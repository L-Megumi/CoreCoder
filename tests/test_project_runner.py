"""Offline engineering controls over actual Workers, tools and private checks.

The scripted actions are known fixture repairs, not model diagnostic accuracy.
Failure injections wrap a real Worker result only where explicitly documented.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects import ProjectRunInput, capture_manifest
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from researchcoder.evaluation.projects.verification import ProjectCheck, ReceiptSpec
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.harness import ResearchHarness
from researchcoder.runtime.protocol import ModelSpec, ProtectedFile, RuntimeLimits, ScriptedCall, ScriptedTurn


BAD = "def increment(value):\n    return value - 1\n\ndef identity(value):\n    return value\n"
GOOD = BAD.replace("return value - 1", "return value + 1")
PARTIAL = BAD.replace("return value - 1", "return value - 2")
REGRESSION = GOOD.replace("def identity(value):\n    return value", "def identity(value):\n    return None")
SECRET = "hidden-oracle-only-m2-98f14a715b"

CHECKER = '''import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
profile = sys.argv[2]
source = root / "module.py"
module = runpy.run_path(str(source))
if profile == "public":
    name, passed = "public-output", module["increment"](3) == 4
elif profile == "f2p":
    name, passed = "private-output", module["increment"](7) == 8
elif profile == "p2p":
    name, passed = "preserve-identity", module["identity"](31) == 31
else:
    raise RuntimeError("Unknown trusted profile")
EMPTY_AT = EMPTY_VALUE
if profile == EMPTY_AT and passed:
    raise SystemExit(0)
print(json.dumps(dict(schema_version=1, profile=profile, origin=str(source),
                     checks=[dict(check=name, passed=bool(passed))], passed=bool(passed))))
raise SystemExit(0 if passed else 1)
'''


def fixture(tmp_path, *, empty_at=None):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text(BAD)
    (source / ".projectrc").write_text("public_mode = stable\n")
    (source / "LICENSE").write_text("Fixture license\n")
    public = tmp_path / "public-check.py"
    private = tmp_path / "private-check.py"
    public.write_text(CHECKER.replace("EMPTY_VALUE", repr(empty_at)))
    private.write_text("# " + SECRET + "\n" + CHECKER.replace("EMPTY_VALUE", repr(empty_at)))
    checks = {}
    for profile, name in (("public", "public-output"), ("f2p", "private-output"), ("p2p", "preserve-identity")):
        path = public if profile == "public" else private
        checks[profile] = ProjectCheck(
            check_id="fixture-" + profile,
            argv_template=(sys.executable, "-I", "-B", str(path), "{project}", profile),
            protected_files=(ProtectedFile(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),),
            receipt=ReceiptSpec(kind="checks_json", origin_field="origin", origin_relative_path="module.py",
                required_fields=("schema_version", "profile", "origin", "checks", "passed"),
                expected_checks=(name,), exact_fields=(("schema_version", 1), ("profile", profile))),
            timeout_seconds=15,
        )
    task = ProjectRunInput(project_id="fixture-project", source_revision="fixture-v1", manifest=capture_manifest(source),
        objective="Fix increment so it increases its input by one; preserve identity and all configuration.",
        public_log="AssertionError: increment(3) returned 2, expected 4", writable_paths=("module.py",))
    return source, task, checks, private


def call(name, args, identity):
    return ScriptedTurn(calls=(ScriptedCall(call_id=identity, name=name, arguments_json=json.dumps(args)),),
                        prompt_tokens=10, completion_tokens=5)


def scripted(new=GOOD, old=BAD, *, first_read="module.py"):
    turns = [call("read_file", {"file_path": str(first_read)}, "read")]
    if new is not None:
        turns.append(call("edit_file", {"file_path": "module.py", "old_string": old, "new_string": new}, "edit"))
    turns.append(ScriptedTurn(content="Hypothesis: module.py increment has the wrong operation. Inspect the actual check result.",
                              prompt_tokens=10, completion_tokens=5))
    return ModelSpec(turns=tuple(turns))


def policy(**changes):
    base = ProjectPolicy(task_limits=RuntimeLimits(max_model_calls=8, max_tool_calls=12, max_rounds=8,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=256, max_total_tokens=50000),
        job_limits=JobLimits(max_model_calls=16, max_tool_calls=30, max_total_tokens=100000, max_runtime_seconds=90),
        max_feedback_rounds=0)
    return replace(base, **changes)


def run(tmp_path, model=None, *, settings=None, empty_at=None, feedback_models=(), cancel_event=None):
    source, task, checks, private = fixture(tmp_path, empty_at=empty_at)
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        model or scripted(), policy=settings or policy(), feedback_models=feedback_models, cancel_event=cancel_event)
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    return result, raw, target, private, io


def requests(raw, io):
    values = []
    for row in raw["rounds"]:
        for value in row["result"]["outputs"]:
            artifact = io.read_json(ArtifactRef.from_dict(value))
            for request in artifact.get("request_records", []):
                values.append(io.read_json(ArtifactRef.from_dict(request)))
    return values


@pytest.mark.parametrize("patch,success,f2p,p2p", [(GOOD, True, True, True), (None, False, False, True),
                                                 (REGRESSION, False, True, False)])
def test_project_real_worker_repair_and_independent_outcomes(tmp_path, patch, success, f2p, p2p):
    result, raw, target, _, io = run(tmp_path, scripted(patch))
    assert result["end_to_end_success"] is success
    assert result["fail_to_pass"] is f2p and result["pass_to_pass"] is p2p
    assert result["status"] == ("succeeded" if success else "failed")
    assert raw["preflight"]["valid"] and len(raw["rounds"]) == 1
    row = raw["rounds"][0]
    assert row["record"]["worker_pid"] and row["record"]["worker_exit_code"] == 0
    assert row["result"]["job_id"] == raw["job_id"]
    assert result["model_requests"] == (2 if patch is None else 3)
    assert result["usage_semantics"] == "scripted_fixture_not_provider_usage"
    assert result["actual_prompt_tokens"] is None and result["cost"] is None
    assert result["diagnosis_correct"] is None and result["localization_correct"] is None
    assert (target / "candidate" / "module.py").read_text() == (BAD if patch is None else patch)
    assert (tmp_path / "source" / "module.py").read_text() == BAD
    assert len(requests(raw, io)) == result["model_requests"]
    if patch == REGRESSION:
        assert row["result"]["status"] == "succeeded"  # Public check is insufficient for E2E success.
        assert not raw["final_checks"]["p2p"]["receipt"]["passed"]


def test_public_feedback_preserves_attempts_evidence_and_one_job_budget(tmp_path):
    result, raw, _, _, io = run(tmp_path, scripted(PARTIAL), settings=policy(max_feedback_rounds=1),
                               feedback_models=(scripted(GOOD, PARTIAL),))
    assert result["end_to_end_success"] and result["feedback_retries"] == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "failed" and first["result"]["failure_kind"] == "verification_failed"
    assert first["public_receipt"]["passed"] is False
    assert second["result"]["status"] == "succeeded"
    assert first["result"]["job_id"] == second["result"]["job_id"] == raw["job_id"]
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert second["before"] == first["after"]
    assert raw["budget"]["charged"]["model_calls"] == 6
    assert set(raw["budget"]["settlements"]) == {"coding-0", "coding-1"}
    assert not raw["budget"]["pending"]
    assert second["budget"]["charged"]["total_tokens"] > first["budget"]["charged"]["total_tokens"]
    recorded = requests(raw, io)
    next_input = json.dumps(next(r for r in recorded if r["task_id"] == "coding-1"), ensure_ascii=False)
    assert first["result"]["attempt_id"] in next_input and "Previous public failures" in next_input
    assert SECRET not in json.dumps(recorded) and "private-output" not in next_input


def test_feedback_cannot_reset_exhausted_job_allowance(tmp_path):
    settings = policy(max_feedback_rounds=1,
        job_limits=JobLimits(max_model_calls=3, max_tool_calls=30, max_total_tokens=100000, max_runtime_seconds=90))
    result, raw, _, _, _ = run(tmp_path, scripted(PARTIAL), settings=settings, feedback_models=(scripted(GOOD, PARTIAL),))
    assert not result["end_to_end_success"] and result["failure_kind"] == "budget_exhausted"
    assert len(raw["rounds"]) == 1 and raw["budget"]["charged"]["model_calls"] == 3
    assert set(raw["budget"]["settlements"]) == {"coding-0"} and not raw["budget"]["pending"]
    assert raw["rounds"][0]["result"]["status"] == "failed"


def test_missing_worker_accounting_retains_liability_and_blocks_feedback(tmp_path, monkeypatch):
    original = ResearchHarness.execute
    def lose_accounting(self, spec, config):
        result = original(self, spec, config)
        if spec.kind == "coding":
            # Fault injection after a real Worker: simulate losing its accounting envelope.
            self.records[result.attempt_id].pop("budget")
        return result
    monkeypatch.setattr(ResearchHarness, "execute", lose_accounting)
    settings = policy(max_feedback_rounds=1)
    result, raw, _, _, _ = run(tmp_path, scripted(PARTIAL), settings=settings, feedback_models=(scripted(GOOD, PARTIAL),))
    assert result["status"] == "needs_review" and result["failure_kind"] == "accounting_unknown"
    assert len(raw["rounds"]) == 1
    settlement = raw["budget"]["settlements"]["coding-0"]
    assert settlement["usage_unknown"] and not settlement["accounting_complete"]
    assert settlement["charged"] == settlement["reserved"] == raw["budget"]["charged"]
    assert result["unknown_requests"] is None and result["model_requests"] is None
    assert result["actual_prompt_tokens"] is None and not result["usage_complete"]


def test_private_checker_read_is_rejected_and_contents_never_enter_model_requests(tmp_path):
    source, task, checks, private = fixture(tmp_path)
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(first_read=private), policy=policy(max_feedback_rounds=1), feedback_models=(scripted(),))
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert not result["end_to_end_success"] and len(raw["rounds"]) == 1 and not raw["final_checks"]
    effective = requests(raw, io)
    assert len(effective) == 3 and SECRET not in json.dumps(effective)
    assert "outside the repair project" in json.dumps(effective)
    assert SECRET not in (target / "public-input.json").read_text()


@pytest.mark.parametrize("empty_at", ["public", "f2p"])
def test_empty_checker_success_cannot_be_promoted_to_repair_success(tmp_path, empty_at):
    result, raw, _, _, _ = run(tmp_path, empty_at=empty_at)
    assert raw["preflight"]["valid"] and len(raw["rounds"]) == 1
    assert raw["rounds"][0]["result"]["status"] == "succeeded"
    assert result["status"] == "needs_review" and result["failure_kind"] == "oracle_invalid"
    assert not result["end_to_end_success"]
    if empty_at == "public":
        assert raw["rounds"][0]["public_receipt"] is None
        assert raw["rounds"][0]["public_receipt_error"] and not raw["final_checks"]
    else:
        assert raw["final_checks"]["f2p"]["receipt_error"] and result["fail_to_pass"] is None


def test_nonwritable_configuration_edit_is_refused(tmp_path):
    model = ModelSpec(turns=(
        call("edit_file", {"file_path": ".projectrc", "old_string": "stable", "new_string": "changed"}, "edit-config"),
        ScriptedTurn(content="Stopped after denied operation", prompt_tokens=10, completion_tokens=5),
    ))
    result, raw, target, _, _ = run(tmp_path, model)
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert not raw["policy_intact"] and not raw["final_checks"]
    assert (target / "candidate" / ".projectrc").read_text() == "public_mode = stable\n"


def test_external_protected_config_change_preserves_the_completed_attempt(tmp_path, monkeypatch):
    original = ResearchHarness.execute
    def modify_after_worker(self, spec, config):
        result = original(self, spec, config)
        if spec.kind == "coding":
            # Real Worker succeeded; an independent writer changes a dotfile before reconciliation.
            (Path(spec.workspace.root) / ".projectrc").write_text("changed externally\n")
        return result
    monkeypatch.setattr(ResearchHarness, "execute", modify_after_worker)
    result, raw, _, _, _ = run(tmp_path)
    assert result["status"] == "needs_review" and result["failure_kind"] == "source_changed"
    assert not result["end_to_end_success"] and not raw["final_checks"]
    assert len(raw["attempt_results"]) == 1
    assert raw["attempt_results"][0]["status"] == "succeeded"
    assert raw["budget"]["charged"]["model_calls"] == 3


def test_cancellation_before_start_preserves_planned_sample_without_worker(tmp_path):
    cancel = threading.Event()
    cancel.set()
    result, raw, target, _, _ = run(tmp_path, cancel_event=cancel)
    assert result["status"] == "cancelled" and result["failure_kind"] == "interrupted"
    assert raw["planned"] == 1 and not raw["started"] and not raw["records"] and not raw["rounds"]
    assert not (target / "candidate").exists() and not raw["budget"]["settlements"]


def test_cancellation_between_attempts_keeps_failure_and_stops_retry(tmp_path, monkeypatch):
    cancel = threading.Event()
    original = ResearchHarness.execute
    def cancel_after_worker(self, spec, config):
        result = original(self, spec, config)
        if spec.kind == "coding":
            cancel.set()
        return result
    monkeypatch.setattr(ResearchHarness, "execute", cancel_after_worker)
    result, raw, _, _, _ = run(tmp_path, scripted(PARTIAL), settings=policy(max_feedback_rounds=1),
        feedback_models=(scripted(GOOD, PARTIAL),), cancel_event=cancel)
    assert result["status"] == "cancelled" and len(raw["rounds"]) == 1
    assert raw["rounds"][0]["result"]["status"] == "failed"
    assert raw["budget"]["charged"]["model_calls"] == 3 and not raw["final_checks"]


def test_recompute_uses_archived_artifact_only_and_rejects_tampering(tmp_path, monkeypatch):
    result, _, target, _, _ = run(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline recomputation must not execute a Worker")
    monkeypatch.setattr(ResearchHarness, "execute", forbidden)
    assert recompute(target) == result
    ref = ArtifactRef.from_json((target / "raw-reference.json").read_text())
    artifact = target / "artifacts" / ref.storage_key
    artifact.write_text("{}")
    with pytest.raises(ValueError):
        recompute(target)


def test_live_model_requires_authorization_before_any_trial_materialization(tmp_path):
    source, task, checks, _ = fixture(tmp_path)
    target = tmp_path / "trial"
    live = ModelSpec(provider="openai", model="explicit-test-model", base_url="https://api.deepseek.com")
    with pytest.raises(ValueError, match="explicit scope authorization"):
        ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"], live)
    assert not target.exists()


def test_configuration_change_after_attempt_receipt_is_reconciled_before_private_checks(tmp_path, monkeypatch):
    original = ProjectCheck.check_protected
    public_checks = 0
    injected = False
    def mutate_after_receipt(self):
        nonlocal public_checks, injected
        original(self)
        if self.check_id == "fixture-public":
            public_checks += 1
            if public_checks == 3:
                # The controller has checked the completed attempt's manifest/receipt.
                # This independent edit occurs in its following checker-integrity pass.
                (tmp_path / "trial" / "candidate" / ".projectrc").write_text("late external change\n")
                injected = True
    monkeypatch.setattr(ProjectCheck, "check_protected", mutate_after_receipt)
    result, raw, _, _, _ = run(tmp_path)
    assert injected and public_checks >= 3
    assert result["status"] == "needs_review" and result["failure_kind"] == "source_changed"
    assert not result["end_to_end_success"] and not raw["final_checks"]
    assert len(raw["attempt_results"]) == 1 and raw["attempt_results"][0]["status"] == "succeeded"
    assert raw["budget"]["charged"]["model_calls"] == 3


def test_private_checker_inside_source_is_rejected_before_candidate_or_agent(tmp_path):
    source, task, checks, private = fixture(tmp_path)
    hidden = source / "hidden-check.py"
    hidden.write_bytes(private.read_bytes())
    task = replace(task, manifest=capture_manifest(source))
    for profile in ("f2p", "p2p"):
        checks[profile] = replace(checks[profile],
            argv_template=(sys.executable, "-I", "-B", str(hidden), "{project}", profile),
            protected_files=(ProtectedFile(path=str(hidden), sha256=hashlib.sha256(hidden.read_bytes()).hexdigest()),))
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(first_read="hidden-check.py"), policy=policy())
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    assert not result["end_to_end_success"]
    assert result["status"] == "incomplete" and result["failure_kind"] == "private_checker_in_source"
    assert not raw["started"] and not (target / "candidate").exists()
    assert not raw["records"] and not raw["rounds"] and not raw["attempt_results"]
    assert not raw["preflight"].get("checks") and not raw["final_checks"]
    assert not requests(raw, io) and SECRET not in (target / "public-input.json").read_text()


def test_final_private_check_cancellation_stays_cancelled_and_stops_p2p(tmp_path, monkeypatch):
    from researchcoder.evaluation.projects import runner as project_runner
    cancel = threading.Event()
    original = project_runner.run_check
    invoked = []
    def cancel_final_f2p(check, project, io, job_id, task_id, harness=None, *, cancel_event=None):
        invoked.append(task_id)
        if task_id == "final-f2p":
            # Exercise the real Harness cancellation path, not a fabricated CheckExecution.
            assert cancel_event is cancel
            cancel.set()
        return original(check, project, io, job_id, task_id, harness=harness, cancel_event=cancel_event)
    monkeypatch.setattr(project_runner, "run_check", cancel_final_f2p)
    result, raw, _, _, _ = run(tmp_path, cancel_event=cancel)
    assert result["status"] == "cancelled" and result["failure_kind"] == "interrupted"
    assert not result["end_to_end_success"]
    assert "final-f2p" in invoked and "final-p2p" not in invoked
    assert set(raw["final_checks"]) == {"f2p"}
    assert raw["final_checks"]["f2p"]["result"]["status"] == "cancelled"
    assert len(raw["attempt_results"]) == 1 and raw["attempt_results"][0]["status"] == "succeeded"
    assert raw["budget"]["charged"]["model_calls"] == 3
