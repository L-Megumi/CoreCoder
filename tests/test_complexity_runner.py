from dataclasses import replace
import hashlib
import os
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.runner import EvaluationRunner, recompute, build_workflow
from researchcoder.evaluation.contracts import TrialSpec
from researchcoder.evaluation.complexity.protocol import offline_config
from researchcoder.evaluation.complexity.dataset import load_manifest, load_complexity, protected_paths


def execute(tmp_path, identity, mode, **changes):
    case = next(c for c in load_manifest().cases if c.case_id == identity)
    config = offline_config(case_ids=(identity,), splits=(case.split,), modes=(mode,), max_trials=1, **changes)
    target = tmp_path / "evaluation"
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths()}
    report = EvaluationRunner().run(target, config, torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
    assert recompute(target) == report
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths()}
    io = ArtifactIO(target / "trials/trial-0000/artifacts")
    result = io.read_json(ArtifactRef.from_json((target / "records/trial-0000.json").read_text()))
    raw = io.read_json(ArtifactRef.from_dict(result["raw"]))
    return result, raw, report, io, target


@pytest.mark.parametrize("identity", ["cpx_03", "cpx_04"])
@pytest.mark.parametrize("mode", ["single", "sequential", "parallel", "evidence"])
def test_existing_runtime_four_modes_new_dataset_with_independent_oracle(tmp_path, identity, mode):
    result, raw, report, io, target = execute(tmp_path, identity, mode)
    assert result["status"] == "succeeded", (result, raw)
    assert result["metrics"]["fail_to_pass"] and result["metrics"]["pass_to_pass"]
    assert result["fail_to_pass_verification"]["spec"]["oracle_profile"] == "benchmark-holdout"
    assert result["pass_to_pass_verification"]["spec"]["oracle_profile"] == "benchmark-regression"
    verifications = [t["verification"] for t in raw["task_results"] if t.get("verification")]
    # Repair can deliberately defer verification to the DAG's separate Verifier.
    # Its not_run receipt has no command and must not be treated as an execution.
    idle = [v for v in verifications if v["status"] == "not_run"]
    assert all(v["spec"] is None and v["exit_code"] is None and v["checked_workspace_revision"] is None for v in idle)
    internal = [v for v in verifications if v["status"] != "not_run"]
    assert internal and all(v["status"] == "passed" and v["spec"]["oracle_profile"] == "benchmark-discovery"
                            and v["spec"]["argv"][-1] == "discovery" for v in internal)
    assert result["fail_to_pass_verification"]["spec"]["argv"][-1] == "holdout"
    assert result["pass_to_pass_verification"]["spec"]["argv"][-1] == "regression"
    assert raw["complexity_diagnostics"]["effective_request_artifacts"] == result["metrics"]["model_requests"]
    assert raw["complexity_diagnostics"]["recoverable_evidence_refs"] == raw["complexity_diagnostics"]["evidence_refs"]
    assert report["complexity"]["levels"][str(2 if identity == "cpx_03" else 3)][mode]["succeeded"] == 1
    assert not raw.get("policy_violations")
    assert result["metrics"]["cost"] is None and not result["metrics"]["usage_complete"]
    labels = load_complexity()
    assert (target / "complexity-manifest.json").read_text() == labels.to_json()


@pytest.mark.parametrize("mode", ["single", "sequential", "parallel", "evidence"])
def test_two_stage_feedback_uses_shared_budget_and_fresh_current_source(tmp_path, mode):
    result, raw, _, io, target = execute(tmp_path, "cpx_04", mode, scripted_behavior="staged_correct", max_feedback_rounds=1)
    assert result["status"] == "succeeded", (result, raw)
    rounds = raw["feedback_rounds"]
    assert len(rounds) == 2 and rounds[0]["verification"]["status"] == "failed" and rounds[1]["verification"]["status"] == "passed"
    assert rounds[0]["workspace_revision"] != rounds[1]["workspace_revision"]
    assert raw["budget"]["charged"]["model_calls"] > rounds[0]["budget"]["charged"]["model_calls"]
    assert raw["budget"]["charged"]["model_calls"] == sum(v["charged"]["model_calls"] for v in raw["budget"]["settlements"].values())
    assert not raw["budget"]["pending"] and raw["feedback_retries"] == 1
    assert raw["replanning_count"] == (0 if mode == "single" else 1)
    assert len({r["attempt_id"] for r in raw["task_results"]}) == len(raw["task_results"])
    if mode != "single":
        assert (target / "trials/trial-0000/source-snapshot-r1").is_dir()
        assert {r["task_id"].split(".")[0] for r in raw["task_results"]} == {"r0", "r1"}
        # Both previous failure and later success stay archived; no relabeling r0.
        assert any(r["status"] == "failed" for r in raw["task_results"] if r["task_id"].startswith("r0."))


@pytest.mark.parametrize("mode", ["single", "evidence"])
@pytest.mark.parametrize("behavior", ["diagnosis_fail", "wrong_patch", "repeat_patch"])
def test_negative_controls_remain_failures_with_full_records(tmp_path, mode, behavior):
    result, raw, report, io, _ = execute(tmp_path, "cpx_04", mode, scripted_behavior=behavior,
        max_feedback_rounds=2 if behavior == "repeat_patch" else 0)
    assert result["status"] != "succeeded" and not result["metrics"]["end_to_end_success"]
    assert report["planned"] == report["recorded"] == 1
    assert raw["records"] and raw["budget"]["charged"]["model_calls"] > 0
    if behavior == "repeat_patch":
        assert len(raw["feedback_rounds"]) == 2
        assert raw["feedback_rounds"][0]["workspace_revision"] == raw["feedback_rounds"][1]["workspace_revision"]
        trace = io.read_json(ArtifactRef.from_dict(raw["feedback_trace"]))["events"]
        assert any(e.get("reason") == "no_new_evidence" for e in trace)
        assert raw["complexity_diagnostics"]["invalid_tool_results"] >= 1


@pytest.mark.parametrize("mode", ["single", "evidence"])
def test_insufficient_budget_cannot_restart_as_free_feedback(tmp_path, mode):
    config = offline_config()
    result, raw, report, _, _ = execute(tmp_path, "cpx_04", mode, max_feedback_rounds=1,
        job_limits=replace(config.job_limits, max_model_calls=1))
    assert result["status"] != "succeeded" and result["failure_kind"] == "budget_exhausted"
    assert raw["budget"]["charged"]["model_calls"] <= 1
    assert report["planned"] == report["recorded"] == 1


def test_same_graph_role_profiles_and_public_access_except_concurrency(tmp_path):
    case = load_manifest().cases[3]
    config = offline_config(case_ids=(case.case_id,))
    built = []
    for mode in ("sequential", "parallel"):
        trial = TrialSpec(trial_id="trial", case_id=case.case_id, mode=mode, repeat=0, ordinal=0)
        built.append(build_workflow(case, trial, tmp_path / mode, config, os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"]))
    a, b = built
    assert a["config"].max_workers == 1 and b["config"].max_workers == 2
    assert a["config"].job_limits == b["config"].job_limits
    assert [(t.task_id, t.role, t.template_id, t.dependencies) for t in a["plan"].tasks] == [(t.task_id, t.role, t.template_id, t.dependencies) for t in b["plan"].tasks]
    for name in a["config"].templates:
        x, y = [item["config"].templates[name] for item in built]
        assert x.spec.objective == y.spec.objective and x.spec.policy == y.spec.policy
        assert x.config.diagnostic_tools == y.config.diagnostic_tools and x.config.limits == y.config.limits
        if x.role == "verification":
            assert x.spec.verification.argv[:3] == y.spec.verification.argv[:3]
            assert x.spec.verification.argv[4:] == y.spec.verification.argv[4:]
        else:
            assert "Public candidate file inventory" in x.spec.objective
        assert x.config.record_requests and y.config.record_requests
