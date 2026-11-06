"""Correction context, including the no-op second writer seen in a live trial."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.correction_context import (
    INSTRUCTION, VERSION, build_edit_window, validate_worker_window,
)
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from researchcoder.evaluation.projects.workspace import capture_manifest
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import RuntimeLimits
from tests.test_project_consultation import expert_model
from tests.test_project_runner import BAD, GOOD, PARTIAL, SECRET, fixture, requests, scripted


def edit_policy():
    return ProjectPolicy(completion_version="explicit-v1", progress_version="progress-v1",
        repair_cycle_version="repair-cycle-v1", correction_context_version=VERSION,
        task_limits=RuntimeLimits(max_model_calls=12, max_tool_calls=24, max_rounds=10,
            max_runtime_seconds=180, max_context_tokens=16000, max_output_tokens=512, max_total_tokens=200000),
        job_limits=JobLimits(max_model_calls=16, max_tool_calls=60, max_total_tokens=400000,
            max_runtime_seconds=600))


def test_legacy_contract_retains_its_serialized_identity():
    legacy = replace(edit_policy(), correction_context_version="off")
    assert "correction_context_version" not in legacy.to_dict()
    assert ProjectPolicy.from_json(legacy.to_json()) == legacy
    assert ProjectPolicy.from_json(edit_policy().to_json()) == edit_policy()
    with pytest.raises(ValueError):
        replace(edit_policy(), repair_cycle_version="off")


def repositories(tmp_path):
    source, target = tmp_path / "source", tmp_path / "candidate"
    for root, value in ((source, BAD), (target, PARTIAL)):
        root.mkdir()
        (root / "module.py").write_text(value)
        (root / "other.py").write_text("UNCHANGED_OTHER_SOURCE")
    return source, target, capture_manifest(source), capture_manifest(target)


def test_failed_edit_window_is_bound_to_both_versions_and_only_changed_sources(tmp_path):
    source, target, before, after = repositories(tmp_path)
    result = build_edit_window(source, target, before, after, ("module.py",))
    assert result["version"] == VERSION
    assert result["before_manifest_sha256"] == before.digest and result["source_manifest_sha256"] == after.digest
    assert len(result["windows"]) == 1
    row = result["windows"][0]
    assert row["path"] == "module.py"
    assert row["before_sha256"] == hashlib.sha256(BAD.encode()).hexdigest()
    assert row["after_sha256"] == hashlib.sha256(PARTIAL.encode()).hexdigest()
    assert row["start_line"] <= row["changed_start_line"] <= 2 <= row["changed_end_line"] <= row["end_line"]
    assert "return value - 2" in json.dumps(row["lines"])
    assert "UNCHANGED_OTHER_SOURCE" not in json.dumps(result)


@pytest.mark.parametrize("unsafe", ["changed_source", "changed_candidate", "outside_writable_paths", "symlink"])
def test_window_rejects_unbound_or_unwritable_source(tmp_path, unsafe):
    source, target, before, after = repositories(tmp_path)
    names = ("module.py",)
    if unsafe == "changed_source":
        (source / "module.py").write_text(GOOD)
    elif unsafe == "changed_candidate":
        (target / "module.py").write_text(GOOD)
    elif unsafe == "outside_writable_paths":
        names = ("other.py",)
    else:
        private = tmp_path / "private-check.py"
        private.write_text(SECRET)
        (target / "module.py").unlink()
        (target / "module.py").symlink_to(private)
    with pytest.raises(ValueError) as caught:
        build_edit_window(source, target, before, after, names)
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("tamper", ["foreign_path", "invented_text", "stale_manifest", "forged_digest"])
def test_receiving_worker_rejects_forged_window_even_when_controller_schema_is_valid(tmp_path, tamper):
    source, target, before, after = repositories(tmp_path)
    window = build_edit_window(source, target, before, after, ("module.py",))
    request = SimpleNamespace(spec=SimpleNamespace(
        policy=SimpleNamespace(writable_paths=("module.py",)), workspace=SimpleNamespace(root=str(target))))
    sources = SimpleNamespace(current=after)
    failure = {"source_digest": after.digest, "changed_paths": ["module.py"]}
    validate_worker_window(window, request, sources, failure)
    row = window["windows"][0]
    if tamper == "foreign_path":
        row["path"] = "../private-check.py"
    elif tamper == "invented_text":
        row["lines"][0] = SECRET
    elif tamper == "stale_manifest":
        failure["source_digest"] = before.digest
    else:
        row["after_sha256"] = hashlib.sha256(b"wrong revision").hexdigest()
    with pytest.raises(ValueError) as caught:
        validate_worker_window(window, request, sources, failure)
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("write_in_correction", [True, False], ids=["effective-edit", "read-only-correction"])
def test_correction_must_actually_fix_current_candidate(tmp_path, write_in_correction):
    source, task, checks, _ = fixture(tmp_path)
    correction = scripted(GOOD if write_in_correction else None, PARTIAL)
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source,
        checks["public"], checks["f2p"], checks["p2p"], scripted(PARTIAL),
        policy=edit_policy(), feedback_models=(correction,),
        expert_policy=ExpertPolicy(mode="public-failure-v1", initial_mode="code-first-v1", writer_min_model_calls=12),
        expert_models={"code_analysis": expert_model()}, permission_recovery_mode="bounded-public-read-v1",
        writer_context_policy=ContextPolicy(role="repair", strategy="evidence", allow_recovery=True,
            # The scripted expert emits the original exchange representation;
            # its immutable excerpts must be checked using the matching profile.
            coding_source_mode="observed-source-v2", repair_state_version="state-v1"))
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    first, second = raw["rounds"]
    assert first["public_receipt"]["passed"] is False and first["changed_paths"] == ["module.py"]
    assert first["after"] == second["before"] and first["before"] != first["after"]
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert first["result"]["job_id"] == second["result"]["job_id"] == raw["job_id"]
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert raw["budget"]["charged"]["model_calls"] <= 16 and not raw["budget"]["pending"]
    assert {"coding-0", "coding-1"} <= set(raw["budget"]["settlements"])
    assert not any(item["usage_unknown"] for item in raw["budget"]["settlements"].values())
    packets = requests(raw, io)
    current = [p for p in packets if p["task_id"] == "coding-1"]
    assert current
    assert SECRET not in json.dumps(packets) and "private-output" not in json.dumps(packets)
    messages = current[0]["request"]["messages"]
    assert any(INSTRUCTION in m["content"] for m in messages if m["role"] == "system")
    assert all("return value - 2" not in m["content"] for m in messages if m["role"] == "system")
    assert any("return value - 2" in m["content"] and "edit_window" in m["content"]
               for m in messages if m["role"] == "user")
    receipt, = raw["correction_contexts"]
    failure = io.read_json(ArtifactRef.from_dict(receipt["reference"]))
    assert failure["attempt_id"] == first["result"]["attempt_id"]
    assert failure["public_observation"]["failed_checks"] == ["public-output"]
    assert (second["after"] != second["before"]) is write_in_correction
    assert first["attempt_changed_paths"] == ["module.py"]
    assert second["attempt_changed_paths"] == (["module.py"] if write_in_correction else [])
    assert (second["progress"]["effective_file_edits"] > 0) is write_in_correction
    assert (target / "candidate/module.py").read_text() == (GOOD if write_in_correction else PARTIAL)
    assert result["end_to_end_success"] is write_in_correction
    assert result["fail_to_pass"] is write_in_correction and result["pass_to_pass"] is True
    assert recompute(target) == result
