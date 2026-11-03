"""Real Worker controls for versioned expert handoff and selected-role dispatch."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.contracts import RepositoryManifest
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import ProtectedFile
from tests.test_project_consultation import expert_model
from tests.test_project_runner import BAD, GOOD, PARTIAL, SECRET, fixture, policy, requests, scripted


def run_trial(tmp_path, *, three_rounds=False, both_models=False):
    source, task, checks, private = fixture(tmp_path)
    private_before = hashlib.sha256(private.read_bytes()).hexdigest()
    if three_rounds:
        # A public observation legitimately changes when the first patch changes
        # the wrong result. The independent private Oracle is never altered.
        public = Path(checks["public"].protected_files[0].path)
        old = 'dict(schema_version=1, profile=profile, origin=str(source),'
        new = 'dict(schema_version=1, profile=profile, origin=str(source), observed=module["increment"](3),'
        contents = public.read_text()
        assert contents.count(old) == 1
        public.write_text(contents.replace(old, new))
        checks["public"] = replace(checks["public"], protected_files=(
            ProtectedFile(path=str(public), sha256=hashlib.sha256(public.read_bytes()).hexdigest()),))
    config = policy(max_feedback_rounds=2 if three_rounds else 1,
        job_limits=JobLimits(max_model_calls=24, max_tool_calls=60, max_total_tokens=400000,
                             max_runtime_seconds=180))
    models = {"code_analysis": expert_model()}
    if both_models:
        # If incorrectly dispatched, this Log fixture cannot produce a valid
        # report. The Code-only decision must ignore its available configuration.
        models["log_analysis"] = expert_model(report=False)
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(None), policy=config, feedback_models=(scripted(PARTIAL), scripted(GOOD, PARTIAL)) if three_rounds
        else (scripted(GOOD),), expert_policy=ExpertPolicy(mode="public-failure-v1"), expert_models=models)
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    assert hashlib.sha256(private.read_bytes()).hexdigest() == private_before
    assert (source / "module.py").read_text() == BAD
    assert SECRET not in json.dumps(requests(raw, io))
    return result, raw, target, io


def test_changed_candidate_discards_current_expert_context_without_erasing_history_or_reconsulting(tmp_path):
    result, raw, target, io = run_trial(tmp_path, three_rounds=True)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"]
    assert len(raw["rounds"]) == 3 and len(raw["attempt_results"]) == 3
    assert [r["result"]["status"] for r in raw["rounds"]] == ["failed", "failed", "succeeded"]
    observed = [json.loads(r["result"]["verification"]["stdout"]["text"])["observed"] for r in raw["rounds"]]
    assert observed == [2, 1, 4]
    assert len(raw["consultations"]) == 1
    consultation = raw["consultations"][0]
    assert consultation["safe_to_continue"] and len(consultation["records"]) == 1
    assert consultation["decision"]["roles"] == ["code_analysis"]
    reference = ArtifactRef.from_dict(consultation["context_ref"])
    original = io.read_json(reference)
    first_version = RepositoryManifest.from_dict(raw["rounds"][0]["after"]).digest
    second_version = RepositoryManifest.from_dict(raw["rounds"][1]["after"]).digest
    assert first_version != second_version
    assert original["complete_source_sha256"] == first_version
    assert original["observations"] and original["hypotheses"]

    captured = requests(raw, io)
    second_requests = json.dumps([r for r in captured if r["task_id"] == "coding-1"])
    third_requests = json.dumps([r for r in captured if r["task_id"] == "coding-2"])
    assert "Source-checked expert observations and hypotheses" in second_requests
    assert "candidate_source_paths" in second_requests
    assert "Source-checked expert observations and hypotheses" not in third_requests
    assert "candidate_source_paths" not in third_requests and "consultation_version" not in third_requests
    assert "Previous public failures" in third_requests
    assert raw["rounds"][0]["result"]["attempt_id"] in third_requests
    assert raw["rounds"][1]["result"]["attempt_id"] in third_requests

    assert raw["context_events"] == [dict(kind="expert_context_stale", generation=2,
        original_manifest_sha256=first_version, current_manifest_sha256=second_version,
        context_ref=reference.to_dict())]
    trace = io.read_json(ArtifactRef.from_dict(raw["trace"]))
    stale = [event for event in trace["events"] if event["event_type"] == "project.consultation.context_stale"]
    assert len(stale) == 1 and stale[0]["count"] == 2 and stale[0]["artifact_id"] == reference.artifact_id
    assert io.read_json(reference) == original  # Immutable historical evidence remains readable.
    assert raw["budget"]["charged"]["model_calls"] == 11
    assert len(raw["budget"]["settlements"]) == 4 and not raw["budget"]["pending"]
    assert len({r["record"]["worker_pid"] for r in raw["rounds"]}) == 3
    assert (target / "candidate" / "module.py").read_text() == GOOD
    assert recompute(target) == result


def test_code_only_decision_projects_available_role_models_and_does_not_start_log_worker(tmp_path):
    result, raw, target, io = run_trial(tmp_path, both_models=True)
    assert result["end_to_end_success"] and result["status"] == "succeeded"
    assert len(raw["consultations"]) == 1 and len(raw["rounds"]) == 2
    consultation = raw["consultations"][0]
    assert consultation["decision"]["roles"] == ["code_analysis"]
    assert consultation["run"]["status"] == "succeeded"
    assert len(consultation["run"]["results"]) == len(consultation["records"]) == 1
    assert consultation["records"][0]["role"] == "code_analysis"
    assert consultation["records"][0]["worker_pid"] > 0
    assert all("expert-log" not in record["task_id"] for record in raw["records"])
    assert all("expert-log" not in identity for identity in raw["budget"]["settlements"])
    assert raw["budget"]["charged"]["model_calls"] == 8 and len(raw["records"]) == 3
    saved = json.loads((target / "config.private.json").read_text())
    assert set(saved["expert_models"]) == {"code_analysis", "log_analysis"}
    assert "candidate_source_paths" in json.dumps(requests(raw, io))
    assert recompute(target) == result
