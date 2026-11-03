"""A public candidate exception is failed evidence, never an Oracle success."""
import hashlib
import json
from dataclasses import replace

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.context.policy import ContextPolicy
from researchcoder.evaluation.projects.public_execution import PublicExecutionPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from tests.test_project_runner import fixture, scripted, policy, BAD, GOOD, SECRET, requests


BROKEN = BAD.replace("return value - 1", "return value + '1'")


def trial(tmp_path, *, enabled=True, context=False, empty_at=None):
    source, task, checks, private = fixture(tmp_path, empty_at=empty_at)
    identities = {p.path: p.sha256 for check in checks.values() for p in check.protected_files}
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(BROKEN), policy=policy(max_feedback_rounds=1),
        feedback_models=(scripted(GOOD, BROKEN),), permission_recovery_mode="bounded-public-read-v1",
        public_execution_policy=PublicExecutionPolicy() if enabled else None,
        writer_context_policy=ContextPolicy(role="repair", strategy="evidence", allow_recovery=True,
            coding_source_mode="observed-source-v2") if context else None)
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    from pathlib import Path
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in identities.items())
    assert (source / "module.py").read_text() == BAD
    return result, raw, io, directory


@pytest.mark.parametrize("context", [False, True])
def test_public_candidate_exception_allows_same_job_feedback_with_original_failure_intact(tmp_path, context):
    result, raw, io, directory = trial(tmp_path, context=context)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["rounds"]) == 2 and result["feedback_retries"] == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "failed"
    assert first["result"]["failure_kind"] == "verification_failed"
    assert first["public_receipt"] is None and "Empty checker output" in first["public_receipt_error"]
    assert first["public_exception_receipt"]["passed"] is False
    assert first["public_exception_receipt"]["exception_type"] == "TypeError"
    assert first["public_exception_receipt"]["attempt_id"] == first["result"]["attempt_id"]
    assert second["result"]["status"] == "succeeded" and second["public_exception_receipt"] is None
    assert second["public_receipt"]["passed"] is True
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert {record["job_id"] for record in raw["records"]} == {raw["job_id"]}
    assert raw["budget"]["charged"]["model_calls"] == 6
    assert len(raw["budget"]["settlements"]) == 2 and not raw["budget"]["pending"]
    assert all(not row["usage_unknown"] for row in raw["budget"]["settlements"].values())
    assert SECRET not in json.dumps(requests(raw, io))
    assert "TypeError" in json.dumps(requests(raw, io)[3:])
    assert first["result"]["verification"]["stdout"]["text"] == ""
    assert len(first["result"]["verification"]["spec"]["argv"]) == 6
    for check in raw["final_checks"].values():
        assert "public_probe.py" not in " ".join(check["result"]["verification"]["spec"]["argv"])
    assert io.read_json(ArtifactRef.from_dict(first["public_exception_ref"])) == first["public_exception_receipt"]
    archived = first["public_execution_artifacts"]
    assert set(archived) == {"probe", "request", "receipt"}
    assert archived["receipt"]["sha256"] == first["public_exception_receipt"]["execution_receipt_sha256"]
    assert io.read_json(ArtifactRef.from_dict(archived["receipt"]))["exception_type"] == "TypeError"
    assert archived["request"]["sha256"] == first["public_exception_receipt"]["request_sha256"]
    assert recompute(directory) == result


def test_disabled_public_probe_keeps_original_needs_review_no_hidden_feedback(tmp_path):
    result, raw, _, _ = trial(tmp_path, enabled=False)
    assert result["status"] == "needs_review" and result["failure_kind"] == "oracle_invalid"
    assert len(raw["rounds"]) == 1 and not raw["final_checks"]
    assert result["fail_to_pass"] is None and result["pass_to_pass"] is None
    assert "public_execution_policy" not in raw
    assert "public_exception_receipt" not in raw["rounds"][0]


def test_empty_success_json_cannot_be_promoted_by_public_exception_policy(tmp_path):
    result, raw, _, _ = trial(tmp_path, empty_at="public")
    assert not result["end_to_end_success"] and result["failure_kind"] == "oracle_invalid"
    assert len(raw["rounds"]) == 2
    assert raw["rounds"][1]["result"]["verification"]["exit_code"] == 0
    assert raw["rounds"][1]["public_receipt"] is None
    assert raw["rounds"][1]["public_exception_receipt"] is None
    assert not raw["final_checks"]


def test_cli_public_policy_roundtrip_and_unknown_mode_rejection(tmp_path):
    from tests.test_project_adaptive_cli import configuration, load
    cfg = configuration(tmp_path)
    assert load(tmp_path, cfg)["public_execution_policy"] is None
    cfg["public_execution_policy"] = PublicExecutionPolicy().to_dict()
    assert load(tmp_path, cfg)["public_execution_policy"] == PublicExecutionPolicy()
    cfg["public_execution_policy"] = {"mode": "accept-any-nonzero"}
    with pytest.raises(ValueError):
        load(tmp_path, cfg)


def test_missing_public_provenance_after_worker_is_archived_as_review_not_controller_crash(tmp_path, monkeypatch):
    from pathlib import Path
    from researchcoder.runtime.harness import ResearchHarness
    original = ResearchHarness.execute
    def remove_receipt_config(self, spec, config):
        result = original(self, spec, config)
        if spec.kind == "coding":
            request = Path(spec.verification.argv[4])
            assert request.name == "execution-request.json" and request.is_relative_to(tmp_path)
            request.unlink()
        return result
    monkeypatch.setattr(ResearchHarness, "execute", remove_receipt_config)
    result, raw, _, directory = trial(tmp_path)
    assert result["status"] == "needs_review" and result["failure_kind"] == "oracle_invalid"
    assert (directory / "raw-reference.json").is_file()
    assert len(raw["rounds"]) == 1 and not raw["final_checks"]
    assert raw["rounds"][0]["public_exception_receipt"] is None
    assert raw["rounds"][0]["public_execution_archive_error"] == "ValueError"
