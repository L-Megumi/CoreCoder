"""Public exception metadata survives context without becoming a diagnosis."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path

import pytest

from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.contracts import ArtifactRef
from tests.test_coding_handoff_context import fixture, projection


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def public_case(tmp_path, *, strategy="evidence", dependency=False):
    manager, agent, hooks, io, envelope, _, history = fixture(tmp_path, strategy)
    # The old view legitimately preserves only its first 700 characters. The
    # failure record remains complete; the new field is structural metadata.
    history[0]["stderr"]["text"] = "trusted wrapper stack\n" * 50 + "TypeError: incompatible argument\n"
    frame = dict(path="module.py", line=2, function="increment",
        source_sha256=hashlib.sha256((Path(manager.request.spec.workspace.root) / "module.py").read_bytes()).hexdigest(),
        code_sha256="1" * 64)
    receipt = dict(schema_version=1, kind="public-candidate-exception-v1", passed=False,
        status="failed", exit_code=1, job_id="job", task_id="coding-0", attempt_id="old-writer",
        verification_id="verified-public-0", invocation_id="public-execution-0", exception_type="TypeError",
        candidate_frames=[frame], attribution="candidate_dependency_execution" if dependency else "candidate_execution",
        dependency_frames=[dict(path="/environment/site-packages/library.py", line=10,
            source_sha256="2" * 64, code_sha256="3" * 64)] if dependency else [],
        source_manifest_sha256=history[0]["source_digest"], checked_workspace_revision=manager.scope.workspace_revision,
        public_check_sha256="4" * 64, execution_receipt_sha256="5" * 64, request_sha256="6" * 64,
        stdout_sha256=sha(history[0]["stdout"]["text"]), stderr_sha256=sha(history[0]["stderr"]["text"]))
    reference = io.put_json("job", receipt)
    history[0]["public_exception_ref"] = reference.to_dict()
    envelope["public_failed_attempts"] = io.put_json("job", history).to_dict()
    request = replace(manager.request, evidence_context=json.dumps(envelope))
    return manager, agent, hooks, io, envelope, history, receipt, reference, request


def load(case):
    manager, _, hooks, io, envelope, *_ = case
    request = replace(manager.request, evidence_context=json.dumps(envelope))
    return EvidenceContextEngine(request, io, hooks.ledger, hooks.trace)


def republish(case, *, receipt=None, history=None):
    _, _, _, io, envelope, original, raw, *_ = case
    data = copy.deepcopy(original if history is None else history)
    data[0]["public_exception_ref"] = io.put_json("job", raw if receipt is None else receipt).to_dict()
    envelope["public_failed_attempts"] = io.put_json("job", data).to_dict()


@pytest.mark.parametrize("strategy", ["original", "evidence"])
@pytest.mark.parametrize("dependency", [False, True])
def test_valid_public_execution_metadata_is_visible_without_changing_truncated_streams(tmp_path, strategy, dependency):
    case = public_case(tmp_path, strategy=strategy, dependency=dependency)
    manager = load(case)
    data, messages = projection(manager)
    failure = data["public_failed_attempts"][0]
    observation = failure["public_execution_observation"]
    assert observation["exception_type"] == "TypeError"
    assert observation["kind"] == "verified-public-execution-observation"
    assert observation["status"] == "failed" and observation["passed"] is False
    assert observation["candidate_frames"] == [dict(path="module.py", line=2)]
    assert observation["attribution"] == ("candidate_dependency_execution" if dependency else "candidate_execution")
    assert observation["reference"] == case[7].to_dict()
    assert observation["current_eligible"] and observation["producer_attempt_id"] == "old-writer"
    assert len(failure["stderr"]["text"]) == 700 and "TypeError" not in failure["stderr"]["text"]
    assert failure["stderr"]["view_truncated"]
    assert all(m["role"] != "system" for m in messages if "verified-public-execution-observation" in m.get("content", ""))
    assert "not a diagnosis, Oracle result, or successful repair" in observation["semantics"]
    assert not case[1].read_cache
    assert case[3].read_json(case[7]) == case[6]


def test_original_failure_and_public_receipt_reference_are_losslessly_restorable(tmp_path):
    case = public_case(tmp_path)
    manager = load(case)
    data, _ = projection(manager)
    failure = data["public_failed_attempts"][0]
    document = next(d for d in manager.store.catalog.documents if d.document_id == failure["document_id"])
    raw = manager.store.io.read_bytes(document.artifact).decode()
    pages = [manager.restore(document.document_id, start_line=n, max_lines=1)["content"]
             for n in range(1, len(raw.splitlines()) + 1)]
    value = json.loads("".join(pages))
    if failure["encoding"] == "json-string-chunks-v1":
        value = json.loads("".join(value["chunks"]))
    assert value == {key: item for key, item in case[5][0].items() if key != "hypothesis"}
    assert value["public_exception_ref"] == case[7].to_dict()
    assert value["stderr"]["text"].endswith("TypeError: incompatible argument\n")


def test_source_edit_marks_public_exception_historical_without_granting_fresh_access(tmp_path):
    case = public_case(tmp_path)
    manager = load(case)
    before, _ = projection(manager)
    root = Path(manager.request.spec.workspace.root)
    (root / "module.py").write_text("def increment(value):\n    return value + 1\n")
    after, _ = projection(manager)
    failure = after["public_failed_attempts"][0]
    assert failure["classification"] == "historical_observation" and not failure["current_eligible"]
    assert not failure["public_execution_observation"]["current_eligible"]
    with pytest.raises(ValueError, match="source changed"):
        manager.restore(before["public_failed_attempts"][0]["document_id"])
    assert not case[1].read_cache


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("kind", "model-claim"),
    ("passed", True), ("status", "passed"), ("exit_code", 0), ("exit_code", True),
    ("job_id", "foreign"), ("attempt_id", "foreign-attempt"), ("task_id", "../forged"),
    ("verification_id", "../invalid"), ("source_manifest_sha256", "0" * 64),
    ("checked_workspace_revision", "0" * 64), ("stdout_sha256", "0" * 64), ("stderr_sha256", "0" * 64),
    ("exception_type", "SystemExit"), ("attribution", "root_cause_proved"),
    ("execution_receipt_sha256", "invalid-digest"), ("candidate_frames", []),
])
def test_foreign_or_changed_public_receipt_is_rejected(tmp_path, field, value):
    case = public_case(tmp_path)
    receipt = copy.deepcopy(case[6])
    receipt[field] = value
    republish(case, receipt=receipt)
    with pytest.raises((ValueError, TypeError)):
        load(case)


@pytest.mark.parametrize("field,value", [("path", "../private.py"), ("path", "invented.py"),
    ("line", 0), ("line", True), ("source_sha256", "0" * 64), ("code_sha256", "invalid")])
def test_public_candidate_frame_is_checked_without_expanding_access(tmp_path, field, value):
    case = public_case(tmp_path)
    receipt = copy.deepcopy(case[6])
    receipt["candidate_frames"][0][field] = value
    republish(case, receipt=receipt)
    with pytest.raises((ValueError, TypeError)):
        load(case)


def test_cross_job_reference_cannot_enter_context_even_with_same_record(tmp_path):
    case = public_case(tmp_path)
    _, _, _, io, envelope, history, receipt, *_ = case
    history[0]["public_exception_ref"] = io.put_json("foreign", receipt).to_dict()
    envelope["public_failed_attempts"] = io.put_json("job", history).to_dict()
    with pytest.raises(ValueError, match="Foreign"):
        load(case)


@pytest.mark.parametrize("change", ["output_changed", "truncated", "extra_history", "current_attempt"])
def test_receipt_requires_matching_complete_prior_history(tmp_path, change):
    case = public_case(tmp_path)
    history = copy.deepcopy(case[5])
    receipt = copy.deepcopy(case[6])
    if change == "output_changed": history[0]["stderr"]["text"] += "forged detail"
    elif change == "truncated": history[0]["stderr"]["truncated"] = True
    elif change == "extra_history": history[0]["invented_fact"] = True
    else:
        history[0]["attempt_id"] = receipt["attempt_id"] = case[0].request.attempt_id
    republish(case, receipt=receipt, history=history)
    with pytest.raises((ValueError, TypeError)):
        load(case)


def test_public_receipt_bytes_are_rechecked_when_projecting_context(tmp_path):
    case = public_case(tmp_path)
    manager = load(case)
    manager.store.io.resolve(case[7]).write_bytes(b"tampered public receipt")
    with pytest.raises(ValueError):
        projection(manager)


def test_legacy_history_card_and_source_document_do_not_gain_new_fields(tmp_path):
    manager, _, _, _, _, _, history = fixture(tmp_path)
    data, _ = projection(manager)
    failure = data["public_failed_attempts"][0]
    assert "public_execution_observation" not in failure
    assert "public_exception_ref" not in failure
    restored = manager.restore(failure["document_id"])
    assert json.loads(restored["content"]) == {key: value for key, value in history[0].items() if key != "hypothesis"}
    assert failure["stderr"] == dict(text="AssertionError", original_truncated=False, view_truncated=False)
