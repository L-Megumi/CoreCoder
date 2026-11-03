"""Changed files must refresh reads without silently mixing observed versions."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from corecoder.completion import CompletionChecklist
from corecoder.diagnose import validate_diagnosis
from corecoder.diagnosis_demo import create_demo_case, run_demo
from corecoder.diagnostic_policy import EvidenceToolPolicy
from corecoder.diagnostic_tools import Investigation
from corecoder.evidence_context import EvidenceContextManager
from corecoder.locator import _Locator
from corecoder.observation_state import current_observation_ids
from corecoder.report_findings import validate_findings


def call(state, name, **arguments):
    return json.loads(state.execute(name, arguments))


@pytest.fixture
def case(tmp_path):
    workspace, root, log, _, _ = create_demo_case(tmp_path / "case")
    return Investigation(log, root, allowed_path_roots=[workspace]), root


def test_repeated_file_read_refreshes_changed_content_and_keeps_old_observation(case):
    state, root = case
    before = call(state, "read_project_file", path="train.py")["observations"][0]
    (root / "train.py").write_text("value = 123\n", encoding="utf-8")
    after = call(state, "read_project_file", path="train.py")["observations"][0]
    assert after["data"]["content"] == "value = 123"
    assert after["data"]["file_sha256"] != before["data"]["file_sha256"]
    assert before["id"] in state.observations


def test_same_length_edit_with_restored_mtime_is_not_a_cache_hit(case):
    state, root = case
    source = root / "entry.py"
    source.write_text("value = 1\n", encoding="utf-8")
    before = call(state, "read_project_file", path="entry.py")["observations"][0]
    previous = source.stat()
    source.write_text("value = 2\n", encoding="utf-8")
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    after = call(state, "read_project_file", path="entry.py")["observations"][0]
    assert after["data"]["file_sha256"] != before["data"]["file_sha256"]
    assert after["data"]["content"] == "value = 2"


def test_deleted_file_is_not_returned_from_cache_and_recreation_is_read(case):
    state, root = case
    source = root / "train.py"
    call(state, "read_project_file", path="train.py")
    source.unlink()
    missing = call(state, "read_project_file", path="train.py")["observations"][0]
    assert missing["data"]["status"] == "missing"
    assert "content" not in missing["data"]
    source.write_text("value = 2\n", encoding="utf-8")
    recreated = call(state, "read_project_file", path="train.py")["observations"][0]
    assert recreated["data"]["content"] == "value = 2"


def test_unchanged_file_still_reuses_cached_snapshot(case):
    state, _ = case
    first = call(state, "read_project_file", path="train.py")
    second = call(state, "read_project_file", path="train.py")
    assert first == second
    assert state.reader.stats["files_read"] == 2  # Recheck bounded bytes; reuse the parsed snapshot.
    assert state.reader.stats["cache_hits"] == 1


def test_missing_then_created_file_recovers_in_same_investigation(case):
    state, root = case
    assert call(state, "read_project_file", path="new.py")["observations"][0]["data"]["status"] == "missing"
    (root / "new.py").write_text("answer = 42\n", encoding="utf-8")
    assert call(state, "read_project_file", path="new.py")["observations"][0]["data"]["status"] == "read"


def test_checklist_uses_new_config_instead_of_old_locator_preview(case):
    state, root = case
    call(state, "extract_log_evidence")
    call(state, "locate_source", evidence_ids=[item["id"] for item in state.evidence])
    new_root = root / "new-dataset"
    (root / "data.yaml").write_text(f"path: {new_root}\nval: next-val\n", encoding="utf-8")
    call(state, "read_project_file", path="data.yaml")
    checks = CompletionChecklist(state).status()["checks"]
    current = [check for check in checks if check["kind"] == "current_validation_path"]
    assert len(current) == 1
    assert current[0]["subject"] == str(new_root / "next-val")


def test_latest_missing_config_does_not_remain_readable_from_an_older_preview(case):
    state, root = case
    call(state, "extract_log_evidence")
    call(state, "locate_source", evidence_ids=[item["id"] for item in state.evidence])
    (root / "data.yaml").unlink()
    call(state, "read_project_file", path="data.yaml")
    check = next(item for item in CompletionChecklist(state).status()["checks"] if item["kind"] == "config_read")
    assert check["status"] == "blocked"
    assert check["blocked_reason"] == "missing"


def test_return_to_old_content_restores_latest_version_order(case):
    state, root = case
    config = root / "data.yaml"
    initial = config.read_bytes()
    call(state, "extract_log_evidence")
    old = call(state, "read_project_file", path="data.yaml")["observations"][0]
    config.write_text(f"path: {root / 'other'}\nval: other-val\n", encoding="utf-8")
    call(state, "read_project_file", path="data.yaml")
    config.write_bytes(initial)
    again = call(state, "read_project_file", path="data.yaml")["observations"][0]
    assert old["id"] == again["id"]
    current = next(item for item in CompletionChecklist(state).status()["checks"] if item["kind"] == "current_validation_path")
    assert current["subject"] == str(root / "dataset/images/val")


def test_known_outdated_source_cannot_be_a_primary_location(case):
    state, root = case
    old = call(state, "read_project_file", path="train.py")["observations"][0]
    (root / "train.py").write_text("value = 2\n", encoding="utf-8")
    call(state, "read_project_file", path="train.py")
    with pytest.raises(ValueError, match="superseded"):
        validate_findings({"failure_category": "unknown", "cause_code": "unknown", "primary_locations": [
            {"observation_id": old["id"], "line_no": 1},
        ]}, state)


def test_current_state_cannot_mix_known_old_and_new_versions(tmp_path):
    result = run_demo(directory=tmp_path / "demo")
    state = SimpleNamespace(evidence=result["evidence"], observations={item["id"]: item for item in result["observations"]})
    source = next(item for item in result["observations"] if item["kind"] == "file_excerpt")
    newer = copy.deepcopy(source)
    newer["id"] = "obs_new_version"
    newer["data"]["file_sha256"] = "new-hash"
    state.observations[newer["id"]] = newer
    prediction = copy.deepcopy(result["diagnosis"])
    prediction["current_state"]["observation_ids"] = [source["id"], newer["id"]]
    with pytest.raises(ValueError, match="superseded"):
        validate_diagnosis(json.dumps(prediction), state)


def test_guided_policy_reopens_location_after_observing_a_changed_source(case):
    state, root = case
    call(state, "extract_log_evidence")
    call(state, "locate_source", evidence_ids=[item["id"] for item in state.evidence])
    assert "locate_source" not in EvidenceToolPolicy().select(state)[0]
    (root / "train.py").write_text("value = 2\n", encoding="utf-8")
    call(state, "read_project_file", path="train.py")
    assert "locate_source" in EvidenceToolPolicy().select(state)[0]


def test_file_changed_during_read_is_not_cached_as_a_stable_snapshot(case, monkeypatch):
    state, root = case
    source = root / "train.py"
    original_read = _Locator._read

    # Change the file immediately after the content read, before the post-read stat.
    original_open = type(source).open
    class ChangingStream:
        def __enter__(self):
            self.stream = original_open(source, "rb")
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size):
            raw = self.stream.read(size)
            with original_open(source, "wb") as writer:
                writer.write(b"changed = True\n")
            return raw

    def changing_open(path, *args, **kwargs):
        return ChangingStream() if path == source and args and args[0] == "rb" else original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(source), "open", changing_open)
    assert original_read(state.reader, source)["status"] == "file_changed_during_read"


def test_new_missing_path_check_supersedes_old_read(case):
    state, root = case
    old = call(state, "read_project_file", path="train.py")["observations"][0]
    (root / "train.py").unlink()
    missing = call(state, "inspect_paths", paths=[str(root / "train.py")])["observations"][0]
    current = current_observation_ids(state)
    assert old["id"] not in current and missing["id"] in current


def test_recreated_file_read_supersedes_older_missing_path_state(case):
    state, root = case
    target = root / "new.py"
    missing = call(state, "inspect_paths", paths=[str(target)])["observations"][0]
    target.write_text("answer = 1\n", encoding="utf-8")
    fresh = call(state, "read_project_file", path="new.py")["observations"][0]
    current = current_observation_ids(state)
    assert missing["id"] not in current and fresh["id"] in current


def test_evidence_memory_marks_old_snapshots_and_keeps_newest_first(case):
    state, root = case
    before = call(state, "read_project_file", path="train.py")["observations"][0]
    (root / "train.py").write_text("answer = 1\n", encoding="utf-8")
    after = call(state, "read_project_file", path="train.py")["observations"][0]
    memory = json.loads(EvidenceContextManager(state).build_memory(50000))
    rows = {item["id"]: item for item in memory["observations"]}
    assert rows[before["id"]]["snapshot_current"] is False
    assert rows[after["id"]]["snapshot_current"] is True
    assert memory["observations"][0]["id"] == after["id"]


def test_atomic_replacement_invalidates_cached_file_identity(case):
    state, root = case
    target = root / "entry.py"
    target.write_text("value = 1\n", encoding="utf-8")
    old = call(state, "read_project_file", path="entry.py")["observations"][0]
    previous = target.stat()
    replacement = root / "replacement.py"
    replacement.write_text("value = 2\n", encoding="utf-8")
    os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    replacement.replace(target)
    new = call(state, "read_project_file", path="entry.py")["observations"][0]
    assert new["data"]["file_sha256"] != old["data"]["file_sha256"]


def test_config_pages_from_different_versions_cannot_form_a_mixed_path(case):
    state, root = case
    config = root / "data.yaml"
    config.write_text(f"path: {root / 'old-root'}\n" + "# comment\n" * 70 + "val: old-val\n", encoding="utf-8")
    call(state, "extract_log_evidence")
    call(state, "locate_source", evidence_ids=[item["id"] for item in state.evidence])
    config.write_text(f"path: {root / 'new-root'}\n" + "# comment\n" * 70 + "val: new-val\n", encoding="utf-8")
    call(state, "read_project_file", path="data.yaml", start_line=61)
    checks = CompletionChecklist(state).status()["checks"]
    assert not any(item["kind"] == "current_validation_path" for item in checks)
    resolution = next(item for item in checks if item["kind"] == "config_resolution")
    assert resolution["status"] == "blocked"
