import hashlib
from dataclasses import replace

import pytest

from researchcoder.adapters.artifacts import ArtifactIO


def test_jobs_are_isolated_content_checked_and_never_overwritten(tmp_path):
    io = ArtifactIO(tmp_path)
    first = io.put_json("job1", {"value": 1}, artifact_id="result")
    second = io.put_json("job2", {"value": 2}, artifact_id="result")
    assert io.read_json(first) == {"value": 1}
    assert io.read_json(second) == {"value": 2}
    assert first.sha256 == hashlib.sha256(io.read_bytes(first)).hexdigest()
    with pytest.raises(FileExistsError):
        io.put_json("job1", {"value": 3}, artifact_id="result")
    assert io.read_json(first) == {"value": 1}
    with pytest.raises(TypeError):
        io.read_bytes("/etc/passwd")


def test_tampering_is_detected(tmp_path):
    io = ArtifactIO(tmp_path)
    ref = io.put_bytes("job", b"original")
    (tmp_path / ref.storage_key).write_bytes(b"changed!")
    with pytest.raises(ValueError, match="digest"):
        io.read_bytes(ref)


def test_job_symlink_and_final_symlink_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    io = ArtifactIO(tmp_path / "store")
    (io.root / "job").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        io.put_bytes("job", b"bad")
    ref = io.put_bytes("good", b"fine")
    target = io.root / ref.storage_key
    target.unlink()
    target.symlink_to(outside / "file")
    with pytest.raises(ValueError, match="Symlink"):
        io.read_bytes(ref)


def test_original_evidence_causality_and_versions_survive_projection(tmp_path):
    io = ArtifactIO(tmp_path)
    value = {
        "evidence": [
            {"id": "ev1", "source": "old.log", "line_no": 1, "end_line_no": 3, "raw_content": "original"},
            {"id": "ev2", "source": "old.log", "line_no": 4, "end_line_no": 5, "cause_id": "ev1"},
        ],
        "observations": [
            {
                "id": "obs1",
                "kind": "file_excerpt",
                "evidence_ids": ["ev2"],
                "data": {
                    "resolved_path": "/project/a.py",
                    "file_sha256": "a" * 64,
                    "start_line": 2,
                    "end_line": 3,
                    "content": "exact\nlines",
                },
            },
            {
                "id": "obs2",
                "kind": "path_status",
                "evidence_ids": [],
                "data": {"requested_path": "/missing", "status": "missing"},
            },
        ],
    }
    artifact = io.put_json("job", value)
    refs = io.index_evidence(artifact, task_id="t1", attempt_id="a1", workspace_revision="b" * 64)
    assert [r.origin_id for r in refs] == ["ev1", "ev2", "obs1", "obs2"]
    assert refs[1].relations == (("cause", "job:ev1"),)
    assert refs[2].relations == (("dependency", "job:ev2"),)
    assert refs[3].source_sha256 is None
    assert io.read_record(refs[0])["raw_content"] == "original"
    assert io.read_record(refs[2])["data"]["content"] == "exact\nlines"
    with pytest.raises(ValueError):
        io.read_record(replace(refs[0], record_key="evidence:unknown"))
    with pytest.raises(ValueError):
        io.read_record(replace(refs[0], record_key="../../file"))


def test_partial_and_invalid_writes_never_publish_reference(tmp_path):
    io = ArtifactIO(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        io.put_json("job", {"runtime": object()})
    with pytest.raises(ValueError):
        io.put_json("job", {"not_finite": float("nan")})
    assert not list(tmp_path.rglob("*.blob"))
