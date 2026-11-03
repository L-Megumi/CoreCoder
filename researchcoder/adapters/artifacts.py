"""Small job-scoped, content-checked file storage; not a checkpoint service."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path

from researchcoder.contracts import ArtifactRef, EvidenceRef, _unique, identifier


class ArtifactIO:
    def __init__(self, root, *, namespace=None):
        given = Path(root).absolute()
        if given.is_symlink():
            raise ValueError("Artifact root cannot be a symlink")
        given.mkdir(parents=True, exist_ok=True)
        self.root = given.resolve()
        if namespace is not None:
            identifier(namespace)
        self.namespace = namespace

    def _path(self, key):
        # ArtifactRef has already checked lexical normalization; also check the filesystem.
        target = self.root / key
        for parent in [target, *target.parents]:
            if parent == self.root:
                break
            if parent.is_symlink():
                raise ValueError("Symlink artifact paths are forbidden")
        if not target.resolve().is_relative_to(self.root):
            raise ValueError("Artifact path escaped storage")
        return target

    def put_bytes(self, job_id, data, *, artifact_id=None, media_type="application/json"):
        if type(data) is not bytes:
            raise TypeError("Artifact content must be bytes")
        identifier(job_id)
        artifact_id = artifact_id or "artifact-" + uuid.uuid4().hex
        identifier(artifact_id)
        ref = ArtifactRef(
            job_id=job_id,
            artifact_id=artifact_id,
            storage_key=f"{job_id}/" + (f"tasks/{self.namespace}/" if self.namespace else "") + f"{artifact_id}.blob",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )
        path = self._path(ref.storage_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = self._path(ref.storage_key)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish bytes and the directory entries before a durable metadata
        # transaction may commit this reference. No overwrite or second store.
        directory = path.parent
        while directory.is_relative_to(self.root):
            directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if directory == self.root:
                break
            directory = directory.parent
        return ref

    def put_json(self, job_id, value, **kwargs):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
        return self.put_bytes(job_id, raw, **kwargs)

    def read_bytes(self, ref):
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ArtifactRef required, not a model path")
        path = self._path(ref.storage_key)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Artifact must be a regular file")
            raw = stream.read(10_000_001)
        if len(raw) != ref.size_bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise ValueError("Artifact digest/size mismatch")
        return raw

    def read_json(self, ref):
        return json.loads(self.read_bytes(ref).decode("utf-8-sig"), object_pairs_hook=_unique)

    def resolve(self, ref):
        self.read_bytes(ref)
        return self._path(ref.storage_key)

    def read_record(self, ref):
        if not isinstance(ref, EvidenceRef):
            raise TypeError("EvidenceRef required")
        value = self.read_json(ref.artifact)
        group, separator, identity = ref.record_key.partition(":")
        if separator != ":" or identity != ref.origin_id or group not in ("evidence", "observations"):
            raise ValueError("Invalid evidence record key")
        records = value.get(group) if isinstance(value, dict) else None
        if not isinstance(records, list):
            raise TypeError("No record collection")
        selected = [record for record in records if isinstance(record, dict) and record.get("id") == identity]
        if len(selected) != 1:
            raise ValueError("Record missing or ambiguous")
        return selected[0]

    def index_evidence(self, artifact, *, task_id, attempt_id, workspace_revision):
        value = self.read_json(artifact)
        if not isinstance(value, dict):
            raise TypeError("Run artifact must be an object")
        refs = []
        for group in ("evidence", "observations"):
            records = value.get(group, [])
            if not isinstance(records, list):
                raise TypeError("Invalid run record collection")
            ids = [r["id"] for r in records]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate original record IDs")
            for record in records:
                identity = record["id"]
                ref_id = f"{artifact.job_id}:{identity}"
                relations = []
                if group == "evidence":
                    data = record
                    kind = "log_evidence"
                    source = data.get("source")
                    start, end = data.get("line_no"), data.get("end_line_no")
                    for key, relation in [("cause_id", "cause"), ("context_id", "context")]:
                        if data.get(key):
                            relations.append((relation, f"{artifact.job_id}:{data[key]}"))
                else:
                    data = record["data"]
                    kind = "path_observation" if record["kind"] == "path_status" else "source_observation"
                    source = data.get("resolved_path") or data.get("requested_path")
                    span = data.get("snippet") or data
                    start, end = span.get("start_line"), span.get("visible_end_line", span.get("end_line"))
                    relations.extend(
                        ("dependency", f"{artifact.job_id}:{i}")
                        for i in (*record.get("evidence_ids", []), *data.get("dependency_observation_ids", []))
                    )
                if not start or not end or end < start:
                    start = end = None
                refs.append(
                    EvidenceRef(
                        job_id=artifact.job_id,
                        ref_id=ref_id,
                        origin_id=identity,
                        record_type=kind,
                        artifact=artifact,
                        record_key=f"{group}:{identity}",
                        producer_task_id=task_id,
                        producer_attempt_id=attempt_id,
                        source_path=source,
                        start_line=start,
                        end_line=end,
                        source_sha256=data.get("file_sha256"),
                        workspace_revision=workspace_revision,
                        relations=tuple(relations),
                    )
                )
        return tuple(refs)
