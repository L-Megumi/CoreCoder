"""Immutable catalog + original ArtifactIO, not a second storage backend or Checkpoint."""

from dataclasses import replace
import hashlib
import json
import os
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from .contracts import AccessScope, DocumentRef, EvidenceCatalog


class EvidenceStore:
    def __init__(self, artifacts, job_id, catalog=None):
        if not isinstance(artifacts, ArtifactIO):
            raise TypeError("ArtifactIO required")
        self.io = artifacts
        self.catalog = catalog or EvidenceCatalog(job_id=job_id)
        if self.catalog.job_id != job_id:
            raise ValueError("Foreign catalog")

    @classmethod
    def load(cls, artifacts, catalog_ref, job_id):
        if catalog_ref.job_id != job_id:
            raise ValueError("Cross-Job catalog read")
        return cls(
            artifacts,
            job_id,
            EvidenceCatalog.from_dict(artifacts.read_json(catalog_ref)),
        )

    def save(self):
        return self.io.put_json(self.catalog.job_id, self.catalog.to_dict())

    def add_record(self, ref):
        if ref.job_id != self.catalog.job_id:
            raise ValueError("Cross-Job evidence")
        original = self.io.index_evidence(
            ref.artifact,
            task_id=ref.producer_task_id,
            attempt_id=ref.producer_attempt_id,
            workspace_revision=ref.workspace_revision,
        )
        if ref not in original:
            raise ValueError("Forged evidence metadata")
        key = (ref.producer_task_id, ref.producer_attempt_id, ref.origin_id)
        prior = [
            r
            for r in self.catalog.records
            if (r.producer_task_id, r.producer_attempt_id, r.origin_id) == key
        ]
        if prior:
            if prior[0] != ref:
                raise ValueError("Conflicting immutable evidence")
            return ref
        self.catalog = replace(self.catalog, records=self.catalog.records + (ref,))
        return ref

    def add_document(
        self,
        raw,
        *,
        task_id,
        attempt_id,
        workspace_revision,
        kind,
        origin_id=None,
        source_path=None,
        source_start_line=None,
    ):
        if not isinstance(raw, bytes):
            raise TypeError("Original document bytes required")
        if len(raw) > 10_000_000:
            raise ValueError("Original document exceeds existing ArtifactIO bound")
        # Shared content hash under the Job, with separate immutable provenance values.
        identity = "context-" + hashlib.sha256(raw).hexdigest()
        shared = ArtifactIO(self.io.root)
        # An identical writer in another Worker must finish fsync before a reader
        # verifies its blob; O_EXCL alone publishes the name before the bytes.
        import fcntl

        lock_path = shared._path(f"{self.catalog.job_id}/{identity}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(
            os.open(
                lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
            ),
            "a+b",
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                artifact = shared.put_bytes(
                    self.catalog.job_id,
                    raw,
                    artifact_id=identity,
                    media_type="text/plain",
                )
            except FileExistsError:
                artifact = ArtifactRef(
                    job_id=self.catalog.job_id,
                    artifact_id=identity,
                    storage_key=f"{self.catalog.job_id}/{identity}.blob",
                    sha256=hashlib.sha256(raw).hexdigest(),
                    size_bytes=len(raw),
                    media_type="text/plain",
                )
                shared.read_bytes(artifact)
        key = json.dumps(
            [
                task_id,
                attempt_id,
                workspace_revision,
                kind,
                artifact.sha256,
                origin_id,
                source_path,
                source_start_line,
            ]
        )
        document = DocumentRef(
            document_id="document-" + hashlib.sha256(key.encode()).hexdigest(),
            job_id=self.catalog.job_id,
            task_id=task_id,
            attempt_id=attempt_id,
            workspace_revision=workspace_revision,
            kind=kind,
            artifact=artifact,
            origin_id=origin_id,
            source_path=source_path,
            source_start_line=source_start_line,
        )
        if document not in self.catalog.documents:
            self.catalog = replace(
                self.catalog, documents=self.catalog.documents + (document,)
            )
        return document

    def _authorize(self, scope, task, attempt, revision):
        if not isinstance(scope, AccessScope) or scope.job_id != self.catalog.job_id:
            raise ValueError("Cross-Job scope")
        if (task, attempt) not in {*scope.producers, (scope.task_id, scope.attempt_id)}:
            raise ValueError("Producer not authorized")
        if revision != scope.workspace_revision and not scope.allow_historical:
            raise ValueError("Stale source version")

    def query(self, scope, *, task_id=None, attempt_id=None, evidence_id=None):
        matches = []
        for ref in self.catalog.records:
            if task_id is not None and ref.producer_task_id != task_id:
                continue
            if attempt_id is not None and ref.producer_attempt_id != attempt_id:
                continue
            if evidence_id is not None and evidence_id not in (
                ref.origin_id,
                ref.ref_id,
            ):
                continue
            try:
                self._authorize(
                    scope,
                    ref.producer_task_id,
                    ref.producer_attempt_id,
                    ref.workspace_revision,
                )
            except ValueError:
                continue
            matches.append(ref)
        return tuple(matches)

    def restore_record(self, ref, scope):
        if ref not in self.catalog.records:
            raise ValueError("Unregistered/forged evidence reference")
        self._authorize(
            scope, ref.producer_task_id, ref.producer_attempt_id, ref.workspace_revision
        )
        record = self.io.read_record(ref)
        return dict(
            record=record,
            reference=ref.to_dict(),
            current_eligible=ref.workspace_revision == scope.workspace_revision,
            classification="observed_record"
            if ref.workspace_revision == scope.workspace_revision
            else "historical_observation",
        )

    def chain(self, ref, scope):
        pending = [ref]
        seen = set()
        result = []
        while pending:
            current = pending.pop()
            key = (
                current.producer_task_id,
                current.producer_attempt_id,
                current.origin_id,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(self.restore_record(current, scope))
            for relation, target in current.relations:
                target_id = target.removeprefix(current.job_id + ":")
                ancestors = self.query(
                    scope,
                    task_id=current.producer_task_id,
                    attempt_id=current.producer_attempt_id,
                    evidence_id=target_id,
                )
                if not ancestors:
                    raise ValueError(
                        "Missing causal/source relation; cannot silently drop it"
                    )
                pending.extend(ancestors)
        return tuple(result)

    def restore_document(self, ref, scope, *, start_line=1, max_lines=None):
        if ref not in self.catalog.documents:
            raise ValueError("Unregistered/forged original document")
        self._authorize(scope, ref.task_id, ref.attempt_id, ref.workspace_revision)
        if (
            type(start_line) is not int
            or start_line < 1
            or (max_lines is not None and (type(max_lines) is not int or max_lines < 1))
        ):
            raise ValueError("Invalid recovery window")
        raw = self.io.read_bytes(ref.artifact)
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        excerpt = "".join(
            lines[start_line - 1 :]
            if max_lines is None
            else lines[start_line - 1 : start_line - 1 + max_lines]
        )
        return dict(
            content=excerpt,
            original_sha256=ref.artifact.sha256,
            reference=ref.to_dict(),
            source_start_line=(ref.source_start_line + start_line - 1)
            if ref.source_start_line
            else None,
            complete=start_line == 1 and (max_lines is None or max_lines >= len(lines)),
            current_eligible=ref.workspace_revision == scope.workspace_revision,
            classification="unverified_hypothesis"
            if ref.kind == "hypothesis"
            else "observed_record"
            if ref.workspace_revision == scope.workspace_revision
            else "historical_observation",
        )
