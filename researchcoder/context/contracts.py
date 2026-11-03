"""Versioned context references; authority is supplied by the Controller."""

from dataclasses import dataclass
from typing import Literal
from researchcoder.contracts import (
    Contract,
    ArtifactRef,
    EvidenceRef,
    identifier,
    digest,
)


@dataclass(frozen=True, kw_only=True)
class AccessScope(Contract):
    job_id: str
    task_id: str
    attempt_id: str
    workspace_revision: str
    producers: tuple[tuple[str, str], ...] = ()
    allow_historical: bool = False
    schema_version: int = 1

    def validate(self):
        for v in (self.job_id, self.task_id, self.attempt_id):
            identifier(v)
        digest(self.workspace_revision)
        for task, attempt in self.producers:
            identifier(task)
            identifier(attempt)


@dataclass(frozen=True, kw_only=True)
class DocumentRef(Contract):
    document_id: str
    job_id: str
    task_id: str
    attempt_id: str
    workspace_revision: str
    kind: Literal["log", "source", "verification", "tool_output", "hypothesis"]
    artifact: ArtifactRef
    origin_id: str | None = None
    source_path: str | None = None
    source_start_line: int | None = None
    schema_version: int = 1

    def validate(self):
        for v in (self.document_id, self.job_id, self.task_id, self.attempt_id):
            identifier(v)
        digest(self.workspace_revision)
        if self.artifact.job_id != self.job_id:
            raise ValueError("Foreign original document")
        if self.origin_id is not None:
            identifier(self.origin_id)
        if self.source_path is not None and (
            not self.source_path
            or len(self.source_path) > 4096
            or "\x00" in self.source_path
        ):
            raise ValueError("Invalid source location metadata")
        if self.source_start_line is not None and (
            type(self.source_start_line) is not int or self.source_start_line < 1
        ):
            raise ValueError("Invalid source page offset")


@dataclass(frozen=True, kw_only=True)
class EvidenceCatalog(Contract):
    job_id: str
    records: tuple[EvidenceRef, ...] = ()
    documents: tuple[DocumentRef, ...] = ()
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        if any(r.job_id != self.job_id for r in (*self.records, *self.documents)):
            raise ValueError("Cross-Job catalog")
        keys = [
            (r.producer_task_id, r.producer_attempt_id, r.origin_id)
            for r in self.records
        ]
        if len(keys) != len(set(keys)) or len(
            {d.document_id for d in self.documents}
        ) != len(self.documents):
            raise ValueError("Duplicate catalog provenance")
