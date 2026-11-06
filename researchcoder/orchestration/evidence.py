"""Job-scoped immutable references with compound provenance; not long-term memory."""

from dataclasses import dataclass
import hashlib
import json
from researchcoder.handoff_codec import INDEXED_VIEW, index_references
from corecoder.citation_scope import UPSTREAM_SCOPE
from researchcoder.contracts import Contract, EvidenceRef, ArtifactRef, identifier, digest


@dataclass(frozen=True, kw_only=True)
class EvidenceItem(Contract):
    exchange_id: str
    reference: EvidenceRef
    classification: str = "observed_record"
    schema_version: int = 1

    def validate(self):
        identifier(self.exchange_id)
        if self.classification != "observed_record":
            raise ValueError("Summaries cannot overwrite observed records")


@dataclass(frozen=True, kw_only=True)
class EvidenceBundle(Contract):
    job_id: str
    workspace_revision: str
    items: tuple[EvidenceItem, ...]
    hypothesis_artifacts: tuple[ArtifactRef, ...]
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        digest(self.workspace_revision)
        if len({i.exchange_id for i in self.items}) != len(self.items):
            raise ValueError("Duplicate exchange identity")
        if any(
            i.reference.job_id != self.job_id or i.reference.workspace_revision != self.workspace_revision for i in self.items
        ):
            raise ValueError("Mixed evidence versions or jobs")
        if any(a.job_id != self.job_id for a in self.hypothesis_artifacts):
            raise ValueError("Foreign hypothesis artifact")


class EvidenceContextCapacityError(ValueError):
    """A bounded view cannot fit; not evidence of a source or permission change."""


class EvidenceExchange:
    def __init__(self, job_id, artifacts):
        self.job_id, self.io = job_id, artifacts
        self._results = {}

    def submit(self, result, expected_revision):
        if result.job_id != self.job_id or result.task_id in self._results:
            raise ValueError("Foreign or duplicate producer")
        items = []
        seen = set()
        indexed = {}
        for ref in result.evidence_refs:
            if (ref.producer_task_id, ref.producer_attempt_id, ref.job_id, ref.workspace_revision) != (
                result.task_id,
                result.attempt_id,
                self.job_id,
                expected_revision,
            ):
                raise ValueError("Evidence provenance mismatch")
            if ref.artifact not in result.outputs:
                raise ValueError("Original evidence artifact not retained in result")
            if ref.artifact.artifact_id not in indexed:
                indexed[ref.artifact.artifact_id] = self.io.index_evidence(
                    ref.artifact, task_id=result.task_id, attempt_id=result.attempt_id, workspace_revision=expected_revision
                )
            if ref not in indexed[ref.artifact.artifact_id]:
                raise ValueError("Evidence metadata disagrees with its original record")
            self.io.read_record(ref)
            key = ":".join((ref.job_id, result.task_id, result.attempt_id, ref.origin_id))
            if key in seen:
                raise ValueError("Duplicate original record within producer")
            seen.add(key)
            items.append(EvidenceItem(exchange_id="exchange-" + hashlib.sha256(key.encode()).hexdigest(), reference=ref))
        for artifact in result.outputs:
            self.io.read_bytes(artifact)
        self._results[result.task_id] = (result, tuple(items))

    def bundle(self, producers, expected_revision):
        if len(set(producers)) != len(producers):
            raise ValueError("Duplicate authorized producers")
        items, hypotheses, seen_sources = [], [], {}
        for producer in producers:
            if producer not in self._results:
                raise ValueError("Producer is not authorized/registered")
            result, refs = self._results[producer]
            for item in refs:
                ref = item.reference
                if ref.workspace_revision != expected_revision:
                    raise ValueError("Stale evidence cannot describe current candidate")
                self.io.read_record(ref)
                if ref.source_sha256:
                    source = (ref.source_path, ref.start_line, ref.end_line)
                    if source in seen_sources and seen_sources[source] != ref.source_sha256:
                        raise ValueError("Conflicting source observations under one revision")
                    seen_sources[source] = ref.source_sha256
                # Preserve all original IDs, including identical IDs from different Tasks.
                items.append(item)
            if result.outputs:
                raw = self.io.read_json(result.outputs[0])
                if isinstance(raw, dict) and raw.get("diagnosis"):
                    hypotheses.append(result.outputs[0])
        return EvidenceBundle(
            job_id=self.job_id, workspace_revision=expected_revision, items=tuple(items), hypothesis_artifacts=tuple(hypotheses)
        )

    def context(self, bundle, *, max_chars=50000, compact_excerpts=False, indexed_refs=False):
        if type(max_chars) is not int or not 1000 <= max_chars <= 50000 or type(compact_excerpts) is not bool:
            raise ValueError("Expected bounded controller-selected view options")
        if type(indexed_refs) is not bool or (indexed_refs and not compact_excerpts):
            raise ValueError("Indexed references require a bounded handoff")
        # Only controller-registered producer results may be rendered. A caller
        # cannot manufacture a bundle with self-declared producer identities.
        producers = tuple(dict.fromkeys(i.reference.producer_task_id for i in bundle.items))
        if bundle.job_id != self.job_id or self.bundle(producers, bundle.workspace_revision) != bundle:
            raise ValueError("Unregistered or forged evidence bundle")
        observed = []
        for item in bundle.items:
            raw = self.io.read_record(item.reference)
            # References always survive; prompt excerpts are explicitly bounded views.
            observed.append(
                dict(
                    exchange_id=item.exchange_id,
                    reference=item.reference.to_dict(),
                    classification="observed_record",
                    citation_scope=UPSTREAM_SCOPE,
                    excerpt=json.dumps(raw, ensure_ascii=False)[:1200],
                )
            )
        claims = []
        for ref in bundle.hypothesis_artifacts:
            report = self.io.read_json(ref)
            claims.append(
                dict(
                    classification="unverified_hypothesis",
                    artifact=ref.to_dict(),
                    hypotheses=report["diagnosis"].get("hypotheses", []),
                )
            )
        view = dict(schema_version=1, workspace_revision=bundle.workspace_revision,
                    citation_scope=UPSTREAM_SCOPE, observations=observed, hypotheses=claims)
        if compact_excerpts:
            excerpt_policy = dict(version=INDEXED_VIEW if indexed_refs else "bounded-excerpts-v1", prefix_chars=1200)
            view["excerpt_policy"] = excerpt_policy
            if indexed_refs:
                view["producer_table"] = index_references(observed)
        context = json.dumps(view, ensure_ascii=False)
        if len(context) > max_chars and compact_excerpts:
            originals = [row["excerpt"] for row in observed]
            def render(limit):
                for row, original in zip(observed, originals):
                    row["excerpt"] = original[:limit]
                excerpt_policy["prefix_chars"] = limit
                return json.dumps(view, ensure_ascii=False)
            context = render(0)
            if len(context) > max_chars:
                raise EvidenceContextCapacityError("Evidence reference metadata exceeds bounded view")
            low, high = 0, 1200
            while low < high:
                middle = (low + high + 1) // 2
                if len(render(middle)) <= max_chars:
                    low = middle
                else:
                    high = middle - 1
            context = render(low)
        if len(context) > max_chars:
            raise EvidenceContextCapacityError("Evidence context exceeds bounded view; narrow authorized dependencies")
        return context
