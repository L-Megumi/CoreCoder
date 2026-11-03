"""Recomputable complexity slices and exact-evidence diagnostics, not model judging."""

import hashlib
import json
import re
from researchcoder.contracts import ArtifactRef
from researchcoder.adapters.artifacts import ArtifactIO
from .contracts import ComplexityManifest


def text_nodes(value):
    if isinstance(value, dict):
        return "\n".join(text_nodes(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(text_nodes(v) for v in value)
    if not isinstance(value, str):
        return ""
    text = re.sub(r"(?m)^\d+\t", "", value).replace("\r\n", "\n")
    if value.lstrip().startswith(("{", "[")):
        try:
            return text + "\n" + text_nodes(json.loads(value))
        except (ValueError, RecursionError):
            pass
    return text


def trial_diagnostics(raw, io, label):
    from .dataset import DATA
    roots = {}
    for site in label.evidence_sites:
        if site.kind != "source":
            continue
        content = (DATA / label.case_id / "faulty" / site.path).read_text()
        lines = content.splitlines()
        roots[site.path + ":" + str(site.start_line)] = "\n".join(lines[site.start_line - 1:site.end_line])
    observed, retained = set(), set()
    opportunities = preserved = request_count = errors = duplicate_reads = reads = 0
    seen = set()
    context_stats = []
    recoverable = total_refs = 0
    for task in raw.get("task_results", []):
        for ref in task.get("evidence_refs", []):
            from researchcoder.contracts import EvidenceRef
            total_refs += 1
            io.read_record(EvidenceRef.from_dict(ref))
            recoverable += 1
        for artifact in task.get("outputs", []):
            if artifact["media_type"] != "application/json":
                continue
            record = io.read_json(ArtifactRef.from_dict(artifact))
            if "request_records" in record:
                for reference in record["request_records"]:
                    request = io.read_json(ArtifactRef.from_dict(reference))["request"]
                    request_count += 1
                    before = text_nodes(request.get("before_context_messages"))
                    after = text_nodes(request["messages"])
                    present = {key for key, span in roots.items() if span and span in before}
                    survived = {key for key in present if roots[key] in after}
                    observed |= present
                    retained |= survived
                    opportunities += len(present)
                    preserved += len(survived)
            if "compression_receipts" in record:
                context_stats.extend(record["statistics"])
            observations = {o["id"]: o for o in record.get("observations", [])}
            for event in record.get("tool_events", record.get("events", [])):
                errors += event.get("status") == "error"
                if event.get("tool") not in ("read_file", "read_project_file"):
                    continue
                roots_by_id = [observations[i] for i in event.get("observation_ids", []) if i in observations]
                if roots_by_id:
                    for observation in roots_by_id:
                        data = observation["data"]
                        span = data.get("snippet") or data
                        path = data.get("requested_path") or data.get("resolved_path", "")
                        path = re.split(r"/(?:source-snapshot(?:-r[0-9]+)?|candidate)/", str(path).replace("\\", "/"))[-1]
                        content = span.get("content")
                        if not isinstance(content, str):
                            continue
                        key = (path, span.get("start_line", 1), hashlib.sha256(content.encode()).hexdigest())
                        reads += 1
                        duplicate_reads += key in seen
                        seen.add(key)
                elif isinstance(event.get("result"), str) and not event["result"].startswith("Error"):
                    path = event.get("arguments", {}).get("file_path", "").replace("\\", "/").split("/candidate/")[-1]
                    content = re.sub(r"^\[Unchanged read reused[^\n]*\n", "", event["result"])
                    content = re.sub(r"(?m)^\d+\t", "", content)
                    key = (path, 1, hashlib.sha256(content.encode()).hexdigest())
                    reads += 1
                    duplicate_reads += key in seen
                    seen.add(key)
    return dict(effective_request_artifacts=request_count,
        required_source_spans=len(roots), exposed_source_spans=len(observed), retained_source_spans=len(retained),
        source_exposure_coverage=len(observed) / len(roots) if roots else None,
        exact_span_retention_when_present=preserved / opportunities if opportunities else None,
        retention_opportunities=opportunities, recoverable_evidence_refs=recoverable, evidence_refs=total_refs,
        comparable_read_observations=reads, duplicate_read_observations=duplicate_reads, invalid_tool_results=errors,
        safety_denials=len(raw.get("policy_violations", [])),
        context_views=len(context_stats), context_build_seconds=sum(s["elapsed_seconds"] for s in context_stats),
        estimated_context_before=sum(s["before_tokens"] for s in context_stats),
        estimated_context_after=sum(s["after_tokens"] for s in context_stats),
        metric_scope="Exact frozen source-span representation and read-excerpt equality, not semantic sufficiency or proof a repeated read was unnecessary")


def enrich_summary(directory, run, manifest, results, summary):
    labels = ComplexityManifest.from_json((directory / "complexity-manifest.json").read_text())
    versions = json.loads((directory / "versions.json").read_text())
    identity = hashlib.sha256(labels.to_json().encode()).hexdigest()
    if identity != versions["complexity_manifest_sha256"] or labels.benchmark_manifest_sha256 != run.manifest_sha256:
        raise ValueError("Frozen complexity revision mismatch")
    by_case = {c.case_id: c for c in labels.cases if c.benchmark_subset == "complexity_benchmark"}
    for case in manifest.cases:
        if by_case[case.case_id].case_sha256 != hashlib.sha256(case.to_json().encode()).hexdigest():
            raise ValueError("Complexity label/case identity mismatch")
    slices = {}
    diagnostics = {}
    for level in (2, 3):
        modes = {}
        for mode in dict.fromkeys(t.mode for t in run.trials):
            planned = [t for t in run.trials if t.mode == mode and by_case[t.case_id].level == level]
            rows = [r for r in results if r.trial in planned]
            modes[mode] = dict(planned=len(planned), recorded=len(rows),
                succeeded=sum(r.metrics.end_to_end_success for r in rows),
                incomplete=sum(r.status == "incomplete" for r in rows),
                failed=sum(r.status == "failed" for r in rows),
                cancelled=sum(r.status == "cancelled" for r in rows),
                e2e_rate=sum(r.metrics.end_to_end_success for r in rows) / len(planned) if planned else None)
        slices[str(level)] = modes
    for result in results:
        io = ArtifactIO(directory / "trials" / result.trial.trial_id / "artifacts")
        raw = io.read_json(result.raw)
        # Diagnostics are saved at collection time, with frozen fixture spans.
        diagnostics[result.trial.trial_id] = raw.get("complexity_diagnostics")
    summary["complexity"] = dict(manifest_sha256=identity, levels=slices, per_trial_diagnostics=diagnostics,
        score_type="Independent dimensions; no scalar fitted to Agent success",
        performance_claim="Scripted runs validate mechanisms only; live evidence requires separate authorization")
    return summary
