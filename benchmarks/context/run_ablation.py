"""Four deterministic views, held-out answers, raw receipts and optional real DAG oracle."""

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import time

from corecoder.context import estimate_tokens
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context import EvidenceStore, AccessScope
from researchcoder.context.policy import ContextPolicy
from researchcoder.context.compression import CompressionPipeline
from .fixtures import materialize
from .probe import diagnose, score


STRATEGIES = {"A": "original", "B": "truncate", "C": "summary", "D": "evidence"}


def data_units(messages):
    """Content-bearing units, excluding provenance/control overhead from duplicate accounting."""
    units = []
    for message in messages:
        if message.get("role") == "system":
            continue
        text = str(message.get("content", ""))
        if text.startswith("[Evidence context:"):
            payload = json.loads(text.split("\n", 1)[1])
            for card in payload["records"]:
                body = {
                    k: v
                    for k, v in card.items()
                    if k
                    not in (
                        "id",
                        "document_id",
                        "reference",
                        "workspace_revision",
                        "classification",
                    )
                }
                units.append(json.dumps(body, ensure_ascii=False, sort_keys=True))
        elif message.get("role") == "tool" and '"context_ref"' in text:
            continue
        else:
            units.extend(line for line in text.splitlines() if line.strip())
    return units


def duplicate_ratio(views):
    seen = set()
    total = duplicate = 0
    for view in views:
        for unit in data_units(view):
            digest = hashlib.sha256(unit.encode()).hexdigest()
            total += len(unit)
            if digest in seen:
                duplicate += len(unit)
            else:
                seen.add(digest)
    return duplicate / max(1, total)


def run(directory, *, max_tokens=4000, include_dag=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).with_name("cases.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        root = directory / "fixtures" / case["id"]
        messages, state = materialize(case["scenario"], root)
        version = hashlib.sha256(
            json.dumps(
                {
                    p.relative_to(root).as_posix(): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in root.rglob("*.py")
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for arm, strategy in STRATEGIES.items():
            io = ArtifactIO(directory / "artifacts" / case["id"] / arm)
            view_list = []
            receipts = []
            error = None
            started = time.perf_counter()
            for role in ("log_analysis", "code_analysis"):
                store = EvidenceStore(io, "benchmark")
                scope = AccessScope(
                    job_id="benchmark",
                    task_id=role,
                    attempt_id="a",
                    workspace_revision=version,
                )
                pipeline = CompressionPipeline(
                    store, scope, ContextPolicy(strategy=strategy, role=role)
                )
                try:
                    view = pipeline.build(messages, max_tokens, original_state=state)
                except (ValueError, RuntimeError) as exc:
                    error = type(exc).__name__
                    view_list.append([])
                    continue
                view_list.append(list(view.messages))
                restored = 0
                for ref in store.catalog.documents:
                    raw = store.restore_document(ref, scope)["content"].encode()
                    assert hashlib.sha256(raw).hexdigest() == ref.artifact.sha256
                    restored += 1
                receipts.append(
                    dict(
                        receipt=view.receipt_ref.to_dict(),
                        catalog=view.catalog_ref.to_dict(),
                        restored_documents=restored,
                        statistics=view.statistics,
                    )
                )
            build_seconds = time.perf_counter() - started
            task_started = time.perf_counter()
            result = diagnose(view_list[-1])
            task_seconds = time.perf_counter() - task_started
            visible = "\n".join(str(m.get("content", "")) for m in view_list[-1])
            metrics = score(result, case, visible)
            row = dict(
                case_id=case["id"],
                arm=arm,
                strategy=strategy,
                input_cap=max_tokens,
                error=error,
                before_tokens=estimate_tokens(messages),
                after_tokens=estimate_tokens(view_list[-1]),
                build_seconds=build_seconds,
                task_seconds=task_seconds,
                diagnosis=result,
                **metrics,
                recoverability=1.0 if receipts else 0.0,
                duplicate_context_ratio=duplicate_ratio(view_list),
                receipts=receipts,
                role_views=view_list,
            )
            rows.append(row)
    summary = []
    for arm, strategy in STRATEGIES.items():
        selected = [r for r in rows if r["arm"] == arm]
        summary.append(
            dict(
                arm=arm,
                strategy=strategy,
                cases=len(selected),
                retained=sum(r["retained_count"] for r in selected),
                expected=sum(r["expected_count"] for r in selected),
                evidence_retention=sum(r["retained_count"] for r in selected)
                / sum(r["expected_count"] for r in selected),
                mean_before_tokens=sum(r["before_tokens"] for r in selected)
                / len(selected),
                mean_after_tokens=sum(r["after_tokens"] for r in selected)
                / len(selected),
                build_seconds=sum(r["build_seconds"] for r in selected),
                task_seconds=sum(r["task_seconds"] for r in selected),
                answer_oracle_passed=sum(r["answer_oracle_passed"] for r in selected),
                diagnosis_matches=sum(r["diagnosis_match"] for r in selected),
                location_matches=sum(r["location_match"] for r in selected),
                construction_failures=sum(r["error"] is not None for r in selected),
                mean_duplicate_ratio=sum(r["duplicate_context_ratio"] for r in selected)
                / len(selected),
                recovered_documents=sum(
                    x["restored_documents"] for r in selected for x in r["receipts"]
                ),
            )
        )
    dag = []
    if include_dag:
        from examples.multi_agent_autograd_demo import prepare_demo
        from researchcoder.orchestration import DAGScheduler

        for arm, strategy in STRATEGIES.items():
            plan, cfg, io, initial = prepare_demo(directory / "autograd" / arm)
            templates = {
                k: replace(
                    t,
                    config=replace(
                        t.config,
                        context_policy=ContextPolicy(
                            strategy=strategy, role=t.role, allow_recovery=True
                        ),
                    ),
                )
                for k, t in cfg.templates.items()
            }
            scheduler = DAGScheduler()
            outcome = scheduler.execute(plan, replace(cfg, templates=templates))
            checks = [r for r in outcome.results if r.kind == "verify"]
            contexts = []
            for result in outcome.results:
                for ref in result.outputs:
                    value = io.read_json(ref)
                    if isinstance(value, dict) and "compression_receipts" in value:
                        contexts.extend(value["statistics"])
            execution = io.read_json(outcome.summary_ref)
            dag.append(
                dict(
                    arm=arm,
                    strategy=strategy,
                    initial_status=initial["status"],
                    status=outcome.status,
                    independent_gradient_oracle=checks[-1].verification.status
                    if checks
                    else "not_run",
                    elapsed_seconds=outcome.elapsed_seconds,
                    context_before_tokens=sum(c["before_tokens"] for c in contexts),
                    context_after_tokens=sum(c["after_tokens"] for c in contexts),
                    context_build_seconds=sum(c["elapsed_seconds"] for c in contexts),
                    worker_context_receipts=len(contexts),
                    result=outcome.to_dict(),
                    execution=execution,
                )
            )
    output = dict(
        schema_version=1,
        mode="offline_deterministic_context_probe_and_scripted_dag",
        provider_requests=0,
        provider_usage=None,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        token_semantics="estimated characters/3; no real model accuracy or billing claim",
        summary_strategy="Existing generic ContextManager deterministic extractive fallback; no paid/model-generated summary.",
        duplicate_definition="Repeated content-bearing character units / all supplied units across two role views; references/control overhead excluded.",
        fixture_oracle="Independent expected evidence/fault/source answers; not patch execution. Separate DAG rows run the held-out PyTorch gradient oracle.",
        summary=summary,
        rows=rows,
        autograd_dag=dag,
    )
    (directory / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-tokens", type=int, default=4000)
    parser.add_argument("--include-dag", action="store_true")
    args = parser.parse_args()
    output = run(
        args.output_dir, max_tokens=args.input_tokens, include_dag=args.include_dag
    )
    print(
        json.dumps(
            dict(
                summary=output["summary"],
                autograd=[
                    {k: v for k, v in r.items() if k not in ("result", "execution")}
                    for r in output["autograd_dag"]
                ],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
