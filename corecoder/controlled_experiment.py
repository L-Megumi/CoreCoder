"""Paired factorial diagnosis experiments with attempt-level usage accounting."""

from __future__ import annotations

import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .experiments import ARMS
from .fault_injection import digest, write_json
from .source_visibility import visible_end_line


def schedule_runs(case_ids, repeats=2, seed=42):
    if len(case_ids) != len(set(case_ids)) or not case_ids or repeats not in (1, 2, 3):
        raise ValueError("Unique cases and 1..3 repeats required")
    rng = random.Random(seed)
    schedule = []
    for repeat in range(repeats):
        cases = list(case_ids)
        rng.shuffle(cases)
        for case in cases:
            arms = list(ARMS)
            rng.shuffle(arms)
            schedule.extend({"run_id": f"r{repeat + 1}-{case}-{arm}", "repeat": repeat + 1,
                             "case_id": case, "arm": arm} for arm in arms)
    return schedule


def score_localization(report, annotation):
    """Exact injected-line coverage, never a semantic cause-accuracy claim."""
    observed = set()
    for item in report.get("observations", []):
        data = item["data"]
        span = data.get("snippet") or data
        if (item["kind"] in ("source_location", "config_file", "file_excerpt")
                and data.get("file_sha256") == annotation["injected_target_sha256"]
                and any(span.get("start_line", 0) <= line <= visible_end_line(span)
                        for line in annotation["injected_lines"])):
            observed.add(item["id"])
    valid = report.get("status") == "completed"
    cited = {key for hypothesis in (report.get("diagnosis") or {}).get("hypotheses", [])
             for key in hypothesis["observation_ids"]}
    current = set(report.get("current_observation_ids", []))
    return {"valid_report": valid, "injected_line_observed": bool(observed),
            "valid_hypothesis_cites_current_injected_line": valid and bool(observed & cited & current),
            "matching_observation_ids": sorted(observed)}


class RequestLedger:
    """Persist before each transport request; no implicit retries or budget borrowing."""

    def __init__(self, path, max_total, max_run):
        self.path, self.max_total, self.max_run = Path(path), max_total, max_run
        if self.path.exists():
            raise ValueError("Existing ledger must never be reset")
        self.events = []
        self.save()

    def save(self):
        write_json(self.path, {"max_total": self.max_total, "max_run": self.max_run, "events": self.events})

    def attach(self, model, run_id):
        original = model._call_with_retry

        def request(params, max_retries=1):
            if len(self.events) >= self.max_total or sum(e["run_id"] == run_id for e in self.events) >= self.max_run:
                raise RuntimeError("Preregistered experiment request budget exhausted")
            event = {"number": len(self.events) + 1, "run_id": run_id, "status": "started",
                     "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "requested_model": params["model"], "has_tools": bool(params.get("tools")),
                     "tool_schema_chars": len(json.dumps(params.get("tools") or [], ensure_ascii=False)),
                     "prompt_tokens": 0, "completion_tokens": 0, "usage_received": False}
            self.events.append(event)
            self.save()
            started = time.monotonic()
            try:
                stream = original(params, max_retries=1)
            except Exception as exc:
                event.update(status="request_error", error_type=type(exc).__name__,
                             elapsed_seconds=time.monotonic() - started)
                self.save()
                raise

            def consume():
                try:
                    for chunk in stream:
                        if getattr(chunk, "model", None):
                            event["response_model"] = chunk.model
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            event.update(prompt_tokens=usage.prompt_tokens or 0,
                                         completion_tokens=usage.completion_tokens or 0, usage_received=True)
                            for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                                value = getattr(usage, name, None)
                                if value is not None:
                                    event[name] = value
                        yield chunk
                    event["status"] = "completed"
                except Exception as exc:
                    event.update(status="stream_error", error_type=type(exc).__name__)
                    raise
                finally:
                    event["elapsed_seconds"] = round(time.monotonic() - started, 4)
                    self.save()
                    close = getattr(stream, "close", None)
                    if close:
                        close()
            return consume()

        model._call_with_retry = request


def source_unchanged(case):
    root = Path(case["project"])
    return all((root / name).is_file() and digest(root / name) == expected
               for name, expected in case["manifest"].items())


def product_hashes(root):
    root = Path(root).resolve()
    return {path.relative_to(root).as_posix(): digest(path)
            for directory in ("corecoder", "examples") for path in (root / directory).rglob("*.py")}


def summarize(rows):
    result = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        result[arm] = {"runs": len(group),
                       **{key: sum(row["scores"][key] for row in group) for key in (
                           "valid_report", "injected_line_observed", "valid_hypothesis_cites_current_injected_line")},
                       "all_sources_unchanged": all(row["source_unchanged"] for row in group),
                       "all_provider_usage_complete": all(row["usage_complete"] for row in group)}
        for key in ("request_attempts", "prompt_tokens", "completion_tokens", "tool_calls", "tool_errors",
                    "tool_schema_chars", "elapsed_seconds"):
            values = [row[key] for row in group]
            result[arm][key] = {"total": sum(values), "mean": statistics.mean(values),
                                "median": statistics.median(values), "min": min(values), "max": max(values)}
    paired = {}
    base = {(row["case_id"], row["repeat"]): row for row in rows if row["arm"] == "baseline"}
    for arm in ("tools", "context", "both"):
        pairs = [(base[(row["case_id"], row["repeat"])], row) for row in rows
                 if row["arm"] == arm and (row["case_id"], row["repeat"]) in base]
        paired[arm] = {"pairs": len(pairs)}
        for key in ("request_attempts", "prompt_tokens", "completion_tokens", "tool_calls", "tool_schema_chars"):
            differences = [b[key] - a[key] for a, b in pairs]
            base_total = sum(a[key] for a, _ in pairs)
            paired[arm][key] = {"mean_paired_delta": statistics.mean(differences) if pairs else None,
                                "total_relative_change": sum(differences) / base_total if base_total else None}
    return {"arms": result, "paired_against_baseline": paired,
            "scope": "All scheduled failures included; exact source coverage is not semantic root-cause accuracy"}
