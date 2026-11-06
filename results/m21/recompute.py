"""Project allowlisted trial metadata from frozen M19/M20/M21 archives; no API access."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


CASES = {"packaging-requirement-hash", "sam-checkpoint-resume", "jsonpointer_scalar",
         "itsdangerous_future", "stream_utf8_chunks", "stable_shard_merge"}
STATES = {"not_started", "started", "succeeded", "failed", "incomplete", "cancelled",
          "cancelled_before_start", "not_started_cancelled"}
GUARDS = {"outside_workspace", "public_read_path_mismatch", "write_not_allowed", "sensitive_target",
          "unknown_target", "path_unavailable", "correction_exhausted", "policy_disabled",
          "current_read_required", "source_changed_since_read", "source_digest_unavailable",
          "tool_not_allowed", "guard_rejected"}
FAILURES = set(("operator_cancel controller_interrupted controller_error controller_trial_exception "
    "permission_violation policy_violation safety_or_permission_violation source_changed "
    "source_or_policy_version_mismatch artifact_integrity_failure trial_usage_unknown "
    "worker_response_lost_usage_unknown pending_request_after_controller_finish provider_parameters_mismatch "
    "context_admission_mismatch unregistered_job scope_identity_mismatch duplicate_transport attempt_scope_mismatch "
    "request_admission_exhausted token_admission_exhausted cost_admission_exhausted job_request_admission_exhausted "
    "job_token_admission_exhausted usage_unknown consecutive_api_errors nonthinking_usage_mismatch "
    "output_limit_mismatch provider_interrupted actual_usage_exceeded_admission actual_job_usage_exceeded_admission "
    "duplicate_settlement response_metadata_invalid permission_receipt_invalid permission_receipt_conflict "
    "correction_allowance_exhausted routing_stopped unknown incomplete interrupted budget_exhausted timeout "
    "independent_checks_failed verification_failed invalid_input failed model_error invalid_json "
    "response_length_exceeded cancelled accounting_unknown worker_crash controller_or_process_anomaly "
    "no_new_evidence consultation_context_unavailable consultation_no_evidence").split())
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def safe(root, key):
    path = (root / key).resolve()
    require(path != root and path.is_relative_to(root), "Record path escaped scope")
    return path


def artifact(directory, ref):
    path = safe(directory / "artifacts", ref["storage_key"])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"] and len(raw) == ref["size_bytes"], "Artifact mismatch")
    return json.loads(raw)


def code(value, allowed):
    return None if value is None else value if value in allowed else "other_recorded"


def count(value):
    require(type(value) is int and value >= 0, "Invalid counter")
    return value


def flag(value):
    require(value is None or type(value) is bool, "Invalid flag")
    return value


def transport(state, started, rates):
    known = Counter()
    statuses = Counter()
    cache_missing = 0
    for record in state.get("requests", {}).values():
        status = record["status"]
        require(status in ("known", "pending", "unknown"), "Unrecognized request status")
        statuses[status] += 1
        if status == "known":
            usage = record["usage"]
            require(record["usage_complete"] is True, "Known request lacks usage")
            for key in TOKEN_KEYS[:3]:
                known[key] += count(usage[key])
            require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "Inconsistent usage")
            if record.get("cache_usage_complete") is True:
                for key in TOKEN_KEYS[3:]:
                    known[key] += count(usage[key])
                require(sum(usage[k] for k in TOKEN_KEYS[3:]) == usage["prompt_tokens"], "Inconsistent cache usage")
            else:
                cache_missing += 1
    available = "requests" in state or not started
    cost = (known["prompt_tokens"] * rates["cache_miss"] + known["completion_tokens"] * rates["output"]) / 1_000_000
    return dict(requests=sum(statuses.values()) if available else None,
        provider_unknown=statuses["unknown"] if available else None, pending=statuses["pending"] if available else None,
        cache_unknown_requests=cache_missing if available else None, known_requests=statuses["known"] if available else None,
        known_usage={k:known[k] for k in TOKEN_KEYS} if available else None,
        cost_complete=available and not (statuses["unknown"] or statuses["pending"]),
        known_peak_upper_usd=float(cost) if available else None)


def oracle(raw, result, profile):
    expected = flag(result.get("fail_to_pass" if profile == "f2p" else "pass_to_pass"))
    check = raw.get("final_checks", {}).get(profile) or {}
    data = check.get("receipt")
    if not data or expected is None:
        require(expected is None, "Scored result lacks independent receipt")
        return None
    bits = [flag(r["passed"]) for r in data["checks"]]
    require(not check.get("receipt_error") and len(bits) == data["check_count"] and None not in bits
            and all(bits) == data["passed"] == expected, "Independent receipt disagrees with scored result")
    return dict(passed=expected, passed_checks=sum(bits), total_checks=len(bits))


def trial(root, slot, row, rates):
    case = slot["case_id"]
    require(case in CASES and slot["scope"] == row["scope"] and case == row["case_id"] and slot["repeat"] == row["repeat"],
            "Trial identity mismatch")
    scope = safe(root, slot["scope"])
    folder = safe(scope / "trials", slot["planned_id"])
    control = scope / "control/state.json"
    state = read(control) if control.exists() else {}
    ref_path = folder / "raw-reference.json"
    ref = read(ref_path) if ref_path.exists() else None
    raw = artifact(folder, ref) if ref else {}
    result = row.get("result") or {}
    native = folder / "result.json"
    if native.exists():
        saved = read(native)
        require(not result or saved == result, "Batch and native results differ")
        result = saved
    require(not result or bool(raw), "Missing raw trial")
    started = flag(row["started"])
    stats = transport(state, started, rates)
    events = 0
    trace_count = 0
    for record in raw.get("records", []):
        if record.get("trace"):
            trace = artifact(folder, record["trace"])
            events += sum(e["event_type"] == "coding.action_phase" for e in trace["events"])
            trace_count += 1
    guards = {}
    for entry in list(raw.get("permission_records", [])) + [v["record"] for v in state.get("permission_denials", {}).values()]:
        rec = entry["receipt"]
        reason = code(rec["reason_code"], GUARDS)
        require(guards.setdefault(rec["denial_id"], reason) == reason and rec["tool_executed"] is False,
                "Conflicting denial receipts")
    rounds = raw.get("rounds", [])
    # A corrective execution requires a separate dispatched writer, not just a routing decision.
    writer_requests = Counter(r["task_id"] for r in state.get("requests", {}).values())
    feedback = sum(writer_requests.get(r["result"]["task_id"], 0) for r in rounds[1:])
    final = row.get("execution_final") or {}
    group = slot.get("arm", "repeat")
    require(group in {"repeat", "new_development"}, "Unexpected trial group")
    retries = result.get("feedback_retries")
    if retries is not None:
        require(count(retries) == max(0, len(rounds) - 1), "Corrective rounds inconsistent")
    wall = result.get("wall_seconds")
    require(wall is None or type(wall) in (int, float) and 0 <= wall < 1e7, "Invalid wall time")
    return dict(case_id=case, repeat=count(slot["repeat"]), arm=group, started=started, terminal=flag(row["terminal"]),
        status=code(row["status"], STATES), archived=ref is not None, native_status=code(result.get("status"), STATES),
        e2e=flag(result.get("end_to_end_success")), f2p=oracle(raw, result, "f2p"), p2p=oracle(raw, result, "p2p"),
        failure_kind=code(result.get("failure_kind") or row.get("failure_kind"), FAILURES),
        scope_stop=code(final.get("stop_reason") or state.get("stop_reason"), FAILURES),
        tool_calls=count(result["tool_calls"]) if result.get("tool_calls") is not None else None,
        wall_seconds=wall, feedback_attempts=retries, corrective_writer_requests=feedback if state and raw else None,
        public_passes=[flag((r.get("public_receipt") or {}).get("passed")) for r in rounds],
        action_phase_events=events if trace_count else None,
        guard_reasons=dict(Counter(guards.values())), runtime_usage_unknown_attempts=sum(
            s["usage_unknown"] is True for s in raw["budget"]["settlements"].values()) if raw.get("budget") else None,
        raw_sha256=ref["sha256"] if ref else None, **stats)


def aggregate(rows):
    with_transport = [r for r in rows if r["known_usage"] is not None]
    return dict(planned=len(rows), started=sum(r["started"] for r in rows), terminal=sum(r["terminal"] for r in rows),
        archived=sum(r["archived"] for r in rows), statuses=dict(Counter(r["status"] for r in rows)),
        e2e_successes=sum(r["e2e"] is True for r in rows), e2e_unscored=sum(r["e2e"] is None for r in rows),
        **{k+"_passes":sum(bool(r[k] and r[k]["passed"]) for r in rows) for k in ("f2p", "p2p")},
        **{k+"_unscored":sum(r[k] is None for r in rows) for k in ("f2p", "p2p")},
        missing_transport_trials=len(rows)-len(with_transport), requests_recorded=sum(r["requests"] for r in with_transport),
        **{k:sum(r[k] for r in with_transport) for k in ("provider_unknown","pending","cache_unknown_requests")},
        known_usage={k:sum(r["known_usage"][k] for r in with_transport) for k in TOKEN_KEYS},
        known_peak_upper_usd=round(sum(r["known_peak_upper_usd"] for r in with_transport), 9),
        cost_complete=all(r["cost_complete"] for r in rows), known_tool_calls=sum(r["tool_calls"] or 0 for r in rows),
        missing_tool_count_started=sum(r["tool_calls"] is None and r["started"] for r in rows),
        feedback_attempts_observed=sum(r["feedback_attempts"] or 0 for r in rows),
        corrective_writer_requests_observed=sum(r["corrective_writer_requests"] or 0 for r in rows),
        action_phase_trials_observed=sum(bool(r["action_phase_events"]) for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    protocol = read(root / "unified-protocol.json")
    ledger = read(root / "batch-ledger.json")
    require(protocol["protocol_id"] == ledger["protocol_id"] and re.fullmatch(r"m\d{2}-[a-z0-9-]{1,48}", protocol["protocol_id"]),
            "Unrecognized protocol ID")
    rows = {r["scope"]:r for r in ledger["rows"]}
    schedule = protocol["schedule"]
    require(len({r["scope"] for r in schedule}) == len(schedule) == protocol["planned"] == len(rows) == len(ledger["rows"]),
            "Trial coverage mismatch")
    rates_raw = read(root / "preflight/price.json")["peak_usd_per_million"]
    rates = {k:Decimal(str(rates_raw[k])) for k in ("cache_hit", "cache_miss", "output")}
    require(all(v.is_finite() and 0 <= v < 1000 for v in rates.values()), "Invalid rate")
    trials = [trial(root, slot, rows[slot["scope"]], rates) for slot in schedule]
    summary = dict(schema_version=1, protocol_id=protocol["protocol_id"], total=aggregate(trials),
        groups={group:aggregate([r for r in trials if r["arm"]==group]) for group in sorted({r["arm"] for r in trials})},
        stop_reason=code(ledger.get("stop_reason"), FAILURES), usd_per_million_tokens={k:float(v) for k,v in rates.items()},
        cost_semantics="Conservative peak estimate for known usage: all input at cache-miss rate; excludes unknowns. Not an invoice.")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary", summary), ("trials", trials)):
        (args.output / (name + ".json")).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["total"]))


if __name__ == "__main__":
    main()
