"""Read only: independently inspect transport, immutable history and both Worker outcomes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from stress_condition import AFTER, MARKER


def get_json(path):
    return json.loads(path.read_bytes())


def get_artifact(trial, ref, job):
    root = (trial / "artifacts").resolve()
    storage = Path(ref["storage_key"])
    if (ref["job_id"] != job or storage.parts != (job, ref["artifact_id"] + ".blob")
            or any(part in (".", "..") for part in storage.parts)):
        raise ValueError("Foreign or invalid artifact")
    original = root / storage
    path = original.resolve(strict=True)
    if path != original or not path.is_relative_to(root):
        raise ValueError("Artifact escaped trial")
    data = path.read_bytes()
    if len(data) != ref["size_bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Archive digest mismatch")
    return json.loads(data)


def mutation_receipt(before, after, writable_paths, workspace, progress, coding_audit):
    """A cumulative baseline diff cannot establish that the corrective worker edited the project."""
    old = {e["path"]: e["sha256"] for e in before["entries"]}
    new = {e["path"]: e["sha256"] for e in after["entries"]}
    if (len(old) != len(before["entries"]) or len(new) != len(after["entries"])
            or old.keys() != new.keys()):
        raise ValueError("Attempt manifests changed shape")
    changed = sorted(path for path in old if old[path] != new[path])
    allowed = set(writable_paths)
    if not set(changed) <= allowed:
        raise ValueError("Unpermitted changes in the corrective attempt")
    workspace = PurePosixPath(workspace)
    if not workspace.is_absolute():
        raise ValueError("Invalid archived workspace")
    writes = []
    for event in coding_audit["events"]:
        if event["tool"] not in ("write_file", "edit_file") or event["status"] != "ok":
            continue
        path = workspace / PurePosixPath(event["arguments"]["file_path"])
        if (not path.is_relative_to(workspace) or ".." in path.parts or
                path.relative_to(workspace).as_posix() not in allowed or event.get("permission_denial")):
            raise ValueError("Invalid corrective write event")
        writes.append(path.relative_to(workspace).as_posix())
    count = progress["effective_file_edits"]
    if (type(count) is not int or count < 0 or count > len(writes)
            or progress != coding_audit["coding_progress"]):
        raise ValueError("Corrective mutation count disagrees with archived tool record")
    return dict(corrective_changed_paths=changed, corrective_effective_edits=count,
                corrective_ok_write_events=len(writes),
                verified_corrective_mutation=(before != after and bool(changed) and count > 0
                                              and set(changed) <= set(writes)))


def receipt(scope, *, controlled=False):
    trials = list((scope / "trials").iterdir())
    if len(trials) != 1:
        raise ValueError("Expected a single registered trial")
    trial = trials[0]
    ref = get_json(trial / "raw-reference.json")
    job = ref["job_id"]
    read = lambda r: get_artifact(trial, r, job)
    raw = read(ref)
    if raw["job_id"] != job:
        raise ValueError("Record on wrong job")
    result = get_json(trial / "result.json")
    state = get_json(scope / "control/state.json")
    transport = list(state["requests"].values())
    rounds = raw["rounds"]
    label = "induced_first_attempt" if controlled else "unmodified_first_attempt"
    out = dict(condition=label, job_id=job, rounds=len(rounds), provider_calls=len(transport),
               raw_sha256=ref["sha256"], terminal_status=result["status"],
               end_to_end=result["end_to_end_success"], corrective_attempt_started=len(rounds) > 1,
               triggered=False, verified_repair=False)
    if len(rounds) < 2:
        return out
    if len(rounds) != 2:
        raise ValueError("Unexpected number of attempts")
    first, second = rounds
    r0, r1 = first["result"], second["result"]
    aid0, aid1 = r0["attempt_id"], r1["attempt_id"]
    spec = get_json(trial / "coding-1.input.json")
    if (spec["task_id"] != r1["task_id"] or spec["job_id"] != job
            or sorted(spec["policy"]["writable_paths"]) != sorted(raw["task"]["writable_paths"])):
        raise ValueError("Correction input identity or write scope differs")
    audits = [a for ref in r1["outputs"] if "coding_progress" in (a := read(ref))]
    if len(audits) != 1 or second.get("progress") is None:
        raise ValueError("Correction progress has no unique original tool audit")
    mutation = mutation_receipt(second["before"], second["after"], raw["task"]["writable_paths"],
                                spec["workspace"]["root"], second["progress"], audits[0])
    requests = []
    for r in (r0, r1):
        packets = []
        for output in r["outputs"]:
            for pkt_ref in read(output).get("request_records", []):
                pkt = read(pkt_ref)
                if pkt["job_id"] != job or pkt["attempt_id"] != r["attempt_id"] or pkt["task_id"] != r["task_id"]:
                    raise ValueError("Request provenance mismatch")
                packets.append(pkt["request"])
        requests.append(packets)
    calls = [r for r in transport if r["attempt_id"] == aid1 and r["task_id"] == "coding-1"]
    handoff = next(h["reference"] for h in raw["writer_handoffs"] if h["task_id"] == "coding-1")
    history = read(read(handoff)["public_failed_attempts"])
    source = next(h["source_digest"] for h in history if h["attempt_id"] == aid0 and h["status"] == "failed")
    user_view = [m["content"] for p in requests[1] for m in p["messages"] if m["role"] == "user"]
    first_system = [m["content"] for p in requests[0] for m in p["messages"] if m["role"] == "system"]
    second_system = [m["content"] for p in requests[1] for m in p["messages"] if m["role"] == "system"]
    exception = first.get("public_exception_receipt")
    if exception is not None and read(first["public_exception_ref"]) != exception:
        raise ValueError("Public exception receipt differs from archived copy")
    first_receipt = first["public_receipt"] or exception
    local = False
    for entry in raw.get("handoffs", []):
        proof = entry["receipt"]
        if proof["attempt_id"] == aid0 and proof["eligible"] and read(entry["reference"]) == proof:
            local = (proof["job_id"] == job and proof["task_id"] == r0["task_id"]
                     and read(proof["result_ref"]) == r0 and read(proof["record_ref"]) == first["record"]
                     and proof["after_sha256"] == source and not r0["workflow_completed"])
            break
    source_bound = hashlib.sha256(json.dumps(first["after"], ensure_ascii=False, allow_nan=False,
                                             sort_keys=True).encode()).hexdigest() == source
    same_scope = ((MARKER in "\n".join(first_system)) == controlled
                  and all(MARKER not in m for m in second_system))
    budget = raw["budget"]
    checks = dict(
        honest_first_public_failure=bool((local or (r0["workflow_completed"] and r0["failure_kind"] == "verification_failed"))
            and first_receipt and first_receipt["passed"] is False and first_receipt["exit_code"] != 0
            and r0["verification"]["status"] == "failed"
            and first_receipt["checked_workspace_revision"] == r0["verification"]["checked_workspace_revision"]),
        precondition_is_archived_change=(not controlled or bool(first_receipt and first_receipt.get("source_sha256") == AFTER)),
        same_job_new_worker_attempt=job == r0["job_id"] == r1["job_id"] and aid0 != aid1
            and r0["task_id"] == "coding-0" and r1["task_id"] == "coding-1"
            and all(type(row["record"]["worker_pid"]) is int and row["record"]["worker_pid"] > 0 for row in rounds)
            and first["record"]["worker_pid"] != second["record"]["worker_pid"],
        same_workspace_state=source_bound and first["after"] == second["before"],
        feedback_in_real_request=any(aid0 in m and source in m and "public_failed_attempts" in m for m in user_view),
        diagnostic_scope_verified=same_scope,
        live_corrective_worker=bool(calls) and len(calls) == len(requests[1]),
        transport_complete=not state["stop_reason"] and len(transport) == result["model_requests"] and
            all(t["job_id"] == job and t["status"] == "known" and t["usage_complete"] for t in transport),
        bounded_same_budget=not budget["pending"] and {"coding-0", "coding-1"} <= set(budget["settlements"]) and
            all(not entry["usage_unknown"] for entry in budget["settlements"].values()) and
            all(0 <= n <= budget["limits"]["max_" + k] for k, n in budget["charged"].items()),
        route_authorized=any(d["decision"]["reason_codes"] == ["bounded_public_correction"] and
                             d["attempt_id"] == aid0 for d in raw["routing_decisions"]),
        expected_source_and_permissions=raw["policy_intact"] is True and raw["preflight"]["valid"] is True,
    )
    checks_pass = all(checks.values())
    final_checks = raw["final_checks"]
    independent = all((final_checks.get(k, {}).get("receipt") or {}).get("passed") is True
                      and not final_checks[k].get("receipt_error")
                      and final_checks[k]["result"]["verification"]["checked_workspace_revision"] == raw["legacy_revision_after"]
                      for k in ("f2p", "p2p"))
    out.update(checks=checks, triggered=checks_pass, first_attempt=aid0, corrective_attempt=aid1, **mutation,
               corrective_requests=len(calls), feedback_source_sha256=source, public_history_sha256=read(handoff)["public_failed_attempts"]["sha256"],
               first_termination="eligible_local_stop" if local else r0["failure_kind"],
               verified_repair=bool(checks_pass and independent and mutation["verified_corrective_mutation"]
                                     and result["end_to_end_success"] and
                                     raw["source_after"] == second["after"] and not raw["stop_reason"] and
                                     r1["status"] == "succeeded" and r1["workflow_completed"] and
                                     second["public_receipt"] and second["public_receipt"]["passed"] is True))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", type=Path)
    parser.add_argument("--controlled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(receipt(args.scope, controlled=args.controlled), indent=2))
