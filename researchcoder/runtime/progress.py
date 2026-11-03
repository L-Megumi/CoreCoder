"""Bounded progress observations over actual tool replies, never model self-reports."""
import hashlib
import json
import copy

VERSION = "coding-progress-v1"
LIMIT = 3
READ_TOOLS = frozenset({"read_file", "read_project_file", "grep", "glob", "ls"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence_fingerprint(name, arguments, result):
    """Only successfully returned read data is progress; IDs and metadata are not."""
    if not isinstance(result, str) or result.startswith(("Error", "⚠ Blocked", "[interrupted]",
                                                        "[Unchanged read reused")):
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict) and ("error" in payload or payload.get("status") in ("error", "failed")):
        return None
    if name == "restore_context_evidence":
        if not isinstance(payload, dict) or payload.get("current_eligible") is not True:
            return None
        if payload.get("classification") in ("unverified_hypothesis", "historical_observation"):
            return None
        if not isinstance(payload.get("content"), str) or not payload["content"].strip():
            return None
        return digest(dict(kind="restored_window", original_sha256=payload.get("original_sha256"),
            source_start_line=payload.get("source_start_line"), content=payload["content"]))
    if name not in READ_TOOLS or not result.strip():
        return None
    return digest(dict(kind=name, arguments=arguments, result=result))


class ProgressTracker:
    """Per-attempt state. Recovery replies are included even outside RepairAgent.events."""
    def __init__(self, trace=None):
        self.trace = trace
        self.seen_calls = set()
        self.seen_fingerprints = set()
        self.records = []
        self.stagnant_rounds = 0
        self.mutations = 0
        self.verification_status = None

    def observe(self, agent):
        calls = {}
        for message in agent.messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls", ()):
                    try:
                        calls[call["id"]] = (call["function"]["name"], json.loads(call["function"]["arguments"]))
                    except (KeyError, TypeError, ValueError):
                        continue
        replies = [m for m in agent.messages if m.get("role") == "tool"
                   and m.get("tool_call_id") in calls and m["tool_call_id"] not in self.seen_calls]
        if not replies:
            return None
        fingerprints = []
        for reply in replies:
            identifier = reply["tool_call_id"]
            if identifier in self.seen_calls:
                continue
            self.seen_calls.add(identifier)
            name, arguments = calls[identifier]
            value = evidence_fingerprint(name, arguments, reply.get("content"))
            if value is not None and value not in fingerprints:
                fingerprints.append(value)
        fresh = [v for v in fingerprints if v not in self.seen_fingerprints]
        changed = agent.mutation_count > self.mutations
        self.stagnant_rounds = 0 if fresh or changed else self.stagnant_rounds + 1
        self.mutations = agent.mutation_count
        self.seen_fingerprints.update(fingerprints)
        row = dict(sequence=len(self.records) + 1, tool_count=len(self.seen_calls),
            mutation_count=self.mutations, fingerprints=fingerprints,
            new_fingerprints=fresh, stagnant_rounds=self.stagnant_rounds)
        self.records.append(row)
        if self.trace is not None:
            self.trace.emit("coding.progress.observed", count=self.stagnant_rounds,
                source_sequence=len(self.seen_calls), revision_id=digest(row),
                reason="new_evidence" if fresh else "source_changed" if changed else "no_new_evidence_or_edit")
        return row

    def verification(self, status):
        # Only RuntimeHooks.after_verification may call this with a real result.
        improved = status == "passed" and self.verification_status != "passed"
        self.verification_status = status
        if self.trace is not None:
            self.trace.emit("coding.progress.verification", status=status,
                reason="public_check_passed" if improved else "no_verified_improvement")
        return improved

    @property
    def stalled(self):
        return self.stagnant_rounds >= LIMIT

    def proof(self, agent):
        return dict(schema_version=1, source=VERSION, limit=LIMIT,
            rounds=self.stagnant_rounds, event_count=len(self.seen_calls),
            agent_event_count=len(agent.events), tool_rounds=len(self.records),
            effective_file_edits=self.mutations, observations=copy.deepcopy(self.records))


def validate_progress_proof(proof):
    """Recompute freshness/stagnation, rejecting invented progress counters."""
    required = {"schema_version", "source", "limit", "rounds", "event_count", "agent_event_count",
                "tool_rounds", "effective_file_edits", "observations"}
    if not isinstance(proof, dict) or set(proof) != required:
        raise ValueError("Malformed progress proof")
    if type(proof["schema_version"]) is not int or proof["schema_version"] != 1 or proof["source"] != VERSION or proof["limit"] != LIMIT:
        raise ValueError("Unsupported progress proof")
    numeric = ("rounds", "event_count", "agent_event_count", "tool_rounds", "effective_file_edits")
    if any(type(proof[k]) is not int or proof[k] < 0 for k in numeric):
        raise ValueError("Invalid progress count")
    rows = proof["observations"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise ValueError("Invalid progress window")
    seen, count, mutations, stalled = set(), 0, 0, 0
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"sequence", "tool_count", "mutation_count", "fingerprints", "new_fingerprints", "stagnant_rounds"}:
            raise ValueError("Malformed progress observation")
        if (type(row["sequence"]) is not int or row["sequence"] != index
                or type(row["tool_count"]) is not int or row["tool_count"] <= count
                or type(row["mutation_count"]) is not int or row["mutation_count"] < mutations):
            raise ValueError("Nonmonotonic progress observation")
        values = row["fingerprints"]
        if (not isinstance(values, list) or len(values) > row["tool_count"] - count
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Invalid evidence fingerprints")
        fresh = [v for v in values if v not in seen]
        stalled = 0 if fresh or row["mutation_count"] > mutations else stalled + 1
        if row["new_fingerprints"] != fresh or type(row["stagnant_rounds"]) is not int or row["stagnant_rounds"] != stalled:
            raise ValueError("Progress proof does not reproduce")
        seen.update(values)
        count, mutations = row["tool_count"], row["mutation_count"]
        if stalled >= LIMIT and index != len(rows):
            raise ValueError("Progress history continued beyond its stop threshold")
    if (len(rows) != proof["tool_rounds"] or count != proof["event_count"]
            or mutations != proof["effective_file_edits"] or stalled != proof["rounds"]
            or stalled != LIMIT or proof["agent_event_count"] > count):
        raise ValueError("Progress stop threshold not proven")
    return True
