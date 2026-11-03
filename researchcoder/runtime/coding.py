"""Versioned, controller-only opt-in for archiving a local Coding stop.

This authorizes neither a retry nor additional resources. The project controller
must separately validate the receipts and reserve the next task in its JobBudget.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal

from researchcoder.contracts import Contract


@dataclass(frozen=True, kw_only=True)
class CodingContinuationPolicy(Contract):
    mode: Literal["local-stop-v1", "local-stop-v2", "local-stop-v3"] = "local-stop-v1"
    closure_version: Literal["off", "reserve-incomplete-v1"] = "off"
    request_version: Literal["off", "explicit-request-v1"] = "off"
    schema_version: int = 1

    def to_dict(self):
        value = super().to_dict()
        if self.closure_version == "off":
            value.pop("closure_version")
        if self.request_version == "off":
            value.pop("request_version")
        return value

    def validate(self):
        if self.closure_version != "off" and self.mode != "local-stop-v3":
            raise ValueError("Reserve closure requires local-stop-v3")
        if self.request_version != "off" and self.mode != "local-stop-v3":
            raise ValueError("Expert request requires local-stop-v3")


CODING_EXPERT_REQUEST = (
    "\n[Controller cooperation option: explicit-request-v1]\n"
    "Before making any edit, you may request one bounded independent Code Analysis expert investigation "
    "when there is a specific unresolved source question, a conflict between observations, or uncertainty "
    "about a proposed repair's public component constraints. First inspect relevant actual source. "
    "Do not request help for work already resolved, invent evidence, or assume an expert grants extra permissions. "
    "To request, end this attempt with exactly one JSON object and no Markdown: "
    '{"action":"request_expert","requested_role":"code_analysis",'
    '"question":"the concrete unresolved question","evidence_paths":["relative/file.py"]}. '
    "Use 1 to 4 normalized candidate-relative paths you successfully read with read_file in this attempt. "
    "If the relevant evidence was only restored from context, read the current file before requesting. "
    "Your question is a hypothesis, not verified evidence. No tools or edits may follow the request. "
    "The controller checks the original Job's remaining budget, source provenance, public failure and one-consultation limit. "
    "It may reject the request; approval is not task success. If no independent investigation is needed, "
    "continue the normal repair and final summary. This option is unavailable once any write has been attempted "
    "or when the controller has entered its finalization-only phase."
)
CODING_EXPERT_REQUEST_SHA256 = hashlib.sha256(CODING_EXPERT_REQUEST.encode()).hexdigest()


def _request_path(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 500 or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("Invalid request source path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or path.as_posix() != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("Request path must be normalized and candidate relative")


def validate_expert_request(response, finish_reason=None):
    """Parse an optional control response, preserving malformed requests as failure.

    Ordinary final summaries keep their historical behavior. The parsed request
    itself is untrusted; actual read provenance and continuation are checked later.
    """
    if not isinstance(response, str):
        return None
    stripped = response.lstrip()
    # Recognize the control key even if the provider truncated the action value;
    # otherwise a partial request could be mistaken for an ordinary completion.
    candidate = (stripped.startswith(("{", "```")) and '"action"' in stripped)
    if not candidate:
        return None
    report = dict(schema_version=1, version="explicit-request-v1",
        prompt_sha256=CODING_EXPERT_REQUEST_SHA256, original_response=response,
        request=None, failure_kind=None)
    if finish_reason == "length":
        report["failure_kind"] = "response_length_exceeded"
        return report
    if len(response) > 10000:
        report["failure_kind"] = "invalid_report"
        return report

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate request key")
            result[key] = value
        return result

    try:
        value = json.loads(response, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite request")))
    except (ValueError, TypeError):
        report["failure_kind"] = "invalid_json"
        return report
    try:
        if (type(value) is not dict or set(value) != {"action", "requested_role", "question", "evidence_paths"}
                or value["action"] != "request_expert" or value["requested_role"] != "code_analysis"
                or type(value["question"]) is not str or not 1 <= len(value["question"].strip()) <= 2000
                or len(value["question"]) > 2000 or "\x00" in value["question"]
                or type(value["evidence_paths"]) is not list or not 1 <= len(value["evidence_paths"]) <= 4):
            raise ValueError("Invalid expert request")
        for path in value["evidence_paths"]:
            _request_path(path)
        if len(set(value["evidence_paths"])) != len(value["evidence_paths"]):
            raise ValueError("Duplicate request source")
    except (TypeError, ValueError):
        report["failure_kind"] = "invalid_report"
        return report
    report["request"] = value
    return report


def validate_request_proof(proof, report, budget):
    """Validate a request after the real verifier, without granting dispatch."""
    fields = {"schema_version", "source", "trigger", "model_calls", "rounds", "event_count", "tool_count",
              "effective_file_edits", "request_sha256", "source_bindings"}
    if (type(proof) is not dict or set(proof) != fields or type(proof["schema_version"]) is not int
            or proof["schema_version"] != 1 or proof["source"] != "controller-explicit-request-v1"
            or proof["trigger"] != "expert_request" or type(report) is not dict or type(budget) is not dict):
        raise ValueError("Invalid explicit request proof")
    expected = validate_expert_request(report.get("original_response"))
    if (expected is None or expected != report or report["failure_kind"] is not None
            or investigation_proof_sha256(report) != proof["request_sha256"]):
        raise ValueError("Request report mismatch")
    if (any(type(proof[key]) is not int or proof[key] < 0
            for key in ("model_calls", "rounds", "event_count", "tool_count", "effective_file_edits"))
            or proof["effective_file_edits"] != 0 or not 1 <= proof["event_count"] <= proof["tool_count"]
            or proof["model_calls"] < 1 or proof["rounds"] < 1
            or any(budget.get(key) != proof[key] for key in ("model_calls", "rounds"))
            or budget.get("tool_calls") != proof["tool_count"] + 1
            or budget.get("pending") != {} or budget.get("unknown_requests") != 0
            or any(budget.get(key) is not None for key in ("stop_kind", "local_limit_detail", "last_failure_kind"))):
        raise ValueError("Request counters do not match settled execution")
    bindings = proof["source_bindings"]
    if (type(bindings) is not list or len(bindings) != len(report["request"]["evidence_paths"])
            or [row.get("path") for row in bindings if type(row) is dict] != report["request"]["evidence_paths"]):
        raise ValueError("Request source identity mismatch")
    for row in bindings:
        if (type(row) is not dict or set(row) != {"path", "file_sha256", "event_sequences"}
                or type(row["file_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", row["file_sha256"])
                or type(row["event_sequences"]) is not list or not row["event_sequences"]
                or any(type(index) is not int or not 1 <= index <= proof["event_count"] for index in row["event_sequences"])
                or row["event_sequences"] != sorted(set(row["event_sequences"]))):
            raise ValueError("Invalid observed request source")
        _request_path(row["path"])
    return True


def request_stop_proof(agent, report, hooks, policy):
    """Build pre-verifier proof from successful, current, read-only tool events."""
    from corecoder.workflow import RepairAgent, file_digest
    from researchcoder.context.engine import RESTORE
    if (type(policy) is not CodingContinuationPolicy or policy.request_version != "explicit-request-v1"
            or not isinstance(agent, RepairAgent) or agent._output_only or agent.budget_exhausted
            or agent.mutation_count != 0 or agent.changes() or hooks.permission_stop
            or type(report) is not dict or report.get("failure_kind") is not None
            or validate_expert_request(report.get("original_response")) != report):
        return None
    ledger = hooks.ledger
    if (ledger.pending or ledger.unknown_requests or ledger.stop_kind is not None
            or ledger.local_limit_detail is not None or ledger.last_failure_kind is not None
            or len(ledger.settlements) != ledger.model_calls
            or any(row.get("unknown") is not False or row.get("status") != "ok" for row in ledger.settlements)):
        return None
    events = agent.events
    if not events or any(type(event) is not dict or event.get("sequence") != index
            or event.get("tool") not in ("read_file", RESTORE) or event.get("status") != "ok"
            or event.get("permission_denial") or event.get("permission_audit_failed")
            for index, event in enumerate(events, 1)):
        return None
    starts = [e for e in hooks.trace.events if e.get("event_type") == "tool.call.start"]
    ends = [e for e in hooks.trace.events if e.get("event_type") == "tool.call.end"]
    if (len(starts) != ledger.tool_calls or len({e["event_id"] for e in starts}) != len(starts)
            or len(ends) != len(starts) or len({e.get("parent_id") for e in ends}) != len(ends)
            or {e["event_id"] for e in starts} != {e.get("parent_id") for e in ends}
            or any(e.get("tool") not in ("read_file", RESTORE) for e in starts)
            or any(e.get("status") != "ok" for e in ends)):
        return None
    root = agent.root.resolve()
    bindings = []
    try:
        for relative in report["request"]["evidence_paths"]:
            target = (root / relative).resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(root):
                return None
            digest = file_digest(target)
            if not any(key[0] == str(target) and key[-1] == digest for key in agent.read_cache):
                return None
            sequences = [event["sequence"] for event in events if event["tool"] == "read_file"
                and (root / event.get("arguments", {}).get("file_path", "")).resolve() == target]
            if not sequences:
                return None
            bindings.append(dict(path=relative, file_sha256=digest, event_sequences=sequences))
    except (ValueError, TypeError, OSError, IndexError):
        return None
    return dict(schema_version=1, source="controller-explicit-request-v1", trigger="expert_request",
        model_calls=ledger.model_calls, rounds=ledger.rounds, event_count=len(events), tool_count=ledger.tool_calls,
        effective_file_edits=0, request_sha256=investigation_proof_sha256(report), source_bindings=bindings)


LOCAL_STOP_DETAILS = frozenset({"model_calls", "rounds", "context_tokens", "total_tokens"})

CODING_FINAL_REPORT = (
    "[Controller coding finalization: explicit-report-v1]\n"
    "The investigation/editing phase is closed for this attempt. Do not call tools or emit tool-call markup. "
    "Using only work already performed, return exactly one JSON object with two keys: "
    '\"status\" (\"completed\" or \"incomplete\") and \"summary\" (a concise nonempty string). '
    "Use incomplete when required changes remain or evidence is insufficient. Do not invent edits or verification. "
    "The controller will run the authorized public verification command; your declaration is not a test result. "
    "No Markdown fences, additional keys, source dump or further actions."
)
CODING_FINAL_REPORT_SHA256 = hashlib.sha256(CODING_FINAL_REPORT.encode()).hexdigest()


def validate_coding_final_report(response, finish_reason=None):
    """Validate a new explicit closing response; never complete malformed JSON."""
    receipt = dict(schema_version=1, version="explicit-report-v1",
        prompt_sha256=CODING_FINAL_REPORT_SHA256, original_response=response,
        declared_status=None, summary="", failure_kind=None)
    if finish_reason == "length":
        receipt["failure_kind"] = "response_length_exceeded"
        return receipt

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate closing key")
            value[key] = item
        return value

    try:
        value = json.loads(response, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (TypeError, ValueError):
        receipt["failure_kind"] = "invalid_json"
        return receipt
    if (not isinstance(value, dict) or set(value) != {"status", "summary"}
            or value["status"] not in ("completed", "incomplete")
            or not isinstance(value["summary"], str) or not 1 <= len(value["summary"].strip()) <= 8000):
        receipt["failure_kind"] = "invalid_report"
        return receipt
    receipt.update(declared_status=value["status"], summary=value["summary"])
    if value["status"] == "incomplete":
        receipt["failure_kind"] = "unknown"
        receipt["reason"] = "model_declared_incomplete"
    return receipt


def investigation_proof_sha256(proof):
    """Bind the complete trusted counter/window proof into its runtime trace."""
    return hashlib.sha256(json.dumps(proof, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_closure_proof(proof, report, budget):
    """Check the narrow reserve declaration; this is not handoff admission.

    The Controller additionally validates immutable artifacts, original source,
    public verification and execution spans before allocating an expert.
    """
    fields = {"schema_version", "source", "trigger", "model_calls", "rounds",
              "max_model_calls", "max_rounds", "event_count", "tool_count",
              "effective_file_edits", "report_sha256"}
    if (not isinstance(proof, dict) or set(proof) != fields
            or type(proof["schema_version"]) is not int or proof["schema_version"] != 1
            or proof["source"] != "controller-reserve-incomplete-v1"
            or proof["trigger"] not in ("round_reserve", "model_call_reserve")
            or not isinstance(report, dict) or not isinstance(budget, dict)):
        raise ValueError("Invalid reserve closure proof")
    numeric = ("model_calls", "rounds", "max_model_calls", "max_rounds", "event_count", "tool_count", "effective_file_edits")
    if any(type(proof[key]) is not int or proof[key] < 0 for key in numeric):
        raise ValueError("Invalid reserve closure counters")
    expected_report = validate_coding_final_report(report.get("original_response"))
    expected_report["trigger"] = proof["trigger"]
    if (report != expected_report or report.get("declared_status") != "incomplete"
            or report.get("failure_kind") != "unknown"
            or report.get("reason") != "model_declared_incomplete"
            or investigation_proof_sha256(report) != proof["report_sha256"]):
        raise ValueError("Invalid explicit incomplete report")
    if (proof["effective_file_edits"] != 0 or not 1 <= proof["event_count"] <= proof["tool_count"]
            or not 1 <= proof["model_calls"] <= proof["max_model_calls"]
            or not 1 <= proof["rounds"] <= proof["max_rounds"]
            or (proof["trigger"] == "round_reserve" and proof["rounds"] != proof["max_rounds"] - 1)
            or (proof["trigger"] == "model_call_reserve" and proof["model_calls"] != proof["max_model_calls"])
            or any(budget.get(key) != proof[key] for key in ("model_calls", "rounds"))
            or budget.get("tool_calls") != proof["tool_count"] + 1
            or budget.get("stop_kind") is not None or budget.get("local_limit_detail") is not None
            or budget.get("last_failure_kind") is not None
            or budget.get("pending") != {} or budget.get("unknown_requests") != 0):
        raise ValueError("Reserve closure does not match settled execution")
    return True


def closure_stop_proof(agent, report, hooks, policy):
    """Archive a completed, read-only investigation without relabeling failure.

    Called after the protected verifier. This receipt preserves the model's
    incomplete response and leaves both local and global budgets untouched.
    """
    from corecoder.workflow import RepairAgent
    from researchcoder.context.engine import RESTORE
    if (type(policy) is not CodingContinuationPolicy or policy.closure_version != "reserve-incomplete-v1"
            or not isinstance(agent, RepairAgent) or not agent._explicit_completion or not agent._output_only
            or getattr(agent, "_completion_reason", None) not in ("round_reserve", "model_call_reserve")
            or hooks.permission_stop or agent.budget_exhausted or agent.mutation_count != 0
            or agent.changes() or not isinstance(report, dict)):
        return None
    events = agent.events
    if not events or any(not isinstance(event, dict) or event.get("sequence") != index
            or event.get("tool") not in ("read_file", RESTORE) or event.get("status") != "ok"
            or event.get("permission_denial") or event.get("permission_audit_failed")
            for index, event in enumerate(events, 1)):
        return None
    ledger = hooks.ledger
    settlements = ledger.settlements
    if (len(settlements) != ledger.model_calls or any(row.get("unknown") is not False
            or row.get("status") != "ok" for row in settlements)):
        return None
    # Context recovery is intercepted by the context engine, so its accounted
    # tool spans are not necessarily present in RepairAgent.events. Count both
    # surfaces explicitly; never turn the missing local event into a free call.
    starts = [e for e in hooks.trace.events if e.get("event_type") == "tool.call.start"]
    ends = [e for e in hooks.trace.events if e.get("event_type") == "tool.call.end"]
    by_id = {e["event_id"]: e for e in starts}
    end_ids = [e.get("parent_id") for e in ends]
    if (len(starts) != ledger.tool_calls - 1 or len(by_id) != len(starts)
            or len(end_ids) != len(set(end_ids)) or set(by_id) != set(end_ids)
            or any(e.get("tool") not in ("read_file", RESTORE) for e in starts)
            or any(e.get("status") != "ok" for e in ends)
            or sum(e.get("tool") == "read_file" for e in starts)
                != sum(e.get("tool") == "read_file" for e in events)
            or sum(e.get("tool") == RESTORE for e in starts)
                < sum(e.get("tool") == RESTORE for e in events)):
        return None
    proof = dict(schema_version=1, source="controller-reserve-incomplete-v1",
        trigger=agent._completion_reason, model_calls=ledger.model_calls, rounds=ledger.rounds,
        max_model_calls=ledger.limits.max_model_calls, max_rounds=ledger.limits.max_rounds,
        event_count=len(events), tool_count=len(starts), effective_file_edits=0,
        report_sha256=investigation_proof_sha256(report))
    try:
        validate_closure_proof(proof, report, ledger.snapshot())
    except (TypeError, ValueError):
        return None
    phases = [e for e in hooks.trace.events if e.get("event_type") == "coding.report_phase"]
    received = [e for e in hooks.trace.events if e.get("event_type") == "coding.report_received"]
    if (len(phases) != 1 or len(received) != 1
            or phases[0].get("reason") != proof["trigger"]
            or phases[0].get("revision_id") != CODING_FINAL_REPORT_SHA256
            or phases[0].get("count") != proof["max_model_calls"] - proof["model_calls"] + 1
            or received[0].get("revision_id") != CODING_FINAL_REPORT_SHA256
            or received[0].get("status") != "incomplete" or received[0].get("failure_kind") != "unknown"):
        return None
    return proof


def known_local_stop(hooks):
    """Require settled accounting and no prior permission/side-effect failure."""
    ledger = hooks.ledger
    return (
        ledger.stop_kind == "budget_exhausted"
        and (ledger.local_limit_detail in LOCAL_STOP_DETAILS
             or (ledger.local_limit_detail == "verification_reserve"
                 and getattr(getattr(hooks, "optimization", None), "version", None) == "explicit-report-v1"))
        and not ledger.pending and ledger.unknown_requests == 0
        and not hooks.permission_stop
        and ledger.last_failure_kind not in ("tool_error", "model_error", "timeout", "interrupted")
    )


def investigation_stop_proof(agent, error, hooks, policy):
    """Validate a typed, synchronous Agent stop without clearing any liability.

    A matching error message, a forged exception, or exhausted global resources
    cannot become a continuation. The Controller separately validates this
    receipt against immutable effects, the final public check and Job budget.
    """
    from corecoder.workflow import RepairAgent, RepairInvestigationLimit
    if (type(policy) is not CodingContinuationPolicy or policy.mode not in ("local-stop-v2", "local-stop-v3")
            or type(error) is not RepairInvestigationLimit or not isinstance(agent, RepairAgent)
            or agent._typed_investigation_stops is not True
            or error is not agent._investigation_limit_error
            or agent.stop_reason != "investigation_limit"
            or agent.investigation_rounds != agent.max_investigation_rounds
            or agent._seen_events != len(agent.events)
            or agent._seen_mutations != agent.mutation_count):
        return None
    ledger = hooks.ledger
    if (ledger.stop_kind is not None or ledger.pending or ledger.unknown_requests
            or ledger.last_failure_kind is not None or hooks.permission_stop):
        return None
    ends = agent._investigation_window_ends
    start = agent._investigation_window_start
    if (len(ends) != agent.investigation_rounds or not ends
            or any(type(v) is not int for v in (start, *ends))
            or not 0 <= start < ends[0] or ends[-1] != len(agent.events)
            or any(a >= b for a, b in zip(ends, ends[1:]))):
        return None
    return dict(schema_version=1, source="repair-agent-investigation-limit-v1",
        limit=agent.max_investigation_rounds, rounds=agent.investigation_rounds,
        event_count=len(agent.events), effective_file_edits=agent.mutation_count,
        window_start_event=start, window_event_ends=list(ends))


def progress_stop_proof(agent, error, hooks, policy):
    """Only the originating tracker in a settled Worker can authorize review."""
    from corecoder.workflow import RepairAgent, RepairProgressStalled
    from .progress import ProgressTracker, validate_progress_proof
    tracker = getattr(hooks, "progress_tracker", None)
    ledger = hooks.ledger
    if (type(policy) is not CodingContinuationPolicy or policy.mode != "local-stop-v3"
            or not isinstance(agent, RepairAgent) or type(error) is not RepairProgressStalled
            or error is not getattr(agent, "_progress_stop_error", None)
            or agent.stop_reason != "progress_stalled" or type(tracker) is not ProgressTracker
            or not tracker.stalled or ledger.stop_kind is not None
            or ledger.pending or ledger.unknown_requests or ledger.last_failure_kind is not None
            or hooks.permission_stop):
        return None
    proof = tracker.proof(agent)
    try:
        validate_progress_proof(proof)
    except ValueError:
        return None
    if proof["event_count"] != ledger.tool_calls:
        return None
    return proof
