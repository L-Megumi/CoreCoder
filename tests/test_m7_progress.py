"""Progress means new returned evidence or actual mutation, not another tool call."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from corecoder.llm import ToolCall
from researchcoder.runtime.budget import BudgetExceeded
from researchcoder.runtime.coding import CodingContinuationPolicy
from researchcoder.runtime.progress import ProgressTracker, evidence_fingerprint, validate_progress_proof
from researchcoder.runtime.protocol import RuntimeLimits
from tests.test_m6_completion import setup, execute, response


def append(agent, number, *, name="restore_context_evidence", content="def step(): pass", start=1,
           historical=False, arguments=None):
    arguments = arguments or dict(reference_id="source", start_line=start, max_lines=9)
    result = (json.dumps(dict(content=content, original_sha256="a" * 64,
        source_start_line=start, current_eligible=not historical,
        classification="historical_observation" if historical else "observed_record",
        reference=dict(document_id="fresh-metadata-" + str(number))))
        if name == "restore_context_evidence" else content)
    agent.messages.extend([dict(role="assistant", tool_calls=[dict(id=str(number),
        function=dict(name=name, arguments=json.dumps(arguments)))]),
        dict(role="tool", tool_call_id=str(number), content=result)])


def empty_agent():
    return SimpleNamespace(messages=[], mutation_count=0, events=[])


def stopped():
    tracker, agent = ProgressTracker(), empty_agent()
    for i in range(4):
        append(agent, i)
        tracker.observe(agent)
    return tracker, agent


def test_recovery_metadata_and_call_ids_do_not_count_as_new_evidence():
    tracker, agent = stopped()
    assert tracker.stalled and tracker.stagnant_rounds == 3
    assert len(tracker.seen_fingerprints) == 1
    proof = tracker.proof(agent)
    assert proof["event_count"] == 4 and proof["agent_event_count"] == 0
    assert validate_progress_proof(proof) is True
    assert tracker.observe(agent) is None  # no double counting on another view build


def test_new_source_window_and_effective_edit_reset_stagnation():
    tracker, agent = ProgressTracker(), empty_agent()
    for i in range(3):
        append(agent, i)
        tracker.observe(agent)
    assert tracker.stagnant_rounds == 2
    append(agent, 3, start=10, content="def restore(): pass")
    tracker.observe(agent)
    assert tracker.stagnant_rounds == 0
    append(agent, 4, start=10, content="def restore(): pass")
    tracker.observe(agent)
    assert tracker.stagnant_rounds == 1
    agent.mutation_count += 1
    append(agent, 5, start=10, content="def restore(): pass")
    tracker.observe(agent)
    assert tracker.stagnant_rounds == 0


def test_stale_hypothesis_and_failed_recovery_do_not_reset_progress():
    tracker, agent = ProgressTracker(), empty_agent()
    for i in range(3):
        append(agent, i, historical=True, content=str(i))
        tracker.observe(agent)
    assert tracker.stalled and not tracker.seen_fingerprints
    assert evidence_fingerprint("restore_context_evidence", {}, "Error: rejected") is None
    assert evidence_fingerprint("bash", {"command": "echo progress"}, "new progress") is None


@pytest.mark.parametrize("field,value", [("rounds", 0), ("event_count", 999), ("limit", 9),
                                       ("agent_event_count", 100), ("effective_file_edits", 1)])
def test_forged_progress_stop_counts_are_rejected(field, value):
    tracker, agent = stopped()
    proof = tracker.proof(agent)
    proof[field] = value
    with pytest.raises(ValueError):
        validate_progress_proof(proof)


def test_forged_freshness_is_rejected_and_proof_cannot_mutate_tracker():
    tracker, agent = stopped()
    proof = tracker.proof(agent)
    proof["observations"][-1]["new_fingerprints"] = ["f" * 64]
    with pytest.raises(ValueError):
        validate_progress_proof(proof)
    assert tracker.records[-1]["new_fingerprints"] == []


def test_real_coding_loop_stops_repeated_read_and_retains_public_check(tmp_path):
    limits = RuntimeLimits(max_model_calls=12, max_rounds=10, max_tool_calls=24,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=512)
    request, io, hooks = setup(tmp_path, explicit=False, source="value = 1\n", limits=limits)
    request = replace(request, coding_continuation=CodingContinuationPolicy(mode="local-stop-v3"))
    hooks.ledger.coding_continuation = request.coding_continuation
    turns = [response(calls=[ToolCall(str(i), "read_file", {"file_path": "module.py"})]) for i in range(4)]
    result, raw, model = execute(request, io, hooks, turns)
    assert result.status == "incomplete" and not result.workflow_completed
    assert result.failure_kind == "budget_exhausted" and result.verification.status == "passed"
    assert hooks.ledger.model_calls == len(model.inputs) == 4
    assert hooks.ledger.tool_calls == 5  # four real reads plus original public verifier
    assert hooks.ledger.local_limit_detail == "progress_stalled"
    proof = raw["coding_stop_receipt"]["progress_stop"]
    assert validate_progress_proof(proof)
    assert proof["event_count"] == proof["agent_event_count"] == 4
    assert raw["coding_progress"] == proof
    assert len([e for e in hooks.trace.events if e["event_type"] == "coding.progress_stalled"]) == 1


@pytest.mark.parametrize("unsafe", ["unknown", "pending", "permission", "effect_error"])
def test_progress_cannot_reclassify_unknown_or_permission_failure(tmp_path, unsafe):
    request, _, hooks = setup(tmp_path)
    tracker, agent = stopped()
    hooks.progress_tracker = tracker
    if unsafe == "unknown": hooks.ledger.unknown_requests = 1
    elif unsafe == "pending": hooks.ledger.pending["unsettled"] = 100
    elif unsafe == "permission": hooks.permission_stop = True
    else: hooks.ledger.last_failure_kind = "tool_error"
    with pytest.raises(BudgetExceeded):
        hooks.observe_coding_progress(agent)
    assert not hasattr(agent, "_progress_stop_error")
    assert hooks.ledger.model_calls == hooks.ledger.tool_calls == 0


def test_real_verification_progress_uses_the_verifier_result():
    tracker = ProgressTracker()
    assert tracker.verification("failed") is False
    assert tracker.verification("passed") is True
    assert tracker.verification("passed") is False
