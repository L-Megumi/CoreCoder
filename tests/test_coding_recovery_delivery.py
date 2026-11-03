"""V2 delivers exact registered recovery windows to a real gateway call.

The provider is offline. Actual tools, RequestRecorder and BudgetedModel remain
in use; these tests do not infer delivery from a predetermined repair result.
"""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from corecoder.llm import LLMResponse, ScriptedLLM
from corecoder.workflow import RepairAgent, project_directory
from researchcoder.context.compression import validate_pairs
from researchcoder.context.engine import EvidenceContextEngine, RESTORE
from researchcoder.context.policy import ContextPolicy
from researchcoder.runtime.budget import BudgetedModel, BudgetExceeded, BudgetLedger, RuntimeHooks
from researchcoder.runtime.request_records import RequestRecorder
from researchcoder.runtime.tracing import TraceSink
from tests.test_coding_context import setup, run_tool


class RecordingModel:
    model = "offline-delivery-test"

    def __init__(self, response=None):
        self.calls = []
        self.response = response or LLMResponse(content="Received the source window.", prompt_tokens=20, completion_tokens=5)

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(copy.deepcopy(messages))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def environment(tmp_path, *, mode="observed-source-v2", strategy="evidence", source=None, context=16000):
    base, _, _, io = setup(tmp_path, source=source, max_context=context)
    policy = ContextPolicy(role="repair", strategy=strategy, allow_recovery=True, coding_source_mode=mode)
    request = replace(base.request, context_policy=policy)
    trace = TraceSink(request.spec.job_id, request.spec.task_id, request.attempt_id)
    ledger = BudgetLedger(request.limits, trace)
    manager = EvidenceContextEngine(request, io, ledger, trace)
    hooks = RuntimeHooks(ledger, trace, (*request.spec.policy.allowed_tool_names, RESTORE))
    hooks.context_engine = manager
    with project_directory(request.spec.workspace.root):
        agent = RepairAgent(ScriptedLLM([]), {"project_root": request.spec.workspace.root, "file_snapshots": {}},
                            allowed_tool_names=request.spec.policy.allowed_tool_names)
    agent.execution_hooks = hooks
    hooks.bind_agent(agent)
    return manager, agent, hooks, io


def restored_environment(tmp_path, **kwargs):
    source = kwargs.pop("source", None)
    if source is None:
        rows = ["# neutral row %03d" % number for number in range(150)]
        rows[69] = "# " + "x" * 280 + " LONG_LINE_TAIL_SENTINEL"
        rows[79] = "# LATE_RECOVERY_SENTINEL"
        source = "\n".join(rows) + "\n"
    manager, agent, hooks, io = environment(tmp_path, source=source, **kwargs)
    _, history = run_tool(manager, agent, "read_file", {"file_path": "module.py"}, "initial-read")
    page = manager.coding_sources.entries[-1]["document_id"]
    output, restored = run_tool(manager, agent, RESTORE,
        dict(reference_id=page, start_line=21, max_lines=100), "restore-window")
    assert "LATE_RECOVERY_SENTINEL" in output and "LONG_LINE_TAIL_SENTINEL" in output
    return manager, agent, hooks, io, history+restored, output


def gateway(manager, hooks, io, model, recorder=None):
    recorder = recorder or RequestRecorder(io, hooks.trace, dict(model=model.model))
    return BudgetedModel(model, hooks.ledger, hooks.trace, context_engine=manager, request_recorder=recorder), recorder


def returned(messages, identity):
    matches = [message for message in messages if message.get("role") == "tool" and message.get("tool_call_id") == identity]
    assert len(matches) == 1
    return json.loads(matches[0]["content"])


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_exact_window_tail_and_long_lines_reach_next_admitted_request(tmp_path, strategy):
    manager, agent, hooks, io, history, output = restored_environment(tmp_path, strategy=strategy)
    model = RecordingModel()
    client, recorder = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    actual = io.read_json(recorder.records[0])["request"]["messages"]
    assert actual == model.calls[0]
    validate_pairs(actual)
    recovered = returned(actual, "restore-window")
    assert recovered == json.loads(output)
    assert "LONG_LINE_TAIL_SENTINEL" in recovered["content"] and "LATE_RECOVERY_SENTINEL" in recovered["content"]
    assert len(recovered["content"].splitlines()) == 100
    entry = manager.recovery_delivery.receipt()["entries"][0]
    assert entry["status"] == "delivered"
    assert entry["content_sha256"] == hashlib.sha256(recovered["content"].encode()).hexdigest()
    assert entry["acknowledged_request"] == manager.recovery_delivery.receipt()["attempts"][0]["request_event_id"]
    assert hooks.ledger.model_calls == 1 and hooks.ledger.tool_calls == 2 and not hooks.ledger.pending
    assert manager.coding_sources.receipt()["mode"] == "observed-source-v2"
    events = [e for e in hooks.trace.events if e["event_type"].startswith("context.recovery.")]
    assert [e["event_type"] for e in events] == ["context.recovery.pending", "context.recovery.delivery_started", "context.recovery.delivered"]
    assert "SENTINEL" not in json.dumps(events) and "content" not in json.dumps(events)


def test_v1_keeps_frozen_recompression_behavior_and_has_no_delivery_registry(tmp_path):
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path, mode="observed-source-v1")
    model = RecordingModel()
    client, _ = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    assert "LATE_RECOVERY_SENTINEL" not in json.dumps(model.calls[0])
    assert "LONG_LINE_TAIL_SENTINEL" not in json.dumps(model.calls[0])
    assert manager.recovery_delivery is None
    assert manager.coding_sources.receipt()["mode"] == "observed-source-v1"


def test_same_document_two_windows_in_one_tool_group_keep_independent_exact_replies(tmp_path):
    source = "\n".join("# independent line %03d" % number for number in range(160)) + "\n"
    manager, agent, hooks, io = environment(tmp_path, source=source)
    _, initial = run_tool(manager, agent, "read_file", {"file_path": "module.py"}, "initial-read")
    page = manager.coding_sources.entries[-1]["document_id"]
    first, replies_a = run_tool(manager, agent, RESTORE, dict(reference_id=page, start_line=20, max_lines=40), "restore-a")
    second, replies_b = run_tool(manager, agent, RESTORE, dict(reference_id=page, start_line=100, max_lines=40), "restore-b")
    batch = [dict(role="assistant", content=None, tool_calls=replies_a[0]["tool_calls"]+replies_b[0]["tool_calls"]),
             replies_a[1], replies_b[1]]
    model = RecordingModel()
    client, recorder = gateway(manager, hooks, io, model)
    client.chat(initial+batch, agent._tool_schemas())
    messages = io.read_json(recorder.records[0])["request"]["messages"]
    assert returned(messages, "restore-a") == json.loads(first)
    assert returned(messages, "restore-b") == json.loads(second)
    assert returned(messages, "restore-a")["content"] != returned(messages, "restore-b")["content"]
    assert [e["source_start_line"] for e in manager.recovery_delivery.receipt()["entries"]] == [20, 100]
    assert all(e["status"] == "delivered" for e in manager.recovery_delivery.receipt()["entries"])
    assert hooks.ledger.tool_calls == 3 and hooks.ledger.model_calls == 1
    validate_pairs(messages)


def test_multiple_prepare_calls_do_not_acknowledge_and_known_response_allows_later_compression(tmp_path):
    manager, agent, hooks, io, history, output = restored_environment(tmp_path)
    for _ in range(2):
        view = manager.prepare(history, agent._tool_schemas())
        assert returned(view, "restore-window") == json.loads(output)
        assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"
    assert not manager.recovery_delivery.receipt()["attempts"] and hooks.ledger.model_calls == 0
    model = RecordingModel()
    client, _ = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    later = manager.prepare(history+[dict(role="assistant", content="Now continue the authorized task.")], agent._tool_schemas())
    assert "LATE_RECOVERY_SENTINEL" not in json.dumps(later)
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "delivered"


def test_complete_window_cannot_be_summarized_to_evade_context_budget(tmp_path):
    rows = ["# line%03d " % n + "a" * 100 for n in range(120)]
    rows[69] += " LONG_LINE_TAIL_SENTINEL"
    rows[79] += " LATE_RECOVERY_SENTINEL"
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path, source="\n".join(rows), context=2500)
    model = RecordingModel()
    client, recorder = gateway(manager, hooks, io, model)
    with pytest.raises(BudgetExceeded):
        client.chat(history, agent._tool_schemas())
    assert not model.calls and not recorder.records
    assert hooks.ledger.model_calls == 0 and hooks.ledger.local_limit_detail == "context_tokens"
    assert not hooks.ledger.pending
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"
    assert not manager.recovery_delivery.receipt()["attempts"]


@pytest.mark.parametrize("failure", ["provider_error", "unknown_usage", "recorder_error", "model_quota"])
def test_failed_or_unknown_model_call_never_acknowledges_or_replays_tools(tmp_path, failure):
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path)
    response = RuntimeError("offline provider failure") if failure == "provider_error" else (
        LLMResponse(content="Reply without confirmed usage", usage_complete=False) if failure == "unknown_usage" else None)
    model = RecordingModel(response)
    class BrokenRecorder:
        def capture(self, *args, **kwargs):
            raise OSError("offline request archive unavailable")
    recorder = BrokenRecorder() if failure == "recorder_error" else None
    if failure == "model_quota":
        hooks.ledger.model_calls = hooks.ledger.limits.max_model_calls
    client, _ = gateway(manager, hooks, io, model, recorder=recorder)
    if failure == "unknown_usage":
        client.chat(history, agent._tool_schemas())
    else:
        with pytest.raises((RuntimeError, OSError)):
            client.chat(history, agent._tool_schemas())
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"
    assert not any(e["event_type"] == "context.recovery.delivered" for e in hooks.trace.events)
    assert hooks.ledger.tool_calls == 2 and not hooks.ledger.pending
    assert len(model.calls) == (1 if failure in ("provider_error", "unknown_usage") else 0)
    if failure != "model_quota":
        assert hooks.ledger.unknown_requests == 1 and hooks.ledger.charged_tokens > 0


def test_retry_with_unknown_first_transport_does_not_claim_confirmed_delivery(tmp_path):
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path)
    class RetriedModel(RecordingModel):
        _call_with_retry = True
        def chat(self, messages, tools=None, on_token=None):
            self.calls.append(copy.deepcopy(messages))
            try:
                with self._transport_context():
                    raise RuntimeError("offline first transport interrupted")
            except RuntimeError:
                pass
            with self._transport_context():
                return self.response
    model = RetriedModel()
    client, _ = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    assert hooks.ledger.model_calls == 2 and hooks.ledger.unknown_requests == 1
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"
    assert manager.recovery_delivery.receipt()["attempts"][0]["status"] == "unconfirmed"


@pytest.mark.parametrize("mutation", ["missing", "content", "arguments", "duplicate"])
def test_missing_or_forged_pending_tool_reply_is_rejected_before_provider(tmp_path, mutation):
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path)
    if mutation == "missing":
        history = history[:-2]
    elif mutation == "content":
        value = json.loads(history[-1]["content"])
        value["content"] = "fabricated replacement"
        history[-1]["content"] = json.dumps(value)
    elif mutation == "arguments":
        history[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps(dict(reference_id="forged"))
    else:
        history += copy.deepcopy(history[-2:])
    model = RecordingModel()
    client, _ = gateway(manager, hooks, io, model)
    with pytest.raises(ValueError):
        client.chat(history, agent._tool_schemas())
    assert not model.calls and hooks.ledger.model_calls == 0
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"


@pytest.mark.parametrize("changed", ["module.py", ".projectrc", "README.rst"])
@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_source_change_before_delivery_changes_only_freshness_not_recovered_content(tmp_path, changed, strategy):
    manager, agent, hooks, io, history, output = restored_environment(tmp_path, strategy=strategy)
    root = Path(manager.request.spec.workspace.root)
    (root / changed).write_text("new public candidate version")
    agent.read_cache.clear()
    model = RecordingModel()
    client, recorder = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    actual = io.read_json(recorder.records[0])["request"]["messages"]
    assert len(model.calls) == 1 and actual == model.calls[0]
    received = returned(actual, "restore-window")
    assert received["content"] == json.loads(output)["content"]
    assert not received["current_eligible"] and received["classification"] == "historical_observation"
    assert json.loads(history[-1]["content"])["current_eligible"]  # Original history is immutable.
    assert not agent.read_cache
    assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "delivered"


@pytest.mark.parametrize("mode", ["observed-source-v1", "observed-source-v2"])
def test_original_historical_marker_preserves_sibling_read_and_v1_behavior(tmp_path, mode):
    manager, agent, hooks, io, history, output = restored_environment(tmp_path, mode=mode, strategy="original")
    _, sibling = run_tool(manager, agent, "read_file", {"file_path": "module.py"}, "sibling-read")
    history = history[:-2] + [dict(role="assistant", content=None,
        tool_calls=history[-2]["tool_calls"] + sibling[0]["tool_calls"]), history[-1], sibling[1]]
    root = Path(manager.request.spec.workspace.root)
    (root / ".projectrc").write_text("changed complete candidate identity")
    agent.read_cache.clear()
    model = RecordingModel()
    client, recorder = gateway(manager, hooks, io, model)
    client.chat(history, agent._tool_schemas())
    actual = io.read_json(recorder.records[0])["request"]["messages"]
    assert len(model.calls) == 1 and actual == model.calls[0]
    validate_pairs(actual)
    replies = {m["tool_call_id"]: m["content"] for m in actual if m.get("role") == "tool"}
    marker = "[Historical source observation; re-read current candidate before use]\n"
    assert replies["sibling-read"].startswith(marker)
    if mode == "observed-source-v1":
        assert replies["restore-window"].startswith(marker)
        assert manager.recovery_delivery is None
    else:
        recovered = json.loads(replies["restore-window"])
        assert recovered["content"] == json.loads(output)["content"]
        assert not recovered["current_eligible"] and recovered["classification"] == "historical_observation"
        assert manager.recovery_delivery.receipt()["entries"][0]["status"] == "delivered"
    assert not agent.read_cache


def test_changed_recovery_artifact_is_rejected_before_model_delivery(tmp_path):
    manager, agent, hooks, io, history, _ = restored_environment(tmp_path)
    entry = manager.recovery_delivery.entries["restore-window"]
    io.resolve(entry["original"].artifact).write_bytes(b"corrupted")
    model = RecordingModel()
    client, _ = gateway(manager, hooks, io, model)
    with pytest.raises(ValueError):
        client.chat(history, agent._tool_schemas())
    assert not model.calls and manager.recovery_delivery.receipt()["entries"][0]["status"] == "pending"
