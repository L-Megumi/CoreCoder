"""Evidence gates, deterministic memory, complete tool groups and measurable ablations."""

import copy
import json

import pytest

from corecoder.diagnose import run_diagnosis
from corecoder.diagnosis_demo import DemoLLM, create_demo_case, run_demo
from corecoder.diagnostic_policy import DiagnosticAgent, EvidenceToolPolicy
from corecoder.diagnostic_tools import Investigation, investigation_tools
from corecoder.evidence_context import EvidenceContextManager, evidence_view, observation_view, project_tool_result
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall


@pytest.fixture
def state(tmp_path):
    workspace, root, log, _, _ = create_demo_case(tmp_path / "case")
    return Investigation(log, root, allowed_path_roots=[workspace])


def extracted(state):
    return json.loads(state.execute("extract_log_evidence", {}))


def test_bootstrap_requires_evidence_and_blocks_unadvertised_tool(state):
    agent = DiagnosticAgent(ScriptedLLM([]), state, tools=investigation_tools(state), tool_selection="guided")
    assert [item["function"]["name"] for item in agent._tool_schemas()] == ["extract_log_evidence"]
    result = json.loads(agent._exec_tool(ToolCall("bad", "read_project_file", {"path": "train.py"})))
    assert result["error"]["type"] == "PolicyError"
    assert not state.observations
    assert state.events[0]["policy_rejection"]


def test_pending_location_gate_then_direct_recovery_tools_remain(state):
    extracted(state)
    policy = EvidenceToolPolicy()
    assert "locate_source" in policy.select(state)[0]
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    assert policy.select(state)[0] == ["extract_log_evidence", "resolve_project_symbol", "read_project_file", "inspect_paths"]


def test_failed_location_does_not_close_location_gate(state):
    extracted(state)
    state.execute("locate_source", {"evidence_ids": ["unknown"]})
    assert "locate_source" in EvidenceToolPolicy().select(state)[0]


def test_numeric_without_stack_does_not_force_useless_location(state):
    state.log_text = "step=5 loss=nan gradient=inf\n"
    extracted(state)
    assert "locate_source" not in EvidenceToolPolicy().select(state)[0]
    assert "read_project_file" in EvidenceToolPolicy().select(state)[0]


def test_pagination_can_reopen_location_for_newly_exposed_records(state):
    first = json.loads(state.execute("extract_log_evidence", {"limit": 1}))
    state.execute("locate_source", {"evidence_ids": [first["evidence"][0]["id"]]})
    assert "locate_source" not in EvidenceToolPolicy().select(state)[0]
    state.execute("extract_log_evidence", {"offset": 1})
    assert "locate_source" in EvidenceToolPolicy().select(state)[0]


@pytest.mark.parametrize("selection,context", [("all", "baseline"), ("guided", "baseline"),
                                               ("all", "evidence"), ("guided", "evidence")])
def test_same_real_tool_demo_completes_in_each_ablation(tmp_path, selection, context):
    result = run_demo(directory=tmp_path / "demo", tool_selection=selection, context_mode=context)
    assert result["status"] == "completed", result["failure_reason"]
    assert result["metrics"]["tool_calls"] == 4
    assert result["metrics"]["llm_calls"] == 5
    assert len(result["request_metrics"]) == 5
    assert result["request_metrics"][0]["tool_schema_count"] == (1 if selection == "guided" else 6)
    assert result["configuration"]["context_mode"] == context


def test_projection_does_not_mutate_full_store_and_retains_status_hash_ids(state):
    extracted(state)
    output = state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    original = copy.deepcopy(state.observations)
    compact = project_tool_result(output)
    assert len(compact) < len(output)
    projected = json.loads(compact)
    for item in projected["observations"]:
        original_data = original[item["id"]]["data"]
        assert item["data"]["status"] == original_data["status"]
        if "file_sha256" in original_data:
            assert item["data"]["file_sha256"] == original_data["file_sha256"]
    assert state.observations == original
    assert project_tool_result(compact) == compact


def test_middle_path_and_numeric_signal_survive_shortened_message():
    message = "x" * 3000 + " missing path '/old/data/images/val' loss=nan " + "y" * 3000
    result = evidence_view({"id": "ev_test", "type": "NUMERIC_ANOMALY", "content": message})
    assert result["message"]["omitted_chars"] > 0
    assert result["quoted_paths"] == ["/old/data/images/val"]
    assert result["numeric_signals"] == ["nan"]


def test_memory_restores_cause_ancestor_not_yet_exposed_by_page(state):
    state.execute("extract_log_evidence", {"offset": 1, "limit": 1})
    manager = EvidenceContextManager(state)
    memory = json.loads(manager.build_memory(12000))
    ids = {item["id"] for item in memory["evidence"]}
    assert ids == {item["id"] for item in state.evidence}
    child = memory["evidence"][1]
    assert child["cause_id"] in ids


def test_critical_memory_overflow_is_explicit_not_silent(state):
    extracted(state)
    with pytest.raises(ValueError, match="Critical evidence memory"):
        EvidenceContextManager(state).build_memory(200)


def test_compaction_never_calls_llm_or_orphans_tool_messages(state):
    extracted(state)
    output = state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    messages = [{"role": "user", "content": "investigate"}]
    for number in range(12):
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": str(number)}], "content": None},
            {"role": "tool", "tool_call_id": str(number), "content": output},
        ])
    manager = EvidenceContextManager(state, 5000)
    manager.reserved_tokens = 500
    assert manager.maybe_compress(messages, ScriptedLLM([]))
    assert manager.memory
    pending = set()
    for message in messages:
        if message["role"] == "assistant":
            pending.update(call["id"] for call in message.get("tool_calls", []))
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
    assert not pending
    memory = json.loads(manager.memory)
    assert memory["evidence"][1]["cause_id"] == memory["evidence"][0]["id"]


def test_budget_guard_includes_system_and_schemas(state):
    agent = DiagnosticAgent(ScriptedLLM([]), state, tools=investigation_tools(state),
                            max_context_tokens=1000, context_mode="evidence")
    agent._system = "system" * 1000
    with pytest.raises(ValueError, match="insufficient"):
        agent._full_messages()


@pytest.mark.parametrize("argument,value", [("tool_selection", "bad"), ("context_mode", "bad")])
def test_unknown_optimization_mode_rejected(state, argument, value):
    with pytest.raises(ValueError):
        run_diagnosis(state.log_file, state.root, ScriptedLLM([]), **{argument: value})


def test_guided_rejection_can_be_recovered_in_same_agent_loop(state):
    class RecoveringLLM(DemoLLM):
        def __init__(self, *args):
            super().__init__(*args)
            self.first = True

        def chat(self, *args, **kwargs):
            if self.first:
                self.first = False
                return LLMResponse(tool_calls=[ToolCall("premature", "read_project_file", {"path": "train.py"})])
            if self.step == 0:
                # DemoLLM's fixture treats tool errors as terminal; the recovery step is explicit here.
                self.step = 1
                return self._tool(0, "extract_log_evidence", {})
            return super().chat(*args, **kwargs)

    current = state.root / "dataset/images/val"
    previous = state.root.parent / "previous/dataset/images/val"
    result = run_diagnosis(state.log_file, state.root, RecoveringLLM(state.root, current, previous),
                           allowed_path_roots=[state.root.parent], tool_selection="guided", context_mode="evidence")
    assert result["status"] == "completed", result["failure_reason"]
    assert result["metrics"]["tool_errors"] == 1
    assert result["tool_events"][0]["policy_rejection"]


def test_history_compaction_does_not_erase_tool_call_metrics(state):
    config = state.root / "data.yaml"
    config.write_text(config.read_text() + ("# padding " + "x" * 400 + "\n") * 80)
    current = state.root / "dataset/images/val"
    previous = state.root.parent / "previous/dataset/images/val"
    baseline = run_diagnosis(state.log_file, state.root, DemoLLM(state.root, current, previous),
                             allowed_path_roots=[state.root.parent])
    assert baseline["status"] == "completed"
    script = [LLMResponse(tool_calls=[ToolCall(str(index), event["tool"], event["arguments"])])
              for index, event in enumerate(baseline["tool_events"])]
    script.append(LLMResponse(content=json.dumps(baseline["diagnosis"])))
    compact = run_diagnosis(state.log_file, state.root, ScriptedLLM(script), allowed_path_roots=[state.root.parent],
                            tool_selection="guided", context_mode="evidence", max_context_tokens=3500)
    assert compact["status"] == "completed", compact["failure_reason"]
    assert any(event["groups_collapsed"] for event in compact["context_events"])
    assert compact["metrics"]["tool_calls"] == len(compact["tool_events"]) == 4


def test_key_configuration_lines_survive_middle_of_preview():
    content = "# long comment\n" * 30 + "path: /real/data\ntrain: images/train\nval: images/val\n" + "# tail\n" * 50
    item = {"id": "obs_config", "kind": "config_file", "evidence_ids": ["ev_root"], "data": {
        "status": "read", "resolved_path": "/project/data.yaml",
        "snippet": {"content": content, "start_line": 5, "end_line": 87},
    }}
    view = observation_view(item, memory=True)
    assert view["data"]["snippet"]["content"]["omitted_chars"] > 0
    assert view["data"]["config_path_lines"] == [
        {"line_no": 35, "text": "path: /real/data"},
        {"line_no": 36, "text": "train: images/train"},
        {"line_no": 37, "text": "val: images/val"},
    ]


def test_configuration_line_projection_is_bounded_and_marks_omissions():
    item = {"id": "obs_config", "kind": "file_excerpt", "evidence_ids": [], "data": {
        "status": "read", "resolved_path": "/project/data.yaml", "start_line": 1,
        "content": ("path: " + "a" * 1000 + "\n") * 15,
    }}
    data = observation_view(item, memory=True)["data"]
    assert len(data["config_path_lines"]) == 12
    assert data["config_path_lines_omitted"] == 3
    assert data["config_path_lines"][0]["text"]["omitted_chars"] > 0
