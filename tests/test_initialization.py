"""Initializer coverage is a source-reading requirement, never proof of blame."""

import ast
import json

import pytest

from corecoder.completion import CompletionChecklist
from corecoder.diagnose import _system_prompt, run_diagnosis
from corecoder.diagnostic_tools import Investigation
from corecoder.evidence_context import observation_view
from corecoder.initialization import initialization_lead
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.locator import _function_scopes


def make_state(tmp_path, initializer="    def __init__(self, axis=0):\n        self.axis = axis\n", suffix=""):
    root = tmp_path / "project"
    root.mkdir()
    text = ("class Mixer:\n" + initializer + "\n" + "    # unrelated\n" * 24
            + "    def forward(self, x):\n        return cat(x, self.axis)\n" + suffix)
    source = root / "ops.py"
    source.write_text(text, encoding="utf-8")
    line = text.splitlines().index("        return cat(x, self.axis)") + 1
    log = tmp_path / "failure.log"
    log.write_text(f'Traceback (most recent call last):\n  File "{source}", line {line}, in forward\n'
                   '    return cat(x, self.axis)\nRuntimeError: Sizes of tensors must match\n', encoding="utf-8")
    state = Investigation(log, root)
    state.extract_log_evidence()
    state.locate_source([state.evidence[0]["id"]])
    return state


def init_check(state):
    return next(check for check in CompletionChecklist(state).status()["checks"] if check["kind"] == "initialization_read")


def do_check(state):
    action = init_check(state)["next_action"]
    return json.loads(state.execute(action["tool"], action["arguments"]))


def test_forward_location_does_not_count_as_observed_default(tmp_path):
    state = make_state(tmp_path)
    check = init_check(state)
    assert check["status"] == "pending" and not check["observation_ids"]
    assert check["next_action"]["arguments"] == {"path": str(state.root / "ops.py"), "start_line": 2, "max_lines": 2}
    assert not CompletionChecklist(state).status()["ready_for_report"]
    result = do_check(state)
    assert "axis=0" in result["observations"][0]["data"]["content"]
    assert init_check(state)["status"] == "observed"
    assert CompletionChecklist(state).status()["ready_for_report"]


def test_status_does_not_read_or_stat_files(tmp_path, monkeypatch):
    state = make_state(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Hidden filesystem access")
    monkeypatch.setattr(type(state.root), "open", forbidden)
    monkeypatch.setattr(type(state.root), "stat", forbidden)
    assert init_check(state)["status"] == "pending"


def test_actual_fragment_pages_must_cover_header_and_assignment(tmp_path):
    state = make_state(tmp_path)
    state.read_project_file("ops.py", 2, 1)
    assert init_check(state)["status"] == "pending"
    assert init_check(state)["next_action"]["arguments"]["start_line"] == 3
    state.read_project_file("ops.py", 3, 1)
    assert init_check(state)["status"] == "observed"


def test_new_version_does_not_reuse_old_initializer_page(tmp_path):
    state = make_state(tmp_path)
    do_check(state)
    target = state.root / "ops.py"
    target.write_text(target.read_text().replace("axis=0", "axis=2"))
    state.read_project_file("ops.py", 28, 10)
    state.locate_source([state.evidence[0]["id"]])
    assert init_check(state)["status"] == "pending"
    assert "axis=2" in do_check(state)["observations"][0]["data"]["content"]


def test_initialization_lines_survive_context_projection(tmp_path):
    state = make_state(tmp_path)
    item = do_check(state)["observations"][0]
    data = observation_view(item, memory=True)["data"]
    assert data["initialization_source_lines"] == [
        {"line_no": 2, "text": "    def __init__(self, axis=0):"},
        {"line_no": 3, "text": "        self.axis = axis"}]


def test_multiline_assignment_and_signature_require_complete_source(tmp_path):
    state = make_state(tmp_path, "    def __init__(\n        self, axis=0\n    ):\n        self.axis = (\n            axis\n        )\n")
    action = init_check(state)["next_action"]["arguments"]
    assert action["start_line"] == 2 and action["max_lines"] == 6
    state.read_project_file("ops.py", 2, 4)
    assert init_check(state)["status"] == "pending"
    do_check(state)
    assert init_check(state)["status"] == "observed"


@pytest.mark.parametrize("initializer", ["", "    def __init__(self):\n        pass\n",
    "    def __init__(self, axis=0):\n        self.axis=axis\n    def __init__(self, axis=1):\n        self.axis=axis\n"])
def test_inherited_dynamic_or_duplicate_initialization_is_unknown(tmp_path, initializer):
    state = make_state(tmp_path, initializer)
    assert init_check(state)["status"] == "blocked"
    assert CompletionChecklist(state).status()["ready_for_report"]
    assert init_check(state)["blocked_reason"]


def test_multiple_methods_do_not_hide_each_others_attribute_checks(tmp_path):
    state = make_state(tmp_path, "    def __init__(self, axis=0):\n        self.axis=axis\n        self.scale=2\n",
                       "    def transform(self, x):\n        return x*self.scale\n")
    scopes = _function_scopes(ast.parse((state.root / "ops.py").read_text()))
    data = next(item["data"] for item in state.observations.values() if item["kind"] == "source_location")
    state.observe("source_location", {**data, "function": next(s for s in scopes if s["name"] == "transform")},
                  [state.evidence[0]["id"]])
    checks = [c for c in CompletionChecklist(state).status()["checks"] if c["kind"] == "initialization_read"]
    assert len(checks) == 2 and all(c["status"] == "pending" for c in checks)


def test_nested_function_receiver_does_not_create_a_method_requirement():
    scopes = _function_scopes(ast.parse("class M:\n def f(self):\n  def inner(): return self.axis\n  return 1\n"))
    assert all("initialization_check" not in scope for scope in scopes)


def test_receiver_names_and_positional_only_parameters_are_supported():
    tree = ast.parse("class M:\n def __init__(obj, axis=2): obj.axis=axis\n def f(obj, /): return obj.axis\n")
    lead = initialization_lead(tree.body[0], tree.body[0].body[1], "M.f")
    assert lead["required_lines"] == [2]


def test_long_header_is_explicitly_incomplete(tmp_path):
    initializer = "    def __init__(\n        self,\n" + "        # comment\n" * 24 + "        axis=0\n    ):\n        self.axis=axis\n"
    state = make_state(tmp_path, initializer)
    do_check(state)
    assert init_check(state)["status"] == "blocked"
    assert "truncated" in init_check(state)["blocked_reason"]


def test_default_expression_is_never_executed(tmp_path):
    sentinel = tmp_path / "executed"
    state = make_state(tmp_path, f"    def __init__(self, axis=open({str(sentinel)!r}, 'w')):\n        self.axis=axis\n")
    do_check(state)
    assert not sentinel.exists()


def test_report_refusal_cannot_skip_initializer_with_checklist(tmp_path):
    state = make_state(tmp_path)
    events = [LLMResponse(tool_calls=[ToolCall("extract", "extract_log_evidence", {})]),
              LLMResponse(tool_calls=[ToolCall("locate", "locate_source", {"evidence_ids": [state.evidence[0]["id"]]})])]
    model = ScriptedLLM(events + [LLMResponse(content="done")] * 5)
    result = run_diagnosis(state.log_file, state.root, model, completion_mode="checklist", max_rounds=5)
    assert result["status"] == "incomplete"
    assert any(c["kind"] == "initialization_read" and c["status"] == "pending"
               for c in result["investigation_checklist"]["checks"])
    assert result["metrics"]["llm_calls"] <= 5


def test_prompt_requires_observed_values_and_contract_uncertainty(tmp_path):
    prompt = _system_prompt(make_state(tmp_path))
    for text in ("禁止凭常见库用法补全默认值", "没有接口契约", "不能断言调用方有错", "limitations"):
        assert text in prompt


def test_oversized_initializer_line_blocks_without_repeat(tmp_path):
    state = make_state(tmp_path, "    def __init__(self, axis=" + " " * 13000 + "0):\n        self.axis=axis\n")
    do_check(state)
    check = init_check(state)
    assert check["status"] == "blocked" and check["next_action"] is None
    assert "character bound" in check["blocked_reason"]


@pytest.mark.parametrize("part", ["method", "initializer"])
def test_ast_bound_is_explicit_unknown(part):
    init = " def __init__(self):\n  self.axis=0\n" + ("  pass\n" * 4001 if part == "initializer" else "")
    method = " def f(self):\n  value=self.axis\n" + ("  pass\n" * 4001 if part == "method" else "")
    tree = ast.parse("class M:\n" + init + method)
    lead = initialization_lead(tree.body[0], tree.body[0].body[1], "M.f")
    assert lead["status"] == "unknown" and "bound reached" in lead["reason"]


def test_unrelated_search_candidate_cannot_expand_initialization_requirements(tmp_path):
    state = make_state(tmp_path)
    source = next(item for item in state.observations.values() if item["kind"] == "source_location")
    state.observations.clear()
    state.observe("source_candidate", source["data"], source["evidence_ids"])
    assert not any(check["kind"].startswith("initialization_") for check in CompletionChecklist(state).status()["checks"])


def test_unverified_stack_location_cannot_create_initialization_requirement(tmp_path):
    state = make_state(tmp_path)
    source = next(item for item in state.observations.values() if item["kind"] == "source_location")
    state.observations.clear()
    state.observe("source_location", {**source["data"], "function_verified": False}, source["evidence_ids"])
    assert not any(check["kind"].startswith("initialization_") for check in CompletionChecklist(state).status()["checks"])


def test_resolved_method_definition_also_requires_initialization_read(tmp_path):
    state = make_state(tmp_path)
    state.observations.clear()
    target = state.root / "ops.py"
    target.write_text(target.read_text().replace("class Mixer:\n", "class Mixer:\n" + "    # class padding\n" * 24))
    (state.root / "entry.py").write_text("from ops import Mixer\nvalue=Mixer.forward(None, 0)\n")
    source = state.read_project_file("entry.py")["observations"][0]
    state.resolve_project_symbol("Mixer.forward", source["id"], [state.evidence[0]["id"]], at_line=2)
    assert init_check(state)["status"] == "pending"
    do_check(state)
    assert init_check(state)["status"] == "observed"
