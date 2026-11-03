"""Static symbol lookup never executes source, guesses dynamic targets or hides ambiguity."""

import json

import pytest

from corecoder import symbol_resolution
from corecoder.completion import CompletionChecklist
from corecoder.diagnostic_policy import EvidenceToolPolicy
from corecoder.diagnostic_tools import Investigation
from corecoder.evidence_context import observation_view
from corecoder.observation_state import current_observation_ids


@pytest.fixture
def state(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    log = tmp_path / "error.log"
    log.write_text("loss=nan\n")
    state = Investigation(log, project)
    state.extract_log_evidence()
    return state


def write(state, path, content):
    target = state.root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def caller(state, content, path="train.py", start_line=1, max_lines=100):
    write(state, path, content)
    return state.read_project_file(path, start_line, max_lines)["observations"][0]


def lookup(state, source, symbol="calc", **kwargs):
    result = json.loads(state.execute("resolve_project_symbol", {"symbol": symbol, "source_observation_id": source["id"],
                                                               "evidence_ids": [state.evidence[0]["id"]], **kwargs}))
    assert "error" not in result, result
    records = result["observations"]
    return records[-1], {record["id"]: record for record in records[:-1]}


def targets(summary, records):
    return [records[item["definition_id"]]["data"] for item in summary["data"]["candidates"]]


@pytest.mark.parametrize("statement,symbol", [
    ("from ops import divide as calc", "calc"),
    ("import ops as m", "m.divide"),
    ("import pkg.ops", "pkg.ops.divide"),
])
def test_import_aliases_find_actual_function_without_execution(state, statement, symbol):
    module = "pkg/ops.py" if "pkg" in statement else "ops.py"
    write(state, module, "raise RuntimeError('must never run')\ndef divide(a,b,eps=0.0):\n    return a/(b+eps)\n")
    source = caller(state, statement + "\nloss = " + symbol + "(0,0)\n")
    summary, records = lookup(state, source, symbol)
    assert summary["data"]["status"] == "resolved"
    target = targets(summary, records)[0]
    assert target["relative_path"] == module and target["definition_kind"] == "function"
    assert target["current_line_no"] == 2 and "eps=0.0" in target["content"]
    assert len(summary["data"]["candidates"][0]["chain_ids"]) == 2
    assert summary["data"]["stats"]["files"] == 2
    assert source["id"] in summary["data"]["dependency_observation_ids"]


def test_reexports_and_relative_imports_preserve_each_binding_snapshot(state):
    write(state, "pkg/__init__.py", "from .ops import divide as exported\n")
    write(state, "pkg/ops.py", "def divide(a,b):\n    return a/b\n")
    source = caller(state, "from pkg import exported as calc\nloss=calc(0,0)\n")
    summary, records = lookup(state, source)
    assert summary["data"]["status"] == "resolved"
    chain = summary["data"]["candidates"][0]["chain_ids"]
    assert [records[item]["data"]["relative_path"] for item in chain] == ["train.py", "pkg/__init__.py", "pkg/ops.py"]
    source = caller(state, "from .ops import divide as calc\nloss=calc(0,0)\n", path="pkg/train.py")
    summary, records = lookup(state, source)
    assert targets(summary, records)[0]["relative_path"] == "pkg/ops.py"


def test_from_package_submodule_and_namespace_package(state):
    write(state, "pkg/__init__.py", "from . import ops\n")
    write(state, "pkg/ops.py", "def divide(a,b):\n    return a/b\n")
    source = caller(state, "from pkg import ops as m\nloss=m.divide(0,0)\n")
    summary, records = lookup(state, source, "m.divide")
    assert targets(summary, records)[0]["relative_path"] == "pkg/ops.py"
    (state.root / "pkg/__init__.py").unlink()
    summary, records = lookup(state, source, "m.divide")
    assert targets(summary, records)[0]["relative_path"] == "pkg/ops.py"


def test_class_constructor_and_method_definitions(state):
    write(state, "ops.py", "class Join:\n    def __init__(self,dim=0):\n        self.dim=dim\n    def forward(self,x):\n        return x\n")
    source = caller(state, "from ops import Join\nvalue=Join()\nresult=Join.forward(None,[])\n")
    summary, records = lookup(state, source, "Join")
    target = targets(summary, records)[0]
    assert target["definition_kind"] == "class" and "dim=0" in target["content"]
    summary, records = lookup(state, source, "Join.forward")
    assert targets(summary, records)[0]["symbol_name"] == "Join.forward"


def test_local_function_import_and_parameter_shadowing(state):
    write(state, "ops.py", "def divide(a,b):\n    return a/b\n")
    source = caller(state, "def run():\n    from ops import divide as calc\n    return calc(0,0)\n")
    summary, records = lookup(state, source)
    assert targets(summary, records)[0]["relative_path"] == "ops.py"
    source = caller(state, "from ops import divide as calc\ndef run(calc):\n    return calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert not summary["data"]["candidates"] and "dynamic_parameter" in summary["data"]["reasons"]


@pytest.mark.parametrize("tail", ["calc = factory()", "del calc", "for calc in values:\n    pass"])
def test_dynamic_rebinding_does_not_report_a_unique_target(state, tail):
    write(state, "ops.py", "def divide(a,b):\n    return a/b\n")
    source = caller(state, "from ops import divide as calc\n" + tail + "\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert summary["data"]["status"] != "resolved"
    assert "wildcard_or_dynamic_binding" in summary["data"]["reasons"]


def test_multiple_conditional_definitions_are_explicitly_ambiguous(state):
    source = caller(state, "if flag:\n    def calc(a,b): return a/b\nelse:\n    def calc(a,b): return 0\nloss=calc(0,0)\n")
    summary, records = lookup(state, source)
    assert summary["data"]["status"] == "ambiguous" and len(targets(summary, records)) == 2
    assert "conditional_binding" in summary["data"]["reasons"]


@pytest.mark.parametrize("call,reason", [("value=lambda calc: calc(0,0)", "dynamic_parameter"),
                                        ("value=[calc(0,0) for calc in functions]", "comprehension_scope_unsupported")])
def test_lambda_and_comprehension_bindings_are_not_confused_with_global_imports(state, call, reason):
    write(state, "ops.py", "def calc(a,b): return a/b\n")
    source = caller(state, "from ops import calc\n" + call + "\n")
    summary, _ = lookup(state, source)
    assert not summary["data"]["candidates"] and reason in summary["data"]["reasons"]


def test_same_reference_in_distinct_scopes_requires_a_line(state):
    source = caller(state, "def a(calc):\n    return calc(0,0)\ndef calc(a,b): return a/b\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert "ambiguous_callsite_use_at_line" in summary["data"]["reasons"]
    summary, records = lookup(state, source, at_line=4)
    assert summary["data"]["status"] == "resolved" and targets(summary, records)[0]["current_line_no"] == 3


def test_import_cycles_star_imports_and_external_modules_are_unknowns(state):
    write(state, "a.py", "from b import calc\n")
    write(state, "b.py", "from a import calc\n")
    source = caller(state, "from a import calc\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert "import_cycle" in summary["data"]["reasons"] and not summary["data"]["candidates"]
    source = caller(state, "from a import *\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert "wildcard_or_dynamic_binding" in summary["data"]["reasons"]
    source = caller(state, "from unavailable_external import calc\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert "module_not_in_project" in summary["data"]["reasons"]


def test_module_package_collision_returns_both_candidates(state):
    write(state, "ops.py", "def calc(a,b): return a/b\n")
    write(state, "ops/__init__.py", "def calc(a,b): return 0\n")
    source = caller(state, "from ops import calc\nloss=calc(0,0)\n")
    summary, records = lookup(state, source)
    assert summary["data"]["status"] == "ambiguous" and len(targets(summary, records)) == 2
    assert "module_package_collision" in summary["data"]["reasons"]


def test_no_unrelated_project_scan_or_symlink_traversal(state, tmp_path):
    other = tmp_path / "external.py"
    other.write_text("def calc(a,b): return a/b\n")
    try:
        (state.root / "ops.py").symlink_to(other)
    except OSError:
        pytest.skip("Symlinks unavailable")
    write(state, "unrelated/secret.py", "def calc(): pass\n")
    source = caller(state, "from ops import calc\nloss=calc(0,0)\n")
    summary, records = lookup(state, source)
    assert "symlink_excluded" in summary["data"]["reasons"]
    assert summary["data"]["stats"]["files"] == 1
    assert all(record["data"]["relative_path"] != "unrelated/secret.py" for record in records.values())


def test_changed_caller_and_later_changed_definition_invalidate_relationships(state):
    target = write(state, "ops.py", "def calc(a,b): return a/b\n")
    source = caller(state, "from ops import calc\nloss=calc(0,0)\n")
    summary, _ = lookup(state, source)
    assert summary["id"] in current_observation_ids(state)
    target.write_text("def calc(a,b): return 0\n")
    state.read_project_file("ops.py")
    assert summary["id"] not in current_observation_ids(state)
    write(state, "train.py", "loss=3\n")
    changed, _ = lookup(state, source)
    assert "source_version_changed" in changed["data"]["reasons"]
    assert source["id"] not in current_observation_ids(state)


@pytest.mark.parametrize("limits,reason", [({"files": 1}, "file_limit"), ({"file_bytes": 2}, "too_large"),
                                         ({"bytes": 2}, "too_large"), ({"depth": 0}, "depth_limit"),
                                         ({"ast_nodes": 2}, "ast_node_limit"), ({"observations": 1}, "observation_limit")])
def test_limits_are_visible_and_no_unbounded_reads_occur(state, monkeypatch, limits, reason):
    write(state, "ops.py", "def calc(a,b): return a/b\n")
    source = caller(state, "from ops import calc\nloss=calc(0,0)\n")
    for key, value in limits.items():
        monkeypatch.setitem(symbol_resolution.SYMBOL_LIMITS, key, value)
    summary, _ = lookup(state, source)
    assert reason in summary["data"]["reasons"] and not summary["data"]["candidates"]
    assert summary["data"]["stats"]["bytes_read"] <= symbol_resolution.SYMBOL_LIMITS["bytes"]


@pytest.mark.parametrize("change", [{"symbol": "eval('bad')"}, {"symbol": "../secret"}, {"symbol": "x..f"},
                                     {"evidence_ids": ["unknown"]}, {"source_observation_id": "invented"},
                                     {"at_line": True}, {"at_line": 1000}, {"max_results": 0}])
def test_invalid_or_unobserved_requests_are_rejected_before_module_reads(state, change):
    source = caller(state, "def calc(a,b): return a/b\nloss=calc(0,0)\n")
    result = json.loads(state.execute("resolve_project_symbol", {"symbol": "calc", "source_observation_id": source["id"],
        "evidence_ids": [state.evidence[0]["id"]], **change}))
    assert "error" in result and state.events[-1]["status"] == "error"


def test_policy_and_checklist_follow_an_observed_imported_numeric_call(state):
    write(state, "ops.py", "def calc(a,b): return a/b\n")
    caller(state, "from ops import calc\nloss=calc(0,0)\n")
    assert "resolve_project_symbol" in EvidenceToolPolicy().select(state)[0]
    state.search_project_source("loss", [state.evidence[0]["id"]])
    plan = CompletionChecklist(state)
    action = next(item["next_action"] for item in plan.status()["checks"] if item["kind"] == "symbol_lookup")
    result = json.loads(state.execute(action["tool"], action["arguments"]))
    assert "error" not in result
    assert plan.status()["ready_for_report"]
    summary = result["observations"][-1]
    projected = observation_view(summary)
    assert projected["data"]["candidates"] and projected["data"]["dependency_observation_ids"]


def test_frameless_symbol_demo_reaches_real_repair_verification(tmp_path):
    from corecoder.symbol_demo import run_demo
    result = run_demo(tmp_path / "demo")
    assert result["diagnosis_status"] == "completed" and result["repair_status"] == "verified"
    assert "resolve_project_symbol" in result["diagnosis_tools"]
    assert result["verification_exit_code"] == 0 and result["provider_requests"] == 0


def test_checklist_prioritizes_evidence_related_symbol_over_an_earlier_unrelated_import(state):
    caller(state, "from helpers import prepare\nfrom metrics import loss_score\ntmp=prepare()\nloss=loss_score(0,0)\n")
    state.search_project_source("loss", [state.evidence[0]["id"]])
    actions = [item["next_action"] for item in CompletionChecklist(state).status()["checks"] if item["kind"] == "symbol_lookup"]
    assert len(actions) == 1 and actions[0]["arguments"]["symbol"] == "loss_score"
    assert actions[0]["arguments"]["at_line"] == 4
