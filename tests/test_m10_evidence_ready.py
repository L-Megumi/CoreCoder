"""Evidence-ready investigation with real hooks/tools and one isolated workflow.

Fixed model actions test budgets, evidence contracts and handoff, not LLM accuracy.
The v1 failure remains reproducible and is never relabelled as a success.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corecoder.diagnose import _has_current_observation, run_diagnosis, validate_diagnosis
from corecoder.diagnostic_tools import Investigation
from corecoder.observation_state import current_observation_ids
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import EXPERT_LIMITS, ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.budget import BudgetedModel, BudgetLedger, RuntimeHooks
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.protocol import CodingTaskSpec, ModelSpec, RuntimeLimits, ScriptedTurn
from researchcoder.runtime.tracing import TraceSink
from researchcoder.runtime.worker import ProtocolScript
from tests.test_orchestration_static import call, report_turn
from tests.test_project_consultation import expert_model
from tests.test_project_runner import BAD, GOOD, SECRET, fixture, policy, requests, scripted
from tests.test_m9_code_first import specialist_requests


VERSION = "evidence-ready-v2"
PROFILE = "readiness-v2"
LIMITS = RuntimeLimits(max_model_calls=6, max_rounds=6, max_tool_calls=10,
    max_total_tokens=80000, max_runtime_seconds=45, max_context_tokens=16000,
    max_output_tokens=2048)


def search_turn(path=None):
    args = dict(query="loss", evidence_ids=["$E0"])
    if path is not None:
        args["paths"] = [path]
    return call("search", "search_project_source", args)


def report(index=2):
    # One literal search hit and its summary precede the actual file excerpt.
    # These placeholders are resolved only from actual local tool observations.
    return replace(report_turn(), content=report_turn().content.replace("$O0", "$O" + str(index)))


def run_diagnostic(tmp_path, turns, *, version=VERSION, limits=LIMITS,
                   unknown_at=None, on_response=None, allowed_tools=None):
    root = tmp_path / "project"
    root.mkdir()
    (root / "training.py").write_text("loss = 1\n")
    log = tmp_path / "run.log"
    log.write_text("step=1 loss=nan\n")
    trace = TraceSink("job", "diagnose", "attempt")
    ledger = BudgetLedger(limits, trace)
    hooks = RuntimeHooks(ledger, trace, allowed_tools)
    hooks.optimization = RuntimeOptimization(compact_reports=True, version=version,
        report_reserve_calls=1 if version == VERSION else 2)
    ledger.optimization = hooks.optimization
    fixed = tuple(replace(turn, prompt_tokens=100, completion_tokens=10) for turn in turns)

    class Capture(ProtocolScript):
        def __init__(self):
            super().__init__(ModelSpec(turns=fixed))
            self.requests = []

        def chat(self, messages, tools=None, on_token=None):
            self.requests.append(deepcopy(dict(messages=messages, tools=tools)))
            ordinal = len(self.requests)
            if on_response:
                on_response(ordinal, ledger, hooks)
            response = super().chat(messages, tools, on_token)
            return replace(response, finish_reason="tool_calls" if response.tool_calls else "stop",
                           usage_complete=ordinal != unknown_at)

    llm = Capture()
    model = BudgetedModel(llm, ledger, trace)
    value = run_diagnosis(log, root, model, max_rounds=limits.max_rounds,
        max_tool_calls=limits.max_tool_calls, max_report_repairs=1,
        context_mode="evidence", execution_hooks=hooks, compact_report=True)
    return value, ledger, llm, trace


def schema_names(request):
    return {schema["function"]["name"] for schema in request["tools"] or []}


def trace_types(trace):
    return [event["event_type"] for event in trace.events]


def test_new_profile_is_explicit_bounded_and_old_cap_remains():
    old = ExpertPolicy(mode="public-failure-v1", report_version="explicit-v1")
    assert old.expert_limits == EXPERT_LIMITS
    assert (old.expert_limits.max_model_calls, old.expert_limits.max_rounds,
            old.expert_limits.max_tool_calls) == (4, 4, 6)
    with pytest.raises(ValueError):
        replace(old, expert_limits=LIMITS)
    new = replace(old, report_version=PROFILE, expert_limits=LIMITS)
    assert ExpertPolicy.from_json(new.to_json()) == new and new.sha256 != old.sha256
    assert new.expert_limits.max_total_tokens == old.expert_limits.max_total_tokens == 80000
    assert new.expert_limits.max_runtime_seconds == old.expert_limits.max_runtime_seconds == 45
    for field in ("max_model_calls", "max_rounds", "max_tool_calls", "max_total_tokens", "max_runtime_seconds"):
        with pytest.raises(ValueError):
            replace(new, expert_limits=replace(LIMITS, **{field: getattr(LIMITS, field) + 1}))
    assert RuntimeOptimization.from_json(RuntimeOptimization(compact_reports=True,
        report_reserve_calls=1, version=VERSION).to_json()).version == VERSION
    with pytest.raises(ValueError):
        RuntimeOptimization(compact_reports=True, report_reserve_calls=2, version=VERSION)
    with pytest.raises(ValueError):
        RuntimeOptimization(compact_reports=True, report_reserve_calls=1, version="explicit-report-v1")


def test_m9_search_only_failure_remains_reproducible_under_old_version(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(), report(0)),
        version="explicit-report-v1", limits=EXPERT_LIMITS)
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert result["failure_reason"] == "No usable current file or path observation was obtained"
    assert result["output_failure_kind"] == "invalid_report"
    assert model.requests[2]["tools"] == [] and ledger.model_calls == 3
    assert result["metrics"]["report_repairs"] == 0
    assert [e["tool"] for e in result["tool_events"]] == ["extract_log_evidence", "search_project_source"]


def test_search_then_real_read_can_complete_under_new_version(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         call("read", "read_project_file", {"path": "training.py"}), report()))
    assert result["status"] == "completed", result
    assert "read_project_file" in schema_names(model.requests[2])
    assert result["output_policy"]["version"] == VERSION
    assert ledger.model_calls == 4 and ledger.tool_calls == 3 and not ledger.pending
    assert not ledger.unknown_requests and len(ledger.settlements) == 4
    observations = {v["id"]: v for v in result["observations"]}
    assert any(observations[i]["kind"] == "file_excerpt"
               for i in result["diagnosis"]["current_state"]["observation_ids"])
    assert [e["tool"] for e in result["tool_events"]] == [
        "extract_log_evidence", "search_project_source", "read_project_file"]


def test_sixth_request_is_final_not_the_fifth_round(tmp_path):
    result, ledger, model, trace = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         call("read", "read_project_file", {"path": "training.py"}),
         call("reread", "read_project_file", {"path": "training.py"}),
         call("reread-again", "read_project_file", {"path": "training.py"}), report()))
    assert result["status"] == "completed", result
    assert "read_project_file" in schema_names(model.requests[4])
    assert model.requests[5]["tools"] == []
    assert ledger.model_calls == ledger.rounds == 6 and ledger.tool_calls == 5
    assert len(ledger.settlements) == 6 and not ledger.pending
    assert trace_types(trace).count("diagnosis.finalization.started") == 1


def test_early_report_gets_only_one_budgeted_real_investigation_continuation(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(), report(0),
         call("read", "read_project_file", {"path": "training.py"}), report()))
    assert result["status"] == "completed", result
    records = result["investigation_continuations"]
    assert len(records) == 1 and records[0]["continued"] is True
    assert records[0]["reason"] == "missing_usable_current_observation"
    original = json.loads(records[0]["response"])
    assert original["current_state"]["observation_ids"] != result["diagnosis"]["current_state"]["observation_ids"]
    assert "read_project_file" in schema_names(model.requests[3])
    assert ledger.model_calls == 5 and ledger.tool_calls == 3 and len(ledger.settlements) == 5
    assert result["metrics"]["report_repairs"] == 0


def test_second_early_answer_does_not_loop_or_fabricate_reading(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(), report(0), report(0),
         call("must-not-run", "read_project_file", {"path": "training.py"}), report()))
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert ledger.model_calls == 4 and ledger.tool_calls == 2
    assert len(model.requests) == 4
    assert sum(v["continued"] for v in result["investigation_continuations"]) == 1
    assert all(v["kind"] != "file_excerpt" for v in result["observations"])


def test_last_slot_without_usable_evidence_stops_before_report_transport(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         search_turn(), search_turn(), search_turn(), report(0)))
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert ledger.model_calls == 5 and len(model.requests) == 5
    assert ledger.tool_calls == 5 and not ledger.pending
    assert result["report_finalization"]["phase"] == "investigate"
    assert all(v["kind"] != "file_excerpt" for v in result["observations"])


@pytest.mark.parametrize("refusal", ["unknown", "permission", "model_error"])
def test_untrusted_accounting_or_error_cannot_fund_extra_investigation(tmp_path, refusal):
    def change(ordinal, ledger, hooks):
        if ordinal == 3 and refusal != "unknown":
            if refusal == "permission":
                hooks.permission_stop = True
            else:
                ledger.last_failure_kind = "model_error"
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(), report(0),
         call("must-not-run", "read_project_file", {"path": "training.py"}), report()),
        unknown_at=3 if refusal == "unknown" else None, on_response=change)
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert ledger.model_calls == 3 and len(model.requests) == 3 and ledger.tool_calls == 2
    assert not any(v["continued"] for v in result["investigation_continuations"])
    if refusal == "unknown":
        assert ledger.unknown_requests == 1 and ledger.settlements[-1]["charged"] > 0


def test_reading_outside_project_is_still_rejected(tmp_path):
    marker = tmp_path / "private.py"
    marker.write_text("PRIVATE_NOT_AGENT_DATA = 42\n")
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         call("outside", "read_project_file", {"path": str(marker)}), report(0), report(0)))
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert result["failure_reason"] == "No usable current observation; bounded investigation continuation unavailable"
    assert ledger.model_calls == len(model.requests) == 5 and ledger.tool_calls == 3
    assert ledger.unknown_requests == 0 and not ledger.pending
    assert [v["continued"] for v in result["investigation_continuations"]] == [True, False]
    # The existing reader returns a successfully observed refusal, not a tool
    # exception: exact outside_project metadata with no contents or file hash.
    reads = [v for v in result["tool_events"] if v["tool"] == "read_project_file"]
    assert len(reads) == 1 and reads[0]["status"] == "ok"
    assert reads[0]["arguments"] == {"path": str(marker)}
    observations = {v["id"]: v for v in result["observations"]}
    assert len(reads[0]["observation_ids"]) == 1
    refusal = observations[reads[0]["observation_ids"][0]]
    assert refusal["kind"] == "file_excerpt"
    assert refusal["data"]["status"] == "outside_project"
    assert refusal["data"]["requested_path"] == str(marker)
    assert refusal["data"]["resolved_path"] == str(marker.resolve())
    assert "content" not in refusal["data"] and "file_sha256" not in refusal["data"]
    assert not _has_current_observation(SimpleNamespace(evidence=result["evidence"], observations=observations))
    assert "PRIVATE_NOT_AGENT_DATA" not in json.dumps(model.requests)
    assert marker.read_text() == "PRIVATE_NOT_AGENT_DATA = 42\n"
    assert (tmp_path / "project/training.py").read_text() == "loss = 1\n"


def test_search_candidates_remain_invalid_evidence_even_with_current_ids(tmp_path):
    root = tmp_path / "source"; root.mkdir(); (root / "training.py").write_text("loss = 1\n")
    log = tmp_path / "log"; log.write_text("loss=nan\n")
    state = Investigation(log, root)
    state.extract_log_evidence()
    state.execute("search_project_source", {"query": "loss", "evidence_ids": [state.evidence[0]["id"]]})
    assert current_observation_ids(state) and not _has_current_observation(state)
    state.execute("read_project_file", {"path": "training.py"})
    assert _has_current_observation(state)
    read = next(v for v in state.observations.values() if v["kind"] == "file_excerpt")
    value = json.loads(report().content)
    value["historical_failure"]["evidence_ids"] = [state.evidence[0]["id"]]
    value["current_state"]["observation_ids"] = [read["id"]]
    value["hypotheses"][0]["evidence_ids"] = [state.evidence[0]["id"]]
    value["hypotheses"][0]["observation_ids"] = [read["id"]]
    assert validate_diagnosis(json.dumps(value), state) == value
    value["current_state"]["observation_ids"] = ["forged-observation"]
    with pytest.raises(ValueError):
        validate_diagnosis(json.dumps(value), state)


def test_v2_soft_tool_reserve_does_not_drop_the_last_usable_tool_slot():
    trace = TraceSink("job", "task", "attempt")
    ledger = BudgetLedger(LIMITS, trace)
    hooks = RuntimeHooks(ledger, trace)
    hooks.optimization = RuntimeOptimization(compact_reports=True, report_reserve_calls=1, version=VERSION)
    ledger.tool_calls = 9
    ledger.model_calls = ledger.rounds = 3
    assert hooks.report_due() is False
    ledger.tool_calls = 10
    assert hooks.report_due() is True


@pytest.mark.parametrize("change", ["model_slots", "round_slots", "tools", "unknown", "pending", "permission", "error", "time", "tokens"])
def test_continuation_requires_investigation_and_final_in_same_known_budget(change):
    trace = TraceSink("job", "task", "attempt"); ledger = BudgetLedger(LIMITS, trace)
    hooks = RuntimeHooks(ledger, trace)
    hooks.optimization = RuntimeOptimization(compact_reports=True, report_reserve_calls=1, version=VERSION)
    assert hooks.investigation_continuation_allowed()
    if change == "model_slots": ledger.model_calls = 5
    elif change == "round_slots": ledger.rounds = 5
    elif change == "tools": ledger.tool_calls = 10
    elif change == "unknown": ledger.unknown_requests = 1
    elif change == "pending": ledger.pending["unsettled"] = 1
    elif change == "permission": hooks.permission_stop = True
    elif change == "error": ledger.last_failure_kind = "model_error"
    elif change == "time": ledger.started -= 46
    elif change == "tokens": ledger.charged_tokens = LIMITS.max_total_tokens
    assert not hooks.investigation_continuation_allowed()


def test_format_retry_uses_only_remaining_original_allowance(tmp_path):
    invalid = ScriptedTurn(content="{broken")
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         call("read", "read_project_file", {"path": "training.py"}), invalid, report()))
    assert result["status"] == "completed", result
    assert result["metrics"]["report_repairs"] == 1 and ledger.model_calls == 5
    assert result["report_attempts"][0]["response"] == "{broken"
    assert model.requests[-1]["tools"] is None and ledger.tool_calls == 3


def test_no_format_retry_credit_after_last_report_request(tmp_path):
    result, ledger, model, _ = run_diagnostic(tmp_path,
        (call("extract", "extract_log_evidence"), search_turn(),
         call("read", "read_project_file", {"path": "training.py"}),
         call("reread", "read_project_file", {"path": "training.py"}),
         call("reread-again", "read_project_file", {"path": "training.py"}),
         ScriptedTurn(content="{broken"), report()))
    assert result["status"] == "incomplete" and result["diagnosis"] is None
    assert ledger.model_calls == len(model.requests) == 6
    assert result["metrics"]["report_repairs"] == 0
    assert result["report_finalization"]["repair_declined"]


def test_real_worker_search_read_handoff_first_writer_and_independent_oracle(tmp_path):
    source, task, checks, _ = fixture(tmp_path)
    settings = policy(completion_version="explicit-v1", progress_version="progress-v1",
        max_feedback_rounds=1,
        task_limits=replace(policy().task_limits, max_model_calls=12, max_rounds=10,
                            max_tool_calls=24, max_total_tokens=200000, max_output_tokens=2048),
        job_limits=JobLimits(max_model_calls=24, max_tool_calls=60,
                            max_total_tokens=400000, max_runtime_seconds=180))
    base = expert_model()
    search = call("search", "search_project_source", {"query": "increment", "evidence_ids": ["$E0"]})
    turns = (base.turns[0], replace(search, prompt_tokens=10, completion_tokens=5), base.turns[1],
             replace(base.turns[2], content=base.turns[2].content.replace("$O0", "$O2")))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source,
        checks["public"], checks["f2p"], checks["p2p"], scripted(), policy=settings,
        expert_policy=ExpertPolicy(mode="public-failure-v1", initial_mode="code-first-v1",
            report_version=PROFILE, expert_limits=LIMITS, writer_min_model_calls=12,
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": replace(base, turns=turns)},
        writer_context_policy=ContextPolicy(role="repair", strategy="evidence", allow_recovery=True,
            coding_source_mode="observed-source-v2", working_set_version="active-v1"),
        handoff_policy=HandoffPolicy(mode="local-stop-v3", closure_version="reserve-incomplete-v1",
                                    request_version="explicit-request-v1"))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["consultations"]) == len(raw["rounds"]) == 1
    assert len(raw["records"]) == 2 and len({r["worker_pid"] for r in raw["records"]}) == 2
    assert {r["job_id"] for r in raw["records"]} == {raw["job_id"]}
    assert raw["records"][0]["budget"]["model_calls"] == 4
    assert raw["records"][0]["budget"]["tool_calls"] == 3
    assert raw["budget"]["charged"]["model_calls"] == 7 and not raw["budget"]["pending"]
    assert not any(s["usage_unknown"] for s in raw["budget"]["settlements"].values())
    first = raw["rounds"][0]
    assert first["generation"] == 0 and first["result"]["task_id"] == "coding-0"
    assert first["result"]["status"] == "succeeded" and first["result"]["workflow_completed"]
    writer_spec = CodingTaskSpec.from_json((directory / "coding-0.input.json").read_text())
    assert writer_spec.policy.profile == "project-expert-reread-v1"
    consultation = raw["consultations"][0]
    assert consultation["safe_to_continue"] and consultation["selected_producers"]
    context = io.read_json(ArtifactRef.from_dict(consultation["context_ref"]))
    assert context["observations"] and context["candidate_source_paths"]
    writer_inputs = requests(raw, io)
    assert consultation["selected_producers"][0] in json.dumps(writer_inputs)
    assert SECRET not in json.dumps(writer_inputs + specialist_requests(raw, io))
    reports = [io.read_json(ArtifactRef.from_dict(ref)) for ref in first["result"]["outputs"]]
    report_value = next(v for v in reports if "baseline_hypothesis" in v)
    dispatched = [e["tool"] for e in report_value["events"]]
    assert dispatched.index("read_file") < dispatched.index("edit_file")
    assert (directory / "candidate/module.py").read_text() == GOOD
    assert (source / "module.py").read_text() == BAD
    assert recompute(directory) == result
