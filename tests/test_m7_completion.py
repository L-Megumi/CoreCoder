"""New progress mode reaches a bounded closing phase before safe local stops."""
from dataclasses import replace
import json

import pytest

from corecoder.llm import ToolCall
from researchcoder.runtime.coding import CODING_FINAL_REPORT, CodingContinuationPolicy
from researchcoder.runtime.protocol import RuntimeLimits
from tests.test_m6_completion import setup, execute, response


def fixture(tmp_path, *, version="local-stop-v3", source="value = 0\n"):
    limits = RuntimeLimits(max_model_calls=12, max_rounds=10, max_tool_calls=24,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=512)
    request, io, hooks = setup(tmp_path, limits=limits, source=source)
    request = replace(request, coding_continuation=CodingContinuationPolicy(mode=version))
    hooks.ledger.coding_continuation = request.coding_continuation
    return request, io, hooks


def varied_reads():
    return [response(calls=[ToolCall(str(i), "read_file", dict(file_path="module.py", offset=i+1, limit=1))])
            for i in range(8)]


def test_progress_mode_closes_before_investigation_guard_without_claiming_unmade_fix(tmp_path):
    request, io, hooks = fixture(tmp_path, source="value = 1\n" + "\n".join(f"# line {i}" for i in range(9)))
    turns = varied_reads() + [response(json.dumps(dict(status="completed", summary="No changes made.")))]
    result, raw, model = execute(request, io, hooks, turns)
    assert len(model.inputs) == 9
    assert model.inputs[-1]["tools"] == []
    assert model.inputs[-1]["messages"][-1] == dict(role="user", content=CODING_FINAL_REPORT)
    assert raw["coding_final_report"]["trigger"] == "investigation_limit"
    assert raw["coding_final_report"]["reason"] == "investigation_ended_without_edit"
    assert raw["file_changes"] == [] and result.verification.status == "passed"
    assert result.status == "incomplete" and not result.workflow_completed
    assert hooks.ledger.model_calls == 9 and hooks.ledger.tool_calls == 9


def test_existing_explicit_v1_local_stop_v2_path_is_preserved(tmp_path):
    request, io, hooks = fixture(tmp_path, version="local-stop-v2",
        source="value = 1\n" + "\n".join(f"# line {i}" for i in range(9)))
    result, raw, model = execute(request, io, hooks, varied_reads())
    assert len(model.inputs) == 8 and hooks.ledger.local_limit_detail == "investigation_limit"
    assert "coding_final_report" not in raw and "coding_progress" not in raw
    assert result.status == "incomplete" and result.verification.status == "passed"


def edited_then_repeated_reads(value=1):
    return [response(calls=[ToolCall("read-first", "read_file", dict(file_path="module.py"))]),
        response(calls=[ToolCall("edit", "edit_file", dict(file_path="module.py", old_string="value = 0", new_string=f"value = {value}"))]),
        *[response(calls=[ToolCall("again-"+str(i), "read_file", dict(file_path="module.py"))]) for i in range(4)]]


@pytest.mark.parametrize("value,expected,verification", [(1,"succeeded","passed"), (2,"failed","failed")])
def test_stalled_post_edit_closes_then_real_public_verifier_decides(tmp_path, value, expected, verification):
    request, io, hooks = fixture(tmp_path)
    turns = edited_then_repeated_reads(value) + [response(json.dumps(dict(status="completed", summary="Edited value; public check pending.")))]
    result, raw, model = execute(request, io, hooks, turns)
    assert len(model.inputs) == 7 and hooks.ledger.tool_calls == 7
    assert model.inputs[-1]["tools"] == []
    assert raw["coding_final_report"]["trigger"] == "progress_stalled_after_edit"
    assert result.status == expected and result.verification.status == verification
    assert result.workflow_completed and "coding_stop_receipt" not in raw
    assert hooks.ledger.stop_kind is None and hooks.ledger.unknown_requests == 0


def test_stalled_post_edit_cannot_auto_complete_invalid_final_report(tmp_path):
    request, io, hooks = fixture(tmp_path)
    result, raw, model = execute(request, io, hooks, edited_then_repeated_reads() + [response("still thinking")])
    assert result.status == "incomplete" and result.failure_kind == "invalid_json"
    assert result.verification.status == "passed" and not result.workflow_completed
    assert raw["coding_final_report"]["original_response"] == "still thinking"
    assert len(model.inputs) == 7
