import copy
import json

import pytest

from corecoder.diagnose import run_diagnosis
from corecoder.diagnosis_demo import DemoLLM, create_demo_case
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.reasoning_consistency import validate_attribution_consistency


def report(statement, uncertainty="无法确定预期语义，因此不能断言调用方或库实现哪一方有错。"):
    return {"summary": statement, "current_state": {"assessment": "已读源码默认dimension=0。"},
            "hypotheses": [{"explanation": "输入与默认参数不匹配。"}], "limitations": [uncertainty]}


@pytest.mark.parametrize("statement", [
    "这属于调用输入与默认参数不匹配，而非库实现缺陷。", "不是模块本身的问题。",
    "库实现本身没有问题。", "根因是调用方输入错误。", "这是调用方错误。",
])
def test_declared_unknown_cannot_coexist_with_categorical_blame(statement):
    with pytest.raises(ValueError, match="Attribution inconsistency"):
        validate_attribution_consistency(report(statement))


@pytest.mark.parametrize("statement", [
    "输入与默认参数不匹配。", "这不是训练过程问题。", "可能是调用方错误，而非库实现缺陷。",
    "如果接口约定沿维度0拼接，则属于调用方错误。", "不能断言不是库实现缺陷。",
    "无法确定库实现是否存在问题。",
])
def test_mechanism_or_explicitly_qualified_hypothesis_is_allowed(statement):
    validate_attribution_consistency(report(statement))


def test_absence_of_explicit_unknown_is_not_semantic_proof_or_automatic_rejection():
    validate_attribution_consistency(report("库实现没有问题。", "未执行验证命令。"))


def test_current_state_and_hypotheses_are_also_checked():
    data = report("输入与默认参数不匹配。")
    data["hypotheses"][0]["explanation"] = "这是调用方错误。"
    with pytest.raises(ValueError, match="Attribution inconsistency"):
        validate_attribution_consistency(data)


def test_two_sentences_do_not_share_an_unrelated_qualifier():
    data = report("可能还有其他因素。库实现没有问题。")
    with pytest.raises(ValueError, match="Attribution inconsistency"):
        validate_attribution_consistency(data)


def test_contradictory_report_uses_existing_single_report_correction_budget(tmp_path):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    good = run_diagnosis(log, root, DemoLLM(root, current, previous), allowed_path_roots=[workspace])
    bad = copy.deepcopy(good["diagnosis"])
    bad["summary"] = "库实现没有问题。"
    bad["limitations"] = ["无法判断责任归属，不能断言调用方或库实现哪一方有错。"]
    script = [LLMResponse(tool_calls=[ToolCall(str(index), event["tool"], event["arguments"])])
              for index, event in enumerate(good["tool_events"])]
    script += [LLMResponse(content=json.dumps(bad)), LLMResponse(content=json.dumps(good["diagnosis"]))]
    result = run_diagnosis(log, root, ScriptedLLM(script), allowed_path_roots=[workspace])
    assert result["status"] == "completed" and result["metrics"]["report_repairs"] == 1
    assert result["report_attempts"][0]["validation_error"].startswith("Attribution inconsistency")
    assert len(result["tool_events"]) == len(good["tool_events"])
