"""Deterministic routing of human requests; source/log text never selects a mode."""

import re
from dataclasses import dataclass

_REPAIR = re.compile(
    r"^(?:(?:请你?|帮我|麻烦|现在|直接|开始|继续|根据(?:这份|上面的)?(?:诊断|报告)|按(?:照)?(?:这份|上面的)?(?:诊断|报告))[\s，,：:]*)*"
    r"(?:修复|修正|修改|改好|改掉|解决)|^(?:please\s+)?(?:fix|repair|patch)\b", re.IGNORECASE,
)
_READONLY_STAGE = re.compile(r"(?:只|仅)(?:分析|解释|读取|查看|诊断)|只读(?!写)|read.only", re.IGNORECASE)
_NEGATED_EDIT = re.compile(
    r"(?:不要|不允许|禁止|先别|别|不必|无需|不能|不准|不)\s*(?:自动)?(?:修改|修复|写入|改动|改)|"
    r"(?:do not|don't|never)\s+(?:edit|modify|write|fix|repair)\b", re.IGNORECASE,
)
_SCOPED_TARGET = re.compile(
    r"(?:输入|测试|校验|验证|数据(?:集)?|权重|入口|配置|参数)|"
    r"(?:(?:the|any|existing|original)\s+)*(?:tests?|checks?|inputs?|data(?:sets?)?|weights?|configuration)\b|"
    r"[\w./\\-]+\.(?:py|yaml|yml|json|toml)\b", re.IGNORECASE,
)
_DIAGNOSE = re.compile(
    r"诊断|排查|报错|错误原因|异常原因|故障|traceback|loss\s*[:=]\s*(?:nan|inf)|"
    r"(?:分析|检查|查看).{0,12}(?:日志|错误|异常)|diagnos|debug|"
    r"(?:analy[sz]e|inspect|investigate).{0,24}(?:log|error|failure)", re.IGNORECASE,
)
_DEVELOP = re.compile(r"^(?:请你?|帮我)?\s*(?:实现|编写|新增|添加|开发|写一个|写个|implement\b|create\b|add\b)", re.IGNORECASE)
_CODE_REVIEW = re.compile(r"(?:解释|讲解|阅读|查看|分析).{0,12}(?:代码|函数|实现)|(?:explain|read|review).{0,24}(?:code|function|implementation)", re.IGNORECASE)
_REVIEW = re.compile(r"解释|讲解|总结|为什么|怎么|如何|方案|建议|explain|review|summari[sz]e|\bhow\b|\bwhy\b", re.IGNORECASE)
_REPAIR_PLAN = re.compile(r"^(?:请你?|帮我)?\s*(?:修复|修改)(?:方案|建议|思路|方法|之前|前)")
_CONTINUE = re.compile(r"^(?:那|好的|好|ok|okay)?[\s，,]*(?:继续|继续吧|接着做|continue|go on)[吧呀。.!！?？\s]*$", re.IGNORECASE)


@dataclass(frozen=True)
class Route:
    mode: str
    reason: str
    missing: tuple = ()


def is_readonly_request(task):
    if _READONLY_STAGE.search(task):
        return True
    # Preserving a named input/test/file is a scope constraint, not a ban on all repair.
    return any(not _SCOPED_TARGET.match(task[match.end():].lstrip()) for match in _NEGATED_EDIT.finditer(task))


def select_route(task, *, mode="auto", has_log=False, has_report=False):
    if not isinstance(task, str) or not task.strip() or len(task) > 6000 or "\x00" in task:
        raise ValueError("task must be nonempty text of at most 6000 characters without nulls")
    if mode not in ("auto", "code", "diagnose", "repair", "review"):
        raise ValueError("mode must be auto, code, diagnose, repair or review")
    task = task.strip()
    if _CONTINUE.fullmatch(task) and mode == "auto":
        return Route("needs_input", "请明确下一项任务，或使用 /code、/diagnose、/repair 指定本轮模式。", ("task",))
    readonly = is_readonly_request(task)
    repair = bool(_REPAIR.search(task)) and not (readonly or _REPAIR_PLAN.search(task))
    if mode == "repair" and readonly:
        return Route("needs_input", "修复模式与本轮只读要求冲突；请使用 diagnose 或 code 表达要做的检查。", ("consistent_task",))
    if mode == "auto":
        if repair and has_report:
            mode = "repair"
        elif has_report and readonly:
            mode = "review"
        elif _DEVELOP.search(task) or _CODE_REVIEW.search(task) and not _DIAGNOSE.search(task):
            mode = "code"
        elif has_report and _REVIEW.search(task):
            mode = "review"
        elif has_log or _DIAGNOSE.search(task):
            mode = "diagnose"
        elif repair and has_report is False:
            mode = "code"  # Ordinary coding fixes do not require an experiment diagnosis.
        else:
            mode = "code"
    if mode == "diagnose" and not has_log:
        return Route("needs_input", "诊断需要日志路径；使用 --log 或 /log 绑定日志。", ("log",))
    if mode == "repair" and not has_report:
        return Route("needs_input", "诊断修复需要有效报告；先诊断，或使用 --report、/report 绑定报告。", ("report",))
    if mode == "review" and not has_report:
        return Route("needs_input", "报告解读需要有效报告；先诊断或绑定报告。", ("report",))
    reasons = {"code": "通用编程任务，使用原编程工具。", "diagnose": "日志调查任务，使用只读证据工具。",
               "repair": "用户明确进入修复；执行前重新校验报告及源码版本。",
               "review": "解读当前诊断报告，不开放执行工具。"}
    return Route(mode, reasons[mode])
