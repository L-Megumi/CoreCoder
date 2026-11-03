"""Narrow consistency checks for explicitly declared attribution uncertainty.

This is a lexical Chinese-report guard, not a semantic truth or blame verifier.
"""

import re

_UNKNOWN = re.compile(
    r"(?:不能|无法|尚不能|尚无法)(?:断言|判定|判断|确定|确认)[^。；;\n]{0,100}"
    r"(?:哪一方|责任|归属|(?:调用方|调用者|实现|库)[^。；;\n]{0,25}(?:有错|错误|有缺陷))"
)
_CATEGORICAL = re.compile(
    r"(?:并非|而非|不是|并不是|不属于)[^。；;\n]{0,12}(?:库|实现|模块)[^。；;\n]{0,15}(?:缺陷|错误|问题)"
    r"|(?:库|实现|模块)[^。；;\n]{0,12}(?:没有|不存在|无)[^。；;\n]{0,6}(?:缺陷|错误|问题)"
    r"|(?:是|属于|根因是)[^。；;\n]{0,12}(?:调用方|调用者)[^。；;\n]{0,12}(?:错误|有错|缺陷)"
)
_QUALIFIED = re.compile(r"可能|假如|如果|若|无法|不能|待核实|不确定|尚未")


def validate_attribution_consistency(report):
    if not any(_UNKNOWN.search(text) for text in report["limitations"]):
        return
    statements = [report["summary"], report["current_state"]["assessment"]]
    statements.extend(hypothesis["explanation"] for hypothesis in report["hypotheses"])
    for statement in statements:
        for sentence in re.split(r"[。；;\n]", statement):
            if _CATEGORICAL.search(sentence) and not _QUALIFIED.search(sentence):
                raise ValueError("Attribution inconsistency: limitations declare responsibility unknown, "
                                 "but summary/current state/hypothesis categorically assigns or excludes implementation blame; "
                                 "retain the mismatch mechanism and make attribution conditional")
