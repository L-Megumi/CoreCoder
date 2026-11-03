"""Compact presentation of the SAME strict diagnosis contract, not a new judge."""

import hashlib
import json

COMPACT_REPORT_V1 = (
    "[Compact diagnosis output policy v1]\n"
    "Keep the original six JSON keys and all validation rules. Output one small JSON object, no fences. "
    "Use at most two concise hypotheses, two actions and two limitations; each text should be a short sentence. "
    "Do not repeat source code, telemetry, full traces, or long paths: immutable originals are retained by "
    "the controller as EvidenceRef/ArtifactRef. Cite actual LOCAL evidence/observation IDs from the live registry. "
    "Upstream references are investigation leads, not local observation IDs. Read locally before asserting current facts. "
    "Distinguish historical failure, current observation and unverified hypothesis. "
    "Do not omit uncertainty or any independently observed root-cause mechanism to shorten the report. "
    "When tools are withheld for finalization, use only existing observations; never invent missing evidence."
)
COMPACT_PROMPT_SHA256 = hashlib.sha256(COMPACT_REPORT_V1.encode()).hexdigest()

# A request-local phase instruction. Keep compact-v1 byte-for-byte unchanged:
# the controller must explicitly select this separate policy version.
EXPLICIT_REPORT_VERSION = "explicit-report-v1"
EXPLICIT_REPORT_V1 = (
    "[ResearchCoder final report phase v1]\n"
    "调查阶段已经结束，本次请求只提交最终诊断报告，工具不可用。"
    "仅依据已经获得且当前有效的证据和观察，按系统规定的完整 JSON 合同输出一个对象。"
    "保留 summary、historical_failure、current_state、hypotheses、recommended_actions、limitations 六个键；"
    "若系统另外启用了结构化预测字段，仍遵守该合同。"
    "引用只使用当前注册表中的合法 ID；证据不足必须在 limitations 和待验证假设中明确说明。"
    "不要继续阅读源码、提出工具请求、输出工具调用标记、Markdown 围栏或 JSON 之外的解释。"
    "不要把假设写成已验证根因，不要补造缺失事实。现在提交 JSON 报告。"
)
EXPLICIT_REPORT_PROMPT_SHA256 = hashlib.sha256(EXPLICIT_REPORT_V1.encode()).hexdigest()


# A separate opt-in version: keep the historical v1 bytes and hashes unchanged.
EVIDENCE_READY_REPORT_VERSION = "evidence-ready-v2"
EVIDENCE_READY_INVESTIGATION_V2 = (
    "[ResearchCoder evidence-ready investigation v2]\n"
    "搜索命中是调查线索，不等于已经读取文件。提交最终报告前，"
    "使用现有工具读取与问题相关的当前源码或核实相关路径；"
    "沿已经观察到的调用关系检查必要的实现与公开接口约束，"
    "区分读到的事实、原因假设和仍未核实的问题。"
    "同一轮可以读取互不依赖的多个文件；依赖搜索结果的读取应在后续轮次进行。"
    "权限、工具和总预算保持不变；不要仅为凑齐步骤读取无关文件。"
)
EVIDENCE_READY_INVESTIGATION_SHA256 = hashlib.sha256(EVIDENCE_READY_INVESTIGATION_V2.encode()).hexdigest()
EVIDENCE_READY_REPORT_V2 = (
    "[ResearchCoder final report phase v2: evidence-ready]\n"
    "当前证据满足报告所需的最低观测条件，本次请求只提交最终诊断报告，工具不可用。"
    "仅依据实际获得且当前有效的证据和观察，输出系统规定的完整 JSON 对象。"
    "保留 summary、historical_failure、current_state、hypotheses、recommended_actions、limitations 六个键；"
    "如启用了其他结构化预测字段，继续遵守该合同。"
    "current_state 至少引用一个实际可用的当前文件或路径观察；搜索候选不是文件读取。"
    "引用使用当前注册表的合法 ID；最低观测条件不代表根因已经证实，"
    "未核实的部分写为假设及局限，不补造事实。不要请求工具或输出 JSON 外的解释。"
)
EVIDENCE_READY_REPORT_PROMPT_SHA256 = hashlib.sha256(EVIDENCE_READY_REPORT_V2.encode()).hexdigest()


def output_failure(response, finish_reason):
    if finish_reason == "length":
        return "response_length_exceeded"
    try:
        json.loads(response)
    except (ValueError, TypeError):
        return "invalid_json"
    return "invalid_report"


# Separate opt-in output identity; v2 instructions and strict validator stay intact.
EVIDENCE_READY_REPAIR_VERSION = "evidence-ready-v3"
EVIDENCE_READY_REPAIR_REPORT = EVIDENCE_READY_REPORT_V2.replace(
    "phase v2: evidence-ready", "phase v3: evidence-ready with bounded correction") + (
    " current_state.observation_ids 只能从最新 Trusted live citation scope 的 current_observation_ids 中选择。"
    "observation_ids 总表包含历史记录，不代表均能证明当前状态。不要引用已被替代的路径观察，"
    "不要猜测或自动映射 ID。若收到报告校验错误，仅在相同剩余预算内修正该报告；不得新增事实。"
)
EVIDENCE_READY_REPAIR_REPORT_SHA256 = hashlib.sha256(EVIDENCE_READY_REPAIR_REPORT.encode()).hexdigest()
