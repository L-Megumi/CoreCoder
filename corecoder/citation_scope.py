"""Live citation scopes, derived from the investigation rather than model output.

An upstream record remains producer-scoped evidence. It is not a local file
observation, even when both records describe identical source bytes.
"""

import json
from .observation_state import current_observation_ids


UPSTREAM_SCOPE = "upstream_reference_not_local_citation"
RULE = (
    "报告中的 evidence_ids / observation_ids 仅允许引用本次调查工具产生的本地 ID。"
    "上游 EvidenceRef 的 origin_id、ref_id、exchange_id、document_id 均不是本地报告引用。"
    "上游记录用于调查线索；需要引用其事实时，先在当前任务调用授权工具独立读取，"
    "使用该工具返回的本地 ID。不得改写或猜测 ID，也不得把上游假设当成已验证事实。"
    "current_state 仅使用 current_observation_ids；历史本地观察不能证明当前源码状态。"
)


def citation_scope(state, identity=None):
    """No file access, import, provenance rewriting or mutation of raw records."""
    return {
        "schema_version": 1,
        "scope": "current_investigation_only",
        "identity": dict(identity or {}),
        "evidence_ids": sorted(item["id"] for item in state.evidence or []),
        "observation_ids": sorted(state.observations),
        "current_observation_ids": sorted(current_observation_ids(state)),
        "upstream_scope": UPSTREAM_SCOPE,
    }


def citation_message(state, identity=None):
    return (
        "[Trusted live citation scope v1]\n" + RULE + "\n"
        + json.dumps(citation_scope(state, identity), ensure_ascii=False, sort_keys=True)
        + "\n[End citation scope]\n"
    )
