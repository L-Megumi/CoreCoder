"""Role priorities and demand-aware estimated input partitions; no separate ledger."""

from dataclasses import dataclass
import re
from corecoder.context import estimate_tokens


WEIGHTS = {
    "log_analysis": (6, 2, 2),
    "code_analysis": (2, 6, 2),
    "diagnosis": (5, 3, 2),
    "repair": (3, 4, 3),
    "verification": (4, 2, 4),
}


@dataclass(frozen=True)
class ContextAllocation:
    instructions: int
    evidence: int
    source: int
    history: int
    output: int
    tool_schema: int
    input_limit: int


class RoleContextBuilder:
    def __init__(self, role, objective=""):
        if role not in WEIGHTS:
            raise ValueError("Unknown domain role")
        self.role = role
        self.focus_terms = set(re.findall(r"[\w.]{4,}", objective.lower()))

    def allocate(self, messages, tools, limits, *, remaining_tokens=None):
        import json

        schema = len(json.dumps(tools or [], ensure_ascii=False)) // 3
        instruction = estimate_tokens(
            [
                m
                for m in messages
                if m.get("role") == "system"
                or (m.get("role") == "user" and not m.get("_context_data"))
                or m.get("_context_recovery")
            ]
        )
        total = limits.max_context_tokens
        if remaining_tokens is not None:
            total = min(total, remaining_tokens)
        input_limit = total - limits.max_output_tokens - schema
        available = max(0, input_limit - instruction)
        demand = {"evidence": 0, "source": 0, "history": 0}
        for m in messages:
            if m.get("_context_recovery"):
                continue  # Required delivery data, not elevated instructions.
            if m.get("role") == "tool" or m.get("_context_data"):
                text = str(m.get("content", ""))
                category = (
                    "source"
                    if m.get("_context_source") or any(
                        s in text
                        for s in (
                            '"observations"',
                            '"snippet"',
                            "file_sha256",
                            ".detach(",
                        )
                    )
                    else "evidence"
                )
                demand[category] += estimate_tokens([m])
            elif m.get("role") == "assistant":
                demand["history"] += estimate_tokens([m])
        weights = dict(zip(demand, WEIGHTS[self.role]))
        allocation = {k: 0 for k in demand}
        # Water filling: unused partitions return to the active demand pool.
        while available and any(allocation[k] < demand[k] for k in demand):
            active = [k for k in demand if allocation[k] < demand[k]]
            weight = sum(weights[k] for k in active)
            used = 0
            for k in active:
                take = min(
                    demand[k] - allocation[k],
                    max(1, available * weights[k] // weight),
                    available - used,
                )
                allocation[k] += take
                used += take
            if not used:
                break
            available -= used
        return ContextAllocation(
            instruction,
            **allocation,
            output=limits.max_output_tokens,
            tool_schema=schema,
            input_limit=input_limit,
        )

    def rank(self, card):
        import json

        kind = card.get("kind")
        if card.get("coding_source"):
            return (0 if card.get("restored_window") and card.get("current_eligible") else
                    1 if card.get("current_eligible") else 3,
                    -sum(term in json.dumps(card, ensure_ascii=False).lower() for term in self.focus_terms),
                    -card.get("read_sequence", 0))
        priority = 3
        if kind == "numeric":
            priority = 2
        if self.role in ("code_analysis", "repair") and kind == "source":
            priority = 1
        if card.get("cause_id") or card.get("context_id") or card.get("exception_type"):
            priority = 1
        if kind == "failed_attempt":
            priority = 0
        text = json.dumps(card, ensure_ascii=False).lower()
        return priority, -sum(term in text for term in self.focus_terms)
