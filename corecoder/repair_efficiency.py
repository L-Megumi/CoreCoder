"""Bounded source handoff, deterministic repair history, and reserved case budgets."""

from __future__ import annotations

import json

from .context import estimate_tokens
from .source_visibility import visible_end_line


def source_context(observations, current, diagnosis, *, max_chars=10000, max_excerpts=6):
    """Project complete, current observed lines; never infer text from requested ranges."""
    cited = {identifier for item in diagnosis.get("hypotheses", []) for identifier in item["observation_ids"]}
    candidates = [item for item in observations.values() if item["id"] in current
                  and item["data"].get("file_sha256") and item["data"].get("resolved_path")
                  and visible_end_line(item["data"].get("snippet") or item["data"]) > 0]
    candidates.sort(key=lambda item: (item["id"] not in cited, item["kind"] not in ("config_file", "file_excerpt")))
    excerpts, seen = [], set()
    for item in candidates:
        data = item["data"]
        span = data.get("snippet") or data
        start = span["start_line"]
        lines = span["content"].split("\n")[:visible_end_line(span) - start + 1]
        anchor = data.get("current_line_no", data.get("hit_line", start))
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            anchor = start
        index = min(max(0, anchor - start), len(lines) - 1)
        left, right = index, index + 1
        # Whole lines around the observed failure, rather than a misleading cut through a line.
        if len(lines[index]) > 1400:
            continue
        while left > 0 or right < len(lines):
            expanded = False
            for direction in (-1, 1):
                lo, hi = (left - 1, right) if direction == -1 else (left, right + 1)
                if lo >= 0 and hi <= len(lines) and len("\n".join(lines[lo:hi])) <= 1400:
                    left, right, expanded = lo, hi, True
            if not expanded:
                break
        key = (data["resolved_path"], start + left, start + right - 1)
        if key in seen:
            continue
        content = "\n".join(lines[left:right])
        excerpt = {"observation_id": item["id"], "path": data["resolved_path"], "file_sha256": data["file_sha256"],
                   "start_line": start + left, "end_line": start + right - 1, "content": content,
                   "omitted_chars": len(span["content"]) - len(content),
                   "source_truncated": bool(span.get("truncated") or span.get("char_truncated"))}
        if len(excerpts) < max_excerpts and len(json.dumps(excerpts + [excerpt], ensure_ascii=False)) <= max_chars:
            excerpts.append(excerpt)
            seen.add(key)
    return {"excerpts": excerpts, "omitted_observations": len(candidates) - len(excerpts),
            "scope": "Read-time source snapshots revalidated at handoff; modifications can invalidate them"}


class RepairContextManager:
    """Keep the explicit task and whole recent tool groups, without model summarization."""

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.reserved_tokens = 0
        self.events = []

    def maybe_compress(self, messages, llm=None):
        available = self.max_tokens - self.reserved_tokens
        if available < 300:
            raise ValueError("Repair context budget cannot fit the task, source memory and tool schemas")
        before = estimate_tokens(messages)
        if before <= int(available * 0.8):
            return False
        if not messages:
            return False
        first = messages[0]
        if estimate_tokens([first]) > available:
            raise ValueError("Repair task and handoff exceed the estimated context budget")
        if len(messages) == 1:
            return False
        groups = []
        for message in messages[1:]:
            if message.get("role") == "tool" and groups:
                groups[-1].append(message)
            else:
                groups.append([message])
        notice = {"role": "user", "content": "[Repair history shortened: complete older tool groups omitted; "
                  "full results remain in the audit. Re-read a needed range if its contents are absent.]"}
        tail = []
        for group in reversed(groups):
            if estimate_tokens([first, notice] + group + tail) > available:
                break
            tail = group + tail
        candidate = [first, notice] + tail
        if estimate_tokens(candidate) > available:
            raise ValueError("Repair task and omission notice exceed context budget")
        messages[:] = candidate
        self.events.append({"tokens_before_estimate": before, "tokens_after_estimate": estimate_tokens(messages),
                            "dropped_messages": max(0, sum(map(len, groups)) - len(tail))})
        return True


def plan_case_requests(case_ids, previous_counts, max_total, max_case):
    """Reserve disjoint allowances before execution; unused quotas are not silently borrowed."""
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case IDs must be unique")
    for value in (max_total, max_case, *previous_counts.values()):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Request budgets and previous counts must be nonnegative integers")
    remaining = max_total - sum(previous_counts.values())
    if remaining < 0:
        raise ValueError("Previous requests already exceed the total budget")
    allocations = dict.fromkeys(sorted(case_ids), 0)
    capacity = {cid: max(0, max_case - previous_counts.get(cid, 0)) for cid in allocations}
    # Water filling reserves a fair share regardless of subsequent case execution order.
    while remaining:
        active = [cid for cid in allocations if allocations[cid] < capacity[cid]]
        if not active:
            break
        share = max(1, remaining // len(active))
        for cid in active:
            count = min(share, capacity[cid] - allocations[cid], remaining)
            allocations[cid] += count
            remaining -= count
    return allocations
