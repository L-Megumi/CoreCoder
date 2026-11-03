"""Atomic, conservative Job reservations over existing per-Worker ledgers."""

from dataclasses import dataclass
import threading
from researchcoder.contracts import Contract
from researchcoder.runtime.budget import BudgetExceeded


@dataclass(frozen=True, kw_only=True)
class JobLimits(Contract):
    max_model_calls: int = 200
    max_tool_calls: int = 300
    max_total_tokens: int = 10_000_000
    max_runtime_seconds: float = 300.0
    schema_version: int = 1

    def validate(self):
        if (
            any(v < 1 for v in (self.max_model_calls, self.max_tool_calls, self.max_total_tokens))
            or not 0.1 <= self.max_runtime_seconds <= 3600
        ):
            raise ValueError("Invalid Job limits")


class JobBudget:
    def __init__(self, limits=JobLimits()):
        self.limits = limits
        self._lock = threading.RLock()
        self.pending, self.settlements = {}, {}
        self.charged = dict(model_calls=0, tool_calls=0, total_tokens=0)

    def reserve(self, identity, limits):
        amount = dict(model_calls=limits.max_model_calls, tool_calls=limits.max_tool_calls, total_tokens=limits.max_total_tokens)
        with self._lock:
            if identity in self.pending or identity in self.settlements:
                raise ValueError("Duplicate Job reservation")
            for key, value in amount.items():
                if self.charged[key] + sum(a[key] for a in self.pending.values()) + value > getattr(self.limits, "max_" + key):
                    raise BudgetExceeded("Job allocation unavailable")
            self.pending[identity] = amount

    def settle(self, identity, record):
        with self._lock:
            if identity not in self.pending:
                raise ValueError("Unknown or repeated Job settlement")
            reserved = self.pending.pop(identity)
            b = (record or {}).get("budget", {})
            complete = all(
                type(b.get(k)) is int and b[k] >= 0 for k in ("model_calls", "tool_calls", "charged_tokens", "unknown_requests")
            ) and not b.get("pending", {})
            actual = (
                dict(model_calls=b["model_calls"], tool_calls=b["tool_calls"], total_tokens=b["charged_tokens"])
                if complete
                else reserved.copy()
            )
            if not complete:
                # A broken response cannot erase already reported consumption.
                # Retain the original allocation and any known overrun, while
                # marking the whole settlement unknown for recovery/admission.
                for key, observed in (("model_calls", "model_calls"), ("tool_calls", "tool_calls"),
                                      ("total_tokens", "charged_tokens")):
                    if type(b.get(observed)) is int and b[observed] >= 0:
                        actual[key] = max(actual[key], b[observed])
            for key in self.charged:
                self.charged[key] += actual[key]
            self.settlements[identity] = dict(
                reserved=reserved,
                charged=actual,
                accounting_complete=complete,
                usage_unknown=(not complete or b["unknown_requests"] > 0),
                known_prompt_tokens=b.get("known_prompt_tokens") if complete else None,
                known_completion_tokens=b.get("known_completion_tokens") if complete else None,
                estimated_input_tokens=b.get("estimated_input_tokens") if complete else None,
            )
            return self.settlements[identity]

    def snapshot(self):
        with self._lock:
            return {
                "limits": self.limits.to_dict(),
                "charged": self.charged.copy(),
                "pending": {k: v.copy() for k, v in self.pending.items()},
                "settlements": {k: v.copy() for k, v in self.settlements.items()},
            }
