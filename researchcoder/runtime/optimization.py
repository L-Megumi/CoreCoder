"""Versioned opt-in controls; no live objects, permissions or model settings."""

from dataclasses import dataclass
from researchcoder.contracts import Contract


@dataclass(frozen=True, kw_only=True)
class RuntimeOptimization(Contract):
    compact_reports: bool = False
    public_training_constraints: bool = False
    report_reserve_calls: int = 2
    version: str = "failure-aware-v1"
    coding_phase_version: str = "off"
    schema_version: int = 1

    def validate(self):
        if self.coding_phase_version not in ("off", "repair-cycle-v1"):
            raise ValueError("Unknown coding phase policy")
        if self.coding_phase_version != "off" and self.version != "explicit-report-v1":
            raise ValueError("Coding phases require explicit completion")
        if self.version not in ("failure-aware-v1", "explicit-report-v1", "evidence-ready-v2", "evidence-ready-v3") or not 1 <= self.report_reserve_calls <= 2:
            raise ValueError("Unsupported bounded runtime optimization")
        if self.version == "explicit-report-v1" and (not self.compact_reports or self.report_reserve_calls != 2):
            raise ValueError("Explicit completion requires two existing request slots, not additional credit")
        if self.version == "evidence-ready-v2" and (not self.compact_reports or self.report_reserve_calls != 1):
            raise ValueError("Evidence-ready completion requires one existing final-report slot")

        if self.version == "evidence-ready-v3" and (not self.compact_reports or self.report_reserve_calls != 2):
            raise ValueError("Evidence-ready v3 requires two existing report/correction slots")

    def to_dict(self):
        value = super().to_dict()
        if self.coding_phase_version == "off":
            value.pop("coding_phase_version")
        return value


STRATEGIES = ("fixed-v1", "budget-v1", "compact-v1", "constraints-v1", "combined-v1")


def runtime_optimization(strategy):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown harness strategy")
    if strategy == "fixed-v1":
        return None
    return RuntimeOptimization(compact_reports=strategy in ("compact-v1", "combined-v1"),
        public_training_constraints=strategy in ("constraints-v1", "combined-v1"))


def budget_remediation_enabled(strategy):
    return strategy in ("budget-v1", "combined-v1")
