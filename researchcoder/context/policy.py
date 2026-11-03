"""Context policy is trusted configuration, not model-proposed execution authority."""

from dataclasses import dataclass
from typing import Literal
from researchcoder.contracts import Contract


@dataclass(frozen=True, kw_only=True)
class ContextPolicy(Contract):
    strategy: Literal["original", "truncate", "summary", "evidence"] = "evidence"
    role: Literal[
        "log_analysis", "code_analysis", "diagnosis", "repair", "verification"
    ] = "diagnosis"
    allow_recovery: bool = False
    max_records: int = 32
    coding_source_mode: Literal["off", "observed-source-v1", "observed-source-v2"] = "off"
    working_set_version: Literal["legacy-v1", "active-v1"] = "legacy-v1"
    repair_state_version: Literal["off", "state-v1"] = "off"
    expert_followup_version: Literal["off", "followup-v1", "followup-v2", "followup-v3"] = "off"
    writer_guidance_version: Literal["legacy-v1", "focused-repair-v1"] = "legacy-v1"
    handoff_view_version: Literal["legacy-v1", "bounded-excerpts-v1"] = "legacy-v1"
    schema_version: int = 1

    def validate(self):
        if self.handoff_view_version != "legacy-v1" and (self.repair_state_version != "state-v1" or self.coding_source_mode != "observed-source-v2"):
            raise ValueError("Bounded handoff view requires trusted repair/source context")
        if self.writer_guidance_version != "legacy-v1" and self.repair_state_version != "state-v1":
            raise ValueError("Focused writer guidance requires trusted repair execution state")
        if self.expert_followup_version != "off" and self.repair_state_version != "state-v1":
            raise ValueError("Expert follow-up requires trusted repair execution state")
        if not 4 <= self.max_records <= 128:
            raise ValueError("Bounded evidence selection required")
        if self.coding_source_mode != "off" and (
            self.role != "repair" or self.strategy not in ("original", "evidence") or not self.allow_recovery
        ):
            raise ValueError("Observed coding source requires explicit repair evidence recovery")
        if self.working_set_version == "active-v1" and self.strategy not in ("original", "evidence"):
            raise ValueError("Active source retention requires original or evidence strategy")
        if self.repair_state_version == "state-v1" and (
            self.role != "repair" or self.strategy not in ("original", "evidence")
            or self.coding_source_mode != "observed-source-v2" or not self.allow_recovery
        ):
            raise ValueError("Repair state retention requires repair observed-source-v2 evidence recovery")

    def to_dict(self):
        value = super().to_dict()
        if self.handoff_view_version == "legacy-v1":
            value.pop("handoff_view_version")
        if self.writer_guidance_version == "legacy-v1":
            value.pop("writer_guidance_version")
        if self.coding_source_mode == "off":
            value.pop("coding_source_mode")
        if self.working_set_version == "legacy-v1":
            value.pop("working_set_version")
        if self.expert_followup_version == "off":
            value.pop("expert_followup_version")
        if self.repair_state_version == "off":
            value.pop("repair_state_version")
        return value
