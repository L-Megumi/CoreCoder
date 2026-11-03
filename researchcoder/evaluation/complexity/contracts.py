"""Frozen program-derived dimensions, never labels fitted to Agent outcomes."""

from dataclasses import dataclass
from typing import Literal
from researchcoder.contracts import Contract, identifier, digest, relative, text


@dataclass(frozen=True, kw_only=True)
class EvidenceSite(Contract):
    path: str
    start_line: int
    end_line: int
    kind: Literal["source", "training", "validation"]
    sha256: str
    schema_version: int = 1

    def validate(self):
        relative(self.path)
        digest(self.sha256)
        if not 1 <= self.start_line <= self.end_line:
            raise ValueError("Invalid evidence span")


@dataclass(frozen=True, kw_only=True)
class RepairStage(Contract):
    stage_id: str
    patch_paths: tuple[str, ...]
    expected_f2p: bool
    expected_p2p: bool = True
    schema_version: int = 1

    def validate(self):
        identifier(self.stage_id)
        if not self.patch_paths or len(set(self.patch_paths)) != len(self.patch_paths):
            raise ValueError("Explicit cumulative reference patch required")
        for path in self.patch_paths:
            relative(path)


@dataclass(frozen=True, kw_only=True)
class ComplexityCase(Contract):
    case_id: str
    benchmark_subset: Literal["pytorch", "complexity_benchmark"]
    level: Literal[1, 2, 3]
    case_sha256: str
    source_files: tuple[str, ...]
    dependencies: tuple[tuple[str, str], ...]
    evidence_sites: tuple[EvidenceSite, ...]
    initial_hypotheses: tuple[str, ...]
    stages: tuple[RepairStage, ...]
    N_files: int
    N_dependencies: int
    EvidenceDispersion: int
    HypothesisAmbiguity: int
    RepairStages: int
    ContextLength: int
    ContextTokensEstimate: int
    FailureVisibility: Literal["filename_only", "indirect_telemetry", "delayed_validation"]
    stage_semantics: Literal["independent_repair_obligations_not_minimum_agent_rounds"] = "independent_repair_obligations_not_minimum_agent_rounds"
    label_basis: Literal["program_dependencies_and_independent_oracle_before_agent_trials"] = "program_dependencies_and_independent_oracle_before_agent_trials"
    schema_version: int = 1

    def validate(self):
        identifier(self.case_id)
        digest(self.case_sha256)
        if self.N_files != len(set(self.source_files)) or not self.source_files:
            raise ValueError("Source file count mismatch")
        for path in self.source_files:
            relative(path)
        if self.N_dependencies != len(set(self.dependencies)):
            raise ValueError("Dependency count mismatch")
        for source, target in self.dependencies:
            if source not in self.source_files or target not in self.source_files or source == target:
                raise ValueError("Dependency is not a local module edge")
        if self.EvidenceDispersion != len({s.path for s in self.evidence_sites}):
            raise ValueError("Evidence dispersion count mismatch")
        if self.HypothesisAmbiguity != len(self.initial_hypotheses) or not self.initial_hypotheses:
            raise ValueError("Explicit pre-trial ambiguity rubric required")
        for hypothesis in self.initial_hypotheses:
            text(hypothesis, 1000)
        if self.RepairStages != len(self.stages) or not self.stages or not self.stages[-1].expected_f2p:
            raise ValueError("Repair sequence must end in complete verification")
        if self.level == 3 and (self.RepairStages < 2 or self.stages[0].expected_f2p):
            raise ValueError("Multistage case needs an independently failing partial repair")
        if self.ContextLength < 1 or self.ContextTokensEstimate != (self.ContextLength + 2) // 3:
            raise ValueError("Context estimate must be explicit, not provider usage")


@dataclass(frozen=True, kw_only=True)
class ComplexityManifest(Contract):
    benchmark_id: str
    version: str
    benchmark_manifest_sha256: str
    cases: tuple[ComplexityCase, ...]
    schema_version: int = 1

    def validate(self):
        identifier(self.benchmark_id)
        identifier(self.version)
        digest(self.benchmark_manifest_sha256)
        if len({(c.benchmark_subset, c.case_id) for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate complexity identity")
        if any(sum(c.level == level for c in self.cases) < (1 if level == 1 else 3) for level in (1, 2, 3)):
            raise ValueError("Keep Level 1 anchors and three distinct cases per higher level")
