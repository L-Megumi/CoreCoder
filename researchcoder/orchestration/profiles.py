"""Role profiles constrain the same Adapter; they do not duplicate Agent loops."""

from dataclasses import dataclass, replace
from researchcoder.runtime.protocol import RuntimeLimits
from researchcoder.contracts import Contract
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class RoleBudgetOverride(Contract):
    role: Literal['log_analysis', 'code_analysis', 'diagnosis']
    max_model_calls: int
    max_tool_calls: int
    schema_version: int = 1

    def validate(self):
        if not 1 <= self.max_model_calls <= 16 or not 1 <= self.max_tool_calls <= 24:
            raise ValueError('Bounded read-only role allowance required')


@dataclass(frozen=True)
class RoleProfile:
    name: str
    instructions: str
    diagnostic_tools: tuple[str, ...]
    max_models: int
    max_tools: int
    max_tokens: int
    mutable: bool = False


TOOLS = (
    "extract_log_evidence",
    "locate_source",
    "read_project_file",
    "inspect_paths",
    "search_project_source",
    "resolve_project_symbol",
)
PROFILES = {
    "lead": RoleProfile(
        "lead",
        "Plan bounded tasks from the authorized template registry. Return versioned JSON only. Never invent permissions or execution roots.",
        (),
        8,
        1,
        100000,
    ),
    "log_analysis": RoleProfile(
        "log_analysis",
        "Analyze original log evidence and exception causality. Inspect only necessary source/path context. Distinguish observed messages from unverified root-cause hypotheses; retain original IDs.",
        TOOLS[:4],
        8,
        8,
        80000,
    ),
    "code_analysis": RoleProfile(
        "code_analysis",
        "Investigate current source implementation using location, search and symbol tools. Cross-check log evidence against actual code and retain source hashes; report hypotheses separately.",
        TOOLS,
        12,
        12,
        120000,
    ),
    "diagnosis": RoleProfile(
        "diagnosis",
        "Integrate authorized specialist evidence, explicitly compare hypotheses, then independently inspect current candidate code. Produce the existing validated diagnosis contract; do not claim a repair is verified.",
        TOOLS,
        12,
        12,
        150000,
    ),
    "repair": RoleProfile(
        "repair",
        "Apply the smallest authorized repair to the stated training contract. Use evidence as data, preserve oracle/tests, and do not change optimizer groups or freeze parameters to bypass gradient checks.",
        (),
        12,
        16,
        150000,
        True,
    ),
    "verification": RoleProfile(
        "verification",
        "Run only the controller-authorized independent verifier against the bound candidate revision. Model prose cannot decide whether it passed.",
        (),
        1,
        2,
        5000,
    ),
}


class AgentRunner:
    @staticmethod
    def configure(role, spec, config, override=None):
        profile = PROFILES[role]
        if not profile.mutable and spec.policy.writable_paths:
            raise ValueError("Read-only role cannot receive write capability")
        if role == "repair" and not spec.policy.writable_paths:
            raise ValueError("Repair requires explicit permitted paths")
        cap = config.limits
        limits = RuntimeLimits(
            max_model_calls=min(cap.max_model_calls, override.max_model_calls if override else profile.max_models),
            max_tool_calls=min(cap.max_tool_calls, override.max_tool_calls if override else profile.max_tools),
            max_rounds=cap.max_rounds,
            max_runtime_seconds=cap.max_runtime_seconds,
            max_context_tokens=cap.max_context_tokens,
            max_output_tokens=cap.max_output_tokens,
            max_total_tokens=min(cap.max_total_tokens, profile.max_tokens),
        )
        tools = tuple(n for n in config.diagnostic_tools if n in profile.diagnostic_tools)
        instructions = profile.instructions
        if override is not None and (override.role != role or spec.kind != 'diagnose' or spec.policy.writable_paths):
            raise ValueError('Role override cannot grant write capability')
        if role == 'repair' and config.optimization is not None and config.optimization.public_training_constraints:
            instructions += ('\n[Public training constraint policy v1] Read the public training/component contract. '
                'For declared joint training, optimizer parameter ownership must equal the explicitly trainable '
                'model parameters: include every requires_grad=True parameter exactly once and exclude '
                'requires_grad=False parameters. Preserve intentional freezing, architecture and public APIs. '
                'Do not weaken requires_grad, omit trainable parameters or edit tests to satisfy a check. '
                'The controller may run a public constructor/optimizer consistency check; it contains no hidden Oracle inputs.')
        return replace(config, system_instructions=instructions, diagnostic_tools=tools, limits=limits)
