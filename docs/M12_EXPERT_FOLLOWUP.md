# Focused expert assignments and follow-up

Two explicit opt-ins preserve the old paths: `ExpertPolicy.assignment_version=focused-v1`
and `ContextPolicy.expert_followup_version=followup-v1` (requires `state-v1`).

The controller derives an investigation question, filename-based entry hints and
completion criteria from public task/inventory only. The assignment is saved as
an immutable same-Job Artifact and included in the existing expert objective.
No new planner, role or execution loop is introduced. Expert reports continue to
use the original strict diagnosis validator and source observations.

The context engine registers actions from complete SHA-validated expert reports.
Each action is scoped to its original Job, report and producer provenance.
`record_expert_decisions` accepts a bounded batch of adopted/rejected/unresolved
writer judgements. Adopted/rejected entries must cite actual current local source
documents. Upstream observations cannot masquerade as local source reads.
The whole batch is validated before publication and charged to the normal tool
budget. This metadata tool has no filesystem, shell or verification capability.

The protected repair execution receipt retains pending items and decisions.
Source changes mark their old source links historical. Every decision is explicitly
`verified=false`: source linkage proves the read happened, not the correctness of
the model's reasoning. Missing decisions remain visible; they do not manufacture
failure or success and never bypass the independent verifier.

This enables audit of whether expert work reached the writer, which proposals were
handled and where duplicate investigation remained. It does not prove expert
causal benefit or better repair accuracy. Those require further frozen experiments.
