# M7B: bounded expert handoff after explicit incomplete closure

This document defines the narrow implementation and acceptance scope. Test
results belong in the separate delivery records; this document does not claim
that either offline or real-model acceptance has already passed.

## Problem and scope

An M7 writer can reach a Controller-selected output-only phase, return a valid
explicit `incomplete` report, and finish with an actually failed public check.
The earlier handoff path accepts accounted hard local stops, but this soft
closure is not a budget exception. It therefore leaves an unused expert path
despite sufficient remaining Job allowance.

M7B adds one explicit admission policy for this case. It reuses the existing
writer, Worker, protected verifier, handoff inspection, expert consultation,
Evidence Exchange and JobBudget. It adds no Agent role, scheduler, token loop,
Oracle change or automatic replay of writes.

## Opt-in configuration and identity

```python
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectPolicy

project_policy = ProjectPolicy(
    completion_version="explicit-v1",
    progress_version="progress-v1",
)
handoff_policy = HandoffPolicy(
    mode="local-stop-v3",
    closure_version="reserve-incomplete-v1",
)
expert_policy = ExpertPolicy(mode="public-failure-v1")
```

Supply these policies to the existing `ProjectEvaluationRunner.run` entry point
with the existing source, public task, permitted checks and model configuration.
Existing deployment-specific limits and expert/context settings should be
retained. This example is not a grant of additional budget or permissions.

The default closure policy is `off`. Its field is omitted from historical
serialization, including nested Worker and handoff records. The enabled policy
has a distinct configuration identity. `CLOSURE_RULES_SHA256` identifies its
routing rules separately from the existing rule versions. Historical trials,
their policy identities and recorded outcomes are not rewritten.

## Admission and preserved semantics

Only Controller-triggered `round_reserve` and `model_call_reserve` closures can
be considered. A normal model refusal, invalid or truncated JSON, another
reserve reason, or an arbitrary `unknown` result does not qualify.

Admission requires a reproducible closure receipt tied to the original Job,
Task and Attempt, the explicit final report, trusted runtime counters and
execution Trace. The final declaration must be valid `incomplete`; the writer
must have performed no writes and the source manifest must be unchanged.
The public verifier must actually have run and failed against that same
workspace. Artifact digests, provenance, execution records and access checks
remain mandatory, including revalidation before expert dispatch.

The original result remains `incomplete`, `workflow_completed=False`, with
`failure_kind="unknown"` and the report reason `model_declared_incomplete`.
That task failure classification is distinct from unknown model Usage.
Unknown or pending Usage prevents continuation. M7B does not change the original
Ledger into a fictitious budget-exhausted state or erase the first Attempt.

The original JobBudget pays for the writer, expert consultation and any later
writer Attempt. The existing expert and writer reservations, remaining retry
limit and one-consultation-episode cap continue to apply. An affirmative
handoff receipt is permission to assess existing routing, not an unconditional
instruction to launch an expert or a claim of repair success.

## Planned focused acceptance

- Run real isolated Workers with ScriptedLLM through valid incomplete closure,
  expert investigation, evidence delivery, subsequent repair and independent
  F2P/P2P checks. Verify that the first incomplete Attempt remains archived and
  all calls are charged to the same Job.
- Check both admitted reserve reasons and reject unreviewed reasons, forged
  counters, altered identities, changed source, invalid artifacts, missing
  Trace linkage, malformed reports and unknown accounting.
- Reject writing attempts, unavailable or nonfailed public verification,
  exhausted Job resources, unavailable writer retry and repeated consultation.
- Confirm that disabling the new policy retains the previous result and
  serialized identity. Check existing local-stop/progress handoff and explicit
  completion paths that the modification directly touches.
- Verify that archived outcomes can still be recomputed without model calls
  and that experts use the existing evidence and permission boundaries.

These are targeted mechanism tests, not a full repository regression. Fixed
ScriptedLLM actions can establish execution correctness but cannot establish
real-model diagnosis accuracy, repair-rate improvement or superiority over a
Single Agent. No paid-model experiment is required for this implementation gate.
