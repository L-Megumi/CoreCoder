# M7: active evidence and progress-aware execution

This document freezes scope and acceptance criteria. Actual execution results,
including any failed runs, belong to the separate M7 delivery report and records.

## Scope and baseline

The baseline is the M6 source manifest with 466 files and canonical SHA256
`4e2cb022d2cfc0a6b110866feb819b412d296653f1f840ca9a8446a611e86ef1`.
The baseline has 1,815 passing tests. This is a historical baseline, not a claim
that the new code has passed those tests. Development uses one isolated copy and
one verified recovery snapshot; historical experiments are immutable.

This batch delivers three independently selectable mechanisms:

1. A bounded active evidence working set. Recently used, version-valid source
   windows remain available to subsequent decisions rather than being immediately
   offloaded again. Existing provenance, access and input-budget checks apply.
2. Correct closure ordering. Safe local investigation stops can reach the explicit
   completion phase when the existing budget permits. A missing/incorrect patch
   or failed public verification remains incomplete. Unknown usage and unsafe
   side effects never gain a retry through this path.
3. Progress-aware consultation. New observations, candidate changes and public
   verification outcomes inform progress. Repeated reads/restores alone do not
   count as continued progress. A bounded stall signal uses the existing expert
   consultation path and the same JobBudget.

No new Scheduler, Agent Loop, database or Agent role is needed. Finite policy
search is a subsequent enhancement, not a dependency of this delivery.

## Evidence and acceptance

The M6 SAM trajectory demonstrates repeated offloading and restoration of a
58-line source file. Its expert obtained the complete source but its final input
omitted the middle of the file. The M6 packaging trajectory separately shows
that investigation-limit handling can precede the explicit closure check.

Acceptance focuses on actual provider inputs and workflow outcomes:

- Old policy serialization and default behavior remain compatible.
- Retained source is current, authorized and within the same context budget.
- Tool-call/result pairing survives compression; forged, stale and foreign
  references remain rejected.
- A recovered active window is not immediately replaced by an offload placeholder
  solely because it moved beyond the latest message.
- Closure is reachable before eligible local stops; false completion remains
  rejected and real public verification is still required.
- A repeated restore loop produces a bounded, traceable stall; changed evidence
  and valid source edits can reset progress. Consultation preserves the original
  failure and consumes the original job's remaining budget.
- The saved real trajectory supports deterministic before/after context replay.
  Such replay demonstrates mechanism behavior, not real-model repair accuracy.

## Efficient validation sequence

Implement related source and tests together. During development use syntax and
import checks only where useful. Run the new and directly affected tests once the
batch is integrated, investigate actual failures, then perform final regression
and packaging checks on a frozen candidate. Avoid repeated baseline audits,
unrelated test runs and paid-model experiments without a specific hypothesis.

New features remain opt-in pending evaluation. This batch does not inherit M6
repair scores as its own or claim that context compression lowers total model
cost merely because a single request is shorter.
