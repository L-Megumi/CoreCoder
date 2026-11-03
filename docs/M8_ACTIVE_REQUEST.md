# M8: bounded voluntary expert requests

This opt-in extends the existing project Coding workflow. The initial writer
may pause after inspecting source and request an independent Code Analysis
investigation before reaching a local limit. It reuses the existing Worker,
public verifier, handoff inspection, JobBudget, expert Scheduler and Evidence
Exchange. It adds no role, Tool authority, token loop or second scheduler.

## Configuration and model-visible contract

Set `HandoffPolicy(mode="local-stop-v3", request_version="explicit-request-v1")`
with the existing `progress-v1` project policy and enabled expert policy. The
older `closure_version="reserve-incomplete-v1"` may remain enabled independently.
The new field defaults to `off` and is omitted from historical serialization.
The Controller exposes the option only to the first writer while a feedback
round, handoff and expert episode remain structurally available. Dispatch still
checks actual remaining time and settled budget; exposure is not a reservation.

The writer can end its ordinary assistant response with this exact object:

```json
{
  "action": "request_expert",
  "requested_role": "code_analysis",
  "question": "A specific unresolved question about the observed source",
  "evidence_paths": ["relative/source.py"]
}
```

This is a versioned response contract rather than a new executable tool. The
existing Agent Loop and model request accounting remain unchanged. The question
is bounded to 2,000 characters; one to four unique normalized candidate-relative
paths must have been successfully read by `read_file` during this attempt.
Context restoration alone requires a current file read before requesting.
`CODING_EXPERT_REQUEST_SHA256` identifies the prompt. The option is voluntary;
normal summaries and successful direct repairs retain their existing behavior.

## Admission and data semantics

The original writer remains `incomplete`, `workflow_completed=False`, with an
`unknown` task failure kind. That deliberate pause is distinct from unknown
provider Usage and does not counterfeit budget exhaustion. Invalid requests
retain their report failure and cannot authorize another Worker.

Admission requires all of the following:

- Original same-Job Task/Attempt identity, immutable report and record Artifacts.
- A valid request, actual read events, per-file hashes and unchanged manifest.
- No write attempts, including no-op edits; no permission rejection or unknown
  side effect. This initial version deliberately supports investigation only.
- Fully settled known provider Usage and matching paired model/tool Trace spans.
- An actually executed, complete protected public check with exit code 1 on the
  same source version. A model declaration cannot stand in for verification.
- The existing one-consultation cap, available writer retry, expert allocation
  and writer reserves from the original JobBudget.

`coding.expert_request` records the fingerprinted request proof before public
verification. `coding.request.archived` binds its archived receipt after the
check. The Controller reproduces both provenance and event order before routing
and again before expert dispatch. `REQUEST_RULES_SHA256` identifies the new
routing rules. Existing closure and local-stop routes preserve their identities.

The validated question and source-read bindings enter the existing expert task
as explicitly untrusted data. The question is never installed as a high-priority
instruction or a verified observation. Code Analysis independently reads the
immutable source snapshot and returns evidence through the existing Exchange.
The follow-up writer receives source-checked evidence and must re-read the
current candidate before editing. It cannot request another episode.

## Intended targeted acceptance

Tests should exercise a real Worker request, expert return, writer continuation,
and independent F2P/P2P checks; retain the first incomplete Attempt; and account
for every request in one Job. Negative controls cover malformed requests,
unobserved paths, writes, unknown accounting, forged bindings or Trace, stale
source, absent retry and insufficient Job allowance. Default-off serialization,
previous local-stop/closure paths and archived result recomputation remain
compatible. Results belong in separate execution records.

ScriptedLLM validates this mechanism only. A real task may still finish directly
without requesting an expert. Neither outcome alone establishes a collaboration
benefit or justifies changing the benchmark, hidden Oracle, permissions or
budgets to force an episode.
