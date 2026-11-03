# M11: opt-in repair execution continuity

## Scope and version

`ContextPolicy.repair_state_version="state-v1"` enables this projection for a
coding task with role `repair`, `observed-source-v2`, recovery enabled, and either
`original` or `evidence` strategy. Default `off` is omitted from serialized policy
and WorkerRequest values. The default prompt, history, handoff projection and
legacy compression path remain unchanged. `working_set_version` is independent;
the new receipt remains protected even when active source windows are evicted.

## Actual runtime state

`RepairStateContext` binds only the current `RepairAgent` and its actual
`RuntimeHooks`/`BudgetLedger`. It does not read or interpret the text of a runtime
marker, log, source comment or model message. Each provider view gets a bounded
USER data receipt containing Job/Task/Attempt, current workspace/all-files hashes,
current read windows and their hashes/registered document IDs, stale read count,
successful write/edit outcomes and hashes, mutation count, read-cache hits,
remaining local limits and unknown/pending token liability. Tool outcomes are
captured after the original guarded tool; an error or no-op is never recorded as
a successful edit. Current read eligibility retains the existing strict complete
source revision rule, so an edit invalidates the previous source snapshot.

Verification data starts at `not_run`. Transparent wrappers observe the existing
verification hooks: admission is `admitted`, not a command success; only the
existing completion callback supplies a terminal status. A captured output does
not imply a pass. Failed/interrupted/launch-error results remain distinguishable.
No additional verification command, Tool, write permission or budget is created.
The receipt describes known successful file-tool effects; it does not reconstruct
arbitrary shell or unknown external side effects.

## Compression and accounting

The raw Agent history stays untouched. Only deep-copied provider messages are
projected. Incoming `_context_repair_state` flags are removed; marker-shaped
conversation text is labeled as untrusted. The sole protected receipt is generated
from the bound runtime, not registered by a model-provided ID. Every receipt has
an immutable EvidenceStore document with Artifact SHA256 and provenance.

`CompressionPipeline` excludes the receipt from lossy summaries and complete
tool-group eviction. Its final capacity pass cannot delete or shorten it. The
full receipt is included in the existing input estimate and the real request
passes through the existing BudgetedModel reservation/settlement. If the receipt
exceeds 16,000 characters or the required provider view cannot fit, the existing
`context_tokens` budget stop applies. A count of retained records is not evidence
that the model understood or used them.

The same policy flag selects `compact-state-v1` in CodingHandoffContext. It
retains complete expert claims/recommendations, scoped evidence identities and
restorable immutable originals while avoiding repeated producer/source metadata.
Both the initial projection and final pipeline reinsertion call the same versioned
handoff implementation. Hypotheses stay unverified; their presence is not a
successful repair or permission grant.

## Guidance, receipts and limits

A static, SHA256-identified `state-v1` system addition asks the writer to check
public task/component constraints after effective edits, recover a specific
remaining evidence gap, then summarize for the existing controller verification.
It contains no Benchmark answer, case ID, private Oracle result or domain-specific
repair rule. It does not force success, disable source checks, raise writer caps,
change tool schemas or add another finalization mechanism.

Context receipts expose the version, estimated retained receipt size and number
of retained messages. Final context artifacts additionally contain the immutable
state references, retained-view flags and exact current/stale read and modified
file counts. Trace emits metadata-only `context.repair_state.retained` events.
`retained_in_view` means admitted into a provider view, not confirmed delivery.
Unknown Usage still carries its original liability. No billing guarantee or
live-model improvement follows from offline replay alone.

Validation is targeted to policy roundtrip/default compatibility, actual guarded
read/edit state, forgery rejection, source invalidation, bounded compression,
budget accounting, exact tool pairing and the existing Worker/verification path.
An archived M10 replay and the fresh cpx_04 trial are recorded separately by the
execution controller; old prompts, artifacts and Oracle results remain immutable.
