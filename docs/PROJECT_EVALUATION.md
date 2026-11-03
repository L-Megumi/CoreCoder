# Real repository Single-Agent evaluation

The `researchcoder.evaluation.projects` package accepts an explicitly reviewed
source tree and trusted verification profiles. It reuses the existing
`ResearchHarness`, process Worker, guarded `RepairAgent`, `BudgetedModel`,
`JobBudget`, `ArtifactIO` and public command Verifier. The built-in benchmark
entry point and historical data formats are unchanged.

This entry point is for bounded local Python repositories, including the M1
packaging and SAM historical cases. It does not install project dependencies,
clone repositories, run arbitrary shell tools, or choose an execution strategy.

## Input and source identity

`ProjectRunInput` contains the project/revision identity, complete immutable
`RepositoryManifest`, public objective/log and explicitly writable existing
paths. `public_input()` is an allowlisted projection without private checks,
reference patches, healthy sources or hidden expected outputs.

`all-files-v1` includes file bytes and sizes for dotfiles, configuration, documents
and binary assets. Capture and materialization reject missing/additional files,
symlinks, special files and known credential paths. Current bounds are 4,000
files, 20,000 entries, 1 MB per file and 20 MB total. Use an exported, reviewed
source tree: `.git` and `.env` are rejected, not silently excluded. These path
rules are not content secret scanning. File modes and empty directories are not
part of this identity. The old `workspace_revision` still accompanies every
TaskSpec/verification for compatibility; it is not replaced by the new digest.

Each trial copies the entire declared tree into a new `candidate` directory.
Inputs and previous outputs are never overwritten. Only named existing files
can change; file addition/deletion or changes to tests/configuration outside the
write list invalidate the trial. Source and verifier provenance are retained.

## Public and private verification

`ProjectCheck` is trusted controller configuration: a bounded argument vector
with one standalone `{project}` placeholder, protected checker SHA256 values,
timeout and a `ReceiptSpec`. It is not model input. `run_check()` uses the existing
Harness/Adapter/Verifier and retains the original TaskResult even when its receipt
is invalid.

`public_json` receipts require valid output fields and a candidate-bound origin
without changing the protected public checker's assertions. `checks_json`
receipts also require the exact unique check-name set, strict booleans,
aggregation, schema/profile values and optional source digest. Empty/truncated
output, collection/launch errors, stale origins and exit codes other than normal
0/1 business outcomes cannot masquerade as completed checks.

Preflight requires a real public failure, private F2P failure and private P2P
pass on the initial source. Only the public failure is supplied to the Agent.
Final private F2P/P2P run after all model work. They cannot trigger another repair
attempt. Public checker code and non-writable candidate files are protected in
the coding Worker; private checker paths/configuration are used only by scoring
Workers. Candidate access uses `read_file` and `edit_file`, with no shell or
implicit subagent tools. This is application-level isolation, not an OS sandbox
against hostile candidate programs.

## Execution and feedback

`ProjectEvaluationRunner.run(...)` creates one CodingTaskSpec per attempt. The
existing Single route reads sources, edits through real tools and performs the
authorized public verification after the Agent finishes. Its final explanation
is `model_prose_not_structured_diagnosis`; this entry point does not fabricate a
validated diagnostic handoff or score root-cause accuracy with keywords.

Only a normal completed public-check failure may trigger a bounded feedback
attempt. All attempts retain the same Job identity and JobBudget. The controller
reuses `single_limits` for remaining allocation, preserves original failures and
changes, rebinds the current source version, and stops on repeated unchanged
failure evidence. A retry is a new Worker, not process/checkpoint restoration.

Unknown usage retains conservative budget responsibility and normally stops
further model work. Permission violations, stale/changed protected input,
interruption, timeout, incomplete execution and invalid verification do not
become successful repairs. A legitimate partial TaskResult is archived before
workspace reconciliation. Complete E2E success additionally requires valid
preflight, completed workflow, public success, both independent profiles and
intact policy/source checks.

Agent JobBudget includes model/tool activity and the public verification within
each coding attempt. Preflight and final independent scoring are evaluator work,
recorded separately under `evaluation_checks`; their time is part of trial wall
time, not claimed to fit the Agent's runtime limit. There is no pricing default.
Scripted token values are simulated accounting inputs and never actual provider
usage or measured model effectiveness.

## Interfaces and CLI

```python
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy

result = ProjectEvaluationRunner().run(
    new_trial_directory, public_task, reviewed_source_directory,
    public_check, private_f2p_check, private_p2p_check, model,
    policy=ProjectPolicy(),
    feedback_models=(),  # Only offline ScriptedLLM fixtures may swap fixed responses.
    allow_live=False,
)
```

The local CLI consumes a versioned trusted configuration containing `task`,
`source_root`, `public_check`, `f2p_check`, `p2p_check`, `model`, and `policy`, plus
optional `feedback_models`/`model_options`. Never put API keys in that file.
Use the existing credential environment mechanism only after authorizing live
model scope and budget. Live execution is rejected unless explicitly opted in;
there are no paid calls in offline acceptance.

```bash
python -m researchcoder.evaluation.projects --config trusted-project.json --output new-trial
python -m researchcoder.evaluation.projects --recompute new-trial
```

Results include `public-input.json`, private execution configuration,
per-attempt task inputs, `raw-reference.json`, immutable Artifact/Trace/request
records and `result.json`. `recompute()` verifies the raw Artifact reference and
rebuilds scores without executing a Worker or model. It does not re-run Oracle
commands. Root-cause and localization fields remain null pending independent
review. Trial-level durable recovery is not added. Optional expert routing is
described below.

## Tests

`test_project_contracts`, `test_project_workspace`, `test_project_verification`,
`test_project_runner` and `test_project_cli` cover serialization, full manifests,
real Worker execution, strict receipts, public feedback, conservative accounting,
private-read rejection, source mutations, cancellation, offline recomputation
and CLI entry. The M1 historical sources and private independent checks remain
an external, versioned intake bundle; the package has no dependency on their
machine-specific absolute paths. Offline fixed-action controls must not be
reported as real-model repair success rates or unseen generalization results.

## Optional public-failure expert consultation

M3 adds `ExpertPolicy(mode="public-failure-v1")`; the default `off` retains the
original Single route. The policy first attempts the same guarded coding task.
After a normal, completed public verification failure, it computes a versioned
`PublicRoutingFeatures` projection and records a deterministic `RoutingDecision`.
No case ID, benchmark difficulty, private Oracle answer or final F2P/P2P result
is accepted by the feature extractor. Decisions retain policy, feature and rule
digests and can be recomputed without a model.

When one remaining writer plus an expert can fit the existing Job budget and
deadline, the controller may consult Code Analysis. A public log containing an
explicit multi-exception traceback chain may additionally select Log Analysis;
if two experts do not fit, Code alone is considered. At most one consultation
episode is permitted. A public pass never triggers experts, even when final
private regression checks subsequently fail. Unknown accounting, pending
responsibility, invalid source, cancellation and policy violations stop progress.

`ProjectSourceSnapshot` copies the entire `all-files-v1` candidate and implements
the old Scheduler's `workspace/verify` interface. It validates both the complete
manifest and legacy revision. The existing Scheduler executes only the selected
read-only diagnostic tasks, with no write or command-verification permission,
using the same JobBudget object. It already reserves and settles expert calls;
the project controller collects those records without charging them again.
Complete snapshot copying does not extend existing diagnostic tool formats:
some formats such as `.rst` remain outside `read_project_file` support.

Successful producers enter the existing EvidenceExchange checks for source,
identity, Artifact SHA and scope. An ordinary local failure from one expert is
retained without discarding another expert's valid evidence. Unsafe failures
and unknown usage stop peers through the existing supervised cancellation path.
No expert's partial report or failed graph is relabeled as successful.

The validated bundle enters the next Single task's `raw_log` as untrusted data,
with original IDs, producer identity and a candidate relative-path mapping.
Expert conclusions remain hypotheses. The writer uses the existing scoped
tools, and the opt-in `project-expert-reread-v1` profile requires a successful
read of the current file digest before editing after expert handoff. This uses
the existing RepairAgent's successful-read cache; the original profiles retain
their previous behavior. A subsequent candidate change removes stale expert
content from current input while retaining its original Artifact and a context
event. It does not trigger another consultation or rewrite historical evidence.

The CLI accepts optional `expert_policy` and, solely for offline ScriptedLLM
fixtures, `expert_models` keyed by the known roles. Live trials cannot swap the
model between roles or feedback attempts. A trusted external experiment
controller may pass a `worker_command` through the Python API for transport
observation; the CLI and public task projection do not accept that field.

Default per-expert limits are four model calls, six tool calls, four rounds,
80,000 Token responsibility, 45 seconds, 16,000 context tokens and 2,048 output
tokens. Admission also retains at least three model calls, three tool calls,
20,000 Token responsibility and 15 seconds for the following writer. These are
bounded policy choices, not learned optimal allocations. The policy is
feedback-driven routing, not RSI or automatic strategy learning.

## Optional bounded correction of public-check path mistakes

M3-R1 adds `permission_recovery_mode="bounded-public-read-v1"` to the Runner
and trusted CLI configuration. The default `off` keeps the previous prompt and
guard behavior. The new `project-single-v2-public-paths` instructions explain
that traceback paths do not confer filesystem authority; prompt identity and
SHA256 are archived with the trial.

Only an exact canonical absolute path appearing in both the controller's public
verification command and its protected-file list is eligible for one corrective
feedback per Job. The read itself is denied and never executes. The model may
then inspect candidate source with the same existing tools and write policy.
Symlinks, private targets, unknown external paths, unauthorized writes and a
second denied public-check read stop execution. Paths supplied by logs are never
converted into permissions. Expert tasks do not receive fresh correction credit.

`PermissionRecoverySpec` is serializable; a `PermissionRecoverySession` maintains
only the Worker-local portion of the allowance. The Runner retains cumulative
usage across feedback attempts. A typed `PermissionDenied` contains fixed reason
codes, an opaque target-path digest and an unexecuted-operation assertion, with
no raw arguments or file contents. RuntimeHooks publishes its immutable Artifact
before returning feedback or stopping. The envelope binds Job, Task, Attempt,
policy digest and workspace version. Partial execution archives preserve legal
events and changes on controlled termination without claiming verification or
workflow success. Runner auditing requires every exempted correction to match
its original tool error and immutable receipt exactly.

This is bounded parameter correction, not automatic retry, a file allowlist,
OS sandbox or new recovery engine. A hard process kill can leave an immutable
receipt without a final result reference; such incomplete records are not a
license to replay an operation. Live experiments may add a separate aggregate
scope gate over the same receipts without replacing JobBudget. Missing Usage,
unknown effects, broken provenance and source changes retain their stop rules.

## M4: observed source context and bounded local-stop handoff

These are explicit, independent strategies. Omitting the new options preserves
the previous project route. Existing experimental records are never relabeled.

`writer_context_policy=ContextPolicy(role="repair", strategy="evidence",
allow_recovery=True, coding_source_mode="observed-source-v2")` connects the
existing EvidenceContextEngine to the actual coding writer. It does not create
another Agent Loop. Successful guarded `read_file` observations produce an
immutable rendered reply, a derived source page with original line numbers,
file SHA256 and the complete `all-files-v1` candidate identity. The legacy
WorkspaceRef revision remains unchanged. Both revisions govern current-source
recovery; old pages require explicit historical recovery. Recovery provides
only previously observed content and never grants the fresh-read qualification
required for an edit after expert handoff.

The model sees bounded source excerpts and a small reference index even after
complete old tool-call/result groups are removed. In v2, a successful recovery
is a pending delivery bound to its actual tool call. Its complete requested
content and matching call/result group must reach the next admitted model input;
it cannot be silently summarized a second time or displaced by a later recovery
of the same document. If the full delivery does not fit the existing budget,
the request stops before invoking the model. Source views and external logs remain
untrusted data. Permission denials always pass through the original Guard and
receipt path; context deduplication cannot swallow a second denial. Final
system instructions, tool schemas, pending responsibility and output reserve
remain subject to the original BudgetedModel admission. `strategy="original"`
with the same source mode and recovery permission supports a controlled context
comparison. Character-based estimates are not provider Token usage or a billing
guarantee, and context processing itself has measurable runtime/storage cost.

`observed-source-v1` remains available to replay the frozen M4 R0 policy. R0 live
request inspection found that restored windows could be immediately reselected
to 20 lines and that two windows of one document could overwrite each other in
the view. Use v2 for recovery delivery. The R0 eight-trial results and immutable
source identity are retained; v2 is a separate implementation and calibration,
not a relabeling of those results. Neither mode grants new filesystem authority.

When this strategy is enabled, the Controller delivers previously checked
consultation references and complete public-failure history through the existing
structured `evidence_context` channel. It does not mix the handoff into free-form
log text and expect an exception parser to rediscover it. The Engine validates
same-Job Artifact digests, producer identities and complete candidate identity;
bounded views retain producer/source mappings and keep model hypotheses separate
from observed records. `current_eligible` describes source-version currency;
it never changes `unverified_hypothesis` into an observed or verified fact.
Full history remains in immutable Artifacts. Combined
tests inspect actual next-writer inputs, since a fixed scripted repair can still
pass its Oracle even when the model-visible handoff has been lost.

`handoff_policy=HandoffPolicy(mode="local-stop-v1")` enables at most one
Controller-audited continuation from a known local budget stop. It also requires
an enabled expert policy. A Worker only archives what happened: settled model
requests, original tool events, file changes, source identities and the actual
protected public verification. The original CodingTaskResult stays incomplete.
The Controller rechecks immutable Artifact SHA, identities, exact candidate
changes, complete accounting and the failed public verification before entering
the existing expert consultation and subsequent writer. All tasks share the
original JobBudget; no consumed or unknown responsibility is reset.

Missing proof, unknown/pending usage, permission errors, unknown write results,
timeouts and exhausted global allocations do not qualify. Tool-budget exhaustion
without capacity for the public check does not qualify. A public pass after an
unfinished workflow is still incomplete. Context-limit handoff additionally
requires an enabled evidence context strategy. There is no implicit replay,
additional verifier, task recovery service or expansion of write permissions.

`ExpertPolicy(mode="public-failure-v1", schedule_version="serial-parallel-v2",
max_workers=1)` and the corresponding `max_workers=2` use the same role selection,
read-only tasks, prompts, DAG, local caps and total JobBudget. Both reserve the
sum of selected expert time caps plus the following writer allowance; only
execution concurrency differs. An episode with just Code Analysis has effective
concurrency one in both configurations. Archived consultation metadata records
the requested/effective concurrency. This comparison can establish scheduling
effects only when multiple independent expert tasks actually execute.

The M4 controlled fixture uses two real source modules, a genuine chained
exception, a protected public checker and separate F2P/P2P checks. Scripted
repairs, incorrect sample-specific patches and regression-producing patches
test the mechanisms; their outcomes are not real-model repair accuracy. Real
trials must separately record activated experts, handoffs, context receipts,
actual usage and independent outcomes. Keep Single results when reporting
serial/parallel or compression effects, and do not infer adaptive superiority
from different stochastic trajectories that never invoke an expert.

## M5: accounted investigation stops and public execution feedback

`HandoffPolicy(mode="local-stop-v2")` extends the opt-in M4 continuation with
the existing eight-round investigation guard. A trusted constructor option
produces a typed stop only on this path; default and v1 retain the original
exception type and stopping behavior. The Worker archives the exact read-only
investigation window and its Trace identity, then runs the already permitted
public verification within the remaining task allowance. The Controller checks
the proof, settled accounting, source version and effects before deciding whether
the existing expert/writer route fits the same Job. The stopped attempt remains
incomplete. A no-op write within the investigation window, unknown usage or
effects, an uncompleted public check, exhausted Job resources or absent feedback
slot prevents this handoff. The policy does not increase the eight-round limit,
create another Scheduler, or reset consumed resources.

`PublicExecutionPolicy(mode="candidate-exception-v1")` is a separate trusted
project-runner option. It wraps only the public Python `-I -B` script invocation
inside the existing Verifier process, keeping the original checker immutable.
An exclusive observation binds the invocation, actual source before/after,
checker/request/probe identities, traceback code origins and exact text outputs.
The ordinary JSON validator remains unchanged. When that validator fails, only
a separately validated, failed public exception observation can reach bounded
feedback; it cannot create a successful task, JSON receipt or F2P/P2P result.
Original invalid JSON errors and raw observations remain archived for audit.

The initial exception profile accepts a source-matched terminal candidate frame.
It additionally observes builtin TypeError across source-matched, preexisting
interpreter-package frames after a candidate call; this records an execution
boundary, not definitive root-cause attribution. Protected checker re-entry,
unknown dependency provenance, launch/timeout/interruption, resource errors,
SystemExit without valid JSON and unmatched output remain review conditions.
Native/binary writes bypassing the Python text observer can cause an output-hash
mismatch and therefore refuse automatic feedback. This is not an OS sandbox.

Each generation receives a distinct exclusive public invocation. The original
public script remains in the trusted command binding, so existing permission
correction validation still applies. Private checks are never wrapped or fed
back to the Agent. Both M5 mechanisms are independently selectable; omitted
options preserve legacy result/config fields. Public failure feedback retains
the same Job limits, writable files, model and independent final scoring.

M4-R1 remains the context baseline: restored content is protected through its
first fully accounted provider delivery. M5 does not extend its retention
duration. Statistics now report the actual v1/v2 coding source mode. Repeated
reads must be evaluated against actual model inputs and fresh-read requirements,
not automatically treated as proof that a longer context cache is beneficial.

## M6: explicit completion inside the existing budget

The defaults and historical JSON identities remain unchanged. Set
`ExpertPolicy.report_version="explicit-v1"` to request an explicit transition
from investigation to the existing six-key diagnostic report. The expert keeps
the same local and Job limits. Two existing model-call slots are reserved for
reporting and at most one strict correction; this does not allocate extra calls.
The trusted runtime uses `RuntimeOptimization(version="explicit-report-v1",
compact_reports=True, report_reserve_calls=2)`. Invalid original responses are
retained. Unknown/pending usage, a stopped budget or unavailable current evidence
prevents automatic correction. Pseudo-tool text is never executed, and citation,
source-version and producer checks are not relaxed.

`ProjectPolicy.completion_version="explicit-v1"` separately enables the coding
writer's bounded completion path, including a public-verifier tool reserve and
correct context/output accounting. Explicit final submission withdraws tools
and asks the model to report completion; the controller still runs its original
protected public command. A known local stop may preserve legal edits and an
actually executed verification even in Single mode, without granting expert
consultation or replaying a write. Partial work plus a passing check remains
incomplete. Permission, unknown-effect/usage and time boundaries still apply.

Both switches are versioned and independently testable. Private F2P/P2P results
remain final scores only. Compare report validity, workflow completion, public
verification, independent repair success and real usage separately. A valid
report is not proof of a correct diagnosis; successful scripted replay is not
real-model accuracy. M5 records and prompts are immutable historical evidence.

## M7: active evidence and bounded progress-aware consultation

M7 has separate, opt-in policy identities. Omitted fields retain their historical
JSON representation and behavior. The following configuration enables the complete
candidate while keeping the original task limits, Job limits, protected public
check, writable paths and independent F2P/P2P scoring:

```python
from dataclasses import replace
from researchcoder.context.policy import ContextPolicy
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy

project_policy = replace(existing_project_policy,
    completion_version="explicit-v1", progress_version="progress-v1")
expert_policy = ExpertPolicy(
    mode="public-failure-v1", schedule_version="serial-parallel-v2",
    max_workers=1, report_version="explicit-v1", working_set_version="active-v1")
handoff_policy = HandoffPolicy(mode="local-stop-v3")
context_policy = ContextPolicy(
    role="repair", strategy="evidence", allow_recovery=True,
    coding_source_mode="observed-source-v2", working_set_version="active-v1")
# Pass these four policies to ProjectEvaluationRunner.run with the original
# source, checks, model and total Job budget. max_workers=2 only changes expert
# concurrency when the existing router selects two independent expert roles.
```

`progress-v1` observes actual successful source/evidence tool responses and
effective source edits. Three complete tool rounds with neither novel response
fingerprints nor edits trigger a typed stop. Novelty is a heuristic over returned
content and retrieval arguments, not proof of root-cause correctness. Changing
read windows can still count as new evidence; no model confidence or hidden
Oracle answer is used. An unchanged repeated read does not reset the window.
With explicit completion, an attempt that already edited source first submits
its final report without further tools, then executes the existing public
Verifier. A report alone does not establish repair success.

An unfinished stop can consult experts only after the Controller reproduces
its observation history, validates the bound Trace and source identity, verifies
settled usage and known effects, and receives an actual failed protected public
check. The previous Attempt stays incomplete and its original result is retained.
Experts receive the concrete public failure, current source snapshot and a bounded
description of the exhausted investigation window with the original result
reference. This is observation metadata, not an inherited diagnosis or a local
Observation ID. The metadata is delivered in a labelled task-data section and
also archived in the raw log: exception extraction alone does not expose general
log metadata to a diagnostic expert. If the complete objective plus receipt
exceeds the existing 6,000-character task limit, consultation stops with
`consultation_context_unavailable` rather than truncating the task. Their fresh
citations still pass the existing Evidence Exchange.
Only the existing one-episode expert DAG and remaining writer feedback slot can
run; all calls consume the same JobBudget. Unknown effects, usage, permissions,
missing proof or absent writer allowance continue to stop execution.

The writer and expert working sets can be enabled independently of progress
routing. Expert `working_set_version="active-v1"` creates a per-role ContextPolicy
inside the existing Worker; it does not reuse writer message history. Trusted,
source-matched tool observations remain available through bounded active-source
selection, while immutable original records and source-version checks remain in
place. It does not pin arbitrary log text or promote tool output to instructions.

For ablations, retain the old policy objects unchanged as the baseline; enable
only writer/expert working sets for context-only, or `progress-v1` plus matching
`local-stop-v3` for progress-only. Progress can run without a handoff policy in
Single mode, where a stop cannot spawn experts. Explicit v3 handoff with legacy
progress, or progress-v1 with a v1/v2 handoff, is rejected before creating a trial.
The new routing rules have a separate SHA256. Original v1/v2 rules and M6 records
are unchanged.

Raw attempts archive `coding_progress`, including all observed rounds, and the
result can be recomputed offline into `progress_metrics`. Report actual model
calls, tool calls, stopped attempts, novel fingerprints, no-progress rounds,
consultation activation, workflow completion and independent F2P/P2P separately.
The real-Worker ScriptedLLM integration verifies mechanism and accounting only;
it does not demonstrate a real-model success-rate or cost advantage.
