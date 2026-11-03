# Phase 3: Multi-Agent DAG orchestration

This outer controller composes the existing Harness, process Worker, Supervisor,
budget instrumentation, CoreCoderAdapter and public verifier. It does not replace
the Agent token loop. The supported execution platform is Linux.

## Static execution

```python
from examples.multi_agent_autograd_demo import prepare_demo
from researchcoder.orchestration import DAGScheduler

plan, config, artifacts, initial_probe = prepare_demo(
    "/tmp/new-multi-autograd-run",
    torch_python="/home/example/miniforge3/envs/M2D/bin/python",
)
scheduler = DAGScheduler()
result = scheduler.execute(plan, config)
summary = artifacts.read_json(result.summary_ref)
```

The example creates a reproducible CPU failure, then executes:

```text
Log Analysis ----\
                  Evidence aggregation -> Diagnosis -> Repair -> Verification
Code Analysis ---/
```

Every domain task executes the real Phase 1 Adapter in its own Worker. Log and
code investigation overlap in separate processes; candidate diagnosis, mutation
and verification take exclusive candidate access. Evidence aggregation is a
controller operation, not a fabricated extra LLM agent. Lead planning can be
enabled through the Planner described below.

## Contracts and authority

PlanRevision contains job/revision identity, ancestry and bounded acyclic tasks.
OrchestrationTask selects a controller-registered template and role, names
dependencies, optionally binds a source revision and records a retry predecessor.
TaskDependency can request an original diagnosis artifact or authorized evidence.
These contracts have no tool policy, shell command, endpoint, credentials or
execution root fields. Unknown fields, invalid roles, duplicate IDs, missing
dependencies, cycles and excessive graph depth are rejected.

TaskTemplate is trusted process-local configuration: the existing TaskSpec and
ExecutionConfig, its role, and whether to use the frozen source or candidate.
The controller compiles execution IDs as `<revision_id>.<logical_task_id>`.
DAGExecutionState is owned by the Scheduler thread; each task has at most one
Attempt per revision. SchedulerResult retains original TaskResult values and
links graph Trace and raw execution summaries.

Phase 1 TaskSpec/TaskResult schema_version=1 remains unchanged. Separate
PlanningTaskSpec/PlanningTaskResult and CodingTaskSpec/CodingTaskResult wire types
extend the runtime for a tool-free Lead and the single-Agent benchmark arm.
Phase 1 decoders still reject `kind=plan` and `kind=coding`. They are never
misrepresented as completed diagnosis reports.

## Multi-Worker supervision and workspace versions

HarnessWorkerPool uses threads only to supervise independent process-based
Harness executions. Each task controller creates its own Repository on its own
thread. No legacy workflow, Agent loop or cwd mutation runs in a pool thread.
The Scheduler alone commits graph state and Job budget. The old synchronous
ResearchHarness.execute API remains usable.

SourceSnapshot copies the exact bounded source/config manifest used by the
existing workspace_revision implementation. It checks hashes before, during and
after copying, sets read-only permissions, and verifies the copy before and after
read tasks. Git HEAD is not treated as an identity for a dirty working tree.
Unsupported assets, hidden files, symlinks, datasets and binary dependencies are
outside this source snapshot contract; they are not silently declared frozen.

Snapshot investigations share Linux flock read leases. Candidate operations use
exclusive leases; this version conservatively serializes candidate reads as well
as writes. Frozen snapshot reads can proceed alongside changes to a separate
candidate, but observations from the old revision cannot be merged as current
candidate facts. Verification binds an explicit source digest and reuses the
existing public verifier. Process-group cleanup remains the Phase 2 Supervisor's
responsibility, including leftover same-group children after a normal exit.

Task output storage uses `<job>/tasks/<revision.task>/<artifact>.blob`. Creation is
exclusive; different tasks may use identical artifact IDs without overwriting
one another. ArtifactRef remains the original content-checked wire value.

## Job budgets

JobBudget atomically reserves a task's full declared model/tool/token caps before
dispatch and settles from its existing Worker BudgetLedger. If a reservation is
temporarily unavailable, ready tasks wait for running tasks to settle. A task
that can never fit the remaining Job budget terminates explicitly. No child gets
an independent copy of the complete Job allowance.

Unknown and failed model attempts retain their per-request liability. Missing
Worker accounting retains the full Job allocation, including counter caps;
observed lower bounds are not mistaken for complete usage. Lead planning and
replanning reserve and settle against the same Job budget; verification counts as
a tool. Actual known counters, estimated input tokens, unknown usage and charged
liability are reported separately. Token admission is estimated, not a strict
billing ceiling. Conservative reservations can reduce parallelism.

## Role profiles and Evidence Exchange

Lead is tool-free; Log Analysis prioritizes log/exception evidence; Code Analysis
adds source search and symbol inspection; Diagnosis integrates authorized
evidence and independently inspects the current candidate; Repair uses the
existing bounded coding tools; Verification executes an explicit controller
command without an LLM verdict. Diagnosis and Repair share a logical specialist
family but have different permissions. Profiles provide system instructions,
capability subsets and budget caps; histories, tools and model counters remain
task-local.

EvidenceExchange registers original EvidenceRef values only after checking the
producer Task/Attempt, artifact membership, original record metadata, hashes and
source revision. Exchange IDs include job, task, attempt and original ID. The
original ID and artifact are retained even when another task reports the same
ID. Source-hash conflicts under the same revision are rejected.

Bundles are limited to controller-authorized dependency producers. Observed tool
records and unverified model hypotheses have separate classifications. A summary
never replaces an original observation. Bounded prompt excerpts retain full
Artifact/Evidence references for later retrieval. Prompt injection occurs before
budget estimation and does not mutate another task's message history. This is a
minimal cooperation layer, not long-term memory or advanced context compression.

## Planning and bounded replanning

`Planner.plan(PlanningJob, available_evidence, PlanningConstraints) -> PlanRevision`
runs one tool-free CoreCoder Agent in the existing isolated Harness. The
controller checks the returned JSON against graph bounds, registered roles and
templates, revision ancestry, retry predecessors, required workflow roles and
diagnosis/repair/verification ordering. Invalid plans still retain their original
output and consume the requests they actually made.

ReplanningController creates a new PlanRevision after verification failure or
insufficient evidence. It preserves the old graph, feedback, raw result artifacts
and Trace. A changed candidate receives a new immutable source snapshot and the
actual failure output becomes an input to subsequent investigation. It enforces
maximum replans, cumulative tasks, Job model/tool/token allocations and a shared
wall-clock deadline. Repeated semantically identical feedback with no new source
or evidence stops before requesting another plan. There is no cyclic DAG edge or
unbounded automatic repair loop.

Completed successes are reusable only if their input fingerprint, role/template,
model, instructions, tool/policy configuration, original input content, upstream
result identity and exact workspace version match. Reuse retains the original
Attempt provenance and performs no Worker execution or Job budget charge. A
source/input/configuration change invalidates it. A mutation is not replayed just
because its previous Attempt succeeded.

## Reproducible offline examples

```sh
cd /home/example/ResearchCoder
export RESEARCHCODER_EXPERIMENT_PYTHON=/home/example/miniforge3/envs/M2D/bin/python
/home/example/miniforge3/envs/researchcoder/bin/python -B -m examples.multi_agent_autograd_demo \
  --output-dir /tmp/new-multi-autograd-demo
/home/example/miniforge3/envs/researchcoder/bin/python -B -m examples.orchestration_comparison \
  --output-dir /tmp/new-orchestration-comparison --repetitions 2
```

Comparison arms: one existing guarded Coding Agent in one Worker; five domain
tasks plus Lead executed sequentially; the same multi-role graph with two
concurrent read Workers. They use the same injected fault, CPU oracle, model
label and Job caps. Fixture construction and the initial fault probe are outside
timing; Worker startup, Lead planning, tools, verification and archival are
included. Order rotates across repetitions. All raw records and traces survive.

The single arm supplies a model hypothesis and verified patch; the multi-role
arms additionally validate structured diagnostic reports. Fixed ScriptedLLM
actions differ by role and do not measure model diagnostic accuracy. This small
CPU example has no simulated LLM latency, so parallelism is not guaranteed to
reduce end-to-end time. Process barriers, unique PIDs and monotonic timestamps
prove actual concurrent execution independently of any speedup claim.

## Boundaries

No SQLite, durable Checkpoint/recovery, automatic Git operations, long-term
memory, complete Context Engineering, API service, MCP server or OS sandbox is
implemented. File permissions and process groups do not contain malicious code
running as the same OS user. External writers ignoring the advisory locks remain
a risk; manifests detect source changes but do not freeze external resources.
Trace/Artifact receipts are audit records, not recovery checkpoints. Partial edits
after cancellation/timeout are retained and versioned, not automatically rolled
back. Snapshot entry/byte bounds, evidence-view limits and conservative Job
allocations fail explicitly when exceeded. Linux is the acceptance platform.
