# Phase 2 Agent Harness

Linux-only, synchronous single-controller/single-worker runtime. It reuses the
Phase 1 contracts and CoreCoderAdapter. It does not implement DAG scheduling,
automatic task retries, durable recovery, a database or an OS security sandbox.

## Entry point and controller configuration

```python
from researchcoder.runtime import ResearchHarness, ExecutionConfig
from researchcoder.runtime.protocol import RuntimeLimits, ModelSpec

harness = ResearchHarness()
config = ExecutionConfig(
    artifact_root='/absolute/job-artifacts',
    allowed_workspaces=(task_spec.workspace.root,),
    max_policy=trusted_policy,
    permitted_verifications=trusted_commands,
    model=ModelSpec(),  # scripted; use explicit turns for model-driven tasks
    limits=RuntimeLimits(max_runtime_seconds=60),
)
result = harness.execute(task_spec, config)
record = harness.records[result.attempt_id]
close_trace = harness.close_job(task_spec.job_id)
```

ExecutionConfig is trusted process-local input, never an LLM tool payload. Its
worker command override exists for trusted deployment/testing, not model routing.
ModelSpec openai uses only a whitelisted environment variable **name**. Credential
values enter the worker environment, never WorkerRequest, Trace or CLI arguments.
Request contracts reject unknown fields, client objects, credential-bearing URLs,
recognizable sk- material and configured credential values. This is not a general
DLP classifier for arbitrary user prose; do not intentionally place secrets in
task descriptions or scripted tool content.

## States and isolation

TaskRepository is owned by its creating controller thread. Jobs are open/closed;
tasks transition pending -> running -> succeeded/failed/incomplete/cancelled.
An Attempt has a distinct identity and follows running -> terminal. Illegal
transitions, duplicate completion, wrong result identity and implicit task replay
are rejected. Phase 2 permits one attempt per submitted task; explicit retry
transitions will be a future controller policy, not a hidden worker operation.

Workers never receive a repository object. They deserialize WorkerRequest,
create a fresh model, ledger, hooks and AdapterContext, execute the existing
Adapter, and return WorkerResponse. Per-run diagnostic and repair tools are
already independently created by the original inner workflows. fresh_tools and
isolated_agent provide the same guarantee for generic Agent use; the unsafe
legacy default ALL_TOOLS list is never used as a Harness task's live toolset.
Recursive agent tools are not exposed by Harness. Original CLI behavior remains.

Worker bootstrap runs from a private temporary directory using the trusted
runtime package path. Legacy cwd changes happen only inside that worker. A
per-user, hashed-workspace flock serializes all task access in Phase 2, including
reads; shared read leases and multiple-worker scheduling are deliberately deferred.

## Supervisor lifecycle

WorkerSupervisor creates a new Linux session/process group, observes timeout or
threading.Event cancellation, sends SIGTERM, then escalates to SIGKILL. It enables
Linux subreaper behavior and reaps only children in the supervised process group.
Cleanup also runs after a leader exits normally, preventing an unwaited child
from being left running. Launch failure, crash, missing/malformed/foreign response,
timeout and cancellation become explicit non-success outcomes.

Task time admission starts at the controller task span. Cleanup has a grace
period and result I/O can add latency after the execution deadline; this is not
an exact hard deadline for the entire execute() return. Process groups cannot
contain deliberately escaping daemon sessions, prevent file/network access, or
provide CPU/RAM quotas. The original verifier captures output before truncation;
it is not a bounded-memory executor for arbitrary noisy programs.

Supervisor drains worker stdout/stderr in bounded chunks and stores byte counts,
SHA256 and omission flags. It does not retain raw worker stream bodies. Original
authorized diagnostic/verification outputs remain in their Phase 1 artifacts.

## Budgets and instrumentation

BudgetLedger checks before model requests and records reservations/settlements,
tool attempts, Agent rounds, wall time, context/output estimates and aggregate
token liability. Unknown or failed requests retain their reserved liability;
they are never interpreted as free. Known Usage and estimates are distinct.
On missing worker accounting, parent counters are observed trace lower bounds,
token Usage remains unknown and the record explicitly says so.

BudgetedModel wraps the existing model interface. A minimal optional transport
context in LLM counts each SDK create attempt, including retries and rejected
stream_options fallbacks; SDK automatic retries are disabled in worker clients.
Context-summary and report-correction requests pass the same gateway. Original
Agent optional hooks observe rounds and actual dispatch without replacing its
token loop. Hooks serialize within-worker tool effects and enforce a trusted
diagnostic capability subset independently of model-proposed names.

The shared run_verification also uses hooks: controller-owned final verification
counts as a tool execution and is traced. After a model/round budget stop, it may
still inspect partial edits only if remaining tool/time allowance permits it.
It cannot resume the Agent. A passing final check never upgrades an incomplete
repair to succeeded. When runtime admission blocks verification, partial edits
and raw investigation records remain and verification is explicitly not_run.

Token admission uses characters/3 plus an output reserve. Live max_tokens is sent
to the provider, but tokenizer mismatch, provider accounting and reasoning-token
semantics can still exceed an estimate; an overrun is recorded and stops future
requests. This is not a guaranteed monetary spending ceiling. Injected/scripted
Usage is synthetic, not real model performance or billing evidence.

## Trace and result archival

Metadata-only events carry job/task/attempt IDs, unique event/parent IDs, ordered
sequence and shared Linux monotonic timestamps, UTC start/end timestamps, status,
model/tool labels, usage, elapsed time and failure kind. Events cover job/task,
worker, model/transport, tool/verification, budget, errors and final results.
Worker events are validated and merged in execution order; malformed events are
rejected. No prompts, tool arguments, raw file bodies, provider exception bodies
or environment dumps are allowed in Trace fields.

The result retains the original Adapter artifacts, an original WorkerResponse,
a receipt with Trace/budget/stream metadata, and an archived result. The archived
result excludes its own artifact reference to avoid a self-hash cycle. Repository
state is finalized only by the controller. If archiving fails, the in-memory
state still becomes terminal and already validated results are retained where
possible; the runtime explicitly reports the missing archive rather than
claiming durability. Trace and receipts are not Checkpoints.

## Tests and demonstration

Run the existing 715 tests plus tests/test_runtime_*.py in the existing Linux
pytest environment, setting RESEARCHCODER_EXPERIMENT_PYTHON to the existing CPU
Torch interpreter. No API key or network model call is required.

```sh
export RESEARCHCODER_EXPERIMENT_PYTHON=/home/example/miniforge3/envs/M2D/bin/python
/home/example/miniforge3/envs/researchcoder/bin/python -B -m examples.harness_autograd_demo --output-dir /tmp/new-harness-autograd-demo
```

The example submits two sequential tasks to two distinct workers. Both call the
existing Adapter; the repair is checked by the existing external CPU oracle on
held-out inputs. It demonstrates engineering correctness with ScriptedLLM, not
diagnosis accuracy, parallel Multi-Agent orchestration or DAG speedup.
