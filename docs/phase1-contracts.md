# Phase 1: task contracts and CoreCoder adapters

Formal execution platform: Linux. No new runtime dependencies. Original coding
commands and tool loops remain in corecoder; researchcoder is an additional Python
package. This phase does not implement Harness, DAG scheduling, worker processes,
Checkpoint, task recovery or a persistent database.

## Public entry points

- `researchcoder.contracts`: TaskSpec, TaskResult, EvidenceRef, VerificationResult;
  ArtifactRef, WorkspaceRef, BudgetSpec, ExecutionPolicy, VerificationSpec,
  UsageSummary, OutputSlice, TaskInput, DiagnosticOptions.
- `researchcoder.adapters.artifacts.ArtifactIO`: put_bytes/put_json, read_bytes/
  read_json, resolve, read_record, index_evidence.
- `researchcoder.adapters.corecoder.CoreCoderAdapter.execute(spec, context)`.
- `corecoder.workflow.run_verification` and `workspace_revision`.

All persistent contracts are frozen dataclasses with schema_version=1, explicit
field validation, and to_dict/to_json/from_dict/from_json. Unknown versions,
unknown/duplicate fields, invalid ranges, bool-as-int, nonfinite numbers and
runtime objects are rejected. Sequences become tuples. job_id identifies the
result namespace, task_id a stable work item, attempt_id an execution attempt.
The latter comes from process-local AdapterContext, not TaskSpec.

TaskSpec kinds are diagnose, repair and verify. A raw_log input must reference a
full artifact, repair needs a full diagnosis_run. No summary substitutes for a
handoff. The original ev_/obs_ IDs and cause/context/dependency relations remain
in immutable raw run artifacts and their EvidenceRef index. Historical recovery
reads the artifact snapshot; it does not silently replace it with current source.

## Execution and permissions

AdapterContext holds a fresh model_factory, ArtifactIO, mode, optional tool and
verification guards, and a trusted tuple of permitted VerificationSpec values.
It is process-local, not JSON. Each execute calls a fresh factory. Existing
diagnostic and repair code creates per-run tools; the adapter never instantiates
the unsafe default shared ALL_TOOLS Agent. No model key is serialized.

TaskSpec.policy.allow_verification alone is insufficient: the exact specification
must also be present in the controller context. It is checked before any model
call/edit and again at command execution. The adapter exposes direct read/edit/
write tools by default, explicit writable_paths, and no Bash or agent tool. Its
optional grep/glob scopes are checked. Original repair guards/root checks remain.
These are application checks, not an OS sandbox; a verifier itself runs trusted
user-provided code. Legacy run_repair retains its explicit-command behavior.

Adapter → run_diagnosis; Adapter → load_handoff → run_repair; standalone verify
and run_repair → the same run_verification. No copied token loop or second
subprocess verifier exists. The verifier has an optional controller guard,
preserves passed/failed/timeout/launch_error/interrupted, and adds actual elapsed
time and checked_workspace_revision to the original output dictionaries.

## Workspace identity

workspace_revision hashes source/config file contents and relative paths, rather
than relying on Git HEAD for dirty workspaces. It includes .py/.yaml/.yml/.json/
.toml/.ini/.cfg/.txt/.md; excludes dotfiles, symlink files, and .git/.tmp/.venv/
venv/__pycache__/node_modules/dist/build directories. Explicit bounds: 20k entries,
4k files, 1 MB per source, 20 MB total. Exceeding limits fails explicitly. This
identity does not cover datasets, weights, external oracle files or arbitrary
shell effects. Controllers guard protected external oracle inputs separately.
The source revision is sampled before verification; it is not a filesystem lock.

## Result semantics

- Diagnosis completed → succeeded means structure/references validated; semantic
  correctness remains unverified, preserved in the raw report.
- Repair verified → succeeded + passed verification.
- Repair completed_unverified → succeeded + not_run, never “verified”.
- Repair incomplete with partial changes + passing test → **incomplete**.
- Nonzero exit/timeout → failed, launch failures retain their distinct category.
- Interruption → cancelled; no automatic retry or additional repair.

Raw events, configuration, evidence, observations, context/selection events,
metrics, file changes and verification are stored intact. Missing provider usage
maps to unknown token values with usage_complete=False; raw counters remain in
the original artifact. Some interrupted legacy calls do not return their
in-memory partial investigation; the adapter records this limitation.

ArtifactIO uses job directories, exclusive creation, bounded bytes, SHA256/size
validation, normalized logical paths and symlink rejection. Do not mutate stored
files: a changed digest is rejected. Files are not database transactions, and
the store does not provide crash recovery or cross-process task coordination.

## Regression and example

Use the existing Linux pytest environment and point Torch integration tests at
the existing experiment interpreter. Run all original 636 tests plus Phase 1
tests. Do not delete assertions or add skips to obtain a green result. Original
CLI demos and installed-wheel imports must also pass. See the autograd_detach
case README for real CPU healthy/faulty/frozen controls and the ScriptedLLM demo.

Windows ctime/path/encoding compatibility remains separately recorded in Phase
0.5 and is not claimed fixed by this phase. Phase 2 can consume the same task
values inside independent workers; it must add task state, budgets, tracing and
tool factories before attempting concurrent execution.
