# M9: Explicit Code-first collaboration

`ExpertPolicy(initial_mode="code-first-v1", mode="public-failure-v1")`
selects a controller-authorized, read-only Code investigation before the first
Coding attempt. The default is `off`; that default is omitted from serialized
policy values to preserve historical policy fingerprints and behavior.

## Execution and authority

1. The existing protected public preflight command runs through the Verifier.
2. A controller-only `InitialConsultationReceipt` binds its actual TaskResult,
   execution record and trace to the Job, preflight Task/Attempt, protected
   command, full candidate manifest and workspace revision. Both source and
   Artifact digests are checked again immediately before dispatch.
3. `decide(..., initial=True)` uses a distinct `INITIAL_RULES_SHA256` identity
   and the original Job allowance. It selects exactly the existing Code role
   once, retaining the original expert caps and writer reserve.
4. The existing `execute_consultation` runs the Scheduler/Worker and validates
   exported observations through Evidence Exchange. The writer receives
   source-bound evidence and hypotheses, not a successful diagnosis fabricated
   from an unfinished task.
5. The first real writer remains `coding-0`. The existing
   `project-expert-reread-v1` profile requires it to inspect current candidate
   source before editing. Optional Evidence Context uses the existing structured
   handoff, including on generation zero. Original public and independent final
   checks determine the result.

The preflight's real `workflow_completed=False`, `status="failed"` and
`failure_kind="verification_failed"` are retained. This flag describes the
failed verification task; the separately checked command start/end and
`worker.result` events prove that the protected command finished normally with
exit code 1 and a valid failing public receipt. No earlier writer, model request,
budget exhaustion or voluntary request is invented.

The immutable initial receipt is recorded as `raw.initial_consultation`. The
route's phase is `initial_public_failure`, with the preflight Task/Attempt IDs.
Real writer attempts alone appear in `raw.rounds`. After this consultation the
existing episode cap prevents another expert call, and the M8 request option is
not advertised to the writer. Failed/unknown expert results retain their actual
records and stop the writer when the existing safety checks require it.

## Bounds and information separation

This is an explicit orchestration configuration, not a learned adaptive router
or proof that the task intrinsically requires multiple agents. It adds no Agent
role, tool authority, write permission, scheduler, budget system or model call
path. Unknown usage, pending accounting, cancellation, changed source and
permission failures retain the existing stop behavior. Initial consultation
requires room for both the expert cap and writer reserve; it does not silently
fall back to a direct run when explicitly requested but inadmissible.

Only the public task, public log, source inventory and public failure projection
enter expert context. The initial receipt's trusted command configuration stays
controller-side. Private F2P/P2P profiles, answers and checker paths are not
projected into prompts. The Code assignment asks for source-backed observations,
public component contracts and unresolved questions; it does not prescribe a
case-specific fix or disclose a hidden assertion.

An unclear public component contract remains a benchmark limitation and must be
reported separately. This mode does not modify frozen project specifications,
benchmarks or Oracles. Scripted tests demonstrate the collaboration mechanism;
real effectiveness requires independently archived model execution.

## Changed surfaces

- `routing.py`: opt-in policy field, distinct decision identity and bounded
  initial admission.
- `consultation.py`: initial receipt validation and generic Code assignment;
  reuse of the existing read-only execution and evidence handoff.
- `runner.py`: preflight-to-consultation orchestration before writer admission;
  existing generation-zero writer and verification paths remain in use.
- `tests/test_m9_code_first.py`: focused execution, default-off and negative
  provenance/accounting checks. Test results belong to the M9 run report, not
  this implementation description.
