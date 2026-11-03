# Phase 6D: complexity-aware benchmark

The separate complexity_benchmark v1.0.1 adds six CPU programs: three cross-module
faults and three cases with independent repair obligations. The original eight
PyTorch cases, their reference inputs and Oracles remain byte-identical. Three
existing development cases are Level 1 anchors. Full rubric, mechanisms,
reproduction commands and boundaries are documented in
[the benchmark README](../benchmarks/complexity_benchmark/README.md).

## Interfaces and reuse

- evaluation/complexity/contracts.py: versioned dimensions, spans, stages and
  manifest identities over existing stdlib Contract dataclasses.
- dataset.py, data/: private labels and reference material, verified source and
  Oracle digests, six-case manifest and 48 genuine CPU observations per program.
- reference.py and oracle.py: evaluator-owned mathematical reference and
  component/architecture/trainability checks outside candidate roots.
- protocol.py: explicit offline profile; it refuses a live provider.
- feedback.py: bounded shared-budget/deadline verification-feedback trials over
  the original Harness and Scheduler, with fresh snapshots. This fixed-revision
  evaluator is not a second Scheduler, learned planner or durable trial resume.
- analysis.py: frozen-level slices, recoverable evidence references, effective
  request counts, exact source-span retention and read-excerpt repetition.
- runtime/request_records.py: opt-in redacted local model inputs and tool schemas,
  with default-off backward-compatible WorkerRequest/ExecutionConfig fields.

The old Runner dispatches to the new dataset/Oracle without copying Agent loops,
AST tools, ArtifactIO, budgets, context compression, subprocess verifier or
DurableRecovery. Existing default paths and old contract canonical JSON remain
compatible. Extra request artifacts append to existing result outputs rather
than replacing diagnostic reports.

## Protocol

All modes share the public candidate inventory, goal, source, allowed write union,
final F2P/P2P Oracle, Job budget and deadline. B/C use one graph and role profiles,
with worker concurrency 1/2. Single retains coding tools, while Multi uses
specialist tools and handoffs; this is an architecture comparison. Single and
Multi receive the same complete feedback data and bounded retry opportunity.
A common JobBudget/deadline spans all rounds; source/attempt identities change,
old failures and unknown charges remain. Role-level quota differences are visible
in saved configs, not hidden as a strict single-variable ablation.

Default engineering profile: 80 model calls, 128 tool calls, 1M estimated Job
Token allowance and 240s. Per-role model/tool/round limits 12/16/12, context
32K and output 2048. Single uses the Job call allowance with the original coding
loop's 30-round/attempt cap. Character estimates are not strict provider billing
limits. Initial 16K context pilot gates failed before sufficient feedback could
be admitted; this was archived and a new common profile frozen before final runs.

## Verification and interpretation

Tests cover health/fault/full repair and P2P; all three partial repairs improve
some checks but still fail F2P. Controls reject constant outputs, factory or
post-step freezing, superficial NaN suppression, entry-only bypasses, undeclared
imports and read-only goal changes. Actual gradients, parameter/buffer values
and module APIs matter; decreasing loss or a model's prose cannot certify repair.
The whitelist/guards are not an OS sandbox against arbitrary hostile Python.

ScriptedLLM executes fixed known actions through real Worker processes, tools,
evidence exchange, context and verifier. Negative controls keep diagnosis failure,
wrong/repeated patches and budget exhaustion in denominators. Separate tests use
the original ReplanningController and genuine long numeric histories with
source-bound recovery/authorization. Default durable DAG tests remain intact.
Feedback-trial orchestration itself is in-memory, not a new multi-round checkpoint.

Raw fixed-feedback `replanning_count` counts graph revisions in Multi modes;
`feedback_retries` and `feedback_semantics` distinguish these from actual Planner
invocations. The standalone ReplanningController test covers actual replanning.
No fixed-script repair rate, lexical root-cause score or timing is evidence of a
real model accuracy improvement. Exact span retention is a representation metric,
not semantic completeness; duplicate excerpts do not prove unnecessary work.

New complex-case process verification uses discovery inputs, while outer F2P
uses holdout and P2P regression inputs. Matched Level 1 experiments explicitly
select discovery through trusted controller configuration; unchanged legacy
profiles and historical results are kept separately. A delivery audit distinguishes
Oracle implementation independence from evaluation-input independence.

The frozen data rubric does not fit a scalar from Agent wins. Ambiguity is a
builder-authored candidate-cause rubric, not empirically calibrated uncertainty.
Levels are not a proof that Multi must outperform Single. Public logs use input
seeds disjoint from Oracle profiles. Validation cpx_02 and held-out cpx_06 are
exercised only as known-action fixture controls, not live strategy-selection data.
This is not an externally blinded benchmark.

The delivery includes full pytest/JUnit records, original-node retention checks,
installed-wheel tests, complete Trial/Trace/Oracle/request records and immutable
source snapshots. No new complex-case paid API calls were made. A separately
proposed development-only real experiment and Phase 7 candidate-generation /
independent-evaluation / validation-selection / versioned-adoption / regression
pipeline remain designs. Static routing is not RSI; gold complexity labels must
not become deployment features or disclose the reference repair.
