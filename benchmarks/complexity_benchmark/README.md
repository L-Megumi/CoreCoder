# Complexity Benchmark v1.0.1

Six CPU PyTorch programs form a separate `complexity_benchmark` subset. The
original eight-case `pytorch` dataset and its Oracle remain unchanged. Three
original development cases provide Level 1 anchors; no anchor is duplicated.

| Case | Level / split | Mechanism | Partial repair still fails |
|---|---|---|---|
| cpx_01 | 2 / development | Optimizer omits an encoder with valid gradients | N/A |
| cpx_02 | 2 / validation | Length-based packing breaks feature/target pairing | N/A |
| cpx_03 | 2 / development | Zero-feature normalization introduces NaN upstream | N/A |
| cpx_04 | 3 / development | Modality detach plus missing optimizer ownership | Restore fusion gradients only |
| cpx_05 | 3 / development | Padding-sensitive pooling plus weight normalization | Restore masked pooling only |
| cpx_06 | 3 / holdout | Validation mode leak plus erased accumulated gradients | Restore validation mode only |

`researchcoder/evaluation/complexity/data/complexity.json` freezes independent
dimensions: module inventory, AST import edges, evidence files/spans, an explicit
initial-hypothesis rubric, cumulative reference patch obligations, deduplicated
context characters and visibility. Token estimates use characters/3 and are not
provider usage. RepairStages counts independent repair/verification obligations,
not the minimum number of Agent rounds. A complete repair may succeed at once.

The builder generates 48 real, independently seeded CPU runs per case: 16 normal,
16 stress, 16 normal. Public seed range 10001..10048 does not overlap the private
Oracle profiles. Records contain measured losses, gradient/update norms and mode
checks; no padding text is generated. Shared historical logs are immutable in both
healthy and faulty projects. Program/reference labels were fixed before Agent
results. v1.0.0 is a retained engineering prototype; v1.0.1 separates public inputs
and strengthens component/trainability checks. Categories and difficulty rubrics
were not selected from Agent success.

## Reproduce offline

For the complete shared-budget, independent-input protocol (24 complex + 12
Level 1 Trials), run `PYTHONPATH=. python benchmarks/complexity_benchmark/run_offline.py
--output-dir "$RUN_ROOT/new-independent-run" --torch-python "$TORCH_PYTHON"`
as one shell command. This offline-only standalone controller records all gateway
requests and selects discovery process validation for the reused Level 1 anchors
without changing their data or historical default protocols. Its scoped builder
override is restored on exit and is not a concurrent service extension.

Use existing environments; the project does not automatically install Torch.
`TORCH_PYTHON` must name an existing CPU-capable PyTorch interpreter.

```sh
python -m researchcoder.evaluation list --subset complexity_benchmark
python -m researchcoder.evaluation run --subset complexity_benchmark \
  --output-dir "$RUN_ROOT/new-unique-run" --torch-python "$TORCH_PYTHON"
python -m researchcoder.evaluation run --subset complexity_benchmark \
  --cases cpx_04 --behavior staged_correct --feedback-rounds 1 \
  --output-dir "$RUN_ROOT/new-staged-run" --torch-python "$TORCH_PYTHON"
python -m researchcoder.evaluation report "$RUN_ROOT/new-unique-run"
python benchmarks/complexity_benchmark/build.py \
  --output "$RUN_ROOT/new-generated-data" --torch-python "$TORCH_PYTHON"
```

The CLI defaults to development cases. All-split fixtures are exercised only as
engineering controls, without paid models or strategy selection. Future paid
experiments must protect validation/holdout. The exact producer of measured v1.0.1
logs is preserved in the delivery snapshot; a subsequent unused-import cleanup of
the builder changes its source hash, not any frozen measurements or Oracle bytes.

## Independent verification and boundaries

The evaluator owns immutable `oracle.py`, `reference.py`, manifests and reference
patches outside candidate roots. Architecture, public read-only material and
candidate file inventory are checked. Independent float64 reference computation
compares predictions, losses, gradients, updates, buffers, trainability, input
preservation and intentional freeze/eval controls. Component APIs are checked
independently to reject entry-only workarounds. Process validation uses the discovery profile; final F2P uses holdout inputs
and P2P uses regression inputs. These profiles are disjoint from one another and
from public controlled-run seeds. Hidden seeds are not published in verification output. Reference patches/answers never enter a live ModelSpec.
Scripted fixtures intentionally know answers; their success is not model accuracy.

Static import/reflection restrictions and file guards cover the tested bypasses;
they are not an OS sandbox or proof against arbitrary hostile Python execution.

All four modes get the same public files, final Oracles, candidate write union,
Job budget and deadline. B/C use identical role profiles and graph, differing in
worker concurrency. Optional bounded verification-feedback rounds preserve one
shared JobBudget/deadline and use fresh source snapshots. This evaluator feedback
path is in-memory and not a persistent multi-round trial checkpoint. The original
DurableRecovery path remains the default when feedback rounds are disabled.
Fixed revisions are not learned replanning or RSI. A separate test exercises the
existing ReplanningController on a partially repaired multi-file case.

Local effective-request capture is opt-in, redacted and disabled by default. It
records gateway messages/tools before SDK transport, not byte-exact HTTP payloads.
It does not export source or records to Langfuse. Exact source-span retention and
read-excerpt duplication are descriptive metrics, not semantic sufficiency or proof
that a repeated investigation was unnecessary. Root-cause lexical scores remain
provisional; future live experiments require independent auditable review.
