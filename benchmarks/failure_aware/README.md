# Failure-aware Harness v1

Opt-in strategies: `fixed-v1` (A0), `budget-v1` (A1), `compact-v1` (A2),
`constraints-v1` (A3), `combined-v1` (A4). Single Agent remains a separate
architecture reference. The ordinary CLI/EvaluationConfig defaults to A0.

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m benchmarks.failure_aware.run_offline \
  --torch-python /path/to/existing/cpu-torch/python \
  --output-dir /new/isolated/ablation --suite all
python -B -m benchmarks.failure_aware.run_offline \
  --output-dir /new/isolated/ablation --recompute
```

No API key or live provider option exists in this launcher. `offline_config`
rejects live models, and the fixture Worker checks the provider before making
a client. Do not use this standalone controller concurrently in the same
Python process: its temporary builder selection is scoped to that process.

The controls compare the same conditional scripted model policy across each
strategy on unchanged development cases (`cpx_03`, `cpx_04`). The public
ownership fixture batches the same independent reads for **every** strategy.
Known-action positive controls exercise local role exhaustion, length-limited
JSON and public parameter-ownership feedback. Negative controls inject unknown
Usage (including after an actual edit), forged citations and forbidden tools.
Anchors additionally run the original scripted actions on `shape_transpose`,
`cpx_03`, `cpx_04`. All use Job limits 48 model calls / 128 tool calls /
1,000,000 estimated/reserved Tokens / 240 seconds, output limit 2,048 and at most
one ordinary verification-feedback round. Unused credit never becomes a new Job.

Workers, file tools, Evidence Exchange, local citation validator, JobBudget,
public Verifier and final independent F2P/P2P Oracles actually execute. Models
and Usage values are synthetic fixtures. Tool-aware finalization, compact
responses and correct second patches are **assumptions of these engineering
controls**, not learned behavior or measured real-LLM gains. Cost is null.
Reference patches belong only to the scripted fixture; no private Oracle
inputs are used by the production public checker. The final independent Oracle
is never fed back to repair. Ordinary failures and safety stops remain rows.

Raw records retain original failed attempts, format errors, every budget
settlement and reference. Recompute reads those records without model calls.
Exact source-span retention and repeated-read equality are proxy measurements;
they do not establish semantic sufficiency or unnecessary work. Latency on a
shared host is not a throughput benchmark.

Production `trainable-optimizer-ownership-v1` is explicitly for **declared
joint training with exact ownership**, not a universal rule for all optimizers.
The evaluation adapter opts in only to projects with the public
`entry:create_model` / `settings:optimizer` convention. Constructor booleans are
checked individually (not every combination). Original public constructor
trainability is protected against freeze-to-pass changes. This check does not
prove gradient correctness; the unchanged independent Oracle still decides
F2P/P2P.

Read-only remediation creates bounded PlanRevisions through the existing
ReplanningController/DAGScheduler. Completed source-valid expert results can
be reused; incomplete results never become succeeded. Unknown Usage, unknown
effects, write/verification already started, invalid evidence, global admission
denial or insufficient remaining Job allocation stop automatic remediation.
Retries are fresh bounded investigations, not restoration of a live message
history. The new multi-round/remediation evaluation path has no durable
trial-level resume claim; the original single-revision durable path is retained.
