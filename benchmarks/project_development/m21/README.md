# M21 prospective development fixtures

These two original, standard-library Python cases were authored after the M20
policy was frozen. They are controlled development problems, separate from the
four pinned open-source historical defects and from validation/holdout. Neither
case is evidence of real-repository generalization. No policy, prompt or
candidate adjustment is required to use them.

| Case | Public symptom | Failure-to-pass objective | Pass-to-pass properties |
| --- | --- | --- | --- |
| `stream_utf8_chunks` | Valid UTF-8 spanning a read boundary raises `UnicodeDecodeError`. | Exact text for single-byte and arbitrary binary partitions, including empty chunks. | Strict errors on invalid/truncated input, empty input, incremental delivery. |
| `stable_shard_merge` | A tie in timestamps attempts to compare dictionary payloads. | Stable merge ordering across and within shards, preserving opaque payloads. | Unambiguous ordering, empty shards, single-pass input with bounded lookahead. |

Both frozen negative controls fix the reported defect by buffering all input.
They pass the public reproducer and both F2P checks but violate the original
streaming constraint. The frozen healthy reference passes all six checks.
The faulty source passes all three P2P checks and fails the public check and
both F2P checks. Results are machine-checked separately from model trials.

On Linux run `prepare.py --runtime RUNTIME --base-config BASE --output NEW`
with the accepted ResearchCoder interpreter. BASE is the frozen M20 Code-first
JSON Pointer config. The program accepts a fresh output directory and
materializes one config per case. Only `source/` (one faulty implementation)
is exposed to the Agent. The three trusted check entry points are distinct,
outside the candidate. Each check binds the imported module path and digest
to the candidate and produces the runner's standard checks receipt. The public
checker includes only the public test. Healthy/negative references and hidden
inputs stay with the evaluator. Isolation relies on the existing tool/access
guards, subject to the existing Python execution trust model.

Preparation validates the frozen configs and executes 18 independent checker
processes before any model run. The portable `--checks-only` option runs the
same fixture controls without importing the Linux runtime. `identities.json`
records each profile's outcome, case and checker hashes. Output directories
contain host-local paths and are for private execution records.
