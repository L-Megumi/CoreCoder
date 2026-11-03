# Pinned historical source-slice trials

`researchcoder.evaluation.projects.historical.load_historical_case` adapts the
two licensed fixtures in `benchmarks/historical` to existing ProjectRunInput,
ProjectCheck and ReceiptSpec contracts. Execution still uses the original
ProjectEvaluationRunner, Harness, Workers, expert exchange and verifier.

Only `faulty/` becomes the candidate. `case.json`, reference versions and the
unchanged independent Oracle remain outside its readable workspace. Both
source versions and original licenses are retained. These are historical
upstream source slices, not whole-repository benchmark results.

The frozen checks cover a bounded lazy reverse slice in more-itertools, and
backslash write/read round-trip across python-dotenv's writer and parser.
Single-component partial dotenv repairs remain negative controls. Scores
are produced by the pre-existing Oracle; the adapter adds origin/version
binding without adding private assertions to public feedback.

M16 uses the existing followup-v3 collaboration strategy. The cpx_05 rerun
is a distinct new Trial, not recovery of the previous unknown provider call.
The original 30-second API timeout remains unchanged; unknown usage retains
its reservation and stops further real calls. No automatic retries or full
regression/installation acceptance are part of this work.
