# Phase 6C-3 engineering closeout

The 271-file R2 baseline was verified against its saved SHA256 manifest before
changes. A new source/worktree snapshot was extracted into an independent
location and all 271 files matched. Historical modified/untracked work and R1/R2
runs were preserved; no Git staging, commit, push or history-changing operation
was performed. A complete candidate list and private-path review are part of the
separate delivery audit; they are not a claim that the repository is already
ready for public release.

R2 tool-schema intersection and tool.rejected batch-stop propagation passed 18
targeted offline tests. One permitted shape_transpose / Evidence Context live
calibration used the unchanged R2 source, model deepseek-flash, non-thinking,
2048 output tokens. It remained incomplete / budget_exhausted: diagnosis tasks
completed, but Repair spent its six-call allocation after unnecessary/missing
file reads. Outer F2P/P2P checks passed the resulting candidate, which does not
make the unfinished workflow successful. All 16 HTTP responses had Usage:
69,406 input and 5,409 output tokens, 43.49 seconds, 15 tool calls. This is one
separate R2 calibration, never a replacement for a failed R1 row or a new 12/12.

The independently rechecked diagnosis names training.py / predict / transpose,
correctly explaining the batch/class layout issue. Original report validation,
producer-scoped references and Artifact digests were independently rechecked.
The patch includes an extra higher-rank fallback not exhaustively verified by the
original small Oracle; no broader correctness is claimed.

At official peak rates checked on 2026-10-26, complete cache-aware Usage gives an
estimated $0.02373756; no invoice was verified. Conservative old-scope admission
stands at 229/288 HTTP and $0.3526389/$1.50, retaining historical unknown Usage.
These remaining allowances do not authorize new complexity cases. See
[official pricing](https://api-docs.deepseek.com/quick_start/pricing/).

Subsequent offline complexity work found and repaired three integration issues:
context projection could move user data before the trusted system registry;
fixed ScriptedLLM placeholders could select compressor-derived IDs; the single
coding baseline sliced long task logs at 20K characters and could hide appended
verification feedback. The production validator remains strict. The existing
ReplanningController now attaches an observed failure envelope to JSON-only
verification feedback, retaining original output bytes and source versions.
No paid evaluation was run on this final extended source.

Final Linux regression, installed-wheel checks, source manifest, complete Git
candidate inventory and raw logs are supplied in the paired Phase 6D delivery.
MIT license and upstream attribution remain intact. No RSI or Phase 7 policy
optimizer was implemented.
