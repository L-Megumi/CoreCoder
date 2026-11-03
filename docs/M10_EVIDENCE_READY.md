# M10: Evidence-ready expert investigation

The opt-in `ExpertPolicy(report_version="readiness-v2")` uses the runtime
`evidence-ready-v2` policy. It is a separate policy identity; the previous
`explicit-v1` / `explicit-report-v1` path and historical experiments remain
available without relabeling their results.

## Problem and scope

The M9 real Code-expert run extracted log evidence and performed source searches.
The former four-request profile reserved two report slots, so its third model
request was forced into report-only mode. Search candidates counted as current
observations for this transition, but the unchanged diagnosis validator required
an actual usable file or path observation. No writer ran after the invalid report.

M10 addresses both the profile size and that inconsistent transition. The
reviewed new profile permits at most six model requests, six rounds and ten
tools, while retaining its Token, time, output and permission bounds. One request
is reserved for the final report; a format retry is optional and uses only
remaining known allowance. These are initial engineering settings, not an
empirically optimal allocation. The existing JobBudget remains authoritative.

## Shared evidence readiness

`corecoder.observation_state.usable_current_observation` and
`has_usable_current_observation` contain the original diagnosis predicate.
`diagnose._usable_current_observation` and `_has_current_observation` remain
import aliases for compatibility. A search candidate is still a lead, even if
its source hash is current. The accepted observation kinds, citation scope and
current-source checks have not been relaxed.

For the new version, both explicit finalization and the compact-tool fallback
use this same readiness predicate. A minimum observation does not establish a
correct diagnosis or complete cross-file coverage. The expert must still state
uncertainty and provide valid citations; independent verification determines
repair correctness.

## Bounded continuation and stopping

The new runtime thresholds distinguish a round already counted by `before_round`
from a future round. Reaching a soft finalization threshold without usable
evidence never creates a successful report. If the same ledger cannot admit one
investigation request and one final report, execution stops incomplete before
another model request is sent. Otherwise the existing authorized tools remain
available. A `diagnosis.finalization.deferred` event records this decision.

If the model voluntarily submits a draft after searching but before acquiring a
usable observation, `run_diagnosis` allows at most one normal investigation
continuation. It requires extracted evidence, a real current source-candidate
record, sufficient model/round/tool allowance, and the runtime's known-usage,
permission and error checks. The original response and checklist are retained
in `investigation_continuations`, with the reason
`missing_usable_current_observation` and whether continuation was admitted.

The continuation asks the model to inspect relevant existing search candidates
using existing tools. It takes no path from the model's draft, grants no new
authority and synthesizes no observation. A repeated early report, unavailable
budget, failed read or still-insufficient evidence remains incomplete. The
existing file/path guard remains responsible for every actual read.

## Versioned prompts and accounting

`diagnostic_output.py` defines separately hashed v2 investigation guidance and
the final-report instruction. The old v1 prompt bytes are unchanged. The report
artifact includes the actual policy version, final prompt SHA and investigation
prompt SHA. The v2 guidance distinguishes search hits from reads and permits
independent reads within one model response; dependent reads still follow the
search response that produced their paths.

All investigation, report and optional format-repair requests still pass through
the existing BudgetedModel. No stage grants additional Job credit, reconstructs a
new Job or bypasses unknown-usage stops. A format-only retry cannot make absent
facts appear and cannot execute tools. The v2 mode adds no new Agent role,
Scheduler, API provider or Oracle information.

## Validation scope

Targeted tests should preserve the historical forced-finalization path, replay
the search-only failure, demonstrate search-to-read-to-valid-report with actual
tools, and cover missing observations at the final request, early-answer
continuation, unknown usage, forbidden reads and exhausted model/round/tool
budgets. ScriptedLLM results establish mechanism behavior only. New real trials
must keep their own protocol, source identity and complete outcomes; previous
M9 and earlier trials must not be overwritten or reclassified.
