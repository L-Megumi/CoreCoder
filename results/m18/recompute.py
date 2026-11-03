"""Recompute trial counts and successful case coverage without model calls."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
rows=read('trials.json')['rows'];summary=read('summary.json')
started=[r for r in rows if r['started']]
latest={r['case_id']:r for r in started}
assert len(rows)==summary['planned_trials']==10
assert len(started)==summary['started_trials']==8
assert len(rows)-len(started)==summary['unstarted_trials']==2
assert sum(r['result']['end_to_end_success'] for r in started)==summary['successes']==4
assert len(latest)==summary['unique_cases']==4
assert sum(r['result']['end_to_end_success'] for r in latest.values())==summary['latest_case_successes']==4
keys=tuple(summary['total'])
totals={k:sum(r['usage'][k] for r in started) for k in keys}
for k,v in totals.items():
    if isinstance(v,float):assert abs(v-summary['total'][k])<1e-9
    else:assert v==summary['total'][k]
assert totals['requests']==76 and totals['tokens']==851201
assert totals['unknown_requests']==totals['pending_requests']==0
for row in latest.values():
    assert row['experts'][0]['safe_to_continue'] and row['experts'][0]['selected_producers']
    assert all(row['independent_checks'][name]['passed'] for name in ('f2p','p2p'))
for name,sha in read('publication-manifest.json')['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
print(json.dumps(dict(planned=10,started=8,unstarted=2,successful_trials=4,successful_cases=4,
    requests=totals['requests'],actual_provider_tokens=totals['tokens'],all_files_sha_verified=True)))
