"""Reproducible A0..A4 fixed-action ablation; never constructs a live model.

Run in a dedicated controller process. All candidate workspaces, independent
Oracles, budget accounting, evidence exchange and Workers are real; model
behavior and known token usage are synthetic fixture inputs, not API outcomes.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation import runner
from researchcoder.evaluation.complexity.protocol import offline_config
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.optimization import STRATEGIES
from .fixtures import decorate, SCENARIOS

VERSION='failure-aware-fixed-actions-v1'


def summarize(directory):
    """Recompute solely from saved raw results, with integrity checks."""
    directory=Path(directory)
    summary=runner.recompute(directory)
    io=ArtifactIO(directory/'trials/trial-0000/artifacts')
    result=io.read_json(ArtifactRef.from_json((directory/'records/trial-0000.json').read_text()))
    raw=io.read_json(ArtifactRef.from_dict(result['raw']))
    counts=Counter()
    output_counts=Counter()
    trace_counts=Counter()
    for t in raw.get('task_results',[]):
        counts[t['failure_kind']]+=bool(t['failure_kind'])
        for ref in t.get('outputs',[]):
            data=io.read_json(ArtifactRef.from_dict(ref))
            for attempt in data.get('report_attempts',[]):
                reason=attempt.get('failure_kind')
                if reason:output_counts[reason]+=1
                elif attempt.get('validation_error'):output_counts['legacy_unclassified_report_error']+=1
    for record in raw.get('records',[]):
        ref=record.get('trace')
        if ref:
            for event in io.read_json(ArtifactRef.from_dict(ref))['events']:
                trace_counts[event['event_type']]+=1
    states=[]
    graph_summary_hashes=[]
    for ref in raw.get('harness_optimization',{}).get('remediation_refs',[]):
        for rev in io.read_json(ArtifactRef.from_dict(ref))['revisions']:
            states.extend(rev['states'])
    if not states and raw.get('scheduler_result'):states=raw['scheduler_result']['states']
    # Existing Scheduler always archives its complete state summary, including
    # blocked tasks. Read those actual states, not a count inferred from results.
    if not states:
        for path in io.root.rglob('*.blob'):
            if path.is_symlink():raise ValueError('Symlink in archived graph summaries')
            try:value=json.loads(path.read_text())
            except (ValueError,UnicodeError):continue
            if isinstance(value,dict) and {'plan','states','budget','records'}<=value.keys():
                if value['plan']['job_id']!=raw['job_id']:raise ValueError('Foreign graph summary')
                states.extend(value['states'])
                graph_summary_hashes.append(dict(storage_key=path.relative_to(io.root).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    local_stops=sum(r.get('budget',{}).get('stop_kind')=='budget_exhausted' for r in raw.get('records',[]))
    return dict(status=result['status'],failure_kind=result['failure_kind'],metrics=result['metrics'],
        failures=dict((k,v) for k,v in counts.items() if k),report_failures=dict(output_counts),
        local_budget_stops=local_stops,
        blocked_dependencies=sum(s['status']=='blocked' for s in states),graph_summary_hashes=graph_summary_hashes,
        budget=raw.get('budget'),remediation_count=raw.get('harness_optimization',{}).get('remediation_count',0),
        feedback_retries=raw.get('feedback_retries',0),report_reservations=trace_counts['diagnosis.report_reserved'],
        diagnostics=raw.get('complexity_diagnostics'),raw_ref=result['raw'],
        recomputed_recorded=summary['recorded'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--torch-python',default=os.environ.get('RESEARCHCODER_EXPERIMENT_PYTHON'))
    p.add_argument('--suite',choices=('controls','anchors','all'),default='all')
    p.add_argument('--recompute',action='store_true')
    args=p.parse_args(argv)
    root=args.output_dir.resolve()
    if args.recompute:
        protocol=json.loads((root/'protocol.json').read_text());rows=[]
        for row in protocol['planned']:
            path=root/row['id']
            rows.append(dict(row,started=path.exists(),result=summarize(path) if (path/'records/trial-0000.json').exists() else None))
        print(json.dumps(dict(version=VERSION,rows=rows),indent=2));return 0
    if not args.torch_python:p.error('An existing CPU Torch interpreter is required')
    root.mkdir(parents=True,exist_ok=False)
    variants=[('single','fixed-v1')]+[('parallel',s) for s in STRATEGIES]
    planned=[]
    if args.suite in ('controls','all'):
        for scenario,case in SCENARIOS.items():
            for mode,strategy in variants if scenario in ('role_quota','report_length','ownership') else variants[1:]:
                planned.append(dict(scenario=scenario,case_id=case,mode=mode,strategy=strategy))
    if args.suite in ('anchors','all'):
        for case in ('shape_transpose','cpx_03','cpx_04'):
            for mode,strategy in variants:planned.append(dict(scenario='unmodified_script',case_id=case,mode=mode,strategy=strategy))
    for i,row in enumerate(planned):row['id']='ablation-'+str(i).zfill(3)
    protocol=dict(version=VERSION,planned=planned,paid_requests=0,job_limits=JobLimits(max_model_calls=48,
        max_tool_calls=128,max_total_tokens=1000000,max_runtime_seconds=240).to_dict(),
        usage_semantics='Synthetic fixture usage, not actual provider Usage; cost remains null',
        input_policy='development only; unchanged Benchmark and independent hidden holdout/regression Oracle profiles',
        model_policy='Common conditional fixture per scenario; short-report and finalization responses are modeled assumptions',
        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    original=runner.build_workflow;rows=[]
    for row in planned:
        scenario=row['scenario']
        def builder(*a,**kw):
            value=original(*a,**kw)
            return decorate(value,scenario) if scenario!='unmodified_script' else value
        runner.build_workflow=builder
        try:
            config=offline_config(case_ids=(row['case_id'],),modes=(row['mode'],),harness_strategy=row['strategy'],
                max_feedback_rounds=1,job_limits=JobLimits.from_dict(protocol['job_limits']),
                benchmark_subset='pytorch' if row['case_id']=='shape_transpose' else 'complexity_benchmark')
            path=root/row['id']
            runner.EvaluationRunner().run(path,config,torch_python=args.torch_python)
            result=summarize(path)
            rows.append(dict(row,started=True,result=result))
            (root/(row['id']+'.json')).write_text(json.dumps(rows[-1],indent=2))
            print(json.dumps(dict(id=row['id'],scenario=scenario,strategy=row['strategy'],status=result['status'],
                e2e=result['metrics']['end_to_end_success'])),flush=True)
        finally:runner.build_workflow=original
    (root/'ablation-summary.json').write_text(json.dumps(dict(version=VERSION,planned=len(planned),recorded=len(rows),rows=rows),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
