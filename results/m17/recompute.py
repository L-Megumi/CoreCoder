"""Recompute public aggregates without model calls or project dependencies."""
from pathlib import Path
from collections import defaultdict
import json

def recompute(root):
    def read(name): return json.loads((root/name).read_text(encoding='utf-8'))
    context = read('context-records.json')
    assert context['provider_requests'] == 0 and len(context['rows']) == 40
    groups = defaultdict(list)
    for row in context['rows']: groups[(row['arm'],row['strategy'])].append(row)
    arms = []
    for (arm,strategy),rows in sorted(groups.items()):
        expected = sum(r['expected_count'] for r in rows)
        retained = sum(r['retained_count'] for r in rows)
        before = sum(r['before_tokens'] for r in rows)/len(rows)
        after = sum(r['after_tokens'] for r in rows)/len(rows)
        arms.append(dict(arm=arm,strategy=strategy,cases=len(rows),expected=expected,retained=retained,
            evidence_retention=retained/expected,mean_before_tokens=before,mean_after_tokens=after,
            reduction_from_uncompressed=1-after/before,
            construction_failures=sum(r['error'] is not None for r in rows),
            recovered_documents=sum(q['restored_documents'] for r in rows for q in r['receipts']),
            build_seconds=sum(r['build_seconds'] for r in rows),
            fixed_rule_probe_passed=sum(bool(r['answer_oracle_passed']) for r in rows),
            mean_duplicate_ratio=sum(r['duplicate_context_ratio'] for r in rows)/len(rows)))
    controls = read('oracle-controls.json')['checks']
    assert len(controls) == 24
    for row in controls:
        expected_pass = row['state'] in ('healthy','repaired') or (row['state']=='faulty' and row['profile']=='p2p') or (row['state']=='negative' and row['profile']=='f2p')
        # Negative controls retain the public failure probe as well as F2P; P2P must reject the bypass.
        if row['state']=='negative' and row['profile']=='public': expected_pass=True
        assert row['result']['passed'] is expected_pass
        assert (row['returncode']==0) is expected_pass
    scripted = read('scripted-e2e.json')
    assert len(scripted['rows']) == 4 and scripted['paid_api_calls'] == 0
    scripted_pass = sum(r['result']['end_to_end_success'] and r['result']['fail_to_pass'] and r['result']['pass_to_pass'] for r in scripted['rows'])
    real = read('real-trials.json')
    assert len(real['trials']) == 4 and sum(r['started'] for r in real['trials']) == 1
    assert real['unknown_requests'] == 1 and real['actual_provider_tokens'] is None
    assert real['stop_reason']=='usage_unknown'
    assert [r['status'] for r in real['trials']].count('cancelled_before_start')==3
    assert real['trials'][0]['result']['fail_to_pass'] is None
    return dict(schema_version=1,context=arms,
        oracle_controls=dict(total=len(controls),matched_expected_outcome=len(controls)),
        scripted=dict(total=4,passed=scripted_pass,paid_api_calls=0,model_accuracy_claim=False),
        real=dict(planned=4,started=1,archived=1,completed_repairs=0,cancelled_after_start=1,
            cancelled_before_start=3,model_requests=1,tool_calls=0,actual_provider_tokens=None,
            unknown_requests=1,unknown_liability=real['unknown_liability'],stop_reason='usage_unknown'),
        full_regression=False,installation_acceptance=False,current_pytest_nodes_run=0)

if __name__=='__main__':
    root = Path(__file__).resolve().parent
    value = recompute(root)
    recorded = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert value == recorded, 'Saved summary differs from recomputed records'
    print(json.dumps(value,ensure_ascii=False,indent=2))
