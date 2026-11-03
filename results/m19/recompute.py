"""Recompute the frozen public numeric projections without model/network calls."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json

def recompute(root):
    root=Path(root)
    manifest=json.loads((root/'publication-manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Manifest path escaped')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Digest mismatch: '+name)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'trials').glob('*.json'))]
    counts=Counter();cases=defaultdict(lambda:Counter())
    for row in rows:
        counts['planned']+=row['planned'];counts['started']+=row['started'];counts['archived']+=row['archived']
        ok=row['result']['end_to_end_success'];counts['successes']+=ok
        cases[row['case_id']]['planned']+=1;cases[row['case_id']]['successes']+=ok
        reqs=row['request_records'];counts['requests']+=len(reqs)
        if len(reqs)!=row['actual_provider_requests']:raise ValueError('Request count mismatch')
        usage=Counter()
        for request in reqs:
            if request['status']!='known' or not request['usage_complete']:raise ValueError('Unknown provider Usage')
            usage.update(request['usage'])
        if dict(usage)!=row['actual_provider_usage']:raise ValueError('Usage mismatch')
        for key,value in usage.items():counts[key]+=value
        counts['f2p']+=row['result']['fail_to_pass'] is True
        counts['p2p']+=row['result']['pass_to_pass'] is True
        counts['unknown_runtime_settlements']+=sum(s['usage_unknown'] for s in row['runtime_budget']['settlements'].values())
    expected=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    for key in ('planned','started','archived','successes'):
        if counts[key]!=expected[key]:raise ValueError('Summary mismatch: '+key)
    for key in ('requests','prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens','f2p','p2p'):
        if counts[key]!=expected['totals'][key]:raise ValueError('Metric mismatch: '+key)
    return dict(**dict(counts),cases={k:dict(v) for k,v in cases.items()},
                e2e_rate=counts['successes']/counts['planned'],model_calls_made_by_recompute=0)

if __name__=='__main__':
    print(json.dumps(recompute(Path(__file__).resolve().parent),ensure_ascii=False,indent=2))
