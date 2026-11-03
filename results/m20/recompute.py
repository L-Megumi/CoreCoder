"""Recompute frozen M20 projections without any model or network calls."""
from collections import Counter
from pathlib import Path
import hashlib,json

def recompute(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'publication-manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Invalid publication record: '+name)
    total=Counter()
    for p in sorted((root/'trials').glob('*.json')):
        row=json.loads(p.read_text(encoding='utf-8'));usage=Counter()
        total['planned']+=row['planned'];total['started']+=row['started'];total['archived']+=row['archived']
        total['successes']+=row['result']['end_to_end_success']
        for request in row['request_records']:
            if request['status']!='known' or not request['usage_complete']:raise ValueError('Unknown Usage')
            usage.update(request['usage'])
        if dict(usage)!=row['actual_provider_usage']:raise ValueError('Usage mismatch')
        total['requests']+=len(row['request_records']);total['total_tokens']+=usage['total_tokens']
        if row['budget']['pending'] or any(s['usage_unknown'] for s in row['budget']['settlements'].values()):
            raise ValueError('Unresolved responsibility')
    expected=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    for k in ('planned','started','archived','successes'):
        if total[k]!=expected[k]:raise ValueError('Summary mismatch')
    for k in ('requests','total_tokens'):
        if total[k]!=expected['totals'][k]:raise ValueError('Metric mismatch')
    return dict(total,model_calls_made_by_recompute=0)

if __name__=='__main__':print(json.dumps(recompute(Path(__file__).parent),indent=2))
