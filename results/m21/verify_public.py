"""Check result file hashes and recompute aggregates without contacting any provider."""
from pathlib import Path
import hashlib,json
from recompute import aggregate

here=Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_bytes())
for n,h in manifest.items():
    if Path(n).name!=n or hashlib.sha256((here/n).read_bytes()).hexdigest()!=h:
        raise ValueError('Mismatched public record')
rows=json.loads((here/'trials.json').read_bytes())
report=json.loads((here/'summary.json').read_bytes())
assert report['total']==aggregate(rows)
assert report['groups']=={g:aggregate([r for r in rows if r['arm']==g]) for g in {r['arm'] for r in rows}}
print(json.dumps(dict(trials=len(rows),end_to_end=report['total']['e2e_successes'],verified=True)))
