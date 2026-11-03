"""Public behavioral reproduction; no reference repair is provided."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True)
args = parser.parse_args()
project = Path(args.project).resolve()
src = project / 'src'
if not src.is_dir():
    raise SystemExit('Missing candidate src directory')
sys.path.insert(0, str(src))
import packaging
from packaging.requirements import Requirement
for name, module in tuple(sys.modules.items()):
    if name == 'packaging' or name.startswith('packaging.'):
        if not Path(module.__file__).resolve().is_relative_to(src):
            raise RuntimeError('Candidate import escaped src: ' + name)
left = Requirement('foo==1.0.0')
right = Requirement('foo==1.0.0.0')
result = {
    'candidate_import': str(Path(packaging.__file__).resolve()),
    'equal': left == right,
    'equal_hash': hash(left) == hash(right),
    'set_size': len({left, right}),
    'dict_lookup': {left: 'available'}.get(right),
}
print(json.dumps(result, indent=2))
assert result['equal'] is True, 'Semantically equivalent requirements must compare equal'
assert result['equal_hash'] is True, 'Equal requirements must have equal hashes'
assert result['set_size'] == 1, 'Equivalent requirements must deduplicate in sets'
assert result['dict_lookup'] == 'available', 'Equivalent requirement must retrieve the same mapping value'
