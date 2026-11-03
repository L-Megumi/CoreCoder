"""Behavioral F2P/P2P evaluator; keep outside the Agent candidate workspace.

Inputs differ from the upstream reproduction. No expected source text, patch,
fixed version, model response or reference implementation is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True)
parser.add_argument('--profile', choices=('f2p', 'p2p', 'all'), default='all')
args = parser.parse_args()
project = Path(args.project).resolve()
src = project / 'src'
if not src.is_dir():
    raise SystemExit('Missing candidate src directory')
sys.path.insert(0, str(src))
import packaging
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
for name, module in tuple(sys.modules.items()):
    if name == 'packaging' or name.startswith('packaging.'):
        if not Path(module.__file__).resolve().is_relative_to(src):
            raise RuntimeError('Candidate import escaped src: ' + name)

def equivalent(left, right):
    a, b = Requirement(left), Requirement(right)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert {a: 17}.get(b) == 17
    assert {b: 19}.get(a) == 19

def different(left, right):
    a, b = Requirement(left), Requirement(right)
    assert a != b
    assert len({a, b}) == 2
    mapping = {a: 17, b: 19}
    assert mapping[a] == 17 and mapping[b] == 19

def rendering():
    text = 'Demo_Pkg[alpha,beta]>=2.4.0; python_version < "3.12"'
    req = Requirement(text)
    assert str(req) == 'Demo_Pkg[alpha,beta]>=2.4.0; python_version < "3.12"'
    assert req.extras == {'alpha', 'beta'}
    assert req.name == 'Demo_Pkg'
    assert req == Requirement(str(req))
    assert req.specifier.contains('2.5')
    assert not req.specifier.contains('2.3')
    assert req.marker.evaluate({'python_version': '3.11'})
    assert not req.marker.evaluate({'python_version': '3.12'})

def serialization():
    req = Requirement('sample_pkg[one]>=3.2; os_name == "posix"')
    restored = pickle.loads(pickle.dumps(req))
    assert restored == req
    assert hash(restored) == hash(req)
    assert str(restored) == str(req)

def invalid():
    for text in ('', 'demo=>2.1', 'demo[', 'demo; python_version = "3.10"'):
        try:
            Requirement(text)
        except InvalidRequirement:
            pass
        else:
            raise AssertionError('Malformed requirement accepted: ' + text)

checks = []
f2p = [
    ('release-padding', lambda: equivalent('widget==2.4', 'widget==2.4.0.0')),
    ('lower-bound-padding', lambda: equivalent('widget>=7.2.0', 'widget>=7.2')),
    ('upper-bound-padding', lambda: equivalent('widget<=5.9', 'widget<=5.9.0')),
    ('name-extras-marker', lambda: equivalent('Demo_Pkg[one,two]>=2.0; python_version<"3.12"', 'demo-pkg[two,one]>=2.0.0; python_version < "3.12"')),
    ('compound-range', lambda: equivalent('tool>=2.1,<4.0', 'tool<4.0.0,>=2.1.0')),
]
p2p = [
    ('canonical-name', lambda: equivalent('Some_Pkg==2.7', 'some-pkg==2.7')),
    ('different-name', lambda: different('alpha==2', 'beta==2')),
    ('different-extras', lambda: different('demo[a]>=2', 'demo[b]>=2')),
    ('different-marker', lambda: different('demo; python_version<"3.11"', 'demo; python_version>="3.11"')),
    ('different-url', lambda: different('demo@https://example.test/a.whl', 'demo@https://example.test/b.whl')),
    ('compatible-boundary', lambda: different('demo~=2.4', 'demo~=2.4.0')),
    ('operator-preserved', lambda: different('demo>2.1', 'demo>=2.1')),
    ('render-parse-membership', rendering),
    ('pickle-roundtrip', serialization),
    ('invalid-inputs', invalid),
]
for profile, cases in (('f2p', f2p), ('p2p', p2p)):
    if args.profile not in (profile, 'all'):
        continue
    for name, function in cases:
        try:
            function()
            checks.append({'profile': profile, 'check': name, 'passed': True})
        except Exception as exc:
            checks.append({'profile': profile, 'check': name, 'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)})
result = {
    'oracle_version': 'packaging-requirement-hash-v1',
    'candidate_import': str(Path(packaging.__file__).resolve()),
    'candidate_requirements_sha256': hashlib.sha256((src/'packaging'/'requirements.py').read_bytes()).hexdigest(),
    'checks': checks,
    'passed': all(check['passed'] for check in checks),
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['passed'] else 1)
