"""Stdlib-only receipt projection; the original Oracle remains unchanged.

Failure types are public diagnostic metadata, not private seeds, expected
answers or exception bodies. Scoring follows the original Oracle aggregate.
"""
import hashlib,json,re
from pathlib import Path

PUBLIC_EXECUTION_CONTRACT = """[Public benchmark execution policy: python-static-v1]
The candidate runs under a restricted Python benchmark policy. Use ordinary
Torch/math operations and declared project modules. Do not introduce reflection
or dynamic execution (including getattr, setattr, eval, exec, compile,
__import__, globals, locals or vars), file access, undeclared imports or mutation
of the evaluator's Torch runtime. These are execution constraints, not hints
about the fault or the reference repair. Preserve the stated public API and
training objective. The independent verifier decides whether the result passes.
"""
PUBLIC_EXECUTION_CONTRACT_SHA256 = hashlib.sha256(PUBLIC_EXECUTION_CONTRACT.encode()).hexdigest()

def adapt_original_result(raw, code, project, *, kind, profile):
    if kind not in ('basic','complexity') or profile not in ('discovery','holdout','regression'):
        raise ValueError('Unknown original Oracle kind/profile')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate original Oracle field')
            result[k]=v
        return result
    value=json.loads(raw,object_pairs_hook=unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if type(code) is not int or code not in (0,1) or type(value.get('passed')) is not bool:
        raise ValueError('Original Oracle did not complete')
    failure_kind=value.get('failure_kind',value.get('error_type'))
    if failure_kind is not None and (not isinstance(failure_kind,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}',failure_kind)):
        raise ValueError('Failure type must be a bounded type name, never an exception body')
    passed=value['passed']
    if passed and failure_kind is not None:
        raise ValueError('A successful original result cannot carry a failure type')
    if (code==0)!=passed:raise ValueError('Original Oracle exit mismatch')
    rows=value.get('rows');checks=[]
    if kind=='complexity':
        if type(rows) is not list or len(rows)!=(3 if profile=='holdout' else 2):raise ValueError('Incomplete complex observations')
        for i,row in enumerate(rows):
            if type(row.get('checks')) is not dict or type(row.get('passed')) is not bool:raise ValueError('Bad original observations')
            if any(type(v) is not bool for v in row['checks'].values()) or row['passed'] is not all(row['checks'].values()):raise ValueError('Original row mismatch')
            checks.extend(dict(check=f'input-{i}:{name}',passed=result) for name,result in sorted(row['checks'].items()))
        if passed is not all(r['passed'] for r in rows):raise ValueError('Original aggregate mismatch')
    else:
        if passed and (value.get('profile')!=profile or type(rows) is not list or len(rows)!=2
                       or any(r.get('passed') is not True for r in rows)):
            raise ValueError('Incomplete successful basic observations')
        # A basic Oracle catches the first assertion and returns only an aggregate
        # on failure. This single check is that actual aggregate, not invented
        # per-assertion/seed scores. Seeds/private tensor values are not projected.
        checks=[dict(check='original_oracle_aggregate',passed=passed)]
    root=Path(project).resolve(strict=True);origin=root/('entry.py' if kind=='complexity' else 'training.py')
    if not origin.is_file() or origin.is_symlink():raise ValueError('Invalid candidate origin')
    return dict(passed=passed,checks=checks,profile=profile,oracle_exit_code=code,
        original_oracle_stdout_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        candidate_origin=str(origin),source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        receipt_adapter_version='original-oracle-receipt-v2', failure_kind=failure_kind,
        scoring_semantics='Original complete check rows' if kind=='complexity' else 'One original aggregate across the fixed Oracle profile; not per-assertion counts')

