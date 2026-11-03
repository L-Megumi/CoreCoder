"""Opt-in joint-training ownership contract using public constructors only.

Run with an existing CPU Torch interpreter. No benchmark IDs, reference patches,
private inputs or external commands are loaded by this checker.
"""

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import sys

POLICY_VERSION = 'trainable-optimizer-ownership-v1'


def public_callable(root, reference):
    if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', reference):
        raise ValueError('Invalid public API reference')
    module, name = reference.split(':')
    path = root.joinpath(*module.split('.')).with_suffix('.py')
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Public API must resolve inside candidate')
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError('Public API is not callable')
    if not Path(inspect.getsourcefile(function)).resolve().is_relative_to(root):
        raise ValueError('Cached/imported public callable belongs to another source')
    return function, path


def check(root, factory='entry:create_model', optimizer='settings:optimizer', *, baseline=None, construct_only=False):
    import torch
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError('Candidate directory required')
    # Torch is imported before adding untrusted candidate module names.
    sys.path.insert(0,str(root))
    try:
        make, source=public_callable(root,factory)
        opt, opt_source=public_callable(root,optimizer)
        paths={source,opt_source};before={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        parameters=inspect.signature(make).parameters
        booleans=[(n,p.default) for n,p in parameters.items() if type(p.default) is bool]
        if len(booleans)>4:raise ValueError('Too many public boolean constructor options')
        variants=[{}]+[{name:not default} for name,default in booleans]
        results=[]
        for kwargs in variants:
            model=make(**kwargs).cpu()
            if not isinstance(model,torch.nn.Module):raise ValueError('Factory must return torch.nn.Module')
            named=dict(model.named_parameters());trainable={id(p) for p in named.values() if p.requires_grad}
            if construct_only:
                results.append(dict(public_constructor_options=kwargs,passed=True,
                    parameter_trainability={n:p.requires_grad for n,p in named.items()}))
                continue
            options={'lr':.03} if 'lr' in inspect.signature(opt).parameters else {}
            optim=opt(model,**options)
            owned=[p for group in optim.param_groups for p in group['params']]
            ids=[id(p) for p in owned];names={id(p):n for n,p in named.items()}
            checks=dict(all_explicitly_trainable_owned=trainable<=set(ids),no_frozen_or_foreign_owned=set(ids)<=trainable,
                        no_duplicate_parameters=len(ids)==len(set(ids)),has_trainable_parameters=bool(trainable))
            results.append(dict(public_constructor_options=kwargs,checks=checks,passed=all(checks.values()),
                parameter_trainability={n:p.requires_grad for n,p in named.items()},
                trainable_names=[n for n,p in named.items() if p.requires_grad],owned_names=[names.get(i,'<foreign>') for i in ids]))
        after={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        if before!=after:raise ValueError('Public source changed during component check')
        value = dict(schema_version=1,policy_version=POLICY_VERSION,passed=all(r['passed'] for r in results),rows=results,
                    source_sha256=before,scope='Opt-in exact trainable-parameter ownership for declared joint training; not all optimizer policies')
    finally:
        sys.path.pop(0)
        for name,module in list(sys.modules.items()):
            file=getattr(module,'__file__',None)
            if file and Path(file).resolve().is_relative_to(root):
                del sys.modules[name]
    if baseline is not None:
        original=check(baseline,factory,optimizer,construct_only=True)
        by_options={json.dumps(r['public_constructor_options'],sort_keys=True):r for r in original['rows']}
        for row in value['rows']:
            prior=by_options.get(json.dumps(row['public_constructor_options'],sort_keys=True))
            row['checks']['public_trainability_preserved']=bool(prior and prior['parameter_trainability']==row['parameter_trainability'])
            row['passed']=all(row['checks'].values())
        value['passed']=all(r['passed'] for r in value['rows'])
        value['baseline_source_sha256']=original['source_sha256']
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True)
    parser.add_argument('--factory',default='entry:create_model')
    parser.add_argument('--optimizer',default='settings:optimizer')
    parser.add_argument('--baseline')
    args=parser.parse_args(argv)
    try:result=check(args.project,args.factory,args.optimizer,baseline=args.baseline)
    except Exception as exc:
        # Candidate exceptions can contain private strings; retain a type only.
        result=dict(schema_version=1,policy_version=POLICY_VERSION,passed=False,error_type=type(exc).__name__,scope='Public API unsupported or check could not run')
    print(json.dumps(result,allow_nan=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
