"""Independent CPU numerical checks; keep outside model-visible candidate roots.

Reference updates follow SAM's published perturb-then-SGD equation. No source
inspection, fixed-patch import, or candidate internal pointer identity decides pass.
The scope uses momentum=0 and weight_decay=0; historical SAM momentum checkpoints
are a different problem not repaired by upstream PR 39.
"""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import torch


def load_sam(project):
    root = Path(project).resolve()
    path = (root / 'sam.py').resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError('Invalid candidate sam.py')
    spec = importlib.util.spec_from_file_location('candidate_sam_independent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Wrong import origin')
    return module.SAM, str(path)


def reference(weights, targets, scales, rates, rho, adaptive):
    grads = [a*(w-t) for w,t,a in zip(weights, targets, scales)]
    norm = torch.stack([((w.abs() if adaptive else 1.0)*g).norm() for w,g in zip(weights,grads)]).norm()
    result = []
    for w,t,a,g,lr in zip(weights, targets, scales, grads, rates):
        perturb = (w.square() if adaptive else 1.0) * g * (rho/(norm+1e-12))
        result.append(w - lr*a*(w+perturb-t))
    return result


def step(parameters, optimizer, targets, scales):
    def loss():
        return sum(0.5*(s*(p-t).square()).sum() for p,t,s in zip(parameters,targets,scales))
    optimizer.zero_grad()
    loss().backward()
    def closure():
        optimizer.zero_grad()
        current=loss()
        current.backward()
        return current
    optimizer.step(closure)


def assert_weights(actual, expected, message):
    if len(actual) != len(expected):
        raise AssertionError(message+': parameter count')
    for parameter, ref in zip(actual,expected):
        if not parameter.requires_grad:
            raise AssertionError(message+': trainable parameter was frozen')
        if not torch.isfinite(parameter).all():
            raise AssertionError(message+': non-finite update')
        if not torch.allclose(parameter.detach(),ref,atol=1e-10,rtol=1e-10):
            raise AssertionError(message+': numeric update differs')


def run_case(SAM, name, *, resume, adaptive=False, rho=0.07, gamma=0.6, multi=True):
    initial = [torch.tensor([1.7,-0.8],dtype=torch.float64)]
    targets = [torch.tensor([0.2,0.5],dtype=torch.float64)]
    scales = [torch.tensor([0.4,1.3],dtype=torch.float64)]
    rates = [0.08]
    if multi:
        initial.append(torch.tensor([-1.2,0.4,2.1],dtype=torch.float64))
        targets.append(torch.tensor([0.3,-0.2,0.9],dtype=torch.float64))
        scales.append(torch.tensor([0.7,1.1,0.2],dtype=torch.float64))
        rates.append(0.025)
    parameters=[torch.nn.Parameter(w.clone()) for w in initial]
    groups=[{'params':[p], 'lr':lr} for p,lr in zip(parameters,rates)]
    optimizer=SAM(groups,torch.optim.SGD,lr=0.2,momentum=0.0,weight_decay=0.0,rho=rho,adaptive=adaptive)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
    expected=[w.clone() for w in initial]
    current_rates=list(rates)
    # Warm up with genuine optimizer steps and changed scheduler state.
    for _ in range(2):
        expected=reference(expected,targets,scales,current_rates,rho,adaptive)
        step(parameters,optimizer,targets,scales)
        assert_weights(parameters,expected,'uninterrupted reference before checkpoint')
        scheduler.step()
        current_rates=[lr*gamma for lr in current_rates]
    if resume:
        buf=io.BytesIO()
        torch.save({'weights':[p.detach().clone() for p in parameters],
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},buf)
        buf.seek(0)
        checkpoint=torch.load(buf,weights_only=True,map_location='cpu')
        resumed=[torch.nn.Parameter(w.clone()) for w in checkpoint['weights']]
        restored=SAM([{'params':[p],'lr':0.33+0.11*i} for i,p in enumerate(resumed)],
                     torch.optim.SGD,lr=0.33,momentum=0.0,weight_decay=0.0,rho=rho,adaptive=adaptive)
        restored.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        resumed_scheduler=torch.optim.lr_scheduler.ExponentialLR(restored,gamma=gamma)
        resumed_scheduler.load_state_dict(checkpoint['scheduler'])
    for iteration in range(3):
        expected=reference(expected,targets,scales,current_rates,rho,adaptive)
        step(parameters,optimizer,targets,scales)
        assert_weights(parameters,expected,'uninterrupted reference')
        if resume:
            step(resumed,restored,targets,scales)
            assert_weights(resumed,expected,'resumed reference at step '+str(iteration))
            assert_weights(resumed,[p.detach() for p in parameters],'resume equivalence')
        scheduler.step()
        if resume:
            resumed_scheduler.step()
        current_rates=[lr*gamma for lr in current_rates]
    if all(torch.allclose(w,p.detach(),atol=1e-12,rtol=1e-12) for w,p in zip(initial,parameters)):
        raise AssertionError('No effective training update')
    return {'case':name,'passed':True,'parameters':[p.detach().tolist() for p in parameters],
            'resume_checked':resume,'adaptive':adaptive,'groups':len(parameters),'steps':5}


def main(profile=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True)
    parser.add_argument('--profile',choices=('f2p','p2p','all'),default=profile or 'all')
    args=parser.parse_args()
    profile=args.profile
    torch.set_num_threads(1)
    SAM,origin=load_sam(args.project)
    f2p_cases = [('restore_multigroup',dict(resume=True,multi=True)),
              ('restore_adaptive',dict(resume=True,adaptive=True,multi=False,gamma=0.8)),
              ('restore_zero_rho',dict(resume=True,rho=0.0,multi=True,gamma=0.5))]
    p2p_cases = [('normal_multigroup',dict(resume=False,multi=True)),
              ('normal_adaptive',dict(resume=False,adaptive=True,multi=False,gamma=0.8)),
              ('zero_rho_sgd',dict(resume=False,rho=0.0,multi=True,gamma=0.5))]
    cases = f2p_cases if profile=='f2p' else p2p_cases if profile=='p2p' else f2p_cases+p2p_cases
    rows=[]
    for name,options in cases:
        try:
            rows.append(run_case(SAM,name,**options))
        except Exception as exc:
            rows.append({'case':name,'passed':False,'error_type':type(exc).__name__,'error':str(exc)[:500]})
    passed=all(row['passed'] for row in rows)
    print(json.dumps({'schema_version':1,'profile':profile,'origin':origin,'passed':passed,'checks':rows},sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
