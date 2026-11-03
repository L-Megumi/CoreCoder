"""Materialize one trusted, portable historical-case config; never call a model."""
from pathlib import Path
import argparse,hashlib,json,os,sys

CASES=('packaging-requirement-hash','sam-checkpoint-resume','jsonpointer_scalar','itsdangerous_future')

def prepare(case,output,*,python=None,torch_python=None):
    if case not in CASES:raise ValueError('Unknown development case')
    root=(Path(__file__).resolve().parents[1]/'benchmarks/repositories'/case).resolve(strict=True)
    config=json.loads((root/'config-template.json').read_text(encoding='utf-8'))
    python=str(Path(python or sys.executable).resolve(strict=True))
    torch_python=str(Path(torch_python or os.environ.get('RESEARCHCODER_EXPERIMENT_PYTHON') or python).resolve(strict=True))
    def bind(v):
        if isinstance(v,dict):return {k:bind(x) for k,x in v.items()}
        if isinstance(v,list):return [bind(x) for x in v]
        if isinstance(v,str):
            if v=='@PYTHON@':return python
            if v=='@TORCH_PYTHON@':return torch_python
            if v.startswith('@CASE@/'):
                p=(root/v[len('@CASE@/'):]).resolve(strict=True)
                if not p.is_relative_to(root):raise ValueError('Case path escaped')
                return str(p)
        return v
    config=bind(config)
    from researchcoder.evaluation.projects.workspace import capture_manifest
    from researchcoder.evaluation.projects.contracts import RepositoryManifest
    from researchcoder.evaluation.projects.verification import ProjectCheck
    if capture_manifest(config['source_root'])!=RepositoryManifest.from_dict(config['task']['manifest']):
        raise ValueError('Frozen candidate source differs')
    for role in ('public_check','f2p_check','p2p_check'):
        ProjectCheck.from_dict(config[role]).check_protected()
    output=Path(output).absolute();output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2)
    return output

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--python')
    p.add_argument('--torch-python')
    args=p.parse_args(argv)
    output=prepare(args.case,args.output,python=args.python,torch_python=args.torch_python)
    print(json.dumps(dict(config=str(output),paid_model_calls=0,checks_and_source_verified=True)))
    return 0

if __name__=='__main__':raise SystemExit(main())
