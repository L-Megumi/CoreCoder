"""Independent stdlib-only checks for two pinned upstream historical defects.

Only synthetic values are used. This is not a hostile-code sandbox. Candidate
processes are bounded by the external runner's timeout; source/Oracle identities
are recorded outside the editable repository slice.
"""
import json
from pathlib import Path
import sys,tempfile


def run(project,identity,profile):
    root=Path(project).resolve(strict=True)
    rows=[]
    def check(name,fn):
        try:fn();rows.append(dict(check=name,passed=True))
        except Exception as error:rows.append(dict(check=name,passed=False,error_type=type(error).__name__))
    if identity=='peekable_negative_slice':
        sys.path.insert(0,str(root));import more_itertools
        if profile=='p2p':
            for index,sl in enumerate((slice(1,8,2),slice(None,5,None),slice(-3,None,None),slice(-1,None,-1),slice(8,1,-2))):
                def existing(sl=sl):
                    p=more_itertools.peekable(range(17))
                    assert p[sl]==list(range(17))[sl]
                    assert next(p)==0
                check('existing_slice_'+str(index),existing)
        else:
            cases=((5,1),) if profile=='public' else ((2,1),(7,2),(11,3),(8,4))
            for start,step in cases:
                def lazy(start=start,step=step):
                    produced=[]
                    def source():
                        for value in range(1000):
                            produced.append(value)
                            if len(produced)>start+1:
                                raise AssertionError('Bounded backwards slice consumed beyond its required prefix')
                            yield value
                    p=more_itertools.peekable(source())
                    assert p[slice(start,None,-step)]==list(range(start+1))[slice(start,None,-step)]
                    assert len(produced)==start+1 and next(p)==0
                check('bounded_prefix_'+str(start)+'_'+str(step),lazy)
    elif identity=='dotenv_backslash_roundtrip':
        sys.path.insert(0,str(root/'src'));import dotenv
        if profile=='p2p':values=['ordinary','',"apostrophe's",'two words','line\nvalue','中文']
        elif profile=='public':values=['Q:\\workspace\\trial']
        else:values=['K:\\logs\\trial_42','suffix'+'\\','\\A[a-z]+\\Z',"v\\'q",'\\\\network\\share\\']
        for index,value in enumerate(values):
            def roundtrip(value=value):
                with tempfile.TemporaryDirectory(prefix='researchcoder-synthetic-') as directory:
                    path=Path(directory)/'synthetic-values.txt';path.write_text('EXISTING=keep\n')
                    assert dotenv.set_key(path,'ROUNDTRIP',value)[0] is True
                    dotenv.set_key(path,'AFTER','sentinel')
                    assert dotenv.get_key(path,'ROUNDTRIP')==value
                    assert dotenv.get_key(path,'AFTER')=='sentinel'
                    assert dotenv.get_key(path,'EXISTING')=='keep'
            check('synthetic_roundtrip_'+str(index),roundtrip)
    else:raise ValueError('Unknown fixed historical case')
    return dict(case_id=identity,profile=profile,passed=all(r['passed'] for r in rows),checks=rows)


if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[3] not in ('public','f2p','p2p'):raise ValueError('Case/profile required')
    result=run(*sys.argv[1:]);print(json.dumps(result,allow_nan=False));raise SystemExit(0 if result['passed'] else 1)
