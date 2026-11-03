"""Independent synthetic checks for two pinned upstream historical defects.

No network, real tokens, wall-clock manipulation or external services are used.
Checks and reference source stay outside the Agent candidate.
"""
import copy,json
from pathlib import Path
import sys


def evaluate(project,identity,profile):
    root=Path(project).resolve(strict=True);rows=[]
    def check(name,fn):
        try:fn();rows.append(dict(check=name,passed=True))
        except Exception as error:rows.append(dict(check=name,passed=False,error_type=type(error).__name__))
    def raises(kind,fn):
        try:fn()
        except kind as error:return error
        raise AssertionError('Expected typed exception was absent')
    if identity=='jsonpointer_scalar':
        sys.path.insert(0,str(root));import jsonpointer as jp
        if profile in ('public','f2p'):
            examples=[({'value':'abc'},'/value/0','function')] if profile=='public' else [
                ({'data':{'message':'abcdef'}},'/data/message/1','function'),
                ('unicode-文字','/2','object'),
                ({'x':['left','right']},'/x/1/0','object'),
                ({'nested':['',{'text':'xyz'}]},'/nested/1/text/2','function')]
            for i,(document,pointer,api) in enumerate(examples):
                def invalid(document=document,pointer=pointer,api=api):
                    original=copy.deepcopy(document)
                    raises(jp.JsonPointerException,lambda:jp.resolve_pointer(document,pointer) if api=='function' else jp.JsonPointer(pointer).resolve(document))
                    assert document==original
                check('reject_scalar_'+str(i),invalid)
        else:
            examples=[({'0':'mapped'},'/0','mapped'),({'a':[10,20]},'/a/1',20),
                ({'a/b':{'~value':3}},'/a~1b/~0value',3),({'':7},'/',7),({'x':'text'},'/x','text')]
            for i,(document,pointer,expected) in enumerate(examples):
                def valid(document=document,pointer=pointer,expected=expected):
                    assert jp.resolve_pointer(document,pointer)==expected
                    assert jp.JsonPointer(pointer).resolve(document)==expected
                check('legal_resolution_'+str(i),valid)
            check('leading_zero_refused',lambda:raises(jp.JsonPointerException,lambda:jp.resolve_pointer([1,2],'/01')))
            def default():
                sentinel=object();assert jp.resolve_pointer({},'/absent',default=sentinel) is sentinel
            check('missing_default',default)
    elif identity=='itsdangerous_future':
        sys.path.insert(0,str(root/'src'))
        from itsdangerous import TimestampSigner,SignatureExpired,BadSignature
        class ClockedSigner(TimestampSigner):
            now=2000000000
            def get_timestamp(self):return self.now
        def signer(clock=2000000000,salt='synthetic-salt'):
            value=ClockedSigner('public-test-key-only',salt=salt);value.now=clock;return value
        if profile in ('public','f2p'):
            examples=[(1,False)] if profile=='public' else [(7,False),(121,True),(3601,False)]
            for i,(offset,return_timestamp) in enumerate(examples):
                def future(offset=offset,return_timestamp=return_timestamp):
                    s=signer();signed=s.sign(b'payload');s.now-=offset
                    error=raises(SignatureExpired,lambda:s.unsign(signed,max_age=60,return_timestamp=return_timestamp))
                    assert error.payload==b'payload'
                    assert error.date_signed==s.timestamp_to_datetime(2000000000)
                check('future_age_'+str(i),future)
        else:
            for age in (0,17,60):
                def valid(age=age):
                    s=signer();signed=s.sign(b'payload');s.now+=age
                    assert s.unsign(signed,max_age=60)==b'payload'
                check('valid_age_'+str(age),valid)
            def expired():
                s=signer();signed=s.sign(b'payload');s.now+=61
                raises(SignatureExpired,lambda:s.unsign(signed,max_age=60))
            check('old_signature_expired',expired)
            def unbounded():
                s=signer();signed=s.sign(b'payload');s.now-=17
                assert s.unsign(signed,max_age=None)==b'payload'
            check('unbounded_compatibility',unbounded)
            def stamp():
                s=signer();signed=s.sign(b'payload');s.now+=2
                assert s.unsign(signed,max_age=60,return_timestamp=True)==(b'payload',s.timestamp_to_datetime(2000000000))
            check('timestamp_return',stamp)
            def tamper():
                s=signer();signed=s.sign(b'payload')
                raises(BadSignature,lambda:s.unsign(b'changed'+signed,max_age=60))
            check('tampered_signature_refused',tamper)
    else:raise ValueError('Unknown frozen case')
    return dict(case_id=identity,profile=profile,passed=all(r['passed'] for r in rows),checks=rows)


if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[3] not in ('public','f2p','p2p'):raise ValueError('Candidate/case/profile required')
    value=evaluate(*sys.argv[1:]);print(json.dumps(value,allow_nan=False));raise SystemExit(0 if value['passed'] else 1)
