from dataclasses import replace
import hashlib,json
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskResult
from researchcoder.orchestration.evidence import EvidenceExchange,EvidenceContextCapacityError

def fixture(tmp_path):
    io=ArtifactIO(tmp_path/'artifacts')
    raw=dict(evidence=[dict(id=f'ev_{i}',source='training.log',line_no=i+1,end_line_no=i+1,
        message=('unicode evidence \\" 中文\n'*150)) for i in range(40)],observations=[],diagnosis=None)
    artifact=io.put_json('job',raw)
    refs=io.index_evidence(artifact,task_id='expert',attempt_id='attempt',workspace_revision='a'*64)
    result=TaskResult(job_id='job',task_id='expert',attempt_id='attempt',kind='diagnose',
        status='succeeded',workflow_completed=True,outputs=(artifact,),evidence_refs=refs)
    ex=EvidenceExchange('job',io);ex.submit(result,'a'*64)
    return ex,ex.bundle(('expert',),'a'*64),io,artifact,raw

def test_opt_in_projection_preserves_every_reference_and_original_record(tmp_path):
    ex,bundle,io,artifact,raw=fixture(tmp_path)
    before=io.read_bytes(artifact)
    with pytest.raises(EvidenceContextCapacityError):ex.context(bundle)
    encoded=ex.context(bundle,max_chars=40000,compact_excerpts=True)
    assert len(encoded)<=40000
    view=json.loads(encoded)
    assert len(view['observations'])==40
    assert [r['reference'] for r in view['observations']]==[i.reference.to_dict() for i in bundle.items]
    assert [r['exchange_id'] for r in view['observations']]==[i.exchange_id for i in bundle.items]
    assert io.read_bytes(artifact)==before
    for i,item in enumerate(bundle.items):assert io.read_record(item.reference)==raw['evidence'][i]
    assert ex.context(bundle,max_chars=40000,compact_excerpts=True)==encoded

def test_metadata_that_cannot_fit_is_explicit_capacity_error(tmp_path):
    ex,bundle,_,_,_=fixture(tmp_path)
    with pytest.raises(EvidenceContextCapacityError,match='reference metadata'):
        ex.context(bundle,max_chars=1000,compact_excerpts=True)

def test_compaction_still_rejects_forged_producer_job_and_corrupted_artifact(tmp_path):
    ex,bundle,io,artifact,_=fixture(tmp_path)
    for forged in (replace(bundle,job_id='foreign-job',items=(),hypothesis_artifacts=()),
        replace(bundle,items=(replace(bundle.items[0],exchange_id='forged'),))):
        with pytest.raises(ValueError):ex.context(forged,compact_excerpts=True)
    path=io.resolve(artifact);path.write_bytes(b'{}')
    with pytest.raises(ValueError):ex.context(bundle,compact_excerpts=True)

@pytest.mark.parametrize('args',[{'max_chars':50001},{'max_chars':True},{'compact_excerpts':'yes'}])
def test_model_shaped_or_unbounded_view_options_rejected(tmp_path,args):
    ex,bundle,_,_,_=fixture(tmp_path)
    with pytest.raises((ValueError,TypeError)):ex.context(bundle,**args)
