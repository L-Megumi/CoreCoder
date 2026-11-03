"""Pinned source slices over real checkers/Workers; fixed actions, not accuracy."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.historical import load_historical_case
from researchcoder.evaluation.projects.verification import run_check

FIXTURES = Path(__file__).resolve().parents[1] / 'benchmarks/historical'
CASES = ('peekable_negative_slice', 'dotenv_backslash_roundtrip')


@pytest.mark.parametrize('identity', CASES)
def test_public_projection_excludes_references_and_evaluator(identity):
    root = FIXTURES / identity
    source, task, checks = load_historical_case(root, sys.executable)
    public = json.dumps(task.public_input())
    assert source.name == 'faulty' and set(checks) == {'public', 'f2p', 'p2p'}
    assert 'oracle.py' not in public and 'case.json' not in public
    metadata = json.loads((root / 'case.json').read_text())
    assert metadata['original_fixed_commit'] not in public
    assert metadata['source']['html_url'] not in public
    assert all((source / p).is_file() for p in task.writable_paths)
    assert all('oracle.py' not in e.path for e in task.manifest.entries)


@pytest.mark.parametrize('identity', CASES)
@pytest.mark.parametrize('state,profile,expected', [('faulty','f2p',False), ('healthy','f2p',True), ('faulty','p2p',True)])
def test_new_receipt_adapter_preserves_original_outcome(tmp_path,identity,state,profile,expected):
    root = FIXTURES / identity
    _, _, checks = load_historical_case(root, sys.executable)
    io = ArtifactIO(tmp_path / 'artifacts')
    record = run_check(checks[profile], root / state, io, 'historical-job', 'check')
    assert record.receipt_error is None and record.receipt['passed'] is expected
    assert record.receipt['reported_source_sha256_verified']


@pytest.mark.parametrize('state', ['writer_only', 'parser_only'])
def test_partial_component_repair_remains_failed(tmp_path,state):
    root = FIXTURES / 'dotenv_backslash_roundtrip'
    _, _, checks = load_historical_case(root, sys.executable)
    record = run_check(checks['f2p'], root / state, ArtifactIO(tmp_path/'artifacts'), 'negative-job', 'check')
    assert record.receipt_error is None and record.receipt['passed'] is False


@pytest.mark.parametrize('identity', CASES)
@pytest.mark.parametrize('target', ['oracle','source','write_scope'])
def test_tampered_oracle_source_and_write_scope_refused(tmp_path,identity,target):
    root = tmp_path / identity
    shutil.copytree(FIXTURES / identity, root)
    if target == 'oracle':
        with (root/'oracle.py').open('a') as f: f.write('\n# changed\n')
    elif target == 'source':
        p=next((root/'faulty').rglob('*.py'))
        with p.open('a') as f: f.write('\n# changed\n')
    else:
        p=root/'case.json';value=json.loads(p.read_text());value['writable_paths']=['../oracle.py'];p.write_text(json.dumps(value))
    with pytest.raises(ValueError):load_historical_case(root,sys.executable)


@pytest.mark.parametrize('identity', CASES)
def test_fixed_actions_real_expert_and_writer_pass_independent_checks(tmp_path,identity):
    from tests.test_project_runner import policy,call,requests
    from tests.test_project_consultation import expert_model
    from tests.test_m9_code_first import specialist_requests
    from tests.test_m11_repair_continuity import state_policy
    from tests.test_m10_evidence_ready import LIMITS
    from researchcoder.runtime.protocol import ModelSpec,ScriptedTurn
    from researchcoder.orchestration.budget import JobLimits
    from researchcoder.evaluation.projects.routing import ExpertPolicy
    from researchcoder.evaluation.projects.handoff import HandoffPolicy
    from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner,recompute
    root=FIXTURES/identity;source,task,checks=load_historical_case(root,sys.executable)
    turns=[]
    for index,path in enumerate(task.writable_paths):
        bad=(source/path).read_text();good=(root/'healthy'/path).read_text()
        prefix=0
        while prefix < min(len(bad),len(good)) and bad[prefix]==good[prefix]:prefix+=1
        suffix=0
        while suffix < min(len(bad),len(good))-prefix and bad[-suffix-1]==good[-suffix-1]:suffix+=1
        # Whole-line hunks keep the real edit guard and the fixed test request bounded.
        start=bad.rfind('\n',0,prefix)+1
        bad_end=bad.find('\n',len(bad)-suffix);good_end=good.find('\n',len(good)-suffix)
        bad_end=len(bad) if bad_end<0 else bad_end+1;good_end=len(good) if good_end<0 else good_end+1
        offset=max(1,bad[:start].count('\n')-3)
        turns.extend((call('read_file',dict(file_path=path,offset=offset,limit=140),'read-'+str(index)),
            call('edit_file',dict(file_path=path,old_string=bad[start:bad_end],new_string=good[start:good_end]),'edit-'+str(index))))
    turns.append(ScriptedTurn(content='Fixed offline source repair complete; inspect independent verification.',prompt_tokens=10,completion_tokens=5))
    expert=expert_model(path=task.writable_paths[0])
    read=expert.turns[1].calls[0]
    expert=replace(expert,turns=(expert.turns[0],replace(expert.turns[1],calls=(replace(read,arguments_json=json.dumps(dict(path=task.writable_paths[0],start_line=420 if identity==CASES[0] else 180,max_lines=100))),)),
        replace(expert.turns[2],content=expert.turns[2].content.replace('increment contract','public component contract'))))
    settings=policy(completion_version='explicit-v1',progress_version='progress-v1',max_feedback_rounds=0,
        task_limits=replace(policy().task_limits,max_model_calls=12,max_rounds=10,max_tool_calls=24,max_total_tokens=200000,max_output_tokens=2048,max_context_tokens=32000),
        job_limits=JobLimits(max_model_calls=24,max_tool_calls=60,max_total_tokens=400000,max_runtime_seconds=180))
    trial=tmp_path/'trial'
    result=ProjectEvaluationRunner().run(trial,task,source,checks['public'],checks['f2p'],checks['p2p'],ModelSpec(turns=tuple(turns)),policy=settings,
        expert_policy=ExpertPolicy(mode='public-failure-v1',initial_mode='code-first-v1',report_version='readiness-v2',assignment_version='focused-v1',expert_limits=LIMITS,writer_min_model_calls=12,schedule_version='serial-parallel-v2',max_workers=1),
        expert_models={'code_analysis':expert},writer_context_policy=replace(state_policy(),expert_followup_version='followup-v3'),
        handoff_policy=HandoffPolicy(mode='local-stop-v3',closure_version='reserve-incomplete-v1',request_version='explicit-request-v1'))
    io=ArtifactIO(trial/'artifacts');raw=io.read_json(ArtifactRef.from_json((trial/'raw-reference.json').read_text()))
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass'],result
    assert recompute(trial)==result and not raw['budget']['pending']
    assert len({r['worker_pid'] for r in raw['records'] if r.get('worker_pid')})==2
    visible=json.dumps(requests(raw,io)+specialist_requests(raw,io))
    assert (root/'oracle.py').read_text() not in visible
    assert json.loads((root/'case.json').read_text())['original_fixed_commit'] not in visible
