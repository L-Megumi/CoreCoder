"""Opt-in contracts and actual expert Workers; fixed actions are not model accuracy."""
from dataclasses import replace
import json

import pytest

from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.consultation import execute_consultation
from researchcoder.evaluation.projects.routing import ExpertPolicy, decide
from researchcoder.evaluation.projects.runner import ProjectPolicy
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.protocol import RuntimeLimits, ScriptedTurn
from tests.test_project_consultation import fixture, expert_model, requests, SECRET


@pytest.mark.parametrize('cls,field', [(ExpertPolicy, 'report_version'), (ProjectPolicy, 'completion_version')])
def test_legacy_policy_serialization_omits_new_opt_in_and_roundtrips(cls, field):
    old = cls().to_dict()
    assert field not in old
    assert cls.from_dict(old).to_dict() == old
    explicit = replace(cls(), **{field: 'explicit-v1'})
    assert cls.from_json(explicit.to_json()) == explicit
    assert explicit.to_dict()[field] == 'explicit-v1'
    assert explicit.to_json() != cls().to_json()


@pytest.mark.parametrize('cls,field', [(ExpertPolicy, 'report_version'), (ProjectPolicy, 'completion_version')])
@pytest.mark.parametrize('value', ['auto', 'explicit-v2', True])
def test_unknown_strategy_cannot_silently_expand_execution(cls, field, value):
    with pytest.raises((TypeError, ValueError)):
        cls(**{field: value})


def test_explicit_reserves_are_inside_existing_limits():
    old = RuntimeOptimization(compact_reports=True)
    assert old.version == 'failure-aware-v1'
    assert RuntimeOptimization.from_json(old.to_json()) == old
    new = replace(old, version='explicit-report-v1')
    assert new.report_reserve_calls == old.report_reserve_calls == 2
    for changes in ({'compact_reports': False}, {'report_reserve_calls': 1}, {'report_reserve_calls': 3}):
        with pytest.raises(ValueError):
            replace(new, **changes)
    with pytest.raises(ValueError):
        ExpertPolicy(report_version='explicit-v1', expert_limits=replace(ExpertPolicy().expert_limits, max_model_calls=1))
    with pytest.raises(ValueError):
        ProjectPolicy(completion_version='explicit-v1', task_limits=RuntimeLimits(max_tool_calls=1))


def repairing_model(*, unknown=False, forged=False):
    model = expert_model()
    bad = ScriptedTurn(content='<｜DSML｜calls><｜DSML｜invoke name="read_project_file">module.py</｜DSML｜invoke></｜DSML｜calls>',
        prompt_tokens=0 if unknown else 10, completion_tokens=0 if unknown else 5)
    final = model.turns[-1]
    if forged:
        value = json.loads(final.content)
        value['current_state']['observation_ids'] = ['forged-local-observation']
        final = replace(final, content=json.dumps(value))
    return replace(model, turns=(*model.turns[:2], bad, final))


@pytest.mark.parametrize('explicit', [False, True])
def test_actual_expert_reuses_strict_repair_without_extra_job_credit(tmp_path, explicit):
    args = fixture(tmp_path)
    args['model'] = repairing_model()
    args['policy'] = replace(args['policy'], report_version='explicit-v1' if explicit else 'legacy-v1')
    args['decision'] = decide(args['features'], args['policy'])
    value = execute_consultation(**args)
    result = value.run.results[0]
    assert result.workflow_completed is explicit
    assert value.safe_to_continue is explicit
    assert value.stop_reason == (None if explicit else 'consultation_no_evidence')
    assert value.records[0]['budget']['model_calls'] == (4 if explicit else 3)
    assert value.budget_after['charged']['model_calls'] == (6 if explicit else 5)
    assert not value.budget_after['pending']
    assert not any(s['usage_unknown'] for s in value.budget_after['settlements'].values())
    docs = [args['io'].read_json(ref) for ref in result.outputs]
    reports = [doc for doc in docs if 'report_attempts' in doc]
    assert len(reports) == 1
    report = reports[0]
    assert len(report['report_attempts']) == (2 if explicit else 1)
    assert 'DSML' in report['report_attempts'][0]['response']
    assert report['report_attempts'][0]['validation_error']
    assert report['metrics']['report_repairs'] == int(explicit)
    captured = requests(value, args['io'])
    assert SECRET not in json.dumps(captured)
    # Only the two original investigation tools ran; textual pseudo-calls never execute.
    assert len(report['tool_events']) == 2
    if explicit:
        assert report['report_attempts'][-1]['validation_error'] is None
        assert value.selected_producers == (result.task_id,)
        assert all(item.reference.producer_task_id == result.task_id for item in value.bundle.items)


@pytest.mark.parametrize('mode', ['unknown', 'forged'])
def test_actual_expert_preserves_unknown_liability_and_illegal_citation_failure(tmp_path, mode):
    args = fixture(tmp_path)
    args['model'] = repairing_model(unknown=mode == 'unknown', forged=mode == 'forged')
    args['policy'] = replace(args['policy'], report_version='explicit-v1')
    args['decision'] = decide(args['features'], args['policy'])
    value = execute_consultation(**args)
    assert not value.safe_to_continue and not value.selected_producers
    assert not value.run.results[0].workflow_completed
    assert value.records[0]['budget']['model_calls'] == (3 if mode == 'unknown' else 4)
    if mode == 'unknown':
        assert value.stop_reason == 'accounting_unknown'
        assert any(s['usage_unknown'] for s in value.budget_after['settlements'].values())
    else:
        assert value.stop_reason == 'consultation_no_evidence'
        assert value.run.results[0].failure_kind == 'invalid_report'


def test_combined_versions_keep_real_worker_handoff_and_independent_checks(tmp_path):
    from researchcoder.context.policy import ContextPolicy
    from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
    from researchcoder.runtime.protocol import ModelSpec
    from tests.test_project_m4_ablation import (
        fixture as project_fixture, policy, partial_writer, writer, expert,
        archived, assert_integrity, SIMULATED_USAGE,
    )
    source, task, checks = project_fixture(tmp_path)
    final = ScriptedTurn(content=json.dumps(dict(status='completed', summary='Inspect the authorized public check.')),
                         **SIMULATED_USAGE)
    # This declaration is deliberately insufficient: the partial patch fails the public check.
    first = ModelSpec(turns=(*partial_writer().turns[:2], final))
    repair = ModelSpec(turns=(*writer().turns[:2], final))
    investigation = expert()
    specialist = replace(investigation, turns=(*investigation.turns[:2],
        ScriptedTurn(content='<｜DSML｜calls>unexecuted pseudo-tool</｜DSML｜calls>', **SIMULATED_USAGE),
        investigation.turns[-1]))
    target = tmp_path / 'trial'
    result = ProjectEvaluationRunner().run(target, task, source,
        checks['public'], checks['f2p'], checks['p2p'], first,
        policy=replace(policy(model_calls=3), completion_version='explicit-v1'),
        feedback_models=(repair,),
        expert_policy=ExpertPolicy(mode='public-failure-v1', schedule_version='serial-parallel-v2',
                                   max_workers=2, report_version='explicit-v1'),
        expert_models={'code_analysis': specialist, 'log_analysis': specialist},
        writer_context_policy=ContextPolicy(strategy='evidence', role='repair', allow_recovery=True,
                                            coding_source_mode='observed-source-v2'))
    raw, io, recorded = archived(target)
    assert_integrity(result, raw, io, recorded, source, task, checks)
    assert recompute(target) == result
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass']
    assert raw['rounds'][0]['result']['status'] == 'failed'
    assert raw['rounds'][0]['result']['failure_kind'] == 'verification_failed'
    assert len(raw['rounds']) == 2 and len(raw['consultations']) == 1
    assert len(raw['consultations'][0]['selected_producers']) == 2
    assert raw['budget']['charged']['model_calls'] == 14
    assert any('Structured expert and public feedback' in json.dumps(r['request'])
               for r in recorded if r['task_id'] == 'coding-1')
