from dataclasses import replace
import json,threading
import pytest
from tests.test_orchestration_static import setup_graph,call,report_turn
from researchcoder.runtime.protocol import ModelSpec,RuntimeLimits,ScriptedCall,ScriptedTurn
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.orchestration import PlanRevision,OrchestrationTask,TaskDependency
from researchcoder.orchestration.budget import JobBudget,JobLimits
from researchcoder.orchestration.planning import ReplanningController
from researchcoder.orchestration.remediation import RemediationPolicy


def graph(tmp_path,*,known=True):
    _,cfg,io=setup_graph(tmp_path)
    extract=call('extract','extract_log_evidence')
    burst=ScriptedTurn(calls=tuple(ScriptedCall(call_id='read'+str(i),name='read_project_file',arguments_json='{"path":"training.py"}') for i in range(7)))
    log=ModelSpec(turns=(extract,burst,call('last','read_project_file',{'path':'training.py'}),report_turn()))
    code=ModelSpec(turns=(extract,call('read','read_project_file',{'path':'training.py'}),report_turn()))
    def usage(model):
        return replace(model,turns=tuple(replace(t,prompt_tokens=100 if known else 0,completion_tokens=10 if known else 0) for t in model.turns))
    base=cfg.templates['read']
    limits=RuntimeLimits(max_model_calls=12,max_tool_calls=16,max_rounds=12,max_runtime_seconds=20,max_total_tokens=150000)
    spec=replace(base.spec,budget=replace(base.spec.budget,max_tool_calls=16,max_rounds=12),diagnostic_options=replace(base.spec.diagnostic_options,completion_mode='off'))
    def template(role,model):
        return replace(base,role=role,spec=spec,config=replace(base.config,limits=limits,model=usage(model),optimization=RuntimeOptimization()))
    cfg=replace(cfg,templates={'log':template('log_analysis',log),'code':template('code_analysis',code),'verify':cfg.templates['verify']},
        role_profiles=True,job_limits=JobLimits(max_model_calls=48,max_tool_calls=128,max_total_tokens=1000000,max_runtime_seconds=120))
    plan=PlanRevision(job_id='job',revision_id='r0',tasks=(
        OrchestrationTask(task_id='log',template_id='log',role='log_analysis'),
        OrchestrationTask(task_id='code',template_id='code',role='code_analysis'),
        OrchestrationTask(task_id='verify',template_id='verify',role='verification',dependencies=(TaskDependency(task_id='log'),TaskDependency(task_id='code')))))
    return plan,cfg,io


def test_real_workers_new_revision_reuses_successful_expert_and_keeps_failure(tmp_path):
    plan,cfg,io=graph(tmp_path);budget=JobBudget(cfg.job_limits)
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy(),budget=budget)
    assert value.latest.status=='succeeded' and len(value.runs)==2
    assert value.runs[0].status=='incomplete'
    original=next(r for r in value.runs[0].results if r.task_id=='r0.log')
    assert original.status=='incomplete' and original.failure_kind=='local_budget_exhausted'
    assert original.evidence_refs and original.workflow_completed is False
    retry=next(r for r in value.runs[1].results if r.task_id.endswith('.log'))
    assert retry.status=='succeeded' and retry.attempt_id!=original.attempt_id
    assert next(s for s in value.latest.states if s.task_id=='code').status=='reused'
    summary=io.read_json(value.summary_ref)
    assert len(summary['remedies'])==1 and summary['budget']==budget.snapshot()
    assert not budget.pending and len(budget.settlements)==4
    assert len({r['attempt_id'] for r in value.records})==4
    assert budget.charged['model_calls']==sum(r['budget']['model_calls'] for r in value.records)
    assert all(r['worker_pid'] for r in value.records)


def test_unknown_usage_retains_debt_and_refuses_remediation(tmp_path):
    plan,cfg,io=graph(tmp_path,known=False);budget=JobBudget(cfg.job_limits)
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy(),budget=budget)
    assert value.latest.status!='succeeded' and len(value.runs)==1 and value.reason=='usage_unknown'
    assert budget.charged['total_tokens']>0 and any(r['usage_unknown'] for r in budget.settlements.values())


def test_no_global_credit_is_created_for_remediation(tmp_path):
    plan,cfg,io=graph(tmp_path)
    # Only one expert: its first allocation fits, the authorized retry cannot.
    plan=replace(plan,tasks=tuple(replace(t,dependencies=(TaskDependency(task_id='log'),)) if t.task_id=='verify' else t
                                  for t in plan.tasks if t.task_id!='code'))
    cfg=replace(cfg,job_limits=replace(cfg.job_limits,max_model_calls=9))
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy())
    assert len(value.runs)==1 and value.reason=='remaining_job_allocation_unavailable'
    assert io.read_json(value.summary_ref)['budget']['charged']['model_calls']<=9


def test_original_global_admission_denial_is_distinct_from_local_limit(tmp_path):
    plan,cfg,io=graph(tmp_path)
    cfg=replace(cfg,job_limits=replace(cfg.job_limits,max_model_calls=9))
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy())
    assert len(value.runs)==1 and value.reason=='global_job_allocation_exhausted'
    assert io.read_json(value.summary_ref)['budget']['charged']['model_calls']<=9


def test_explicit_zero_remedies_leaves_original_failure(tmp_path):
    plan,cfg,io=graph(tmp_path)
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy(max_revisions=0))
    assert len(value.runs)==1 and value.latest.status=='incomplete'


def test_precancelled_graph_is_archived_and_not_retried(tmp_path):
    plan,cfg,io=graph(tmp_path);cancel=threading.Event();cancel.set()
    value=ReplanningController().execute_authorized(plan,replace(cfg,cancel_event=cancel),policy=RemediationPolicy())
    assert value.latest.status=='cancelled' and len(value.runs)==1 and not value.records


def test_workspace_changed_between_revisions_is_not_silently_reused(tmp_path,monkeypatch):
    from pathlib import Path
    import researchcoder.orchestration.remediation as module
    plan,cfg,io=graph(tmp_path);old=module.eligible_experts
    def mutate(*args):
        decision=old(*args)
        (Path(cfg.templates['verify'].spec.workspace.root)/'training.py').write_text('unrelated = 2\n')
        return decision
    monkeypatch.setattr(module,'eligible_experts',mutate)
    value=ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy())
    assert len(value.runs)==1 and value.reason=='workspace_changed'
