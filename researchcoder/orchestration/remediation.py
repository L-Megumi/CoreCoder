"""Finite read-only expert retries. Original failures stay in original revisions."""

from dataclasses import dataclass, replace
import time
from corecoder.workflow import workspace_revision
from researchcoder.contracts import Contract
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.runtime.tracing import TraceSink
from .scheduler import DAGScheduler
from .budget import JobBudget
from .profiles import PROFILES, RoleBudgetOverride


@dataclass(frozen=True, kw_only=True)
class RemediationPolicy(Contract):
    max_revisions: int = 1
    max_expert_retries: int = 2
    extra_tool_calls: int = 4
    extra_model_calls: int = 2
    max_total_tasks: int = 32
    schema_version: int = 1

    def validate(self):
        if not 0 <= self.max_revisions <= 2 or not 0 <= self.max_expert_retries <= 4:
            raise ValueError('Finite remediation required')
        if not 1 <= self.extra_tool_calls <= 4 or not 1 <= self.extra_model_calls <= 2 or not 1 <= self.max_total_tasks <= 64:
            raise ValueError('Remediation exceeds reviewed allowance')


@dataclass
class RemediationExecution:
    latest: object
    runs: tuple
    records: tuple
    summary_ref: object
    reason: str


def eligible_experts(plan, config, run, scheduler, budget):
    """Decide from trusted results/records, never from LLM requests for credit."""
    if budget.snapshot()['pending'] or any(s['usage_unknown'] for s in budget.snapshot()['settlements'].values()):
        return (), 'usage_unknown'
    if run.status == 'cancelled':
        return (), 'cancelled'
    if any(r.kind != 'diagnose' for r in run.results):
        return (), 'side_effect_or_verification_already_started'
    if not any(r.evidence_refs for r in run.results):
        return (), 'no_legal_evidence'
    candidates = []
    states = {s.task_id:s for s in run.states}
    for task in plan.tasks:
        state = states[task.task_id]
        if state.status in ('succeeded', 'reused', 'blocked'):
            continue
        result = next((r for r in run.results if r.task_id == state.execution_task_id), None)
        if result is None and state.reason == 'budget_exhausted':
            return (), 'global_job_allocation_exhausted'
        if (task.role not in ('log_analysis','code_analysis','diagnosis') or result is None
                or result.status != 'incomplete' or result.failure_kind not in ('local_budget_exhausted','budget_exhausted')):
            return (), 'nonremediable_failure'
        template = config.templates[task.template_id]
        if template.spec.policy.writable_paths or template.spec.policy.allow_verification:
            return (), 'not_read_only'
        record = scheduler.records.get(task.task_id, {})
        b = record.get('budget', {})
        dimension = b.get('local_limit_detail')
        if dimension not in ('tool_calls','model_calls','rounds'):
            return (), 'not_a_bounded_call_limit'
        if record.get('resulting_revision') != state.workspace_revision:
            return (), 'workspace_changed'
        # This also checks all Artifact/EvidenceRef metadata and source versions.
        scheduler.exchange.bundle([result.task_id], state.workspace_revision)
        candidates.append(task.role)
    return tuple(dict.fromkeys(candidates)), 'local_role_limit' if candidates else 'no_eligible_expert'


def execute_authorized(plan, config, *, policy, budget=None):
    if not isinstance(policy, RemediationPolicy) or config.role_overrides:
        raise ValueError('Explicit nonstacked remediation policy required')
    if len(plan.revision_id)+8>45 or plan.generation+policy.max_revisions>10:
        raise ValueError('Insufficient revision identity/ancestry space')
    budget = budget or JobBudget(config.job_limits)
    io=ArtifactIO(config.artifact_root)
    trace=TraceSink(plan.job_id,'remediation',plan.revision_id)
    trace.parent_id=trace.emit('remediation.started', reason='bounded-read-only-v1')
    start=time.monotonic();runs=[];records=[];reuse={};remedies=[];used=0
    scheduler=DAGScheduler();reason='revision_limit';current=plan;cfg=config
    for generation in range(policy.max_revisions+1):
        remaining=config.job_limits.max_runtime_seconds-(time.monotonic()-start)
        if remaining<.1:reason='timeout';break
        if runs and config.cancel_event is not None and config.cancel_event.is_set():reason='cancelled';break
        if (generation+1)*len(plan.tasks)>policy.max_total_tasks:reason='task_limit';break
        cfg=replace(cfg,job_limits=replace(config.job_limits,max_runtime_seconds=remaining))
        run=scheduler.execute(current,cfg,budget=budget,reuse=reuse);runs.append(run)
        summary=io.read_json(run.summary_ref)
        records.extend(r for r in summary['records'].values() if not r.get('reused'))
        if run.status=='succeeded':reason='completed';break
        if generation==policy.max_revisions:break
        roles,reason=eligible_experts(current,cfg,run,scheduler,budget)
        if not roles or used+len(roles)>policy.max_expert_retries:
            if roles:reason='expert_retry_limit'
            break
        config.snapshot.verify()
        # Refuse unrelated candidate edits between revisions, even when experts used a snapshot.
        if any(workspace_revision(t.spec.workspace.root)!=t.spec.workspace.source_manifest_sha256
               for t in config.templates.values()):
            reason='workspace_changed';break
        overrides=[]
        for role in roles:
            profile=PROFILES[role]
            overrides.append(RoleBudgetOverride(role=role,max_model_calls=profile.max_models+policy.extra_model_calls,
                max_tool_calls=profile.max_tools+policy.extra_tool_calls))
        snap=budget.snapshot()
        left={key:getattr(budget.limits,'max_'+key)-snap['charged'][key] for key in ('model_calls','tool_calls','total_tokens')}
        available=True
        for override in overrides:
            t=next(t for t in config.templates.values() if t.role==override.role)
            allowed=(min(override.max_model_calls,t.config.limits.max_model_calls),
                     min(override.max_tool_calls,t.config.limits.max_tool_calls),
                     min(PROFILES[override.role].max_tokens,t.config.limits.max_total_tokens))
            if any(left[key]<amount for key,amount in zip(left,allowed)):available=False
        if not available:reason='remaining_job_allocation_unavailable';break
        used+=len(roles);reuse=dict(scheduler.completed)
        parent=current.revision_id
        current=replace(plan,revision_id=plan.revision_id+'.remedy'+str(generation+1),generation=plan.generation+generation+1,
            parent_revision_id=parent,tasks=tuple(replace(t,retry_of=parent+'.'+t.task_id if t.role in roles else t.retry_of) for t in plan.tasks))
        cfg=replace(cfg,role_overrides=tuple(overrides))
        row=dict(parent_revision_id=parent,revision_id=current.revision_id,roles=list(roles),policy=policy.to_dict())
        remedies.append(row);trace.emit('remediation.revision_authorized',revision_id=current.revision_id,
                                       reason=','.join(roles),count=used)
    if not runs:raise ValueError('No admitted execution revision')
    trace.emit('remediation.finished',reason=reason,count=len(remedies))
    ref=io.put_json(plan.job_id,dict(schema_version=1,policy=policy.to_dict(),reason=reason,
        revisions=[r.to_dict() for r in runs],remedies=remedies,budget=budget.snapshot(),events=trace.events))
    return RemediationExecution(runs[-1],tuple(runs),tuple(records),ref,reason)
