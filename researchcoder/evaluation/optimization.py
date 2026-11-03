"""Trusted opt-in strategy mapping; never inspect gold causes/repair paths."""

from dataclasses import replace
from pathlib import Path
import hashlib
from researchcoder.runtime.optimization import runtime_optimization
from researchcoder.runtime.protocol import ModelSpec, ProtectedFile
from researchcoder.contracts import TaskSpec,ExecutionPolicy,VerificationSpec
from researchcoder.orchestration import TaskTemplate,OrchestrationTask,TaskDependency


def configure(built,case,config,torch_python):
    strategy=config.harness_strategy
    if strategy=='fixed-v1' or 'plan' not in built:
        return built
    cfg=built['config'];runtime=runtime_optimization(strategy)
    templates={k:replace(t,config=replace(t.config,optimization=runtime)) for k,t in cfg.templates.items()}
    tasks=list(built['plan'].tasks)
    # Explicit supported public API convention, not a case-ID/gold-answer rule.
    inventory={a.path for a in case.assets}
    if runtime.public_training_constraints and {'entry.py','settings.py'}<=inventory:
        from researchcoder.verification import public_training
        script=Path(public_training.__file__).resolve()
        command=VerificationSpec(argv=(str(torch_python),'-B',str(script),'--project',str(built['project']),
            '--factory','entry:create_model','--optimizer','settings:optimizer','--baseline',cfg.snapshot.workspace.root),timeout_seconds=30,
            oracle_profile='public-joint-training-ownership-v1')
        prototype=templates['verification']
        spec=replace(prototype.spec,task_id='public-contract',objective='Check declared joint-training public parameter ownership',verification=command)
        worker=replace(prototype.config,model=ModelSpec(),model_options=None,permitted_verifications=(command,),
            protected_files=prototype.config.protected_files+(ProtectedFile(path=str(script),sha256=hashlib.sha256(script.read_bytes()).hexdigest()),)
                +tuple(ProtectedFile(path=str(Path(cfg.snapshot.workspace.root)/name),sha256=digest) for name,digest in cfg.snapshot.manifest))
        templates['public_contract']=TaskTemplate('verification',spec,worker,'candidate')
        original=next(t for t in tasks if t.task_id=='verify')
        tasks[tasks.index(original)]=replace(original,dependencies=original.dependencies+(TaskDependency(task_id='public_contract'),))
        tasks.insert(-1,OrchestrationTask(task_id='public_contract',template_id='public_contract',role='verification',
            objective=spec.objective,dependencies=(TaskDependency(task_id='repair'),)))
    built['plan']=replace(built['plan'],tasks=tuple(tasks))
    built['config']=replace(cfg,templates=templates)
    return built
