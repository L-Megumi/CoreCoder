from dataclasses import replace
import hashlib
import json
import os
import pytest
from corecoder.workflow import workspace_revision
from researchcoder.evaluation.contracts import TrialSpec
from researchcoder.evaluation.runner import build_workflow
from researchcoder.evaluation.dataset import materialize
from researchcoder.evaluation.complexity.dataset import DATA, load_manifest
from researchcoder.evaluation.complexity.protocol import offline_config
from researchcoder.evaluation.complexity.feedback import repair_model, diagnosis_model
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn
from researchcoder.orchestration.planning import ReplanningController, PlanningJob
from researchcoder.context import EvidenceStore, AccessScope
from researchcoder.context.policy import ContextPolicy
from researchcoder.context.compression import CompressionPipeline, validate_pairs
from tests.test_context_compression import messages


def test_real_middle_numeric_history_compression_restore_and_scope(tmp_path):
    case = next(c for c in load_manifest().cases if c.case_id == "cpx_03")
    project = materialize(case, tmp_path / "candidate")
    from researchcoder.adapters.artifacts import ArtifactIO
    io = ArtifactIO(tmp_path / "artifacts")
    store = EvidenceStore(io, "complex-context")
    scope = AccessScope(job_id="complex-context", task_id="log", attempt_id="a0",
                        workspace_revision=workspace_revision(project))
    raw = (project / "records/training.txt").read_bytes()
    pipeline = CompressionPipeline(store, scope, ContextPolicy(strategy="evidence"))
    original = messages(raw.decode(), True)
    view = pipeline.build(original, 4000)
    validate_pairs(view.messages)
    assert view.statistics["after_tokens"] < view.statistics["before_tokens"]
    assert "nan" in json.dumps(view.messages).lower()
    ref = next(d for d in store.catalog.documents if d.artifact.sha256 == hashlib.sha256(raw).hexdigest())
    rows = raw.decode().splitlines(keepends=True)
    middle = next(i for i, row in enumerate(rows) if '"scenario":"stress"' in row)
    restored = store.restore_document(ref, scope, start_line=middle + 1, max_lines=10)
    assert restored["content"] == "".join(rows[middle:middle + 10])
    assert restored["original_sha256"] == hashlib.sha256(raw).hexdigest()
    other = replace(scope, task_id="diagnosis", attempt_id="a1")
    with pytest.raises(ValueError):
        store.restore_document(ref, other)
    allowed = replace(other, producers=(("log", "a0"),))
    assert store.restore_document(ref, allowed, start_line=middle + 1, max_lines=1)["current_eligible"]
    (project / "pipeline.py").write_text((project / "pipeline.py").read_text() + "\n# changed source\n")
    stale = replace(allowed, workspace_revision=workspace_revision(project))
    with pytest.raises(ValueError):
        store.restore_document(ref, stale)
    assert not store.restore_document(ref, replace(stale, allow_historical=True))["current_eligible"]
    with pytest.raises(ValueError):
        store.restore_document(ref, replace(scope, job_id="foreign"))


def test_existing_replanning_controller_complex_partial_then_complete(tmp_path):
    case = next(c for c in load_manifest().cases if c.case_id == "cpx_04")
    config = offline_config(scripted_behavior="staged_correct", max_feedback_rounds=1)
    trial = TrialSpec(trial_id="trial", case_id=case.case_id, mode="parallel", repeat=0, ordinal=0)
    built = build_workflow(case, trial, tmp_path / "trial", config, os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
    cfg, plan = built["config"], built["plan"]
    partial = materialize(case, tmp_path / "expected-partial")
    (partial / "fusion.py").write_bytes((DATA / case.case_id / "healthy/fusion.py").read_bytes())
    templates = dict(cfg.templates)
    for name, template in cfg.templates.items():
        if template.role == "repair":
            templates[name] = replace(template, config=replace(template.config, model=repair_model(
                case, "staged_correct", built["project"], 0, template.config.model)))
            final_model = repair_model(case, "staged_correct", partial, 1, template.config.model)
        elif template.role in ("log_analysis", "code_analysis", "diagnosis"):
            final_model = diagnosis_model(case, "staged_correct", partial, 1, template.config.model)
            report = json.loads(final_model.turns[-1].content)
            final_model = replace(final_model, turns=final_model.turns[:-1] + (ScriptedTurn(content=json.dumps(report)),))
        else:
            final_model = template.config.model
        templates[name + "_final"] = replace(template, config=replace(template.config, model=final_model))
    cfg = replace(cfg, templates=templates)
    second = replace(plan, revision_id="r1", generation=1, parent_revision_id="r0", tasks=tuple(
        replace(t, template_id=t.template_id + "_final", retry_of="r0." + t.task_id) for t in plan.tasks))
    models = tuple(ModelSpec(turns=(ScriptedTurn(content=p.to_json()),)) for p in (plan, second))
    result = ReplanningController().execute(PlanningJob(built["job_id"], case.objective), cfg, models)
    (tmp_path / "complex-replanning.json").write_text(json.dumps(result, indent=2))
    assert result["status"] == "succeeded" and result["replans"] == 1, result
    assert [r["status"] for r in result["runs"]] == ["failed", "succeeded"]
    assert [r["results"][-1]["verification"]["status"] for r in result["runs"]] == ["failed", "passed"]
    assert all(r["accepted"] for r in result["planning_records"])
    assert not result["budget"]["pending"]
    assert result["budget"]["charged"]["model_calls"] == sum(
        s["charged"]["model_calls"] for s in result["budget"]["settlements"].values())


def test_live_workflow_construction_exposes_public_inputs_only_without_calling_api(tmp_path):
    case = next(c for c in load_manifest().cases if c.case_id == "cpx_04")
    config = replace(offline_config(), allow_live=True,
                     model=ModelSpec(provider="openai", model="deepseek-flash",
                                     base_url="https://api.deepseek.com", credential_env="DEEPSEEK_API_KEY"))
    for mode in config.modes:
        trial = TrialSpec(trial_id="trial", case_id=case.case_id, mode=mode, repeat=0, ordinal=0)
        built = build_workflow(case, trial, tmp_path / mode, config, os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
        pairs = [(built["spec"], built["config"])] if mode == "single" else [
            (t.spec, t.config) for t in built["config"].templates.values() if t.role != "verification"]
        for spec, cfg in pairs:
            assert cfg.model == config.model and not cfg.model.turns
            assert case.reference_cause not in spec.objective
            assert "Public candidate file inventory" in spec.objective
            assert "oracle.py" not in spec.objective and "reference.py" not in spec.objective
            for item in spec.inputs:
                if item.role == "raw_log":
                    assert built["io"].read_bytes(item.reference).decode() == case.public_log
        assert not (built["project"] / "oracle.py").exists()
        assert not (built["project"] / "reference.py").exists()
        verifier = built["spec"] if mode == "single" else built["config"].templates["verification"].spec
        assert verifier.verification.oracle_profile == "benchmark-discovery"


def test_single_agent_receives_latest_feedback_after_long_real_history(tmp_path):
    from researchcoder.contracts import TaskInput
    from researchcoder.runtime import ResearchHarness
    case = next(c for c in load_manifest().cases if c.case_id == "cpx_04")
    trial = TrialSpec(trial_id="trial", case_id=case.case_id, mode="single", repeat=0, ordinal=0)
    built = build_workflow(case, trial, tmp_path / "trial", offline_config(), os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
    text = (built["project"] / "records/training.txt").read_text() + "\nERROR latest validation still failed; investigate missing parameter updates.\n"
    assert 20000 < len(text) < 70000
    ref = built["io"].put_bytes(built["job_id"], text.encode(), media_type="text/plain")
    spec = replace(built["spec"], inputs=(TaskInput(role="raw_log", reference=ref),))
    cfg = replace(built["config"], model=ModelSpec(turns=(ScriptedTurn(content="Unable to repair"),)))
    harness = ResearchHarness()
    result = harness.execute(spec, cfg)
    outputs = [built["io"].read_json(a) for a in result.outputs]
    manifest = next(x for x in outputs if "request_records" in x)
    from researchcoder.contracts import ArtifactRef
    request = built["io"].read_json(ArtifactRef.from_dict(manifest["request_records"][0]))["request"]
    assert any(text in m.get("content", "") for m in request["messages"])
    assert result.status != "succeeded" and result.verification.status == "failed"
    harness.close_job(built["job_id"])
