"""Dataset self-acceptance, independent of any Agent/model success rate."""

from dataclasses import replace
import hashlib
import json
import os
import pytest
from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.evaluation.contracts import EvaluationConfig
from researchcoder.evaluation.dataset import load_manifest, materialize, public_input
from researchcoder.evaluation.complexity.contracts import ComplexityManifest
from researchcoder.evaluation.complexity.dataset import DATA, load_complexity
from researchcoder.evaluation.verification import verify_candidate


def case(identity):
    return next(c for c in load_manifest("complexity_benchmark").cases if c.case_id == identity)


def check(c, project, io, task, profile):
    result = verify_candidate(c, project, os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"], profile, io, "dataset-check", task)
    assert result.verification.checked_workspace_revision == workspace_revision(project)
    return result.verification


@pytest.mark.parametrize("identity", [f"cpx_{i:02d}" for i in range(1, 7)])
def test_health_fault_correct_repair_and_p2p(identity, tmp_path):
    c = case(identity)
    io = ArtifactIO(tmp_path / "artifacts")
    healthy = materialize(c, tmp_path / "healthy", healthy=True)
    faulty = materialize(c, tmp_path / "faulty")
    assert check(c, healthy, io, "healthy-f2p", "holdout").status == "passed"
    assert check(c, healthy, io, "healthy-p2p", "regression").status == "passed"
    assert check(c, faulty, io, "fault-f2p", "holdout").status == "failed"
    assert check(c, faulty, io, "fault-p2p", "regression").status == "passed"
    for path in c.writable_paths:
        (faulty / path).write_bytes((DATA / identity / "healthy" / path).read_bytes())
    assert check(c, faulty, io, "repair-f2p", "holdout").status == "passed"
    assert check(c, faulty, io, "repair-p2p", "regression").status == "passed"


@pytest.mark.parametrize("identity", ["cpx_04", "cpx_05", "cpx_06"])
def test_partial_repair_changes_observations_but_does_not_succeed(identity, tmp_path):
    c = case(identity)
    label = next(x for x in load_complexity().cases if x.case_id == identity)
    project = materialize(c, tmp_path / "candidate")
    io = ArtifactIO(tmp_path / "artifacts")
    before = check(c, project, io, "before", "holdout")
    for path in label.stages[0].patch_paths:
        (project / path).write_bytes((DATA / identity / "healthy" / path).read_bytes())
    after = check(c, project, io, "partial", "holdout")
    assert after.status == "failed" and check(c, project, io, "partial-p2p", "regression").status == "passed"
    initial_rows = json.loads(before.stdout.text)["rows"]
    partial_rows = json.loads(after.stdout.text)["rows"]
    assert sum(sum(r["checks"].values()) for r in partial_rows) > sum(sum(r["checks"].values()) for r in initial_rows)
    assert before.checked_workspace_revision != after.checked_workspace_revision


@pytest.mark.parametrize("identity", [f"cpx_{i:02d}" for i in range(1, 7)])
def test_constant_result_and_frozen_factory_cannot_bypass_training(identity, tmp_path):
    c = case(identity)
    io = ArtifactIO(tmp_path / "artifacts")
    constant = materialize(c, tmp_path / "constant", healthy=True)
    (constant / "entry.py").write_text((constant / "entry.py").read_text() + '\ndef step(model, batch, lr=0.03):\n    return {"loss": 0.0}\n')
    assert check(c, constant, io, "constant", "holdout").status == "failed"
    frozen = materialize(c, tmp_path / "frozen", healthy=True)
    (frozen / "entry.py").write_text((frozen / "entry.py").read_text() +
        '\noriginal_factory = create_model\ndef create_model(**kwargs):\n    model = original_factory(**kwargs)\n    for p in model.parameters():\n        p.requires_grad_(False)\n    return model\n')
    assert check(c, frozen, io, "frozen", "holdout").status == "failed"


def test_surface_nan_patch_and_oracle_import_are_not_repairs(tmp_path):
    c = case("cpx_03")
    io = ArtifactIO(tmp_path / "artifacts")
    for name, patch in (("surface", 'import torch\ndef compute(output, target):\n    return torch.nan_to_num(torch.nn.functional.mse_loss(output, target))\n'),
                        ("oracle-import", 'import os\ndef compute(output, target):\n    return 0.0\n')):
        project = materialize(c, tmp_path / name)
        (project / "objective.py").write_text(patch)
        assert check(c, project, io, name, "holdout").status == "failed"
    project = materialize(c, tmp_path / "readonly", healthy=True)
    (project / "README.md").write_text("The task is now solved")
    assert check(c, project, io, "goal-tamper", "holdout").status == "failed"


def test_complexity_dimensions_are_pretrial_version_bound_and_private():
    labels = load_complexity()
    assert ComplexityManifest.from_json(labels.to_json()) == labels
    assert [sum(c.level == level for c in labels.cases) for level in (1, 2, 3)] == [3, 3, 3]
    assert len(load_manifest().cases) == 8
    assert len(load_manifest("complexity_benchmark").cases) == 6
    assert {c.split for c in load_manifest("complexity_benchmark").cases} == {"development", "validation", "holdout"}
    for label in labels.cases:
        assert label.label_basis.endswith("before_agent_trials")
        assert label.N_files == len(label.source_files) and label.N_dependencies == len(label.dependencies)
        with pytest.raises(ValueError):
            replace(label, N_dependencies=label.N_dependencies + 1)
    for c in load_manifest("complexity_benchmark").cases:
        public = public_input(c)
        assert set(public) == {"objective", "log", "source_files"}
        assert c.reference_cause not in json.dumps(public) and c.case_id not in c.public_log
        assert len(c.writable_paths) > len(c.fault_locations)  # allowed paths are not gold labels
    config = EvaluationConfig()
    assert "benchmark_subset" not in config.to_dict() and "max_feedback_rounds" not in config.to_dict()
    assert EvaluationConfig.from_json(config.to_json()) == config
    with pytest.raises(ValueError):
        replace(config, max_feedback_rounds=3)
    with pytest.raises(ValueError):
        load_manifest("../../outside")


def test_real_log_provenance_and_middle_stress_observations():
    metadata = json.loads((DATA / "generation.json").read_text())
    for identity, record in metadata.items():
        assert record["device"] == "cpu" and record["controlled_repetitions"] == 48
        assert record["public_seed_range"] == [10001, 10048]
        for name in ("training", "validation"):
            lines = (DATA / identity / "faulty/records" / (name + ".txt")).read_text().splitlines()
            rows = [json.loads(line) for line in lines]
            assert {row["repeat"] for row in rows} == set(range(48))
            assert rows[0]["scenario"] == rows[-1]["scenario"] == "normal"
            first = next(i for i, row in enumerate(rows) if row["scenario"] == "stress")
            assert len(rows) // 4 < first < len(rows) * 3 // 4
    c = case("cpx_06")
    assert len(c.public_log) > 10000
    # Logs contain varied measured values, not repeated meaningless filler.
    rows = [json.loads(line) for line in (DATA / "cpx_06/faulty/records/training.txt").read_text().splitlines()]
    assert len({str(r.get("loss")) for r in rows if "loss" in r}) > 30
    assert hashlib.sha256((DATA / "cpx_06/faulty/records/training.txt").read_bytes()).hexdigest() == next(a.faulty_sha256 for a in c.assets if a.path == "records/training.txt")


def test_entry_only_bypass_and_post_step_freezing_do_not_pass(tmp_path):
    c = case("cpx_04")
    io = ArtifactIO(tmp_path / "artifacts")
    candidate = materialize(c, tmp_path / "inline")
    source = (candidate / "entry.py").read_text().replace(
        "settings.optimizer(model, lr)",
        "torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)",
    ).replace("fusion.combine(model, image, text)",
              "torch.cat((torch.tanh(model.image(image)), torch.tanh(model.text(text))), dim=-1)")
    (candidate / "entry.py").write_text("import torch\n" + source)
    result = check(c, candidate, io, "inline-bypass", "holdout")
    assert result.status == "failed"
    checks = json.loads(result.stdout.text)["rows"][0]["checks"]
    assert checks["parameter_and_buffer_values"]
    assert not checks["optimizer_parameter_ownership"] and not checks["fusion_gradient_interface"]
    candidate = materialize(c, tmp_path / "late-freeze", healthy=True)
    (candidate / "entry.py").write_text((candidate / "entry.py").read_text() +
        '\noriginal_step = step\ndef step(model, batch, lr=0.03):\n    value = original_step(model, batch, lr)\n    for p in model.head.parameters():\n        p.requires_grad_(False)\n    return value\n')
    result = check(c, candidate, io, "late-freeze", "holdout")
    assert result.status == "failed"
    assert not json.loads(result.stdout.text)["rows"][0]["checks"]["trainability_preserved"]
