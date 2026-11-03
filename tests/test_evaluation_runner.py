from dataclasses import replace
import json
import os
import threading
import pytest
from researchcoder.evaluation.contracts import EvaluationConfig
from researchcoder.evaluation.runner import (
    EvaluationRunner,
    recompute,
    build_workflow,
    planned_trials,
)
from researchcoder.evaluation.dataset import load_manifest
from researchcoder.contracts import ArtifactRef
from researchcoder.adapters.artifacts import ArtifactIO


@pytest.mark.parametrize("mode", ["single", "sequential", "parallel", "evidence"])
def test_real_original_runtime_eval_mode_and_recompute_without_model(
    tmp_path, mode, monkeypatch
):
    config = EvaluationConfig(modes=(mode,), case_ids=("autograd_detach",))
    output = tmp_path / "evaluation"
    from researchcoder.evaluation.observability import LangfuseExporter

    class BrokenExporter:
        def export_result(self, *args):
            raise ConnectionError("private exporter connection detail")

    exporter = (
        BrokenExporter()
        if mode == "parallel"
        else LangfuseExporter(mode="disabled" if mode == "single" else "offline")
    )
    report = EvaluationRunner().run(
        output,
        config,
        torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
        exporter=exporter,
    )
    summary = report["modes"][mode]
    assert summary["rates"]["end_to_end_success"]["passed"] == 1, report
    assert (
        summary["rates"]["pass_to_pass"]["passed"] == 1
        and summary["rates"]["fail_to_pass"]["passed"] == 1
    )
    assert (
        summary["actual_prompt_tokens"] is None
        and summary["unknown_requests_observed"] > 0
    )
    assert summary["cost"] is None
    monkeypatch.setattr(
        "researchcoder.runtime.ResearchHarness.execute",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("offline recompute must not execute")
        ),
    )
    assert recompute(output) == report
    exported = json.loads((output / "exports/trial-0000.json").read_text())
    assert exported["status"] == (
        "export_error"
        if mode == "parallel"
        else "disabled"
        if mode == "single"
        else "offline"
    )
    assert "private exporter connection detail" not in str(exported)
    record_path = next((output / "records").glob("*.json"))
    ref = ArtifactRef.from_json(record_path.read_text())
    io = ArtifactIO(output / "trials/trial-0000/artifacts")
    result = io.read_json(ref)
    raw = io.read_json(ArtifactRef.from_dict(result["raw"]))
    assert raw["workspace_before"] != raw["workspace_after"]
    if mode != "single":
        assert (output / "trials/trial-0000/state.sqlite").is_file()
        times = {
            r["logical_task_id"]: (r["started_ns"], r["ended_ns"])
            for r in raw["records"]
        }
        if mode == "sequential":
            assert times["code"][0] >= times["log"][1]
        else:
            assert max(times["code"][0], times["log"][0]) < min(
                times["code"][1], times["log"][1]
            )


def test_failed_patch_is_recorded_not_dropped(tmp_path):
    config = EvaluationConfig(
        modes=("single",),
        case_ids=("shape_transpose",),
        scripted_behavior="no_patch",
        repeats=2,
        stop_after_failures=1,
    )
    report = EvaluationRunner().run(
        tmp_path / "run",
        config,
        torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
    )
    summary = report["modes"]["single"]
    assert summary["planned"] == summary["recorded"] == 2
    assert summary["statuses"]["failed"] == 1 and summary["statuses"]["cancelled"] == 1
    assert (
        summary["rates"]["end_to_end_success"]["passed"] == 0
        and summary["rates"]["fail_to_pass"]["passed"] == 0
    )
    assert summary["rates"]["pass_to_pass"]["passed"] == 1


def test_cancel_and_setup_failures_preserve_planned_samples(tmp_path, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    cfg = EvaluationConfig(modes=("single", "parallel"), case_ids=("shape_transpose",))
    report = EvaluationRunner().run(
        tmp_path / "cancel",
        cfg,
        torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
        cancel_event=cancel,
    )
    assert all(r["statuses"]["cancelled"] == 1 for r in report["modes"].values())
    monkeypatch.setattr(
        "researchcoder.evaluation.runner.build_workflow",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("setup fault")),
    )
    report = EvaluationRunner().run(
        tmp_path / "failed",
        replace(cfg, modes=("single",)),
        torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
        cancel_event=cancel,
    )
    assert report["modes"]["single"]["statuses"]["incomplete"] == 1


def test_round_rotation_and_private_answer_not_in_task_inputs(tmp_path):
    cases = load_manifest().cases[:2]
    config = EvaluationConfig(repeats=2)
    rows = planned_trials(cases, config)
    assert len(rows) == 16 and [r.mode for r in rows[:4]] == list(config.modes)
    assert [r.mode for r in rows[4:8]] == [
        "sequential",
        "parallel",
        "evidence",
        "single",
    ]
    built = build_workflow(
        cases[0],
        rows[2],
        tmp_path / "trial",
        config,
        os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
    )
    for t in built["config"].templates.values():
        wire = t.spec.to_json()
        assert (
            "reference_cause" not in wire
            and "cause_concepts" not in wire
            and "healthy_sha256" not in wire
        )
        assert all(i.role in ("raw_log", "diagnosis_run") for i in t.spec.inputs)
        if t.role != "verification":
            assert t.spec.verification is None
