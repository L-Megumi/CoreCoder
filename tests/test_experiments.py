"""Experiment outputs, separation of scripted/real usage and refusal to overwrite runs."""

import json

import pytest

from corecoder.diagnose import main as diagnose_main
from corecoder.diagnosis_demo import DemoLLM, create_demo_case
from corecoder.experiments import ARMS, main, run_live, run_offline


def test_offline_four_arms_and_stress_preserve_exact_facts_without_provider(tmp_path):
    result = run_offline(tmp_path)
    assert {row["arm"] for row in result["runs"]} == set(ARMS)
    assert all(row["status"] == "completed" for row in result["runs"])
    reports = [json.loads((tmp_path / f"{arm}.json").read_text()) for arm in ARMS]
    assert len({report["log_file"] for report in reports}) == 1
    assert len({report["evidence"][0]["id"] for report in reports}) == 1
    assert all(not arm["all_provider_usage_complete"] for arm in result["summary"].values())
    baseline = result["summary"]["baseline"]["mean_request_tokens_estimate"]
    assert result["summary"]["both"]["mean_request_tokens_estimate"] < baseline
    assert len(result["context_stress"]) == 3
    for case in result["context_stress"]:
        assert case["constructed_transcript"]
        variants = case["variants"]
        assert variants["evidence"]["structured_fact_retention"] == 1
        assert variants["evidence"]["facts_total"] == variants["baseline"]["facts_total"]
    assert "provider" in " ".join(result["limitations"])


def test_live_runner_order_manifest_outputs_and_metrics_are_auditable(tmp_path, monkeypatch):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    directory = tmp_path / "results"
    directory.mkdir()
    monkeypatch.setattr("corecoder.experiments.build_deepseek_llm", lambda **kwargs: DemoLLM(root, current, previous))
    result = run_live(directory, log, root, allowed_path_roots=[workspace])
    assert len(result["runs"]) == 4
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["seed_scope"] == "arm ordering only; does not seed the model"
    assert len(manifest["schedule"]) == 4
    assert len({row["arm"] for row in result["runs"]}) == 4
    for row in result["runs"]:
        report = json.loads((directory / row["artifact"]).read_text())
        assert row["status"] == report["status"] == "completed"
        assert not report["metrics"]["provider_usage_complete"]


def test_experiment_cli_refuses_existing_directory_and_missing_live_inputs(tmp_path):
    with pytest.raises(SystemExit) as missing:
        main(["--live", "--output-dir", str(tmp_path / "new")])
    assert missing.value.code == 2
    assert not (tmp_path / "new").exists()
    with pytest.raises(SystemExit) as existing:
        main(["--offline", "--output-dir", str(tmp_path)])
    assert existing.value.code == 2


@pytest.mark.parametrize("repeats", [0, 4, True])
def test_live_repetition_budget_is_bounded_before_provider_call(tmp_path, repeats):
    with pytest.raises(ValueError, match="repeats"):
        run_live(tmp_path, "unused.log", tmp_path, repeats=repeats)


def test_relative_fixture_directory_produces_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace, root, log, current, previous = create_demo_case("relative-case")
    assert all(path.is_absolute() for path in (workspace, root, log, current, previous))
    assert current.is_dir()
    assert not previous.exists()


def test_diagnosis_cli_keeps_baseline_default_and_optimization_is_opt_in(tmp_path):
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    assert diagnose_main(["--demo", "--output", str(baseline)]) == 0
    assert diagnose_main(["--demo", "--tool-selection", "guided", "--context-mode", "evidence",
                          "--output", str(optimized)]) == 0
    assert json.loads(baseline.read_text())["configuration"]["context_mode"] == "baseline"
    assert json.loads(optimized.read_text())["configuration"]["context_mode"] == "evidence"
