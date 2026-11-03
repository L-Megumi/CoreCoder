import json
import pytest
from researchcoder.evaluation.projects.__main__ import load_config, main
from researchcoder.evaluation.projects.runner import ProjectPolicy


def test_project_policy_roundtrip_and_feedback_limits():
    value = ProjectPolicy()
    assert ProjectPolicy.from_json(value.to_json()) == value
    with pytest.raises(ValueError): ProjectPolicy(max_feedback_rounds=3)
    with pytest.raises(ValueError): ProjectPolicy.from_dict(dict(value.to_dict(), unknown=True))


@pytest.mark.parametrize("value", [{}, {"schema_version":2}, {"api_key":"not-accepted"}])
def test_project_cli_refuses_incomplete_or_unknown_config(tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError): load_config(path)
    assert main(["--config", str(path), "--output", str(tmp_path / "output")]) == 2
    assert not (tmp_path / "output").exists()


def test_project_cli_recompute_does_not_accept_execution_options(tmp_path):
    with pytest.raises(SystemExit) as caught:
        main(["--recompute", str(tmp_path), "--allow-live"])
    assert caught.value.code == 2


def test_project_cli_help_has_explicit_live_switch(capsys):
    with pytest.raises(SystemExit) as caught: main(["--help"])
    assert caught.value.code == 0
    assert "--allow-live" in capsys.readouterr().out
