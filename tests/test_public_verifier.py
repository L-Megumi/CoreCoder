import sys

import pytest

from corecoder.workflow import run_verification, workspace_revision


def test_only_real_command_can_pass_and_revision_covers_dirty_source(tmp_path):
    source = tmp_path / "value.py"
    source.write_text("value=1\n")
    initial = workspace_revision(tmp_path)
    result = run_verification([sys.executable, "-c", "print('检查通过')"], tmp_path)
    assert result["status"] == "passed" and result["exit_code"] == 0
    assert "检查通过" in result["stdout"]["text"]
    assert result["checked_workspace_revision"] == initial
    source.write_text("value=2\n")
    assert workspace_revision(tmp_path) != initial


@pytest.mark.parametrize("code,status,exit_code", [(0, "passed", 0), (7, "failed", 7)])
def test_actual_exit_code(tmp_path, code, status, exit_code):
    result = run_verification([sys.executable, "-c", f"import sys;sys.exit({code})"], tmp_path)
    assert (result["status"], result["exit_code"]) == (status, exit_code)


def test_timeout_preserves_partial_output_without_exit_code(tmp_path):
    result = run_verification([sys.executable, "-c", "import time;print('partial',flush=True);time.sleep(5)"], tmp_path, 1)
    assert result["status"] == "timeout" and "exit_code" not in result
    assert "partial" in result["stdout"]["text"]


def test_launch_error_interruption_and_guard_never_invent_pass(tmp_path, monkeypatch):
    assert run_verification(["/missing-phase1-executable"], tmp_path)["status"] == "launch_error"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("corecoder.workflow.subprocess.run", interrupted)
    assert run_verification(["anything"], tmp_path)["status"] == "interrupted"

    def denied(*args):
        raise ValueError("blocked")

    with pytest.raises(ValueError, match="blocked"):
        run_verification(["anything"], tmp_path, guard=denied)


def test_truncation_and_invalid_arguments(tmp_path):
    result = run_verification([sys.executable, "-c", "print('A'*20000)"], tmp_path)
    output = result["stdout"]
    assert output["truncated"] and output["original_chars"] == 20001 and output["omitted_chars"] == 8001
    for args in [("shell command", 30), ([], 30), (["echo"], True), (["bad\0"], 30)]:
        with pytest.raises(ValueError):
            run_verification(args[0], tmp_path, args[1])
