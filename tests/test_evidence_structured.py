"""Regression tests for complete messages, provenance, chains and the offline CLI."""

import json
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

from corecoder.evidence import extract_evidence

FIXTURE = Path(__file__).resolve().parents[1] / "examples/logs/dronevehicle_missing_path.log"
CAUSE = "The above exception was the direct cause of the following exception:"
CONTEXT = "During handling of the above exception, another exception occurred:"


def traceback_text(exception="ValueError: broken", path="train.py"):
    return f'Traceback (most recent call last):\n  File "{path}", line 9, in train\n    run()\n{exception}'


def test_dronevehicle_full_messages_and_explicit_cause():
    text = FIXTURE.read_text(encoding="utf-8")
    root, wrapper = extract_evidence(text, source="dronevehicle.log")
    assert root["exception_type"] == "FileNotFoundError"
    assert wrapper["exception_type"] == "RuntimeError"
    for item in (root, wrapper):
        assert "missing path '/data/example/M2D-LIF/dataset/DroneVehicle_train_val_crop/images/val'" in item["message"]
        assert "settings.yaml" in item["message"]
        assert item["truncated"] is False
        assert item["source"] == "dronevehicle.log"
    assert root["message"].startswith("\nDataset")
    assert root["cause_id"] is None
    assert wrapper["cause_id"] == root["id"]
    assert wrapper["context_id"] is None
    assert wrapper["chain_id"] == root["chain_id"] == root["id"]
    assert len(root["frames"]) == 2
    assert len(wrapper["frames"]) == 5
    assert root["frames"][-1] == {
        "file_path": "/home/example/M2D-LIF/ultralytics/data/utils.py",
        "line_no": 312,
        "function": "check_det_dataset",
        "log_line_no": 4,
        "source_line": "raise FileNotFoundError(m)",
    }
    assert (root["line_no"], root["end_line_no"]) == (1, 8)
    assert (wrapper["line_no"], wrapper["end_line_no"]) == (12, 25)
    for item in (root, wrapper):
        recovered = "\n".join(text.splitlines()[item["line_no"] - 1:item["end_line_no"]])
        assert recovered == item["raw_content"] == item["content"]
        assert CAUSE not in item["content"]


def test_context_is_distinct_from_explicit_causality():
    text = traceback_text() + f"\n\n{CONTEXT}\n\n" + traceback_text("RuntimeError: wrapper")
    root, wrapper = extract_evidence(text)
    assert wrapper["context_id"] == root["id"]
    assert wrapper["cause_id"] is None
    assert wrapper["chain_id"] == root["id"]


def test_three_exception_chain_keeps_immediate_parent_and_chain_root():
    text = (
        traceback_text("OSError: root") + f"\n\n{CAUSE}\n\n"
        + traceback_text("ValueError: middle") + f"\n\n{CAUSE}\n\n"
        + traceback_text("RuntimeError: outer")
    )
    root, middle, outer = extract_evidence(text)
    assert middle["cause_id"] == root["id"]
    assert outer["cause_id"] == middle["id"]
    assert outer["chain_id"] == root["id"]


def test_standalone_cause_without_its_own_traceback():
    text = "ValueError: original\n\n" + CAUSE + "\n\n" + traceback_text("RuntimeError: outer")
    root, outer = extract_evidence(text)
    assert root["frames"] == []
    assert outer["cause_id"] == root["id"]


def test_separate_tracebacks_are_not_implicitly_linked():
    first, second = extract_evidence(traceback_text() + "\n\n" + traceback_text("OSError: next"))
    assert second["cause_id"] is None
    assert second["context_id"] is None
    assert second["chain_id"] != first["chain_id"]


def test_orphan_chain_marker_does_not_invent_a_cause():
    item, = extract_evidence(CAUSE + "\n" + traceback_text())
    assert item["cause_id"] is None


def test_intervening_log_prevents_stale_chain_link():
    text = traceback_text() + "\nINFO: another run\n" + CAUSE + "\n" + traceback_text("OSError: next")
    first, second = extract_evidence(text)
    assert second["cause_id"] is None
    assert second["chain_id"] != first["chain_id"]


@pytest.mark.parametrize("boundary", ["INFO: retrying", "[worker] INFO: retrying", "2026-10-20 10:00:00 INFO: retrying"])
def test_explicit_log_boundary_stops_multiline_message(boundary):
    text = traceback_text("FileNotFoundError:") + "\nmissing path /old/data\n" + boundary + "\nloss = NaN"
    error, numeric = extract_evidence(text)
    assert error["message"] == "\nmissing path /old/data"
    assert boundary not in error["content"]
    assert numeric["type"] == "NUMERIC_ANOMALY"


def test_warning_and_metric_after_traceback_remain_separate_evidence():
    text = traceback_text() + "\nUserWarning: retry\nloss = -Inf"
    assert [item["type"] for item in extract_evidence(text)] == ["ERROR", "WARNING", "NUMERIC_ANOMALY"]


def test_multiline_message_with_error_words_or_file_like_text_is_retained():
    message = 'RuntimeError: configuration error\nError details are shown below\n  File "note.txt", line 20\nextra exception details'
    item, = extract_evidence(traceback_text(message))
    assert item["message"].endswith("extra exception details")
    assert len(item["frames"]) == 1


def test_indented_code_and_caret_are_not_exception_headers():
    text = 'Traceback (most recent call last):\n  File "train.py", line 3, in train\n    RuntimeError\n    ^^^^^^^^^^^^\nTrainingFailure: custom problem'
    item, = extract_evidence(text)
    assert item["exception_type"] == "TrainingFailure"
    assert item["message"] == "custom problem"
    assert item["frames"][0]["source_line"] == "RuntimeError"


def test_syntax_error_frame_without_function_name():
    text = 'Traceback (most recent call last):\n  File "train.py", line 3\n    invalid syntax\n            ^^^^^^\nSyntaxError: invalid syntax'
    item, = extract_evidence(text)
    assert item["exception_type"] == "SyntaxError"
    assert item["frames"][0]["function"] is None


def test_incomplete_traceback_does_not_swallow_next_one():
    text = 'Traceback (most recent call last):\n  File "first.py", line 1, in train\n    run()\n' + traceback_text()
    incomplete, complete = extract_evidence(text)
    assert incomplete["truncated"] is True
    assert incomplete["exception_type"] is None
    assert complete["truncated"] is False
    assert complete["line_no"] == 4


def test_incomplete_traceback_stops_at_log_boundary():
    text = 'Traceback (most recent call last):\n  File "train.py", line 1\nINFO: disconnected\nWARNING: partial log'
    incomplete, warning = extract_evidence(text)
    assert incomplete["truncated"] is True
    assert incomplete["end_line_no"] == 2
    assert warning["type"] == "WARNING"


def test_ids_are_reproducible_source_sensitive_and_label_sensitive():
    text = "  WARNING: loss = NaN  "
    first = extract_evidence(text, source="run-a.log")
    assert first == extract_evidence(text, source="run-a.log")
    assert {x["id"] for x in first}.isdisjoint(x["id"] for x in extract_evidence(text, source="run-b.log"))
    assert first[0]["id"] != first[1]["id"]
    assert first[0]["raw_content"] == text
    assert first[0]["content"] == text.strip()


def test_crlf_and_uniform_indentation_preserve_line_provenance():
    text = "\r\n".join("  " + line for line in traceback_text().splitlines())
    item, = extract_evidence(text)
    assert item["exception_type"] == "ValueError"
    assert item["line_no"] == 1
    assert item["end_line_no"] == 4
    assert item["frames"][0]["log_line_no"] == 2


@pytest.mark.parametrize("value", ["StopIteration", "SystemExit: 2", "KeyboardInterrupt", "package.CustomError: broken"])
def test_non_error_named_control_exceptions_and_qualified_names(value):
    item, = extract_evidence(traceback_text(value))
    assert item["exception_type"] == value.split(":")[0]
    assert item["truncated"] is False


def test_unstructured_trailing_text_is_retained_until_a_known_boundary():
    # Without a delimiter an arbitrary exception message and an arbitrary log
    # line cannot be distinguished. Preserve it rather than silently dropping it.
    item, = extract_evidence(traceback_text() + "\nadditional unstructured details")
    assert item["message"].endswith("additional unstructured details")


def test_blank_lines_inside_exception_message_are_preserved():
    text = traceback_text("ValueError: first paragraph") + "\n\nsecond paragraph\n\nINFO: next run"
    item, = extract_evidence(text)
    assert item["message"] == "first paragraph\n\nsecond paragraph"


def test_real_python_formatted_exception_chain_roundtrip():
    class TrainingFailure(Exception):
        pass

    try:
        try:
            raise FileNotFoundError("missing dataset\n\npath: /old/images/val")
        except FileNotFoundError as cause:
            raise TrainingFailure("training could not start") from cause
    except TrainingFailure:
        text = traceback.format_exc()

    root, wrapper = extract_evidence(text, source="actual-python-traceback.log")
    assert root["message"] == "missing dataset\n\npath: /old/images/val"
    assert wrapper["exception_type"].endswith("TrainingFailure")
    assert wrapper["cause_id"] == root["id"]
    assert root["frames"] and wrapper["frames"]
    assert not root["truncated"] and not wrapper["truncated"]


def test_cli_writes_json_file_and_keeps_unicode(tmp_path):
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.evidence", str(FIXTURE), "--output", str(output)],
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert "⚠️" in data[0]["raw_content"]
    assert data[1]["cause_id"] == data[0]["id"]


def test_cli_stdout_and_missing_file(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.evidence", str(FIXTURE)],
        capture_output=True, check=False,
    )
    assert result.returncode == 0
    # The CLI emits UTF-8 even when the surrounding terminal uses another codec.
    data = json.loads(result.stdout)
    assert data[0]["exception_type"] == "FileNotFoundError"
    missing = subprocess.run(
        [sys.executable, "-m", "corecoder.evidence", str(tmp_path / "missing.log")],
        capture_output=True, check=False,
    )
    assert missing.returncode == 2
    assert missing.stderr


def test_cli_cannot_overwrite_its_input_log(tmp_path):
    log = tmp_path / "experiment.log"
    original = traceback_text()
    log.write_text(original, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.evidence", str(log), "--output", str(log)],
        capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert log.read_text(encoding="utf-8") == original
