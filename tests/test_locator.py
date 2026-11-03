"""Behavioral tests for current-source verification and bounded file inspection."""

import json
import os
import subprocess
import sys

import pytest

from corecoder.locator import locate_evidence


def write_source(root, text="def train():\n    run()\n", name="train.py"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def evidence(path, line=2, function="train", source_line="run()", identifier="ev_test", message=""):
    return {
        "id": identifier, "message": message,
        "frames": [{"file_path": str(path), "line_no": line, "function": function, "source_line": source_line}],
    }


def locate_one(root, item, **kwargs):
    result = locate_evidence([item], root, **kwargs)
    return result["locations"][0]


def test_exact_location_has_function_span_snapshot_hash_and_provenance(tmp_path):
    path = write_source(tmp_path)
    frame = locate_one(tmp_path, evidence(path))
    assert frame["status"] == "matched"
    assert frame["current_line_no"] == 2
    assert frame["logged_frame"]["line_no"] == 2
    assert frame["evidence_id"] == "ev_test"
    assert frame["relative_path"] == "train.py"
    assert frame["function"]["qualname"] == "train"
    assert (frame["function"]["start_line"], frame["function"]["end_line"]) == (1, 2)
    assert frame["function_verified"] is True
    assert len(frame["file_sha256"]) == 64
    assert frame["snippet"]["content"] == "def train():\n    run()"


def test_unique_anchor_relocates_after_inserted_lines(tmp_path):
    path = write_source(tmp_path, "# new\n# new\n\ndef train():\n    run()\n")
    frame = locate_one(tmp_path, evidence(path))
    assert frame["status"] == "relocated"
    assert frame["current_line_no"] == 5
    assert frame["logged_frame"]["line_no"] == 2
    assert frame["current_line_at_logged_line"] == "# new"
    assert frame["logged_line_matches"] is False
    assert frame["candidate_lines"] == [5]


def test_relocation_filters_same_code_in_an_unrelated_function(tmp_path):
    path = write_source(tmp_path, "def other():\n    run()\n\ndef train():\n    run()\n")
    frame = locate_one(tmp_path, evidence(path))
    assert frame["status"] == "relocated"
    assert frame["current_line_no"] == 5
    assert frame["function"]["name"] == "train"


def test_ambiguous_anchor_does_not_choose_an_arbitrary_line(tmp_path):
    path = write_source(tmp_path, "def train():\n    if a:\n        run()\n    if b:\n        run()\n")
    frame = locate_one(tmp_path, evidence(path, line=1))
    assert frame["status"] == "ambiguous"
    assert frame["current_line_no"] is None
    assert frame["candidate_lines"] == [3, 5]
    assert frame["snippet_basis"] == "logged_line_unverified"


def test_changed_code_is_not_reported_as_a_verified_location(tmp_path):
    path = write_source(tmp_path, "def train():\n    different_call()\n")
    frame = locate_one(tmp_path, evidence(path))
    assert frame["status"] == "source_changed"
    assert frame["current_line_no"] is None
    assert frame["current_line_at_logged_line"].strip() == "different_call()"
    assert frame["function_candidates"][0]["name"] == "train"


def test_outdated_line_beyond_eof_can_still_relocate_by_unique_anchor(tmp_path):
    path = write_source(tmp_path)
    frame = locate_one(tmp_path, evidence(path, line=999))
    assert frame["status"] == "relocated"
    assert frame["current_line_no"] == 2


def test_no_source_snapshot_is_explicitly_unverified(tmp_path):
    path = write_source(tmp_path)
    frame = locate_one(tmp_path, evidence(path, source_line=None))
    assert frame["status"] == "unverified"
    assert frame["logged_line_matches"] is None
    assert frame["current_line_no"] == 2


def test_function_mismatch_without_snapshot(tmp_path):
    path = write_source(tmp_path, "def other():\n    run()\n")
    frame = locate_one(tmp_path, evidence(path, source_line=None))
    assert frame["status"] == "function_mismatch"
    assert frame["current_line_no"] is None


def test_out_of_range_without_snapshot_has_no_invented_snippet(tmp_path):
    path = write_source(tmp_path)
    frame = locate_one(tmp_path, evidence(path, line=999, source_line=None))
    assert frame["status"] == "line_out_of_range"
    assert frame["snippet"] is None


def test_nested_async_method_and_decorator_ranges(tmp_path):
    path = write_source(tmp_path, "class Model:\n    @decorator\n    async def train(self):\n        def inner():\n            return 1\n        return inner()\n")
    outer = locate_one(tmp_path, evidence(path, line=6, source_line="return inner()"))
    inner = locate_one(tmp_path, evidence(path, line=5, function="inner", source_line="return 1"))
    assert outer["function"]["qualname"] == "Model.train"
    assert outer["function"]["start_line"] == 2
    assert outer["function_verified"] is True
    assert inner["function"]["qualname"] == "Model.train.inner"


def test_module_frame_on_function_definition_is_not_function_execution(tmp_path):
    path = write_source(tmp_path)
    frame = locate_one(tmp_path, evidence(path, line=1, function="<module>", source_line="def train():"))
    assert frame["status"] == "matched"
    assert frame["function"] is None
    assert frame["function_verified"] is True


def test_relative_path_resolves_from_project_root(tmp_path):
    write_source(tmp_path, name="pkg/train.py")
    frame = locate_one(tmp_path, evidence("pkg/train.py"))
    assert frame["status"] == "matched"
    assert frame["path_resolution"] == "relative"
    assert frame["relative_path"] == "pkg/train.py"


def test_explicit_mapping_rebases_old_absolute_root(tmp_path):
    write_source(tmp_path)
    frame = locate_one(
        tmp_path, evidence("/old/repo/train.py"), path_mappings={"/old/repo": str(tmp_path)},
    )
    assert frame["status"] == "matched"
    assert frame["path_resolution"] == "mapped"


def test_windows_log_path_can_map_to_native_project(tmp_path):
    write_source(tmp_path)
    frame = locate_one(
        tmp_path, evidence(r"C:\old\repo\train.py"),
        path_mappings={r"C:\old\repo": str(tmp_path)},
    )
    assert frame["status"] == "matched"


def test_longest_mapping_wins(tmp_path):
    write_source(tmp_path)
    frame = locate_one(
        tmp_path, evidence("/old/repo/train.py"),
        path_mappings={"/old": str(tmp_path / "wrong"), "/old/repo": str(tmp_path)},
    )
    assert frame["status"] == "matched"


def test_mapping_prefix_respects_path_components(tmp_path):
    frame = locate_one(
        tmp_path, evidence("/old/repository/train.py"), path_mappings={"/old/repo": str(tmp_path)},
    )
    assert frame["status"] in ("outside_project", "unmapped_path")
    assert frame["path_resolution"] != "mapped"


def test_foreign_absolute_path_requires_mapping(tmp_path):
    foreign = "/foreign/repo/train.py" if os.name == "nt" else r"Z:\foreign\train.py"
    frame = locate_one(tmp_path, evidence(foreign))
    assert frame["status"] == "unmapped_path"


def test_outside_project_is_not_read(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = write_source(tmp_path)
    result = locate_evidence([evidence(outside)], root)
    assert result["locations"][0]["status"] == "outside_project"
    assert result["locations"][0]["snippet"] is None
    assert result["stats"]["files_read"] == 0


@pytest.mark.parametrize("reference", ["../outside.py", "pkg/../../outside.py"])
def test_parent_traversal_cannot_escape_project(tmp_path, reference):
    frame = locate_one(tmp_path, evidence(reference))
    assert frame["status"] == "outside_project"


def test_mapping_target_cannot_escape_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    frame = locate_one(root, evidence("/old/train.py"), path_mappings={"/old": str(tmp_path)})
    assert frame["status"] == "outside_project"


def test_symlink_target_outside_project_is_not_read(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = write_source(tmp_path)
    link = root / "link.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this platform")
    result = locate_evidence([evidence(link)], root)
    assert result["locations"][0]["status"] == "outside_project"
    assert result["stats"]["files_read"] == 0


@pytest.mark.parametrize("reference", ["<stdin>", "<frozen importlib._bootstrap>"])
def test_virtual_frames_do_not_become_real_paths(tmp_path, reference):
    assert locate_one(tmp_path, evidence(reference))["status"] == "virtual_path"


@pytest.mark.parametrize("line", [0, -1, True, "2", None])
def test_invalid_line_numbers_are_reported(tmp_path, line):
    path = write_source(tmp_path)
    result = locate_evidence([evidence(path, line=line)], tmp_path)
    assert result["locations"][0]["status"] == "invalid_line"
    assert result["stats"]["files_read"] == 0


def test_missing_files_and_directories_are_distinguished(tmp_path):
    assert locate_one(tmp_path, evidence("missing.py"))["status"] == "missing"
    assert locate_one(tmp_path, evidence(tmp_path))["status"] == "not_file"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is unavailable")
def test_fifo_is_not_opened_as_source(tmp_path):
    pipe = tmp_path / "pipe.py"
    os.mkfifo(pipe)
    assert locate_one(tmp_path, evidence(pipe))["status"] == "not_file"


def test_file_size_limit_prevents_large_reads(tmp_path):
    path = write_source(tmp_path)
    result = locate_evidence([evidence(path)], tmp_path, max_file_bytes=5)
    assert result["locations"][0]["status"] == "too_large"
    assert result["stats"]["files_read"] == 0


def test_broken_python_keeps_text_snapshot_but_reports_parse_error(tmp_path):
    path = write_source(tmp_path, "def train(:\n    run()\n")
    frame = locate_one(tmp_path, evidence(path))
    assert frame["status"] == "matched"
    assert frame["parse_status"] == "parse_error"
    assert frame["parse_error"]
    assert frame["function"] is None
    assert frame["function_verified"] is None


def test_python_encoding_cookie_is_respected(tmp_path):
    path = tmp_path / "latin.py"
    path.write_bytes("# coding: latin-1\ndef train():\n    return 'café'\n".encode("latin-1"))
    frame = locate_one(tmp_path, evidence(path, line=3, source_line="return 'café'"))
    assert frame["status"] == "matched"
    assert "café" in frame["snippet"]["content"]


def test_invalid_encoding_is_a_structured_failure(tmp_path):
    path = tmp_path / "bad.py"
    path.write_bytes(b"\xff\xfe\x00\x01")
    assert locate_one(tmp_path, evidence(path))["status"] == "decode_error"


def test_short_snippets_and_character_limits_are_enforced(tmp_path):
    text = "def train():\n" + "".join("    long_call_name()\n" for _ in range(30))
    path = write_source(tmp_path, text)
    frame = locate_one(
        tmp_path, evidence(path, line=16, source_line="long_call_name()"),
        context_lines=10, max_snippet_lines=3, max_snippet_chars=12,
    )
    snippet = frame["snippet"]
    assert snippet["end_line"] - snippet["start_line"] + 1 <= 3
    assert len(snippet["content"]) <= 12
    assert snippet["char_truncated"] is True
    assert snippet["function_truncated"] is True


def test_cache_is_per_invocation_and_source_updates_are_visible(tmp_path):
    path = write_source(tmp_path)
    item = evidence(path)
    item["frames"].append(dict(item["frames"][0]))
    first = locate_evidence([item], tmp_path)
    assert first["stats"]["files_read"] == 1
    assert first["stats"]["cache_hits"] == 1
    old_hash = first["locations"][0]["file_sha256"]
    path.write_text("def train():\n    other_call()\n", encoding="utf-8")
    second = locate_evidence([item], tmp_path)
    assert second["locations"][0]["status"] == "source_changed"
    assert second["locations"][0]["file_sha256"] != old_hash


def test_configuration_references_deduplicate_evidence_ids(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text("path: ./dataset\nval: images/val\n", encoding="utf-8")
    items = [
        {"id": "root", "message": f"Dataset '{config}' missing"},
        {"id": "outer", "message": f"Dataset '{config}' error"},
    ]
    result = locate_evidence(items, tmp_path)
    located, = result["config_files"]
    assert located["status"] == "read"
    assert located["evidence_ids"] == ["root", "outer"]
    assert located["snippet"]["content"] == "path: ./dataset\nval: images/val"
    assert result["stats"]["files_read"] == 1


def test_external_config_hint_is_retained_without_reading_contents(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "settings.yaml"
    outside.write_text("private: value\n", encoding="utf-8")
    config, = locate_evidence([{"id": "e", "message": f"see '{outside}'"}], root)["config_files"]
    assert config["status"] == "outside_project"
    assert config["snippet"] is None


def test_configuration_preview_is_bounded(tmp_path):
    config = tmp_path / "data.json"
    config.write_text("0123456789\n" * 20, encoding="utf-8")
    result = locate_evidence(
        [{"id": "e", "message": f"see '{config}'"}], tmp_path,
        max_snippet_lines=2, max_snippet_chars=5,
    )
    snippet = result["config_files"][0]["snippet"]
    assert snippet["end_line"] == 2
    assert len(snippet["content"]) == 5
    assert snippet["truncated"] is True


def test_frame_and_configuration_budgets_report_omitted_items(tmp_path):
    path = write_source(tmp_path)
    first = evidence(path, message="'first.yaml' 'second.yaml'")
    second = evidence(path, identifier="another")
    result = locate_evidence([first, second], tmp_path, max_frames=1, max_config_files=1)
    assert len(result["locations"]) == len(result["config_files"]) == 1
    assert result["stats"]["frames_requested"] == 2
    assert result["stats"]["frames_omitted"] == 1
    assert result["stats"]["config_references_omitted"] == 1


def test_project_source_is_never_imported_or_executed(tmp_path):
    marker = tmp_path / "executed.txt"
    text = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\ndef train():\n    run()\n"
    path = write_source(tmp_path, text)
    frame = locate_one(tmp_path, evidence(path, line=4))
    assert frame["status"] == "matched"
    assert not marker.exists()


@pytest.mark.parametrize("kwargs", [{"max_frames": 0}, {"context_lines": -1}, {"max_file_bytes": True}])
def test_invalid_budgets_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        locate_evidence([], tmp_path, **kwargs)


def test_project_root_must_exist(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        locate_evidence([], tmp_path / "missing")


def cli_log(tmp_path):
    path = write_source(tmp_path)
    log = tmp_path / "failed.log"
    log.write_text(
        f'Traceback (most recent call last):\n  File "{path}", line 2, in train\n    run()\nRuntimeError: failed\n',
        encoding="utf-8",
    )
    return path, log


def test_cli_combines_extraction_and_location(tmp_path):
    _, log = cli_log(tmp_path)
    output = tmp_path / "locations.json"
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.locator", str(log), "--project", str(tmp_path), "--output", str(output)],
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["locations"][0]["status"] == "matched"
    assert report["locations"][0]["evidence_id"].startswith("ev_")


@pytest.mark.parametrize("target", ["log", "source"])
def test_cli_cannot_overwrite_log_or_located_source(tmp_path, target):
    path, log = cli_log(tmp_path)
    destination = log if target == "log" else path
    original = destination.read_bytes()
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.locator", str(log), "--project", str(tmp_path), "--output", str(destination)],
        capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert destination.read_bytes() == original


def test_cli_rejects_malformed_mapping(tmp_path):
    _, log = cli_log(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.locator", str(log), "--project", str(tmp_path), "--map-root", "invalid"],
        capture_output=True, check=False,
    )
    assert result.returncode == 2
