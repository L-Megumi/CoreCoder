"""Investigation scope, provenance and tool error behavior."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from corecoder.diagnosis_demo import create_demo_case
from corecoder.diagnostic_tools import Investigation, investigation_tools


@pytest.fixture
def case(tmp_path):
    return create_demo_case(tmp_path / "case")


@pytest.fixture
def state(case):
    workspace, root, log, _, _ = case
    return Investigation(log, root, allowed_path_roots=[workspace])


def call(state, name, **arguments):
    return json.loads(state.execute(name, arguments))


def test_real_evidence_pagination_keeps_store_and_chain(state):
    first = call(state, "extract_log_evidence", limit=1)
    second = call(state, "extract_log_evidence", offset=first["next_offset"], limit=1)
    assert first["total"] == 2
    assert second["next_offset"] is None
    assert second["evidence"][0]["cause_id"] == first["evidence"][0]["id"]
    assert "raw_content" not in first["evidence"][0]
    assert "raw_content" in state.evidence[0]
    assert "missing path" in state.evidence[0]["raw_content"]


def test_locator_requires_extracted_known_evidence(state):
    assert "error" in call(state, "locate_source", evidence_ids=["unknown"])
    call(state, "extract_log_evidence")
    assert "error" in call(state, "locate_source", evidence_ids=["unknown"])
    ids = [item["id"] for item in state.evidence]
    report = call(state, "locate_source", evidence_ids=ids)
    assert report["observations"]
    assert all(observation["id"] in state.observations for observation in report["observations"])
    assert {e for item in report["observations"] for e in item["evidence_ids"]} == set(ids)


def test_read_project_file_paginates_and_does_not_execute(state, case):
    _, root, _, _, _ = case
    result = call(state, "read_project_file", path=str(root / "train.py"), max_lines=1)
    data = result["observations"][0]["data"]
    assert data["status"] == "read"
    assert data["content"] == "def check_data():"
    assert data["truncated"] is True
    beyond = call(state, "read_project_file", path=str(root / "train.py"), start_line=999)
    assert beyond["observations"][0]["data"]["status"] == "line_out_of_range"


def test_reading_metadata_root_does_not_grant_external_content_access(state, case):
    workspace, _, _, _, _ = case
    outside = workspace / "external.yaml"
    outside.write_text("private: value\n", encoding="utf-8")
    read = call(state, "read_project_file", path=str(outside))
    assert read["observations"][0]["data"]["status"] == "outside_project"
    assert "content" not in read["observations"][0]["data"]
    meta = call(state, "inspect_paths", paths=[str(outside)])
    assert meta["observations"][0]["data"]["status"] == "exists"
    assert "content" not in meta["observations"][0]["data"]


def test_environment_file_is_not_a_diagnostic_read_target(state, case):
    _, root, _, _, _ = case
    env = root / ".env"
    env.write_text("PRIVATE_KEY=not-for-the-model\n", encoding="utf-8")
    result = call(state, "read_project_file", path=str(env))
    assert "error" in result
    assert "not-for-the-model" not in json.dumps(result)


def test_metadata_checks_literal_old_path_even_when_source_mapping_exists(case):
    workspace, root, log, current, previous = case
    state = Investigation(
        log, root, path_mappings={str(workspace / "previous"): str(root)},
        allowed_path_roots=[workspace],
    )
    result = call(state, "inspect_paths", paths=[str(current), str(previous)])
    assert [item["data"]["status"] for item in result["observations"]] == ["exists", "missing"]
    assert result["observations"][1]["data"]["resolved_path"] == str(previous.resolve())


def test_external_path_is_not_reported_as_missing(state, tmp_path):
    outside = tmp_path / "not-allowed"
    data = call(state, "inspect_paths", paths=[str(outside)])["observations"][0]["data"]
    assert data["status"] == "outside_allowed_roots"
    assert "is_file" not in data


def test_symlink_does_not_expand_metadata_scope(state, case, tmp_path):
    _, root, _, _, _ = case
    outside = tmp_path / "not-allowed"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable")
    data = call(state, "inspect_paths", paths=[str(link)])["observations"][0]["data"]
    assert data["status"] == "outside_allowed_roots"


def test_repeated_observation_ids_are_stable(state, case):
    _, _, _, current, _ = case
    first = call(state, "inspect_paths", paths=[str(current)])
    second = call(state, "inspect_paths", paths=[str(current)])
    assert first == second
    assert len(state.observations) == 1
    assert len(state.events) == 2


def test_parallel_tool_dispatch_keeps_auditable_sequence(state, case):
    _, root, _, _, _ = case
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [
            pool.submit(state.execute, "read_project_file", {"path": str(root / filename)})
            for filename in ("data.yaml", "train.py")
        ]
        outputs = [json.loads(request.result()) for request in requests]
    assert all("observations" in output for output in outputs)
    assert [event["sequence"] for event in state.events] == [1, 2]


def test_tool_budget_blocks_further_investigation(case):
    _, root, log, current, _ = case
    state = Investigation(log, root, max_tool_calls=1)
    assert "evidence" in call(state, "extract_log_evidence")
    failure = call(state, "inspect_paths", paths=[str(current)])
    assert "budget" in failure["error"]["message"]
    assert not state.observations
    assert state.events[-1]["status"] == "error"


@pytest.mark.parametrize("name,args", [
    ("extract_log_evidence", {"offset": -1}),
    ("extract_log_evidence", {"limit": True}),
    ("extract_log_evidence", {"unexpected": "argument"}),
    ("inspect_paths", {"paths": "not-a-list"}),
    ("inspect_paths", {"paths": [123]}),
    ("read_project_file", {"path": "train.py", "start_line": 0}),
    ("not_a_tool", {}),
])
def test_invalid_requests_produce_structured_errors(state, name, args):
    result = json.loads(state.execute(name, args))
    assert "error" in result
    assert state.events[-1]["status"] == "error"


def test_diagnostic_tools_exclude_general_shell_and_mutation(state):
    tools = {tool.name: tool for tool in investigation_tools(state)}
    assert set(tools) == {"extract_log_evidence", "locate_source", "read_project_file", "inspect_paths", "search_project_source",
                          "resolve_project_symbol"}
    assert all(tool.schema()["function"]["parameters"]["additionalProperties"] is False for tool in tools.values())


def test_display_truncation_does_not_destroy_original_evidence(case):
    _, root, log, _, _ = case
    log.write_text("RuntimeError: " + "x" * 13000, encoding="utf-8")
    state = Investigation(log, root)
    displayed = call(state, "extract_log_evidence")["evidence"][0]
    assert displayed["message_truncated"] is True
    assert len(displayed["message"]) == 12000
    assert len(state.evidence[0]["message"]) == 13000


def test_log_size_limit_rejects_incomplete_silent_tail_reads(case):
    _, root, log, _, _ = case
    log.write_bytes(b"x" * 5_000_001)
    with pytest.raises(ValueError, match="5 MB"):
        Investigation(log, root)
