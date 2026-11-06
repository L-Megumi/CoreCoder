"""Guarded writer handoff, independent of benchmark-specific scoring."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from researchcoder.context.policy import ContextPolicy
from researchcoder.context.repair_state import (
    FOCUSED_GUIDANCE_V2_SHA256, FOCUSED_GUIDANCE_SHA256, MAX_EDIT_EXCERPT_CHARS,
)
from tests.test_coding_context import run_tool, setup as coding_setup
from tests.test_m11_repair_continuity import bind, project, state_policy, state_record


def writer(tmp_path, version="focused-repair-v2"):
    base, _, _, io = coding_setup(tmp_path)
    policy = replace(state_policy(), writer_guidance_version=version)
    return bind(replace(base.request, context_policy=policy), io)


def test_writer_policy_requires_trusted_state_and_distinct_identity(tmp_path):
    with pytest.raises(ValueError):
        ContextPolicy(writer_guidance_version="focused-repair-v2")
    manager, agent, _, _ = writer(tmp_path)
    assert ContextPolicy.from_json(manager.policy.to_json()) == manager.policy
    assert "Focused repair policy v2" in agent._system
    assert manager.repair_state.receipt()["guidance_sha256"] == FOCUSED_GUIDANCE_V2_SHA256
    assert FOCUSED_GUIDANCE_V2_SHA256 != FOCUSED_GUIDANCE_SHA256


@pytest.mark.parametrize("version", ["focused-repair-v1", "focused-repair-v2"])
def test_guarded_edit_receipt_is_bound_to_revision_and_grants_no_read_or_success(tmp_path, version):
    manager, agent, hooks, io = writer(tmp_path, version)
    path = Path(agent.root) / "module.py"
    _, read = run_tool(manager, agent, "read_file", {"file_path": "module.py"})
    wrong, _ = run_tool(manager, agent, "edit_file", {
        "file_path": "module.py", "old_string": "not in this file", "new_string": "bogus"}, "bad")
    assert wrong.startswith("Error") and manager.repair_state.latest_edit_excerpt is None
    _, edit = run_tool(manager, agent, "edit_file", {"file_path": "module.py",
        "old_string": "return value + 0", "new_string": "return value + 1"}, "good")
    view = state_record(project(manager, agent, hooks, io, read + edit))
    assert view["verification"]["status"] == "not_run"
    assert not view["current_reads"] and view["stale_read_count"] >= 1
    assert ("latest_edit_excerpt" in view) == (version == "focused-repair-v2")
    if version == "focused-repair-v2":
        excerpt = view["latest_edit_excerpt"]
        assert excerpt["file_sha256"] == sha256(path.read_bytes()).hexdigest()
        assert excerpt["full_manifest_sha256"] == view["full_manifest_sha256"]
        assert excerpt["event_sequence"] == view["modified_files"][0]["event_sequence"]
        assert "return value + 1" in excerpt["excerpt"] and str(tmp_path) not in excerpt["excerpt"]
        assert not excerpt["truncated"]
        assert len(excerpt["excerpt"]) <= MAX_EDIT_EXCERPT_CHARS
        (path.parent / "other.py").write_text("external_change = True\n")
        changed = state_record(project(manager, agent, hooks, io, read + edit))
        assert "latest_edit_excerpt" not in changed


def test_excerpt_is_bounded_even_for_large_effective_replacements(tmp_path):
    manager, agent, hooks, io = writer(tmp_path)
    _, read = run_tool(manager, agent, "read_file", {"file_path": "module.py"})
    replacement = "return value + 1  #" + "padding" * 600
    _, edit = run_tool(manager, agent, "edit_file", {
        "file_path": "module.py", "old_string": "return value + 0", "new_string": replacement}, "large")
    view = state_record(project(manager, agent, hooks, io, read + edit))
    assert view["latest_edit_excerpt"]["truncated"]
    assert len(view["latest_edit_excerpt"]["excerpt"]) == MAX_EDIT_EXCERPT_CHARS
