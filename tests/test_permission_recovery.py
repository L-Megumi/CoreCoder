"""Metadata-only permission correction; no provider or unguarded file reads."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from corecoder.permission import PermissionDenied, validate_permission_receipt
from researchcoder.runtime.permissions import (
    PermissionRecoverySession, PermissionRecoverySpec,
    classify_permission_denial, normalized_target_sha256,
)


@pytest.fixture
def public_check(tmp_path):
    path = tmp_path / "public-check.py"
    path.write_text("PRIVATE_BODY_MARKER = 'never expose this body'\n", encoding="utf-8")
    return path


def enabled(path, **kwargs):
    return PermissionRecoverySpec(
        public_check_paths=(path.as_posix(),), remaining_corrections=1,
        mode="bounded-public-read-v1", **kwargs,
    )


def test_policy_roundtrip_versioned_hash_and_runtime_free_fields(public_check):
    spec = enabled(public_check)
    assert PermissionRecoverySpec.from_json(spec.to_json()) == spec
    assert PermissionRecoverySpec.from_dict(spec.to_dict()) == spec
    assert spec.sha256 == hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest()
    assert replace(spec, remaining_corrections=0).sha256 != spec.sha256
    assert set(spec.to_dict()) == {"schema_version", "mode", "public_check_paths", "remaining_corrections"}
    with pytest.raises((ValueError, TypeError)):
        PermissionRecoverySpec.from_dict({**spec.to_dict(), "client": object()})


@pytest.mark.parametrize("change", [
    {"remaining_corrections": -1}, {"remaining_corrections": 2},
    {"remaining_corrections": True}, {"remaining_corrections": 1.0},
    {"mode": "read_external"}, {"schema_version": 2},
    {"public_check_paths": ("relative.py",)},
    {"public_check_paths": ("/tmp/../check.py",)},
    {"public_check_paths": ("/tmp//check.py",)},
    {"public_check_paths": ("//server/check.py",)},
    {"public_check_paths": ("~/check.py",)},
    {"public_check_paths": ("/tmp/check.py/",)},
    {"public_check_paths": ("/tmp/check.py\x00",)},
    {"public_check_paths": ("/tmp/check\n.py",)},
    {"public_check_paths": ("/tmp/check.py", "/tmp/check.py")},
    {"public_check_paths": ("/",)},
    {"public_check_paths": tuple(f"/tmp/check{i}.py" for i in range(17))},
])
def test_invalid_policy_values_are_rejected(change):
    with pytest.raises((TypeError, ValueError)):
        PermissionRecoverySpec(**change)


@pytest.mark.parametrize("path", [
    "/tmp/.env", "/tmp/.env.test", "/tmp/.ssh/check.py", "/tmp/.git/config",
    "/tmp/.aws/credentials", "/tmp/ID_ED25519", "/tmp/secrets.json",
    "/tmp/credentials.json", "/tmp/key.pem", "/tmp/key.p12", "/tmp/key.key",
])
def test_sensitive_path_cannot_become_a_public_check(path):
    with pytest.raises(ValueError):
        PermissionRecoverySpec(public_check_paths=(path,))
    denial = classify_permission_denial(PermissionRecoverySpec(), "read_file", path)
    assert denial["action"] == "stop"
    assert denial["reason_code"] == "sensitive_target"
    assert denial["target_category"] == "sensitive"
    assert path not in json.dumps(denial)


def test_default_off_preserves_rejection_even_for_explicit_public_path(public_check):
    spec = PermissionRecoverySpec(public_check_paths=(public_check.as_posix(),), remaining_corrections=1)
    denial = classify_permission_denial(spec, "read_file", public_check.as_posix())
    assert denial["action"] == "stop"
    assert denial["reason_code"] == "policy_disabled"
    assert denial["correction_index"] == 0


def test_exact_public_read_is_denied_with_one_feedback_and_never_reads_content(public_check, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Permission classifier must not open file contents")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    denial = classify_permission_denial(enabled(public_check), "read_file", public_check.as_posix())
    validate_permission_receipt(denial)
    assert denial["reason_code"] == "public_read_path_mismatch"
    assert denial["target_category"] == "public_check"
    assert denial["action"] == "correct_parameters"
    assert denial["correction_index"] == 1
    assert denial["tool_executed"] is False
    assert denial["target_sha256"] == hashlib.sha256(public_check.as_posix().encode()).hexdigest()
    assert denial["denial_id"].startswith("denial-")
    assert public_check.as_posix() not in json.dumps(denial)
    assert "PRIVATE_BODY_MARKER" not in json.dumps(denial)
    exception = PermissionDenied.from_receipt(denial)
    assert exception.receipt == denial
    assert public_check.as_posix() not in str(exception)


def test_session_budget_is_consumed_once_and_unique_denials_are_auditable(public_check):
    session = PermissionRecoverySession(enabled(public_check))
    first = session.deny("read_file", public_check.as_posix())
    second = session.deny("read_file", public_check.as_posix())
    assert first["action"] == "correct_parameters"
    assert second["action"] == "stop"
    assert second["reason_code"] == "correction_exhausted"
    assert second["correction_index"] == 0
    assert session.corrections_used == 1 and session.remaining_corrections == 0
    assert first["denial_id"] != second["denial_id"]
    first["action"] = "stop"
    assert session.corrections_used == 1
    assert session.spec.remaining_corrections == 1  # Immutable admission is separate from mutable usage.


def test_controller_carried_job_allowance_does_not_reset_for_next_attempt_or_expert(public_check):
    initial = enabled(public_check)
    first_attempt = PermissionRecoverySession(initial)
    assert first_attempt.deny("read_file", public_check.as_posix())["action"] == "correct_parameters"
    # Controller persists the debit and supplies the residual to every new Worker.
    residual = replace(initial, remaining_corrections=first_attempt.remaining_corrections)
    for _ in ("second-attempt", "code-expert", "log-expert"):
        later = PermissionRecoverySession(PermissionRecoverySpec.from_json(residual.to_json()))
        assert later.deny("read_file", public_check.as_posix())["action"] == "stop"
        assert later.corrections_used == 0


def test_concurrent_calls_cannot_double_spend_session_allowance(public_check):
    session = PermissionRecoverySession(enabled(public_check))
    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(executor.map(lambda _: session.deny("read_file", public_check.as_posix()), range(20)))
    assert sum(r["action"] == "correct_parameters" for r in receipts) == 1
    assert len({r["denial_id"] for r in receipts}) == 20
    assert session.corrections_used == 1


@pytest.mark.parametrize("tool", ["write_file", "edit_file", "bash", "agent", "grep", "glob", "unknown", "read_project_file"])
def test_non_read_file_tools_never_receive_correction(public_check, tool):
    session = PermissionRecoverySession(enabled(public_check))
    denial = session.deny(tool, public_check.as_posix())
    assert denial["action"] == "stop"
    assert denial["tool_executed"] is False
    assert denial["correction_index"] == 0
    assert session.corrections_used == 0


@pytest.mark.parametrize("reason", [
    "write_not_allowed", "current_read_required", "tool_not_allowed", "guard_rejected",
    "sensitive_target", "path_unavailable", "public_read_path_mismatch", "unknown_user_reason",
])
def test_public_allowlist_cannot_downgrade_another_guard_failure(public_check, reason):
    denial = classify_permission_denial(enabled(public_check), "read_file", public_check.as_posix(), reason_code=reason)
    assert denial["action"] == "stop" and denial["correction_index"] == 0


def test_unknown_external_and_hidden_oracle_targets_never_qualify(public_check, tmp_path, monkeypatch):
    hidden = tmp_path / "hidden-check.py"
    hidden.write_text("HIDDEN_ANSWER_MARKER\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unknown external targets must not even be inspected")

    monkeypatch.setattr(Path, "lstat", forbidden)
    for target in (hidden.as_posix(), "/some/dependency/library.py", "/old/source/project.py"):
        denial = classify_permission_denial(enabled(public_check), "read_file", target)
        assert denial["action"] == "stop" and denial["reason_code"] == "unknown_target"
        assert target not in json.dumps(denial)
        assert "HIDDEN_ANSWER_MARKER" not in json.dumps(denial)


@pytest.mark.parametrize("target", [None, {}, [], 2, "", "relative.py", "~/x.py", "${ROOT}/x.py", "/tmp/\x00x", "/tmp/\ud800x", "//host/x", "C:\\temp\\x.py"])
def test_invalid_tool_targets_are_safely_rejected(public_check, target):
    denial = classify_permission_denial(enabled(public_check), "read_file", target)
    assert denial["action"] == "stop" and denial["reason_code"] == "unknown_target"
    assert denial["target_sha256"] is None


def test_lexical_aliases_never_qualify_even_when_target_digest_matches(public_check):
    target = public_check.as_posix()
    for alias in (str(public_check.parent) + "/./" + public_check.name,
                  str(public_check.parent) + "/unused/../" + public_check.name,
                  str(public_check.parent) + "//" + public_check.name):
        assert normalized_target_sha256(alias) == normalized_target_sha256(target)
        denial = classify_permission_denial(enabled(public_check), "read_file", alias)
        assert denial["action"] == "stop" and denial["reason_code"] == "unknown_target"


def test_missing_or_directory_target_is_not_correctable(public_check, tmp_path):
    spec = enabled(public_check)
    public_check.unlink()
    missing = classify_permission_denial(spec, "read_file", public_check.as_posix())
    assert missing["action"] == "stop" and missing["reason_code"] == "path_unavailable"
    directory = classify_permission_denial(enabled(tmp_path), "read_file", tmp_path.as_posix())
    assert directory["action"] == "stop" and directory["reason_code"] == "path_unavailable"


def test_file_and_ancestor_symlinks_are_not_correctable(public_check, tmp_path):
    linked = tmp_path / "linked-check.py"
    linked.symlink_to(public_check)
    link_dir = tmp_path / "linked-directory"
    real_dir = tmp_path / "real-directory"
    real_dir.mkdir()
    (real_dir / "check.py").write_text("print('public')", encoding="utf-8")
    link_dir.symlink_to(real_dir, target_is_directory=True)
    for target in (linked, link_dir / "check.py"):
        denial = classify_permission_denial(enabled(target), "read_file", target.as_posix())
        assert denial["action"] == "stop" and denial["target_category"] == "symlink"


def test_unknown_tool_and_reason_cannot_inject_receipt_or_feedback(public_check):
    marker = "SECRET_ARBITRARY_PROVIDER_TEXT"
    denial = classify_permission_denial(enabled(public_check), marker, public_check.as_posix(), reason_code=marker)
    assert denial["tool"] == "unknown" and denial["action"] == "stop"
    assert marker not in json.dumps(denial)
    assert marker not in str(PermissionDenied.from_receipt(denial))


def test_forged_receipt_cannot_claim_execution_or_add_unknown_fields(public_check):
    denial = classify_permission_denial(enabled(public_check), "read_file", public_check.as_posix())
    for change in ({"tool_executed": True}, {"original_path": public_check.as_posix()},
                   {"tool": "write_file"}, {"correction_index": 0}, {"target_sha256": None}):
        with pytest.raises((ValueError, TypeError)):
            validate_permission_receipt({**denial, **change})


def test_non_contract_session_and_classifier_input_is_rejected(public_check):
    with pytest.raises(TypeError):
        PermissionRecoverySession({"remaining_corrections": 1})
    with pytest.raises(TypeError):
        classify_permission_denial({}, "read_file", public_check.as_posix())
