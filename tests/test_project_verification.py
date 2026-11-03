from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import OutputSlice, VerificationResult, VerificationSpec
from researchcoder.evaluation.projects.verification import ProjectCheck, ReceiptSpec, run_check, validate_receipt
from researchcoder.runtime.protocol import ProtectedFile


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def checks_spec(**changes):
    return replace(ReceiptSpec(kind="checks_json", origin_field="origin", origin_relative_path="module.py",
                               exact_fields=(("schema_version", 1), ("profile", "f2p")),
                               expected_checks=("one", "two"), check_name_field="case",
                               digest_field="source_sha256", digest_relative_path="module.py"), **changes)


def report_for(root, passed=True):
    return dict(schema_version=1, profile="f2p", origin=str(root / "module.py"),
                source_sha256=hashlib.sha256((root / "module.py").read_bytes()).hexdigest(),
                checks=[dict(case="one", passed=passed), dict(case="two", passed=True)], passed=passed)


def verification(root, report, *, status=None, exit_code=None, raw=False, truncated=False):
    passed = not isinstance(report, dict) or report.get("passed", True)
    text = report if raw else json.dumps(report)
    status = status or ("passed" if passed else "failed")
    if exit_code is None and status in ("passed", "failed"):
        exit_code = 0 if status == "passed" else 1
    return VerificationResult(job_id="job", task_id="task", attempt_id="attempt", verification_id="verification",
                              status=status, exit_code=exit_code,
                              spec=VerificationSpec(argv=(sys.executable, "checker.py")),
                              stdout=OutputSlice(text=text, truncated=truncated, original_chars=len(text) + (5 if truncated else 0),
                                                 omitted_chars=5 if truncated else 0),
                              checked_workspace_revision=workspace_revision(root))


@pytest.mark.parametrize("passed", (True, False))
def test_independent_receipt_preserves_real_boolean_and_exit(candidate, passed):
    value = verification(candidate, report_for(candidate, passed))
    receipt = validate_receipt(value, checks_spec(), candidate)
    assert receipt["passed"] is passed
    assert receipt["check_count"] == 2
    assert receipt["reported_source_sha256_verified"] is True
    assert receipt["source_sha256"] == report_for(candidate)["source_sha256"]


def test_named_check_and_per_check_profile_format_is_generic(candidate):
    report = report_for(candidate)
    report["oracle_version"] = "fixture-v1"
    for row in report["checks"]:
        row["check"] = row.pop("case")
        row["profile"] = "f2p"
    spec = checks_spec(check_name_field="check", exact_fields=(("oracle_version", "fixture-v1"),),
                       check_exact_fields=(("profile", "f2p"),))
    assert validate_receipt(verification(candidate, report), spec, candidate)["passed"]
    report["checks"][0]["profile"] = "p2p"
    with pytest.raises(ValueError, match="field mismatch"):
        validate_receipt(verification(candidate, report), spec, candidate)


@pytest.mark.parametrize("fault", ("empty", "duplicate", "missing", "string_bool", "aggregate", "schema_bool", "profile", "digest"))
def test_independent_invalid_receipts_are_not_business_failures(candidate, fault):
    report = report_for(candidate)
    if fault == "empty":
        report["checks"] = []
    elif fault == "duplicate":
        report["checks"][1]["case"] = "one"
    elif fault == "missing":
        report["checks"][1]["case"] = "unexpected"
    elif fault == "string_bool":
        report["checks"][0]["passed"] = "true"
    elif fault == "aggregate":
        report["checks"][0]["passed"] = False
    elif fault == "schema_bool":
        report["schema_version"] = True
    elif fault == "profile":
        report["profile"] = "p2p"
    elif fault == "digest":
        report["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_receipt(verification(candidate, report), checks_spec(), candidate)


@pytest.mark.parametrize("raw", ("", "{}", "[]", "not-json", '{"passed":true,"passed":false}', '{"value":NaN}'))
def test_missing_or_invalid_json_rejected(candidate, raw):
    with pytest.raises(ValueError):
        validate_receipt(verification(candidate, raw, raw=True), checks_spec(), candidate)


def test_nonbusiness_exit_truncation_and_stale_workspace_rejected(candidate):
    raw = report_for(candidate)
    values = (verification(candidate, raw, status="failed", exit_code=2),
              verification(candidate, raw, status="interrupted"),
              verification(candidate, raw, truncated=True))
    for value in values:
        with pytest.raises(ValueError):
            validate_receipt(value, checks_spec(), candidate)
    value = verification(candidate, raw)
    (candidate / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        validate_receipt(value, checks_spec(), candidate)


def test_foreign_candidate_origin_rejected(candidate, tmp_path):
    other = tmp_path / "other.py"
    other.write_text("VALUE = 1\n", encoding="utf-8")
    report = report_for(candidate)
    report["origin"] = str(other)
    with pytest.raises(ValueError, match="origin mismatch"):
        validate_receipt(verification(candidate, report), checks_spec(), candidate)


def test_public_receipt_retains_exit_semantics_without_hidden_assertions(candidate):
    spec = ReceiptSpec(kind="public_json", origin_field="origin", origin_relative_path="module.py",
                       required_fields=("origin", "left", "right", "equal"),
                       field_types=(("left", "list"), ("right", "list"), ("equal", "bool")),
                       nonempty_fields=("left", "right"))
    report = dict(origin=str(candidate / "module.py"), left=[1.0], right=[1.0], equal=True)
    for status in ("passed", "failed"):
        receipt = validate_receipt(verification(candidate, report, status=status), spec, candidate)
        assert receipt["passed"] is (status == "passed")
        assert receipt["reported_source_sha256_verified"] is False
    for bad in (dict(report, equal="true"), dict(report, left=[]), {k: v for k, v in report.items() if k != "right"}):
        with pytest.raises(ValueError):
            validate_receipt(verification(candidate, bad), spec, candidate)


def test_receipt_contract_roundtrip_and_unknown_fields():
    spec = checks_spec()
    assert ReceiptSpec.from_json(spec.to_json()) == spec
    with pytest.raises(ValueError):
        ReceiptSpec.from_dict(spec.to_dict() | {"hidden_extra": "not accepted"})
    for changes in ({"origin_relative_path": "../hidden.py"}, {"expected_checks": ()},
                    {"digest_field": None}, {"expected_checks": ("one", "one")}):
        with pytest.raises(ValueError):
            replace(spec, **changes)


def make_check(tmp_path, candidate, *, passed=True, empty=False):
    checker = tmp_path / "checker.py"
    checker.write_text(
        "import hashlib,json,pathlib,sys\n"
        "p=pathlib.Path(sys.argv[1]).resolve()/'module.py'\n"
        f"passed={passed!r}\n"
        "r=dict(schema_version=1,profile='f2p',origin=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),"
        "checks=[dict(case='one',passed=passed),dict(case='two',passed=True)],passed=passed)\n"
        + ("" if empty else "print(json.dumps(r))\n")
        + "raise SystemExit(0 if passed else 1)\n", encoding="utf-8")
    return ProjectCheck(check_id="fixture-independent", argv_template=(sys.executable, "-I", "-B", str(checker), "{project}"),
                        protected_files=(ProtectedFile(path=str(checker), sha256=hashlib.sha256(checker.read_bytes()).hexdigest()),),
                        receipt=checks_spec(), timeout_seconds=10)


def test_project_check_roundtrip_fixed_command_and_protected_hash(candidate, tmp_path):
    check = make_check(tmp_path, candidate)
    assert ProjectCheck.from_json(check.to_json()) == check
    assert check.command(candidate).argv[-1] == str(candidate.resolve())
    assert check.command(candidate).oracle_profile == check.check_id
    check.check_protected()
    with pytest.raises(ValueError):
        ProjectCheck.from_dict(check.to_dict() | {"environment": {"SECRET": "no"}})
    for changes in ({"argv_template": (sys.executable, "-c", "{project}")},
                    {"argv_template": (sys.executable, str(tmp_path / "checker.py"), "x{project}")},
                    {"protected_files": ()}, {"timeout_seconds": 301}):
        with pytest.raises(ValueError):
            replace(check, **changes)
    Path(check.protected_files[0].path).write_text("raise SystemExit(0)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        check.check_protected()


@pytest.mark.parametrize("passed", (True, False))
def test_check_uses_actual_harness_worker_and_existing_verifier(candidate, tmp_path, passed):
    check = make_check(tmp_path, candidate, passed=passed)
    io = ArtifactIO(tmp_path / "artifacts")
    execution = run_check(check, candidate, io, "project-check", "trusted-verification")
    assert execution.receipt_error is None
    assert execution.receipt["passed"] is passed
    assert execution.result.status == ("succeeded" if passed else "failed")
    assert execution.record["worker_pid"] > 0
    assert execution.result.usage.llm_calls == execution.record["budget"]["model_calls"] == 0
    assert execution.record["budget"]["tool_calls"] == 1
    assert execution.to_dict()["result"] == execution.result.to_dict()
    for artifact in execution.result.outputs:
        io.read_bytes(artifact)


def test_empty_successful_process_is_oracle_invalid_without_rewriting_result(candidate, tmp_path):
    check = make_check(tmp_path, candidate, empty=True)
    execution = run_check(check, candidate, ArtifactIO(tmp_path / "artifacts"), "empty-check", "verification")
    assert execution.result.status == "succeeded"
    assert execution.result.verification.status == "passed"
    assert execution.receipt is None
    assert execution.receipt_error == "Empty checker output"


def test_tampered_checker_cannot_produce_a_valid_receipt(candidate, tmp_path):
    check = make_check(tmp_path, candidate)
    Path(check.protected_files[0].path).write_text("raise SystemExit(0)\n", encoding="utf-8")
    execution = run_check(check, candidate, ArtifactIO(tmp_path / "artifacts"), "tampered-check", "verification")
    assert execution.result.status == "incomplete"
    assert execution.result.verification is None
    assert execution.receipt is None and execution.receipt_error


def test_cancelled_check_keeps_cancelled_state_and_no_receipt(candidate, tmp_path):
    check = make_check(tmp_path, candidate)
    cancel = threading.Event()
    cancel.set()
    execution = run_check(check, candidate, ArtifactIO(tmp_path / "artifacts"), "cancelled-check", "verification", cancel_event=cancel)
    assert execution.result.status == "cancelled"
    assert execution.receipt is None and execution.receipt_error
