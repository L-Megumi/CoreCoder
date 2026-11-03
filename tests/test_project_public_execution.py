"""Real shared-Verifier processes; no synthetic stderr-to-success conversion."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from corecoder.workflow import run_verification
from researchcoder.contracts import OutputSlice, VerificationResult
from researchcoder.evaluation.projects.public_execution import (
    PreparedPublicExecution, PublicExecutionPolicy, prepare_public_execution, validate_public_exception,
)
from researchcoder.evaluation.projects.verification import ProjectCheck, ReceiptSpec, validate_receipt
from researchcoder.evaluation.projects.workspace import capture_manifest
from researchcoder.runtime.protocol import ProtectedFile


MODULE = 'def execute():\n    return 3\n'
CHECKER = '''import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
source = root / "module.py"
spec = importlib.util.spec_from_file_location("candidate", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.execute()
print(json.dumps(dict(origin=str(source), value=result)))
raise SystemExit(0 if result == 3 else 1)
'''


def build(tmp_path, module=MODULE, checker=CHECKER, *, timeout=10):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text(module, encoding="utf-8")
    (root / ".public-config").write_text("unchanged\n", encoding="utf-8")
    script = tmp_path / "public-check.py"
    script.write_text(checker, encoding="utf-8")
    check = ProjectCheck(check_id="public-fixture", argv_template=(sys.executable, "-I", "-B", str(script), "{project}"),
        protected_files=(ProtectedFile(path=str(script), sha256=hashlib.sha256(script.read_bytes()).hexdigest()),),
        receipt=ReceiptSpec(kind="public_json", origin_field="origin", origin_relative_path="module.py",
            required_fields=("origin", "value"), field_types=(("value", "int"),)), timeout_seconds=timeout)
    prepared = prepare_public_execution(check, root, tmp_path / "probe", job_id="job", task_id="coding-0")
    return root, check, prepared


def execute(root, prepared):
    def guard(command, project):
        assert tuple(command) == prepared.command.argv
        assert project == root.resolve()
        prepared.check_protected()
    raw = run_verification(prepared.command.argv, root, prepared.command.timeout_seconds, guard=guard)
    result = VerificationResult(job_id=prepared.job_id, task_id=prepared.task_id, attempt_id="attempt-0",
        verification_id="public-verification", spec=prepared.command, status=raw["status"],
        exit_code=raw.get("exit_code"), stdout=OutputSlice(**raw.get("stdout", {})),
        stderr=OutputSlice(**raw.get("stderr", {})), checked_workspace_revision=raw["checked_workspace_revision"],
        elapsed_seconds=raw["elapsed_seconds"], timeout_seconds=raw.get("timeout_seconds"))
    return result


def exception_module(name="TypeError"):
    return f'def execute():\n    raise {name}("candidate failure")\n'


def test_public_policy_version_roundtrip_and_illegal_modes():
    policy = PublicExecutionPolicy()
    assert PublicExecutionPolicy.from_json(policy.to_json()) == policy
    for data in ({"mode": "anything", "schema_version": 1},
                 {"mode": "candidate-exception-v1", "schema_version": 2},
                 policy.to_dict() | {"allow_hidden_oracle": True}):
        with pytest.raises((TypeError, ValueError)):
            PublicExecutionPolicy.from_dict(data)


def test_prepared_execution_roundtrip_and_fixed_existing_verifier_command(tmp_path):
    root, check, prepared = build(tmp_path)
    assert PreparedPublicExecution.from_json(prepared.to_json()) == prepared
    assert prepared.command.argv[:3] == (sys.executable, "-I", "-B")
    assert prepared.command.timeout_seconds == check.timeout_seconds
    assert prepared.command.argv[-1] == check.argv_template[3]
    assert len(prepared.protected_files) == 3
    prepared.check_protected()
    request = json.loads(Path(prepared.request_path).read_text())
    assert request["original_argv"] == list(check.command(root).argv)
    assert request["checker_arguments"] == [str(root.resolve())]
    assert set(prepared.read_files()) == {"probe", "request"}
    for changes in ({"request_path": str(root / "execution-request.json")},
                    {"command": check.command(root)}, {"protected_files": prepared.protected_files[:-1]},
                    {"receipt_path": str(tmp_path / "other.json")}):
        with pytest.raises(ValueError):
            replace(prepared, **changes)


@pytest.mark.parametrize("value", (3, 4))
def test_normal_checker_stdout_exit_and_json_semantics_unchanged(tmp_path, value):
    root, check, prepared = build(tmp_path, module=MODULE.replace("return 3", f"return {value}"))
    original = run_verification(check.command(root).argv, root)
    result = execute(root, prepared)
    assert result.status == original["status"] and result.exit_code == original["exit_code"]
    assert result.stdout == OutputSlice(**original["stdout"]) and result.stderr == OutputSlice(**original["stderr"])
    receipt = validate_receipt(result, check.receipt, root)
    assert receipt["passed"] is (value == 3)
    with pytest.raises(ValueError):
        validate_public_exception(result, prepared, check, root)


@pytest.mark.parametrize("name", ("TypeError", "ValueError", "RuntimeError", "AssertionError", "AttributeError",
                                 "IndexError", "KeyError", "ZeroDivisionError", "OverflowError"))
def test_real_candidate_exceptions_are_failed_public_evidence_not_success(tmp_path, name):
    root, check, prepared = build(tmp_path, module=exception_module(name))
    before = capture_manifest(root)
    result = execute(root, prepared)
    with pytest.raises(ValueError, match="Empty checker output"):
        validate_receipt(result, check.receipt, root)
    value = validate_public_exception(result, prepared, check, root)
    assert value["passed"] is False and value["status"] == result.status == "failed" and result.exit_code == 1
    assert value["exception_type"] == name
    assert value["candidate_frames"][0]["path"] == "module.py"
    assert value["candidate_frames"][0]["function"] == "execute"
    assert value["source_manifest_sha256"] == before.digest == capture_manifest(root).digest
    assert value["attempt_id"] == result.attempt_id
    assert f"{name}:" in result.stderr.text
    assert value["execution_receipt_sha256"] == hashlib.sha256(Path(prepared.receipt_path).read_bytes()).hexdigest()


def test_prepared_command_accepts_authorized_candidate_edit_before_actual_verification(tmp_path):
    root, check, prepared = build(tmp_path)
    (root / "module.py").write_text(exception_module())
    result = execute(root, prepared)
    value = validate_public_exception(result, prepared, check, root)
    assert value["source_manifest_sha256"] == capture_manifest(root).digest


def test_preexisting_dependency_type_error_is_observation_not_root_cause_claim(tmp_path):
    module = 'from rich._loop import loop_first\ndef execute():\n    return next(loop_first(3))\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    receipt = validate_public_exception(result, prepared, check, root)
    assert receipt["exception_type"] == "TypeError" and receipt["attribution"] == "candidate_dependency_execution"
    assert receipt["dependency_frames"] and receipt["candidate_frames"]
    assert "root_cause" not in receipt and receipt["passed"] is False
    record = prepared.read_record()
    assert record["dependency_snapshot"]["complete"] is True
    assert record["dependency_snapshot"]["entries"] > 0
    assert record["frames"][-1]["dependency_path"].endswith("rich/_loop.py")


def test_helper_outside_candidate_and_interpreter_packages_is_not_trusted_dependency(tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text('def fail():\n    raise TypeError("helper bug")\n')
    module = ('import importlib.util\n'
              f'spec = importlib.util.spec_from_file_location("helper", {str(helper)!r})\n'
              'helper = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(helper)\n'
              'def execute():\n    return helper.fail()\n')
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    with pytest.raises(ValueError, match="terminal frame"):
        validate_public_exception(result, prepared, check, root)


def test_dependency_snapshot_and_stat_tampering_cannot_authorize_feedback(tmp_path):
    module = 'from rich._loop import loop_first\ndef execute():\n    return next(loop_first(3))\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    path = Path(prepared.receipt_path)
    original = prepared.read_record()
    for mutation in (lambda r: r["dependency_snapshot"].update(complete=False),
                     lambda r: r["dependency_snapshot"].update(interpreter_executable="/untrusted/python"),
                     lambda r: r["frames"][-1].update(dependency_before=[0, 0, 0, 0]),
                     lambda r: r["frames"][-1].update(source_sha256="0" * 64),
                     lambda r: r["frames"][-1].update(source_code_matched=False)):
        record = json.loads(json.dumps(original))
        mutation(record)
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            validate_public_exception(result, prepared, check, root)


@pytest.mark.parametrize("name", ("ModuleNotFoundError", "ImportError", "OSError", "MemoryError", "KeyboardInterrupt", "SystemExit"))
def test_infrastructure_control_flow_and_resource_failures_do_not_authorize_feedback(tmp_path, name):
    root, check, prepared = build(tmp_path, module=exception_module(name))
    result = execute(root, prepared)
    with pytest.raises(ValueError):
        validate_public_exception(result, prepared, check, root)


def test_checker_bug_after_candidate_return_is_not_candidate_failure(tmp_path):
    checker = CHECKER.replace("print(json.dumps", 'raise TypeError("checker bug")\nprint(json.dumps')
    root, check, prepared = build(tmp_path, checker=checker)
    result = execute(root, prepared)
    with pytest.raises(ValueError, match="outside candidate"):
        validate_public_exception(result, prepared, check, root)


def test_checker_callback_failure_is_not_attributed_to_candidate_ancestor(tmp_path):
    module = 'def execute(callback):\n    return callback()\n'
    checker = CHECKER.replace("result = module.execute()",
        'def callback():\n    raise TypeError("checker callback bug")\nresult = module.execute(callback)')
    root, check, prepared = build(tmp_path, module=module, checker=checker)
    result = execute(root, prepared)
    assert "checker callback bug" in result.stderr.text
    with pytest.raises(ValueError, match="terminal frame"):
        validate_public_exception(result, prepared, check, root)


def test_printed_traceback_and_exit_do_not_prove_candidate_exception(tmp_path):
    module = 'import sys\ndef execute():\n    print("Traceback: module.py TypeError candidate failure", file=sys.stderr)\n    raise SystemExit(1)\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    assert "TypeError" in result.stderr.text
    with pytest.raises(ValueError, match="review"):
        validate_public_exception(result, prepared, check, root)


def test_forged_code_filename_does_not_match_actual_candidate_source(tmp_path):
    module = 'def execute():\n    exec(compile("raise TypeError(123)", __file__, "exec"))\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    with pytest.raises(ValueError, match="provenance"):
        validate_public_exception(result, prepared, check, root)


def test_candidate_change_inside_public_execution_is_rejected(tmp_path):
    module = 'from pathlib import Path\ndef execute():\n    Path(__file__).with_name(".public-config").write_text("changed")\n    raise TypeError("candidate failure")\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    with pytest.raises(ValueError, match="revision|changed"):
        validate_public_exception(result, prepared, check, root)


def test_candidate_change_after_public_execution_is_rejected(tmp_path):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    (root / "module.py").write_text(MODULE)
    with pytest.raises(ValueError, match="stale"):
        validate_public_exception(result, prepared, check, root)


@pytest.mark.parametrize("target", ("request", "probe", "checker"))
def test_protected_execution_inputs_are_verified_after_process_completion(tmp_path, target):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    path = Path(prepared.request_path if target == "request" else
                prepared.command.argv[3] if target == "probe" else check.protected_files[0].path)
    path.write_text(path.read_text() + "\n ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_public_exception(result, prepared, check, root)


@pytest.mark.parametrize("field,value", (("job_id", "other-job"), ("task_id", "other-task"),
    ("invocation_id", "other-invocation"), ("request_sha256", "0" * 64),
    ("checker_sha256", "0" * 64), ("exception_module", "custom"), ("candidate_after", [])))
def test_mismatched_or_ambiguous_receipts_are_refused(tmp_path, field, value):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    path = Path(prepared.receipt_path)
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        validate_public_exception(result, prepared, check, root)


@pytest.mark.parametrize("content", ("", "{", '{"schema_version":1,"schema_version":1}', '[]', '{"extra":NaN}'))
def test_missing_malformed_duplicate_and_nonfinite_receipts_are_not_feedback(tmp_path, content):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    Path(prepared.receipt_path).write_text(content)
    assert prepared.read_files()["receipt"] == content.encode()
    with pytest.raises((ValueError, TypeError)):
        validate_public_exception(result, prepared, check, root)


def test_receipt_file_symlink_is_refused(tmp_path):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    path = Path(prepared.receipt_path)
    other = tmp_path / "copied-receipt.json"
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="symlink"):
        validate_public_exception(result, prepared, check, root)


def test_public_probe_never_reuses_existing_execution_output(tmp_path):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    validate_public_exception(result, prepared, check, root)
    receipt = Path(prepared.receipt_path).read_bytes()
    second = execute(root, prepared)
    assert second.status == "failed" and "already exists" in second.stderr.text
    assert Path(prepared.receipt_path).read_bytes() == receipt
    with pytest.raises(ValueError, match="output differs"):
        validate_public_exception(second, prepared, check, root)


def test_native_output_not_observed_by_probe_fails_closed(tmp_path):
    module = 'import os\ndef execute():\n    os.write(2, b"native diagnostic\\n")\n    raise TypeError("candidate")\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    assert "native diagnostic" in result.stderr.text
    with pytest.raises(ValueError, match="output differs"):
        validate_public_exception(result, prepared, check, root)


def test_missing_receipt_after_hard_process_exit_stays_unknown(tmp_path):
    module = 'import os\ndef execute():\n    os._exit(1)\n'
    root, check, prepared = build(tmp_path, module=module)
    result = execute(root, prepared)
    assert result.status == "failed"
    with pytest.raises(ValueError, match="missing"):
        validate_public_exception(result, prepared, check, root)


def test_timeout_never_becomes_a_business_failure_receipt(tmp_path):
    module = 'import time\ndef execute():\n    time.sleep(3)\n    raise TypeError("late")\n'
    root, check, prepared = build(tmp_path, module=module, timeout=1)
    result = execute(root, prepared)
    assert result.status == "timeout"
    with pytest.raises(ValueError, match="complete failed"):
        validate_public_exception(result, prepared, check, root)


def test_truncated_output_or_foreign_result_cannot_reuse_a_receipt(tmp_path):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    bad = (replace(result, job_id="foreign"), replace(result, task_id="foreign"),
           replace(result, stderr=OutputSlice(text="partial", truncated=True, original_chars=20, omitted_chars=13)),
           replace(result, spec=check.command(root)), replace(result, exit_code=2))
    for value in bad:
        with pytest.raises(ValueError):
            validate_public_exception(value, prepared, check, root)


def test_different_oracle_configuration_cannot_reuse_public_execution(tmp_path):
    root, check, prepared = build(tmp_path, module=exception_module())
    result = execute(root, prepared)
    hidden = replace(check, check_id="private-f2p")
    with pytest.raises(ValueError, match="identity"):
        validate_public_exception(result, prepared, hidden, root)


def test_unknown_interpreter_profile_and_reused_directory_are_refused(tmp_path):
    root, check, prepared = build(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_public_execution(check, root, tmp_path / "probe", job_id="job", task_id="coding-1")
    unsafe = replace(check, argv_template=(sys.executable, "-B", check.argv_template[3], "{project}"))
    with pytest.raises(ValueError, match="profile"):
        prepare_public_execution(unsafe, root, tmp_path / "unsafe", job_id="job", task_id="coding-1")
