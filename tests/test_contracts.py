"""Task contracts preserve identities and execution semantics through JSON."""

from dataclasses import FrozenInstanceError, replace

import pytest

from researchcoder.contracts import (
    ArtifactRef,
    BudgetSpec,
    DiagnosticOptions,
    EvidenceRef,
    ExecutionPolicy,
    OutputSlice,
    TaskInput,
    TaskResult,
    TaskSpec,
    UsageSummary,
    VerificationResult,
    VerificationSpec,
    WorkspaceRef,
)


def artifact(job="job"):
    return ArtifactRef(job_id=job, artifact_id="log", storage_key=job + "/log.json", sha256="a" * 64, size_bytes=0)


def workspace():
    return WorkspaceRef(workspace_id="workspace", root="/tmp/project", source_manifest_sha256="b" * 64)


def task():
    return TaskSpec(
        job_id="job",
        task_id="diagnose",
        kind="diagnose",
        objective="Investigate training",
        workspace=workspace(),
        inputs=(TaskInput(role="raw_log", reference=artifact()),),
    )


def verification(**kwargs):
    values = {
        "job_id": "job",
        "task_id": "repair",
        "attempt_id": "attempt-1",
        "verification_id": "check",
        "status": "passed",
        "exit_code": 0,
        "spec": VerificationSpec(argv=("python", "oracle.py")),
        "checked_workspace_revision": "b" * 64,
    }
    return VerificationResult(**(values | kwargs))


@pytest.mark.parametrize(
    "value",
    [
        artifact(),
        workspace(),
        BudgetSpec(),
        ExecutionPolicy(),
        VerificationSpec(argv=("python", "check.py")),
        UsageSummary(),
        DiagnosticOptions(),
        OutputSlice(),
        task(),
        EvidenceRef(
            job_id="job",
            ref_id="job:ev1",
            origin_id="ev1",
            record_type="log_evidence",
            artifact=artifact(),
            record_key="evidence:ev1",
            producer_task_id="diagnose",
            producer_attempt_id="a1",
            relations=(("cause", "job:ev0"),),
        ),
        verification(),
        TaskResult(
            job_id="job",
            task_id="repair",
            attempt_id="attempt-1",
            kind="repair",
            status="incomplete",
            failure_kind="model_error",
            verification=verification(),
            workflow_completed=False,
        ),
    ],
)
def test_every_contract_roundtrips(value):
    assert type(value).from_json(value.to_json()) == value
    assert type(value).from_dict(value.to_dict()) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "../escape"),
        ("schema_version", 2),
        ("schema_version", True),
        ("objective", object()),
        ("budget", object()),
        ("inputs", (object(),)),
        ("kind", "delegate"),
        ("objective", ""),
    ],
)
def test_task_rejects_invalid_values_and_runtime_objects(field, value):
    with pytest.raises((ValueError, TypeError)):
        replace(task(), **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        {"max_tool_calls": True},
        {"max_rounds": 0},
        {"max_context_tokens": 999},
        {"max_report_repairs": 2},
    ],
)
def test_invalid_budget(change):
    with pytest.raises((ValueError, TypeError)):
        BudgetSpec(**change)


@pytest.mark.parametrize("path", ["../job/x", "/tmp/x", "job/../x", "job//x", "job\\x", "job/x:y"])
def test_artifact_key_must_be_logical_job_path(path):
    with pytest.raises(ValueError):
        replace(artifact(), storage_key=path)


def test_unknown_missing_version_and_duplicate_fields_are_rejected():
    for raw in [task().to_dict() | {"extra": 1}, {k: v for k, v in task().to_dict().items() if k != "schema_version"}]:
        with pytest.raises(ValueError):
            TaskSpec.from_dict(raw)
    with pytest.raises(ValueError):
        BudgetSpec.from_json('{"schema_version":1,"schema_version":2}')
    with pytest.raises(ValueError):
        UsageSummary.from_json('{"schema_version":1,"elapsed_seconds":NaN}')


def test_deep_values_are_immutable():
    policy = ExecutionPolicy(writable_paths=["training.py"])
    assert policy.writable_paths == ("training.py",)
    with pytest.raises(FrozenInstanceError):
        policy.profile = "other"


def test_job_roles_and_verification_permissions():
    with pytest.raises(ValueError):
        replace(task(), inputs=())
    with pytest.raises(ValueError):
        replace(task(), inputs=(TaskInput(role="raw_log", reference=artifact("other")),))
    with pytest.raises(ValueError):
        replace(task(), inputs=task().inputs * 2)
    with pytest.raises(ValueError):
        replace(task(), kind="verify", inputs=())
    with pytest.raises(ValueError):
        ExecutionPolicy(allowed_tool_names=("bash",))
    with pytest.raises(ValueError):
        ExecutionPolicy(allowed_tool_names=("agent",))
    with pytest.raises(ValueError):
        VerificationSpec(argv=())


@pytest.mark.parametrize(
    "change",
    [
        {"exit_code": True},
        {"exit_code": 1},
        {"status": "failed", "exit_code": 0},
        {"status": "timeout", "exit_code": None},
        {"checked_workspace_revision": None},
        {"spec": None},
    ],
)
def test_verification_cannot_invent_success(change):
    with pytest.raises((ValueError, TypeError)):
        verification(**change)


def test_partial_workflow_with_passed_test_stays_incomplete():
    result = TaskResult(
        job_id="job",
        task_id="repair",
        attempt_id="attempt-1",
        kind="repair",
        status="incomplete",
        failure_kind="budget_exhausted",
        verification=verification(),
    )
    assert TaskResult.from_json(result.to_json()).status == "incomplete"
    with pytest.raises(ValueError):
        replace(result, status="succeeded", failure_kind=None)
    with pytest.raises(ValueError):
        replace(result, verification=replace(verification(), attempt_id="different"))


@pytest.mark.parametrize("bounds", [(0, 1), (3, 2), (1, None), (None, 2)])
def test_invalid_evidence_ranges(bounds):
    with pytest.raises(ValueError):
        EvidenceRef(
            job_id="job",
            ref_id="ref",
            origin_id="ev",
            record_type="log_evidence",
            artifact=artifact(),
            record_key="evidence:ev",
            producer_task_id="t",
            producer_attempt_id="a",
            start_line=bounds[0],
            end_line=bounds[1],
        )


def test_source_reference_requires_revision():
    with pytest.raises(ValueError):
        EvidenceRef(
            job_id="job",
            ref_id="ref",
            origin_id="obs",
            record_type="source_observation",
            artifact=artifact(),
            record_key="observations:obs",
            producer_task_id="t",
            producer_attempt_id="a",
        )


def test_standalone_verification_result_cannot_report_success_without_command():
    with pytest.raises(ValueError):
        TaskResult(job_id="job", task_id="verify", attempt_id="a1", kind="verify", status="succeeded", workflow_completed=True)
