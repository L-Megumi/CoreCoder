from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import pytest
from researchcoder.contracts import TaskSpec, TaskResult, WorkspaceRef, ExecutionPolicy, VerificationSpec
from researchcoder.runtime.state import TaskRepository


def spec():
    return TaskSpec(
        job_id="job",
        task_id="verify",
        kind="verify",
        objective="Check",
        workspace=WorkspaceRef(workspace_id="p", root="/tmp/project", source_manifest_sha256="a" * 64),
        policy=ExecutionPolicy(allow_verification=True),
        verification=VerificationSpec(argv=("true",)),
    )


@pytest.mark.parametrize(
    "status,reason",
    [
        ("succeeded", None),
        ("failed", "timeout"),
        ("failed", "tool_error"),
        ("incomplete", "budget_exhausted"),
        ("cancelled", "interrupted"),
    ],
)
def test_controller_state_is_distinct_from_result_and_all_terminal_paths(status, reason):
    repo = TaskRepository()
    repo.create_job("job")
    repo.add_task(spec())
    repo.start("job", "verify", "a1")
    assert repo.task("job", "verify").status == "running"
    # Successful diagnosis is used here to avoid claiming a verification without a command result.
    result = (
        TaskResult(
            job_id="job",
            task_id="verify",
            attempt_id="a1",
            kind="verify",
            status=status,
            failure_kind=reason,
            workflow_completed=status == "succeeded",
        )
        if status != "succeeded"
        else None
    )
    if result is None:
        from researchcoder.contracts import VerificationResult

        result = TaskResult(
            job_id="job",
            task_id="verify",
            attempt_id="a1",
            kind="verify",
            status=status,
            workflow_completed=True,
            verification=VerificationResult(
                job_id="job",
                task_id="verify",
                attempt_id="a1",
                verification_id="v",
                status="passed",
                spec=spec().verification,
                exit_code=0,
                checked_workspace_revision="a" * 64,
            ),
        )
    repo.finish(result)
    assert repo.task("job", "verify").status == status and repo.attempt("a1").status == status
    assert repo.result("a1") == result
    with pytest.raises(ValueError):
        repo.finish(result)
    repo.close_job("job")
    with pytest.raises(ValueError):
        repo.add_task(replace(spec(), task_id="other"))


def test_illegal_state_identity_and_duplicate_transitions():
    repo = TaskRepository()
    with pytest.raises(ValueError):
        repo.add_task(spec())
    repo.create_job("job")
    repo.add_task(spec())
    with pytest.raises(ValueError):
        repo.close_job("job")
    with pytest.raises(ValueError):
        repo.add_task(spec())
    bad = TaskResult(job_id="job", task_id="verify", attempt_id="a1", kind="verify", status="failed", failure_kind="tool_error")
    with pytest.raises(ValueError):
        repo.finish(bad)
    repo.start("job", "verify", "a1")
    with pytest.raises(ValueError):
        repo.start("job", "verify", "a2")
    with pytest.raises(ValueError):
        repo.finish(replace(bad, attempt_id="other"))
    with pytest.raises(ValueError):
        repo.finish(replace(bad, kind="diagnose"))
    assert repo.task("job", "verify").status == "running"


def test_repository_is_controller_owned_not_worker_thread_mutable():
    repo = TaskRepository()
    with ThreadPoolExecutor() as pool:
        with pytest.raises(RuntimeError):
            pool.submit(repo.create_job, "job").result()
