"""Independent evaluator checks reuse the existing Adapter/public verifier."""

import uuid
from corecoder.workflow import workspace_revision
from researchcoder.adapters.corecoder import CoreCoderAdapter, AdapterContext
from researchcoder.contracts import (
    TaskSpec,
    WorkspaceRef,
    ExecutionPolicy,
    VerificationSpec,
)
from researchcoder.runtime.protocol import ProtectedFile
from .dataset import DATA, ORACLE, sha_bytes


def protected_inputs(case=None):
    from .complexity.dataset import is_complexity, protected_paths
    paths = protected_paths() if case is not None and is_complexity(case) else (
        ORACLE, DATA / "cases.json", DATA / "support/legacy.py", DATA / "support/case.json",
    )
    return tuple(
        ProtectedFile(path=str(p.resolve()), sha256=sha_bytes(p.read_bytes()))
        for p in paths
    )


def check_protected(protected):
    from pathlib import Path

    if any(sha_bytes(Path(p.path).read_bytes()) != p.sha256 for p in protected):
        raise ValueError("Benchmark oracle changed")


def oracle_command(case, project, python, profile):
    if profile not in ("discovery", "holdout", "regression"):
        raise ValueError("Unknown fixed oracle profile")
    from .complexity.dataset import is_complexity, ORACLE as COMPLEX_ORACLE
    oracle = COMPLEX_ORACLE if is_complexity(case) else ORACLE
    return VerificationSpec(
        argv=(python, "-B", str(oracle.resolve()), str(project), case.case_id, profile),
        timeout_seconds=case.timeout_seconds,
        oracle_profile="benchmark-" + profile,
    )


def verify_candidate(case, project, python, profile, io, job_id, task_id):
    protected = protected_inputs(case)
    check_protected(protected)
    command = oracle_command(case, project, python, profile)
    spec = TaskSpec(
        job_id=job_id,
        task_id=task_id,
        kind="verify",
        objective="Independent benchmark checks",
        workspace=WorkspaceRef(
            workspace_id="candidate",
            root=str(project),
            source_manifest_sha256=workspace_revision(project),
        ),
        policy=ExecutionPolicy(allowed_tool_names=(), allow_verification=True),
        verification=command,
    )
    result = CoreCoderAdapter().execute(
        spec,
        AdapterContext(
            attempt_id="oracle-" + uuid.uuid4().hex,
            artifacts=io,
            permitted_verifications=(command,),
            verification_guard=lambda *args: check_protected(protected),
        ),
    )
    check_protected(protected)
    return result
