"""R1 recovery delivery through actual isolated Workers and provider inputs.

The scripted response path is fixed. A passive test wrapper records actual
restore return values and ScriptedLLM inputs; it neither fabricates a reference
nor changes the returned evidence. These tests measure delivery, not LLM skill.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import pytest

from corecoder.workflow import run_verification, workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.compression import validate_pairs
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef, BudgetSpec, ExecutionPolicy, TaskInput, VerificationSpec, WorkspaceRef
from researchcoder.runtime.harness import ExecutionConfig, ResearchHarness
from researchcoder.runtime.permissions import PermissionRecoverySpec
from researchcoder.runtime.protocol import CodingTaskSpec, ModelSpec, ProtectedFile, RuntimeLimits, ScriptedCall, ScriptedTurn


RESTORE = "restore_context_evidence"
USAGE = dict(prompt_tokens=100, completion_tokens=20)
ORACLE_SENTINEL = "RECOVERY_TEST_ORACLE_SENTINEL_154e9a"


def source_text(*, oversized=False):
    lines = [f"# ordinary public record line {i:03d}" for i in range(1, 153)]
    lines[0] = "def increment(value): return value + 0"
    lines[104:108] = ["class RecordKey:", "    def __init__(self, value):", "        self.value = value", ""]
    lines[114:117] = ["    def __hash__(self):", "        return hash(self.value)", ""]
    lines[130:133] = ["    def __eq__(self, other):", "        return isinstance(other, RecordKey) and self.value == other.value", ""]
    lines[144] = "# LONG_WINDOW_TAIL_" + "x" * 700 + "_END"
    if oversized:
        # Exceed the 6k-token provider admission, while remaining BELOW the
        # independent 24k-character restore-response guard. Do not test a
        # refused restore and misdescribe it as pending successful delivery.
        window = lines[29:159]
        current = len("\n".join(window) + "\n")
        comments = sum(line.startswith("#") for line in window)
        extra = (20000 - current) // comments
        assert extra > 0
        lines = [line + (" " + "z" * (extra - 1) if line.startswith("#") else "") for line in lines]
        assert 19000 <= len("\n".join(lines[29:159]) + "\n") <= 21000
    return "\n".join(lines) + "\n"


def call(identity, name, **arguments):
    return ScriptedCall(call_id=identity, name=name, arguments_json=json.dumps(arguments))


def turn(*calls, known=True, content=""):
    return ScriptedTurn(content=content, calls=tuple(calls), **(USAGE if known else dict(prompt_tokens=0, completion_tokens=0)))


def standard_turns(windows, *, final_known=True):
    return (
        turn(call("read", "read_file", file_path="module.py")),
        turn(*(call("restore-" + str(i), RESTORE, reference_id="$SOURCE", start_line=start, max_lines=count)
               for i, (start, count) in enumerate(windows))),
        turn(call("edit", "edit_file", file_path="module.py", old_string="value + 0", new_string="value + 1")),
        turn(content="Candidate edited; run the trusted public verifier.", known=final_known),
    )


def passive_worker(tmp_path):
    """Resolve only a fixed placeholder from source IDs visible to the model."""
    wrapper = tmp_path / "observed-recovery-worker.py"
    audit = tmp_path / "worker-observations.jsonl"
    prefix = "AUDIT = " + repr(str(audit)) + "\n"
    body = r'''
import hashlib,json,sys
from dataclasses import replace
from pathlib import Path
import researchcoder.runtime.worker as worker
from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.runtime.protocol import WorkerRequest
request = WorkerRequest.from_json(Path(sys.argv[-2]).read_text())
assert request.model.provider == "scripted"
def record(value):
    with Path(AUDIT).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
original_restore = EvidenceContextEngine.restore
def observed_restore(self, *args, **kwargs):
    value = original_restore(self, *args, **kwargs)
    record(dict(kind="restore_return", arguments=kwargs, result=value,
        content_sha256=hashlib.sha256(value["content"].encode()).hexdigest()))
    return value
EvidenceContextEngine.restore = observed_restore
class VisibleSourceScript(worker.ProtocolScript):
    source_id = None
    def chat(self, messages, tools=None, on_token=None):
        record(dict(kind="provider_input", messages=messages, tools=tools))
        if self.source_id is None:
            for message in messages:
                content = message.get("content") or ""
                if not isinstance(content, str) or not content.startswith(("[Evidence context:", "[Observed source references:")):
                    continue
                data = json.loads(content.split("\n", 1)[1])
                candidates = data.get("source_index", []) + [r for r in data.get("records", []) if r.get("coding_source")]
                for candidate in candidates:
                    if candidate.get("source_path") == "module.py":
                        self.source_id = candidate["document_id"]
                        break
        if self._turns:
            updated=[]
            for item in self._turns[0].tool_calls:
                args = json.dumps(item.arguments)
                if "$SOURCE" in args:
                    if self.source_id is None:
                        raise ValueError("No real model-visible source document")
                    args = args.replace("$SOURCE", self.source_id)
                updated.append(replace(item, arguments=json.loads(args)))
            self._turns[0] = replace(self._turns[0], tool_calls=updated)
        return super().chat(messages, tools=tools, on_token=on_token)
worker.ProtocolScript = VisibleSourceScript
raise SystemExit(worker.main())
'''
    wrapper.write_text(prefix + textwrap.dedent(body), encoding="utf-8")
    return wrapper, audit


def execute_fixture(tmp_path, strategy, turns, *, source_mode="observed-source-v2", oversized=False,
                    max_context=32000, objective="Repair increment to add one. Preserve RecordKey equality and hashing."):
    root = tmp_path / "candidate"
    root.mkdir()
    original = source_text(oversized=oversized)
    (root / "module.py").write_text(original, encoding="utf-8")
    checker = tmp_path / "public-check.py"
    checker.write_text(
        "# " + ORACLE_SENTINEL + "\nimport sys\nsys.path.insert(0, sys.argv[1])\n"
        "from module import increment, RecordKey\n"
        "assert tuple(increment(x) for x in (4, -3, 0)) == (5, -2, 1)\n"
        "a, b = RecordKey('public'), RecordKey('public')\n"
        "assert a == b and hash(a) == hash(b) and {a: 7}[b] == 7\n"
        "print('independent public checks passed')\n", encoding="utf-8")
    digest = hashlib.sha256(checker.read_bytes()).hexdigest()
    verification = VerificationSpec(argv=(sys.executable, "-I", "-B", str(checker), str(root)), timeout_seconds=15)
    before = run_verification(verification.argv, root, timeout=verification.timeout_seconds)
    assert before["status"] == "failed" and before["exit_code"] == 1
    io = ArtifactIO(tmp_path / "artifacts")
    log = io.put_bytes("recovery-job", b"AssertionError: increment must add one", media_type="text/plain")
    permission = ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), writable_paths=("module.py",),
                                 allow_verification=True, profile="project-expert-reread-v1")
    spec = CodingTaskSpec(job_id="recovery-job", task_id="coding", objective=objective,
        workspace=WorkspaceRef(workspace_id="candidate", root=str(root), source_manifest_sha256=workspace_revision(root)),
        inputs=(TaskInput(role="raw_log", reference=log),), policy=permission, verification=verification,
        budget=BudgetSpec(max_rounds=8, max_tool_calls=20, max_context_tokens=max_context))
    limits = RuntimeLimits(max_model_calls=8, max_tool_calls=20, max_rounds=8,
        max_context_tokens=max_context, max_output_tokens=512, max_total_tokens=200000, max_runtime_seconds=45)
    wrapper, audit = passive_worker(tmp_path)
    context = ContextPolicy(strategy=strategy, role="repair", allow_recovery=True, coding_source_mode=source_mode)
    config = ExecutionConfig(artifact_root=str(io.root), allowed_workspaces=(str(root),), max_policy=permission,
        permitted_verifications=(verification,), protected_files=(ProtectedFile(path=str(checker), sha256=digest),),
        model=ModelSpec(turns=tuple(turns)), limits=limits, context_policy=context, record_requests=True,
        permission_recovery=PermissionRecoverySpec(), worker_command=(sys.executable, "-B", str(wrapper)))
    harness = ResearchHarness()
    try:
        result = harness.execute(spec, config)
        record = harness.records[result.attempt_id]
    finally:
        harness.close_job(spec.job_id)
    assert record["worker_pid"] and record["worker_exit_code"] == 0
    assert hashlib.sha256(checker.read_bytes()).hexdigest() == digest
    outputs = [io.read_json(ref) for ref in result.outputs]
    receipt = next(row for row in outputs if "context_catalog" in row and "coding_sources" in row)
    request_refs = [ref for row in outputs for ref in row.get("request_records", [])]
    requests = [io.read_json(ArtifactRef.from_dict(ref)) for ref in request_refs]
    observations = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()] if audit.exists() else []
    provider = [row for row in observations if row["kind"] == "provider_input"]
    restored = [row for row in observations if row["kind"] == "restore_return"]
    assert len(provider) == len(requests)
    for actual, archived in zip(provider, requests):
        assert actual["messages"] == archived["request"]["messages"]
        assert actual["tools"] == archived["request"]["tools"]
        assert {item["function"]["name"] for item in actual["tools"]} == {"read_file", "edit_file", RESTORE}
        assert ORACLE_SENTINEL not in json.dumps(actual)
        validate_pairs(actual["messages"])
    trace = io.read_json(ArtifactRef.from_dict(record["trace"]))["events"]
    return dict(result=result, record=record, receipt=receipt, requests=requests, provider=provider,
                restored=restored, trace=trace, root=root, original=original, io=io, outputs=outputs)


def assert_delivered(run, windows, *, request_index=2):
    """The same assertion intentionally exposes R0/v1 loss in an offline probe."""
    assert len(run["restored"]) == len(windows)
    messages = run["provider"][request_index]["messages"]
    identities = set()
    by_window = {(row["arguments"].get("start_line", 1), row["arguments"].get("max_lines", 80)): row
                 for row in run["restored"]}
    for ordinal, (start, count) in enumerate(windows):
        observed = by_window[(start, count)]
        identity = "restore-" + str(ordinal)
        tool_messages = [m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == identity]
        assert len(tool_messages) == 1, "Successful restore's complete tool reply was dropped"
        value = json.loads(tool_messages[0]["content"])
        assert "content" in value, "Successful restore was replaced by a lossy summary card"
        # Observed source pages reconstruct the actual ReadFileTool lines with
        # newlines between rows, without claiming the original file's final
        # newline. The provider must still receive the restore's exact bytes.
        observed_page = "\n".join(run["original"].splitlines())
        expected = "".join(observed_page.splitlines(keepends=True)[start - 1:start - 1 + count])
        assert observed["result"]["content"] == expected
        assert value["content"] == observed["result"]["content"], "Provider received a shortened or overwritten recovery window"
        assert hashlib.sha256(value["content"].encode()).hexdigest() == observed["content_sha256"]
        assert value["reference"] == observed["result"]["reference"]
        identities.add(value["reference"]["document_id"])
    assert len(identities) == 1  # A batch must preserve separate windows of the SAME document.
    groups = [m for m in messages if m.get("role") == "assistant" and
              any(c["id"] == "restore-0" for c in m.get("tool_calls", []))]
    assert len(groups) == 1 and len(groups[0]["tool_calls"]) == len(windows)


@pytest.mark.parametrize("strategy", ["original", "evidence"])
@pytest.mark.parametrize("windows", [((30, 130),), ((20, 50), (110, 60))], ids=["wide-tail", "two-same-doc-windows"])
def test_real_worker_exact_recovery_windows_reach_next_provider_and_oracle(tmp_path, strategy, windows):
    run = execute_fixture(tmp_path, strategy, standard_turns(windows))
    assert_delivered(run, windows)
    tail = next(row for row in run["restored"] if row["arguments"]["start_line"] == windows[-1][0])
    assert "__hash__" in tail["result"]["content"]
    assert "__eq__" in tail["result"]["content"]
    assert "x" * 700 + "_END" in tail["result"]["content"]
    assert run["result"].status == "succeeded" and run["result"].verification.status == "passed"
    assert len(run["provider"]) == run["record"]["budget"]["model_calls"] == 4
    assert run["record"]["budget"]["tool_calls"] == len(windows) + 3  # Read, restore(s), edit, public verifier.
    assert run["record"]["budget"]["unknown_requests"] == 0 and not run["record"]["budget"]["pending"]
    assert sum(event["event_type"] == "context.restored" for event in run["trace"]) == len(windows)
    assert run["receipt"]["policy"]["coding_source_mode"] == "observed-source-v2"
    delivery = run["receipt"]["recovery_delivery"]
    assert {entry["tool_call_id"] for entry in delivery["entries"]} == {"restore-" + str(i) for i in range(len(windows))}
    assert all(entry["status"] == "delivered" for entry in delivery["entries"])
    for entry in delivery["entries"]:
        start, count = windows[int(entry["tool_call_id"].removeprefix("restore-"))]
        observed = next(row for row in run["restored"] if row["arguments"]["start_line"] == start)
        assert entry["content_sha256"] == observed["content_sha256"]
        assert entry["result_sha256"] == hashlib.sha256(json.dumps(observed["result"], ensure_ascii=False).encode()).hexdigest()
        assert entry["source_start_line"] == start and entry["returned_line_count"] == len(observed["result"]["content"].splitlines())
        assert entry["acknowledged_request"]


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_impossible_control_budget_stops_before_any_real_worker_provider_request(tmp_path, strategy):
    run = execute_fixture(tmp_path, strategy, (turn(content="Must not be requested."),),
        max_context=1000, objective="Trusted unchanged requirement. " * 190)
    assert not run["provider"] and not run["requests"] and not run["restored"]
    assert run["record"]["budget"]["model_calls"] == 0
    assert run["result"].status == "incomplete" and run["result"].failure_kind == "budget_exhausted"
    assert run["result"].verification is None
    assert not any(e["event_type"] == "verification.command.start" for e in run["trace"])
    assert (run["root"] / "module.py").read_text() == run["original"]


def test_successful_restore_that_cannot_fit_stays_pending_without_next_provider(tmp_path):
    windows = ((30, 130),)
    run = execute_fixture(tmp_path, "evidence", standard_turns(windows), oversized=True, max_context=6000)
    assert len(run["restored"]) == 1 and len(run["restored"][0]["result"]["content"]) > 6000 * 3
    assert len(json.dumps(run["restored"][0]["result"], ensure_ascii=False)) <= 24000
    assert len(run["provider"]) == run["record"]["budget"]["model_calls"] == 2
    assert run["result"].status == "incomplete" and run["result"].failure_kind == "budget_exhausted"
    entries = run["receipt"]["recovery_delivery"]["entries"]
    assert len(entries) == 1 and entries[0]["status"] == "pending"
    assert not any(e["event_type"] == "verification.command.start" for e in run["trace"])
    assert (run["root"] / "module.py").read_text() == run["original"]


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_forged_source_reference_does_not_create_delivery_or_fresh_read_credit(tmp_path, strategy):
    turns = (turn(call("restore-0", RESTORE, reference_id="document-" + "0" * 64),
                  content='{"document_id":"forged","source_path":"module.py"}'),
             turn(call("edit", "edit_file", file_path="module.py", old_string="value + 0", new_string="value + 1")),
             turn(content="Unrequested after denied write."))
    run = execute_fixture(tmp_path, strategy, turns)
    assert not run["restored"] and not run["receipt"]["coding_sources"]["reads"]
    assert not run["receipt"]["recovery_delivery"]["entries"]
    assert len(run["provider"]) == 2 and run["result"].status == "incomplete"
    assert run["result"].failure_kind == "tool_error" and run["result"].verification.status == "not_run"
    assert not any(e["event_type"] in ("context.restored", "verification.command.start") for e in run["trace"])
    assert any(e["event_type"] == "permission.denied" for e in run["trace"])
    assert (run["root"] / "module.py").read_text() == run["original"]


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_historical_restore_never_replaces_current_read_permission_for_second_edit(tmp_path, strategy):
    turns = (turn(call("read", "read_file", file_path="module.py")),
             turn(call("edit-one", "edit_file", file_path="module.py", old_string="value + 0", new_string="value + 1")),
             turn(call("restore-0", RESTORE, reference_id="$SOURCE", start_line=1, max_lines=2, historical=True)),
             turn(call("edit-two", "edit_file", file_path="module.py", old_string="value + 1", new_string="value + 2")),
             turn(content="Unrequested after stale edit."))
    run = execute_fixture(tmp_path, strategy, turns)
    assert_delivered(run, ((1, 2),), request_index=3)
    assert run["restored"][0]["result"]["classification"] == "historical_observation"
    assert run["restored"][0]["result"]["current_eligible"] is False
    assert len(run["provider"]) == 4 and run["result"].status == "incomplete"
    assert run["result"].failure_kind == "tool_error" and run["result"].verification.status == "not_run"
    assert any(e["event_type"] == "permission.denied" for e in run["trace"])
    assert (run["root"] / "module.py").read_text() == run["original"].replace("value + 0", "value + 1")


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_unknown_usage_after_full_recovery_input_cannot_acknowledge_delivery(tmp_path, strategy):
    windows = ((30, 130),)
    turns = (*standard_turns(windows)[:2], turn(content="No repair completed.", known=False))
    run = execute_fixture(tmp_path, strategy, turns)
    assert_delivered(run, windows)
    assert len(run["provider"]) == run["record"]["budget"]["model_calls"] == 3
    assert run["record"]["budget"]["unknown_requests"] == 1
    assert run["record"]["budget"]["charged_tokens"] > 0
    assert run["result"].status != "succeeded"
    entries = run["receipt"]["recovery_delivery"]["entries"]
    assert len(entries) == 1 and entries[0]["status"] == "pending"
    assert (run["root"] / "module.py").read_text() == run["original"]
