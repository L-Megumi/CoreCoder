import json
from types import SimpleNamespace
import pytest
from researchcoder.evaluation.observability import (
    projection,
    send_payload,
    LangfuseExporter,
)
from researchcoder.evaluation.contracts import EvaluationMetrics


def sample():
    return SimpleNamespace(
        run_id="run",
        job_id="job",
        trial=SimpleNamespace(
            trial_id="trial-0000", case_id="case", mode="single", repeat=0
        ),
        status="failed",
        metrics=EvaluationMetrics(wall_seconds=1.25),
    )


def test_allowlist_pseudonyms_parent_pairing_and_no_sensitive_content():
    events = [
        dict(
            event_id="a",
            parent_id=None,
            event_type="model.request.start",
            job_id="job",
            task_id="task",
            attempt_id="attempt",
            status="running",
            model="sk-secret-model-private",
            role="repair",
            revision_id="revision",
            start_timestamp="2026-10-25T00:00:00+00:00",
            input="source body",
            environment={"password": "secret"},
        ),
        dict(
            event_id="b",
            parent_id="a",
            event_type="tool.start",
            status="running",
            tool="private log",
            usage=None,
        ),
        dict(
            event_id="c",
            parent_id="b",
            event_type="tool.end",
            status="ok",
            elapsed_seconds=0.25,
        ),
        dict(
            event_id="d",
            parent_id="a",
            event_type="model.request.end",
            status="ok",
            elapsed_seconds=1.0,
            usage=dict(prompt_tokens=12, completion_tokens=3, unknown=False),
            end_timestamp="2026-10-25T00:00:01+00:00",
        ),
    ]
    payload = projection(sample(), dict(source="secret source"), events)
    raw = json.dumps(payload)
    assert all(
        value not in raw
        for value in (
            "sk-secret-model-private",
            "source body",
            "password",
            "private log",
            "secret source",
        )
    )
    assert len(payload["observations"]) == 3
    generation = next(o for o in payload["observations"] if o["type"] == "generation")
    tool = next(o for o in payload["observations"] if o["type"] == "tool")
    assert (
        generation["usage_details"] == dict(input=12, output=3)
        and tool["parent_id"] == generation["id"]
    )
    assert generation["metadata"]["elapsed_seconds"] == 1.0


def test_current_sdk_method_surface_via_mock_and_network_failure_is_separate():
    class Client:
        def __init__(self):
            self.calls = []
            self.scores = []
            self.flushed = False

        def start_observation(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(id=str(len(self.calls)).zfill(16), end=lambda: None)

        def create_score(self, **kwargs):
            self.scores.append(kwargs)

        def flush(self):
            self.flushed = True

    payload = projection(
        sample(),
        {},
        [dict(event_id="a", parent_id=None, event_type="worker.ready", status="ok")],
    )
    client = Client()
    send_payload(payload, client)
    assert (
        client.flushed
        and len(client.calls) == 2
        and client.calls[1]["trace_context"]["parent_span_id"] == "0000000000000001"
    )
    assert all("input" not in c and "output" not in c for c in client.calls)
    assert any(c["name"] == "end_to_end_success" for c in client.scores)


def test_disabled_offline_missing_credentials_and_network_authorization(monkeypatch):
    assert LangfuseExporter().export_result(None, None) == dict(
        status="disabled", sent=False
    )
    monkeypatch.setattr(
        "researchcoder.evaluation.observability.payload_for_result",
        lambda *a: projection(sample(), {}, []),
    )
    assert (
        LangfuseExporter(mode="offline").export_result(None, None)["status"]
        == "offline"
    )
    assert (
        LangfuseExporter(mode="langfuse").export_result(None, None)["status"]
        == "network_not_authorized"
    )
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert (
        LangfuseExporter(mode="langfuse", allow_network=True).export_result(None, None)[
            "status"
        ]
        == "not_configured"
    )


def test_export_error_does_not_expose_exception_body(monkeypatch):
    monkeypatch.setattr(
        "researchcoder.evaluation.observability.payload_for_result",
        lambda *a: (_ for _ in ()).throw(
            ValueError("sk-supersecret /private/source.py")
        ),
    )
    result = LangfuseExporter(mode="offline").export_result(None, None)
    assert result == dict(status="export_error", sent=False, error_type="ValueError")


def test_unknown_usage_not_exported_as_actual_zero():
    payload = projection(
        sample(),
        {},
        [
            dict(
                event_id="a",
                parent_id=None,
                event_type="model.request.start",
                status="running",
            ),
            dict(
                event_id="b",
                parent_id="a",
                event_type="model.request.end",
                status="error",
                usage=dict(
                    prompt_tokens=None,
                    completion_tokens=None,
                    unknown=True,
                    estimated_tokens=45,
                ),
            ),
        ],
    )
    generation = next(o for o in payload["observations"] if o["type"] == "generation")
    assert "usage_details" not in generation and generation["metadata"]["usage_unknown"]


def test_invalid_parent_cycles_rejected():
    with pytest.raises(ValueError):
        projection(
            sample(),
            {},
            [
                dict(event_id="a", parent_id="b", event_type="worker.ready"),
                dict(event_id="b", parent_id="a", event_type="worker.ready"),
            ],
        )


def test_mock_current_sdk_connection_failure_is_nonfatal(monkeypatch):
    import sys

    class FakeSDK:
        def __init__(self, **kwargs):
            pass

        def start_observation(self, **kwargs):
            raise ConnectionError("private network exception body")

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-test")
    monkeypatch.setattr(
        "researchcoder.evaluation.observability.payload_for_result",
        lambda *a: projection(sample(), {}, []),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "5.0.0")
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(Langfuse=FakeSDK))
    assert LangfuseExporter(mode="langfuse", allow_network=True).export_result(
        None, None
    ) == dict(status="export_error", sent=False, error_type="ConnectionError")
