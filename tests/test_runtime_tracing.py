import json
import pytest
from researchcoder.runtime.tracing import TraceSink


def test_ordered_metadata_trace_and_parent_links(tmp_path):
    trace = TraceSink("j", "t", "a", path=tmp_path / "trace.jsonl")
    root = trace.emit("job.created")
    span = trace.start("model.request", parent_id=root, model="fake-model")
    trace.finish(span, "model.request", usage={"prompt_tokens": None, "completion_tokens": None, "unknown": True})
    events = trace.merged([])
    assert [e["sequence"] for e in events] == [1, 2, 3]
    assert events[1]["parent_id"] == events[0]["event_id"] and events[2]["parent_id"] == events[1]["event_id"]
    assert events[1]["end_timestamp"] is None and events[2]["end_timestamp"]
    assert all(e["job_id"] == "j" and e["task_id"] == "t" and e["attempt_id"] == "a" for e in events)
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 3


def test_sensitive_payload_fields_and_usage_backdoors_are_rejected():
    trace = TraceSink("j", "t", "a")
    with pytest.raises(ValueError):
        trace.emit("tool.call", arguments={"private": "data"})
    with pytest.raises(ValueError):
        trace.emit("model.request", env={"OPENAI_API_KEY": "secret"})
    with pytest.raises(ValueError):
        trace.emit("model.request", usage={"prompt_tokens": "secret"})
    trace.emit("model.request", model="sk-EXAMPLESECRET000000")
    assert "EXAMPLESECRET" not in json.dumps(trace.events)


def test_corrupt_or_cross_attempt_trace_is_rejected():
    trace = TraceSink("j", "t", "a")
    trace.emit("job.created")
    wrong = TraceSink("j", "t", "other")
    wrong.emit("worker.ready")
    with pytest.raises(ValueError):
        trace.merged(wrong.events)
    with pytest.raises(ValueError):
        trace.merged([{"arbitrary": "payload"}])
