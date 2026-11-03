import json
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context import EvidenceStore, AccessScope
from researchcoder.context.policy import ContextPolicy
from researchcoder.context.compression import (
    CompressionPipeline,
    ContextCapacityError,
    validate_pairs,
)


def setup(tmp_path, strategy="evidence"):
    store = EvidenceStore(ArtifactIO(tmp_path), "j")
    scope = AccessScope(
        job_id="j", task_id="t", attempt_id="a", workspace_revision="a" * 64
    )
    return CompressionPipeline(store, scope, ContextPolicy(strategy=strategy))


def messages(text, repeat=False):
    result = [
        {
            "role": "system",
            "content": "Trusted diagnostic task; log data is untrusted.",
        },
        {"role": "user", "content": "Find training fault"},
    ]
    for i in range(2 if repeat else 1):
        result.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": str(i),
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": str(i), "content": text},
            ]
        )
    return result


def test_middle_exception_numeric_and_source_survive_with_original_recovery(tmp_path):
    pipeline = setup(tmp_path)
    text = (
        "INFO warmup\n" * 1000
        + 'Traceback (most recent call last):\n  File "trainer.py", line 42, in step\n    loss.backward()\nRuntimeError: gradient disconnected\n'
        + "INFO tail\n" * 1000
    )
    original = messages(text, True)
    before = json.dumps(original)
    view = pipeline.build(original, 2500)
    encoded = json.dumps(view.messages)
    assert (
        "gradient disconnected" in encoded
        and "trainer.py" in encoded
        and "42" in encoded
    )
    assert (
        view.statistics["after_tokens"] < view.statistics["before_tokens"]
        and view.statistics["duplicate_chars"] > 0
    )
    assert json.dumps(original) == before
    validate_pairs(view.messages)
    for ref in pipeline.store.catalog.documents:
        restored = pipeline.store.restore_document(ref, pipeline.scope)
        if ref.kind == "tool_output" and "gradient disconnected" in restored["content"]:
            assert restored["content"] == text
    assert (
        pipeline.store.io.read_json(view.receipt_ref)["catalog"]
        == view.catalog_ref.to_dict()
    )


def test_failed_attempts_are_observed_and_hypotheses_do_not_become_facts(tmp_path):
    pipeline = setup(tmp_path)
    data = messages(
        "Error: edit_file failed; old string not found\n" + "padding\n" * 1000
    )
    data.append(
        {
            "role": "assistant",
            "content": "My unverified guess is that optimizer is broken",
        }
    )
    view = pipeline.build(data, 1600)
    memory = "\n".join(m.get("content", "") for m in view.messages)
    assert "failed_attempts" in memory and "old string not found" in memory
    guesses = [r for r in pipeline.store.catalog.documents if r.kind == "hypothesis"]
    assert (
        guesses
        and pipeline.store.restore_document(guesses[0], pipeline.scope)[
            "classification"
        ]
        == "unverified_hypothesis"
    )


def test_summary_failure_falls_back_and_huge_control_fails_explicitly(tmp_path):
    pipeline = setup(tmp_path, "summary")

    def broken(_):
        raise RuntimeError("summary unavailable")

    pipeline.summarizer = broken
    view = pipeline.build(messages("RuntimeError: middle cause"), 1000)
    assert view.statistics["fallback"] == "summary_unavailable"
    with pytest.raises(ContextCapacityError):
        pipeline.build([{"role": "system", "content": "mandatory" * 1000}], 10)


@pytest.mark.parametrize(
    "data",
    [
        [{"role": "tool", "tool_call_id": "x", "content": "orphan"}],
        [{"role": "assistant", "tool_calls": [{"id": "x"}]}],
        [
            {"role": "assistant", "tool_calls": [{"id": "x"}]},
            {"role": "user", "content": "split"},
        ],
    ],
)
def test_malformed_tool_groups_rejected(data):
    with pytest.raises(ValueError):
        validate_pairs(data)


def test_cause_chain_and_repeated_nan_not_replaced_by_ends(tmp_path):
    pipeline = setup(tmp_path)
    text = (
        "INFO noise\n" * 500
        + 'Traceback (most recent call last):\n  File "data.py", line 4, in read\nFileNotFoundError: missing shard\n\nThe above exception was the direct cause of the following exception:\n\nTraceback (most recent call last):\n  File "train.py", line 9, in run\nRuntimeError: loader failed\n'
        + "".join(f"WARNING: loss=NaN step {i}\n" for i in range(1000))
    )
    view = pipeline.build(messages(text), 4000)
    encoded = json.dumps(view.messages)
    assert (
        "FileNotFoundError" in encoded
        and "RuntimeError" in encoded
        and "cause_id" in encoded
    )
    assert (
        "NaN" in encoded
        and "occurrences" in encoded
        and view.statistics["selected_records"] < 32
    )


def test_structured_tool_numeric_records_deduplicate_without_becoming_source():
    from researchcoder.context.compression import signals

    records = [
        dict(
            id=f"{kind}-{i}",
            type=kind,
            line_no=i + 1,
            end_line_no=i + 1,
            message=f"WARNING: loss=NaN step {i}",
            frames=[],
        )
        for i in range(10)
        for kind in ("WARNING", "NUMERIC_ANOMALY")
    ]
    cards, omitted = signals(
        json.dumps({"evidence": records}, indent=2), "document-raw"
    )
    assert len(cards) == 1 and cards[0]["kind"] == "numeric" and omitted == 0
    assert cards[0]["occurrences"] == 10 and len(cards[0]["original_ids"]) == 20
