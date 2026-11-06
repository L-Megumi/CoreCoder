from dataclasses import replace
import json

import pytest

from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.handoff_codec import INDEXED_VIEW, index_references
from tests.test_handoff_projection_contract import projected


def indexed_fixture(tmp_path):
    request, io, runtime, envelope, context = projected(tmp_path)
    request = replace(request, context_policy=replace(request.context_policy, handoff_view_version=INDEXED_VIEW))
    context["producer_table"] = index_references(context["observations"])
    context["excerpt_policy"]["version"] = INDEXED_VIEW
    return request, io, runtime, envelope, context


def receive(request, io, runtime, envelope, context):
    envelope["consultations"][0]["context_ref"] = io.put_json("job", context).to_dict()
    request = replace(request, evidence_context=json.dumps(envelope))
    return EvidenceContextEngine(request, io, runtime.ledger, runtime.trace)


def test_full_refs_reconstructed_even_with_colliding_origin_ids(tmp_path):
    request, io, runtime, envelope, context = indexed_fixture(tmp_path)
    engine = receive(request, io, runtime, envelope, context)
    handoff = engine.coding_handoff
    assert len(handoff.observations) == 4 and len(handoff.selected) == 2
    refs = [item["binding"]["provenance"]["reference"] for item in handoff.observations]
    assert {(r["origin_id"], r["producer_task_id"]) for r in refs} == {
        (origin, producer) for origin in ("E0", "O0") for producer in ("code-expert", "log-expert")}
    assert len({item["binding"]["document_id"] for item in handoff.observations}) == 4
    assert len(handoff.message()["content"]) <= 32000


@pytest.mark.parametrize("change", ("scope", "foreign_job", "producer", "attempt", "artifact", "stale",
                                    "source_sha", "bad_index", "negative_index", "boolean_index",
                                    "inject_field", "swap_index", "extra_receipt", "excerpt"))
def test_indexed_handoff_rejects_tampering(tmp_path, change):
    request, io, runtime, envelope, context = indexed_fixture(tmp_path)
    row = context["observations"][0]
    group = context["producer_table"][0]
    if change == "scope":
        request = replace(request, context_policy=replace(request.context_policy, handoff_view_version="bounded-excerpts-v1"))
    elif change == "foreign_job":
        group["job_id"] = "other"
    elif change == "producer":
        group["producer_task_id"] = "outside-selected"
    elif change == "attempt":
        group["producer_attempt_id"] = "forged-attempt"
    elif change == "artifact":
        group["artifact"]["sha256"] = "e" * 64
    elif change == "stale":
        group["workspace_revision"] = "a" * 64
    elif change == "source_sha":
        context["candidate_source_paths"][0]["sha256"] = "c" * 64
    elif change == "bad_index":
        row["producer_index"] = 222
    elif change == "negative_index":
        row["producer_index"] = -1
    elif change == "boolean_index":
        row["producer_index"] = True
    elif change == "inject_field":
        row["reference"]["job_id"] = "forged"
    elif change == "swap_index":
        row["producer_index"] = 1
    elif change == "extra_receipt":
        context["producer_table"].append(group.copy())
    else:
        row["excerpt"] = "forged"
    with pytest.raises((ValueError, TypeError)):
        receive(request, io, runtime, envelope, context)
