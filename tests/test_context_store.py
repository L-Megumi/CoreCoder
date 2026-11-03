from dataclasses import replace
import hashlib
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context import EvidenceStore, AccessScope, EvidenceCatalog
from tests.test_orchestration_evidence import sample


def setup(tmp_path):
    io = ArtifactIO(tmp_path)
    store = EvidenceStore(io, "job")
    result = sample(io, "producer")
    for ref in result.evidence_refs:
        store.add_record(ref)
    scope = AccessScope(
        job_id="job",
        task_id="consumer",
        attempt_id="c1",
        workspace_revision="a" * 64,
        producers=(("producer", "attempt-producer"),),
    )
    return store, scope, result


def test_catalog_roundtrip_queries_and_immutable_originals(tmp_path):
    store, scope, result = setup(tmp_path)
    ref = store.query(
        scope, task_id="producer", attempt_id="attempt-producer", evidence_id="same"
    )[0]
    assert store.restore_record(ref, scope)["record"]["id"] == "same"
    catalog = store.save()
    loaded = EvidenceStore.load(store.io, catalog, "job")
    assert loaded.catalog == EvidenceCatalog.from_json(store.catalog.to_json())
    assert loaded.restore_record(ref, scope) == store.restore_record(ref, scope)
    with pytest.raises(ValueError):
        loaded.restore_record(replace(ref, source_path="/forged"), scope)
    with pytest.raises(ValueError):
        EvidenceStore.load(store.io, catalog, "other")
    with pytest.raises(ValueError):
        store.add_record(replace(ref, source_path="/forged"))


@pytest.mark.parametrize(
    "change",
    [{"job_id": "foreign"}, {"producers": ()}, {"workspace_revision": "c" * 64}],
)
def test_authority_and_source_versions_enforced(tmp_path, change):
    store, scope, result = setup(tmp_path)
    with pytest.raises(ValueError):
        store.restore_record(result.evidence_refs[0], replace(scope, **change))


def test_full_original_documents_and_explicit_historical_recovery(tmp_path):
    store, scope, _ = setup(tmp_path)
    text = (
        "INFO padding\n" * 2000 + "RuntimeError: middle cause\n" + "INFO tail\n" * 2000
    ).encode()
    refs = [
        store.add_document(
            text,
            task_id="producer",
            attempt_id="attempt-producer",
            workspace_revision="a" * 64,
            kind=k,
        )
        for k in ("log", "verification", "source")
    ]
    assert len({r.artifact for r in refs}) == 1
    for ref in refs:
        restored = store.restore_document(ref, scope)
        assert (
            restored["content"].encode() == text
            and restored["original_sha256"] == hashlib.sha256(text).hexdigest()
        )
        assert (
            store.restore_document(ref, scope, start_line=2001, max_lines=1)["content"]
            == "RuntimeError: middle cause\n"
        )
        with pytest.raises(ValueError):
            store.restore_document(ref, replace(scope, workspace_revision="b" * 64))
        old = store.restore_document(
            ref, replace(scope, workspace_revision="b" * 64, allow_historical=True)
        )
        assert (
            not old["current_eligible"]
            and old["classification"] == "historical_observation"
        )
    with pytest.raises(ValueError):
        store.restore_document(replace(ref, document_id="fake"), scope)
    hypothesis = store.add_document(
        b"Guess only",
        task_id="producer",
        attempt_id="attempt-producer",
        workspace_revision="a" * 64,
        kind="hypothesis",
    )
    assert (
        store.restore_document(hypothesis, scope)["classification"]
        == "unverified_hypothesis"
    )


def test_causal_and_source_relations_remain_producer_scoped(tmp_path):
    io = ArtifactIO(tmp_path)
    store = EvidenceStore(io, "j")
    raw = {
        "evidence": [
            {"id": "root", "source": "x.log", "line_no": 1},
            {"id": "wrap", "source": "x.log", "line_no": 3, "cause_id": "root"},
        ],
        "observations": [],
    }
    for task in ("one", "two"):
        artifact = io.put_json("j", raw)
        for ref in io.index_evidence(
            artifact, task_id=task, attempt_id="a", workspace_revision="a" * 64
        ):
            store.add_record(ref)
    scope = AccessScope(
        job_id="j",
        task_id="consumer",
        attempt_id="c",
        workspace_revision="a" * 64,
        producers=(("one", "a"), ("two", "a")),
    )
    ref = store.query(scope, task_id="one", attempt_id="a", evidence_id="wrap")[0]
    chain = store.chain(ref, scope)
    assert {r["reference"]["origin_id"] for r in chain} == {"wrap", "root"}
    assert {r["reference"]["producer_task_id"] for r in chain} == {"one"}
    missing = EvidenceStore(io, "j", replace(store.catalog, records=(ref,)))
    with pytest.raises(ValueError):
        missing.chain(ref, scope)


def test_tampered_original_and_path_escape_fail(tmp_path):
    store, scope, _ = setup(tmp_path)
    ref = store.add_document(
        b"original",
        task_id="producer",
        attempt_id="attempt-producer",
        workspace_revision="a" * 64,
        kind="log",
    )
    (store.io.root / ref.artifact.storage_key).write_bytes(b"changed")
    with pytest.raises(ValueError):
        store.restore_document(ref, scope)
    with pytest.raises(ValueError):
        replace(ref.artifact, storage_key="job/../escape")
