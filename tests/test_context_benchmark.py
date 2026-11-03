import json
from pathlib import Path
from benchmarks.context.run_ablation import run
from benchmarks.context.probe import diagnose, score


def test_four_arm_ablation_has_independent_oracle_and_restorable_receipts(tmp_path):
    result = run(tmp_path)
    assert result["provider_requests"] == 0 and result["provider_usage"] is None
    assert len(result["rows"]) == 40 and len(result["summary"]) == 4
    assert all(r["after_tokens"] <= 4000 for r in result["rows"])
    assert {r["case_id"] for r in result["rows"]} == {
        c["id"]
        for c in json.loads(Path("benchmarks/context/cases.json").read_text())["cases"]
    }
    d = next(
        r for r in result["rows"] if r["case_id"] == "middle_source" and r["arm"] == "D"
    )
    b = next(
        r for r in result["rows"] if r["case_id"] == "middle_source" and r["arm"] == "B"
    )
    assert (
        d["evidence_retention"] > b["evidence_retention"] and d["answer_oracle_passed"]
    )
    assert all(r["recoverability"] == 1 for r in result["rows"] if r["arm"] == "D")
    assert all(r["receipts"] for r in result["rows"] if r["arm"] == "D")
    # Wrong or missing diagnostic text cannot make the independent oracle pass.
    expected = {
        "fault": "detached_features",
        "location": "train/joint.py",
        "expected_evidence": ["features.detach()"],
    }
    assert not score(diagnose([]), expected, "")["answer_oracle_passed"]
    assert (
        json.loads((tmp_path / "results.json").read_text())["manifest_sha256"]
        == result["manifest_sha256"]
    )
