"""Fixed diagnostic actions. No access to the expected-answer manifest or originals."""


def diagnose(messages):
    text = "\n".join(
        str(m.get("content", "")) for m in messages if m.get("role") != "system"
    )
    rules = (
        (
            "mat1 and mat2 shapes cannot be multiplied",
            "shape_mismatch",
            "train/head.py",
        ),
        ("missing validation images", "dataset_path", "data/loader.py"),
        ("features.detach()", "detached_features", "models/backbone.py"),
        (
            "features = model.backbone(x).detach()",
            "detached_features",
            "train/joint.py",
        ),
        ("torch.log(prob)", "unstable_log", "train/loss.py"),
        ("features / std", "division_zero", "train/normalize.py"),
    )
    for clue, fault, location in rules:
        if clue in text and location in text:
            return dict(fault=fault, location=location, decision="fixed_rule_probe")
    return dict(fault=None, location=None, decision="insufficient_visible_evidence")


def score(result, expected, visible_text):
    """Independent labels; no model self-assessment."""
    found = [
        clue
        for clue in expected["expected_evidence"]
        if any(c in visible_text for c in (clue if isinstance(clue, list) else [clue]))
    ]
    diagnosis = result["fault"] == expected["fault"]
    location = result["location"] == expected["location"]
    return dict(
        expected_count=len(expected["expected_evidence"]),
        retained_count=len(found),
        retained=found,
        evidence_retention=len(found) / len(expected["expected_evidence"]),
        diagnosis_match=diagnosis,
        location_match=location,
        answer_oracle_passed=diagnosis and location,
    )
