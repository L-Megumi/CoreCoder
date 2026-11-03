"""Versioned public investigation assignment; no hidden evaluator inputs."""
import hashlib
import json
from pathlib import PurePosixPath

VERSION = "focused-v1"
INSTRUCTIONS = (
    "Answer one investigation question: which cross-file behavior can explain the public failure, "
    "and which public component behaviors must the writer preserve? Start with the public task, "
    "then read the relevant entry point and actual implementations on its call path. "
    "Prefer those source reads over repeated broad searches. Check each suspected operation against "
    "its callers and configuration; filename hints alone are not observations. "
    "Use the existing strict diagnosis format: cite observed source for each conclusion; use "
    "recommended_actions for concrete checks or changes and limitations for unresolved gaps. "
    "An assignment is complete only to the extent supported by actual observations. "
    "Do not claim an uninspected implementation is faulty. Do not edit code."
)
PROMPT_SHA256 = hashlib.sha256(INSTRUCTIONS.encode()).hexdigest()


def build_assignment(objective, manifest):
    paths = [entry.path for entry in manifest.entries]
    def rank(path):
        name = PurePosixPath(path).stem.lower()
        priority = 0 if name in ("entry", "main", "train", "training", "trainer") else 1
        return priority, path.count("/"), path
    candidates = sorted((p for p in paths if p.endswith(".py")), key=rank)[:4]
    return dict(schema_version=1, version=VERSION, public_objective=objective,
        source_manifest_sha256=manifest.digest, candidate_entry_paths=candidates,
        scope="Filename-based entry hints only; follow actual imports/calls. No Oracle or private scores.",
        question="Explain the public failure along its execution path and identify public behaviors to preserve.",
        completion_criteria=["Cite actual implementation observations, distinguishing hypotheses.",
            "Return specific writer checks and explicitly retain uninspected gaps."],
        instructions_sha256=PROMPT_SHA256)


def assignment_text(value):
    return "\n[Controller focused investigation assignment: focused-v1]\n" + INSTRUCTIONS + "\n" + json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
