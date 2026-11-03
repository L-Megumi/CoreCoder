"""Bounded source snapshots: copied bytes, verified manifests, no dirty-tree aliasing."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from corecoder.workflow import workspace_revision
from researchcoder.runtime.isolation import WorkspaceLease
from researchcoder.contracts import WorkspaceRef

EXCLUDED = {".git", ".tmp", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".txt", ".md"}


def source_bytes(root):
    root = Path(root).resolve()
    # Enforce the exact existing manifest limits before allocating/copying bytes.
    revision = workspace_revision(root)
    records = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED and not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            p = Path(directory) / name
            if not name.startswith(".") and p.suffix.lower() in SUFFIXES and not p.is_symlink():
                raw = p.read_bytes()
                if len(raw) > 1_000_000:
                    raise ValueError("Source changed during capture")
                records[p.relative_to(root).as_posix()] = raw
    manifest = {p: hashlib.sha256(b).hexdigest() for p, b in records.items()}
    actual = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != revision or workspace_revision(root) != revision:
        raise ValueError("Workspace changed while snapshotting")
    return revision, records, manifest


@dataclass(frozen=True)
class SourceSnapshot:
    workspace: WorkspaceRef
    manifest: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, source, target):
        source, target = Path(source).resolve(), Path(target).absolute()
        if target.is_relative_to(source) or target.exists():
            raise ValueError("Snapshot needs a new directory outside source")
        with WorkspaceLease(source):
            revision, records, manifest = source_bytes(source)
            target.mkdir(parents=True)
            for name, raw in records.items():
                p = target / name
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("xb") as stream:
                    stream.write(raw)
                p.chmod(0o444)
            if workspace_revision(target) != revision:
                raise ValueError("Snapshot copy mismatch")
            for p in sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                p.chmod(0o555)
            target.chmod(0o555)
        result = cls(
            WorkspaceRef(workspace_id="source-snapshot", root=str(target), source_manifest_sha256=revision),
            tuple(sorted(manifest.items())),
        )
        result.verify()
        return result

    def verify(self):
        revision, _, manifest = source_bytes(self.workspace.root)
        if revision != self.workspace.source_manifest_sha256 or tuple(sorted(manifest.items())) != self.manifest:
            raise ValueError("Immutable source snapshot changed")
        return self.workspace
