"""Bounded exact-content copies for real repository candidates.

No source metadata is silently excluded. A private path, symlink or unsupported
file type causes rejection. Failed materialization leaves its new partial target
for inspection; it never deletes or overwrites a caller's directory.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import stat
from researchcoder.contracts import WorkspaceRef

from .contracts import (
    MAX_ENTRIES, MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES,
    RepositoryManifest, SourceFile, source_path,
)


def _directory(root):
    root = Path(root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Symlink directory refused")
    if not root.is_dir():
        raise ValueError("Repository directory must exist")
    return root.resolve(strict=True)


def _file_bytes(path, root):
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Source path escaped repository")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError("Unsupported or oversized source file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("Source changed while opening")
        raw = stream.read(MAX_FILE_BYTES + 1)
        closed = os.fstat(stream.fileno())
    if (len(raw) > MAX_FILE_BYTES or opened.st_size != len(raw)
            or (opened.st_size, opened.st_mtime_ns) != (closed.st_size, closed.st_mtime_ns)
            or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root)):
        raise ValueError("Source changed while reading")
    return raw


def capture_manifest(root):
    """Capture every regular file, bounded and including hidden/configuration files."""
    root = _directory(root)
    entries = []
    total = seen = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(root):
            raise ValueError("Source directory escaped repository")
        with os.scandir(directory) as iterator:
            # Bound enumeration rather than allocating an unbounded directory list.
            children = []
            for child in iterator:
                seen += 1
                if seen > MAX_ENTRIES:
                    raise ValueError("Repository entry limit exceeded")
                children.append(child)
        for child in sorted(children, key=lambda value: value.name):
            path = Path(child.path)
            logical = path.relative_to(root).as_posix()
            source_path(logical)
            if child.is_symlink():
                raise ValueError("Source symlink refused")
            if child.is_dir(follow_symlinks=False):
                stack.append(path)
                continue
            if not child.is_file(follow_symlinks=False):
                raise ValueError("Unsupported source file type")
            if len(entries) >= MAX_FILES:
                raise ValueError("Repository file limit exceeded")
            raw = _file_bytes(path, root)
            total += len(raw)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Repository byte limit exceeded")
            entries.append(SourceFile(path=logical, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
    return RepositoryManifest(entries=tuple(sorted(entries, key=lambda item: item.path)))


def materialize(source, target, manifest):
    """Copy a validated complete source into a fresh independent candidate."""
    if type(manifest) is not RepositoryManifest:
        raise TypeError("Typed repository manifest required")
    source = _directory(source)
    target = Path(target).absolute()
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Symlink destination refused")
    resolved = target.resolve()
    if target.exists() or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("Candidate must be a fresh directory outside the source")
    if capture_manifest(source) != manifest:
        raise ValueError("Source differs from frozen manifest")
    target.mkdir(parents=True, exist_ok=False)
    for entry in manifest.entries:
        raw = _file_bytes(source / entry.path, source)
        if len(raw) != entry.size_bytes or hashlib.sha256(raw).hexdigest() != entry.sha256:
            raise ValueError("Source changed during materialization")
        destination = target / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (destination, *destination.parents)) or not destination.resolve().is_relative_to(resolved):
            raise ValueError("Candidate path escaped target")
        with destination.open("xb") as stream:
            stream.write(raw)
    if capture_manifest(source) != manifest or capture_manifest(target) != manifest:
        raise ValueError("Source or candidate changed during materialization")
    return target


def assert_allowed_changes(before, after, writable_paths):
    """Allow content changes only to explicitly named existing files."""
    if type(before) is not RepositoryManifest or type(after) is not RepositoryManifest:
        raise TypeError("Typed repository manifests required")
    if not isinstance(writable_paths, (tuple, list)) or any(type(p) is not str for p in writable_paths):
        raise TypeError("Explicit writable path sequence required")
    for path in writable_paths:
        source_path(path)
    old = {entry.path: entry for entry in before.entries}
    new = {entry.path: entry for entry in after.entries}
    allowed = set(writable_paths)
    if len(allowed) != len(writable_paths) or not allowed <= old.keys():
        raise ValueError("Invalid or duplicate writable paths")
    if old.keys() != new.keys():
        raise ValueError("Candidate added or deleted source files")
    changed = tuple(sorted(path for path in old if old[path] != new[path]))
    if not set(changed) <= allowed:
        raise ValueError("Candidate changed protected files")
    return changed


@dataclass(frozen=True)
class ProjectSourceSnapshot:
    """Complete project copy implementing the existing Scheduler snapshot interface.

    The legacy revision remains available to the Adapter and EvidenceExchange.
    The independent all-files manifest prevents omitted suffixes or dotfiles from
    silently changing under that revision. Read-only modes are an extra guard,
    not an OS sandbox or a new source version algorithm for existing contracts.
    """

    workspace: WorkspaceRef
    manifest: RepositoryManifest

    @classmethod
    def create(cls, source, target, manifest):
        from corecoder.workflow import workspace_revision
        from researchcoder.contracts import WorkspaceRef
        from researchcoder.runtime.isolation import WorkspaceLease

        if type(manifest) is not RepositoryManifest:
            raise TypeError("Typed complete source manifest required")
        source = _directory(source)
        with WorkspaceLease(source):
            revision = workspace_revision(source)
            target = materialize(source, target, manifest)
            if workspace_revision(target) != revision or workspace_revision(source) != revision:
                raise ValueError("Legacy revision changed while snapshotting")
            for entry in manifest.entries:
                (target / entry.path).chmod(0o444)
            for directory in sorted((p for p in target.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                directory.chmod(0o555)
            target.chmod(0o555)
        result = cls(WorkspaceRef(workspace_id="project-source-snapshot", root=str(target),
                                 source_manifest_sha256=revision), manifest)
        result.verify()
        return result

    def verify(self):
        from corecoder.workflow import workspace_revision

        if capture_manifest(self.workspace.root) != self.manifest:
            raise ValueError("Complete source snapshot changed")
        if workspace_revision(self.workspace.root) != self.workspace.source_manifest_sha256:
            raise ValueError("Legacy source snapshot revision changed")
        return self.workspace

    def to_dict(self):
        return dict(schema_version=1, workspace=self.workspace.to_dict(),
                    manifest=self.manifest.to_dict(), manifest_sha256=self.manifest.digest)
