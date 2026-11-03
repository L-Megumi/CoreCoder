"""Versioned public project inputs, without reference patches or private answers.

The all-files-v1 identity includes file content and size, including dotfiles and
files without extensions. It is separate from the legacy WorkspaceRef revision;
it does not claim to identify file permissions, directories or an OS sandbox.
"""

from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Literal

from researchcoder.contracts import Contract, digest, identifier, relative, text


MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 20_000_000
MAX_FILES = 4000
MAX_ENTRIES = 20000


def source_path(value):
    """Reject known private/configuration paths; this is not a secret scanner."""
    text(value, 4096)
    relative(value)
    parts = PurePosixPath(value).parts
    lowered = tuple(part.lower() for part in parts)
    forbidden_directories = {".git", ".hg", ".svn", ".ssh", ".aws", ".azure", ".gcloud", ".gnupg"}
    if any(part in forbidden_directories for part in lowered):
        raise ValueError("Private repository metadata or credential path refused")
    name = lowered[-1]
    if (name == ".env" or name.startswith(".env.")
            or name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "credentials.json", "secrets.json", "secrets.toml"}
            or PurePosixPath(name).suffix in {".key", ".pem", ".p12", ".pfx", ".jks"}):
        raise ValueError("Environment or credential file refused")


@dataclass(frozen=True, kw_only=True)
class SourceFile(Contract):
    path: str
    sha256: str
    size_bytes: int
    schema_version: int = 1

    def validate(self):
        source_path(self.path)
        digest(self.sha256)
        if not 0 <= self.size_bytes <= MAX_FILE_BYTES:
            raise ValueError("Source file size exceeds project bound")


@dataclass(frozen=True, kw_only=True)
class RepositoryManifest(Contract):
    entries: tuple[SourceFile, ...]
    algorithm: Literal["all-files-v1"] = "all-files-v1"
    schema_version: int = 1

    def validate(self):
        paths = tuple(item.path for item in self.entries)
        if not 1 <= len(paths) <= MAX_FILES or len(set(paths)) != len(paths):
            raise ValueError("Nonempty bounded unique source entries required")
        if paths != tuple(sorted(paths)):
            raise ValueError("Manifest entries must use canonical path order")
        if sum(item.size_bytes for item in self.entries) > MAX_TOTAL_BYTES:
            raise ValueError("Repository content exceeds project bound")
        # A file cannot simultaneously be the parent directory of another entry.
        names = set(paths)
        for path in paths:
            if any(str(parent) in names for parent in PurePosixPath(path).parents if str(parent) != "."):
                raise ValueError("Source file/directory path collision")

    @property
    def digest(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ProjectRunInput(Contract):
    project_id: str
    source_revision: str
    manifest: RepositoryManifest
    objective: str
    public_log: str
    writable_paths: tuple[str, ...]
    schema_version: int = 1

    def validate(self):
        identifier(self.project_id)
        identifier(self.source_revision)
        text(self.objective)
        text(self.public_log, 200_000)
        paths = {item.path for item in self.manifest.entries}
        if (not self.writable_paths or len(set(self.writable_paths)) != len(self.writable_paths)
                or not set(self.writable_paths) <= paths):
            raise ValueError("Explicit unique writes must refer to existing source files")
        for path in self.writable_paths:
            source_path(path)

    def public_input(self):
        """Explicit model-facing projection; execution/Oracle configuration is absent."""
        return {
            "project_id": self.project_id,
            "source_revision": self.source_revision,
            "manifest_sha256": self.manifest.digest,
            "objective": self.objective,
            "log": self.public_log,
            "source_files": [entry.path for entry in self.manifest.entries],
            "writable_paths": list(self.writable_paths),
        }
