"""Explicit real-repository inputs for the existing ResearchCoder runtime."""

from .contracts import ProjectRunInput, RepositoryManifest, SourceFile
from .workspace import assert_allowed_changes, capture_manifest, materialize

__all__ = [
    "ProjectRunInput", "RepositoryManifest", "SourceFile",
    "assert_allowed_changes", "capture_manifest", "materialize",
]
