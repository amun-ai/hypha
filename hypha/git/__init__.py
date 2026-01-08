"""Git-compatible version control for Hypha artifacts.

This module provides Git repository functionality backed by S3 storage,
allowing standard Git clients to clone, push, and pull from Hypha artifacts.

Key components:
- S3GitObjectStore: Async S3-backed Git object store
- S3RefsContainer: Git references stored in S3
- GitHTTPHandler: FastAPI endpoints for Git Smart HTTP protocol
- GitLFSHandler: Git LFS (Large File Storage) support via S3
"""

from hypha.git.object_store import S3GitObjectStore
from hypha.git.refs import S3RefsContainer
from hypha.git.repo import S3GitRepo
from hypha.git.lfs import GitLFSHandler, create_lfs_router

__all__ = [
    "S3GitObjectStore",
    "S3RefsContainer",
    "S3GitRepo",
    "GitLFSHandler",
    "create_lfs_router",
]
