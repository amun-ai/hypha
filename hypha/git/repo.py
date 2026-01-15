"""S3-backed Git repository.

This module provides a Git repository implementation backed by S3,
combining the S3GitObjectStore and S3RefsContainer.

Uses fully async S3 operations:
- Read operations: Use async aiobotocore client factory
- Write operations: Use async aiohttp-based S3 client with AWS Signature V4
"""

import logging
from typing import Callable, Optional

from dulwich.objects import ObjectID
from dulwich.repo import BaseRepo

from hypha.git.object_store import S3GitObjectStore
from hypha.git.refs import S3InfoRefsContainer
from hypha.git.s3_async import create_async_s3_client

logger = logging.getLogger(__name__)


class S3GitRepo(BaseRepo):
    """A Git repository backed by S3 storage.

    This repository stores all objects as pack files in S3 and
    references as individual S3 objects. It supports the full Git
    protocol via HTTP Smart transport.

    Uses fully async S3 operations:
    - Read operations: Use async aiobotocore client factory
    - Write operations: Use async aiohttp-based S3 client

    Usage:
        def s3_client_factory():
            return session.create_client("s3", ...)

        repo = S3GitRepo(s3_client_factory, "bucket", "prefix")
        await repo.initialize()

        # Use as a standard Git repo
        head = repo.head()
    """

    def __init__(
        self,
        s3_client_factory: Callable,
        bucket: str,
        prefix: str,
        object_format=None,
        s3_config: Optional[dict] = None,
    ):
        """Initialize S3 Git repository.

        Args:
            s3_client_factory: Factory function that returns an async context manager
                               for S3 client (e.g., lambda: session.create_client("s3", ...))
            bucket: S3 bucket name
            prefix: Prefix path in bucket (e.g., "workspace/artifacts/123/.git")
            object_format: Git object format (defaults to SHA1)
            s3_config: S3 config dict with endpoint_url, access_key_id,
                       secret_access_key, region_name for async write operations
        """
        self._s3_client_factory = s3_client_factory
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._s3_config = s3_config

        # Create object store and refs container
        self._object_store = S3GitObjectStore(
            s3_client_factory, bucket, prefix, object_format=object_format, s3_config=s3_config
        )
        self._refs = S3InfoRefsContainer(
            s3_client_factory, bucket, prefix, object_store=self._object_store, s3_config=s3_config
        )

        # Initialize base repo
        super().__init__(self._object_store, self._refs)

        self._initialized = False
        self.bare = True  # S3 repos are always bare

    @property
    def s3_client_factory(self):
        return self._s3_client_factory

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    async def initialize(self):
        """Initialize the repository by loading refs and pack cache."""
        if self._initialized:
            return

        await self._object_store.initialize()
        await self._refs.initialize()
        self._initialized = True

    @classmethod
    async def init_bare(
        cls,
        s3_client_factory: Callable,
        bucket: str,
        prefix: str,
        default_branch: bytes = b"main",
        s3_config: Optional[dict] = None,
    ) -> "S3GitRepo":
        """Initialize a new bare repository in S3.

        Args:
            s3_client_factory: Factory function that returns an async context manager
                               for S3 client
            bucket: S3 bucket name
            prefix: Prefix path in bucket
            default_branch: Name of the default branch
            s3_config: S3 config dict with endpoint_url, access_key_id,
                       secret_access_key, region_name for async write operations

        Returns:
            Initialized S3GitRepo instance
        """
        if not s3_config:
            raise RuntimeError(
                "s3_config is required for init_bare. "
                "Pass s3_config dict with endpoint_url, access_key_id, "
                "secret_access_key, and region_name."
            )

        repo = cls(s3_client_factory, bucket, prefix, s3_config=s3_config)

        # Create initial HEAD pointing to default branch
        await repo._refs.set_symbolic_ref_async(b"HEAD", b"refs/heads/" + default_branch)

        # Create the pack directory marker using async S3 client
        pack_dir_key = f"{prefix}/objects/pack/.gitkeep"
        async_client = create_async_s3_client(s3_config)
        try:
            await async_client.put_object(bucket, pack_dir_key, b"")
        except Exception:
            raise
        finally:
            await async_client.close()

        repo._initialized = True
        return repo

    async def is_empty(self) -> bool:
        """Check if the repository has any commits."""
        if not self._initialized:
            await self.initialize()

        refs = await self._refs.get_all_refs_async()
        # Filter out symbolic refs
        for name, value in refs.items():
            if not value.startswith(b"ref: "):
                return False
        return True

    def head(self) -> ObjectID:
        """Get the SHA1 of the HEAD commit."""
        return self._refs[b"HEAD"]

    async def head_async(self) -> Optional[ObjectID]:
        """Get the SHA1 of HEAD asynchronously."""
        if not self._initialized:
            await self.initialize()

        try:
            return self._refs[b"HEAD"]
        except KeyError:
            return None

    async def get_refs_async(self) -> dict[bytes, ObjectID]:
        """Get all references."""
        if not self._initialized:
            await self.initialize()
        return await self._refs.get_all_refs_async()

    async def set_ref_async(
        self,
        name: bytes,
        value: ObjectID,
        old_value: Optional[ObjectID] = None,
    ) -> bool:
        """Set a reference."""
        if not self._initialized:
            await self.initialize()
        return await self._refs.set_ref_async(name, value, old_value)

    async def generate_info_refs_async(self) -> bytes:
        """Generate info/refs content for HTTP protocol."""
        if not self._initialized:
            await self.initialize()
        return await self._refs.generate_info_refs()

    def _determine_file_mode(self) -> bool:
        """Determine if file modes should be trusted."""
        return False  # S3 doesn't have file modes

    def _put_named_file(self, path: str, contents: bytes):
        """Store a named file. Not typically used for S3 repos."""
        raise NotImplementedError("Use async methods for S3 operations")

    def get_named_file(self, path: str):
        """Get a named file. Not typically used for S3 repos."""
        raise NotImplementedError("Use async methods for S3 operations")

    def close(self):
        """Close the repository."""
        self._object_store.close()

    async def receive_pack_async(
        self,
        pack_data: bytes,
        ref_updates: dict[bytes, tuple[bytes, bytes]],
    ) -> dict[bytes, Optional[str]]:
        """Process an incoming pack from a push operation.

        Args:
            pack_data: Raw pack file data
            ref_updates: Dict mapping ref names to (old_sha, new_sha) tuples

        Returns:
            Dict mapping ref names to error messages (None if successful)
        """
        if not self._initialized:
            await self.initialize()

        results = {}

        # First, add the pack to the object store
        if pack_data:
            try:
                basename, checksum = await self._object_store.add_pack_async(pack_data)
                logger.info(f"Added pack {basename}")
            except Exception as e:
                logger.error(f"Failed to add pack: {e}")
                # Return error for all refs
                for ref_name in ref_updates:
                    results[ref_name] = f"failed to process pack: {e}"
                return results

        # Process ref updates
        for ref_name, (old_sha, new_sha) in ref_updates.items():
            try:
                # Handle delete (new_sha is all zeros)
                if new_sha == b"0" * 40:
                    success = await self._refs.delete_ref_async(
                        ref_name,
                        old_sha if old_sha != b"0" * 40 else None,
                    )
                    if not success:
                        results[ref_name] = "reference update rejected"
                    else:
                        results[ref_name] = None
                else:
                    # Create or update ref
                    success = await self._refs.set_ref_async(
                        ref_name,
                        new_sha,
                        old_sha if old_sha != b"0" * 40 else None,
                    )
                    if not success:
                        results[ref_name] = "reference update rejected"
                    else:
                        results[ref_name] = None

            except Exception as e:
                logger.error(f"Failed to update ref {ref_name}: {e}")
                results[ref_name] = str(e)

        return results

    async def get_object_async(self, sha: ObjectID):
        """Get a Git object by SHA.

        Args:
            sha: Object SHA - can be 40-char hex bytes or 20-byte binary.
                 Pack operations handle both formats.
        """
        if not self._initialized:
            await self.initialize()

        type_num, data = await self._object_store.get_raw_async(sha)
        from dulwich.objects import ShaFile

        # ShaFile.from_raw_string expects 40-char hex SHA
        # Convert binary SHA (20 bytes) to hex if needed
        sha_for_shafile = sha
        if len(sha) == 20:
            sha_for_shafile = sha.hex().encode()

        return ShaFile.from_raw_string(type_num, data, sha=sha_for_shafile)

    async def has_object_async(self, sha: ObjectID) -> bool:
        """Check if an object exists."""
        if not self._initialized:
            await self.initialize()
        return await self._object_store.contains_async(sha)

    async def get_tree_at_path_async(
        self,
        path: str,
        commit_sha: Optional[ObjectID] = None,
    ) -> Optional[ObjectID]:
        """Get the tree SHA at a given path.

        Args:
            path: Path to traverse (e.g., "src/utils" or "")
            commit_sha: Commit to start from (defaults to HEAD)

        Returns:
            Tree SHA at the path, or None if not found
        """
        if not self._initialized:
            await self.initialize()

        # Get commit
        if commit_sha is None:
            commit_sha = await self.head_async()
            if commit_sha is None:
                return None

        # Get the commit object
        commit = await self.get_object_async(commit_sha)
        tree_sha = commit.tree

        # If path is empty or root, return the root tree
        if not path or path == "/" or path == ".":
            return tree_sha

        # Traverse the path
        parts = [p for p in path.strip("/").split("/") if p]
        current_tree_sha = tree_sha

        for part in parts:
            tree = await self.get_object_async(current_tree_sha)

            # Find the entry matching this path component
            found = False
            for entry in tree.items():
                name, mode, sha = entry
                if name.decode("utf-8", errors="replace") == part:
                    current_tree_sha = sha
                    found = True
                    break

            if not found:
                return None

        return current_tree_sha

    async def list_tree_async(
        self,
        path: str = "",
        commit_sha: Optional[ObjectID] = None,
    ) -> list[dict]:
        """List files and directories in a tree at the given path.

        Args:
            path: Path to list (e.g., "src/utils" or "")
            commit_sha: Commit to start from (defaults to HEAD)

        Returns:
            List of dicts with name, type (blob/tree), mode, sha, size
        """
        if not self._initialized:
            await self.initialize()

        # Get commit
        if commit_sha is None:
            commit_sha = await self.head_async()
            if commit_sha is None:
                return []

        # Get the commit object and root tree
        commit = await self.get_object_async(commit_sha)
        tree_sha = commit.tree

        # Navigate to the target path
        if path and path != "/" and path != ".":
            tree_sha = await self.get_tree_at_path_async(path, commit_sha)
            if tree_sha is None:
                return []

        # Get the tree object
        tree = await self.get_object_async(tree_sha)

        # Check if this is actually a tree
        from dulwich.objects import Tree

        if not isinstance(tree, Tree):
            # It's a blob (file), not a tree
            return []

        result = []
        for entry in tree.items():
            name, mode, sha = entry
            # Determine type based on mode
            # Mode 40000 is tree (directory)
            # Mode 100644 is regular file
            # Mode 100755 is executable
            # Mode 120000 is symlink
            # Mode 160000 is gitlink (submodule)
            if mode == 0o40000:
                entry_type = "tree"
                size = None
            else:
                entry_type = "blob"
                # Get blob size
                blob = await self.get_object_async(sha)
                size = len(blob.data)

            result.append(
                {
                    "name": name.decode("utf-8", errors="replace"),
                    "type": entry_type,
                    "mode": oct(mode),
                    "sha": sha.hex() if isinstance(sha, bytes) else sha,
                    "size": size,
                }
            )

        return result

    async def get_file_content_async(
        self,
        path: str,
        commit_sha: Optional[ObjectID] = None,
    ) -> Optional[bytes]:
        """Get the content of a file at the given path.

        Args:
            path: Path to the file (e.g., "src/main.py")
            commit_sha: Commit to read from (defaults to HEAD)

        Returns:
            File content as bytes, or None if not found
        """
        if not self._initialized:
            await self.initialize()

        if not path or path == "/" or path == ".":
            return None  # Can't read a directory as a file

        # Get commit
        if commit_sha is None:
            commit_sha = await self.head_async()
            if commit_sha is None:
                return None

        # Get the commit object
        commit = await self.get_object_async(commit_sha)
        tree_sha = commit.tree

        # Navigate to the parent directory
        parts = [p for p in path.strip("/").split("/") if p]
        current_tree_sha = tree_sha

        # Traverse to the parent of the target
        for i, part in enumerate(parts[:-1]):
            tree = await self.get_object_async(current_tree_sha)

            found = False
            for entry in tree.items():
                name, mode, sha = entry
                if name.decode("utf-8", errors="replace") == part:
                    current_tree_sha = sha
                    found = True
                    break

            if not found:
                return None

        # Get the final tree and find the file
        tree = await self.get_object_async(current_tree_sha)
        target_name = parts[-1]

        for entry in tree.items():
            name, mode, sha = entry
            if name.decode("utf-8", errors="replace") == target_name:
                # Check if it's a blob (not a tree)
                if mode == 0o40000:
                    return None  # It's a directory

                # Get the blob content
                blob = await self.get_object_async(sha)
                return blob.data

        return None

    async def get_file_info_async(
        self,
        path: str,
        commit_sha: Optional[ObjectID] = None,
    ) -> Optional[dict]:
        """Get information about a file at the given path.

        Args:
            path: Path to the file (e.g., "src/main.py")
            commit_sha: Commit to read from (defaults to HEAD)

        Returns:
            Dict with name, type, mode, sha, size, or None if not found
        """
        if not self._initialized:
            await self.initialize()

        if not path or path == "/" or path == ".":
            return None

        # Get commit
        if commit_sha is None:
            commit_sha = await self.head_async()
            if commit_sha is None:
                return None

        # Get the commit object
        commit = await self.get_object_async(commit_sha)
        tree_sha = commit.tree

        # Navigate to the parent directory
        parts = [p for p in path.strip("/").split("/") if p]
        current_tree_sha = tree_sha

        # Traverse to the parent of the target
        for i, part in enumerate(parts[:-1]):
            tree = await self.get_object_async(current_tree_sha)

            found = False
            for entry in tree.items():
                name, mode, sha = entry
                if name.decode("utf-8", errors="replace") == part:
                    current_tree_sha = sha
                    found = True
                    break

            if not found:
                return None

        # Get the final tree and find the entry
        tree = await self.get_object_async(current_tree_sha)
        target_name = parts[-1]

        for entry in tree.items():
            name, mode, sha = entry
            if name.decode("utf-8", errors="replace") == target_name:
                if mode == 0o40000:
                    entry_type = "tree"
                    size = None
                else:
                    entry_type = "blob"
                    blob = await self.get_object_async(sha)
                    size = len(blob.data)

                return {
                    "name": name.decode("utf-8", errors="replace"),
                    "type": entry_type,
                    "mode": oct(mode),
                    "sha": sha.hex() if isinstance(sha, bytes) else sha,
                    "size": size,
                }
