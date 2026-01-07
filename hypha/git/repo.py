"""S3-backed Git repository.

This module provides a Git repository implementation backed by S3,
combining the S3GitObjectStore and S3RefsContainer.
"""

import asyncio
import logging
from typing import Optional

from dulwich.objects import ObjectID
from dulwich.repo import BaseRepo

from hypha.git.object_store import S3GitObjectStore
from hypha.git.refs import S3InfoRefsContainer

logger = logging.getLogger(__name__)


class S3GitRepo(BaseRepo):
    """A Git repository backed by S3 storage.

    This repository stores all objects as pack files in S3 and
    references as individual S3 objects. It supports the full Git
    protocol via HTTP Smart transport.

    Usage:
        async with create_s3_client(config) as s3_client:
            repo = S3GitRepo(s3_client, "bucket", "prefix")
            await repo.initialize()

            # Use as a standard Git repo
            head = repo.head()
    """

    def __init__(
        self,
        s3_client,
        bucket: str,
        prefix: str,
        object_format=None,
    ):
        """Initialize S3 Git repository.

        Args:
            s3_client: Async S3 client (aiobotocore)
            bucket: S3 bucket name
            prefix: Prefix path in bucket (e.g., "workspace/artifacts/123/git")
            object_format: Git object format (defaults to SHA1)
        """
        self._s3_client = s3_client
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

        # Create object store and refs container
        self._object_store = S3GitObjectStore(
            s3_client, bucket, prefix, object_format=object_format
        )
        self._refs = S3InfoRefsContainer(
            s3_client, bucket, prefix, object_store=self._object_store
        )

        # Initialize base repo
        super().__init__(self._object_store, self._refs)

        self._initialized = False
        self.bare = True  # S3 repos are always bare

    @property
    def s3_client(self):
        return self._s3_client

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
        s3_client,
        bucket: str,
        prefix: str,
        default_branch: bytes = b"main",
    ) -> "S3GitRepo":
        """Initialize a new bare repository in S3.

        Args:
            s3_client: Async S3 client
            bucket: S3 bucket name
            prefix: Prefix path in bucket
            default_branch: Name of the default branch

        Returns:
            Initialized S3GitRepo instance
        """
        repo = cls(s3_client, bucket, prefix)

        # Create initial HEAD pointing to default branch
        await repo._refs.set_symbolic_ref_async(b"HEAD", b"refs/heads/" + default_branch)

        # Create the pack directory marker (not strictly necessary for S3)
        pack_dir_key = f"{prefix}/objects/pack/.gitkeep"
        await s3_client.put_object(Bucket=bucket, Key=pack_dir_key, Body=b"")

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
        """Get a Git object by SHA."""
        if not self._initialized:
            await self.initialize()

        type_num, data = await self._object_store.get_raw_async(sha)
        from dulwich.objects import ShaFile

        return ShaFile.from_raw_string(type_num, data, sha=sha)

    async def has_object_async(self, sha: ObjectID) -> bool:
        """Check if an object exists."""
        if not self._initialized:
            await self.initialize()
        return await self._object_store.contains_async(sha)
