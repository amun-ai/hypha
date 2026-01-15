"""S3-backed Git reference container.

This module provides Git reference storage (branches, tags) backed by S3,
with no local file system usage.

Uses fully async S3 operations:
- Read operations: Use async aiobotocore client factory
- Write operations: Use async aiohttp-based S3 client with AWS Signature V4
  (bypasses aiobotocore's put_object which hangs with MinIO)
"""

import logging
from typing import Callable, Optional

from botocore.exceptions import ClientError

from dulwich.objects import ObjectID
from dulwich.protocol import PEELED_TAG_SUFFIX
from dulwich.refs import (
    HEADREF,
    LOCAL_TAG_PREFIX,
    Ref,
    RefsContainer,
)

from hypha.git.s3_async import create_async_s3_client

logger = logging.getLogger(__name__)


class S3RefsContainer(RefsContainer):
    """Git references container backed by S3.

    Stores references as individual S3 objects:
    - refs/heads/{branch} -> branch references
    - refs/tags/{tag} -> tag references
    - HEAD -> symbolic reference to default branch

    This follows the "info/refs" format used by Git's dumb HTTP protocol,
    but stores refs as individual objects for atomic updates.

    Uses fully async S3 operations:
    - Read operations: Use async aiobotocore client factory
    - Write operations: Use async aiohttp-based S3 client
    """

    def __init__(
        self,
        s3_client_factory: Callable,
        bucket: str,
        prefix: str,
        object_store=None,
        s3_config: Optional[dict] = None,
    ):
        """Initialize S3 refs container.

        Args:
            s3_client_factory: Factory function that returns an async context manager
                               for S3 client (e.g., lambda: session.create_client("s3", ...))
            bucket: S3 bucket name
            prefix: Prefix path in bucket (e.g., "workspace/artifacts/123/.git")
            object_store: Optional object store for peeled tag resolution
            s3_config: S3 config dict with endpoint_url, access_key_id,
                       secret_access_key, region_name for async write operations
        """
        super().__init__()
        self._s3_client_factory = s3_client_factory
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._object_store = object_store
        self._refs_cache: dict[bytes, bytes] = {}
        self._initialized = False
        self._s3_config = s3_config

    async def initialize(self):
        """Initialize by loading all refs from S3."""
        if self._initialized:
            return
        await self._load_all_refs()
        self._initialized = True

    async def _load_all_refs(self):
        """Load all references from S3."""
        self._refs_cache.clear()

        async with self._s3_client_factory() as s3_client:
            # Load HEAD
            head = await self._read_ref_with_client(s3_client, HEADREF)
            if head:
                self._refs_cache[HEADREF] = head

            # List all refs
            paginator = s3_client.get_paginator("list_objects_v2")
            refs_prefix = f"{self._prefix}/refs/"

            try:
                async for page in paginator.paginate(
                    Bucket=self._bucket, Prefix=refs_prefix
                ):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        # Convert S3 key to ref name
                        ref_path = key[len(self._prefix) + 1 :]  # e.g., "refs/heads/main"
                        ref_name = ref_path.encode("utf-8")

                        # Read the ref value
                        value = await self._read_ref_with_client(s3_client, ref_name)
                        if value:
                            self._refs_cache[ref_name] = value
            except ClientError:
                pass  # No refs yet

    async def _read_ref_with_client(self, s3_client, name: bytes) -> Optional[bytes]:
        """Read a reference from S3 using provided client."""
        key = f"{self._prefix}/{name.decode('utf-8')}"
        try:
            response = await s3_client.get_object(Bucket=self._bucket, Key=key)
            data = await response["Body"].read()
            return data.strip()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return None
            raise

    async def _read_ref_async(self, name: bytes) -> Optional[bytes]:
        """Read a reference from S3."""
        async with self._s3_client_factory() as s3_client:
            return await self._read_ref_with_client(s3_client, name)

    async def _write_ref_async(self, name: bytes, value: bytes):
        """Write a reference to S3 using async aiohttp-based client.

        Uses aiohttp with AWS Signature V4 signing, which works reliably
        with MinIO unlike aiobotocore's put_object.
        """
        if not self._s3_config:
            raise RuntimeError(
                "s3_config is required for write operations. "
                "Pass s3_config dict with endpoint_url, access_key_id, "
                "secret_access_key, and region_name."
            )

        key = f"{self._prefix}/{name.decode('utf-8')}"
        async_client = create_async_s3_client(self._s3_config)
        try:
            await async_client.put_object(self._bucket, key, value + b"\n")
            self._refs_cache[name] = value
            logger.debug(f"Wrote ref {name.decode()} = {value.decode()}")
        except Exception:
            raise
        finally:
            await async_client.close()

    async def _delete_ref_async(self, name: bytes):
        """Delete a reference from S3 using native async."""
        key = f"{self._prefix}/{name.decode('utf-8')}"
        async with self._s3_client_factory() as s3_client:
            try:
                await s3_client.delete_object(Bucket=self._bucket, Key=key)
            except ClientError:
                pass
        self._refs_cache.pop(name, None)
        logger.debug(f"Deleted ref {name.decode()}")

    # =========================================================================
    # Async public methods
    # =========================================================================

    async def get_ref_async(self, name: Ref) -> Optional[ObjectID]:
        """Get a reference value asynchronously."""
        if not self._initialized:
            await self.initialize()

        if name in self._refs_cache:
            return self._refs_cache[name]

        # Try to read from S3 (might be new)
        value = await self._read_ref_async(name)
        if value:
            self._refs_cache[name] = value
        return value

    async def set_ref_async(
        self,
        name: Ref,
        value: ObjectID,
        old_value: Optional[ObjectID] = None,
    ) -> bool:
        """Set a reference value asynchronously.

        Args:
            name: Reference name (e.g., b"refs/heads/main")
            value: New SHA1 value
            old_value: Expected current value for CAS (optional)

        Returns:
            True if successful, False if CAS failed
        """
        if not self._initialized:
            await self.initialize()

        # Check old value if provided (CAS)
        if old_value is not None:
            current = self._refs_cache.get(name)
            if current != old_value:
                return False

        await self._write_ref_async(name, value)
        return True

    async def delete_ref_async(
        self,
        name: Ref,
        old_value: Optional[ObjectID] = None,
    ) -> bool:
        """Delete a reference asynchronously."""
        if not self._initialized:
            await self.initialize()

        if old_value is not None:
            current = self._refs_cache.get(name)
            if current != old_value:
                return False

        await self._delete_ref_async(name)
        return True

    async def get_all_refs_async(self) -> dict[Ref, ObjectID]:
        """Get all references."""
        if not self._initialized:
            await self.initialize()
        return dict(self._refs_cache)

    async def set_symbolic_ref_async(self, name: Ref, target: Ref):
        """Set a symbolic reference (e.g., HEAD -> refs/heads/main)."""
        value = b"ref: " + target
        await self._write_ref_async(name, value)

    async def get_symref_async(self, name: Ref) -> Optional[Ref]:
        """Get the target of a symbolic reference."""
        value = await self.get_ref_async(name)
        if value and value.startswith(b"ref: "):
            return value[5:]
        return None

    # =========================================================================
    # Synchronous methods required by RefsContainer
    # =========================================================================

    def allkeys(self):
        """Return all reference names."""
        return self._refs_cache.keys()

    def keys(self, base: Optional[Ref] = None):
        """Return reference names, optionally filtered by prefix."""
        for key in self._refs_cache:
            if base is None or key.startswith(base):
                yield key

    def __contains__(self, name: Ref) -> bool:
        """Check if reference exists."""
        return name in self._refs_cache

    def __getitem__(self, name: Ref) -> ObjectID:
        """Get a reference value."""
        value = self._refs_cache.get(name)
        if value is None:
            raise KeyError(name)

        # Handle symbolic refs
        if value.startswith(b"ref: "):
            target = value[5:]
            return self[target]

        return value

    def get_peeled(self, name: Ref) -> Optional[ObjectID]:
        """Get the peeled value of a reference."""
        # Check for cached peeled value
        peeled_name = name + PEELED_TAG_SUFFIX
        if peeled_name in self._refs_cache:
            return self._refs_cache[peeled_name]

        try:
            sha = self[name]
        except KeyError:
            return None

        # If we have an object store, resolve tags to their target
        if self._object_store is not None:
            try:
                from dulwich.objects import Tag

                obj = self._object_store[sha]
                while isinstance(obj, Tag):
                    sha = obj.object[1]
                    obj = self._object_store[sha]
            except KeyError:
                pass

        return sha

    def set_if_equals(
        self,
        name: Ref,
        old_ref: Optional[ObjectID],
        new_ref: ObjectID,
        message: Optional[bytes] = None,
    ) -> bool:
        """Atomic compare-and-swap for references.

        Note: This is synchronous. For async use set_ref_async.
        """
        if old_ref is not None:
            current = self._refs_cache.get(name)
            if current != old_ref:
                return False

        self._refs_cache[name] = new_ref
        return True

    def add_if_new(
        self,
        name: Ref,
        ref: ObjectID,
        message: Optional[bytes] = None,
    ) -> bool:
        """Add a reference only if it doesn't exist."""
        if name in self._refs_cache:
            return False
        self._refs_cache[name] = ref
        return True

    def remove_if_equals(
        self,
        name: Ref,
        old_ref: Optional[ObjectID],
        message: Optional[bytes] = None,
    ) -> bool:
        """Remove a reference if it matches the expected value."""
        if old_ref is not None:
            current = self._refs_cache.get(name)
            if current != old_ref:
                return False

        self._refs_cache.pop(name, None)
        return True

    def get_symrefs(self) -> dict[Ref, Ref]:
        """Get all symbolic references."""
        result = {}
        for name, value in self._refs_cache.items():
            if value.startswith(b"ref: "):
                result[name] = value[5:]
        return result

    def set_symbolic_ref(
        self,
        name: Ref,
        other: Ref,
        committer: Optional[bytes] = None,
        timestamp: Optional[float] = None,
        timezone: Optional[int] = None,
        message: Optional[bytes] = None,
    ) -> None:
        """Set a symbolic reference."""
        self._refs_cache[name] = b"ref: " + other


class S3InfoRefsContainer(S3RefsContainer):
    """Refs container optimized for Git info/refs format.

    This container can generate the info/refs file format used by
    Git's HTTP protocol for reference advertisement.
    """

    async def generate_info_refs(self) -> bytes:
        """Generate info/refs content for HTTP protocol."""
        if not self._initialized:
            await self.initialize()

        lines = []
        for name, value in sorted(self._refs_cache.items()):
            # Skip symbolic refs (they're advertised via capabilities)
            if value.startswith(b"ref: "):
                continue
            # Skip peeled entries (we'll add them separately)
            if name.endswith(PEELED_TAG_SUFFIX):
                continue

            lines.append(value + b"\t" + name + b"\n")

            # Add peeled value for tags
            if name.startswith(LOCAL_TAG_PREFIX):
                peeled = self.get_peeled(name)
                if peeled and peeled != value:
                    lines.append(peeled + b"\t" + name + PEELED_TAG_SUFFIX + b"\n")

        return b"".join(lines)

    async def update_from_dict(self, refs: dict[Ref, ObjectID]):
        """Update refs from a dictionary.

        This is useful for bulk updates after a push.
        """
        for name, value in refs.items():
            await self._write_ref_async(name, value)
