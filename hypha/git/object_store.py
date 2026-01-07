"""Async S3-backed Git object store.

This module provides an async implementation of a Git object store that
stores all objects (as pack files) in S3, with no local file system usage.
"""

import asyncio
import hashlib
import io
import logging
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any, BinaryIO, Optional

from botocore.exceptions import ClientError

from dulwich.object_store import (
    PACK_SPOOL_FILE_MAX_SIZE,
    BucketBasedObjectStore,
)
from dulwich.objects import ObjectID, ShaFile
from dulwich.pack import (
    Pack,
    PackData,
    iter_sha1,
    load_pack_index_file,
    write_pack_index,
)

logger = logging.getLogger(__name__)


class AsyncPackDataReader:
    """Wrapper to read pack data from S3 asynchronously."""

    def __init__(self, s3_client, bucket: str, key: str):
        self._s3_client = s3_client
        self._bucket = bucket
        self._key = key
        self._data: Optional[bytes] = None
        self._offset = 0

    async def load(self):
        """Load the pack data from S3."""
        if self._data is None:
            response = await self._s3_client.get_object(
                Bucket=self._bucket, Key=self._key
            )
            self._data = await response["Body"].read()
        return self._data

    def read(self, size: int = -1) -> bytes:
        """Synchronous read for dulwich compatibility."""
        if self._data is None:
            raise RuntimeError("Pack data not loaded. Call load() first.")
        if size == -1:
            result = self._data[self._offset :]
            self._offset = len(self._data)
        else:
            result = self._data[self._offset : self._offset + size]
            self._offset += size
        return result

    def seek(self, offset: int, whence: int = 0):
        """Seek to a position in the data."""
        if self._data is None:
            raise RuntimeError("Pack data not loaded. Call load() first.")
        if whence == 0:
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = len(self._data) + offset

    def tell(self) -> int:
        """Return current position."""
        return self._offset

    def close(self):
        """Close the reader."""
        self._data = None
        self._offset = 0


class S3Pack:
    """A Git pack stored in S3."""

    def __init__(
        self,
        basename: str,
        s3_client,
        bucket: str,
        pack_dir: str,
        object_format=None,
    ):
        self._basename = basename
        self._s3_client = s3_client
        self._bucket = bucket
        self._pack_dir = pack_dir
        self._object_format = object_format
        self._pack: Optional[Pack] = None
        self._pack_data: Optional[bytes] = None
        self._index_data: Optional[bytes] = None

    @property
    def name(self) -> str:
        return self._basename

    async def _ensure_loaded(self):
        """Ensure pack and index are loaded from S3."""
        if self._pack is not None:
            return

        # Load pack and index data in parallel
        pack_key = f"{self._pack_dir}/{self._basename}.pack"
        idx_key = f"{self._pack_dir}/{self._basename}.idx"

        pack_response, idx_response = await asyncio.gather(
            self._s3_client.get_object(Bucket=self._bucket, Key=pack_key),
            self._s3_client.get_object(Bucket=self._bucket, Key=idx_key),
        )

        self._pack_data = await pack_response["Body"].read()
        self._index_data = await idx_response["Body"].read()

        # Create in-memory pack
        pack_file = io.BytesIO(self._pack_data)
        idx_file = io.BytesIO(self._index_data)

        pack_data = PackData(self._basename + ".pack", file=pack_file, object_format=self._object_format)
        pack_index = load_pack_index_file(
            self._basename + ".idx", idx_file, self._object_format
        )

        self._pack = Pack.from_objects(pack_data, pack_index)

    def __contains__(self, sha: ObjectID) -> bool:
        """Check if SHA is in this pack (sync, requires pre-loading)."""
        if self._pack is None:
            return False
        return sha in self._pack

    async def contains_async(self, sha: ObjectID) -> bool:
        """Check if SHA is in this pack."""
        await self._ensure_loaded()
        return sha in self._pack

    async def get_raw_async(self, sha: ObjectID) -> tuple[int, bytes]:
        """Get raw object data."""
        await self._ensure_loaded()
        return self._pack.get_raw(sha)

    def get_raw(self, sha: ObjectID) -> tuple[int, bytes]:
        """Synchronous get_raw for dulwich compatibility."""
        if self._pack is None:
            raise RuntimeError("Pack not loaded. Use get_raw_async.")
        return self._pack.get_raw(sha)

    def get_stored_checksum(self) -> bytes:
        """Get the stored checksum of the pack."""
        if self._pack is None:
            raise RuntimeError("Pack not loaded.")
        return self._pack.get_stored_checksum()

    def close(self):
        """Close the pack."""
        if self._pack is not None:
            self._pack.close()
            self._pack = None
        self._pack_data = None
        self._index_data = None


class S3GitObjectStore(BucketBasedObjectStore):
    """Async S3-backed Git object store.

    This object store keeps all objects in pack files stored in S3.
    It does not support loose objects (all objects are packed).

    Usage:
        async with create_s3_client(config) as s3_client:
            store = S3GitObjectStore(s3_client, "my-bucket", "artifacts/123/git")
            await store.initialize()
            # Use store...
    """

    def __init__(
        self,
        s3_client,
        bucket: str,
        prefix: str,
        object_format=None,
    ):
        """Initialize the S3 Git object store.

        Args:
            s3_client: Async S3 client (aiobotocore)
            bucket: S3 bucket name
            prefix: Prefix path in bucket (e.g., "workspace/artifacts/123/git")
            object_format: Git object format (defaults to SHA1)
        """
        super().__init__(object_format=object_format)
        self._s3_client = s3_client
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._pack_dir = f"{self._prefix}/objects/pack"
        self._initialized = False

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
        """Initialize the object store, loading pack cache."""
        if self._initialized:
            return
        await self._update_pack_cache_async()
        self._initialized = True

    # =========================================================================
    # Async methods for S3 operations
    # =========================================================================

    async def _iter_pack_names_async(self) -> AsyncIterator[str]:
        """List all pack files in S3."""
        paginator = self._s3_client.get_paginator("list_objects_v2")
        prefix = self._pack_dir + "/"

        try:
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(".pack"):
                        # Extract basename: pack-{sha}.pack -> pack-{sha}
                        basename = key[len(prefix) :].replace(".pack", "")
                        yield basename
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                return
            raise

    async def _update_pack_cache_async(self) -> list[S3Pack]:
        """Update the pack cache by scanning S3."""
        pack_names = set()
        async for name in self._iter_pack_names_async():
            pack_names.add(name)

        # Create new packs
        new_packs = []
        for name in pack_names:
            if name not in self._pack_cache:
                pack = S3Pack(
                    name,
                    self._s3_client,
                    self._bucket,
                    self._pack_dir,
                    self.object_format,
                )
                self._pack_cache[name] = pack
                new_packs.append(pack)

        # Remove old packs
        for name in list(self._pack_cache.keys()):
            if name not in pack_names:
                old_pack = self._pack_cache.pop(name)
                old_pack.close()

        return new_packs

    def _upload_pack_sync(
        self, basename: str, pack_data: bytes, index_data: bytes
    ):
        """Synchronous pack upload - runs in thread pool.

        Uses synchronous botocore to avoid aiobotocore connection pool issues.
        """
        import boto3
        from botocore.config import Config

        pack_key = f"{self._pack_dir}/{basename}.pack"
        idx_key = f"{self._pack_dir}/{basename}.idx"

        # Create sync client with same credentials
        endpoint_url = self._s3_client._endpoint.host

        sync_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self._s3_client._request_signer._credentials.access_key,
            aws_secret_access_key=self._s3_client._request_signer._credentials.secret_key,
            region_name=self._s3_client._client_config.region_name,
            config=Config(signature_version='s3v4'),
        )

        try:
            logger.debug(f"Uploading pack file: {pack_key}")
            sync_client.put_object(
                Bucket=self._bucket, Key=pack_key, Body=pack_data
            )
            logger.debug(f"Uploading index file: {idx_key}")
            sync_client.put_object(
                Bucket=self._bucket, Key=idx_key, Body=index_data
            )
            logger.info(f"Uploaded pack {basename} to S3")
        finally:
            sync_client.close()

    async def _upload_pack_async(
        self, basename: str, pack_data: bytes, index_data: bytes
    ):
        """Upload pack and index files to S3.

        Uses synchronous boto3 in thread pool to avoid connection pool issues.
        """
        await asyncio.to_thread(
            self._upload_pack_sync, basename, pack_data, index_data
        )

    async def _delete_pack_async(self, basename: str):
        """Delete a pack from S3."""
        pack_key = f"{self._pack_dir}/{basename}.pack"
        idx_key = f"{self._pack_dir}/{basename}.idx"

        await asyncio.gather(
            self._s3_client.delete_object(Bucket=self._bucket, Key=pack_key),
            self._s3_client.delete_object(Bucket=self._bucket, Key=idx_key),
        )

        logger.info(f"Deleted pack {basename} from S3")

    async def contains_async(self, sha: ObjectID) -> bool:
        """Check if an object exists in the store."""
        for pack in self._pack_cache.values():
            if await pack.contains_async(sha):
                return True
        return False

    async def get_raw_async(self, sha: ObjectID) -> tuple[int, bytes]:
        """Get raw object data asynchronously."""
        for pack in self._pack_cache.values():
            if await pack.contains_async(sha):
                return await pack.get_raw_async(sha)
        raise KeyError(sha)

    def _parse_pack_sync(self, pack_data: bytes) -> tuple[list, bytes, bytes]:
        """Synchronous pack parsing - runs in thread pool.

        Returns:
            Tuple of (entries, checksum, index_data)
        """
        pack_file = io.BytesIO(pack_data)
        p = PackData("incoming.pack", file=pack_file, object_format=self.object_format)
        entries = p.sorted_entries()

        if not entries:
            return [], None, None

        # Get checksum and generate index
        checksum = p.get_stored_checksum()
        idx_file = io.BytesIO()
        write_pack_index(idx_file, entries, checksum, version=self.pack_index_version)
        idx_file.seek(0)
        index_data = idx_file.read()

        return entries, checksum, index_data

    async def add_pack_async(
        self, pack_data: bytes
    ) -> tuple[str, bytes]:
        """Add a pack to the object store.

        Args:
            pack_data: Raw pack file data

        Returns:
            Tuple of (basename, checksum)
        """
        logger.debug(f"add_pack_async: parsing pack data ({len(pack_data)} bytes)")

        # Run synchronous pack parsing in thread pool to avoid blocking event loop
        entries, checksum, index_data = await asyncio.to_thread(
            self._parse_pack_sync, pack_data
        )
        logger.debug(f"add_pack_async: got {len(entries)} entries")

        if not entries:
            logger.warning("Empty pack, not storing")
            return None, None

        # Generate basename from SHA of all entries
        basename = "pack-" + iter_sha1(entry[0] for entry in entries).decode("ascii")
        logger.debug(f"add_pack_async: basename = {basename}")

        # Check if pack already exists
        if basename in self._pack_cache:
            logger.debug("add_pack_async: pack already exists in cache")
            existing = self._pack_cache[basename]
            await existing._ensure_loaded()
            return basename, existing.get_stored_checksum()

        # Upload to S3
        logger.debug("add_pack_async: uploading to S3")
        await self._upload_pack_async(basename, pack_data, index_data)
        logger.debug("add_pack_async: upload complete")

        # Add to cache
        pack = S3Pack(
            basename,
            self._s3_client,
            self._bucket,
            self._pack_dir,
            self.object_format,
        )
        self._pack_cache[basename] = pack

        return basename, checksum

    async def add_objects_async(
        self,
        objects: list[tuple[ShaFile, Optional[str]]],
        progress: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """Add multiple objects to the store.

        Args:
            objects: List of (object, path) tuples
            progress: Optional progress callback

        Returns:
            Pack basename if objects were added, None otherwise
        """
        if not objects:
            return None

        # Create a new pack from the objects
        from dulwich.pack import write_pack_objects

        pack_file = io.BytesIO()

        # Write pack
        def generate_objects():
            for obj, path in objects:
                yield obj

        entries, data_sum = write_pack_objects(
            pack_file,
            generate_objects(),
            object_format=self.object_format,
        )

        pack_file.seek(0)
        pack_data = pack_file.read()

        basename, checksum = await self.add_pack_async(pack_data)
        return basename

    # =========================================================================
    # Synchronous methods required by BucketBasedObjectStore
    # These are used internally by dulwich but we provide async alternatives
    # =========================================================================

    def _iter_pack_names(self) -> Iterator[str]:
        """Synchronous pack name iteration - returns cached names."""
        return iter(self._pack_cache.keys())

    def _get_pack(self, name: str) -> S3Pack:
        """Get a pack by name."""
        return self._pack_cache[name]

    def _remove_pack_by_name(self, name: str):
        """Remove a pack by name. Use delete_pack_async instead."""
        raise NotImplementedError("Use delete_pack_async for S3 operations")

    def _upload_pack(
        self, basename: str, pack_file: BinaryIO, index_file: BinaryIO
    ):
        """Synchronous upload - not supported. Use add_pack_async."""
        raise NotImplementedError("Use add_pack_async for S3 operations")

    def _update_pack_cache(self) -> list[S3Pack]:
        """Synchronous cache update - returns empty, use async version."""
        return list(self._pack_cache.values())

    def _iter_loose_objects(self) -> Iterator[ObjectID]:
        """No loose objects in S3 store."""
        return iter([])

    def _get_loose_object(self, sha: ObjectID):
        """No loose objects in S3 store."""
        return None

    def contains_loose(self, sha: ObjectID) -> bool:
        """No loose objects in S3 store."""
        return False

    def contains_packed(self, sha: ObjectID) -> bool:
        """Check if object is in a pack (sync version)."""
        for pack in self._pack_cache.values():
            if sha in pack:
                return True
        return False

    def __contains__(self, sha: ObjectID) -> bool:
        """Check if object exists (sync version)."""
        return self.contains_packed(sha)

    def get_raw(self, sha: ObjectID) -> tuple[int, bytes]:
        """Get raw object (sync version, packs must be loaded)."""
        for pack in self._pack_cache.values():
            if sha in pack:
                return pack.get_raw(sha)
        raise KeyError(sha)

    def add_pack(self) -> tuple[BinaryIO, Callable[[], None], Callable[[], None]]:
        """Add a new pack - returns file object for writing.

        Note: The commit function stores pack data in memory.
        Use add_pack_async for actual S3 upload.
        """
        pf = tempfile.SpooledTemporaryFile(
            max_size=PACK_SPOOL_FILE_MAX_SIZE, prefix="incoming-"
        )

        pack_data_holder = {"data": None, "basename": None}

        def commit():
            if pf.tell() == 0:
                pf.close()
                return None

            pf.seek(0)
            pack_data_holder["data"] = pf.read()
            pf.close()

            # Store for later async upload
            return pack_data_holder["data"]

        return pf, commit, pf.close

    def add_object(self, obj: ShaFile):
        """Add a single object. Use add_objects_async for S3 upload."""
        raise NotImplementedError("Use add_objects_async for S3 operations")

    def add_objects(
        self,
        objects: list[tuple[ShaFile, Optional[str]]],
        progress: Optional[Callable[..., None]] = None,
    ):
        """Add objects. Use add_objects_async for S3 upload."""
        raise NotImplementedError("Use add_objects_async for S3 operations")

    def close(self):
        """Close all packs."""
        for pack in self._pack_cache.values():
            pack.close()
        self._pack_cache.clear()
