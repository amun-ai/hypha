"""Utilities for working with ZIP files.

This module provides utilities for working with ZIP files stored in S3, particularly
for streaming the content of files from inside ZIP archives without having to download
the entire ZIP file first (for large archives).

The main functions are:
- fetch_zip_tail: Retrieve just the central directory of a ZIP file to read its structure
- get_zip_file_content: Stream content from a specific file inside a ZIP or list directory contents
- read_file_from_zip: Extract a single file from a ZIP archive
- stream_file_from_zip: Stream a file from a ZIP archive in chunks

The implementation uses an adaptive approach to handle ZIP files of different sizes:
- For small files (<50MB), it downloads the entire ZIP file for simplicity and reliability
- For larger files, it attempts to fetch just the central directory (tail) of the ZIP file
- The implementation supports both standard ZIP and ZIP64 formats
- For very large files, it implements streaming decompression to avoid loading the entire
  file content into memory at once

ZIP64 handling: when a partial tail is fetched from S3, fetch_zip_tail parses the
ZIP64 EOCD locator + record (or the regular EOCD record for non-ZIP64 archives) to
determine the actual offset and size of the central directory, then ensures the tail
covers [cd_offset, EOF]. _open_zip_from_tail wraps the partial tail in a sparse
file-like view so Python's zipfile module can resolve absolute offsets in the EOCD64
record and central directory entries — without that wrapper, Python would seek to
absolute offsets that lie before the start of the partial buffer and fail with
"Corrupt zip64 end of central directory locator/record".
"""

import asyncio
import io
import logging
import struct
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union, AsyncGenerator
import zipfile
from botocore.exceptions import ClientError
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.responses import Response
from datetime import datetime
import os
import zlib

# Setup logger
logger = logging.getLogger("zip_utils")


# --- ZIP / ZIP64 record signatures and sizes (per APPNOTE.TXT) ---------------
EOCD_SIGNATURE = b"PK\x05\x06"          # End of Central Directory Record
ZIP64_EOCD_LOCATOR_SIGNATURE = b"PK\x06\x07"  # ZIP64 EOCD Locator
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"    # ZIP64 EOCD Record
ZIP64_EOCD_LOCATOR_SIZE = 20            # 4 sig + 4 disk + 8 offset + 4 disks
ZIP64_EOCD_SIZE = 56                    # minimum, without extensible data sector
EOCD_SIZE = 22                          # minimum, without zip comment


class _SparseZipReader(io.RawIOBase):
    """Read-only file-like view of a partial ZIP archive tail.

    Bytes [0, tail_start) are virtual zeros; bytes [tail_start, total_size)
    come from `tail`. This lets ``zipfile.ZipFile`` resolve absolute offsets
    in the ZIP64 EOCD locator/record and central-directory entries even when
    we only have the tail of the file in memory — Python seeks to absolute
    offsets and we serve bytes correctly for any offset >= tail_start.
    Reads in the virtual-zero region return zero-filled bytes (zipfile only
    seeks there during ``open()``/``read()`` of an entry, which we never
    call: the higher-level streaming code uses ranged S3 reads keyed off
    ``ZipInfo.header_offset`` directly).
    """

    def __init__(self, tail: bytes, tail_start: int, total_size: int):
        if tail_start < 0:
            raise ValueError(f"tail_start must be >= 0, got {tail_start}")
        if tail_start + len(tail) != total_size:
            raise ValueError(
                f"tail must end at total_size: tail_start={tail_start}, "
                f"len(tail)={len(tail)}, total_size={total_size}"
            )
        self._tail = tail
        self._tail_start = tail_start
        self._total = total_size
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._total + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError(f"negative seek position: {new_pos}")
        self._pos = new_pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n=-1) -> bytes:
        if n is None or n < 0:
            n = max(0, self._total - self._pos)
        end = min(self._pos + n, self._total)
        if end <= self._pos:
            return b""
        out = bytearray()
        # Virtual-zero region [pos, min(end, tail_start))
        zero_end = min(end, self._tail_start)
        if self._pos < zero_end:
            out.extend(b"\x00" * (zero_end - self._pos))
        # Real-data region [max(pos, tail_start), end)
        actual_start = max(self._pos, self._tail_start)
        if actual_start < end:
            buf_off = actual_start - self._tail_start
            out.extend(self._tail[buf_off : buf_off + (end - actual_start)])
        self._pos = end
        return bytes(out)

    # zipfile.ZipFile may call readinto on a RawIOBase; provide it so that
    # zlib decompression and friends work even though we don't expect those
    # paths to fire here.
    def readinto(self, b) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def _open_zip_from_tail(
    zip_tail: bytes, tail_start_offset: int, total_size: int
) -> zipfile.ZipFile:
    """Open a ZipFile from a (possibly partial) tail of a ZIP archive.

    For complete files (tail_start_offset == 0 and tail length == total_size)
    uses BytesIO directly. For partial tails, wraps the bytes in a
    :class:`_SparseZipReader` so absolute offsets in EOCD64/CD records resolve
    correctly. With this wrapper, ``ZipInfo.header_offset`` returned from
    ``infolist()`` is the real absolute offset in the original archive — no
    post-hoc adjustment required.
    """
    if tail_start_offset == 0 and len(zip_tail) == total_size:
        return zipfile.ZipFile(BytesIO(zip_tail))
    return zipfile.ZipFile(_SparseZipReader(zip_tail, tail_start_offset, total_size))


def _parse_zip64_locator(buf: bytes, locator_offset: int) -> int:
    """Parse a ZIP64 EOCD locator and return the absolute offset of the
    ZIP64 EOCD record in the original archive.

    locator format (20 bytes total):
        4: signature (PK\\x06\\x07)
        4: number of the disk with EOCD64 (uint32)
        8: relative offset of EOCD64 record (uint64)
        4: total number of disks (uint32)
    """
    if buf[locator_offset : locator_offset + 4] != ZIP64_EOCD_LOCATOR_SIGNATURE:
        raise ValueError("Buffer does not start with ZIP64 EOCD locator signature")
    diskno, eocd64_offset, total_disks = struct.unpack(
        "<LQL", buf[locator_offset + 4 : locator_offset + 20]
    )
    if diskno != 0 or total_disks not in (0, 1):
        raise ValueError(
            f"ZIP spans multiple disks (diskno={diskno}, total_disks={total_disks}); "
            "multi-disk archives are not supported"
        )
    return eocd64_offset


def _parse_zip64_eocd(buf: bytes, record_offset: int) -> tuple[int, int]:
    """Parse a ZIP64 EOCD record and return (cd_offset, cd_size).

    record format (56 bytes minimum):
        4: signature (PK\\x06\\x06)
        8: size of zip64 EOCD record - 12
        2: version made by
        2: version needed to extract
        4: this disk number
        4: number of the disk with start of CD
        8: total entries on this disk
        8: total entries
        8: size of central directory (cd_size)
        8: offset of start of CD with respect to starting disk (cd_offset)
        ...: extensible data sector (variable)
    """
    if buf[record_offset : record_offset + 4] != ZIP64_EOCD_SIGNATURE:
        raise ValueError("Buffer does not start with ZIP64 EOCD record signature")
    cd_size, cd_offset = struct.unpack("<QQ", buf[record_offset + 40 : record_offset + 56])
    return cd_offset, cd_size


def _parse_eocd(buf: bytes, eocd_offset: int) -> tuple[int, int]:
    """Parse a regular (non-ZIP64) EOCD record and return (cd_offset, cd_size).

    Returns 0xFFFFFFFF for either field if the value overflowed and the real
    value lives in the ZIP64 EOCD record.
    """
    if buf[eocd_offset : eocd_offset + 4] != EOCD_SIGNATURE:
        raise ValueError("Buffer does not start with EOCD signature")
    cd_size, cd_offset = struct.unpack("<II", buf[eocd_offset + 12 : eocd_offset + 20])
    return cd_offset, cd_size


async def fetch_zip_tail(
    s3_client, bucket: str, key: str, content_length: int, cache_instance=None
) -> tuple[bytes, int]:
    """
    Fetch the tail part of the zip file that contains the central directory.
    Uses an adaptive approach to find the central directory, supporting both standard ZIP and ZIP64 formats.

    Args:
        s3_client: S3 client to use for fetching
        bucket: S3 bucket name
        key: S3 object key
        content_length: Total size of the ZIP file
        cache_instance: Optional Redis cache instance

    Returns:
        tuple: (zip_tail_bytes, tail_start_offset) where tail_start_offset is where the tail begins in the full file
    """
    # Check cache first if provided
    if cache_instance:
        cache_key = f"zip_tail:{bucket}:{key}:{content_length}"
        cached_data = await cache_instance.get(cache_key)
        if cached_data:
            logger.debug(f"Using cached ZIP tail for {key}")
            return cached_data

    # For small files (< 50MB), just download the entire file
    # This is more reliable for tests and small files
    if content_length < 50 * 1024 * 1024:  # < 50MB
        try:
            logger.debug(
                f"File is small ({content_length} bytes), downloading entire ZIP file"
            )
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            complete_zip = await response["Body"].read()

            # Verify this is a valid ZIP file
            try:
                with zipfile.ZipFile(BytesIO(complete_zip)) as zf:
                    # Just checking if it opens correctly
                    num_files = len(zf.namelist())
                    logger.debug(
                        f"Successfully loaded complete ZIP file ({len(complete_zip)} bytes) with {num_files} files"
                    )

                # Cache the entire file with tail start offset of 0 (full file)
                result = (complete_zip, 0)
                if cache_instance:
                    logger.debug(f"Caching entire ZIP file for {key}")
                    await cache_instance.set(
                        cache_key, result, ttl=3600
                    )  # Cache for 1 hour

                return result
            except zipfile.BadZipFile as e:
                logger.error(f"Downloaded file is not a valid ZIP file: {str(e)}")
                # Continue with partial download approach

        except Exception as e:
            logger.error(f"Failed to download entire ZIP file: {str(e)}")
            # Continue with partial download approach

    # Adaptive tail size based on file size
    MAX_TAIL = 128 * 1024 * 1024  # 128MB absolute maximum for very large ZIP files

    # Determine starting tail size based on file size
    # For small files (<10MB), start with 64KB
    # For medium files (10MB-100MB), start with 256KB
    # For large files (>100MB), start with 1MB
    # For very large files (>1GB), start with 4MB
    if content_length < 10 * 1024 * 1024:  # < 10MB
        min_tail = 64 * 1024  # 64KB
    elif content_length < 100 * 1024 * 1024:  # < 100MB
        min_tail = 256 * 1024  # 256KB
    elif content_length < 1024 * 1024 * 1024:  # < 1GB
        min_tail = 1024 * 1024  # 1MB
    else:
        min_tail = 4 * 1024 * 1024  # 4MB

    # Limit max tail to 10% of file size for small/medium files to avoid overhead
    if content_length < 100 * 1024 * 1024:
        MAX_TAIL = min(MAX_TAIL, max(content_length // 10, min_tail * 2))

    tail_length = min_tail
    found = False
    zip_tail = b""
    tail_start_offset = max(content_length - tail_length, 0)  # Initialize properly

    async def _ensure_tail_covers(target_start: int) -> None:
        """Extend zip_tail backwards so it covers [target_start, content_length).

        Mutates the enclosing zip_tail / tail_start_offset locals via nonlocal.
        """
        nonlocal zip_tail, tail_start_offset
        target_start = max(0, target_start)
        if target_start >= tail_start_offset:
            return
        # Fetch [target_start, tail_start_offset - 1] and prepend.
        range_header = f"bytes={target_start}-{tail_start_offset - 1}"
        logger.debug(
            f"Extending ZIP tail backwards to cover central directory: {range_header}"
        )
        response = await s3_client.get_object(
            Bucket=bucket, Key=key, Range=range_header
        )
        prefix = await response["Body"].read()
        zip_tail = prefix + zip_tail
        tail_start_offset = target_start

    # Adaptive approach: start small and grow until we find an EOCD/ZIP64
    # locator signature anywhere in the tail.
    eocd_offset_in_tail = -1
    zip64_locator_offset_in_tail = -1
    while tail_length <= MAX_TAIL:
        new_tail_start = max(content_length - tail_length, 0)
        # Only fetch the new part if we've already read a smaller tail
        if zip_tail and new_tail_start < tail_start_offset:
            # Fetch only the missing prefix
            range_header = f"bytes={new_tail_start}-{tail_start_offset - 1}"
            logger.debug(f"Fetching additional ZIP tail: {range_header}")
            response = await s3_client.get_object(
                Bucket=bucket, Key=key, Range=range_header
            )
            new_part = await response["Body"].read()
            zip_tail = new_part + zip_tail
            tail_start_offset = new_tail_start
        else:
            range_header = f"bytes={new_tail_start}-{content_length - 1}"
            logger.debug(f"Fetching ZIP tail: {range_header}")
            response = await s3_client.get_object(
                Bucket=bucket, Key=key, Range=range_header
            )
            zip_tail = await response["Body"].read()
            tail_start_offset = new_tail_start

        eocd_offset_in_tail = zip_tail.rfind(EOCD_SIGNATURE)
        zip64_locator_offset_in_tail = zip_tail.rfind(ZIP64_EOCD_LOCATOR_SIGNATURE)

        if eocd_offset_in_tail != -1 or zip64_locator_offset_in_tail != -1:
            found = True
            logger.debug(
                f"Found ZIP directory signatures in tail of size {len(zip_tail)} "
                f"(eocd@{eocd_offset_in_tail}, zip64_locator@{zip64_locator_offset_in_tail})"
            )
            break

        # Double the tail size and try again
        tail_length *= 2
        logger.debug(f"Increasing tail size to {tail_length}")

    if not found and content_length <= 100 * 1024 * 1024:
        # Last-resort fallback for moderate-sized files: download the whole thing.
        # We do NOT extend this fallback to bigger files — for true large ZIP64
        # archives the proper path below (parse EOCD64 → range-fetch CD) is far
        # cheaper.
        logger.warning(
            f"Central directory not found in tail, downloading entire ZIP file for {key}"
        )
        try:
            response = await s3_client.get_object(Bucket=bucket, Key=key)
            zip_tail = await response["Body"].read()
            tail_start_offset = 0
            try:
                with zipfile.ZipFile(BytesIO(zip_tail)):
                    found = True
                    logger.debug(
                        f"Successfully loaded complete ZIP file ({len(zip_tail)} bytes)"
                    )
            except zipfile.BadZipFile:
                logger.error(f"File is not a valid ZIP file: {key}")
        except Exception as e:
            logger.error(f"Failed to download entire ZIP file: {str(e)}")

    if not found:
        logger.warning(
            f"ZIP directory signatures not found in last {tail_length//2} bytes "
            f"of zip file {key}. Returning largest tail."
        )

    # Once we've located an EOCD or ZIP64 locator, parse it to determine the
    # actual offset and size of the central directory, and ensure the tail
    # covers [cd_offset, EOF]. Without this step, ZIP64 archives whose CD
    # spills past the initial tail (e.g. >65535 entries) can't be opened —
    # see comments at the top of this module.
    if found and tail_start_offset > 0:
        try:
            cd_start_required = await _resolve_cd_start(
                zip_tail,
                tail_start_offset,
                eocd_offset_in_tail,
                zip64_locator_offset_in_tail,
                s3_client,
                bucket,
                key,
                content_length,
            )
            if cd_start_required is not None and cd_start_required < tail_start_offset:
                await _ensure_tail_covers(cd_start_required)
        except Exception as e:
            logger.error(
                f"Failed to resolve central directory bounds for {key}: {e}",
                exc_info=True,
            )

    # Validation: open the (possibly partial) tail via the sparse-reader helper.
    # For partial tails this resolves absolute offsets correctly; for full files
    # it's equivalent to BytesIO. Failures here are logged but not fatal — the
    # caller can still surface a meaningful error.
    try:
        with _open_zip_from_tail(zip_tail, tail_start_offset, content_length) as zf:
            num_files = len(zf.namelist())
            logger.debug(
                f"Validated ZIP tail contains {num_files} files, "
                f"tail starts at offset {tail_start_offset}"
            )
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file structure: {str(e)}")

    result = (zip_tail, tail_start_offset)

    # Cache result if cache instance provided and we found something valid
    if cache_instance and found:
        logger.debug(f"Caching ZIP tail for {key}")
        await cache_instance.set(cache_key, result, ttl=3600)  # Cache for 1 hour

    return result


async def _resolve_cd_start(
    zip_tail: bytes,
    tail_start_offset: int,
    eocd_offset_in_tail: int,
    zip64_locator_offset_in_tail: int,
    s3_client,
    bucket: str,
    key: str,
    content_length: int,
) -> Optional[int]:
    """Determine the absolute offset where the central directory starts.

    For ZIP64 archives this requires parsing the ZIP64 EOCD locator + record;
    for plain ZIP archives it's the cd_offset field of the EOCD record.

    Returns None if the EOCD/EOCD64 records can't be located. Otherwise
    returns the absolute offset of the CD start (which may be smaller than
    tail_start_offset, indicating the tail must be extended).
    """
    # Prefer ZIP64 if its locator was found — the ZIP64 record is authoritative
    # when present, and the regular EOCD's cd_offset will be 0xFFFFFFFF in that
    # case anyway.
    if zip64_locator_offset_in_tail != -1:
        eocd64_record_offset = _parse_zip64_locator(
            zip_tail, zip64_locator_offset_in_tail
        )
        # The EOCD64 record itself may live before our current tail.
        if eocd64_record_offset < tail_start_offset:
            range_header = (
                f"bytes={eocd64_record_offset}-"
                f"{eocd64_record_offset + ZIP64_EOCD_SIZE - 1}"
            )
            logger.debug(f"Fetching ZIP64 EOCD record: {range_header}")
            resp = await s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
            record_bytes = await resp["Body"].read()
            cd_offset, cd_size = _parse_zip64_eocd(record_bytes, 0)
        else:
            local_offset = eocd64_record_offset - tail_start_offset
            cd_offset, cd_size = _parse_zip64_eocd(zip_tail, local_offset)
        logger.debug(
            f"ZIP64 central directory: cd_offset={cd_offset}, cd_size={cd_size}"
        )
        return cd_offset

    if eocd_offset_in_tail != -1:
        cd_offset, cd_size = _parse_eocd(zip_tail, eocd_offset_in_tail)
        # If the value overflowed (0xFFFFFFFF), this is actually a ZIP64
        # archive but we didn't find the locator — caller will surface the
        # error via _open_zip_from_tail.
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            logger.warning(
                f"EOCD reports overflow values for {key} but no ZIP64 locator was "
                "found in the tail; the tail may need to be extended further."
            )
            return None
        logger.debug(
            f"Standard central directory: cd_offset={cd_offset}, cd_size={cd_size}"
        )
        return cd_offset

    return None


async def get_zip_file_content(
    s3_client,
    bucket: str,
    key: str,
    path: str = "",
    zip_tail: Optional[Union[bytes, Tuple[bytes, int]]] = None,
    cache_instance=None,
) -> Response:
    """
    Extract and stream content from a ZIP file stored in S3.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        path: Path within the ZIP file to retrieve
        zip_tail: Optional pre-fetched ZIP tail data (can be bytes or tuple of (bytes, offset))
        cache_instance: Optional Redis cache instance

    Returns:
        Response: Either a StreamingResponse with file content or a JSONResponse with directory listing
    """
    try:
        # Fetch the ZIP file metadata to get its size
        try:
            zip_file_metadata = await s3_client.head_object(Bucket=bucket, Key=key)
            content_length = zip_file_metadata["ContentLength"]
            logger.info(f"[zip] ZIP file size: {content_length} bytes")
        except ClientError as e:
            # Check if the error is due to the file not being found (404)
            if e.response["Error"]["Code"] == "404":
                logger.error(f"[zip] ZIP file not found: {key}")
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "detail": f"ZIP file not found: {key}"},
                )
            else:
                # For other types of errors, raise a 500 error
                logger.error(f"[zip] Failed to fetch file metadata: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Failed to fetch file metadata: {str(e)}",
                    },
                )

        # Fetch the ZIP's central directory if not provided
        if not zip_tail:
            try:
                zip_tail, tail_start_offset = await fetch_zip_tail(
                    s3_client, bucket, key, content_length, cache_instance
                )
            except Exception as e:
                logger.error(f"[zip] Failed to fetch ZIP tail: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "detail": f"Failed to fetch ZIP file structure: {str(e)}",
                    },
                )
        else:
            # Handle both tuple and bytes formats
            if isinstance(zip_tail, tuple):
                # If it's already a tuple, unpack it
                zip_tail, tail_start_offset = zip_tail
            else:
                # If zip_tail was provided as bytes, assume it's the full file
                # (this is for backward compatibility with existing callers)
                tail_start_offset = 0

        # Open the in-memory ZIP tail and parse it. _open_zip_from_tail uses
        # a sparse-file wrapper for partial tails so absolute offsets in
        # EOCD64/CD records resolve correctly — without it, ZIP64 archives
        # whose CD lives at an offset > tail_start fail with "Corrupt zip64
        # end of central directory locator/record".
        try:
            zip_tail_with_offset = (zip_tail, tail_start_offset)
            with _open_zip_from_tail(
                zip_tail, tail_start_offset, content_length
            ) as zip_file:
                # If `path` ends with "/" or is empty, treat it as a directory listing
                if not path or path.endswith("/"):
                    logger.info(f"[zip] Listing directory: path={path}")
                    directory_contents = []

                    for zip_info in zip_file.infolist():
                        # Handle root directory or subdirectory files
                        if not path:
                            # Root directory - extract immediate children
                            relative_path = zip_info.filename.strip("/")
                            if "/" not in relative_path:
                                # Top-level file
                                directory_contents.append(
                                    {
                                        "type": (
                                            "file"
                                            if not zip_info.is_dir()
                                            else "directory"
                                        ),
                                        "name": relative_path,
                                        "size": (
                                            zip_info.file_size
                                            if not zip_info.is_dir()
                                            else None
                                        ),
                                        "last_modified": datetime(
                                            *zip_info.date_time
                                        ).timestamp(),
                                    }
                                )
                            else:
                                # Top-level directory
                                top_level_dir = relative_path.split("/")[0]
                                if not any(
                                    d["name"] == top_level_dir
                                    and d["type"] == "directory"
                                    for d in directory_contents
                                ):
                                    directory_contents.append(
                                        {"type": "directory", "name": top_level_dir}
                                    )
                        else:
                            # Subdirectory: Include only immediate children
                            if (
                                zip_info.filename.startswith(path)
                                and zip_info.filename != path
                            ):
                                relative_path = zip_info.filename[len(path) :].strip(
                                    "/"
                                )
                                if "/" in relative_path:
                                    # Subdirectory case
                                    child_name = relative_path.split("/")[0]
                                    if not any(
                                        d["name"] == child_name
                                        and d["type"] == "directory"
                                        for d in directory_contents
                                    ):
                                        directory_contents.append(
                                            {"type": "directory", "name": child_name}
                                        )
                                else:
                                    # File case
                                    directory_contents.append(
                                        {
                                            "type": "file",
                                            "name": relative_path,
                                            "size": zip_info.file_size,
                                            "last_modified": datetime(
                                                *zip_info.date_time
                                            ).timestamp(),
                                        }
                                    )

                    logger.info(
                        f"[zip] Directory listing complete, returning {len(directory_contents)} items"
                    )
                    return JSONResponse(status_code=200, content=directory_contents)

                # Otherwise, find the file inside the ZIP
                try:
                    zip_info = zip_file.getinfo(path)
                    logger.info(
                        f"[zip] Found file in ZIP: {path}, size: {zip_info.file_size}, compression: {zip_info.compress_type}"
                    )
                except KeyError:
                    logger.error(f"[zip] File not found inside ZIP: {path}")
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"File not found inside ZIP: {path}",
                        },
                    )

                # Set content type based on file extension if possible
                content_type = "application/octet-stream"
                file_extension = os.path.splitext(path)[1].lower()
                if file_extension:
                    # Basic content type mapping
                    content_type_map = {
                        ".txt": "text/plain",
                        ".html": "text/html",
                        ".htm": "text/html",
                        ".json": "application/json",
                        ".xml": "application/xml",
                        ".csv": "text/csv",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".pdf": "application/pdf",
                        ".js": "application/javascript",
                        ".css": "text/css",
                        ".md": "text/markdown",
                    }
                    content_type = content_type_map.get(file_extension, content_type)

                # Use our streaming implementation with proper chunking
                # Check file size to decide on the best chunk size for streaming
                if zip_info.file_size > 10 * 1024 * 1024:  # > 10MB
                    # For larger files, use larger chunks to improve throughput
                    chunk_size = 256 * 1024  # 256KB
                else:
                    # For smaller files, use smaller chunks for lower latency
                    chunk_size = 64 * 1024  # 64KB

                logger.info(
                    f"[zip] Returning StreamingResponse for {path} with chunk size {chunk_size}"
                )

                # Stream the file data
                return StreamingResponse(
                    stream_file_from_zip(
                        s3_client, bucket, key, zip_info, zip_tail_with_offset, chunk_size
                    ),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"',
                        "Content-Length": str(
                            zip_info.file_size
                        ),  # Add file size for client progress tracking
                    },
                )
        except zipfile.BadZipFile as e:
            logger.error(f"[zip] Bad ZIP file: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "detail": f"Invalid ZIP file: {str(e)}"},
            )

    except Exception as e:
        logger.error(f"[zip] Unhandled exception in get_zip_file_content: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": f"Internal server error: {str(e)}"},
        )


async def read_file_from_zip(s3_client, bucket, key, zip_info, zip_tail_data):
    """
    Extract a file from a ZIP archive efficiently by directly reading the compressed data
    and decompressing it on the fly.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail_data: Tuple of (zip_tail, tail_start_offset)

    Returns:
        bytes: The decompressed file content
    """
    try:
        # Unpack the tuple
        zip_tail, tail_start_offset = zip_tail_data
            
        # Get the total ZIP file size to determine if zip_tail contains the full file
        try:
            zip_file_metadata = await s3_client.head_object(Bucket=bucket, Key=key)
            total_zip_size = zip_file_metadata["ContentLength"]
        except Exception as e:
            logger.error(f"Failed to get ZIP file size: {str(e)}")
            # If we can't get the size, assume zip_tail is partial and use streaming approach
            total_zip_size = len(zip_tail) + 1  # Force streaming approach

        # Only use the simple approach if zip_tail contains the full ZIP file
        # AND the file inside the ZIP is small
        zip_tail_is_full_file = (tail_start_offset == 0 and len(zip_tail) == total_zip_size)

        if (
            zip_tail_is_full_file and zip_info.file_size < 10 * 1024 * 1024
        ):  # < 10MB and zip_tail is the full file
            with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                return zf.read(zip_info.filename)

        # For larger files or when zip_tail is partial, use streaming approach
        chunks = []
        async for chunk in stream_file_chunks_from_zip(
            s3_client, bucket, key, zip_info, (zip_tail, tail_start_offset)
        ):
            chunks.append(chunk)
        return b"".join(chunks)
    except Exception as e:
        logger.error(f"Error reading file from ZIP: {str(e)}")
        raise


async def stream_file_from_zip(
    s3_client, bucket, key, zip_info, zip_tail_data, chunk_size=64 * 1024
):
    """
    Stream a file from a ZIP archive, yielding chunks of decompressed data.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail_data: Tuple of (zip_tail, tail_start_offset)
        chunk_size: Size of decompressed chunks to yield

    Yields:
        bytes: Chunks of decompressed file data
    """
    try:
        # Unpack the tuple
        zip_tail, tail_start_offset = zip_tail_data
            
        # Get the total ZIP file size to determine if zip_tail contains the full file
        try:
            zip_file_metadata = await s3_client.head_object(Bucket=bucket, Key=key)
            total_zip_size = zip_file_metadata["ContentLength"]
        except Exception as e:
            logger.error(f"Failed to get ZIP file size: {str(e)}")
            # If we can't get the size, assume zip_tail is partial and use streaming approach
            total_zip_size = len(zip_tail) + 1  # Force streaming approach

        # Only use the simple approach if zip_tail contains the full ZIP file
        # AND the file inside the ZIP is small or uncompressed
        zip_tail_is_full_file = (tail_start_offset == 0 and len(zip_tail) == total_zip_size)

        if zip_tail_is_full_file and (
            zip_info.file_size < 1 * 1024 * 1024
            or zip_info.compress_type == zipfile.ZIP_STORED
        ):  # < 1MB and zip_tail is the full file
            with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                file_content = zf.read(zip_info.filename)
                for i in range(0, len(file_content), chunk_size):
                    yield file_content[i : i + chunk_size]
            return

        # For larger files or when zip_tail is partial, use the streaming implementation
        async for chunk in stream_file_chunks_from_zip(
            s3_client, bucket, key, zip_info, (zip_tail, tail_start_offset), chunk_size
        ):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming file from ZIP: {str(e)}")
        raise


async def stream_file_chunks_from_zip(
    s3_client, bucket, key, zip_info, zip_tail_data, chunk_size=64 * 1024
):
    """
    Stream a file from a ZIP archive using true streaming without loading the entire file into memory.
    This fetches compressed data in chunks and decompresses on-the-fly.

    Args:
        s3_client: S3 client
        bucket: S3 bucket name
        key: S3 object key for the ZIP file
        zip_info: ZipInfo object for the file to extract
        zip_tail_data: Tuple of (zip_tail, tail_start_offset)
        chunk_size: Size of decompressed chunks to yield

    Yields:
        bytes: Chunks of decompressed file data
    """
    # Unpack the tuple
    zip_tail, tail_start_offset = zip_tail_data
        
    # Get the total size of the ZIP file
    try:
        zip_file_metadata = await s3_client.head_object(Bucket=bucket, Key=key)
        total_zip_size = zip_file_metadata["ContentLength"]
    except Exception as e:
        logger.error(f"Failed to get ZIP file size: {str(e)}")
        raise
    
    # _open_zip_from_tail provides Python's zipfile with a sparse view that
    # makes ZipInfo.header_offset the real absolute offset in the original
    # archive, regardless of whether zip_tail covers the full file or only
    # the tail. No post-hoc offset adjustment is needed.
    zip_tail_is_partial = tail_start_offset > 0
    file_header_offset = zip_info.header_offset
    logger.debug(
        f"Using absolute offset for {zip_info.filename}: "
        f"header_offset={file_header_offset}, "
        f"tail_start={tail_start_offset}, "
        f"is_partial={zip_tail_is_partial}"
    )

    # Calculate data offset: header offset + size of the local file header + filename length + extra field length
    try:
        # Read the local file header from S3 to get the exact extra field length
        # The local header format is:
        # 4 bytes: signature (0x04034b50)
        # 2 bytes: version needed
        # 2 bytes: general purpose bit flag
        # 2 bytes: compression method
        # 2 bytes: last mod file time
        # 2 bytes: last mod file date
        # 4 bytes: crc-32
        # 4 bytes: compressed size
        # 4 bytes: uncompressed size
        # 2 bytes: filename length
        # 2 bytes: extra field length
        # Total: 30 bytes
        
        local_header_size = 30
        
        # Fetch the local file header to get the actual filename and extra field lengths
        range_header = f"bytes={file_header_offset}-{file_header_offset + local_header_size - 1}"
        response = await s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
        local_header = await response["Body"].read()
        
        # Parse the local header to get the actual lengths
        if len(local_header) >= 30:
            # Check signature
            signature = struct.unpack('<I', local_header[0:4])[0]
            if signature != 0x04034b50:
                logger.error(f"Invalid local header signature: {hex(signature)}")
                raise ValueError(f"Invalid ZIP local header for {zip_info.filename}")
            
            # Unpack the filename length and extra field length from the local header
            # These are at bytes 26-27 and 28-29 respectively
            filename_len_local = struct.unpack('<H', local_header[26:28])[0]
            extra_field_len_local = struct.unpack('<H', local_header[28:30])[0]
            
            # Calculate the offset where the actual file data begins
            data_offset = (
                file_header_offset
                + local_header_size
                + filename_len_local
                + extra_field_len_local
            )
        else:
            # Fallback: use the lengths from the central directory
            logger.warning(f"Could not read full local header, using central directory values")
            data_offset = (
                file_header_offset
                + local_header_size
                + len(zip_info.filename)
                + len(zip_info.extra)
            )

        compression_method = zip_info.compress_type
        compressed_size = zip_info.compress_size
        uncompressed_size = zip_info.file_size

        logger.info(
            f"Streaming file from ZIP: filename={zip_info.filename}, "
            f"compression={compression_method}, "
            f"compressed_size={compressed_size}, "
            f"uncompressed_size={uncompressed_size}, "
            f"data_offset={data_offset}, "
            f"zip_size={total_zip_size}, "
            f"tail_size={len(zip_tail)}, "
            f"is_partial={zip_tail_is_partial}"
        )
    except Exception as e:
        logger.error(f"Error calculating data offset: {str(e)}")
        # Fall back to simple approach only if zip_tail contains the full ZIP file
        if not zip_tail_is_partial:
            try:
                with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                    file_content = zf.read(zip_info.filename)
                    for i in range(0, len(file_content), chunk_size):
                        yield file_content[i : i + chunk_size]
                return
            except Exception as fallback_error:
                logger.error(f"Fallback approach also failed: {str(fallback_error)}")
                raise e  # Re-raise the original exception
        else:
            logger.error(
                f"Cannot use fallback approach: zip_tail is partial (size: {len(zip_tail)}, full size: {total_zip_size})"
            )
            raise e  # Re-raise the original exception since fallback is not possible

    # For DEFLATE compression, we need to use zlib to decompress
    if compression_method == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-15)  # -15 for raw deflate data

        # Fetch compressed data in chunks and decompress on-the-fly
        # Optimize chunk size based on compression ratio
        fetch_size = max(
            64 * 1024,
            min(1 * 1024 * 1024, int(chunk_size * compressed_size / uncompressed_size)),
        )

        # Stream the compressed data
        bytes_read = 0
        buffer = b""  # Buffer for decompressed data

        while bytes_read < compressed_size:
            # Calculate the byte range to fetch
            start = data_offset + bytes_read
            end = min(start + fetch_size - 1, data_offset + compressed_size - 1)
            range_header = f"bytes={start}-{end}"

            try:
                # Fetch a chunk of compressed data
                response = await s3_client.get_object(
                    Bucket=bucket, Key=key, Range=range_header
                )

                # Read the data chunk
                chunk = await response["Body"].read()
                bytes_read += len(chunk)

                # Decompress the chunk
                if bytes_read >= compressed_size:
                    # Last chunk - use flush
                    decompressed = (
                        buffer + decompressor.decompress(chunk) + decompressor.flush()
                    )
                else:
                    decompressed = buffer + decompressor.decompress(chunk)

                # Yield complete chunks
                for i in range(0, len(decompressed) - chunk_size + 1, chunk_size):
                    yield decompressed[i : i + chunk_size]

                # Keep remainder in buffer
                buffer = decompressed[
                    len(decompressed) - (len(decompressed) % chunk_size) :
                ]

            except Exception as e:
                logger.error(f"Error fetching or decompressing chunk: {str(e)}")
                raise

        # Yield any remaining data in the buffer
        if buffer:
            yield buffer

    elif compression_method == zipfile.ZIP_STORED:
        # For uncompressed files, directly stream the data
        bytes_read = 0
        while bytes_read < uncompressed_size:
            # Calculate the byte range to fetch
            start = data_offset + bytes_read
            end = min(start + chunk_size - 1, data_offset + uncompressed_size - 1)
            range_header = f"bytes={start}-{end}"

            try:
                # Fetch a chunk of data
                response = await s3_client.get_object(
                    Bucket=bucket, Key=key, Range=range_header
                )

                # Read and yield the data chunk
                chunk = await response["Body"].read()
                bytes_read += len(chunk)
                yield chunk

            except Exception as e:
                logger.error(f"Error fetching chunk: {str(e)}")
                raise
    else:
        # For other compression methods, fall back to the zipfile module only if zip_tail is the full file
        logger.info(
            f"Unsupported compression method {compression_method}, falling back to zipfile module"
        )
        if not zip_tail_is_partial:
            try:
                with zipfile.ZipFile(BytesIO(zip_tail)) as zf:
                    file_content = zf.read(zip_info.filename)
                    for i in range(0, len(file_content), chunk_size):
                        yield file_content[i : i + chunk_size]
            except Exception as fallback_error:
                logger.error(f"Zipfile fallback failed: {str(fallback_error)}")
                raise ValueError(
                    f"Unsupported compression method {compression_method}: {str(fallback_error)}"
                )
        else:
            logger.error(
                f"Cannot use zipfile fallback: zip_tail is partial (size: {len(zip_tail)}, full size: {total_zip_size})"
            )
            raise ValueError(
                f"Unsupported compression method {compression_method} and zip_tail is partial"
            )
