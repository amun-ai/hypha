"""Unit tests for ZIP64 archive handling in hypha.utils.zip_utils.

Regression tests for the bug where /zip-files/ endpoint failed with
"Corrupt zip64 end of central directory locator" on Zip64 archives
(>4 GB or >65535 entries). See peer report on artifact
reef-imaging/poc-hpa-plate1-zip.

These tests use an in-memory mock S3 client so no docker / postgres /
minio is required — they exercise fetch_zip_tail and get_zip_file_content
directly against synthesized Zip64 archives.
"""

import asyncio
import io
import json
import struct
import zipfile
from io import BytesIO

import pytest

from hypha.utils.zip_utils import (
    fetch_zip_tail,
    get_zip_file_content,
    read_file_from_zip,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Mock async S3 client
# ---------------------------------------------------------------------------


class _MockBody:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self):
        return self._data


class MockAsyncS3Client:
    """Minimal aiobotocore-shaped client backed by an in-memory bytes object.

    Supports head_object and get_object with Range, which is all the
    zip_utils code paths need.
    """

    def __init__(self, data: bytes):
        self._data = data
        self.range_calls = []  # for assertions in tests

    async def head_object(self, Bucket, Key):
        return {"ContentLength": len(self._data)}

    async def get_object(self, Bucket, Key, Range=None):
        if Range is None:
            self.range_calls.append(None)
            return {"Body": _MockBody(self._data)}
        assert Range.startswith("bytes="), Range
        spec = Range[len("bytes="):]
        start_str, end_str = spec.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else len(self._data) - 1
        self.range_calls.append((start, end))
        return {"Body": _MockBody(self._data[start : end + 1])}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_zip64_archive(num_entries: int = 65540, padding_bytes: int = 0) -> bytes:
    """Build a ZIP archive that uses ZIP64 EOCD format.

    Python's zipfile writes ZIP64 EOCD when total_entries > 0xFFFF, so we
    use 65540 small entries (just over 65535) to force ZIP64. An optional
    padding entry is added to push total file size past arbitrary
    thresholds in fetch_zip_tail.
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        # Add a couple of named files we can read back in tests.
        zf.writestr("metadata/.zattrs", json.dumps({"shape": [10, 10]}))
        zf.writestr("data/chunk-0", b"hello-zip64-world")
        # Bulk small entries to force the ZIP64 EOCD record.
        for i in range(num_entries):
            zf.writestr(f"e/{i:06d}", b"x")
        if padding_bytes:
            zf.writestr("padding/big.bin", b"\x00" * padding_bytes)
    return buf.getvalue()


def _assert_is_zip64(data: bytes) -> None:
    """Sanity check: the archive really uses ZIP64 EOCD format."""
    # ZIP64 End of Central Directory Locator signature
    assert b"PK\x06\x07" in data[-512:], (
        "Test archive does not contain a ZIP64 EOCD locator — "
        "test would not actually exercise the bug."
    )
    # ZIP64 End of Central Directory Record signature must also be present
    assert b"PK\x06\x06" in data, "Test archive is missing ZIP64 EOCD record"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_zip64_archive_is_actually_zip64():
    """Sanity: our fixture builder produces a ZIP64 archive."""
    archive = _build_zip64_archive(num_entries=65540)
    _assert_is_zip64(archive)


async def test_fetch_zip_tail_handles_zip64_above_50mb():
    """Regression for /zip-files/ failure on Zip64 archives.

    Builds a >50 MB ZIP64 archive (forcing fetch_zip_tail's partial path),
    runs fetch_zip_tail, then opens the result with the in-tree helper
    used by get_zip_file_content. Before the fix this raised
    BadZipFile('Corrupt zip64 end of central directory locator').
    """
    # 65540 entries (~5 MB of ZIP overhead) + 50 MB padding → ~55 MB archive.
    # That clears the < 50 MB shortcut in fetch_zip_tail, taking the partial
    # path that the bug actually lives on.
    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)
    assert len(archive) > 50 * 1024 * 1024, len(archive)

    s3 = MockAsyncS3Client(archive)
    zip_tail, tail_start_offset = await fetch_zip_tail(
        s3, bucket="test", key="zip64.zip", content_length=len(archive)
    )

    # The tail must cover from cd_offset onward — i.e. the tail length
    # must be at least (content_length - cd_offset). We don't have direct
    # access to cd_offset here, but we can verify by *opening* the ZIP via
    # the in-tree helper that get_zip_file_content uses.
    from hypha.utils.zip_utils import _open_zip_from_tail

    with _open_zip_from_tail(zip_tail, tail_start_offset, len(archive)) as zf:
        names = zf.namelist()
        assert "metadata/.zattrs" in names
        assert "data/chunk-0" in names
        assert len(names) >= 65540 + 3  # bulk entries + 3 named files


async def test_get_zip_file_content_lists_zip64_archive():
    """End-to-end against the get_zip_file_content path that the
    /zip-files/ endpoint calls. Lists the root directory of a Zip64
    archive that's large enough to hit the partial-fetch code path.
    """
    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)

    s3 = MockAsyncS3Client(archive)
    response = await get_zip_file_content(s3, bucket="b", key="big.zip", path="")
    # Directory listing returns JSONResponse with a list body.
    assert response.status_code == 200, response.body
    contents = json.loads(response.body)
    names = {item["name"] for item in contents}
    # Top-level entries we put in the archive: metadata/, data/, e/, padding/
    assert {"metadata", "data", "e", "padding"}.issubset(names), names


async def test_get_zip_file_content_extracts_small_file_from_zip64():
    """End-to-end: extract a small file by path from a Zip64 archive."""
    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)

    s3 = MockAsyncS3Client(archive)
    response = await get_zip_file_content(
        s3, bucket="b", key="big.zip", path="metadata/.zattrs"
    )
    assert response.status_code == 200, getattr(response, "body", response)
    # StreamingResponse — collect via body_iterator
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    body = b"".join(chunks)
    assert body == json.dumps({"shape": [10, 10]}).encode()


async def test_get_zip_file_content_extracts_data_chunk_from_zip64():
    """Extract a non-trivial path — covers the streaming-by-LFH-offset path."""
    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)

    s3 = MockAsyncS3Client(archive)
    response = await get_zip_file_content(
        s3, bucket="b", key="big.zip", path="data/chunk-0"
    )
    assert response.status_code == 200, getattr(response, "body", response)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    assert b"".join(chunks) == b"hello-zip64-world"


async def test_non_zip64_still_works_above_50mb():
    """Regression-guard: non-Zip64 archives in the partial-fetch path
    must still work after the fix (the existing code path that already
    handled them must not be broken)."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("hello.txt", b"plain-zip-hello")
        zf.writestr("padding.bin", b"\x00" * (55 * 1024 * 1024))
    archive = buf.getvalue()
    assert len(archive) > 50 * 1024 * 1024
    # Sanity: this archive has no ZIP64 markers.
    assert b"PK\x06\x07" not in archive[-512:]

    s3 = MockAsyncS3Client(archive)
    response = await get_zip_file_content(s3, bucket="b", key="plain.zip", path="hello.txt")
    assert response.status_code == 200, getattr(response, "body", response)
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    assert b"".join(chunks) == b"plain-zip-hello"


# ---------------------------------------------------------------------------
# Performance regression: parse the CD only once across requests
# ---------------------------------------------------------------------------


async def test_repeated_requests_parse_central_directory_only_once():
    """Regression for production perf cliff on a 236 GB / 3.6M-entry archive
    where /zip-files/ took 28–55 s per request.

    Even when the raw CD bytes are cached, ``zipfile.ZipFile()`` re-parses
    the entire central directory on every call (~20–30 s for a 270 MB CD
    with 3.6M entries). The fix is an in-process memo of the parsed
    {filename: ZipInfo} dict, keyed on (bucket, key, content_length), so
    sequential requests on the same replica skip the parse entirely.

    This test counts how many times the underlying parse fires across
    several requests against the same archive — it must stay at 1.
    """
    from hypha.utils import zip_utils

    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)

    s3 = MockAsyncS3Client(archive)

    # Reset the parse counter and any pre-existing memo state for this archive.
    zip_utils._zip_index_memo.clear()
    zip_utils._zip_index_parse_count = 0

    paths = ["metadata/.zattrs", "data/chunk-0", "metadata/.zattrs", "e/000000", "data/chunk-0"]
    for p in paths:
        response = await get_zip_file_content(s3, bucket="b", key="big.zip", path=p)
        assert response.status_code == 200, getattr(response, "body", response)
        # drain streaming response so the request fully completes
        if hasattr(response, "body_iterator"):
            async for _ in response.body_iterator:
                pass

    # The 5 sequential requests must share a single parse — that's the whole
    # point of the memo. If this fails, we're back to the perf cliff.
    assert zip_utils._zip_index_parse_count == 1, (
        f"Expected exactly 1 CD parse across 5 requests, "
        f"got {zip_utils._zip_index_parse_count}"
    )


async def test_directory_listing_uses_memo_too():
    """Directory listing must use the same in-process memo as file lookup."""
    from hypha.utils import zip_utils

    archive = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    _assert_is_zip64(archive)

    s3 = MockAsyncS3Client(archive)
    zip_utils._zip_index_memo.clear()
    zip_utils._zip_index_parse_count = 0

    # Mix of directory listings and file extractions on the same archive.
    response = await get_zip_file_content(s3, bucket="b", key="big.zip", path="")
    assert response.status_code == 200
    response = await get_zip_file_content(
        s3, bucket="b", key="big.zip", path="metadata/.zattrs"
    )
    assert response.status_code == 200
    if hasattr(response, "body_iterator"):
        async for _ in response.body_iterator:
            pass
    response = await get_zip_file_content(s3, bucket="b", key="big.zip", path="data/")
    assert response.status_code == 200

    assert zip_utils._zip_index_parse_count == 1, (
        f"Expected exactly 1 CD parse across mixed-mode requests, "
        f"got {zip_utils._zip_index_parse_count}"
    )


async def test_memo_is_keyed_on_archive_identity():
    """Different archives must each produce their own memo entry."""
    from hypha.utils import zip_utils

    archive_a = _build_zip64_archive(num_entries=65540, padding_bytes=50 * 1024 * 1024)
    archive_b = _build_zip64_archive(num_entries=65541, padding_bytes=50 * 1024 * 1024)
    s3_a = MockAsyncS3Client(archive_a)
    s3_b = MockAsyncS3Client(archive_b)

    zip_utils._zip_index_memo.clear()
    zip_utils._zip_index_parse_count = 0

    # Two different archives → two parses. Then repeated access → still two.
    for _ in range(3):
        await get_zip_file_content(s3_a, bucket="b", key="a.zip", path="metadata/.zattrs")
        await get_zip_file_content(s3_b, bucket="b", key="b.zip", path="metadata/.zattrs")

    assert zip_utils._zip_index_parse_count == 2, (
        f"Expected 2 parses (one per archive), got {zip_utils._zip_index_parse_count}"
    )
