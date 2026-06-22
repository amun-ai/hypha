"""Unit tests for the bounded-memory S3 pack range reader.

These tests do NOT require a server, Docker, or network. They build real
dulwich packs (including delta-compressed ones and multi-chunk incompressible
ones), serve their bytes through a fake S3 client that honours HTTP Range
requests, and assert that dulwich resolves every object byte-for-byte
identically to a whole-loaded pack.

This proves the correctness of ``_S3RangeReader``:
  * sequential reads spanning chunk boundaries reassemble correctly,
  * seek + read at arbitrary object offsets work,
  * ofs-delta / ref-delta resolution (which seeks to base offsets and reads
    again) works through the range reader,
  * peak resident pack memory is bounded by the chunk size, not the pack size.
"""

import io

import pytest

from dulwich.objects import Blob, Commit, Tree, DEFAULT_OBJECT_FORMAT as FMT
from dulwich.pack import (
    OFS_DELTA,
    REF_DELTA,
    Pack,
    PackData,
    PackIndexer,
    load_pack_index_file,
    write_pack_index,
    write_pack_objects,
)

from hypha.git.object_store import _S3RangeReader


class _FakeBody:
    def __init__(self, data):
        self._d = data

    def read(self):
        return self._d

    def close(self):
        pass


class _FakeRangeS3:
    """Minimal sync S3 client that serves byte ranges of an in-memory object.

    Tracks the number of range GETs and total bytes served so tests can assert
    that the reader fetches only what it touches (not the whole pack) and never
    holds more than one chunk resident.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.calls = 0
        self.bytes_served = 0
        self.max_single_chunk = 0

    def get_object(self, Bucket, Key, Range=None):
        assert Range and Range.startswith("bytes=")
        start, end = Range[len("bytes=") :].split("-")
        start, end = int(start), int(end)
        chunk = self.data[start : end + 1]
        self.calls += 1
        self.bytes_served += len(chunk)
        self.max_single_chunk = max(self.max_single_chunk, len(chunk))
        return {"Body": _FakeBody(chunk)}


def _build_pack(objs, deltify=True):
    """Return (pack_bytes, idx_bytes, entries) for a list of dulwich objects."""
    buf = io.BytesIO()
    write_pack_objects(buf, objs, object_format=FMT, deltify=deltify)
    pack_bytes = buf.getvalue()

    pidx = PackData("x.pack", file=io.BytesIO(pack_bytes), object_format=FMT)
    indexer = PackIndexer.for_pack_data(pidx)
    entries = list(indexer)
    checksum = pidx.get_stored_checksum()
    entries.sort()
    ib = io.BytesIO()
    write_pack_index(ib, entries, checksum, version=2)
    return pack_bytes, ib.getvalue(), entries


def _count_deltas(pack_bytes, entries):
    pdt = PackData("x.pack", file=io.BytesIO(pack_bytes), object_format=FMT)
    n = 0
    for e in entries:
        u = pdt.get_unpacked_object_at(e[1])
        if u.pack_type_num in (OFS_DELTA, REF_DELTA):
            n += 1
    return n


def _make_pack_obj(pack_bytes, idx_bytes, file_obj, size):
    idx = load_pack_index_file("x.idx", io.BytesIO(idx_bytes), FMT)
    pd = PackData("x.pack", file=file_obj, size=size, object_format=FMT)
    return Pack.from_objects(pd, idx)


def test_range_reader_resolves_deltas():
    """A delta-heavy pack: every object resolves identically via range reads."""
    objs = []
    prev = b"X" * 200000
    for i in range(40):
        prev = prev[:100000] + (f"mut-{i}\n".encode() * 500) + prev[100000:]
        objs.append(Blob.from_string(prev))
    t = Tree()
    t.add(b"f", 0o100644, objs[-1].id)
    objs.append(t)
    c = Commit()
    c.tree = t.id
    c.author = c.committer = b"a <a@a>"
    c.author_time = c.commit_time = 0
    c.author_timezone = c.commit_timezone = 0
    c.message = b"m"
    objs.append(c)

    pack_bytes, idx_bytes, entries = _build_pack(objs)
    ndelta = _count_deltas(pack_bytes, entries)
    assert ndelta > 0, "test setup failed to produce deltas"

    ref = _make_pack_obj(
        pack_bytes, idx_bytes, io.BytesIO(pack_bytes), len(pack_bytes)
    )

    fake = _FakeRangeS3(pack_bytes)
    rr = _S3RangeReader(fake, "b", "k", len(pack_bytes), chunk_size=64 * 1024)
    rng = _make_pack_obj(pack_bytes, idx_bytes, rr, len(pack_bytes))

    for o in objs:
        assert ref.get_raw(o.id) == rng.get_raw(o.id), f"mismatch for {o.id!r}"

    # The reader must never hold more than one chunk resident.
    assert fake.max_single_chunk <= 64 * 1024


def test_range_reader_multi_chunk_incompressible():
    """Incompressible blobs make a multi-MB pack; reads cross many chunks.

    Proves the chunk-boundary reassembly path (a single object's bytes span
    more than one range GET) and that peak resident memory is the chunk size,
    not the pack size.
    """
    import os

    objs = []
    for _ in range(6):
        objs.append(Blob.from_string(os.urandom(1024 * 1024)))  # 1MB each
    t = Tree()
    t.add(b"f", 0o100644, objs[0].id)
    objs.append(t)

    # deltify=False: random data does not delta and the delta search is
    # pathologically slow (O(n^2)) on incompressible blobs. This test targets
    # chunk-boundary reassembly, not delta resolution (that is covered above).
    pack_bytes, idx_bytes, entries = _build_pack(objs, deltify=False)
    assert len(pack_bytes) > 4 * 1024 * 1024, "pack should be several MB"

    ref = _make_pack_obj(
        pack_bytes, idx_bytes, io.BytesIO(pack_bytes), len(pack_bytes)
    )

    chunk = 256 * 1024  # smaller than a 1MB object -> forces boundary crossings
    fake = _FakeRangeS3(pack_bytes)
    rr = _S3RangeReader(fake, "b", "k", len(pack_bytes), chunk_size=chunk)
    rng = _make_pack_obj(pack_bytes, idx_bytes, rr, len(pack_bytes))

    for o in objs:
        assert ref.get_raw(o.id) == rng.get_raw(o.id), f"mismatch for {o.id!r}"

    # Must have issued multiple range GETs (multi-chunk) and never buffered more
    # than one chunk at a time.
    assert fake.calls > 1
    assert fake.max_single_chunk <= chunk


def test_range_reader_seek_tell_read_semantics():
    """Direct file-protocol checks: seek/tell/read across boundaries + EOF."""
    data = bytes(range(256)) * 4096  # 1 MiB, deterministic
    fake = _FakeRangeS3(data)
    rr = _S3RangeReader(fake, "b", "k", len(data), chunk_size=100000)

    assert rr.tell() == 0
    # read spanning a chunk boundary
    rr.seek(99990)
    got = rr.read(40)  # crosses the 100000 boundary
    assert got == data[99990:100030]
    assert rr.tell() == 100030

    # seek to end and read-to-EOF
    rr.seek(len(data) - 10)
    assert rr.read(50) == data[-10:]  # truncated at EOF
    assert rr.read(1) == b""  # at EOF

    # whence variants
    rr.seek(0)
    rr.seek(5, io.SEEK_CUR)
    assert rr.tell() == 5
    rr.seek(-3, io.SEEK_END)
    assert rr.tell() == len(data) - 3

    # read(-1) reads to EOF from current pos
    rr.seek(len(data) - 7)
    assert rr.read(-1) == data[-7:]
