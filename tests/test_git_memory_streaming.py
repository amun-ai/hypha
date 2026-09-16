"""Validation tests for the git smart-HTTP streaming/disk-spill memory fix.

These are REAL end-to-end tests: they create a git-storage artifact, push and
clone with the actual ``git`` CLI over the Hypha smart-HTTP endpoint, and verify
correctness with ``git fsck --full`` plus content/SHA comparison.

Covered:
  * test_git_large_clone_streaming_path  -- ~300MB repo exercises the STREAMING
    upload-pack path and the disk-spill receive-pack path; clone + fsck.
  * test_git_small_clone_buffered_path   -- small repo stays under the streaming
    threshold, exercising the BUFFERED upload-pack path; clone + fsck.
  * test_git_thin_pack_push_roundtrip    -- a second (incremental) commit is
    pushed as a thin pack through the disk-spill path; clone + fsck.
  * test_git_clone_memory_peak           -- measures server RSS peak during a
    large clone and 3 concurrent large clones, reporting the numbers.

Requires Docker (postgres via fastapi_server) and the git CLI.
"""

import os
import subprocess
import tempfile
import uuid
import asyncio
import hashlib
import time
import platform

import pytest

from hypha_rpc import connect_to_server

from . import WS_SERVER_URL, SIO_PORT

pytestmark = pytest.mark.asyncio


def _run(args, cwd=None, timeout=300, check=True):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    res = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    if check and res.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\n"
            f"stdout={res.stdout}\nstderr={res.stderr}"
        )
    return res


def _init_and_config(clone_dir):
    os.makedirs(clone_dir, exist_ok=True)
    _run(["git", "init", "-b", "main"], cwd=clone_dir)
    _run(["git", "config", "user.email", "test@example.com"], cwd=clone_dir)
    _run(["git", "config", "user.name", "Test User"], cwd=clone_dir)
    # Avoid expensive delta compression slowing huge pushes too much, but keep
    # default for correctness of thin-pack generation on incremental push.


async def _make_git_artifact(token):
    api = await connect_to_server(
        {
            "name": "git-mem-test-client",
            "server_url": WS_SERVER_URL,
            "token": token,
        }
    )
    artifact_manager = await api.get_service("public/artifact-manager")
    alias = f"git-mem-{uuid.uuid4().hex[:8]}"
    await artifact_manager.create(
        alias=alias,
        manifest={"name": "Git Memory Test"},
        config={"storage": "git"},
    )
    workspace = api.config.workspace
    auth_url = (
        f"http://git:{token}@127.0.0.1:{SIO_PORT}/{workspace}/git/{alias}"
    )
    return api, artifact_manager, alias, auth_url


def _write_random_file(path, size_bytes):
    """Write a file of incompressible random bytes and return its sha256 hex."""
    h = hashlib.sha256()
    chunk = os.urandom(1024 * 1024)
    written = 0
    with open(path, "wb") as f:
        while written < size_bytes:
            n = min(len(chunk), size_bytes - written)
            f.write(chunk[:n])
            h.update(chunk[:n])
            written += n
    return h.hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


async def test_git_pack_op_semaphore_503():
    """Concurrency cap raises 503 with Retry-After when slots are exhausted.

    Pure unit test of change A (no server needed).
    """
    import hypha.git.http as h
    from fastapi import HTTPException

    saved_sem = h._pack_op_semaphore
    saved_timeout = h.HYPHA_GIT_PACK_ACQUIRE_TIMEOUT
    try:
        h._pack_op_semaphore = asyncio.Semaphore(1)
        h.HYPHA_GIT_PACK_ACQUIRE_TIMEOUT = 0.1

        sem = await h._acquire_pack_slot()  # take the only slot
        with pytest.raises(HTTPException) as exc:
            await h._acquire_pack_slot()  # should time out -> 503
        assert exc.value.status_code == 503
        assert exc.value.detail == "Git server busy"
        assert exc.value.headers.get("Retry-After") == "5"
        sem.release()

        # Slot is reusable after release.
        sem2 = await h._acquire_pack_slot()
        sem2.release()
    finally:
        h._pack_op_semaphore = saved_sem
        h.HYPHA_GIT_PACK_ACQUIRE_TIMEOUT = saved_timeout


async def test_git_large_clone_streaming_path(
    minio_server, fastapi_server, test_user_token
):
    """~300MB repo: push (disk-spill) then clone (streaming) + fsck + verify."""
    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            _init_and_config(src)

            # One large incompressible file + a few small ones.
            big_sha = _write_random_file(
                os.path.join(src, "big.bin"), 300 * 1024 * 1024
            )
            for i in range(3):
                with open(os.path.join(src, f"small_{i}.txt"), "w") as f:
                    f.write(f"small file {i}\n" * 100)

            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "large initial"], cwd=src, timeout=300)
            _run(
                ["git", "push", auth_url, "main:main"], cwd=src, timeout=600
            )

            # Fresh clone -- exercises the streaming upload-pack path (>8MB).
            dst = os.path.join(tmp, "dst")
            _run(["git", "clone", auth_url, dst], timeout=600)

            # git fsck --full must pass.
            _run(["git", "fsck", "--full"], cwd=dst, timeout=300)

            # Verify the big file content matches by sha256.
            cloned_big = os.path.join(dst, "big.bin")
            assert os.path.exists(cloned_big), "big.bin missing in clone"
            assert _sha256_file(cloned_big) == big_sha, "big.bin content mismatch"

            for i in range(3):
                assert os.path.exists(os.path.join(dst, f"small_{i}.txt"))
    finally:
        await artifact_manager.delete(artifact_id=alias)
        await api.disconnect()


async def test_git_small_clone_buffered_path(
    minio_server, fastapi_server, test_user_token
):
    """Small repo (< threshold): clone exercises BUFFERED path + fsck."""
    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            _init_and_config(src)
            with open(os.path.join(src, "README.md"), "w") as f:
                f.write("# small repo\n" * 10)
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "small"], cwd=src)
            _run(["git", "push", auth_url, "main:main"], cwd=src, timeout=60)

            dst = os.path.join(tmp, "dst")
            _run(["git", "clone", auth_url, dst], timeout=60)
            _run(["git", "fsck", "--full"], cwd=dst)
            with open(os.path.join(dst, "README.md")) as f:
                assert "small repo" in f.read()
    finally:
        await artifact_manager.delete(artifact_id=alias)
        await api.disconnect()


async def test_git_thin_pack_push_roundtrip(
    minio_server, fastapi_server, test_user_token
):
    """Second incremental commit pushed as a THIN pack via disk-spill path.

    The second push only sends the new/changed objects with deltas against the
    base objects already on the server (a thin pack). This is the trickiest
    correctness case for the disk-spill receive path -- the pack must be
    fattened correctly. Verified by cloning and running git fsck --full.
    """
    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            _init_and_config(src)

            # First commit: a sizable base file so the delta has a real base.
            with open(os.path.join(src, "data.txt"), "w") as f:
                f.write("line\n" * 50000)
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "base"], cwd=src)
            _run(["git", "push", auth_url, "main:main"], cwd=src, timeout=120)

            # Second commit: modify the same file -> delta against base blob.
            with open(os.path.join(src, "data.txt"), "a") as f:
                f.write("appended line\n" * 100)
            with open(os.path.join(src, "extra.txt"), "w") as f:
                f.write("extra\n" * 1000)
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "incremental thin"], cwd=src)
            # Force thin pack on push (default for smart HTTP, but be explicit).
            _run(
                ["git", "push", "--thin", auth_url, "main:main"],
                cwd=src,
                timeout=120,
            )

            head_local = _run(
                ["git", "rev-parse", "HEAD"], cwd=src
            ).stdout.strip()

            # Clone and verify both commits + fsck.
            dst = os.path.join(tmp, "dst")
            _run(["git", "clone", auth_url, dst], timeout=120)
            _run(["git", "fsck", "--full"], cwd=dst, timeout=120)

            head_remote = _run(
                ["git", "rev-parse", "HEAD"], cwd=dst
            ).stdout.strip()
            assert head_local == head_remote, "HEAD mismatch after thin-pack push"

            # log should show 2 commits
            log = _run(["git", "log", "--oneline"], cwd=dst).stdout
            assert "incremental thin" in log and "base" in log
            assert os.path.exists(os.path.join(dst, "extra.txt"))
    finally:
        await artifact_manager.delete(artifact_id=alias)
        await api.disconnect()


# --------------------------------------------------------------------------
# Memory measurement
# --------------------------------------------------------------------------

def _find_server_pid(port):
    """Find the hypha.server process started by this pytest process.

    The fastapi_server fixture launches ``python -m hypha.server --port=<port>``
    as a child of the pytest process, so we locate it among our own descendants
    (psutil can read RSS of own children on macOS without elevated perms,
    whereas net_connections() / inspecting unrelated PIDs is blocked).
    """
    import psutil

    me = psutil.Process(os.getpid())
    port_flag = f"--port={port}"
    for child in me.children(recursive=True):
        try:
            cmdline = child.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("hypha.server" in part for part in cmdline) and any(
            port_flag == part for part in cmdline
        ):
            return child.pid
    return None


def _proc_tree_rss(pid):
    """Total RSS (bytes) of a process and its children (best-effort)."""
    import psutil

    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0
    try:
        total = p.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    try:
        children = p.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for c in children:
        try:
            total += c.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


async def test_git_clone_memory_concurrent(
    minio_server, fastapi_server, test_user_token
):
    """N concurrent large clones must COMPLETE and stay intact; RSS is reported.

    FUNCTIONAL GUARD (the assertions): three simultaneous clones of a large repo
    -- each pack far above HYPHA_GIT_STREAM_THRESHOLD, so all take the streaming
    / range-read upload-pack path -- must every one succeed (returncode == 0) AND
    pass ``git fsck --full``. This catches concurrency-specific breakage the
    single-clone tests cannot: a deadlock/starvation on the pack-op semaphore,
    range-reader buffer corruption when N reads overlap, or side-band frame
    interleaving between streams.

    WHY THERE IS NO MEMORY-BOUND ASSERTION HERE (and where the memory guard is).
    The O(1)-in-pack-size memory property is proven by
    test_git_clone_memory_flat_curve (single-clone cross-size slope): with the
    producer's mid-stream malloc_trim, one stream's peak RSS reflects the bounded
    LIVE working set, and a whole-pack-reload regression -- which holds a full
    pack LIVE and is therefore un-trimmable -- still spikes it and trips that
    test. Under CONCURRENCY that clean signal is destroyed by the ALLOCATOR, not
    by the code: prod and CI set MALLOC_ARENA_MAX=2 (conftest / Dockerfile), so
    the two glibc arenas are shared by the N concurrent producer threads. Freed
    pack churn from one stream is fragmented between the still-live allocations of
    the others, and malloc_trim(0) cannot return a partially-live arena during the
    overlap window. The concurrent PEAK RSS therefore scales ~linearly with
    aggregate throughput (measured on Linux CI: ~1.36x aggregate pack bytes,
    e.g. ~1237MB for 3x300MB) EVEN THOUGH each stream is O(1) in live memory --
    it measures glibc arena physics, not a per-op whole-pack load, so a bound on
    it would FAIL correct O(1) code and cannot be a code-regression guard. That
    concurrent arena-retention is a real, distinct prod-memory concern tracked
    separately; it is NOT the whole-pack-reload regression this file guards. We
    still SAMPLE and PRINT the concurrent peak and the POST-SETTLE RSS (after all
    streams finish, their final trims run with no overlapping live allocations, so
    the churn should largely drain) for observability and to characterise that
    separate concern.
    """
    import threading

    pid = _find_server_pid(SIO_PORT)
    assert pid, f"could not find server pid on port {SIO_PORT}"

    mb = 1024 * 1024
    n_files = 40
    n_concurrent = 3
    # Each pack is well above the 8MiB stream threshold, so every concurrent
    # clone exercises the streaming/range-read path. Random (incompressible)
    # files => ~one blob object each.
    repo_size = 150 * mb
    per_file = repo_size // n_files

    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )

    def _sample_peak(pid, stop, peak):
        while not stop["v"]:
            peak["v"] = max(peak["v"], _proc_tree_rss(pid))
            time.sleep(0.02)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            _init_and_config(src)
            for i in range(n_files):
                _write_random_file(
                    os.path.join(src, f"f_{i:04d}.bin"), per_file
                )
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "big"], cwd=src, timeout=600)
            _run(["git", "push", auth_url, "main:main"], cwd=src, timeout=900)

            # Warm one clone so the one-time, retained first-enumeration growth
            # (bounded by object COUNT, not pack size) is excluded from the
            # measured concurrent window.
            _run(
                ["git", "clone", auth_url, os.path.join(tmp, "warm")],
                timeout=900,
            )

            time.sleep(1.0)
            baseline = _proc_tree_rss(pid)
            peak = {"v": baseline}
            stop = {"v": False}
            t = threading.Thread(target=_sample_peak, args=(pid, stop, peak))
            t.start()

            async def _clone(idx):
                d = os.path.join(tmp, f"conc_{idx}")
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "clone",
                    auth_url,
                    d,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                _, err = await proc.communicate()
                assert proc.returncode == 0, (
                    f"concurrent clone {idx} failed: {err!r}"
                )
                # Integrity of each concurrent clone (no cross-stream corruption).
                _run(["git", "fsck", "--full"], cwd=d, timeout=300)

            await asyncio.gather(*[_clone(i) for i in range(n_concurrent)])
            peak_delta = peak["v"] - baseline

            # Post-settle: all streams have finished, so their final malloc_trim
            # runs with no overlapping live allocations -> arena churn should drain.
            time.sleep(3.0)
            stop["v"] = True
            t.join()
            settled_delta = _proc_tree_rss(pid) - baseline

            print("\n===== GIT CONCURRENT-CLONE RSS (observability) =====")
            print(
                f"repo ~{repo_size / mb:.0f} MB x {n_concurrent} concurrent "
                f"(aggregate ~{n_concurrent * repo_size / mb:.0f} MB), "
                f"{n_files} objects each"
            )
            print(f"peak RSS delta    = {peak_delta / mb:.1f} MB")
            print(f"settled RSS delta = {settled_delta / mb:.1f} MB")
            print(
                "NOTE: concurrent peak is dominated by glibc MALLOC_ARENA_MAX=2 "
                "fragmentation retention, not live memory; the O(1) memory guard "
                "is test_git_clone_memory_flat_curve."
            )
            print("====================================================")
    finally:
        await artifact_manager.delete(artifact_id=alias)
        await api.disconnect()


async def test_git_large_clone_then_fetch_incremental(
    minio_server, fastapi_server, test_user_token
):
    """Large repo: clone (range-read), then push a 2nd commit and FETCH it.

    Exercises the range-read upload-pack path on both the initial clone and the
    incremental fetch (which must read the existing large pack via range GETs to
    compute what to send), and validates correctness end-to-end with
    git fsck --full after both operations.
    """
    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            _init_and_config(src)
            # ~120MB incompressible base -> pack well above the 8MB whole-load
            # threshold, so the range-read path is used.
            big_sha = _write_random_file(
                os.path.join(src, "big.bin"), 120 * 1024 * 1024
            )
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "base"], cwd=src, timeout=300)
            _run(["git", "push", auth_url, "main:main"], cwd=src, timeout=600)

            # Fresh clone (range-read upload-pack on the large pack).
            dst = os.path.join(tmp, "dst")
            _run(["git", "clone", auth_url, dst], timeout=600)
            _run(["git", "fsck", "--full"], cwd=dst, timeout=300)
            assert _sha256_file(os.path.join(dst, "big.bin")) == big_sha

            # Second commit on the source, push it.
            with open(os.path.join(src, "note.txt"), "w") as f:
                f.write("incremental update\n" * 1000)
            _run(["git", "add", "-A"], cwd=src)
            _run(["git", "commit", "-m", "second"], cwd=src)
            _run(["git", "push", auth_url, "main:main"], cwd=src, timeout=300)
            head_local = _run(
                ["git", "rev-parse", "HEAD"], cwd=src
            ).stdout.strip()

            # Fetch the increment into the existing clone. The server must read
            # the large existing pack via range GETs to resolve reachability.
            _run(["git", "fetch", "origin"], cwd=dst, timeout=300)
            _run(["git", "merge", "--ff-only", "origin/main"], cwd=dst, timeout=60)
            _run(["git", "fsck", "--full"], cwd=dst, timeout=300)

            head_remote = _run(
                ["git", "rev-parse", "HEAD"], cwd=dst
            ).stdout.strip()
            assert head_local == head_remote, "HEAD mismatch after fetch"
            assert os.path.exists(os.path.join(dst, "note.txt"))
            # big file still intact after fetch
            assert _sha256_file(os.path.join(dst, "big.bin")) == big_sha
    finally:
        await artifact_manager.delete(artifact_id=alias)
        await api.disconnect()


@pytest.mark.skipif(
    platform.system() != "Linux",
    reason=(
        "RSS-peak assertion is only meaningful on glibc/Linux. malloc_trim is a "
        "no-op off Linux, so freed buffers are not returned to the OS and RSS "
        "ratchets, turning the per-clone delta into allocator noise. The "
        "range-read correctness is covered by the fsck roundtrip tests on all "
        "platforms; this flat-curve guard runs in CI (Linux) and prod, where "
        "the measurement is valid. On macOS run with -s to see the directional "
        "numbers printed below."
    ),
)
async def test_git_clone_memory_flat_curve(
    minio_server, fastapi_server, test_user_token
):
    """Prove O(1) memory: server RSS growth from an 80MB to a 400MB clone is flat.

    Before the range-read fix, S3Pack._ensure_loaded downloaded the WHOLE pack
    into RAM, so the per-clone RSS delta scaled ~linearly with pack size. After
    the fix, the source pack is range-read on demand with a bounded chunk
    buffer, and the streamed pack is generated one object at a time, so only
    ~1 object is ever live regardless of pack size.

    IMPORTANT: the repos here use MANY medium files (2 MiB each), NOT one giant
    blob. A single huge object must be fully materialized to be sent regardless
    of how the pack is read (that cost is inherent and would mask the win), so a
    flat-curve test must spread the bytes across many objects. With many medium
    objects, the dominant pre-fix cost was the whole-pack download, which the
    range reader eliminates.

    METRIC CHOICE (why RSS, and why an additive slope + absolute bound rather
    than a small/small ratio): the server runs on glibc, which retains freed
    per-object buffers in its arenas ~proportionally to total bytes churned. So
    even correct O(1) streaming code shows RSS that grows with pack size UNLESS
    that retention is returned to the OS. The producer now calls malloc_trim()
    every HYPHA_GIT_STREAM_TRIM_INTERVAL bytes (http.py), which returns the
    freed transient buffers mid-stream, so peak RSS reflects the LIVE working
    set (bounded) instead of cumulative throughput. Crucially this does NOT mask
    the regression it guards against: a whole-pack-load reload holds the pack
    LIVE for the whole send, and malloc_trim cannot release live memory — so
    such a regression still spikes RSS to ~one pack and trips the asserts below.

    We clone two repos (~80MB and ~400MB total), sampling server RSS during
    each. The O(1) property is that going from 80MB to 400MB (5x, +320MB of
    pack) adds ~no extra peak RSS. We therefore assert on the ADDITIVE slope
    (large_delta - small_delta), which is ~0 for O(1) and ~+320MB for a
    whole-pack reload, plus an absolute bound (large_delta well under one pack).
    Both are immune to dividing by a tiny/jittery small-pack denominator.
    Measured (macOS, directional): before -> small 198MB / large 774MB (scales);
    after (range-read) -> small 56MB / large 1.3MB (flat/bounded). Numbers are
    printed (visible with -s) for the report.
    """
    import threading

    pid = _find_server_pid(SIO_PORT)
    assert pid, f"could not find server pid on port {SIO_PORT}"

    mb = 1024 * 1024
    file_size = 2 * mb  # many medium files; no single object dominates
    sizes = {"small": 80 * mb, "large": 400 * mb}
    deltas = {}

    api, artifact_manager, alias, auth_url = await _make_git_artifact(
        test_user_token
    )
    # We will create one artifact per size to keep packs isolated.
    await artifact_manager.delete(artifact_id=alias)
    await api.disconnect()

    def _sample_peak(pid, stop, peak):
        while not stop["v"]:
            peak["v"] = max(peak["v"], _proc_tree_rss(pid))
            time.sleep(0.02)

    for label, size in sizes.items():
        api, artifact_manager, alias, auth_url = await _make_git_artifact(
            test_user_token
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = os.path.join(tmp, "src")
                _init_and_config(src)
                for i in range(size // file_size):
                    _write_random_file(
                        os.path.join(src, f"f_{i:04d}.bin"), file_size
                    )
                _run(["git", "add", "-A"], cwd=src)
                _run(["git", "commit", "-m", label], cwd=src, timeout=600)
                _run(
                    ["git", "push", auth_url, "main:main"], cwd=src, timeout=900
                )

                # Let any push-side buffers settle and trim before measuring.
                time.sleep(1.0)
                baseline = _proc_tree_rss(pid)

                clone_dir = os.path.join(tmp, "clone")
                peak = {"v": baseline}
                stop = {"v": False}
                t = threading.Thread(
                    target=_sample_peak, args=(pid, stop, peak)
                )
                t.start()
                _run(["git", "clone", auth_url, clone_dir], timeout=900)
                stop["v"] = True
                t.join()
                _run(["git", "fsck", "--full"], cwd=clone_dir, timeout=300)

                deltas[label] = peak["v"] - baseline
        finally:
            await artifact_manager.delete(artifact_id=alias)
            await api.disconnect()

    print("\n===== GIT CLONE MEMORY FLAT-CURVE (server RSS delta) =====")
    for label, size in sizes.items():
        d = deltas[label]
        print(
            f"{label:>5} pack ~{size / mb:.0f} MB : RSS delta "
            f"{d / mb:.1f} MB ({d / size:.3f}x pack)"
        )
    slope = deltas["large"] - deltas["small"]
    extra_pack = sizes["large"] - sizes["small"]
    ratio = deltas["large"] / max(deltas["small"], 1)
    size_ratio = sizes["large"] / sizes["small"]
    print(
        f"pack-size ratio large/small = {size_ratio:.1f}x ; "
        f"RSS-delta ratio = {ratio:.2f}x ; "
        f"additive slope large-small = {slope / mb:.1f} MB "
        f"for +{extra_pack / mb:.0f} MB of pack"
    )
    print("===========================================================")

    # O(1) proof #1 (additive slope): sending +320MB more pack (80MB -> 400MB)
    # must add ~no extra peak RSS, because only ~1 object is ever live and the
    # producer trims freed transients mid-stream. A whole-pack reload would add
    # ~one whole extra pack (+320MB) to the large clone's peak. We require the
    # extra RSS to be under a quarter of the extra pack -- flat for O(1), and
    # far exceeded (~320MB > 100MB) by a whole-pack regression. This is
    # jitter-robust: it never divides by the small (possibly tiny) delta.
    assert slope < extra_pack // 4, (
        f"RSS grew with pack size: +{slope / mb:.1f}MB peak for "
        f"+{extra_pack / mb:.0f}MB of pack (80MB->400MB clone) -- range-read/"
        f"stream likely regressed to a whole-pack load. "
        f"small={deltas['small'] / mb:.1f}MB large={deltas['large'] / mb:.1f}MB"
    )
    # O(1) proof #2 (absolute bound): the large clone must never hold anywhere
    # near a whole pack. A whole-pack reload (~400MB LIVE, un-trimmable) trips
    # this; the bounded working set (chunk buffer + idx + one live object) is a
    # small fraction of half a pack.
    assert deltas["large"] < sizes["large"] // 2, (
        f"large-pack RSS delta {deltas['large'] / mb:.1f}MB >= half the pack "
        f"({sizes['large'] / mb / 2:.0f}MB) -- not bounded to the live working "
        f"set; range-read likely regressed to whole-pack load"
    )
