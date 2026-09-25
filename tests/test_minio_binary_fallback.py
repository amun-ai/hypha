"""MinIO binaries must resolve even though dl.min.io stopped serving them.

In 2026 MinIO removed every free binary archive download from ``dl.min.io``:
each ``server``/``client`` archive URL — pinned version, most-recent version,
and the unversioned "latest" path — now returns HTTP 410 Gone, and no raw
binaries are published as GitHub release assets. Docker Hub's ``minio/minio``
is also access-restricted (401 anonymously). The only anonymously reachable
source left is the ``quay.io`` container images (``quay.io/minio/minio`` and
``quay.io/minio/mc``), which still bundle the ``/usr/bin/minio`` and
``/usr/bin/mc`` binaries.

Before the fix, ``setup_minio_executables`` gave up when the HTTP download
failed, leaving ``./bin`` empty; the ``minio_server`` test fixture then fell
back to ``docker run minio/minio:latest`` (Docker Hub, now 401) and raised
``RuntimeError``, halting the whole ``pytest -x`` matrix at the first
S3-dependent test. The fix adds a Docker-image extraction fallback so both the
server and client binaries are recovered from quay.io.

These are REAL tests: they invoke the genuine ``setup_minio_executables`` /
``_extract_binary_from_docker_image`` against real Docker and real quay.io
images — no mocks. They are gated on the same infrastructure the S3 test suite
already requires (Linux host + a working Docker daemon), because the images
carry Linux binaries that only execute on a Linux host.
"""

import os
import shutil
import sys

import pytest

from hypha.minio import (
    _extract_binary_from_docker_image,
    setup_minio_executables,
)

_DOCKER = shutil.which("docker") is not None

# The images ship Linux binaries; extraction is only meaningful on a Linux host
# with Docker (exactly the CI environment that runs the S3 suite).
requires_docker_linux = pytest.mark.skipif(
    sys.platform != "linux" or not _DOCKER,
    reason="requires a Linux host with Docker to extract MinIO binaries from "
    "the quay.io image (the S3 test suite already requires this)",
)

# Pinned versions matching the defaults in hypha/minio.py::setup_minio_executables.
MINIO_VERSION = "RELEASE.2024-07-16T23-46-41Z"
MC_VERSION = "RELEASE.2025-04-08T15-39-49Z"


@requires_docker_linux
def test_extract_mc_binary_from_quay_image(tmp_path):
    """The extraction helper copies a runnable ``mc`` binary out of the image."""
    dst = str(tmp_path / "mc")
    ok = _extract_binary_from_docker_image(
        f"quay.io/minio/mc:{MC_VERSION}", "/usr/bin/mc", dst
    )
    assert ok is True, "extraction from quay.io/minio/mc should succeed"
    assert os.path.exists(dst), "mc binary must be written to the destination"
    assert os.path.getsize(dst) > 1_000_000, "mc binary looks truncated"
    assert os.access(dst, os.X_OK), "extracted mc binary must be executable"


@requires_docker_linux
def test_setup_minio_executables_recovers_via_quay(tmp_path):
    """End-to-end: with dl.min.io dead, both binaries are recovered from quay.io.

    ``setup_minio_executables`` targets a fresh, empty directory, so the direct
    HTTP download is attempted first (and fails with 410 today). The fix's
    Docker-image fallback must then leave BOTH a runnable ``minio`` server and
    ``mc`` client on disk — the exact precondition the ``minio_server`` fixture
    needs to start and the ``--s3-admin-type=minio`` fixtures need for admin ops.
    """
    exe_dir = str(tmp_path / "bin")
    minio_version, mc_version, minio_path, mc_path = setup_minio_executables(exe_dir)

    assert minio_version == MINIO_VERSION
    assert mc_version == MC_VERSION

    assert os.path.exists(minio_path), "minio server binary must be present"
    assert os.access(minio_path, os.X_OK), "minio server binary must be executable"
    assert os.path.getsize(minio_path) > 1_000_000, "minio binary looks truncated"

    assert os.path.exists(mc_path), "mc client binary must be present"
    assert os.access(mc_path, os.X_OK), "mc client binary must be executable"
    assert os.path.getsize(mc_path) > 1_000_000, "mc binary looks truncated"


@pytest.mark.skipif(
    sys.platform == "linux",
    reason="characterizes the non-Linux guard; on Linux extraction is attempted",
)
def test_extraction_refuses_on_non_linux_host(tmp_path):
    """On non-Linux hosts the helper must refuse (a Linux binary won't run)."""
    dst = str(tmp_path / "mc")
    assert (
        _extract_binary_from_docker_image(
            f"quay.io/minio/mc:{MC_VERSION}", "/usr/bin/mc", dst
        )
        is False
    )
    assert not os.path.exists(dst), "no binary should be written on a non-Linux host"
