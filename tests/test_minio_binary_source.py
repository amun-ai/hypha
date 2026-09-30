"""Issue #0007 — the MinIO binaries must come from a source that is actually
reachable, and the download must be integrity-checked.

Context: CI was blocked repo-wide because the binary supply disappeared. Every
``dl.min.io`` server/client archive URL (pinned, latest, and non-archive alike)
returns **HTTP 410 Gone**, and the container images that #1060 fell back to
(``quay.io/minio/*``) now reject anonymous pulls with **HTTP 401** — so every PR
died at the first S3 fixture with "Unable to find image ... locally". The
remaining free source is the GitHub release assets, which are published for
every version this project pins, on every platform, with a ``.sha256sum``
sidecar.

These are real network tests against the real release hosting — no mocks. That
is the entire point: a mocked URL builder would have stayed green through both
the 410 and the 401. The binary fetches use HTTP Range so the suite costs a few
hundred KB rather than the ~100 MB of a full download.
"""

import hashlib
import urllib.error
import urllib.request

import pytest

from hypha.minio import _download_verified, _github_release_url, _platform_tag

# The versions hypha pins in setup_minio_executables: the defaults and the pair
# used for file_system_mode. All four must be fetchable or some code path breaks.
PINNED = [
    ("minio", "minio", "RELEASE.2024-07-16T23-46-41Z"),
    ("mc", "mc", "RELEASE.2025-04-08T15-39-49Z"),
    ("minio", "minio", "RELEASE.2022-10-24T18-35-07Z"),
    ("mc", "mc", "RELEASE.2022-10-29T10-09-23Z"),
]

# Platforms hypha runs on: CI (linux-amd64), Apple Silicon and Intel dev
# machines, plus arm64 Linux images.
PLATFORM_TAGS = ["linux-amd64", "linux-arm64", "darwin-arm64", "darwin-amd64"]


def _range_get(url, nbytes=65536):
    """Fetch the first ``nbytes`` of ``url``; return (status, body)."""
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.read()


def test_platform_tag_matches_release_asset_naming():
    """The tag must be one MinIO actually publishes assets for."""
    assert _platform_tag() in PLATFORM_TAGS


def test_github_release_url_shape():
    """URL/sidecar construction, including the Windows .exe suffix."""
    url, sha_url = _github_release_url(
        "minio", "minio", "RELEASE.2024-07-16T23-46-41Z", "linux-amd64"
    )
    assert url == (
        "https://github.com/minio/minio/releases/download/"
        "RELEASE.2024-07-16T23-46-41Z/minio.linux-amd64.RELEASE.2024-07-16T23-46-41Z"
    )
    assert sha_url == url + ".sha256sum"

    win_url, win_sha = _github_release_url(
        "minio", "minio", "RELEASE.2024-07-16T23-46-41Z", "windows-amd64"
    )
    assert win_url.endswith(".exe")
    assert win_sha.endswith(".exe.sha256sum")


@pytest.mark.parametrize("repo,base,version", PINNED)
@pytest.mark.parametrize("platform_tag", PLATFORM_TAGS)
def test_pinned_binaries_are_anonymously_downloadable(
    repo, base, version, platform_tag
):
    """REGRESSION for #0007: every pinned binary must be fetchable WITHOUT auth.

    This is the check that was missing. It fails loudly the moment MinIO gates
    or removes a source — which is exactly how CI broke twice (410, then 401) —
    instead of surfacing as an unrelated-looking S3 fixture timeout.
    """
    url, sha_url = _github_release_url(repo, base, version, platform_tag)

    status, body = _range_get(url)
    assert status in (200, 206), f"{url} returned {status}"
    assert body, f"{url} returned an empty body"

    with urllib.request.urlopen(sha_url, timeout=60) as response:
        checksum = response.read().decode().split()[0].strip()
    assert len(checksum) == 64, f"{sha_url} is not a sha256 digest: {checksum!r}"
    int(checksum, 16)  # must be hex


def test_download_verified_writes_file_and_matches_checksum(tmp_path):
    """The happy path: a real (small) asset downloads and verifies end to end.

    Uses the sha256sum sidecar itself as the payload — a few dozen bytes — so
    the test exercises the real download/verify/rename path against real
    hosting without pulling a 100 MB binary.
    """
    payload_url = (
        "https://github.com/minio/minio/releases/download/"
        "RELEASE.2024-07-16T23-46-41Z/"
        "minio.linux-amd64.RELEASE.2024-07-16T23-46-41Z.sha256sum"
    )
    with urllib.request.urlopen(payload_url, timeout=60) as response:
        payload = response.read()
    expected = hashlib.sha256(payload).hexdigest()

    # Serve the expected digest from a local file:// sidecar.
    sha_file = tmp_path / "payload.sha256sum"
    sha_file.write_text(f"{expected}  payload\n")

    dst = tmp_path / "payload.bin"
    _download_verified(payload_url, sha_file.as_uri(), str(dst))

    assert dst.read_bytes() == payload
    assert not (tmp_path / "payload.bin.part").exists(), "temp file was not cleaned up"


def test_download_verified_rejects_checksum_mismatch(tmp_path):
    """A corrupted/substituted download must RAISE and leave no file behind.

    Verification is mandatory precisely because these binaries get executed; a
    silent pass here would turn a supply-chain problem into a mysterious MinIO
    startup failure much later.
    """
    payload_url = (
        "https://github.com/minio/minio/releases/download/"
        "RELEASE.2024-07-16T23-46-41Z/"
        "minio.linux-amd64.RELEASE.2024-07-16T23-46-41Z.sha256sum"
    )
    sha_file = tmp_path / "wrong.sha256sum"
    sha_file.write_text(f"{'0' * 64}  payload\n")

    dst = tmp_path / "payload.bin"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        _download_verified(payload_url, sha_file.as_uri(), str(dst))

    assert not dst.exists(), "a checksum-failed download must not be left in place"
    assert not (tmp_path / "payload.bin.part").exists(), "temp file was not cleaned up"


def test_dl_min_io_is_still_dead():
    """Documents WHY the source moved, and will fail if MinIO ever restores it.

    If this starts failing, dl.min.io is back and the comments in hypha/minio.py
    claiming it is permanently gone need revisiting. Asserting the actual status
    keeps that claim honest rather than frozen as folklore.
    """
    url = (
        "https://dl.min.io/server/minio/release/linux-amd64/archive/"
        "minio.RELEASE.2024-07-16T23-46-41Z"
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(url, timeout=60)
    assert exc.value.code == 410, f"dl.min.io now returns {exc.value.code}, not 410"
