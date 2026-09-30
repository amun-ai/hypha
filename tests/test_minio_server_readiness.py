"""Issue #0008 — ``start_minio_server`` must return only when MinIO is USABLE,
not merely when its socket is open.

The gate used to poll ``/minio/health/live``. That is the **liveness** probe: it
succeeds as soon as the HTTP listener accepts connections, which is *before* the
object layer and the IAM subsystem have initialized. Every caller immediately
constructs a ``MinioClient``, whose first act is ``mc alias set`` — an IAM
operation — so on a loaded machine it could lose the race:

    Unable to initialize new alias from the provided credentials.
    Server not initialized yet, please try again.

In CI that surfaced as the far less obvious ``TimeoutError: Local auth server did
not start in time`` from the ``local_auth_server`` fixture (the real error was
buried in captured stdout), and because CI runs ``pytest -x`` it killed a whole
~40 minute run at roughly 50%. Observed on run 36725019021, build 3.11.

HONEST NOTE ON REPRODUCE-BEFORE-FIX: this is a race whose window is MinIO's
internal init time, which we cannot control from outside the process. On an idle
machine ``live`` and ``ready`` pass within ~1 ms of each other, so a test cannot
be made to fail deterministically against the old code — the evidence that it was
real is the CI failure above, not a red local test. What these tests *can* do,
and do, is pin the contract that was violated: when ``start_minio_server``
returns, the server is ready and an IAM operation succeeds immediately.

Real tests: a real MinIO binary, a real server process, a real ``mc`` client — no
mocks, because a mocked HTTP probe would pass against either endpoint and prove
nothing.
"""

import asyncio
import socket

import pytest
import requests

from hypha.minio import MinioClient, start_minio_server


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


@pytest.fixture(name="minio_proc")
def minio_proc_fixture(tmp_path):
    """Start a real MinIO server and hand back (url, user, password)."""
    port = _free_port()
    console_port = _free_port()
    proc, server_url, workdir = start_minio_server(
        executable_path="./bin",
        workdir=str(tmp_path / "data"),
        port=port,
        console_port=console_port,
        root_user="minioadmin",
        root_password="minioadmin",
        timeout=30,
    )
    assert proc is not None, "MinIO server failed to start"
    assert server_url, "start_minio_server returned no URL"
    yield server_url
    proc.terminate()
    proc.wait(timeout=30)


def test_readiness_probe_passes_the_moment_start_returns(minio_proc):
    """When ``start_minio_server`` returns, READINESS must already be satisfied.

    This is the contract the old ``live`` gate did not provide: it could return
    while the object layer was still initializing.
    """
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{minio_proc}/minio/health/ready", timeout=10)
    assert response.ok, (
        "start_minio_server returned before /minio/health/ready passed — the "
        f"gate is not waiting for readiness (got HTTP {response.status_code})"
    )


def test_iam_is_usable_immediately_after_start(minio_proc):
    """The actual regression: ``mc alias set`` must succeed with NO retry.

    ``MinioClient.__init__`` runs ``mc alias set``, which needs IAM. This is the
    exact call that failed in CI with "Server not initialized yet". Constructing
    the client right after the fixture returns is the real-world sequence, and an
    admin call afterwards proves IAM is genuinely serving rather than merely
    having accepted the alias.
    """
    client = MinioClient(
        minio_proc,
        "minioadmin",
        "minioadmin",
        executable_path="./bin",
    )

    async def _probe():
        # An admin read: fails if the IAM subsystem is not actually up.
        return await client.admin_user_list()

    users = asyncio.run(_probe())
    assert isinstance(users, list), f"admin_user_list returned {users!r}"


def test_liveness_is_not_a_sufficient_gate(minio_proc):
    """Document WHY the endpoint changed, so the choice is not undone casually.

    Both probes pass here — the point is that they are different endpoints with
    different meanings, and only ``ready`` speaks to usability. If MinIO ever
    removes ``/minio/health/ready`` this fails loudly instead of silently
    reverting the gate to a liveness check.
    """
    session = requests.Session()
    session.trust_env = False
    live = session.get(f"{minio_proc}/minio/health/live", timeout=10)
    ready = session.get(f"{minio_proc}/minio/health/ready", timeout=10)
    assert live.ok, "liveness probe unexpectedly missing"
    assert ready.ok, (
        "/minio/health/ready is not available on the pinned MinIO release — "
        "the start gate in hypha/minio.py depends on it"
    )
