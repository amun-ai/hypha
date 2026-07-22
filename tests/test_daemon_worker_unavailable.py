"""#0005 — daemon apps installed for an UNAVAILABLE worker type must not
ERROR-spam on every restart.

Prod (scilifelab-2-dev) logged `ERROR:apps:Failed to start daemon app ... 'No
server app worker found for type: compute-app'` ~11x on every restart, for demo
apps (cap-test / canary-demo / ab-demo / rolling-demo / git-probe) whose type
(`compute-app`) has no registered worker in that deployment. `compute-app` is
not a hypha worker type at all — so this is an ENVIRONMENTAL condition (worker
absent), not a broken app: it should be a single clear WARNING, not an ERROR.

Fix: `hypha/apps.py` raises a dedicated `WorkerNotFoundError` when no worker
supports an app type, and `prepare_workspace`'s daemon-autostart loop catches it
BEFORE the generic handler → WARNING (skip), keeping ERROR for genuine failures.
"""
import inspect

import pytest

from hypha_rpc import connect_to_server

from hypha.apps import ServerAppController, WorkerNotFoundError

from . import SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


def test_worker_not_found_error_is_exception_subclass():
    """WorkerNotFoundError must subclass Exception so existing `except Exception`
    handlers keep catching it (backward compatible), while allowing callers to
    catch it specifically."""
    assert issubclass(WorkerNotFoundError, Exception)


def test_worker_resolution_raises_worker_not_found_not_generic():
    """Every 'no server app worker found' site raises the dedicated
    WorkerNotFoundError (not a bare Exception), so the daemon loop can classify
    it. Source-contract check (server-side log levels aren't observable through
    the RPC/subprocess boundary; same static-assertion pattern as
    test_apps_scan_vs_keys.py)."""
    src = inspect.getsource(ServerAppController)
    # No bare `raise Exception("No server app worker found ...")` should remain.
    assert 'raise Exception(f"No server app worker found' not in src, (
        "worker-not-found must raise WorkerNotFoundError, not a bare Exception"
    )
    # And the dedicated type is used for the condition.
    assert "raise WorkerNotFoundError(" in src, (
        "expected WorkerNotFoundError to be raised on missing worker"
    )
    assert src.count("No server app worker found") >= 1


def test_prepare_workspace_downgrades_unavailable_worker_to_warning():
    """The daemon-autostart loop must catch WorkerNotFoundError BEFORE the
    generic Exception handler, logging it at WARNING (not ERROR). Locks the
    log-hygiene contract for #0005."""
    src = inspect.getsource(ServerAppController.prepare_workspace)

    # The specific handler exists and precedes the generic one.
    idx_specific = src.find("except WorkerNotFoundError")
    idx_generic = src.find("except Exception")
    assert idx_specific != -1, (
        "prepare_workspace must catch WorkerNotFoundError specifically"
    )
    assert idx_generic != -1, "prepare_workspace must still catch genuine failures"
    assert idx_specific < idx_generic, (
        "WorkerNotFoundError must be caught BEFORE the generic Exception handler"
    )

    # The WorkerNotFoundError branch logs at WARNING; the generic branch at ERROR.
    specific_branch = src[idx_specific:idx_generic]
    generic_branch = src[idx_generic:]
    assert "logger.warning" in specific_branch, (
        "unavailable-worker daemon app must be logged at WARNING, not ERROR"
    )
    assert "logger.error" not in specific_branch, (
        "the WorkerNotFoundError branch must not log at ERROR"
    )
    assert "logger.error" in generic_branch, (
        "a genuine daemon-start failure must still log at ERROR"
    )


async def test_unknown_worker_type_raises_clear_worker_not_found_over_rpc(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """End-to-end: the worker-not-found condition surfaces as a clear
    'No server app worker found' error (WorkerNotFoundError) over RPC — a clean,
    identifiable failure, not a crash or opaque error.

    This is the same condition the daemon-autostart loop now downgrades to a
    WARNING (rather than ERROR) on restart, when an app installed for a
    previously-available worker type finds that worker absent. Here we trigger it
    at install time, where the worker-availability check also lives (line ~1666):
    an app for a nonexistent worker type is rejected with the clear message, and
    the server stays healthy."""
    api = await connect_to_server(
        {
            "name": "worker-unavailable-test",
            "server_url": SERVER_URL_SQLITE,
            "token": test_user_token,
        }
    )
    controller = await api.get_service("public/server-apps")

    # An app for a worker type that does not exist in this deployment (mirrors
    # the prod 'compute-app' demo apps) is rejected with the clear worker-not-
    # found message — not an opaque error, not a crash.
    with pytest.raises(Exception, match=r"[Nn]o server app worker found"):
        await controller.install(
            manifest={
                "name": "Unavailable Worker Probe",
                "type": "nonexistent-worker-type-xyz",
                "entry_point": "index.html",
            },
            overwrite=True,
        )

    # The server is still healthy afterwards — a subsequent call succeeds.
    apps = await controller.list_apps()
    assert isinstance(apps, list), (
        "controller must stay healthy after a worker-not-found rejection"
    )

    await api.disconnect()
