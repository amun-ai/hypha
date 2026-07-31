"""#0017 — a missing service on the HTTP/ASGI apps path -> clean 404.

A GET to ``/{workspace}/apps/{service_id}/...`` for a service that does not
exist (in an *existing*, accessible workspace) previously raised a bare
``KeyError('Service not found: ...')`` inside ``get_service_info``; the ASGI
handler caught it with its generic ``except Exception`` -> HTTP **500** + a full
traceback logged at ERROR:http. A missing service is a client-side lookup miss
(**404**), not a server fault, and must not spew a traceback into the error log.

This is the sibling case to #0014 (unknown *workspace* -> 404), but with a
crucial difference: #0014's ``WorkspaceNotFoundError`` is raised *locally* in
the same process as the ASGI middleware, so a local ``except`` catches it. The
service lookup here runs on the **workspace-manager** side and crosses the RPC
boundary, so a missing service arrives at the middleware as a
``hypha_rpc.rpc.RemoteException`` (the original ``KeyError`` type is lost — only
its message survives). A local ``except KeyError`` would therefore never fire.

Fix: the ASGI handler gains an ``except RemoteException`` arm (mirroring the
three ``/services`` function endpoints) that maps the exception to an HTTP
status via ``_get_status_for_remote_exception`` -> a missing service ("KeyError:
... not found") becomes **404** + ``logger.warning`` (no traceback), instead of
falling into the generic ``except Exception`` -> 500 + ERROR:http stack trace.
A *multiple*-match ambiguity raises ``AssertionError`` -> mapped to **400** with
its explanatory message, so it is surfaced, NOT masked. Genuine server faults
still map to 500 + an ERROR-level traceback (non-masking).
"""
import httpx
import pytest

from . import SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


async def test_missing_service_apps_path_returns_404(fastapi_server_sqlite):
    """A request for a nonexistent service under an existing workspace -> 404.

    The ``public`` workspace always exists and is readable, so this exercises
    the service-not-found path (NOT the unknown-workspace path of #0014).
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{SERVER_URL_SQLITE}/public/apps/no-such-service-xyz/whatever"
        )

    assert resp.status_code == 404, (
        f"expected 404 for a missing service, got {resp.status_code}: {resp.text}"
    )
    # Clean, client-facing message naming the missing service, no server traceback.
    assert "Service not found" in resp.text, (
        f"expected a 'Service not found' message, got: {resp.text}"
    )
