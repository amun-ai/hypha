"""#0014 — unknown/inaccessible workspace on the HTTP/ASGI apps path -> 404.

A GET to ``/{workspace}/apps/{service}/...`` for a workspace that does not exist
(and cannot be auto-created for the caller) previously raised a bare ``KeyError``
inside ``get_workspace_interface``; the ASGI handler caught it with its generic
``except Exception`` -> HTTP **500** + a full traceback logged at ERROR. An
unknown workspace is a client error (**404**), not a server fault, and must not
spew a traceback into the ASGI error log.

Fix: ``store.load_or_create_workspace`` raises ``WorkspaceNotFoundError`` (a
``KeyError`` subclass, so the existing message contract in ``test_server.py`` is
preserved), and the ASGI handler catches it specifically -> 404 +
``logger.warning`` (no traceback). Any *other* ``KeyError`` still surfaces as a
500 (non-masking).
"""
import httpx
import pytest

from . import SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


async def test_unknown_workspace_apps_path_returns_404(fastapi_server_sqlite):
    """An app request under a nonexistent workspace returns 404, not 500."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(
            f"{SERVER_URL_SQLITE}/ws-does-not-exist-xyz/apps/no-such-svc/whatever"
        )

    assert resp.status_code == 404, (
        f"expected 404 for unknown workspace, got {resp.status_code}: {resp.text}"
    )
    # Clean, client-facing message (the workspace name), no server traceback.
    assert "does not exist or is not accessible" in resp.text
