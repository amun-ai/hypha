"""Workspace unload debounce / grace period (#0007 server-side).

Background: on last-client-disconnect the workspace manager unloads an empty
non-persistent workspace immediately (`delete_client(unload=True)` ->
`unload_if_empty`). Apps that intentionally close+reconnect on a cadence (e.g.
a feedback sink reconnecting every ~31s) therefore churn the workspace
load/unload path thousands of times a day.

The debounce (`HYPHA_WORKSPACE_UNLOAD_GRACE_PERIOD`, seconds) defers that
last-client unload by a grace period; a reconnect within the window cancels the
pending unload (via the `client_connected` hook), and as a backstop the delayed
task itself re-checks emptiness before unloading. Default 0 = original
immediate-unload behavior (no fleet change).

Observability: a non-persistent workspace, once unloaded, is `hdel`'d from the
Redis "workspaces" hash and therefore disappears from `admin-utils
.list_workspaces()`. An anonymous connection lands in its own unique
non-persistent `ws-user-<id>` (websocket.py), which is the exact churn scenario.

The activity manager (inactive_period=300s) cannot fire within these few-second
windows and only tracks persistent workspaces, so it does not confound these
observations.
"""

import asyncio
import uuid

import pytest
from hypha_rpc import connect_to_server

from . import (
    SERVER_URL_SQLITE,
    SERVER_URL_UNLOAD_GRACE,
    UNLOAD_GRACE_PERIOD,
    find_item,
)

pytestmark = pytest.mark.asyncio


async def _connect_admin(server_url, root_user_token, client_id):
    """Root/admin connection with the admin-utils service for observation."""
    api = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": server_url,
            "token": root_user_token,
        }
    )
    admin_service = await api.get_service("admin-utils")
    return api, admin_service


async def _workspace_present(admin_service, ws_id):
    workspaces = await admin_service.list_workspaces()
    return find_item(workspaces, "id", ws_id) is not None


async def _wait_absent(admin_service, ws_id, deadline):
    """Poll until the workspace disappears; return True if it did within deadline."""
    loop_deadline = asyncio.get_event_loop().time() + deadline
    while asyncio.get_event_loop().time() < loop_deadline:
        if not await _workspace_present(admin_service, ws_id):
            return True
        await asyncio.sleep(0.2)
    return not await _workspace_present(admin_service, ws_id)


async def test_unload_debounced_then_eventually_unloads(
    fastapi_server_unload_grace, root_user_token
):
    """With a grace period > 0 the last-client unload is DEFERRED, not immediate,
    and the workspace is still eventually unloaded (no leak)."""
    admin_api, admin_service = await _connect_admin(
        SERVER_URL_UNLOAD_GRACE, root_user_token, "obs-root-grace"
    )

    # Anonymous client -> its own unique non-persistent ws-user-<id>.
    anon = await connect_to_server(
        {"client_id": "anon-grace", "server_url": SERVER_URL_UNLOAD_GRACE}
    )
    anon_ws = anon.config["workspace"]
    assert anon_ws.startswith("ws-user-"), f"unexpected anon workspace {anon_ws}"

    # Present while connected.
    assert await _workspace_present(admin_service, anon_ws)

    await anon.disconnect()

    # Within the grace window the workspace must still be loaded (deferred).
    # (Sampled well before the grace deadline; with grace=0 it would already be
    # gone by now — see test_unload_immediate_when_grace_zero.)
    await asyncio.sleep(1.0)
    assert await _workspace_present(admin_service, anon_ws), (
        f"workspace {anon_ws} unloaded before grace period elapsed — "
        f"debounce did not defer the unload"
    )

    # After the grace period (+ margin) the delayed task must unload it.
    unloaded = await _wait_absent(
        admin_service, anon_ws, deadline=UNLOAD_GRACE_PERIOD + 4.0
    )
    assert unloaded, (
        f"workspace {anon_ws} was not unloaded after the grace period — "
        f"debounced unload leaked"
    )

    await admin_api.disconnect()


async def test_unload_immediate_when_grace_zero(
    fastapi_server_sqlite, root_user_token
):
    """Regression guard: with no grace period configured (default 0) the
    last-client unload happens immediately — the original behavior is unchanged."""
    admin_api, admin_service = await _connect_admin(
        SERVER_URL_SQLITE, root_user_token, "obs-root-immediate"
    )

    anon = await connect_to_server(
        {"client_id": "anon-immediate", "server_url": SERVER_URL_SQLITE}
    )
    anon_ws = anon.config["workspace"]
    assert anon_ws.startswith("ws-user-")
    assert await _workspace_present(admin_service, anon_ws)

    await anon.disconnect()

    # grace=0 -> immediate unload. It must be gone well before the grace-fixture's
    # grace period would have elapsed (proves default behavior is unchanged).
    unloaded = await _wait_absent(
        admin_service, anon_ws, deadline=max(2.0, UNLOAD_GRACE_PERIOD - 0.5)
    )
    assert unloaded, (
        f"workspace {anon_ws} not unloaded with grace=0 — immediate unload "
        f"(default behavior) regressed"
    )

    await admin_api.disconnect()


async def test_reconnect_within_grace_keeps_workspace(
    fastapi_server_unload_grace, root_user_token
):
    """A client that reconnects within the grace window and STAYS connected must
    not be torn down at the original unload deadline.

    Timing (grace = G):
      t0        : client A disconnects   -> unload scheduled @ t0+G
      t0+1      : client B connects+stays -> connect hook cancels the pending
                  unload; as a backstop the delayed task would also re-check
                  emptiness and abort because B is present.
      t0+G+1    : assert PRESENT + B works -> the returning client was NOT torn
                  down out from under it at the original deadline.
      after B   : assert ABSENT           -> once B leaves it eventually unloads
                  (no leak).

    This guards BOTH the connect-time cancel hook and the delayed task's
    emptiness re-check (a naive `unload(force=True)` at the deadline would kill
    B's live workspace and fail here). Uses a token-bound non-persistent named
    workspace so A and B rejoin the SAME workspace (anonymous reconnects would
    each get a fresh ws-user-<id>).
    """
    admin_api, admin_service = await _connect_admin(
        SERVER_URL_UNLOAD_GRACE, root_user_token, "obs-root-recon"
    )

    ws_name = f"ws-debounce-recon-{uuid.uuid4().hex[:8]}"
    await admin_api.create_workspace(
        {
            "name": ws_name,
            "description": "debounce reconnect test (non-persistent)",
            "persistent": False,
        }
    )
    ws_token = await admin_api.generate_token({"workspace": ws_name})

    async def _connect_member(client_id):
        return await connect_to_server(
            {
                "client_id": client_id,
                "server_url": SERVER_URL_UNLOAD_GRACE,
                "token": ws_token,
                "workspace": ws_name,
            }
        )

    # Client A loads the workspace, then disconnects (schedules unload @ t0+G).
    api_a = await _connect_member("member-a")
    assert api_a.config["workspace"] == ws_name
    assert await _workspace_present(admin_service, ws_name)

    t0 = asyncio.get_event_loop().time()
    await api_a.disconnect()

    # Reconnect B within the grace window and keep it connected.
    await asyncio.sleep(1.0)
    api_b = await _connect_member("member-b")
    assert api_b.config["workspace"] == ws_name

    # Sleep past the ORIGINAL deadline (t0 + G) while B stays connected.
    now = asyncio.get_event_loop().time()
    target = t0 + UNLOAD_GRACE_PERIOD + 1.0
    if target > now:
        await asyncio.sleep(target - now)

    # The workspace must still be loaded and B must still be functional: the
    # returning client was not torn down at the original deadline.
    assert await _workspace_present(admin_service, ws_name), (
        f"workspace {ws_name} was unloaded at the original deadline while a "
        f"reconnected client was still connected"
    )
    services = await api_b.list_services()
    assert isinstance(services, list)

    # Once B leaves, the workspace must eventually unload (no leak).
    await api_b.disconnect()
    unloaded = await _wait_absent(
        admin_service, ws_name, deadline=UNLOAD_GRACE_PERIOD + 4.0
    )
    assert unloaded, (
        f"workspace {ws_name} was not unloaded after B's grace period — leak"
    )

    await admin_api.disconnect()
