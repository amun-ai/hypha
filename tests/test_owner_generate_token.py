"""Integration test for amun-ai/hypha#958.

A workspace owner who is NOT a global admin must be able to ``generate_token``
(and therefore start a server-app) in their OWN workspace, even when their
token carries a broad ``*`` scope at a permission *lower* than admin.

The bug: ``UserInfo.get_permission`` short-circuited on the ``*`` wildcard and
never consulted the stronger per-workspace admin grant that
``auth.get_user_info`` installs for the caller's own workspace. As a result
``generate_token`` (called by ``apps.start``) raised
``PermissionError: Only admin can generate token``.
"""

import pytest
from hypha_rpc import connect_to_server

from . import SERVER_URL_SQLITE

# All test coroutines will be treated as marked with pytest.mark.asyncio
pytestmark = pytest.mark.asyncio


async def test_owner_with_wildcard_scope_can_generate_token(
    fastapi_server_sqlite, wildcard_owner_user_token
):
    """Owner of ``ws-user-wildcard-owner`` can mint a token in their own ws.

    Reproduces the exact failing call in ``apps.start``:
    ``await ws.generate_token({"client_id": ...})``. The caller is a non-admin
    user whose token holds only ``*#rw`` (below admin). Before the fix this
    raised ``Only admin can generate token``.
    """
    api = await connect_to_server(
        {
            "name": "wildcard owner",
            "server_url": SERVER_URL_SQLITE,
            "token": wildcard_owner_user_token,
        }
    )
    try:
        # Connected into the user's own workspace.
        assert api.config.workspace == "ws-user-wildcard-owner"

        # This is the call that apps.start performs internally and that
        # previously failed with "Only admin can generate token".
        token = await api.generate_token({"client_id": "_rapp_test__rlb"})
        assert isinstance(token, str) and token

        # An explicit admin-permission token for the owned workspace must
        # also be allowed (owner elevation must reach admin).
        admin_token = await api.generate_token(
            {"workspace": "ws-user-wildcard-owner", "permission": "admin"}
        )
        assert isinstance(admin_token, str) and admin_token
    finally:
        await api.disconnect()
