"""Test workspace loader."""

import base64
import json

import pytest
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    find_item,
)


def _decode_jwt_claims(token):
    """Decode JWT payload without signature verification (tests only)."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_loading_workspace(fastapi_server, test_user_token):
    """Test workspace loader."""
    async with connect_to_server(
        {
            "name": "my app",
            "server_url": WS_SERVER_URL,
            "client_id": "my-app",
            "token": test_user_token,
        }
    ) as api:
        ws = await api.create_workspace(
            dict(
                name="test-2",
                description="test workspace",
                owners=[],
                persistent=True,
                read_only=False,
            ),
        )
        token = await api.generate_token(
            {"expires_in": 3600000, "workspace": ws.id, "permission": "read_write"}
        )
        workspaces = await api.list_workspaces()
        assert find_item(workspaces, "id", ws.id)

    async with connect_to_server(
        {
            "name": "my app",
            "server_url": WS_SERVER_URL,
            "client_id": "my-app",
            "workspace": ws.id,
            "token": token,
        }
    ) as api:
        assert api.config.workspace == ws.id

        workspaces = await api.list_workspaces()
        assert find_item(workspaces, "name", ws.id)


async def test_delete_workspace(fastapi_server, test_user_token):
    """Test workspace loader."""
    async with connect_to_server(
        {
            "name": "my app",
            "server_url": WS_SERVER_URL,
            "client_id": "my-app",
            "token": test_user_token,
        }
    ) as api:
        ws = await api.create_workspace(
            dict(
                name="test-3",
                description="test workspace",
                owners=[],
                persistent=True,
                read_only=False,
            ),
        )

        # check if workspace exists
        workspaces = await api.list_workspaces()
        assert find_item(workspaces, "name", ws.id)

        await api.delete_workspace(ws.id)


        # workspace will only be deleted after some time
        # workspaces = await api.list_workspaces()
        # assert not find_item(workspaces, "name", ws.id)


async def test_generate_token_email_override_admin(
    fastapi_server, admin_role_user_token
):
    """Admins may mint user-attributed tokens by supplying ``email``.

    This unlocks downstream services (e.g. quota-enforcing LLM gateways)
    that require an email claim for attribution but are reached through
    operator-minted tokens that would otherwise carry a null email.
    """
    async with connect_to_server(
        {
            "name": "admin-app",
            "server_url": WS_SERVER_URL,
            "client_id": "admin-app",
            "token": admin_role_user_token,
        }
    ) as api:
        minted = await api.generate_token(
            {
                "workspace": "ws-user-admin-role-user",
                "expires_in": 3600,
                "email": "attributed-user@example.com",
            }
        )
        claims = _decode_jwt_claims(minted)
        assert claims["https://amun.ai/email"] == "attributed-user@example.com"


async def test_generate_token_email_override_non_admin_rejected(
    fastapi_server, test_user_token
):
    """Non-admin callers must NOT be able to forge ``email`` on tokens."""
    async with connect_to_server(
        {
            "name": "regular-app",
            "server_url": WS_SERVER_URL,
            "client_id": "regular-app",
            "token": test_user_token,
        }
    ) as api:
        with pytest.raises(Exception) as exc_info:
            await api.generate_token(
                {
                    "workspace": "ws-user-user-1",
                    "expires_in": 3600,
                    "email": "forged@example.com",
                }
            )
        # Error may surface as PermissionError via RPC — check message
        assert "admin" in str(exc_info.value).lower()
