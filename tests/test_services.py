"""Test services."""
import pytest
import requests
import asyncio
import httpx

from imjoy_rpc.hypha import login
from . import (
    SERVER_URL,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_login(fastapi_server):
    """Test login to the server."""

    TOKEN = "sf31df234"

    async def callback(context):
        print(f"By passing login: {context['login_url']}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(context["login_url"] + "?key=" + context["key"])
            assert resp.status_code == 200
            assert "Hypha Login" in resp.text
            assert "{{ report_url }}" not in resp.text
            assert context["report_url"] in resp.text
            resp = await client.get(
                context["report_url"] + "?key=" + context["key"] + "&token=" + TOKEN
            )
            assert resp.status_code == 200

    token = await login(
        {
            "server_url": SERVER_URL,
            "login_callback": callback,
            "login_timeout": 3,
        }
    )
    assert token == TOKEN
