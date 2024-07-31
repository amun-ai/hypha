"""Test S3 services."""
import pytest
import requests
from hypha_rpc.websocket_client import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_card_controller(minio_server, fastapi_server):
    """Test card controller."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": WS_SERVER_URL}
    )
    card_controller = await api.get_service("public/*:card")

    source = "api.log('hello')"
    await card_controller.save(name="test.js", source=source)
    apps = await card_controller.list()
    assert find_item(apps, "name", "test.js")
    app = find_item(apps, "name", "test.js")
    response = requests.get(f"{SERVER_URL}/{app['url']}")
    assert response.ok
    assert response.text == source
    await card_controller.remove(name="test.js")
    apps = await card_controller.list()
    assert not find_item(apps, "name", "test.js")
