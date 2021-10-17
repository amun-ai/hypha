"""Test S3 services."""
import pytest
import requests
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_rdf_controller(minio_server, socketio_server):
    """Test rdf controller."""
    api = await connect_to_server(
        {"name": "test deploy client", "server_url": SIO_SERVER_URL}
    )
    s3controller = await api.get_service("rdf")

    source = "api.log('hello')"
    await s3controller.save(name="test.js", source=source)
    apps = await s3controller.list()
    assert find_item(apps, "name", "test.js")
    app = find_item(apps, "name", "test.js")
    response = requests.get(f"{SIO_SERVER_URL}/{app['url']}")
    assert response.ok
    assert response.text == source
    await s3controller.remove(name="test.js")
    apps = await s3controller.list()
    assert not find_item(apps, "name", "test.js")
