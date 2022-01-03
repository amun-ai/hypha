from hypha.core.store import RedisStore
from hypha.websocket import connect_to_websocket
from hypha.core import WorkspaceInfo, VisibilityEnum
import pytest
from . import SIO_PORT, find_item

pytestmark = pytest.mark.asyncio


async def test_redis_store():
    """Test the redis store."""
    store = RedisStore.get_instance()

    # Test adding a workspace
    workspace = WorkspaceInfo(
        name="test",
        owners=[],
        visibility=VisibilityEnum.protected,
        persistent=True,
        read_only=False,
    )

    await store.register_workspace(workspace)
    workspace_info = store.get_workspace(workspace.name)
    assert workspace_info.name == "test"
    assert "test" in store.list_workspace()

    rpc = store.connect_to_workspace("test", client_id="test-plugin-1")
    api = await rpc.get_remote_service("/")
    await api.log("hello", _rscope="session")
    assert len(await api.list_services()) == 2

    def echo(data):
        return data

    interface = {
        "name": "my service",
        "id": "test-service",
        "config": {},
        "setup": print,
        "echo": echo,
    }
    await rpc.register_service(interface)

    rpc = store.connect_to_workspace("test", client_id="test-plugin-2")
    service = await rpc.get_remote_service("/test-plugin-1/test-service")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"

    # Test deleting a workspace
    store.delete_workspace(workspace.name)
    assert "test" not in store.list_workspace()


async def test_websocket_server(socketio_server):
    """Test the websocket server."""
    store = RedisStore.get_instance()

    workspace = WorkspaceInfo(
        name="test-workspace",
        owners=[],
        visibility=VisibilityEnum.protected,
        persistent=True,
        read_only=False,
    )

    await store.register_workspace(workspace)
    rpc = await connect_to_websocket(
        url=f"ws://127.0.0.1:{SIO_PORT}",
        workspace="test-workspace",
        client_id="test-plugin-1",
        token="123",
    )

    ws = await rpc.get_remote_service("/")
    await ws.log("hello")

    def echo(data):
        return data

    await rpc.register_service(
        {
            "name": "my service",
            "id": "test-service",
            "config": {},
            "setup": print,
            "echo": echo,
        }
    )

    # Relative service means from the same client
    assert await rpc.get_remote_service("test-service")

    await rpc.register_service(
        {
            "name": "my service",
            "config": {},
            "setup": print,
            "echo": echo,
        }
    )

    svc = await rpc.get_remote_service("/test-plugin-1/test-service")

    assert await svc.echo("hello") == "hello"

    services = await ws.list_services()
    assert find_item(services, "id", "/test-plugin-1/")

    rpc2 = await connect_to_websocket(
        url=f"ws://127.0.0.1:{SIO_PORT}",
        workspace="test-workspace",
        client_id="test-plugin-2",
        token="123",
    )

    svc2 = await rpc2.get_remote_service("/test-plugin-1/test-service")
    assert await svc2.echo("hello") == "hello"

    rpc.disconnect()
