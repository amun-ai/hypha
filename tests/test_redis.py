from hypha.core.store import store
from hypha.websocket import connect_to_websocket
from hypha.core import WorkspaceInfo, VisibilityEnum
import pytest
from . import SIO_PORT

pytestmark = pytest.mark.asyncio


async def test_redis_store():
    """Test the redis store."""

    # Test adding a workspace
    workspace = WorkspaceInfo(
        name="test",
        owners=[],
        visibility=VisibilityEnum.protected,
        persistent=True,
        read_only=False,
    )

    store.register_workspace(workspace)
    workspace_info = store.get_workspace(workspace.name)
    assert workspace_info.name == "test"
    assert "test" in store.list_workspace()

    rpc = store.create_rpc(workspace, "test-plugin-1")
    interface = store.get_workspace_interface("test", client_id="test-plugin-1")
    api = rpc.decode(interface, with_promise=True, target_id="test-plugin-1")
    await api.log("hello")
    assert len(await api.list_services()) == 1

    def echo(data):
        return data

    interface = {
        "name": "my service",
        "id": "test-service",
        "config": {},
        "setup": print,
        "echo": echo,
    }
    encoded = rpc.encode_service(interface, target_id="test-plugin-1")
    await api.register_service(encoded)

    rpc = store.create_rpc(workspace, "test-plugin-2")
    interface = store.get_workspace_interface("test", client_id="test-plugin-2")
    api = rpc.decode(interface, with_promise=True, target_id="test-plugin-2")

    interface = await api.get_service("test-service")
    service = rpc.decode_service(interface, target_id="test-plugin-2")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"

    # Test deleting a workspace
    store.delete_workspace(workspace.name)
    assert "test" not in store.list_workspace()


async def test_websocket_server(socketio_server):
    """Test the websocket server."""
    workspace = WorkspaceInfo(
        name="test-workspace",
        owners=[],
        visibility=VisibilityEnum.protected,
        persistent=True,
        read_only=False,
    )

    store.register_workspace(workspace)
    rpc = await connect_to_websocket(
        url=f"ws://127.0.0.1:{SIO_PORT}",
        workspace="test-workspace",
        client_id="test-plugin-1",
        token="123",
    )
    api = await rpc.get_service()
    await api.log("hello")

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

    svc = await rpc.get_service("test-service")

    assert await svc.echo("hello") == "hello"
