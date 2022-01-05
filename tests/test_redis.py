from hypha.core.store import RedisStore
from hypha.websocket import connect_to_websocket
import pytest
import time
from . import SIO_PORT, find_item

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="session", autouse=True)
def redis_store():
    yield RedisStore.get_instance()


async def test_redis_store(event_loop, redis_store):
    """Test the redis store."""
    # Test adding a workspace
    await redis_store.clear_workspace("test")
    await redis_store.register_workspace(
        dict(
            name="test",
            owners=[],
            visibility="protected",
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    workspace_info = await redis_store.get_workspace("test")
    assert workspace_info.name == "test"
    assert "test" in await redis_store.list_workspace()

    rpc = await redis_store.connect_to_workspace("test", client_id="test-plugin-1")
    api = await rpc.get_remote_service("workspace-manager:default")
    await api.log("hello")
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

    rpc = await redis_store.connect_to_workspace("test", client_id="test-plugin-2")
    service = await rpc.get_remote_service("test-plugin-1:test-service")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"

    # Test deleting a workspace
    await redis_store.delete_workspace("test")
    assert "test" not in await redis_store.list_workspace()


async def test_websocket_server(event_loop, socketio_server, redis_store):
    """Test the websocket server."""
    await redis_store.clear_workspace("test-workspace")
    await redis_store.register_workspace(
        dict(
            name="test-workspace",
            owners=[],
            visibility="protected",
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    rpc = await connect_to_websocket(
        url=f"ws://127.0.0.1:{SIO_PORT}",
        workspace="test-workspace",
        client_id="test-plugin-1",
        token="123",
    )

    ws = await rpc.get_remote_service("workspace-manager:default")
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

    svc = await rpc.get_remote_service("test-plugin-1:test-service")

    assert await svc.echo("hello") == "hello"

    services = await ws.list_services()
    assert find_item(services, "uri", "test-workspace/workspace-manager:default")

    rpc2 = await connect_to_websocket(
        url=f"ws://127.0.0.1:{SIO_PORT}",
        workspace="test-workspace",
        client_id="test-plugin-2",
        token="123",
        method_timeout=2,  # set to very short timeout
    )

    svc2 = await rpc2.get_remote_service("test-plugin-1:test-service")
    assert await svc2.echo("hello") == "hello"
    assert len(rpc2._object_store) == 1
    svc3 = await svc2.echo(svc2)
    assert len(rpc2._object_store) == 1
    assert await svc3.echo("hello") == "hello"

    svc4 = await svc2.echo(
        {
            "add_one": lambda x: x + 1,
        }
    )
    assert len(rpc2._object_store) > 0

    # It should fail because add_one is not a service and will be destroyed after the session
    with pytest.raises(Exception, match=r".*Method not found: test-plugin-2:.*"):
        assert await svc4.add_one(99) == 100

    svc5 = await rpc2.register_service(
        {
            "add_one": lambda x: x + 1,
        }
    )
    svc6 = await svc2.echo(svc5)
    assert await svc6.add_one(99) == 100
    with pytest.raises(Exception, match=r".*Service already exists: default.*"):
        await rpc2.register_service(
            {
                "add_two": lambda x: x + 2,
            }
        )

    await rpc2.register_service({"id": "add-two", "blocking_sleep": time.sleep})
    svc5 = await rpc2.get_remote_service("test-plugin-2:add-two")
    with pytest.raises(Exception, match=r".*Method call time out:.*"):
        await svc5.blocking_sleep(3)

    await svc5.blocking_sleep(0.5)

    rpc.disconnect()
