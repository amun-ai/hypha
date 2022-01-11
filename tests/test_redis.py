import time

import numpy as np
import pytest

from hypha.core.store import RedisStore
from hypha.websocket_client import connect_to_server

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
    manager = await redis_store.get_workspace_manager("test")
    workspace_info = await manager.get_workspace_info()
    assert workspace_info.name == "test"
    assert "test" in await redis_store.list_workspace()

    wm = await redis_store.connect_to_workspace("test", client_id="test-plugin-1")
    rpc = wm.rpc
    api = await rpc.get_remote_service("workspace-manager:default")
    await api.log("hello")
    services = await api.list_services()
    assert len(services) == 4 # 2 services: built-in and default
    assert await api.generate_token()
    ws = await api.create_workspace(
        dict(
            name="test-2",
            owners=[],
            visibility="protected",
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    assert ws.config["workspace"] == "test-2"

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

    wm = await redis_store.connect_to_workspace("test", client_id="test-plugin-2")
    rpc = wm.rpc
    service = await rpc.get_remote_service("test-plugin-1:test-service")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"

    # Test deleting a workspace
    await redis_store.delete_workspace("test")
    assert "test" not in await redis_store.list_workspace()


async def test_websocket_server(
    event_loop, socketio_server, redis_store, test_user_token
):
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
    wm = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            workspace="test-workspace",
            client_id="test-plugin-1",
            token=test_user_token,
        )
    )
    await wm.log("hello")
    rpc = wm.rpc

    def echo(data):
        return data

    await rpc.register_service(
        {
            "name": "my service",
            "id": "test-service",
            "config": {"visibility": "public"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x ** 2,
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

    services = await wm.list_services()
    assert find_item(services, "id", "workspace-manager:default")

    svc = await wm.get_service("test-plugin-1:test-service")
    assert await svc.echo("hello") == "hello"

    # Get public service from another workspace
    wm3 = await connect_to_server({"server_url": f"ws://127.0.0.1:{SIO_PORT}/ws"})
    rpc3 = wm3.rpc
    svc7 = await rpc3.get_remote_service("test-workspace/test-plugin-1:test-service")
    svc7 = await wm3.get_service("test-workspace/test-plugin-1:test-service")
    assert await svc7.square(9) == 81

    # Change the service to protected
    await rpc.register_service(
        {
            "name": "my service",
            "id": "test-service",
            "config": {"visibility": "protected"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x ** 2,
        },
        overwrite=True,
    )

    # It should fail due to permission error
    with pytest.raises(
        Exception, match=r".*Permission denied for service: test-service.*"
    ):
        await rpc3.get_remote_service("test-workspace/test-plugin-1:test-service")

    # Should fail if we try to bypass the get_remote_service call
    remote_echo = rpc3._generate_remote_method(
        {
            "_rtarget": "test-workspace/test-plugin-1",
            "_rmethod": "services.test-service.echo",
            "_rpromise": True,
        }
    )
    with pytest.raises(Exception, match=r".*Permission denied for protected method.*"):
        await remote_echo(123) == 123

    wm2 = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            workspace="test-workspace",
            client_id="test-plugin-2",
            token=test_user_token,
            method_timeout=3,
        )
    )
    rpc2 = wm2.rpc

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
    with pytest.raises(
        Exception, match=r".*Method not found: test-workspace/test-plugin-2.*"
    ):
        assert await svc4.add_one(99) == 100

    svc5_info = await rpc2.register_service(
        {"add_one": lambda x: x + 1, "inner": {"square": lambda y: y ** 2}}
    )
    svc5 = await rpc2.get_remote_service(svc5_info["id"])
    svc6 = await svc2.echo(svc5)
    assert await svc6.add_one(99) == 100
    assert await svc6.inner.square(10) == 100
    array = np.zeros([2048, 1000, 1])
    array2 = await svc6.add_one(array)
    np.testing.assert_array_equal(array2, array + 1)

    # Test large data transfer
    # array = np.zeros([2048, 2048, 4])
    # array2 = await svc6.add_one(array)
    # np.testing.assert_array_equal(array2, array + 1)

    with pytest.raises(Exception, match=r".*Service already exists: default.*"):
        await rpc2.register_service(
            {
                "add_two": lambda x: x + 2,
            }
        )

    await rpc2.unregister_service("default")
    await rpc2.register_service(
        {
            "add_two": lambda x: x + 2,
        }
    )

    await rpc2.register_service({"id": "add-two", "blocking_sleep": time.sleep})
    svc5 = await rpc2.get_remote_service("test-plugin-2:add-two")
    # This will fail because the service is blocking
    with pytest.raises(Exception, match=r".*Method call time out:.*"):
        await svc5.blocking_sleep(4)

    await svc5.blocking_sleep(0.5)

    await rpc2.register_service(
        {
            "id": "executor-test",
            "config": {"run_in_executor": True, "visibility": "public"},
            "blocking_sleep": time.sleep,
        }
    )
    svc5 = await rpc2.get_remote_service("test-plugin-2:executor-test")
    # This should be fine because it is run in executor
    await svc5.blocking_sleep(3)
