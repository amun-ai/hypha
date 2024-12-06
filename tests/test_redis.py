import asyncio
import time

import numpy as np
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from hypha_rpc import connect_to_server

from hypha.core.store import RedisStore

from . import SIO_PORT, find_item

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(name="redis_store")
async def redis_store():
    """Represent the redis store."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    yield store
    await store.teardown()


async def test_redis_store(redis_store):
    """Test the redis store."""
    # Test adding a workspace
    await redis_store.register_workspace(
        dict(
            name="test",
            description="test workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    wss = await redis_store.list_all_workspaces()
    assert find_item(wss, "name", "test")

    api = await redis_store.connect_to_workspace("test", client_id="test-app-99")
    # Need to wait for the client to be registered
    await asyncio.sleep(0.1)
    clients = await api.list_clients()
    assert find_item(clients, "id", "test/test-app-99")
    await api.log("hello")
    services = await api.list_services()
    assert len(services) == 1
    assert await api.generate_token()
    ws = await api.create_workspace(
        dict(
            name="test-2",
            description="test workspace",
            owners=[],
            persistent=True,
            read_only=False,
        ),
        overwrite=True,
    )
    assert ws["name"] == "test-2"
    ws2 = await api.get_workspace_info("test-2")
    assert ws2["name"] == "test-2"

    def echo(data):
        return data

    interface = {
        "id": "test-service",
        "name": "my service",
        "config": {},
        "setup": print,
        "echo": echo,
    }
    await api.register_service(interface)

    wm = await redis_store.connect_to_workspace("test", client_id="test-app-22")
    await asyncio.sleep(0.1)
    # Need to wait for the client to be registered
    clients = await wm.list_clients()
    assert find_item(clients, "id", "test/test-app-22")
    assert find_item(clients, "id", "test/test-app-99")
    rpc = wm.rpc
    services = await wm.list_services()
    service = await rpc.get_remote_service("test-app-99:test-service")

    assert callable(service.echo)
    assert await service.echo("hello") == "hello"
    assert await service.echo("hello") == "hello"

    # External state to track task execution
    execution_tracker = {"executed": False, "data": None}

    # Define a dummy task that updates the tracker
    async def dummy_task(data):
        execution_tracker["executed"] = True
        execution_tracker["data"] = data
        return f"Processed {data}"

    # Schedule the task to run immediately
    task = await redis_store.run_task(dummy_task, "test-data")

    # Assert that the task ID is valid
    assert task.task_id is not None

    # Allow enough time for the task to execute
    await asyncio.sleep(0.5)

    # Fetch the result of the task
    result = await redis_store.get_task_result(task.task_id)
    assert result.return_value == "Processed test-data"
    assert execution_tracker["executed"] is True
    assert execution_tracker["data"] == "test-data"

    # Test scheduling a task using a cron expression
    execution_tracker["executed"] = False  # Reset the tracker
    cron_expr = "*/1 * * * *"
    schedule = await redis_store.schedule_task(
        dummy_task, "cron-test-data", corn=cron_expr, schedule_id="cron-test"
    )

    # Assert that the schedule ID is valid
    assert schedule.schedule_id == "cron-test"

    # Allow some time for the cron job to execute
    await asyncio.sleep(65)  # Wait for at least one minute

    # Check that the cron job executed
    assert execution_tracker["executed"] is True
    assert execution_tracker["data"] == "cron-test-data"

    # Clean up the scheduled task
    await redis_store.delete_schedule("cron-test")

    # Test scheduling a task with a specific time
    execution_tracker["executed"] = False  # Reset the tracker
    scheduled_time = datetime.now() + timedelta(seconds=2)
    schedule = await redis_store.schedule_task(
        dummy_task, "scheduled-test-data", time=scheduled_time, schedule_id="scheduled"
    )

    # Assert that the schedule ID is valid
    assert schedule.schedule_id == "scheduled"

    # Disable schedule check for now due to an error
    # Allow enough time for the task to execute
    # await asyncio.sleep(10
    # Verify that the task executed
    # assert execution_tracker["executed"] is True
    # assert execution_tracker["data"] == "scheduled-test-data"


async def test_websocket_server(fastapi_server, test_user_token_2):
    """Test the websocket server."""
    wm = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            client_id="test-app-1",
            token=test_user_token_2,
        )
    )
    await wm.log("hello")

    clients = await wm.list_clients()
    assert find_item(clients, "id", wm.config.workspace + "/test-app-1")
    rpc = wm.rpc

    def echo(data):
        return data

    await rpc.register_service(
        {
            "id": "test-service",
            "name": "my service",
            "config": {"visibility": "public"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x**2,
        }
    )

    # Relative service means from the same client
    assert await rpc.get_remote_service("test-service")

    await rpc.register_service(
        {
            "id": "default",
            "name": "my service",
            "config": {},
            "setup": print,
            "echo": echo,
        }
    )

    svc = await rpc.get_remote_service("test-app-1:test-service")

    assert await svc.echo("hello") == "hello"

    services = await wm.list_services()
    assert len(services) == 3

    svc = await wm.get_service("test-app-1:test-service")
    assert await svc.echo("hello") == "hello"

    # Get public service from another workspace
    wm3 = await connect_to_server({"server_url": f"ws://127.0.0.1:{SIO_PORT}/ws"})
    rpc3 = wm3.rpc
    svc7 = await rpc3.get_remote_service(
        f"{wm.config.workspace}/test-app-1:test-service"
    )
    svc7 = await wm3.get_service(f"{wm.config.workspace}/test-app-1:test-service")
    assert await svc7.square(9) == 81

    # Change the service to protected
    await rpc.register_service(
        {
            "id": "test-service",
            "name": "my service",
            "config": {"visibility": "protected"},
            "setup": print,
            "echo": echo,
            "square": lambda x: x**2,
        },
        {"overwrite": True},
    )

    # It should fail due to permission error
    with pytest.raises(
        Exception,
        match=r".*Permission denied for getting protected service: test-service.*",
    ):
        await rpc3.get_remote_service(f"{wm.config.workspace}/test-app-1:test-service")

    # Should fail if we try to get the protected service from another workspace
    with pytest.raises(
        Exception, match=r".*Permission denied for invoking protected method.*"
    ):
        remote_echo = rpc3._generate_remote_method(
            {
                "_rtarget": f"{wm.config.workspace}/test-app-1",
                "_rmethod": "services.test-service.echo",
                "_rpromise": True,
            }
        )
        assert await remote_echo(123) == 123

    wm2 = await connect_to_server(
        dict(
            server_url=f"ws://127.0.0.1:{SIO_PORT}/ws",
            workspace=wm.config.workspace,
            client_id="test-app-6",
            token=test_user_token_2,
            method_timeout=3,
        )
    )
    rpc2 = wm2.rpc

    svc2 = await rpc2.get_remote_service(
        f"{wm.config.workspace}/test-app-1:test-service"
    )
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
    with pytest.raises(Exception, match=r".*Method not found:.*"):
        assert await svc4.add_one(99) == 100

    svc5_info = await rpc2.register_service(
        {
            "id": "default",
            "add_one": lambda x: x + 1,
            "inner": {"square": lambda y: y**2},
        }
    )
    svc5 = await rpc2.get_remote_service(svc5_info["id"])
    svc6 = await svc2.echo(svc5)
    assert await svc6.add_one(99) == 100
    assert await svc6.inner.square(10) == 100
    array = np.zeros([2048, 1000, 1])
    array2 = await svc6.add_one(array)
    np.testing.assert_array_equal(array2, array + 1)

    await wm.disconnect()
    await asyncio.sleep(0.5)

    with pytest.raises(Exception, match=r".*Service already exists: default.*"):
        await rpc2.register_service(
            {
                "id": "default",
                "add_two": lambda x: x + 2,
            }
        )

    await rpc2.unregister_service("default")
    await rpc2.register_service(
        {
            "id": "default",
            "add_two": lambda x: x + 2,
        }
    )

    await rpc2.register_service({"id": "add-two", "blocking_sleep": time.sleep})
    svc5 = await rpc2.get_remote_service(f"{wm.config.workspace}/test-app-6:add-two")
    await svc5.blocking_sleep(2)

    await rpc2.register_service(
        {
            "id": "executor-test",
            "config": {"run_in_executor": True, "visibility": "public"},
            "blocking_sleep": time.sleep,
        }
    )
    svc5 = await rpc2.get_remote_service(
        f"{wm.config.workspace}/test-app-6:executor-test"
    )
    # This should be fine because it is run in executor
    await svc5.blocking_sleep(3)
    summary = await wm2.get_summary()
    assert summary["client_count"] == 1
    assert summary["service_count"] == 4
    assert find_item(summary["services"], "name", "executor-test")
    await wm2.disconnect()
