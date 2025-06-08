"""Test services."""

import pytest
import httpx

from hypha_rpc import login, connect_to_server
from . import (
    SERVER_URL,
)

from hypha_rpc.utils.schema import schema_function
from pydantic import Field

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_singleton_service(fastapi_server, test_user_token):
    """Test a singleton service."""
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "token": test_user_token}
    ) as api:
        await api.register_service(
            {
                "id": "test-service-1",
                "name": "Test Service",
                "config": {"singleton": True},
                "description": "A test service",
                "tools": {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                },
            }
        )

        # Registering the same service from another client should raise an error
        async with connect_to_server(
            {
                "name": "test client 2",
                "server_url": SERVER_URL,
                "token": test_user_token,
            }
        ) as api:
            with pytest.raises(
                Exception, match=".*A singleton service with the same name.*"
            ):
                await api.register_service(
                    {
                        "id": "test-service-1",
                        "name": "Test Service",
                        "config": {"singleton": True},
                        "description": "A test service",
                        "tools": {
                            "add": lambda a, b: a + b,
                            "sub": lambda a, b: a - b,
                        },
                    },
                    {"overwrite": True},
                )


async def test_typed_service(fastapi_server):
    """Test a typed service."""
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL}
    ) as api:

        @schema_function
        async def add(
            a: int = Field(..., description="first arg"),
            b: int = Field(2, description="second arg"),
        ) -> int:
            return a + b

        service = {
            "id": "test-service",
            "name": "Test Service",
            "description": "A test service",
            "tools": {"add": add},
        }
        service_schema = await api.get_service_schema(service)

        await api.register_service_type(
            {
                "id": "test-service-type",
                "name": "Test Service Type",
                "definition": service_schema,
            }
        )

        svc_type = await api.get_service_type("test-service-type")
        assert svc_type["id"] == f"{api.config.workspace}/test-service-type"

        service["type"] = "test-service-type"
        svc_info = await api.register_service(service, {"check_type": True})
        assert svc_info["id"].endswith(":test-service")
        assert svc_info["type"] == "test-service-type"

        def add2(a: int, b: int) -> int:
            return a + b

        svc_info2 = await api.register_service(
            {
                "id": "test-service2",
                "name": "Test Service 2",
                "description": "A test service 2",
                "type": api.config.workspace + "/test-service-type",
                "tools": {"add": add2},
            },
            {"check_type": True},
        )
        assert svc_info2["id"].endswith(":test-service2")
        assert svc_info2["service_schema"]


async def test_login_service(fastapi_server):
    """Test login to the server."""
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL}
    ) as api:
        svc = await api.get_service("public/hypha-login")
        assert svc and callable(svc.start)
        info = await svc.start()
        key = info["key"]
        data = await svc.check(key, timeout=-1)
        assert data is None
        await svc.report(key, "test")
        token = await svc.check(key, timeout=1)
        assert token == "test"

        # with user info
        info = await svc.start()
        key = info["key"]
        data = await svc.check(key, timeout=-1)
        assert data is None
        await svc.report(key, "test", email="abc@example.com")
        user_profile = await svc.check(key, timeout=1, profile=True)
        assert user_profile["token"] == "test"
        assert user_profile["email"] == "abc@example.com"


async def test_cleanup_workspace(fastapi_server, root_user_token):
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL, "token": root_user_token}
    ) as api:
        admin = await api.get_service("admin-utils")
        servers = await admin.list_servers()
        assert len(servers) == 1
        summary = await api.cleanup("public")
        assert "removed_clients" not in summary
        assert len(summary) == 0


async def test_login(fastapi_server):
    """Test login to the server."""
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL}
    ) as api:
        svc = await api.get_service("public/hypha-login")
        assert svc and callable(svc.start)

        TOKEN = "sf31df234"

        async def callback(context):
            print(f"By passing login: {context['login_url']}")
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(context["login_url"])
                assert resp.status_code == 200, resp.text
                assert "Hypha Account" in resp.text
                assert "{{ report_url }}" not in resp.text
                resp = await client.get(
                    context["report_url"] + "?key=" + context["key"] + "&token=" + TOKEN
                )
                assert resp.status_code == 200, resp.text

        token = await login(
            {
                "server_url": SERVER_URL,
                "login_callback": callback,
                "login_timeout": 3,
            }
        )
        assert token == TOKEN

        user_profile = await login(
            {
                "server_url": SERVER_URL,
                "login_callback": callback,
                "login_timeout": 3,
                "profile": True,
            }
        )
        assert user_profile["token"] == TOKEN


async def test_register_service_type(fastapi_server):
    """Test registering a new service type."""
    api = await connect_to_server({"name": "my app", "server_url": SERVER_URL})

    service_type_info = {
        "id": "test-service-type",
        "name": "Test Service Type",
        "definition": {"type": "object", "properties": {"name": {"type": "string"}}},
        "description": "A test service type",
    }
    registered_service_type = await api.register_service_type(service_type_info)
    assert registered_service_type["id"] == service_type_info["id"]
    assert registered_service_type["name"] == service_type_info["name"]
    assert registered_service_type["definition"] == service_type_info["definition"]
    assert registered_service_type["description"] == service_type_info["description"]


async def test_get_service_type(fastapi_server):
    """Test getting a service type by ID."""
    api = await connect_to_server({"name": "my app", "server_url": SERVER_URL})

    service_type_info = {
        "id": "test-service-type",
        "name": "Test Service Type",
        "definition": {"type": "object", "properties": {"name": {"type": "string"}}},
        "description": "A test service type",
    }

    await api.register_service_type(service_type_info)
    retrieved_service_type = await api.get_service_type("test-service-type")

    assert (
        retrieved_service_type["id"]
        == api.config.workspace + "/" + service_type_info["id"]
    )
    assert retrieved_service_type["name"] == service_type_info["name"]
    assert retrieved_service_type["definition"] == service_type_info["definition"]
    assert retrieved_service_type["description"] == service_type_info["description"]


async def test_list_service_types(fastapi_server):
    """Test listing all service types in the workspace."""
    api = await connect_to_server({"name": "my app", "server_url": SERVER_URL})

    service_type_info_1 = {
        "id": "test-service-type-1",
        "name": "Test Service Type 1",
        "definition": {"type": "object", "properties": {"name": {"type": "string"}}},
        "description": "First test service type",
    }

    service_type_info_2 = {
        "id": "test-service-type-2",
        "name": "Test Service Type 2",
        "definition": {"type": "object", "properties": {"name": {"type": "string"}}},
        "description": "Second test service type",
    }

    await api.register_service_type(service_type_info_1)
    await api.register_service_type(service_type_info_2)

    service_types = await api.list_service_types()

    assert len(service_types) == 2
    assert any(st["id"] == service_type_info_1["id"] for st in service_types)
    assert any(st["id"] == service_type_info_2["id"] for st in service_types)


async def test_simple_select_service_syntax(fastapi_server, test_user_token):
    """Test the simple select service syntax: select:criteria:function."""
    import asyncio
    import time

    # Create multiple service instances with different characteristics
    class SimpleService:
        def __init__(self, instance_id: str, load_value: float = 0.0):
            self.instance_id = instance_id
            self.load_value = load_value
            self.cpu_usage = load_value * 100  # Convert to percentage
            self.priority = 10 - load_value * 10  # Higher load = lower priority

        async def get_load(self):
            """Return current load as a simple number."""
            return self.load_value

        async def get_cpu_usage(self):
            """Return CPU usage percentage."""
            return self.cpu_usage

        async def get_priority(self):
            """Return priority score."""
            return self.priority

        async def process_task(self, task_data: str):
            """Process a task."""
            return f"Task '{task_data}' completed by service {self.instance_id}"

    # Create service instances with different load values
    service_a = SimpleService("A", 0.1)  # Low load
    service_b = SimpleService("B", 0.5)  # Medium load
    service_c = SimpleService("C", 0.8)  # High load

    # Register service instances
    async with connect_to_server(
        {
            "name": "simple-test-client-a",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    ) as api_a:
        await api_a.register_service(
            {
                "id": "simple-select-service",
                "name": "Simple Service Instance A",
                "get_load": service_a.get_load,
                "get_cpu_usage": service_a.get_cpu_usage,
                "get_priority": service_a.get_priority,
                "process_task": service_a.process_task,
            }
        )

        async with connect_to_server(
            {
                "name": "simple-test-client-b",
                "server_url": SERVER_URL,
                "token": test_user_token,
            }
        ) as api_b:
            await api_b.register_service(
                {
                    "id": "simple-select-service",
                    "name": "Simple Service Instance B",
                    "get_load": service_b.get_load,
                    "get_cpu_usage": service_b.get_cpu_usage,
                    "get_priority": service_b.get_priority,
                    "process_task": service_b.process_task,
                }
            )

            async with connect_to_server(
                {
                    "name": "simple-test-client-c",
                    "server_url": SERVER_URL,
                    "token": test_user_token,
                }
            ) as api_c:
                await api_c.register_service(
                    {
                        "id": "simple-select-service",
                        "name": "Simple Service Instance C",
                        "get_load": service_c.get_load,
                        "get_cpu_usage": service_c.get_cpu_usage,
                        "get_priority": service_c.get_priority,
                        "process_task": service_c.process_task,
                    }
                )

                # Wait for services to register
                await asyncio.sleep(0.5)

                # Create a client to test the new syntax
                async with connect_to_server(
                    {
                        "name": "simple-select-client",
                        "server_url": SERVER_URL,
                        "token": test_user_token,
                    }
                ) as client_api:

                    print("Testing simple select syntax...")

                    # Test 1: select:min:get_load (should select service A - lowest load)
                    print("\n1. Testing select:min:get_load")
                    for i in range(5):
                        service = await client_api.get_service(
                            "simple-select-service", mode="select:min:get_load"
                        )
                        result = await service.process_task(f"min-load-task-{i}")
                        print(f"  {result}")
                        # Should consistently select service A (lowest load)
                        assert "service A" in result

                    # Test 2: select:max:get_load (should select service C - highest load)
                    print("\n2. Testing select:max:get_load")
                    for i in range(3):
                        service = await client_api.get_service(
                            "simple-select-service", mode="select:max:get_load"
                        )
                        result = await service.process_task(f"max-load-task-{i}")
                        print(f"  {result}")
                        # Should consistently select service C (highest load)
                        assert "service C" in result

                    # Test 3: select:max:get_priority (should select service A - highest priority)
                    print("\n3. Testing select:max:get_priority")
                    for i in range(3):
                        service = await client_api.get_service(
                            "simple-select-service", mode="select:max:get_priority"
                        )
                        result = await service.process_task(f"max-priority-task-{i}")
                        print(f"  {result}")
                        # Should consistently select service A (highest priority)
                        assert "service A" in result

                    # Test 4: select:min:get_cpu_usage (should select service A - lowest CPU)
                    print("\n4. Testing select:min:get_cpu_usage")
                    service = await client_api.get_service(
                        "simple-select-service", mode="select:min:get_cpu_usage"
                    )
                    result = await service.process_task("min-cpu-task")
                    print(f"  {result}")
                    assert "service A" in result

                    # Test 5: first_success criteria
                    print("\n5. Testing select:first_success:get_load")
                    service = await client_api.get_service(
                        "simple-select-service", mode="select:first_success:get_load"
                    )
                    result = await service.process_task("first-success-task")
                    print(f"  {result}")
                    # Should work with any service that responds first
                    assert "service" in result
