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


async def test_singleton_service(fastapi_server):
    """Test a singleton service."""
    async with connect_to_server(
        {"name": "test client", "server_url": SERVER_URL}
    ) as api:
        await api.register_service(
            {
                "id": "test-service",
                "name": "Test Service",
                "config": {"singleton": True},
                "description": "A test service",
                "tools": {
                    "add": lambda a, b: a + b,
                    "sub": lambda a, b: a - b,
                },
            }
        )

        # Registering the same service again should raise an error
        with pytest.raises(Exception, match=".*Failed to notify workspace manager.*"):
            await api.register_service(
                {
                    "id": "test-service",
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
