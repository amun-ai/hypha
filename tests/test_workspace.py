"""Test the hypha workspace."""

import pytest
from hypha_rpc import connect_to_server
import time
import asyncio

from . import (
    WS_SERVER_URL,
    find_item,
    wait_for_workspace_ready,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_workspace(fastapi_server, test_user_token):
    """Test workspace."""
    api = await connect_to_server(
        {
            "client_id": "my-app",
            "name": "my app",
            "server_url": WS_SERVER_URL,
            "method_timeout": 20,
            "token": test_user_token,
        }
    )
    assert api.config.workspace is not None
    assert api.config.public_base_url.startswith("http")
    await api.log("hi")
    token = await api.generate_token()
    assert token is not None

    public_svc = await api.list_services("public")
    assert len(public_svc) > 0

    current_svcs = await api.list_services(api.config.workspace)
    assert len(current_svcs) >= 1

    current_svcs = await api.list_services()
    assert len(current_svcs) >= 1

    login_svc = find_item(public_svc, "name", "Hypha Login")
    assert len(login_svc["description"]) > 0

    s3_svc = await api.list_services({"workspace": "public", "id": "s3-storage"})
    assert len(s3_svc) == 1

    # workspace=* means search both the public and the current workspace
    s3_svc = await api.list_services({"workspace": "*", "id": "s3-storage"})
    assert len(s3_svc) == 1

    ws = await api.create_workspace(
        {
            "name": "my-new-test-workspace",
            "description": "This is a test workspace",
            "owners": ["user1@imjoy.io", "user2@imjoy.io"],
        },
        overwrite=True,
    )
    assert ws["name"] == "my-new-test-workspace"

    def test(context=None):
        return context

    # we should not get it because api is in another workspace
    ss2 = await api.list_services({"type": "test-type"})
    assert len(ss2) == 0

    # let's generate a token for the my-new-test-workspace
    token = await api.generate_token({"workspace": "my-new-test-workspace"})

    # now if we connect directly to the workspace
    # we should be able to get the my-new-test-workspace services
    api2 = await connect_to_server(
        {
            "client_id": "my-app-2",
            "name": "my app 2",
            "workspace": "my-new-test-workspace",
            "server_url": WS_SERVER_URL,
            "token": token,
            "method_timeout": 20,
        }
    )
    await api2.log("hi")

    service_info = await api2.register_service(
        {
            "id": "test_service_2",
            "name": "test_service_2",
            "type": "test-type",
            "config": {"require_context": True, "visibility": "public"},
            "test": test,
        }
    )
    service = await api2.get_service(service_info.id)
    context = await service.test()
    assert "from" in context and "to" in context and "user" in context

    svcs = await api2.list_services(service_info.config.workspace)
    assert find_item(svcs, "name", "test_service_2")

    svcs = await api2.list_services({"type": "test-type"})
    assert find_item(svcs, "name", "test_service_2")

    assert api2.config["workspace"] == "my-new-test-workspace"
    await api2.export({"foo": "bar"})
    # services = api2.rpc.get_all_local_services()
    clients = await api2.list_clients()
    assert find_item(clients, "id", "my-new-test-workspace/my-app-2")

    ss3 = await api2.list_services({"type": "test-type"})
    assert len(ss3) == 1

    app = await api2.get_app("my-app-2")
    assert app.foo == "bar"

    await api2.export({"foo2": "bar2"})
    app = await api2.get_app("my-app-2")
    assert not hasattr(app, "foo")
    assert app.foo2 == "bar2"

    plugins = await api2.list_apps()
    assert find_item(plugins, "id", "my-new-test-workspace/my-app-2:default")

    try:
        await api.get_app("my-new-test-workspace/my-app-2")
    except Exception as e:
        assert "Permission denied" in str(e)

    ws2 = await api.get_workspace_info("my-new-test-workspace")
    assert ws.id == ws2.id

    # Skip disconnect to keep the connection alive so other tests won't be affected
    # await api.disconnect()


async def test_create_workspace_token(fastapi_server, test_user_token):
    server_url = WS_SERVER_URL
    # Connect to the Hypha server
    user = await connect_to_server(
        {
            "server_url": server_url,
            "token": test_user_token,
        }
    )

    workspace = await user.create_workspace(
        {
            "id": "my-new-test-workspace-3",
            "name": "my-new-test-workspace-3",
            "description": "This is a test workspace",
            "visibility": "public",  # public/protected
            "persistent": True,  # keeps the workspace alive even after a server restart
        },
        overwrite=True,
    )

    # Wait for the workspace to be ready
    status = await wait_for_workspace_ready(user, timeout=5)
    assert status["status"] == "ready"
    assert "errors" not in status or status["errors"] is None

    print(f"Workspace created: {workspace['name']}")

    # Generate a workspace token
    token = await user.generate_token(
        {
            "workspace": "my-new-test-workspace-3",
            "expires_in": 60 * 60 * 24 * 30,  # 30 days
            "permission": "read_write",
        }
    )
    print(f"Token generated: {token}")


async def test_workspace_ready(fastapi_server):
    """Test workspace."""
    server_url = WS_SERVER_URL
    server = await connect_to_server(
        {
            "server_url": server_url,
        }
    )

    status = await wait_for_workspace_ready(server, timeout=1)
    assert status["status"] == "ready"
    assert "errors" not in status or status["errors"] is None


async def test_workspace_env_variables(fastapi_server, test_user_token):
    """Test workspace environment variables with proper permission controls."""
    server_url = WS_SERVER_URL
    
    # Connect with authorized user
    authorized_api = await connect_to_server(
        {
            "client_id": "env-test-client",
            "name": "Environment Test Client",
            "server_url": server_url,
            "method_timeout": 20,
            "token": test_user_token,
        }
    )
    
    # Test setting environment variables
    await authorized_api.set_env("TEST_VAR1", "value1")
    await authorized_api.set_env("TEST_VAR2", "value2")
    await authorized_api.set_env("DATABASE_URL", "postgres://localhost:5432/testdb")
    
    # Test getting individual environment variables
    var1 = await authorized_api.get_env("TEST_VAR1")
    assert var1 == "value1"
    
    var2 = await authorized_api.get_env("TEST_VAR2") 
    assert var2 == "value2"
    
    db_url = await authorized_api.get_env("DATABASE_URL")
    assert db_url == "postgres://localhost:5432/testdb"
    
    # Test getting all environment variables
    all_vars = await authorized_api.get_env()
    assert isinstance(all_vars, dict)
    assert all_vars["TEST_VAR1"] == "value1"
    assert all_vars["TEST_VAR2"] == "value2"
    assert all_vars["DATABASE_URL"] == "postgres://localhost:5432/testdb"
    assert len(all_vars) == 3
    
    # Test getting non-existent variable
    try:
        await authorized_api.get_env("NON_EXISTENT_VAR")
        assert False, "Should have raised KeyError"
    except Exception as e:
        assert "not found" in str(e)
    
    # Test overwriting existing variable
    await authorized_api.set_env("TEST_VAR1", "new_value1")
    updated_var1 = await authorized_api.get_env("TEST_VAR1")
    assert updated_var1 == "new_value1"
    
    # Test with unauthorized client (different workspace)
    # Create a second workspace with different user
    second_workspace_user = await connect_to_server(
        {
            "client_id": "second-workspace-user",
            "server_url": server_url,
            "method_timeout": 10,
            "token": test_user_token,  # Same user but will create different workspace
        }
    )
    
    second_workspace = await second_workspace_user.create_workspace(
        {
            "id": "test-env-workspace-2",
            "name": "Test Environment Workspace 2", 
            "description": "Second test workspace for env variables",
            "visibility": "protected",
            "persistent": False,
        },
        overwrite=True,
    )
    
    # Anonymous user should not be able to connect to protected workspace
    connection_failed = False
    try:
        unauthorized_api = await connect_to_server(
            {
                "client_id": "unauthorized-client",
                "server_url": server_url, 
                "workspace": "test-env-workspace-2",  # Try to connect to protected workspace
                "method_timeout": 5,
            }
        )
        # If connection succeeds, then try operations (they should still fail)
        # This should fail due to lack of read_write permission on protected workspace
        await unauthorized_api.set_env("UNAUTHORIZED_VAR", "should_fail")
        assert False, "Should have raised PermissionError"
    except Exception as e:
        # Connection or operation should fail
        assert "Permission denied" in str(e) or "Failed to connect" in str(e), f"Expected permission error, got: {e}"
        connection_failed = True
    
    # Test env variables persist across different clients in same workspace
    workspace_token = await authorized_api.generate_token()
    
    second_client = await connect_to_server(
        {
            "client_id": "env-test-client-2",
            "name": "Second Environment Test Client", 
            "server_url": server_url,
            "token": workspace_token,
            "method_timeout": 10,
        }
    )
    
    # Should be able to access env vars set by first client
    shared_var = await second_client.get_env("TEST_VAR1")
    assert shared_var == "new_value1"
    
    # Should be able to set new env vars
    await second_client.set_env("SHARED_VAR", "shared_value")
    
    # First client should see the new variable
    shared_from_first = await authorized_api.get_env("SHARED_VAR")
    assert shared_from_first == "shared_value"
    
    # Verify all variables are accessible from both clients
    all_vars_second = await second_client.get_env()
    all_vars_first = await authorized_api.get_env()
    
    assert all_vars_first == all_vars_second
    assert "SHARED_VAR" in all_vars_first
    assert all_vars_first["SHARED_VAR"] == "shared_value"
