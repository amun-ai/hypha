"""Test the hypha workspace."""
import pytest
from hypha_rpc import connect_to_server

from . import (
    WS_SERVER_URL,
    find_item,
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

    await api.disconnect()


async def test_create_workspace_token(test_user_token):
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
            "name": "my-new-test-workspace",
            "description": "This is a test workspace",
            "visibility": "public",  # public/protected
            "persistent": True,  # keeps the workspace alive even after a server restart
        },
        overwrite=True,
    )

    print(f"Workspace created: {workspace['name']}")

    # Generate a workspace token
    token = await user.generate_token(
        {
            "workspace": "my-new-test-workspace",
            "expires_in": 60 * 60 * 24 * 30,  # 30 days
            "permission": "read_write",
        }
    )
    print(f"Token generated: {token}")
