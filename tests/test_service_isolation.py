"""Test suite for service isolation and cross-workspace access.

This test verifies that protected services cannot be accessed across workspace boundaries.
"""

import pytest
import uuid
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL

pytestmark = pytest.mark.asyncio


class TestServiceIsolation:
    """Test service registration and discovery isolation."""

    async def test_cannot_access_protected_service_in_other_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that protected services in other workspaces are not accessible."""
        # Create two workspaces
        api1 = await connect_to_server({
            "client_id": "service-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"service-victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace with protected service",
        })
        ws2 = ws2_info["id"]

        # Connect to ws2 and register a protected service
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "service-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # Register a protected service in ws2
        async def secret_function(data):
            return f"SECRET: {data}"

        await api2.register_service({
            "id": "secret-service",
            "name": "Secret Service",
            "config": {"visibility": "protected"},  # Default: only accessible within workspace
            "process": secret_function,
        })

        # Now try to access this service from ws1
        ws1_token = await api1.generate_token({"workspace": ws1})
        api_attacker = await connect_to_server({
            "client_id": "attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": ws1_token,
        })

        # Attempt 1: Try to get service using full service ID
        with pytest.raises(Exception) as exc_info:
            await api_attacker.get_service(f"{ws2}/service-creator:secret-service")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        # Attempt 2: Try to get service info (should also fail)
        ws_manager = await api_attacker.get_service("~")
        with pytest.raises(Exception) as exc_info:
            await ws_manager.get_service_info(f"{ws2}/service-creator:secret-service")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        # Attempt 3: Try with wildcard client (should also fail)
        with pytest.raises(Exception) as exc_info:
            await api_attacker.get_service(f"{ws2}/*:secret-service")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg or "not found" in error_msg, \
            f"Expected permission/not found error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()

    async def test_public_service_accessible_cross_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that public services ARE accessible across workspaces."""
        api1 = await connect_to_server({
            "client_id": "public-service-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Register a public service in ws1
        async def public_function(data):
            return f"PUBLIC: {data}"

        await api1.register_service({
            "id": "public-service",
            "name": "Public Service",
            "config": {"visibility": "public"},  # Accessible to all
            "process": public_function,
        })

        # Create second workspace and access the public service
        ws2_info = await api1.create_workspace({
            "name": f"public-reader-ws-{uuid.uuid4().hex[:8]}",
            "description": "Reader's workspace",
        })
        ws2 = ws2_info["id"]

        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "public-reader",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        # Should be able to access public service from different workspace
        svc = await api2.get_service(f"{ws1}/public-service-owner:public-service")
        result = await svc.process("test")
        assert result == "PUBLIC: test"

        await api1.disconnect()
        await api2.disconnect()

    async def test_authorized_workspaces_grants_access(
        self, fastapi_server, test_user_token
    ):
        """Test that authorized_workspaces allows specific workspaces to access protected services."""
        api1 = await connect_to_server({
            "client_id": "auth-service-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create workspace 2 (authorized)
        ws2_info = await api1.create_workspace({
            "name": f"authorized-ws-{uuid.uuid4().hex[:8]}",
            "description": "Authorized workspace",
        })
        ws2 = ws2_info["id"]

        # Create workspace 3 (NOT authorized)
        ws3_info = await api1.create_workspace({
            "name": f"unauthorized-ws-{uuid.uuid4().hex[:8]}",
            "description": "Unauthorized workspace",
        })
        ws3 = ws3_info["id"]

        # Register a protected service with authorized_workspaces
        async def shared_function(data):
            return f"SHARED: {data}"

        await api1.register_service({
            "id": "shared-service",
            "name": "Shared Service",
            "config": {
                "visibility": "protected",
                "authorized_workspaces": [ws2]  # Only ws2 is authorized
            },
            "process": shared_function,
        })

        # Connect to ws2 (authorized) - should succeed
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "authorized-client",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        svc = await api2.get_service(f"{ws1}/auth-service-owner:shared-service")
        result = await svc.process("test")
        assert result == "SHARED: test"

        # Connect to ws3 (NOT authorized) - should fail
        ws3_token = await api1.generate_token({"workspace": ws3})
        api3 = await connect_to_server({
            "client_id": "unauthorized-client",
            "workspace": ws3,
            "server_url": WS_SERVER_URL,
            "token": ws3_token,
        })

        with pytest.raises(Exception) as exc_info:
            await api3.get_service(f"{ws1}/auth-service-owner:shared-service")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api3.disconnect()

    async def test_cannot_list_services_in_other_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that listing services is restricted to authorized workspaces."""
        api1 = await connect_to_server({
            "client_id": "list-test-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"list-victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace",
        })
        ws2 = ws2_info["id"]

        # Register a protected service in ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "list-service-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        async def private_func():
            return "private"

        await api2.register_service({
            "id": "private-list-service",
            "name": "Private Service",
            "config": {"visibility": "protected"},
            "run": private_func,
        })

        # Try to list services in ws2 from ws1
        ws1_token = await api1.generate_token({"workspace": ws1})
        api_attacker = await connect_to_server({
            "client_id": "list-attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": ws1_token,
        })

        # List services in ws2 - protected services should NOT be visible
        services = await api_attacker.list_services(f"{ws2}/*")

        # Check that the private service is NOT in the list
        service_ids = [s["id"] for s in services]
        private_services = [sid for sid in service_ids if "private-list-service" in sid]
        assert len(private_services) == 0, \
            f"Protected service should not be visible in cross-workspace listing: {private_services}"

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()
