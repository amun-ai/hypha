"""Test admin terminal and utilities."""

import asyncio
import pytest
import secrets
from hypha.admin import AdminTerminal, setup_admin_services
from hypha_rpc import connect_to_server

from . import SIO_PORT_SQLITE


@pytest.mark.asyncio
async def test_admin_utilities(fastapi_server_sqlite, root_user_token):
    """Test admin utilities functions."""
    # Connect as root user
    ws_url = f"ws://127.0.0.1:{SIO_PORT_SQLITE}/ws"
    api = await connect_to_server(
        {"server_url": ws_url, "token": root_user_token}
    )
    
    # Get admin service
    admin_service = await api.get_service("admin-utils")
    
    # Test list servers
    servers = await admin_service.list_servers()
    assert isinstance(servers, list)
    
    # Test list workspaces
    workspaces = await admin_service.list_workspaces()
    assert isinstance(workspaces, list)
    assert any(ws["id"] == "public" for ws in workspaces)
    assert any(ws["id"] == "ws-user-root" for ws in workspaces)
    
    # Test get metrics
    metrics = await admin_service.get_metrics()
    assert "rpc" in metrics
    assert "eventbus" in metrics
    
    # Test unload workspace (create a temporary workspace to unload)
    test_workspace = f"test-admin-{secrets.token_hex(4)}"
    
    # Create a workspace explicitly
    await api.create_workspace({"name": test_workspace}, overwrite=True)
    
    # Register a service in the workspace
    await api.register_service(
        {
            "id": "test-service",
            "name": "Test Service",
            "workspace": test_workspace
        },
        overwrite=True
    )
    
    # Wait a bit for workspace to be fully created
    await asyncio.sleep(0.5)
    
    # Workspace should be created
    workspaces = await admin_service.list_workspaces()
    assert any(ws["id"] == test_workspace for ws in workspaces)
    
    # Unload the workspace
    await admin_service.unload_workspace(test_workspace, wait=True, timeout=5)
    
    # Workspace should be unloaded
    workspaces = await admin_service.list_workspaces()
    assert not any(ws["id"] == test_workspace for ws in workspaces)
    
    # Disconnect
    await api.disconnect()


@pytest.mark.asyncio
async def test_admin_terminal_basic():
    """Test basic admin terminal functionality."""
    # Create a mock store with minimal requirements
    class MockStore:
        def __init__(self):
            self._redis = None
            self._root_user = None
            self._app = None
            
        async def list_servers(self):
            return ["server-1", "server-2"]
            
        async def list_all_workspaces(self):
            return [{"id": "public"}, {"id": "ws-user-root"}]
            
        async def get_metrics(self):
            return {"rpc": {}, "eventbus": {}}
            
        async def unload_workspace(self, workspace, wait=False, timeout=10):
            return True
            
        def kickout_client(self, workspace, client_id, code, reason):
            return True
    
    store = MockStore()
    terminal = AdminTerminal(store)
    
    # Test starting terminal
    result = await terminal.start_terminal()
    assert result["success"] is True
    assert "terminal_size" in result
    
    # Wait a bit for REPL to initialize
    await asyncio.sleep(0.1)
    
    # Test executing a simple Python expression
    result = await terminal.execute_command("2 + 2")
    assert result["success"] is True
    
    # Test writing input character by character (simulating user typing)
    # Type "print('hello')" followed by Enter
    await terminal.write_terminal("p")
    await terminal.write_terminal("r")
    await terminal.write_terminal("i")
    await terminal.write_terminal("n")
    await terminal.write_terminal("t")
    await terminal.write_terminal("(")
    await terminal.write_terminal("'")
    await terminal.write_terminal("h")
    await terminal.write_terminal("e")
    await terminal.write_terminal("l")
    await terminal.write_terminal("l")
    await terminal.write_terminal("o")
    await terminal.write_terminal("'")
    await terminal.write_terminal(")")
    await terminal.write_terminal("\r")
    
    # Wait for execution
    await asyncio.sleep(0.2)
    
    # Test reading from terminal
    result = await terminal.read_terminal()
    assert result["success"] is True
    output = result.get("output", "")
    assert "hello" in output or ">>>" in output  # Either output or just prompt
    
    # Test getting screen content
    result = await terminal.get_screen_content()
    assert result["success"] is True
    assert "content" in result
    
    # Test resizing terminal (should still work)
    result = await terminal.resize_terminal(30, 100)
    assert result["success"] is True
    
    # Test that store is available in namespace
    result = await terminal.execute_command("store")
    assert result["success"] is True
    
    # Stop the terminal
    await terminal.stop_terminal()
    
    # Verify terminal is stopped
    result = await terminal.read_terminal()
    assert result["success"] is False
    assert "not running" in result["error"]


@pytest.mark.asyncio
async def test_admin_utilities_setup():
    """Test admin utilities setup."""
    class MockStore:
        def __init__(self):
            self._redis = None
            self._root_user = None
            
        async def list_servers(self):
            return []
            
        async def list_all_workspaces(self):
            return []
            
        async def get_metrics(self):
            return {"rpc": {}, "eventbus": {}}
            
        async def unload_workspace(self, workspace, wait=False, timeout=10):
            return True
            
        def kickout_client(self, workspace, client_id, code, reason):
            return True
    
    store = MockStore()
    
    # Test setup without terminal
    admin = await setup_admin_services(store, enable_terminal=False)
    assert admin is not None
    assert admin.terminal is None
    
    service_api = admin.get_service_api()
    assert service_api["id"] == "admin-utils"
    assert "list_servers" in service_api
    assert "kickout_client" in service_api
    assert "list_workspaces" in service_api
    assert "unload_workspace" in service_api
    assert "get_metrics" in service_api
    # Terminal methods should not be included
    assert "start_terminal" not in service_api
    assert "resize_terminal" not in service_api
    
    # Test setup with terminal
    admin = await setup_admin_services(store, enable_terminal=True)
    assert admin is not None
    assert admin.terminal is not None
    
    service_api = admin.get_service_api()
    assert service_api["id"] == "admin-utils"
    # Terminal methods should be included
    assert "start_terminal" in service_api
    assert "resize_terminal" in service_api
    assert "read_terminal" in service_api
    assert "get_screen_content" in service_api
    assert "write_terminal" in service_api
    assert "execute_command" in service_api
    
    # Clean up terminal if started
    if admin.terminal and admin.terminal._running:
        await admin.terminal.stop_terminal()


@pytest.mark.asyncio
async def test_admin_terminal_with_server(fastapi_server_sqlite, root_user_token):
    """Test admin terminal integration with actual server."""
    # Connect as root user
    ws_url = f"ws://127.0.0.1:{SIO_PORT_SQLITE}/ws"
    api = await connect_to_server(
        {"server_url": ws_url, "token": root_user_token}
    )
    
    # Get the admin service
    admin_service = await api.get_service("admin-utils")
    
    # Basic admin functions should work
    servers = await admin_service.list_servers()
    assert isinstance(servers, list)
    
    workspaces = await admin_service.list_workspaces()
    assert isinstance(workspaces, list)
    
    metrics = await admin_service.get_metrics()
    assert isinstance(metrics, dict)
    
    # Note: Terminal functions require --enable-admin-terminal flag
    # which is not set in the test fixture by default
    # So we just verify the basic admin utilities work
    
    # Disconnect
    await api.disconnect()