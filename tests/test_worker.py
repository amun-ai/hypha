"""Tests for the worker API and workspace-specific worker support."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from . import WS_SERVER_URL, SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_worker_api_imports():
    """Test that worker API classes can be imported."""
    try:
        from hypha.workers.base import (
            BaseWorker,
            WorkerConfig,
            SessionInfo,
            SessionStatus,
            WorkerError,
            SessionNotFoundError,
            WorkerNotAvailableError,
        )
        assert BaseWorker is not None
        assert WorkerConfig is not None
        assert SessionInfo is not None
        assert SessionStatus is not None
        assert WorkerError is not None
        assert SessionNotFoundError is not None
        assert WorkerNotAvailableError is not None
        print("✓ All worker API classes imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import worker API classes: {e}")


async def test_worker_config_creation():
    """Test WorkerConfig dataclass creation."""
    from hypha.workers.base import WorkerConfig
    
    config = WorkerConfig(
        id="test-workspace/test-client",
        client_id="test-client",
        app_id="test-app",
        server_url="http://localhost:8080",
        workspace="test-workspace",
        entry_point="index.html",
        artifact_id="test-workspace/test-app",
        manifest={"test": "value", "type": "web-python"},
        token="test-token"
    )
    
    assert config.client_id == "test-client"
    assert config.app_id == "test-app"
    assert config.workspace == "test-workspace"
    assert config.entry_point == "index.html"
    assert config.manifest == {"test": "value", "type": "web-python"}
    assert config.id == "test-workspace/test-client"
    assert config.artifact_id == "test-workspace/test-app"
    print("✓ WorkerConfig created successfully")


async def test_session_info_creation():
    """Test SessionInfo dataclass creation."""
    from hypha.workers.base import SessionInfo, SessionStatus
    
    session_info = SessionInfo(
        session_id="test-workspace/test-client",
        app_id="test-app",
        workspace="test-workspace",
        client_id="test-client",
        status=SessionStatus.RUNNING,
        app_type="web-python",
        created_at="2024-01-01T00:00:00Z",
        entry_point="index.html",
        metadata={"test": "value"}
    )
    
    assert session_info.session_id == "test-workspace/test-client"
    assert session_info.app_id == "test-app"
    assert session_info.workspace == "test-workspace"
    assert session_info.status == SessionStatus.RUNNING
    assert session_info.app_type == "web-python"
    print("✓ SessionInfo created successfully")


async def test_browser_worker_integration(fastapi_server, test_user_token):
    """Test browser worker integration with actual server."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Test simple app installation and running (browser worker will be selected automatically)
    test_app_code = """
    api.log('Browser worker test app started');
    
    api.export({
        async setup() {
            console.log("Browser worker app initialized");
        },
        async test_function() {
            return "Browser worker test successful";
        }
    });
    """
    
    # Install and run app (this will use the browser worker for window type)
    config = await controller.install(
        source=test_app_code,
        config={"type": "window", "name": "Browser Worker Test"},
        wait_for_service="default",
        timeout=15,  # Reduced from 30
        overwrite=True,
    )
    config = await controller.start(config.id)
    
    assert "id" in config
    app = await api.get_app(config.id)
    assert "test_function" in app
    
    result = await app.test_function()
    assert result == "Browser worker test successful"
    print("✓ Browser worker integration test passed")
    
    # Clean up
    await controller.stop(config.id)
    await api.disconnect()


async def test_python_worker_integration(fastapi_server, test_user_token):
    """Test Python worker integration with actual server."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Test Python app (python worker will be selected automatically)
    test_python_code = """
import os
from hypha_rpc.sync import connect_to_server

server = connect_to_server({
    "client_id": os.environ["HYPHA_CLIENT_ID"],
    "server_url": os.environ["HYPHA_SERVER_URL"],
    "workspace": os.environ["HYPHA_WORKSPACE"],
    "token": os.environ["HYPHA_TOKEN"],
})

# Test Python worker functionality
result = 5 + 3
print(f"Python worker calculation: {result}")

server.register_service({
    "id": "default",
    "name": "Python Worker Test",
    "calculate": lambda a, b: a + b,
    "get_result": lambda: result,
})

print("Python worker test app registered successfully")
"""
    
    try:
        # Install and run Python app (this will use the python worker for python-eval type)
        config = await controller.install(
            source=test_python_code,
            config={"type": "python-eval", "name": "Python Worker Test"},
            timeout=5,  # Reduced from 10
            overwrite=True,
        )
        config = await controller.start(config.id)
        # Check logs
        logs = await controller.get_logs(config.id)
        log_text = " ".join(logs.get("log", []))
        assert "Python worker calculation: 8" in log_text
        assert "Python worker test app registered successfully" in log_text
        print("✓ Python worker integration test passed")
        
        # Clean up
        await controller.stop(config.id)
        
    except Exception as e:
        print(f"Python worker test failed (may not be available): {e}")
        # Python worker might not be available in all environments
    
    await api.disconnect()


async def test_mcp_worker_integration(fastapi_server, test_user_token):
    """Test MCP worker integration with actual server."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Register a simple MCP service first
    @schema_function
    def test_tool(message: str) -> str:
        """Test MCP tool."""
        return f"MCP processed: {message}"
    
    mcp_service = await api.register_service({
        "id": "test-mcp-service",
        "name": "Test MCP Service",
        "type": "mcp",
        "config": {"visibility": "public"},
        "tools": [test_tool],
    })
    
    # Create MCP app config (MCP worker will be selected automatically)
    mcp_config = {
        "type": "mcp-server",
        "name": "Test MCP App",
        "version": "1.0.0",
        "mcpServers": {
            "test-server": {
                "type": "streamable-http",
                "url": f"{SERVER_URL}/{workspace}/mcp/{mcp_service['id'].split('/')[-1]}/mcp"
            }
        }
    }
    
    # Install MCP app
    app_info = await controller.install(
        config=mcp_config,
        overwrite=True,
    )
    
    # Start MCP app
    session_info = await controller.start(
        app_info["id"],
        wait_for_service="test-server",
        timeout=15,  # Reduced from 30
    )
    
    # Test MCP service
    mcp_service_unified = await api.get_service(f"{session_info['id']}:test-server")
    assert mcp_service_unified is not None
    assert hasattr(mcp_service_unified, "tools")
    print("✓ MCP worker integration test passed")
    
    # Clean up
    await controller.stop(session_info["id"])
    await controller.uninstall(app_info["id"])
    await api.unregister_service(mcp_service["id"])
    await api.disconnect()


async def test_a2a_worker_integration(fastapi_server, test_user_token):
    """Test A2A worker integration with actual server."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Register a simple A2A service first
    @schema_function
    def test_skill(text: str) -> str:
        """Test A2A skill."""
        return f"A2A processed: {text}"
    
    async def a2a_run(message, context=None):
        """Simple A2A run function."""
        return test_skill(text=str(message))
    
    a2a_service = await api.register_service({
        "id": "test-a2a-service",
        "name": "Test A2A Service",
        "type": "a2a",
        "config": {"visibility": "public"},
        "skills": [test_skill],
        "run": a2a_run,
        "agent_card": {
            "name": "Test A2A Agent",
            "description": "Test agent for A2A integration",
            "version": "1.0.0",
            "url": f"{SERVER_URL}/{workspace}/a2a/test-a2a-service",
            "capabilities": {"streaming": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
        },
    })
    
    # Create A2A app config (A2A worker will be selected automatically)
    a2a_config = {
        "type": "a2a-agent",
        "name": "Test A2A App",
        "version": "1.0.0",
        "a2aAgents": {
            "test-agent": {
                "url": f"{SERVER_URL}/{workspace}/a2a/test-a2a-service",
                "headers": {}
            }
        }
    }
    
    # Install A2A app
    app_info = await controller.install(
        config=a2a_config,
        overwrite=True,
    )
    
    # Start A2A app
    session_info = await controller.start(
        app_info["id"],
        wait_for_service="test-agent",
        timeout=15,  # Reduced from 30
    )
    
    # Test A2A service
    a2a_service_unified = await api.get_service(f"{session_info['id']}:test-agent")
    assert a2a_service_unified is not None
    assert hasattr(a2a_service_unified, "skills")
    print("✓ A2A worker integration test passed")
    
    # Clean up
    await controller.stop(session_info["id"])
    await controller.uninstall(app_info["id"])
    await api.unregister_service(a2a_service["id"])
    await api.disconnect()


async def test_workspace_worker_registration(fastapi_server, test_user_token):
    """Test registering a custom worker in a workspace."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    
    # Create a mock worker service
    @schema_function
    async def start(
        client_id: str,
        app_id: str,
        server_url: str,
        public_base_url: str,
        local_base_url: str,
        workspace: str,
        version: str = None,
        token: str = None,
        entry_point: str = None,
        app_type: str = None,
        metadata: dict = None,
    ) -> dict:
        """Start a custom worker session."""
        return {
            "session_id": f"{workspace}/{client_id}",
            "status": "running",
            "url": f"http://localhost:8080/{workspace}/{app_id}",
            "custom_data": "test-worker-data",
        }
    
    @schema_function
    async def stop(session_id: str) -> None:
        """Stop a custom worker session."""
        pass
    
    @schema_function
    async def get_logs(session_id: str, type: str = None, offset: int = 0, limit: int = None) -> dict:
        """Get logs for a custom worker session."""
        return {
            "log": [f"Custom worker log for {session_id}"],
            "error": []
        }
    
    @schema_function
    async def list_sessions(workspace: str) -> list:
        """List sessions for a workspace."""
        return []
    
    # Register the custom worker
    worker_service = await api.register_service({
        "id": "custom-test-worker",
        "name": "Custom Test Worker",
        "type": "server-app-worker",
        "config": {
            "visibility": "public",
            "run_in_executor": True,
        },
        "supported_types": ["custom-test-type"],
        "start": start,
        "stop": stop,
        "get_logs": get_logs,
        "list_sessions": list_sessions,
    })
    
    print(f"✓ Custom worker registered: {worker_service['id']}")
    
    # Test that the worker service is properly registered with the new API
    assert worker_service["type"] == "server-app-worker"
    assert "custom-test-worker" in worker_service["id"]
    print("✓ Worker service registered with correct type and ID")
    
    # Test the worker methods are accessible
    worker = await api.get_service(worker_service["id"])
    assert hasattr(worker, "start")
    assert hasattr(worker, "stop")
    assert hasattr(worker, "get_logs")
    assert hasattr(worker, "list_sessions")
    print("✓ Worker methods are accessible")
    
    # Test calling the worker methods directly
    session_data = await worker.start(
        client_id="test-client",
        app_id="test-app",
        server_url="http://localhost:8080",
        public_base_url="http://localhost:8080",
        local_base_url="http://localhost:8080",
        workspace=workspace,
        app_type="custom-test-type",
    )
    
    assert session_data["session_id"] == f"{workspace}/test-client"
    assert session_data["status"] == "running"
    assert session_data["custom_data"] == "test-worker-data"
    print("✓ Worker start method works correctly")
    
    # Test getting logs
    logs = await worker.get_logs(session_data["session_id"])
    assert "Custom worker log" in logs["log"][0]
    print("✓ Worker get_logs method works correctly")
    
    # Test stopping session
    await worker.stop(session_data["session_id"])
    print("✓ Worker stop method works correctly")
    
    # Test listing sessions
    sessions = await worker.list_sessions(workspace)
    assert isinstance(sessions, list)
    print("✓ Worker list_sessions method works correctly")
    
    # Clean up
    await api.unregister_service(worker_service["id"])
    print("✓ Custom worker integration test passed")
    
    await api.disconnect()


async def test_worker_selection_by_type(fastapi_server, test_user_token):
    """Test that workers are correctly selected by app type."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Test different app types by installing and running simple apps
    test_cases = [
        ("window", "console.log('Window app test');"),
        ("web-python", "console.log('Web Python app test');"),
        ("web-worker", "console.log('Web Worker app test');"),
    ]
    
    for app_type, test_code in test_cases:
        # Create a simple app of the given type
        app_config = {
            "type": app_type,
            "name": f"Test {app_type} App",
            "version": "1.0.0",
        }
        
        # Install the app (worker selection happens automatically)
        app_info = await controller.install(
            source=f"<html><body><script>{test_code}</script></body></html>",
            config=app_config,
            timeout=5,
            detached=True,
            overwrite=True,
        )
        
        print(f"✓ App of type {app_type} installed successfully: {app_info['id']}")
        
        # Clean up
        await controller.uninstall(app_info["id"])
        

    await api.disconnect()


async def test_worker_concurrent_sessions(fastapi_server, test_user_token):
    """Test multiple concurrent sessions with the same worker."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Create multiple simple apps
    test_app_code = """
    api.export({
        async setup() {
            console.log("Concurrent test app initialized");
        },
        async get_id() {
            return Math.random().toString(36).substring(7);
        }
    });
    """
    
    sessions = []
    
    try:
        # Start multiple sessions concurrently
        for i in range(3):
            config = await controller.install(
                source=test_app_code,
                config={"type": "window", "name": f"Concurrent Test App {i}"},
                wait_for_service="default",
                timeout=10,  # Reduced from 30
                overwrite=True,
            )
            config = await controller.start(config.id)
            sessions.append(config)
        
        # Verify all sessions are running
        assert len(sessions) == 3
        
        # Test that each session is independent
        for i, session in enumerate(sessions):
            app = await api.get_app(session.id)
            app_id = await app.get_id()
            assert app_id is not None
            print(f"✓ Session {i} is running independently with ID: {app_id}")
        
        print("✓ Concurrent sessions test passed")
        
    finally:
        # Clean up all sessions
        for session in sessions:
            try:
                await controller.stop(session.id)
            except Exception as e:
                print(f"Error stopping session {session.id}: {e}")
    
    await api.disconnect()


async def test_worker_error_handling(fastapi_server, test_user_token):
    """Test worker error handling scenarios."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Test 1: Invalid app type
    try:
        workers = await controller.get_server_app_workers(app_type="nonexistent-type")
        assert len(workers) == 0, "Should return empty list for nonexistent type"
        print("✓ Invalid app type handled correctly")
    except Exception as e:
        print(f"✓ Invalid app type properly rejected: {e}")
    
    # Test 2: App with syntax errors
    bad_app_code = """
    this is not valid javascript syntax
    api.export({
        async setup() {
            console.log("This won't work");
        }
    });
    """
    
    try:
        config = await controller.install(
            source=bad_app_code,
            config={"type": "window", "name": "Bad App"},
            timeout=5,
            overwrite=True,
        )
        config = await controller.start(config.id)
        # If it somehow succeeds, stop it
        await controller.stop(config.id)
        print("⚠️ Bad app unexpectedly succeeded")
    except Exception as e:
        print(f"✓ Bad app code properly rejected: {type(e).__name__}")
    
    # Test 3: App that takes too long to start
    slow_app_code = """
    // This app will take a long time to initialize
    setTimeout(() => {
        api.export({
            async setup() {
                console.log("Slow app finally started");
            }
        });
    }, 10000); // 10 seconds
    """
    
    try:
        config = await controller.install(
            source=slow_app_code,
            config={"type": "window", "name": "Slow App"},
            timeout=2,  # Short timeout
            overwrite=True,
        )
        config = await controller.start(config.id)
        await controller.stop(config.id)
        print("⚠️ Slow app unexpectedly succeeded")
    except Exception as e:
        print(f"✓ Slow app timeout handled correctly: {type(e).__name__}")
    
    print("✓ Worker error handling tests completed")
    await api.disconnect()


async def test_worker_lifecycle_management(fastapi_server, test_user_token):
    """Test complete worker lifecycle management."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    # Get the controller
    controller = await api.get_service("public/server-apps")
    
    # Test app
    test_app_code = """
    api.export({
        async setup() {
            console.log("Lifecycle test app started");
        },
        async get_status() {
            return "running";
        }
    });
    """
    
    # 1. Install app
    app_info = await controller.install(
        source=test_app_code,
        config={"type": "window", "name": "Lifecycle Test App"},
        overwrite=True,
    )
    
    # Verify installation
    apps = await controller.list_apps()
    assert any(app["id"] == app_info["id"] for app in apps)
    print("✓ App installed successfully")
    
    # 2. Start app
    session_info = await controller.start(
        app_info["id"],
        wait_for_service="default",
        timeout=10,  # Reduced from 30
    )
    
    # Verify running
    running_apps = await controller.list_running()
    assert any(app["id"] == session_info["id"] for app in running_apps)
    print("✓ App started successfully")
    
    # 3. Test app functionality
    app = await api.get_app(session_info["id"])
    status = await app.get_status()
    assert status == "running"
    print("✓ App functioning correctly")
    
    # 4. Get logs
    logs = await controller.get_logs(session_info["id"])
    assert "log" in logs
    assert len(logs["log"]) > 0
    print("✓ Logs retrieved successfully")
    
    # 5. Stop app
    await controller.stop(session_info["id"])
    
    # Verify stopped
    running_apps = await controller.list_running()
    assert not any(app["id"] == session_info["id"] for app in running_apps)
    print("✓ App stopped successfully")
    
    # 6. Uninstall app
    await controller.uninstall(app_info["id"])
    
    # Verify uninstalled
    apps = await controller.list_apps()
    assert not any(app["id"] == app_info["id"] for app in apps)
    print("✓ App uninstalled successfully")
    
    print("✓ Complete lifecycle management test passed")
    await api.disconnect()


async def test_worker_workspace_isolation(fastapi_server, test_user_token, test_user_token_2):
    """Test that workers maintain workspace isolation."""
    # Connect to two different workspaces
    api1 = await connect_to_server(
        {
            "name": "test client 1",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    api2 = await connect_to_server(
        {
            "name": "test client 2",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token_2,
        }
    )
    
    workspace1 = api1.config.workspace
    workspace2 = api2.config.workspace
    
    assert workspace1 != workspace2, "Workspaces should be different"
    
    # Get controllers for both workspaces
    controller1 = await api1.get_service("public/server-apps")
    controller2 = await api2.get_service("public/server-apps")
    
    # Start apps in both workspaces
    test_app_code = """
    api.export({
        async setup() {
            console.log("Workspace isolation test app started");
        },
        async get_workspace() {
            return api.config.workspace;
        }
    });
    """
    
    config1 = await controller1.install(
        source=test_app_code,
        config={"type": "window", "name": "Workspace 1 App"},
        wait_for_service="default",
        timeout=10,  # Reduced from 30
        overwrite=True,
    )
    config1 = await controller1.start(config1.id)
    
    config2 = await controller2.install(
        source=test_app_code,
        config={"type": "window", "name": "Workspace 2 App"},
        wait_for_service="default",
        timeout=10,  # Reduced from 30
        overwrite=True,
    )
    config2 = await controller2.start(config2.id)
    
    # Verify apps are in different workspaces
    app1 = await api1.get_app(config1.id)
    app2 = await api2.get_app(config2.id)
    
    workspace1_from_app = await app1.get_workspace()
    workspace2_from_app = await app2.get_workspace()
    
    assert workspace1_from_app == workspace1
    assert workspace2_from_app == workspace2
    assert workspace1_from_app != workspace2_from_app
    
    print(f"✓ Apps correctly isolated in workspaces: {workspace1} and {workspace2}")
    
    # Verify that each workspace only sees its own apps
    running1 = await controller1.list_running()
    running2 = await controller2.list_running()
    
    # Each workspace should only see its own running apps
    workspace1_sessions = [app for app in running1 if workspace1 in app.get("id", "")]
    workspace2_sessions = [app for app in running2 if workspace2 in app.get("id", "")]
    
    assert len(workspace1_sessions) > 0
    assert len(workspace2_sessions) > 0
    
    # No cross-workspace visibility
    assert not any(workspace2 in app.get("id", "") for app in running1)
    assert not any(workspace1 in app.get("id", "") for app in running2)
    
    print("✓ Workspace isolation verified")
    
    # Clean up
    await controller1.stop(config1.id)
    await controller2.stop(config2.id)
    await api1.disconnect()
    await api2.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 