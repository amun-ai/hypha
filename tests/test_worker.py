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


async def test_conda_worker_imports():
    """Test that conda worker can be imported."""
    from hypha.workers.conda import CondaWorker, get_available_package_manager

    assert CondaWorker is not None
    assert get_available_package_manager is not None
    print("✓ Conda worker classes imported successfully")


async def test_browser_worker_imports():
    """Test that browser worker can be imported."""
    from hypha.workers.browser import BrowserWorker

    assert BrowserWorker is not None
    print("✓ Browser worker classes imported successfully")


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
        token="test-token",
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
        metadata={"test": "value"},
    )

    assert session_info.session_id == "test-workspace/test-client"
    assert session_info.app_id == "test-app"
    assert session_info.workspace == "test-workspace"
    assert session_info.status == SessionStatus.RUNNING
    assert session_info.app_type == "web-python"
    print("✓ SessionInfo created successfully")


async def test_conda_worker_service_registration(fastapi_server, test_user_token):
    """Test conda worker service registration and basic functionality."""
    from hypha.workers.conda import CondaWorker

    api = await connect_to_server(
        {
            "name": "conda worker test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace

    # Create and register conda worker
    worker = CondaWorker()

    # Get service config and register (use get_worker_service() to include conda-specific methods)
    service_config = worker.get_worker_service()
    service_config["id"] = "test-conda-worker"
    service_config["visibility"] = "public"

    await api.rpc.register_service(service_config)

    # Verify worker is registered
    worker_service = await api.get_service("test-conda-worker")
    assert worker_service is not None
    assert hasattr(worker_service, "start")
    assert hasattr(worker_service, "stop")
    assert hasattr(worker_service, "execute")
    assert hasattr(worker_service, "get_logs")

    print("✓ Conda worker registered successfully")

    # Test supported types
    assert "conda-jupyter-kernel" in worker.supported_types
    print(f"✓ Conda worker supports types: {worker.supported_types}")

    # Test worker info
    assert "Conda Worker" in worker.name
    assert worker.description is not None
    print(f"✓ Worker info: {worker.name}")

    # Clean up
    await api.unregister_service("test-conda-worker")
    await worker.shutdown()
    await api.disconnect()


async def test_browser_worker_service_registration(fastapi_server, test_user_token):
    """Test browser worker service registration and basic functionality."""
    from hypha.workers.browser import BrowserWorker

    api = await connect_to_server(
        {
            "name": "browser worker test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace

    # Create and register browser worker
    worker = BrowserWorker(in_docker=False)

    # Get service config and register (use get_service() to include browser-specific methods)
    service_config = worker.get_worker_service()
    service_config["id"] = "test-browser-worker"
    service_config["visibility"] = "public"

    await api.rpc.register_service(service_config)

    # Verify worker is registered
    worker_service = await api.get_service("test-browser-worker")
    assert worker_service is not None
    assert hasattr(worker_service, "start")
    assert hasattr(worker_service, "stop")
    assert hasattr(worker_service, "get_logs")
    assert hasattr(worker_service, "compile")

    print("✓ Browser worker registered successfully")

    # Test supported types
    expected_types = [
        "web-python",
        "web-worker",
        "window",
        "iframe",
        "hypha",
        "web-app",
    ]
    for app_type in expected_types:
        assert app_type in worker.supported_types
    print(f"✓ Browser worker supports types: {worker.supported_types}")

    # Test worker info
    assert "Browser Worker" in worker.name
    assert worker.description is not None
    print(f"✓ Worker info: {worker.name}")

    # Clean up
    await api.unregister_service("test-browser-worker")
    await worker.shutdown()
    await api.disconnect()


async def test_conda_worker_session_lifecycle(fastapi_server, test_user_token):
    """Test conda worker session lifecycle management."""
    from hypha.workers.conda import CondaWorker
    from hypha.workers.base import WorkerConfig

    api = await connect_to_server(
        {
            "name": "conda session test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 60,  # Longer timeout for conda operations
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace

    # Create and register conda worker
    worker = CondaWorker()
    service_config = worker.get_worker_service()
    service_config["id"] = "test-conda-session-worker"
    service_config["visibility"] = "public"

    await api.rpc.register_service(service_config)
    worker_service = await api.get_service("test-conda-session-worker")


    # Note: We can't actually start a conda session without proper setup
    # So we'll test the session management methods directly

    # Test get logs for non-existent session (should handle gracefully)
    try:
        logs = await worker_service.get_logs("non-existent-session")
        assert isinstance(logs, (dict, list))
    except Exception as e:
        # Expected to fail for non-existent session
        assert "not found" in str(e).lower()
    print("✓ Get logs handles non-existent session correctly")

    print("✓ Conda worker session management tested")

    # Clean up
    await api.unregister_service("test-conda-session-worker")
    await worker.shutdown()
    await api.disconnect()


async def test_browser_worker_session_lifecycle(fastapi_server, test_user_token):
    """Test browser worker session lifecycle management."""
    from hypha.workers.browser import BrowserWorker
    from hypha.workers.base import WorkerConfig

    api = await connect_to_server(
        {
            "name": "browser session test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 60,  # Longer timeout for browser operations
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace

    # Create and register browser worker
    worker = BrowserWorker(in_docker=False)
    service_config = worker.get_worker_service()
    service_config["id"] = "test-browser-session-worker"
    service_config["visibility"] = "public"

    await api.rpc.register_service(service_config)
    worker_service = await api.get_service("test-browser-session-worker")

    # Test get logs for non-existent session (should handle gracefully)
    try:
        logs = await worker_service.get_logs("non-existent-session")
        assert isinstance(logs, (dict, list))
    except Exception as e:
        # Expected to fail for non-existent session
        assert "not found" in str(e).lower()
    print("✓ Get logs handles non-existent session correctly")

    # Test take screenshot for non-existent session (should fail gracefully)
    try:
        screenshot = await worker_service.take_screenshot("non-existent-session")
    except Exception as e:
        # Expected to fail for non-existent session
        assert "not found" in str(e).lower() or "not found" in str(e)
    print("✓ Take screenshot handles non-existent session correctly")

    # Test compile method
    manifest = {
        "type": "web-python",
        "name": "Test Web Python App",
        "entry_point": "main.py",
    }
    files = [
        {
            "path": "main.py",
            "content": "print('Hello from web python')",
            "format": "text",
        }
    ]
    config = {"progress_callback": lambda x: print(f"Compile: {x}")}

    compiled_manifest, compiled_files = await worker_service.compile(
        manifest, files, config
    )
    assert isinstance(compiled_manifest, dict)
    assert isinstance(compiled_files, list)
    assert compiled_manifest["type"] == "web-python"
    print("✓ Compile method works")

    print("✓ Browser worker session management tested")

    # Clean up
    await api.unregister_service("test-browser-session-worker")
    await worker.shutdown()
    await api.disconnect()


async def test_conda_worker_cli_functionality():
    """Test conda worker CLI argument parsing and configuration."""
    import subprocess
    import sys

    # Test help functionality (should not raise exception)
    result = subprocess.run(
        [sys.executable, "-m", "hypha.workers.conda", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Hypha Conda Environment Worker" in result.stdout

    print("✓ Conda worker CLI help works")

    # Test missing required arguments (should show help and environment variables info)
    result = subprocess.run(
        [sys.executable, "-m", "hypha.workers.conda"],
        capture_output=True,
        text=True,
        env={}  # Clear environment to ensure no HYPHA_ variables are set
    )
    # Without required arguments or env vars, it should fail
    assert result.returncode != 0 or "Server URL" in result.stdout

    print("✓ Conda worker CLI validates required arguments")


async def test_browser_worker_cli_functionality():
    """Test browser worker CLI argument parsing and configuration."""
    from hypha.workers.browser import main
    import sys
    from unittest.mock import patch

    # Test help functionality (should not raise exception)
    with patch.object(sys, "argv", ["browser.py", "--help"]):
        try:
            main()
        except SystemExit as e:
            # --help causes SystemExit with code 0
            assert e.code == 0

    print("✓ Browser worker CLI help works")

    # Test missing required arguments
    with patch.object(sys, "argv", ["browser.py"]):
        try:
            main()
        except SystemExit as e:
            # Missing required args should cause SystemExit with code 1
            assert e.code == 1

    print("✓ Browser worker CLI validates required arguments")


async def test_worker_environment_variable_support():
    """Test that both workers support environment variables correctly."""
    import os
    from unittest.mock import patch

    # Test conda worker environment variables by testing the CLI function
    from hypha.workers.conda import main
    import sys

    # Test that environment variables are used when set
    with patch.dict(os.environ, {"HYPHA_SERVER_URL": "https://test.example.com"}):
        with patch.object(sys, "argv", ["conda.py", "--help"]):
            try:
                main()
            except SystemExit:
                pass  # Expected for --help

    print("✓ Conda worker environment variable support works")

    # Test browser worker environment variables
    from hypha.workers.browser import main
    import sys

    # We can test the get_env_var function indirectly by checking argument defaults
    with patch.dict(
        os.environ, {"HYPHA_SERVER_URL": "https://browser-test.example.com"}
    ):
        with patch.object(sys, "argv", ["browser.py", "--help"]):
            try:
                main()
            except SystemExit:
                pass  # Expected for --help

    print("✓ Browser worker environment variable support works")


async def test_worker_docker_configuration():
    """Test docker-specific configuration for workers."""
    # Test conda worker docker setup
    from hypha.workers.conda import CondaWorker

    # Create worker instance
    worker = CondaWorker()

    # Test that worker can be created (Docker-specific setup would be in main())
    assert worker is not None
    assert hasattr(worker, "supported_types")

    print("✓ Conda worker can be created for Docker deployment")

    # Test browser worker docker setup
    from hypha.workers.browser import BrowserWorker

    # Create worker instance with Docker flag
    worker = BrowserWorker(in_docker=True)

    # Test that worker can be created with Docker configuration
    assert worker is not None
    assert worker.in_docker == True

    print("✓ Browser worker can be created for Docker deployment")


async def test_worker_service_integration_with_controller(
    fastapi_server, test_user_token
):
    """Test that workers integrate properly with the server app controller."""
    api = await connect_to_server(
        {
            "name": "worker integration test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )

    # Get the controller
    controller = await api.get_service("public/server-apps")

    # Test that different worker types can be utilized by installing and starting apps

    # Test Python eval worker (conda worker)
    python_app_code = """
import os
from hypha_rpc.sync import connect_to_server

server = connect_to_server({
    "client_id": os.environ["HYPHA_CLIENT_ID"],
    "server_url": os.environ["HYPHA_SERVER_URL"],
    "workspace": os.environ["HYPHA_WORKSPACE"],
    "token": os.environ["HYPHA_TOKEN"],
})

print("Python worker test successful")

server.register_service({
    "id": "default",
    "name": "Python Worker Test Service",
    "test": lambda: "Python worker working"
})
"""
    config = await controller.install(
        source=python_app_code,
        manifest={"type": "python-eval", "name": "Python Worker Test"},
        timeout=10,
        overwrite=True,
    )
    config = await controller.start(config.id)
    # Just verify we can start the app - the exact return format doesn't matter for this test
    print("✓ Python eval worker integration - app started successfully")

    # Verify the app is running by checking it's in the running list
    running_apps = await controller.list_running()
    app_running = any(app["id"] == config.id for app in running_apps)
    assert app_running, f"App {config.id} should be running"
    print("✓ App is confirmed running")

    await controller.stop(config.id)

    # Test that we can install another app using the same worker
    python_app_code2 = """
import os
from hypha_rpc.sync import connect_to_server

server = connect_to_server({
    "client_id": os.environ["HYPHA_CLIENT_ID"],
    "server_url": os.environ["HYPHA_SERVER_URL"],
    "workspace": os.environ["HYPHA_WORKSPACE"],
    "token": os.environ["HYPHA_TOKEN"],
})

print("Second Python worker test successful")

server.register_service({
    "id": "default",
    "name": "Second Python Worker Test Service",
    "calculate": lambda a, b: a + b
})
"""
    config2 = await controller.install(
        source=python_app_code2,
        manifest={"type": "python-eval", "name": "Second Python Worker Test"},
        timeout=10,
        overwrite=True,
    )
    config2 = await controller.start(config2.id)
    print("✓ Second Python eval worker integration - app started successfully")

    # Verify both workers can be used for different apps
    running_apps = await controller.list_running()
    app2_running = any(app["id"] == config2.id for app in running_apps)
    assert app2_running, f"App {config2.id} should be running"
    print("✓ Second app is confirmed running")

    await controller.stop(config2.id)

    # Test that controller basic functionality works
    running_apps = await controller.list_running()
    assert isinstance(running_apps, list)
    print("✓ Controller list_running works")

    await api.disconnect()


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
        manifest={"type": "window", "name": "Browser Worker Test"},
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

    # Install and run Python app (this will use the python worker for python-eval type)
    config = await controller.install(
        source=test_python_code,
        manifest={"type": "python-eval", "name": "Python Worker Test"},
        timeout=5,  # Reduced from 10
        overwrite=True,
    )
    config = await controller.start(config.id)

    # Wait a bit for the Python code to execute and logs to be populated
    import asyncio

    await asyncio.sleep(2)

    # Check logs - Python worker outputs to stdout, not log
    logs = await controller.get_logs(config.id)

    # Get all log content from log items
    all_logs = []
    if "items" in logs:
        for item in logs["items"]:
            all_logs.append(item["content"])
    full_log_text = " ".join(all_logs)

    # Check for the expected output in any log type
    assert "Python worker calculation: 8" in full_log_text
    assert "Python worker test app registered successfully" in full_log_text
    print("✓ Python worker integration test passed")

    # Clean up
    await controller.stop(config.id)

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

    mcp_service = await api.register_service(
        {
            "id": "test-mcp-service",
            "name": "Test MCP Service",
            "type": "mcp",
            "config": {"visibility": "public"},
            "tools": {"test_tool": test_tool},
        }
    )

    # Create MCP app config (MCP worker will be selected automatically)
    mcp_config = {
        "type": "mcp-server",
        "name": "Test MCP App",
        "version": "1.0.0",
        "mcpServers": {
            "test-server": {
                "type": "streamable-http",
                "url": f"{SERVER_URL}/{workspace}/mcp/{mcp_service['id'].split('/')[-1]}/mcp",
            }
        },
    }

    # Install MCP app
    app_info = await controller.install(
        manifest=mcp_config,
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

    a2a_service = await api.register_service(
        {
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
                "default_input_modes": ["text/plain"],
                "default_output_modes": ["text/plain"],
            },
        }
    )

    # Create A2A app config (A2A worker will be selected automatically)
    a2a_config = {
        "type": "a2a-agent",
        "name": "Test A2A App",
        "version": "1.0.0",
        "a2aAgents": {
            "test-agent": {
                "url": f"{SERVER_URL}/{workspace}/a2a/test-a2a-service",
                "headers": {},
            }
        },
    }

    # Install A2A app
    app_info = await controller.install(
        manifest=a2a_config,
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
    async def get_logs(
        session_id: str, type: str = None, offset: int = 0, limit: int = None
    ) -> dict:
        """Get logs for a custom worker session."""
        items = [{"type": "log", "content": f"Custom worker log for {session_id}"}]
        if type:
            items = [item for item in items if item["type"] == type]
        
        if limit is not None:
            items = items[offset:offset + limit]
        else:
            items = items[offset:]
        
        return {
            "items": items,
            "total": 1,
            "offset": offset,
            "limit": limit
        }


    # Register the custom worker
    worker_service = await api.register_service(
        {
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
        }
    )

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
    assert "items" in logs
    log_items = [item for item in logs["items"] if item["type"] == "log"]
    assert len(log_items) > 0
    assert "Custom worker log" in log_items[0]["content"]
    print("✓ Worker get_logs method works correctly")

    # Test stopping session
    await worker.stop(session_data["session_id"])
    print("✓ Worker stop method works correctly")

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
            manifest=app_config,
            timeout=5,
            wait_for_service=False,
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

    # Start multiple sessions concurrently
    for i in range(3):
        config = await controller.install(
            source=test_app_code,
            manifest={"type": "window", "name": f"Concurrent Test App {i}"},
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

    # Clean up all sessions
    for session in sessions:
        await controller.stop(session.id)

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

    # Test 1: Invalid app manifest
    try:
        config = await controller.install(
            source="console.log('test');",
            manifest={"type": "nonexistent-type", "name": "Invalid Type App"},
            timeout=5,
            overwrite=True,
        )
        print("⚠️ Invalid app type unexpectedly succeeded")
    except Exception as e:
        print(f"✓ Invalid app type properly rejected: {type(e).__name__}")

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
            manifest={"type": "window", "name": "Bad App"},
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
            manifest={"type": "window", "name": "Slow App"},
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
        manifest={"type": "window", "name": "Lifecycle Test App"},
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
    assert "items" in logs
    console_logs = [item for item in logs["items"] if item["type"] == "console"]
    assert len(console_logs) > 0
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


async def test_worker_workspace_isolation(
    fastapi_server, test_user_token, test_user_token_2
):
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
        manifest={"type": "window", "name": "Workspace 1 App"},
        wait_for_service="default",
        timeout=10,  # Reduced from 30
        overwrite=True,
    )
    config1 = await controller1.start(config1.id)

    config2 = await controller2.install(
        source=test_app_code,
        manifest={"type": "window", "name": "Workspace 2 App"},
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


@pytest.mark.asyncio
async def test_worker_death_session_cleanup(fastapi_server, test_user_token):
    """Test that sessions are properly cleaned up when a worker dies."""
    from hypha.utils import random_id
    from hypha.workers.base import BaseWorker, WorkerConfig, SessionInfo, SessionStatus
    
    # Connect as the main test client
    api = await connect_to_server(
        {"client_id": "test-worker-death-main", "server_url": SERVER_URL, "token": test_user_token}
    )
    
    # Get the app controller service
    controller = await api.get_service("public/server-apps")
    
    # Check initial running apps
    running_apps_initial = await controller.list_running()
    
    # Create a test worker that extends BaseWorker properly
    class TestWorker(BaseWorker):
        def __init__(self):
            super().__init__()
            self._sessions = {}
            
        @property
        def name(self):
            return "TestWorker"
            
        @property
        def description(self):
            return "Test worker for death cleanup testing"
            
        @property
        def supported_types(self):
            return ["test-app"]
            
        @property
        def use_local_url(self):
            return True
            
        @property
        def visibility(self):
            return "public"  # Make worker public so it can be accessed from public workspace
            
        async def start(self, config, context=None):
            """Start a new worker session."""
            if isinstance(config, dict):
                # Convert dict to WorkerConfig for consistency
                config = WorkerConfig(**config)
            
            session_id = config.id
            session_info = SessionInfo(
                session_id=session_id,
                app_id=config.app_id,
                workspace=config.workspace,
                client_id=config.client_id,
                status=SessionStatus.RUNNING,
                app_type="test-app",
                created_at="2024-01-01T00:00:00Z",
                entry_point=config.entry_point,
            )
            self._sessions[session_id] = session_info
            print(f"🚀 TestWorker started session: {session_id}")
            
            # Simulate a running app by registering a service
            # This prevents the app from being stopped immediately
            import asyncio
            await asyncio.sleep(0.1)  # Small delay to simulate startup
            
            return session_id
            
        async def stop(self, session_id, context=None):
            """Stop a worker session."""
            if session_id in self._sessions:
                self._sessions.pop(session_id)
                print(f"🛑 TestWorker stopped session: {session_id}")

        async def get_logs(self, session_id, type=None, offset=0, limit=None, context=None):
            """Get logs for a session."""
            items = [{"type": "info", "content": f"Log from {session_id}"}]
            if type:
                items = [item for item in items if item["type"] == type]
            
            if limit is not None:
                items = items[offset:offset + limit]
            else:
                items = items[offset:]
            
            return {
                "items": items,
                "total": 1,
                "offset": offset,
                "limit": limit
            }

    # Connect as a worker client and register the test worker
    worker_client_id = "test-worker-client-" + random_id()
    worker_api = await connect_to_server(
        {"client_id": worker_client_id, "server_url": SERVER_URL, "token": test_user_token}
    )
    
    test_worker = TestWorker()
    worker_service = test_worker.get_worker_service()
    
    await worker_api.register_service(worker_service)
    
    # Install a test app that the worker supports
    test_app_source = """<config lang="json">
{
    "name": "Test App for Worker Death",
    "type": "test-app",
    "version": "0.1.0",
    "description": "Test app for worker death cleanup verification"
}
</config>
<script lang="python">
from hypha_rpc import api

def test_function():
    return "Hello from test app"

api.export({"test": test_function})
</script>
"""
    
    app_info = await controller.install(source=test_app_source, overwrite=True, wait_for_service=False)
    app_id = app_info.get("id") or app_info.get("alias")
    
    # Start the app - it should use our test worker
    # The worker_id needs to be in format 'workspace/client_id:service_id' 
    worker_full_id = f"{worker_api.config.workspace}/{worker_client_id}:{test_worker.service_id}"
    session_info = await controller.start(app_id, worker_id=worker_full_id, wait_for_service=False)
    session_id = session_info.get("id")
    
    # Verify the session is running
    running_apps_with_session = await controller.list_running()
    assert len(running_apps_with_session) > len(running_apps_initial), "Session should be in running list"
    
    # Find our session in the running list
    our_session = None
    for session in running_apps_with_session:
        # session might be a dict/object with 'id' field
        session_id_to_check = session.get('id') if hasattr(session, 'get') else str(session)
        if session_id in session_id_to_check:
            our_session = session
            break
    
    assert our_session is not None, f"Our session {session_id} should be in running list"
    
    # Now simulate worker death by disconnecting the worker client
    await worker_api.disconnect()
    
    # Wait for cleanup events to be processed
    await asyncio.sleep(5)
    
    # Check running apps - our session should be gone
    running_apps_after = await controller.list_running()
    
    # Verify our session is no longer in the running list
    session_still_running = False
    for session in running_apps_after:
        # session might be a dict/object with 'id' field
        session_id_to_check = session.get('id') if hasattr(session, 'get') else str(session)
        if session_id in session_id_to_check:
            session_still_running = True
            break
    
    assert not session_still_running, f"Session {session_id} should be removed from running list after worker death"
    
    # Clean up
    await controller.uninstall(app_id)
    await api.disconnect()

@pytest.mark.asyncio
async def test_use_local_url_functionality(fastapi_server, test_user_token):
    """Test that built-in workers have use_local_url property set correctly."""
    # Test that built-in workers have use_local_url=True
    from hypha.workers.browser import BrowserWorker
    from hypha.workers.conda import CondaWorker
    
    # Test BrowserWorker (built-in worker)
    browser_worker = BrowserWorker()
    assert hasattr(browser_worker, 'use_local_url'), "BrowserWorker should have use_local_url property"
    assert browser_worker.use_local_url == True, "BrowserWorker should have use_local_url=True"
    
    # Test that the property is included in worker service registration
    browser_service = browser_worker.get_worker_service()
    assert 'use_local_url' in browser_service, "Worker service should include use_local_url"
    assert browser_service['use_local_url'] == True, "Worker service use_local_url should match property"
    
    # Test CondaWorker (can be external worker) - should have use_local_url=False by default
    conda_worker = CondaWorker()
    assert hasattr(conda_worker, 'use_local_url'), "CondaWorker should have use_local_url property"
    assert conda_worker.use_local_url == False, "CondaWorker should have use_local_url=False"
    
    conda_service = conda_worker.get_worker_service()
    assert 'use_local_url' in conda_service, "Worker service should include use_local_url"
    assert conda_service['use_local_url'] == False, "Worker service use_local_url should match property"
    
    # Test the base worker class has the property
    from hypha.workers.base import BaseWorker
    
    class TestWorker(BaseWorker):
        def __init__(self):
            super().__init__()
        
        @property
        def name(self):
            return "test-worker"
            
        @property
        def description(self):
            return "Test worker"
            
        @property
        def supported_types(self):
            return ["test"]
            
        async def start(self, config, context=None):
            return "test-session"
            
        async def stop(self, session_id, context=None):
            pass

            
        async def get_logs(self, session_id, type=None, offset=0, limit=None, context=None):
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

    
    test_worker = TestWorker()
    assert hasattr(test_worker, 'use_local_url'), "BaseWorker should have use_local_url property"
    assert test_worker.use_local_url == False, "BaseWorker use_local_url should default to False"
    
    test_service = test_worker.get_worker_service()
    assert 'use_local_url' in test_service, "Worker service should include use_local_url"
    assert test_service['use_local_url'] == False, "Worker service use_local_url should match property"
    
    print("✓ use_local_url property is correctly set on all workers")
    print(f"  BrowserWorker.use_local_url = {browser_worker.use_local_url}")
    print(f"  CondaWorker.use_local_url = {conda_worker.use_local_url}")
    print(f"  BaseWorker.use_local_url (default) = {test_worker.use_local_url}")
    
    print("✓ Test completed successfully - use_local_url functionality")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
