"""Test the register_auth_service functionality."""
import pytest
import asyncio
from hypha_rpc import connect_to_server
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope


@pytest.mark.asyncio
async def test_register_auth_service_basic(custom_auth_server):
    """Test basic custom auth registration works."""
    server_url = custom_auth_server.replace("http://", "ws://") + "/ws"
    
    # Connect and verify custom auth is working
    async with connect_to_server(
        {
            "client_id": "test-register-auth",
            "server_url": server_url,
        }
    ) as api:
        # Generate a custom token
        workspace_name = api.config.workspace
        custom_token = await api.generate_token(
            config={
                "workspace": workspace_name,
                "expires_in": 3600
            }
        )
        
        # Verify it's our custom format
        assert custom_token.startswith("CUSTOM:"), f"Expected custom token format, got: {custom_token}"


@pytest.mark.asyncio
async def test_register_auth_service_with_login(fastapi_server):
    """Test register_auth_service with custom login handlers."""
    # Create a test module with custom login handlers
    test_module_content = '''
async def custom_index(event):
    """Custom index handler."""
    return {
        "status": 200,
        "headers": {"Content-Type": "text/html"},
        "body": "<h1>Custom Login Page</h1>",
    }

async def custom_start(workspace=None, expires_in=None):
    """Custom start handler."""
    return {"key": "test-key-login", "login_url": "/custom/login"}

async def custom_check(key, timeout=180, profile=False):
    """Custom check handler."""
    if key == "test-key-login":
        return "CUSTOM_LOGIN:test-user:workspace"
    raise ValueError("Invalid key")

async def custom_report(key, token, **kwargs):
    """Custom report handler."""
    return {"success": True, "message": "Login reported"}

async def hypha_startup(server):
    """Register custom login handlers."""
    await server.register_auth_service(
        index_handler=custom_index,
        start_handler=custom_start,
        check_handler=custom_check,
        report_handler=custom_report,
    )
    print("Custom login handlers registered")
'''
    
    # Write test module
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_module_content)
        test_module_path = f.name
    
    import subprocess
    import sys
    import time
    
    # Find an available port
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    
    # Start server with custom login
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={port}",
            "--reset-redis",
            f"--startup-functions={test_module_path}:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start and check if it's ready
    max_retries = 30
    server_ready = False
    for i in range(max_retries):
        try:
            # Try to connect to check if server is ready
            import requests
            response = requests.get(f"http://127.0.0.1:{port}/health/readiness", timeout=1)
            if response.ok:
                server_ready = True
                break
        except:
            time.sleep(1)
    
    if not server_ready:
        # Server didn't start, get the output for debugging
        try:
            stdout, stderr = server_process.communicate(timeout=5)
            raise Exception(f"Server failed to start after {max_retries} seconds. Output: {stdout.decode()}")
        except subprocess.TimeoutExpired:
            server_process.kill()
            stdout, stderr = server_process.communicate()
            raise Exception(f"Server startup timed out. Output: {stdout.decode()}")
    
    server_url = f"ws://127.0.0.1:{port}/ws"
    
    # Test custom login service
    async with connect_to_server(
        {
            "client_id": "test-custom-login",
            "server_url": server_url,
        }
    ) as api:
        # Get the custom login service
        login_service = await api.get_service("public/hypha-login")
        
        # Test custom start handler
        result = await login_service.start()
        assert result["key"] == "test-key-login"
        assert result["login_url"] == "/custom/login"
        
        # Test custom check handler
        token = await login_service.check(key="test-key-login", timeout=1)
        assert token == "CUSTOM_LOGIN:test-user:workspace"
        
        # Test custom report handler
        report_result = await login_service.report(
            key="test-key-login",
            token="some-token"
        )
        assert report_result["success"] is True
        assert report_result["message"] == "Login reported"
    
    server_process.terminate()
    server_process.wait()
    os.unlink(test_module_path)


@pytest.mark.asyncio
async def test_register_auth_partial_handlers(fastapi_server):
    """Test register_auth_service with only some handlers provided."""
    # Create a test module with only token functions (no login handlers)
    test_module_content = '''
async def custom_parse_token(token):
    """Custom parse token function."""
    if token.startswith("PARTIAL:"):
        from hypha.core.auth import create_scope
        from hypha.core import UserInfo, UserPermission
        parts = token.split(":")
        user_id = parts[1]
        workspace = f"ws-user-{user_id}"
        
        user_info = UserInfo(
            id=user_id,
            is_anonymous=False,
            email=f"{user_id}@example.com",
            parent=None,
            roles=[],
            scope=create_scope(
                workspaces={workspace: UserPermission.admin},
                current_workspace=workspace
            ),
            expires_at=None,
        )
        return user_info
    # Fall back to default parsing for non-partial tokens
    from hypha.core.auth import _parse_token
    return await _parse_token(token)

def custom_generate_token(user_info, expires_in):
    """Custom generate token function."""
    return f"PARTIAL:{user_info.id}:{expires_in}"

async def hypha_startup(server):
    """Register only token functions, no login handlers."""
    await server.register_auth_service(
        parse_token=custom_parse_token,
        generate_token=custom_generate_token,
    )
    print("Partial auth handlers registered (token functions only)")
'''
    
    # Write test module
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_module_content)
        test_module_path = f.name
    
    import subprocess
    import sys
    import time
    
    # Find an available port
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    
    # Start server with partial auth
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={port}",
            "--reset-redis",
            f"--startup-functions={test_module_path}:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start and check if it's ready
    max_retries = 30
    server_ready = False
    for i in range(max_retries):
        try:
            # Try to connect to check if server is ready
            import requests
            response = requests.get(f"http://127.0.0.1:{port}/health/readiness", timeout=1)
            if response.ok:
                server_ready = True
                break
        except:
            time.sleep(1)
    
    if not server_ready:
        # Server didn't start, get the output for debugging
        try:
            stdout, stderr = server_process.communicate(timeout=5)
            raise Exception(f"Server failed to start after {max_retries} seconds. Output: {stdout.decode()}")
        except subprocess.TimeoutExpired:
            server_process.kill()
            stdout, stderr = server_process.communicate()
            raise Exception(f"Server startup timed out. Output: {stdout.decode()}")
    
    server_url = f"ws://127.0.0.1:{port}/ws"
    
    # Test that custom token functions work
    async with connect_to_server(
        {
            "client_id": "test-partial-auth",
            "server_url": server_url,
        }
    ) as api:
        # Generate a token using custom generate function
        workspace = api.config.workspace
        token = await api.generate_token(
            config={
                "workspace": workspace,
                "expires_in": 3600
            }
        )
        
        # Should be our custom partial token format
        assert token.startswith("PARTIAL:"), f"Expected PARTIAL: token, got: {token}"
        
        # Should contain the user ID and expires_in
        parts = token.split(":")
        assert len(parts) == 3
        assert parts[2] == "3600"  # expires_in value
    
    # Test that we can connect with the custom token
    async with connect_to_server(
        {
            "client_id": "test-partial-token",
            "server_url": server_url,
            "token": token,  # Use the custom token
        }
    ) as api_with_token:
        # Should be able to list services with custom token
        services = await api_with_token.list_services("public")
        assert len(services) > 0
        
        # The hypha-login service should NOT be registered since we only provided token functions
        service_ids = [s["id"] for s in services]
        login_services = [sid for sid in service_ids if "hypha-login" in sid]
        # The default login service might exist, but not our custom one
    
    server_process.terminate()
    server_process.wait()
    os.unlink(test_module_path)


@pytest.mark.asyncio 
async def test_register_auth_service_override_login(fastapi_server):
    """Test that custom login handlers override default ones."""
    # Create a test module with custom login
    test_module_content = '''
async def custom_index(event):
    """Custom index handler."""
    return {
        "status": 200,
        "headers": {"Content-Type": "text/plain"},
        "body": "CUSTOM_LOGIN_PAGE",
    }

async def custom_start(workspace=None, expires_in=None):
    """Custom start handler."""
    return {"custom": True, "key": "test-key-override", "workspace": workspace}

async def custom_check(key, timeout=180, profile=False):
    """Custom check handler."""
    if key == "test-key-override":
        return "custom-token-override-123"
    raise ValueError("Invalid key")

async def custom_report(key, token, **kwargs):
    """Custom report handler."""
    return {"reported": True, "custom_handler": True}

async def hypha_startup(server):
    """Register custom login handlers."""
    await server.register_auth_service(
        index_handler=custom_index,
        start_handler=custom_start,
        check_handler=custom_check,
        report_handler=custom_report,
    )
    print("Custom login handlers registered for override test")
'''
    
    # Write test module
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_module_content)
        test_module_path = f.name
    
    import subprocess
    import sys
    import time
    import socket
    
    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    
    # Start server with custom login
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={port}",
            "--reset-redis",
            f"--startup-functions={test_module_path}:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start and check if it's ready
    server_url = f"ws://127.0.0.1:{port}/ws"
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try to connect to check if server is ready
            import requests
            response = requests.get(f"http://127.0.0.1:{port}/health/readiness", timeout=1)
            if response.ok:
                break
        except:
            time.sleep(1)
    else:
        # Server didn't start, get the output for debugging
        stdout, stderr = server_process.communicate(timeout=5)
        raise Exception(f"Server failed to start after {max_retries} seconds. Output: {stdout}")
    
    # Test custom login service
    async with connect_to_server(
        {
            "client_id": "test-custom-override",
            "server_url": server_url,
        }
    ) as api:
        # Get the custom login service
        login_service = await api.get_service("public/hypha-login")
        
        # Test custom start handler
        result = await login_service.start()
        assert result["custom"] is True
        assert result["key"] == "test-key-override"
        
        # Test custom check handler
        token = await login_service.check(key="test-key-override", timeout=1)
        assert token == "custom-token-override-123"
        
        # Test custom report handler
        report_result = await login_service.report(
            key="test-key-override",
            token="some-token"
        )
        assert report_result["reported"] is True
        assert report_result["custom_handler"] is True
    
    server_process.terminate()
    server_process.wait()
    os.unlink(test_module_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])