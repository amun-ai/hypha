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
    from tests import SIO_PORT
    import subprocess
    import sys
    import os
    import time
    
    # Start server with custom auth that includes login service
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={39001}",
            "--redis-uri=redis://localhost:6379/12",
            "--startup-functions=./tests/custom_auth_with_login_test.py:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        server_url = f"ws://127.0.0.1:39001/ws"
        
        # Test connection with custom auth
        async with connect_to_server(
            {
                "client_id": "test-login-service",
                "server_url": server_url,
            }
        ) as api:
            # Try to access the custom login service
            login_service = await api.get_service("public/hypha-login")
            
            # Start a login session
            login_info = await login_service.start()
            assert "key" in login_info
            assert "login_url" in login_info
            
            # Simulate reporting login completion
            await login_service.report(
                key=login_info["key"],
                user_id="test-user",
                email="test@custom-login.com",
                workspace="test-workspace"
            )
            
            # Check login status
            token = await login_service.check(key=login_info["key"], timeout=1)
            assert token.startswith("CUSTOM_LOGIN:"), f"Expected custom login token, got: {token}"
            
            # Verify we can use the token
            async with connect_to_server(
                {
                    "client_id": "test-with-custom-token",
                    "server_url": server_url,
                    "token": token,
                }
            ) as auth_api:
                # Should be able to list services
                services = await auth_api.list_services("public")
                assert len(services) > 0
    
    finally:
        server_process.terminate()
        server_process.wait()


@pytest.mark.asyncio
async def test_register_auth_partial_handlers(fastapi_server):
    """Test register_auth_service with only some handlers provided."""
    # Create a test module with partial handlers
    test_module_content = '''
import time
from hypha.core import UserInfo, UserPermission
from hypha.core.auth import create_scope

def partial_generate_token(user_info: UserInfo, expires_in: int) -> str:
    """Generate a partial test token."""
    return f"PARTIAL:{user_info.id}:{int(time.time() + expires_in)}"

async def partial_parse_token(token: str) -> UserInfo:
    """Parse partial test token."""
    if token.startswith("PARTIAL:"):
        parts = token.split(":")
        user_id = parts[1]
        return UserInfo(
            id=user_id,
            email=f"{user_id}@partial.test",
            roles=["user"],
            scope=create_scope(workspaces={"partial-ws": UserPermission.admin}),
        )
    else:
        from hypha.core.auth import _parse_token
        return _parse_token(token)

async def hypha_startup(server):
    """Register partial auth handlers."""
    # Only provide token functions, not login handlers
    await server.register_auth_service(
        parse_token=partial_parse_token,
        generate_token=partial_generate_token,
    )
    print("Partial auth registered")
'''
    
    # Write test module
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_module_content)
        test_module_path = f.name
    
    import subprocess
    import sys
    import os
    import time
    
    # Start server with partial auth
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={39002}",
            "--redis-uri=redis://localhost:6379/13",
            f"--startup-functions={test_module_path}:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        server_url = f"ws://127.0.0.1:39002/ws"
        
        # Test connection
        async with connect_to_server(
            {
                "client_id": "test-partial-auth",
                "server_url": server_url,
            }
        ) as api:
            # Generate a token with partial auth
            workspace_name = api.config.workspace
            token = await api.generate_token(
                config={
                    "workspace": workspace_name,
                    "expires_in": 3600
                }
            )
            
            # Verify it's our partial format
            assert token.startswith("PARTIAL:"), f"Expected partial token format, got: {token}"
            
            # Default login service should still be available
            login_service = await api.get_service("public/hypha-login")
            assert login_service is not None
    
    finally:
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
    return {"custom": True, "key": "test-key"}

async def custom_check(key, timeout=180, profile=False):
    """Custom check handler."""
    if key == "test-key":
        return "custom-token-123"
    raise ValueError("Invalid key")

async def custom_report(key, token, **kwargs):
    """Custom report handler."""
    return {"reported": True}

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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_module_content)
        test_module_path = f.name
    
    import subprocess
    import sys
    import os
    import time
    
    # Start server with custom login
    server_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            "--host=0.0.0.0",
            f"--port={39003}",
            "--redis-uri=redis://localhost:6379/14",
            f"--startup-functions={test_module_path}:hypha_startup",
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        server_url = f"ws://127.0.0.1:39003/ws"
        
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
            assert result["custom"] is True
            assert result["key"] == "test-key"
            
            # Test custom check handler
            token = await login_service.check(key="test-key", timeout=1)
            assert token == "custom-token-123"
            
            # Test custom report handler
            report_result = await login_service.report(
                key="test-key",
                token="some-token"
            )
            assert report_result["reported"] is True
    
    finally:
        server_process.terminate()
        server_process.wait()
        os.unlink(test_module_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])