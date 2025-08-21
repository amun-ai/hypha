"""Test unified authentication integration with templates."""

import pytest
import asyncio
import json
import subprocess
import time
import os
import sys
import tempfile
from pathlib import Path
from hypha_rpc import connect_to_server, login
import requests


@pytest.mark.asyncio 
async def test_unified_auth_with_templates(tmp_path):
    """Test that templates work with unified authentication approach."""
    
    # Start server with local auth enabled
    port = 39777
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "hypha.server", 
         f"--port={port}", 
         "--enable-local-auth",
         "--enable-s3",
         "--reset-redis",
         "--start-minio-server",
         "--minio-root-user=minioadmin",
         "--minio-root-password=minioadmin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "HYPHA_LOGLEVEL": "INFO"}
    )
    
    server_url = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}/ws"
    
    # Wait for server to start with longer timeout for MinIO initialization
    max_retries = 60  # Increased timeout for MinIO
    server_ready = False
    last_error = None
    config = None
    
    for i in range(max_retries):
        try:
            # First check if the server process is still running
            if server_proc.poll() is not None:
                # Server process has exited, capture output for debugging
                stdout, stderr = server_proc.communicate(timeout=1)
                error_msg = f"Server process exited with code {server_proc.returncode}"
                if stderr:
                    error_msg += f"\nStderr: {stderr.decode('utf-8', errors='ignore')[-1000:]}"
                raise RuntimeError(error_msg)
            
            # Try to access the config endpoint
            response = requests.get(f"{server_url}/assets/config.json", timeout=2)
            if response.status_code == 200:
                config = response.json()
                # Also verify that the health endpoint is ready
                health_response = requests.get(f"{server_url}/health/readiness", timeout=2)
                if health_response.status_code == 200:
                    server_ready = True
                    break
        except requests.RequestException as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
            if "Server process exited" in str(e):
                raise  # Re-raise if server crashed
        
        time.sleep(1)
    
    if not server_ready:
        # Try to get more diagnostic info before failing
        try:
            server_proc.terminate()
            stdout, stderr = server_proc.communicate(timeout=5)
            error_details = f"Server failed to start after {max_retries} seconds."
            if last_error:
                error_details += f"\nLast error: {last_error}"
            if stderr:
                error_details += f"\nServer stderr (last 500 chars): {stderr.decode('utf-8', errors='ignore')[-500:]}"
            raise TimeoutError(error_details)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            raise TimeoutError(f"Server failed to start and didn't terminate gracefully. Last error: {last_error}")
    
    # Test that config contains login_service_url (for unified auth)
    assert "login_service_url" in config, "login_service_url should be in config"
    assert "hypha-login" in config["login_service_url"], "Should use hypha-login service"
    
    # Test that templates are served correctly
    response = requests.get(f"{server_url}/", timeout=5)
    assert response.status_code == 200
    html_content = response.text
    
    # Check that the template uses hypha-rpc instead of Auth0
    assert "hypha-rpc-websocket.js" in html_content, "Should load hypha-rpc library"
    assert "hyphaWebsocketClient.login" in html_content, "Should use hypha-rpc login"
    assert "auth0" not in html_content.lower(), "Should not contain Auth0 references"
    
    # Test the login flow using hypha-rpc
    async with connect_to_server({
        "client_id": "test-unified-auth",
        "server_url": ws_url,
    }) as api:
        # Get the hypha-login service
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_found = any("hypha-login" in sid for sid in service_ids)
        assert hypha_login_found, "hypha-login service should be available"
        
        # Test signup
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        result = await login_service.signup(
            name="Test User",
            email="test@example.com",
            password="testpass123"
        )
        assert result["success"] is True, f"Signup failed: {result.get('error')}"
        
        # Test the unified login API
        # This simulates what the template JavaScript does
        login_session = await login_service.start(expires_in=3600)
        assert "login_url" in login_session
        assert "key" in login_session
        
        # Simulate direct login (as would happen in the popup)
        login_result = await login_service.login(
            email="test@example.com",
            password="testpass123",
            key=login_session["key"]
        )
        assert login_result["success"] is True
        assert "token" in login_result
        
        # Check that the session was updated
        token = await login_service.check(key=login_session["key"], timeout=0)
        assert token is not None, "Token should be available after login"
        
    # Test that the workspace template also works
    response = requests.get(f"{server_url}/ws-user-test", timeout=5)
    assert response.status_code == 200
    ws_html = response.text
    
    # Check that workspace template has localStorage fallback for tokens
    assert "localStorage.getItem('hypha_token')" in ws_html, "Should have localStorage fallback"
    
    server_proc.terminate()
    server_proc.wait()


@pytest.mark.asyncio
async def test_login_function_compatibility():
    """Test that the login() function works with local auth."""
    
    port = 39778
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "hypha.server",
         f"--port={port}",
         "--enable-local-auth", 
         "--enable-s3",
         "--reset-redis",
         "--start-minio-server",
         "--minio-root-user=minioadmin",
         "--minio-root-password=minioadmin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "HYPHA_LOGLEVEL": "INFO"}
    )
    
    server_url = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}/ws"
    
    # Wait for server to start with longer timeout for MinIO initialization
    max_retries = 60  # Increased from 30 to 60 seconds
    server_ready = False
    last_error = None
    
    for i in range(max_retries):
        try:
            # First check if the server process is still running
            if server_proc.poll() is not None:
                # Server process has exited, capture output for debugging
                stdout, stderr = server_proc.communicate(timeout=1)
                error_msg = f"Server process exited with code {server_proc.returncode}"
                if stderr:
                    error_msg += f"\nStderr: {stderr.decode('utf-8', errors='ignore')[-1000:]}"  # Last 1000 chars
                raise RuntimeError(error_msg)
            
            # Try to access the config endpoint
            response = requests.get(f"{server_url}/assets/config.json", timeout=2)
            if response.status_code == 200:
                # Also verify that the health endpoint is ready
                health_response = requests.get(f"{server_url}/health/readiness", timeout=2)
                if health_response.status_code == 200:
                    server_ready = True
                    break
        except requests.RequestException as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)
            if "Server process exited" in str(e):
                raise  # Re-raise if server crashed
        
        time.sleep(1)
    
    if not server_ready:
        # Try to get more diagnostic info before failing
        try:
            server_proc.terminate()
            stdout, stderr = server_proc.communicate(timeout=5)
            error_details = f"Server failed to start after {max_retries} seconds."
            if last_error:
                error_details += f"\nLast error: {last_error}"
            if stderr:
                error_details += f"\nServer stderr (last 500 chars): {stderr.decode('utf-8', errors='ignore')[-500:]}"
            raise TimeoutError(error_details)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            raise TimeoutError(f"Server failed to start and didn't terminate gracefully. Last error: {last_error}")
    
    # First create a user
    async with connect_to_server({
        "client_id": "test-create-user",
        "server_url": ws_url,
    }) as api:
        services = await api.list_services("public")
        service_ids = [s["id"] for s in services]
        hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
        login_service = await api.get_service(hypha_login_service_id)
        
        await login_service.signup(
            name="API Test User",
            email="apitest@example.com",
            password="apipass123"
        )
    
    # Now test the login() function with a mock callback
    login_completed = False
    received_token = None
    
    async def mock_login_callback(context):
        """Mock callback that simulates automatic login."""
        nonlocal login_completed, received_token
        
        # Extract key from login URL
        login_url = context.get("login_url", "")
        key_start = login_url.find("key=")
        if key_start != -1:
            key = login_url[key_start + 4:].split("&")[0]
            
            # Connect and login directly
            async with connect_to_server({
                "client_id": "test-login-callback",
                "server_url": ws_url,
            }) as api:
                services = await api.list_services("public")
                service_ids = [s["id"] for s in services]
                hypha_login_service_id = next(sid for sid in service_ids if "hypha-login" in sid)
                login_service = await api.get_service(hypha_login_service_id)
                
                # Perform login
                result = await login_service.login(
                    email="apitest@example.com",
                    password="apipass123",
                    key=key
                )
                
                if result["success"]:
                    received_token = result["token"]
                    login_completed = True
                else:
                    print("Failed to login:", result.get("error"))

    # Test the unified login function
    token = await login({
        "server_url": ws_url,
        "login_callback": mock_login_callback,
        "login_timeout": 10
    })
    
    assert token is not None, "Should receive a token"
    assert login_completed is True, "Login should have completed"
    assert received_token == token, "Token should match"
    
    # Verify the token works
    async with connect_to_server({
        "client_id": "test-with-token",
        "server_url": ws_url,
        "token": token
    }) as api:
        # Should be able to connect with the token
        # Parse the token to verify user info
        user_info = await api.parse_token(token)
        assert user_info["email"] == "apitest@example.com"
        assert user_info["is_anonymous"] is False
        
        # Also verify we can list services
        services = await api.list_services("public")
        assert len(services) > 0, "Should be able to list services with token"
    
    server_proc.terminate()
    server_proc.wait()


if __name__ == "__main__":
    # Allow running directly for debugging
    asyncio.run(test_unified_auth_with_templates(Path("/tmp")))
    asyncio.run(test_login_function_compatibility())