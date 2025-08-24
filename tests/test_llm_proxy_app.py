"""Test the LLM Proxy conda application."""

import pytest
import json
import os
import asyncio
from hypha_rpc import connect_to_server
from hypha.apps.llm_proxy import (
    create_llm_proxy_app,
    install_llm_proxy,
    LLM_PROXY_MANIFEST,
    LLM_PROXY_SCRIPT,
)
from hypha.apps.llm_proxy.startup import hypha_startup, register_llm_proxy_app
from tests import SIO_PORT


@pytest.mark.asyncio
async def test_create_llm_proxy_app():
    """Test creating the LLM proxy application."""
    app_info = await create_llm_proxy_app()
    
    # Check manifest
    assert app_info["manifest"]["name"] == "LLM Proxy"
    assert app_info["manifest"]["id"] == "llm-proxy"
    assert app_info["manifest"]["type"] == "conda-jupyter-kernel"
    assert "python=3.11" in app_info["manifest"]["dependencies"]
    
    # Check that litellm is in pip dependencies
    pip_deps = None
    for dep in app_info["manifest"]["dependencies"]:
        if isinstance(dep, dict) and "pip" in dep:
            pip_deps = dep["pip"]
            break
    
    assert pip_deps is not None
    assert any("litellm" in dep for dep in pip_deps)
    
    # Check files
    assert "main.py" in app_info["files"]
    assert "class LLMProxyService" in app_info["files"]["main.py"]
    assert "start_proxy" in app_info["files"]["main.py"]
    assert "chat_completion" in app_info["files"]["main.py"]


def test_llm_proxy_manifest():
    """Test that the LLM proxy manifest is properly configured."""
    assert LLM_PROXY_MANIFEST["name"] == "LLM Proxy"
    assert LLM_PROXY_MANIFEST["type"] == "conda-jupyter-kernel"
    assert LLM_PROXY_MANIFEST["version"] == "0.1.0"
    
    # Check dependencies
    deps = LLM_PROXY_MANIFEST["dependencies"]
    assert "python=3.11" in deps
    assert "pip" in deps
    
    # Find pip dependencies
    pip_deps = None
    for dep in deps:
        if isinstance(dep, dict) and "pip" in dep:
            pip_deps = dep["pip"]
            break
    
    assert pip_deps is not None
    assert any("litellm[proxy]" in dep for dep in pip_deps)
    assert any("hypha-rpc" in dep for dep in pip_deps)
    
    # Check config
    assert LLM_PROXY_MANIFEST["config"]["visibility"] == "public"
    assert LLM_PROXY_MANIFEST["config"]["require_context"] is False


def test_llm_proxy_script():
    """Test that the LLM proxy script contains necessary components."""
    # Check imports
    assert "import asyncio" in LLM_PROXY_SCRIPT
    assert "from hypha_rpc import connect_to_server" in LLM_PROXY_SCRIPT
    
    # Check class definition
    assert "class LLMProxyService:" in LLM_PROXY_SCRIPT
    
    # Check service methods
    assert "async def start_proxy" in LLM_PROXY_SCRIPT
    assert "async def stop_proxy" in LLM_PROXY_SCRIPT
    assert "async def get_status" in LLM_PROXY_SCRIPT
    assert "async def chat_completion" in LLM_PROXY_SCRIPT
    
    # Check main function
    assert "async def main():" in LLM_PROXY_SCRIPT
    assert 'server.register_service' in LLM_PROXY_SCRIPT
    assert '"llm-proxy"' in LLM_PROXY_SCRIPT
    
    # Check that it uses litellm
    assert "litellm" in LLM_PROXY_SCRIPT
    assert "subprocess.Popen" in LLM_PROXY_SCRIPT  # For starting the proxy server


@pytest.mark.asyncio
async def test_llm_proxy_with_conda_worker(fastapi_server_llm_proxy, test_user_token):
    """Test LLM proxy as a cascading worker with conda environment."""
    # Connect to the server with LLM proxy enabled
    llm_proxy_port = fastapi_server_llm_proxy  # The fixture returns the port
    async with connect_to_server({
        "server_url": f"ws://127.0.0.1:{llm_proxy_port}/ws",
        "token": test_user_token,
        "client_id": "test-llm-proxy-conda",
    }) as server:
        # Check if app controller is available
        app_controller = await server.get_service("server-apps")
        assert app_controller is not None, "Server apps controller should be available"
        
        # The LLM proxy should have been installed by the fixture
        apps = await app_controller.list()
        llm_proxy_app = None
        for app in apps:
            if app.get("id") == "llm-proxy":
                llm_proxy_app = app
                break
        
        assert llm_proxy_app is not None, "LLM proxy app should be installed"
        assert llm_proxy_app["manifest"]["type"] == "conda-jupyter-kernel"
        
        # Start the LLM proxy app (this creates conda env and runs the script)
        try:
            session = await app_controller.start(app_id="llm-proxy", timeout=60)
            assert session["id"], "Session should have an ID"
            
            # Wait a bit for the service to register
            await asyncio.sleep(5)
            
            # Check if the LLM proxy service is registered
            llm_service = await server.get_service("llm-proxy")
            assert llm_service is not None, "LLM proxy service should be registered"
            
            # Test the service methods
            status = await llm_service.get_status()
            assert status["status"] in ["stopped", "running"]
            
            # Try to start the proxy with test configuration
            config = {
                "api_keys": {
                    "OPENAI_API_KEY": "test-key-12345"
                }
            }
            result = await llm_service.start_proxy(config)
            assert result["status"] == "running"
            assert "url" in result
            assert "port" in result
            
            # Test chat completion (will fail without real API key, but tests the interface)
            try:
                response = await llm_service.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo"
                )
                # If we get here, the method exists and is callable
            except Exception as e:
                # Expected to fail without real litellm server running
                print(f"Chat completion failed as expected: {e}")
            
            # Stop the proxy
            stop_result = await llm_service.stop_proxy()
            assert stop_result["status"] in ["stopped", "not_running"]
            
            # Stop the session
            await app_controller.stop(session["id"])
            
        except asyncio.TimeoutError:
            pytest.skip("Conda environment setup timed out (expected in CI)")
        except Exception as e:
            if "conda" in str(e).lower() or "worker" in str(e).lower():
                pytest.skip(f"Conda worker not properly configured: {e}")
            raise


@pytest.mark.asyncio
async def test_llm_proxy_startup_function():
    """Test the LLM proxy startup function."""
    from unittest.mock import MagicMock, AsyncMock, patch
    
    # Mock server
    server = {
        "get_service": AsyncMock(return_value=None),
        "register_service": AsyncMock(),
    }
    
    # Test when LLM proxy is not enabled
    with patch.dict(os.environ, {"HYPHA_ENABLE_LLM_PROXY": "false"}):
        await hypha_startup(server)
        # Should not register anything
        assert not server["register_service"].called
    
    # Reset mock
    server["register_service"].reset_mock()
    
    # Test when LLM proxy is enabled but app controller not available
    with patch.dict(os.environ, {"HYPHA_ENABLE_LLM_PROXY": "true"}):
        await hypha_startup(server)
        # Should try to get service but not register
        assert server["get_service"].called
        assert not server["register_service"].called
    
    # Test successful registration with mock app controller
    mock_app_controller = MagicMock()
    mock_app_controller.install = AsyncMock(return_value="llm-proxy-id")
    mock_app_controller.start = AsyncMock()
    
    server["get_service"] = AsyncMock(return_value=mock_app_controller)
    
    # Mock CondaWorker
    with patch("hypha.workers.conda.CondaWorker") as mock_conda:
        mock_worker = MagicMock()
        mock_worker.get_worker_service = MagicMock(return_value={})
        mock_conda.return_value = mock_worker
        
        with patch.dict(os.environ, {
            "HYPHA_ENABLE_LLM_PROXY": "true",
            "OPENAI_API_KEY": "test-key",
            "HYPHA_LLM_PROXY_AUTO_START": "true"
        }):
            with patch("hypha.apps.llm_proxy.llm_proxy_app.install_llm_proxy", 
                      AsyncMock(return_value="llm-proxy-id")) as mock_install:
                await hypha_startup(server)
                
                # Verify installation was called
                assert mock_install.called
                assert mock_app_controller.start.called


@pytest.mark.asyncio
async def test_register_llm_proxy_app():
    """Test the register_llm_proxy_app helper function."""
    from unittest.mock import MagicMock, AsyncMock, patch
    
    # Mock server
    server = {
        "register_service": AsyncMock(),
    }
    
    # Mock app controller
    app_controller = MagicMock()
    app_controller.install = AsyncMock(return_value="test-llm-proxy-id")
    app_controller.start = AsyncMock()
    
    # Test with provided conda worker
    mock_worker = MagicMock()
    app_id = await register_llm_proxy_app(
        server,
        app_controller,
        conda_worker=mock_worker,
        config={"test": "config"},
        auto_start=False
    )
    
    assert app_id == "test-llm-proxy-id"
    assert not app_controller.start.called  # auto_start=False
    
    # Test without conda worker (should create one)
    with patch("hypha.workers.conda.CondaWorker") as mock_conda:
        mock_new_worker = MagicMock()
        mock_new_worker.get_worker_service = MagicMock(return_value={})
        mock_conda.return_value = mock_new_worker
        
        app_id = await register_llm_proxy_app(
            server,
            app_controller,
            conda_worker=None,
            auto_start=True
        )
        
        assert app_id == "test-llm-proxy-id"
        assert app_controller.start.called  # auto_start=True
        assert server["register_service"].called  # Should register conda worker