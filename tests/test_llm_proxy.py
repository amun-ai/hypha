"""Tests for the LLM proxy worker."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hypha.workers.llm_proxy import LLMProxyWorker
from hypha.core import UserInfo


@pytest.fixture
def mock_store():
    """Create a mock store."""
    store = MagicMock()
    # Create async context manager for get_workspace_interface
    mock_workspace_api = AsyncMock()
    mock_workspace_api.register_service = AsyncMock()
    
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__ = AsyncMock(return_value=mock_workspace_api)
    mock_context_manager.__aexit__ = AsyncMock(return_value=None)
    
    store.get_workspace_interface = MagicMock(return_value=mock_context_manager)
    return store


@pytest.fixture
def mock_workspace_manager():
    """Create a mock workspace manager."""
    return MagicMock()


@pytest.fixture
def llm_worker(mock_store, mock_workspace_manager):
    """Create an LLM proxy worker instance."""
    worker = LLMProxyWorker(mock_store, mock_workspace_manager, "test-llm-worker")
    # The worker stores these in _store attribute
    worker._store = mock_store
    return worker


@pytest.mark.asyncio
async def test_llm_worker_start_stop(llm_worker):
    """Test starting and stopping the LLM worker."""
    # Create a valid app manifest (this is what ServerAppController passes to compile)
    app_manifest = {
        "config": {
            "model_list": [
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "mock_response": "Test response"
                    }
                }
            ],
            "litellm_settings": {}
        }
    }
    
    context = {
        "ws": "test-workspace",
        "from": "test-client",
        "user": {
            "id": "test-user",
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    # First compile the app to get the session config
    compiled_manifest, files = await llm_worker.compile(app_manifest, [], context=context)
    
    # The start method expects config with connection info and the compiled manifest
    start_config = {
        "manifest": compiled_manifest,
        "client_id": "test-client",
        "app_id": "test-app",
        "workspace": "test-workspace",
        "server_url": "ws://localhost:9527/ws",  # mock server URL
        "token": "test-token",
    }
    
    # Mock the connect_to_server call since we don't have a real server in unit tests
    async def mock_connect_to_server(config):
        """Mock async connect_to_server function."""
        mock_client = AsyncMock()
        mock_service_info = {"id": "test-service-id"}
        mock_client.register_service = AsyncMock(return_value=mock_service_info)
        return mock_client
    
    with patch('hypha_rpc.connect_to_server', side_effect=mock_connect_to_server):
        # Start the session with the config
        session_id = await llm_worker.start(start_config, context=context)
        
        assert session_id is not None
        assert session_id in llm_worker._sessions
    
    # Stop the session
    await llm_worker.stop(session_id)
    assert session_id not in llm_worker._sessions


@pytest.mark.asyncio
async def test_llm_worker_compile(llm_worker):
    """Test compiling an LLM proxy session."""
    # Create a test manifest (what ServerAppController passes)
    manifest = {
        "config": {
            "model_list": [
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "test-key"
                    }
                }
            ],
            "litellm_settings": {
                "debug": False,
                "drop_params": True
            }
        }
    }
    
    context = {
        "ws": "test-workspace", 
        "from": "test-client",
        "user": {
            "id": "test-user",
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    result, files = await llm_worker.compile(manifest, [], context=context)
    
    # Should return the manifest with session_id added
    assert "session_id" in result
    assert result["config"] == manifest["config"]
    
    # Session should be stored in worker
    session_id = result["session_id"]
    assert session_id in llm_worker._sessions


@pytest.mark.asyncio
async def test_llm_worker_compile_no_manifest(llm_worker):
    """Test compiling without a manifest raises an error."""
    config = {}
    context = {
        "ws": "test-workspace",
        "from": "test-client", 
        "user": {
            "id": "test-user",
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    with pytest.raises(ValueError, match="No manifest found"):
        await llm_worker.compile(config, context=context)


@pytest.mark.asyncio
async def test_llm_worker_compile_no_model_list(llm_worker):
    """Test compiling without model_list raises an error."""
    manifest = {
        "config": {
            "litellm_settings": {}
        }
    }
    context = {
        "ws": "test-workspace",
        "from": "test-client",
        "user": {
            "id": "test-user", 
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    with pytest.raises(ValueError, match="No model_list found"):
        await llm_worker.compile(manifest, [], context=context)


@pytest.mark.asyncio
async def test_llm_worker_execute(llm_worker):
    """Test executing a session operation (LLM proxy worker doesn't have execute method)."""
    # The LLM proxy worker doesn't have an execute method - it operates via ASGI
    # This test should verify that the worker correctly indicates it doesn't support execute
    api = llm_worker.get_service_api()
    
    # The API should not include an execute method
    assert "execute" not in api
    
    # But it should have the other expected methods
    assert "compile" in api
    assert "start" in api
    assert "stop" in api
    assert "get_logs" in api


@pytest.mark.asyncio
async def test_llm_worker_execute_invalid_session(llm_worker):
    """Test that the worker behaves correctly for invalid sessions."""
    # Since the worker doesn't have execute, test get_logs with invalid session
    context = {
        "ws": "test-workspace",
        "from": "test-client",
        "user": {
            "id": "test-user",
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    # Should return empty logs for non-existent session
    result = await llm_worker.get_logs("invalid-session-id", context=context)
    assert result["items"] == []
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_llm_worker_get_logs(llm_worker):
    """Test getting logs from the LLM worker."""
    # Create a session with some logs
    session_id = "test-session"
    llm_worker._sessions[session_id] = {
        "logs": ["Starting LLM proxy", "Model loaded successfully", "Error: Connection failed"]
    }
    
    context = {
        "ws": "test-workspace",
        "from": "test-client",
        "user": {
            "id": "test-user",
            "roles": ["user"],
            "is_anonymous": False,
        }
    }
    
    # Test getting all logs
    result = await llm_worker.get_logs(session_id, context=context)
    assert result["total"] == 3
    assert len(result["items"]) == 3
    assert result["items"][0]["content"] == "Starting LLM proxy"
    assert result["items"][2]["type"] == "error"  # Should detect error logs
    
    # Test with limit
    result = await llm_worker.get_logs(session_id, limit=2, context=context)
    assert len(result["items"]) == 2
    assert result["limit"] == 2
    
    # Test with offset
    result = await llm_worker.get_logs(session_id, offset=1, context=context)
    assert len(result["items"]) == 2
    assert result["items"][0]["content"] == "Model loaded successfully"


@pytest.mark.asyncio
async def test_llm_worker_cleanup_sessions(llm_worker):
    """Test session cleanup."""
    # Create test sessions with different last_access times
    current_time = asyncio.get_event_loop().time()
    
    # Active session (accessed recently)
    llm_worker._sessions["active"] = {
        "last_access": current_time - 100,  # 100 seconds ago
        "info": {"session_id": "active", "status": "running"}
    }
    
    # Inactive session (not accessed for > 5 minutes)
    llm_worker._sessions["inactive"] = {
        "last_access": current_time - 400,  # 400 seconds ago
        "info": {"session_id": "inactive", "status": "running"}
    }
    
    # Mock the cleanup method to test it directly
    await llm_worker._cleanup_session("inactive")
    
    # Verify inactive session was removed
    assert "inactive" not in llm_worker._sessions
    assert "active" in llm_worker._sessions


@pytest.mark.asyncio
async def test_llm_middleware():
    """Test the LLM routing middleware."""
    from hypha.llm import LLMRoutingMiddleware
    
    # Create mock app and store
    mock_app = AsyncMock()
    mock_store = MagicMock()
    
    # Create middleware with correct signature
    middleware = LLMRoutingMiddleware(mock_app, base_path="/api", store=mock_store)
    
    # Test that middleware was created successfully
    assert middleware.app == mock_app
    assert middleware.store == mock_store
    assert middleware.base_path == "/api"


@pytest.mark.asyncio
async def test_llm_service_api(llm_worker):
    """Test the service API definition."""
    api = llm_worker.get_service_api()
    
    assert api["id"] == "test-llm-worker"
    assert api["name"] == "LLM Proxy Worker"
    assert "compile" in api
    # The LLM proxy worker doesn't have execute - it uses ASGI serving instead
    assert "execute" not in api
    assert "start" in api
    assert "stop" in api
    assert "get_logs" in api
    assert api["config"]["visibility"] == "public"
    assert api["config"]["require_context"] is True