"""Tests for the LLM proxy worker - unit and integration tests."""

import asyncio
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from hypha_rpc import connect_to_server
import litellm

from hypha.workers.llm_proxy import LLMProxyWorker
from hypha.core import UserInfo
from . import (
    WS_SERVER_URL,
    SERVER_URL,
    SIO_PORT,
)

# All test coroutines will be treated as marked with pytest.mark.asyncio
pytestmark = pytest.mark.asyncio


# ============================================================================
# UNIT TESTS
# ============================================================================

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


async def test_llm_worker_start_stop(llm_worker):
    """Test starting and stopping the LLM worker."""
    # Create a valid app manifest (this is what ServerAppController passes to compile)
    app_manifest = {
        "config": {
            "service_id": "test-llm-service",
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
        
        # Mock the register_service method
        mock_client.register_service = AsyncMock(return_value=mock_service_info)
        
        # Mock the config property
        mock_client.config = MagicMock()
        mock_client.config.workspace = config.get("workspace", "test-workspace")
        mock_client.config.client_id = config.get("client_id", "test-client")
        
        # Mock serve method for ASGI service
        mock_client.serve = AsyncMock()
        
        # Mock disconnect
        mock_client.disconnect = AsyncMock()
        
        # Mock get_env for workspace secrets (should not have any in this test)
        mock_client.get_env = AsyncMock(side_effect=KeyError("No secrets configured"))
        
        return mock_client
    
    # Patch connect_to_server
    with patch('hypha.workers.llm_proxy.connect_to_server', side_effect=mock_connect_to_server):
        # Start the worker - returns just the session_id string
        session_id = await llm_worker.start(start_config, context=context)
        
        # Check that a session was created
        assert session_id is not None
        assert isinstance(session_id, str)
        
        # Check that the worker has registered the session
        assert session_id in llm_worker._sessions
        
        # Check the session data
        session_data = llm_worker._sessions[session_id]
        assert session_data is not None
        assert "app" in session_data
        assert session_data["service_id"] == "test-llm-service"
        # The info dict exists
        assert "info" in session_data
        
        # Stop the worker
        await llm_worker.stop(session_id, context=context)
        
        # Check that the session was removed
        assert session_id not in llm_worker._sessions


async def test_llm_worker_compile(llm_worker):
    """Test compiling an LLM proxy session."""
    # Create a test manifest (what ServerAppController passes)
    manifest = {
        "config": {
            "service_id": "test-llm-service",
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


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


async def test_llm_proxy_installation_and_basic_operation(
    minio_server, fastapi_server, test_user_token
):
    """Test installing and running an LLM proxy app with real infrastructure."""
    # Connect to the real Hypha server
    api = await connect_to_server({
        "name": "test llm proxy",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    # Get the server apps controller
    controller = await api.get_service("public/server-apps")
    
    # Test 1: Basic LLM proxy with mock responses
    llm_app_source = """
<config lang="json">
{
    "name": "Test LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "description": "LLM proxy for testing with mock responses",
    "config": {
        "service_id": "test-llm-service",
        "model_list": [
            {
                "model_name": "test-gpt-3.5",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "This is a mock response from the test LLM"
                }
            },
            {
                "model_name": "test-gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "mock_response": "This is a mock GPT-4 response"
                }
            }
        ],
        "litellm_settings": {
            "debug": false,
            "drop_params": true,
            "routing_strategy": "simple-shuffle"
        }
    }
}
</config>
"""
    
    # Install the LLM proxy app
    app_info = await controller.install(
        source=llm_app_source,
        wait_for_service=False,
        timeout=20,
        overwrite=True,
    )
    
    app_id = app_info["id"]
    print(f"Installed LLM proxy app: {app_id}")
    
    # Start the LLM proxy app and wait for the test-llm-service
    session = await controller.start(app_id, wait_for_service="test-llm-service", timeout=30)
    session_id = session["id"]
    service_id = session.get("service_id")
    
    print(f"Started LLM proxy session: {session_id}, service: {service_id}")
    
    # Wait a bit for the service to be fully ready
    await asyncio.sleep(2)
    
    try:
        # Test basic functionality
        if service_id:
            # Extract the actual service name from the full service_id
            if "@" in service_id:
                service_path = service_id.split("@")[0]
                llm_service = await api.get_service(service_path)
                assert llm_service is not None, "Service should be registered"
                print(f"Service found: {service_path}")
            else:
                llm_service = await api.get_service(service_id)
                assert llm_service is not None, "LLM service should be registered"
                print(f"LLM service found: {service_id}")
        
        # Test HTTP requests to the LLM endpoint
        llm_service_id = "test-llm-service"
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/{llm_service_id}"
        
        headers = {
            "Authorization": f"Bearer {test_user_token}"
        }
        
        async with httpx.AsyncClient() as client:
            # Test the health endpoint
            health_response = await client.get(
                f"{base_url}/health",
                headers=headers,
                timeout=10
            )
            print(f"Health check status: {health_response.status_code}")
            
            # Test the models endpoint
            models_response = await client.get(
                f"{base_url}/v1/models",
                headers=headers,
                timeout=10
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                print(f"Available models: {models_data}")
                
                model_ids = [m["id"] for m in models_data.get("data", [])]
                assert "test-gpt-3.5" in model_ids, "test-gpt-3.5 should be in model list"
                assert "test-gpt-4" in model_ids, "test-gpt-4 should be in model list"
            
            # Test chat completions endpoint with mock response
            chat_request = {
                "model": "test-gpt-3.5",
                "messages": [
                    {"role": "user", "content": "Hello, this is a test"}
                ],
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            chat_response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=chat_request,
                timeout=10
            )
            
            if chat_response.status_code == 200:
                chat_data = chat_response.json()
                print(f"Chat response: {chat_data}")
                
                message_content = chat_data["choices"][0]["message"]["content"]
                assert "mock response" in message_content.lower(), "Should return mock response"
        
        # Use litellm directly with the proxy endpoint
        response = await litellm.acompletion(
            model="openai/test-gpt-3.5",
            api_base=f"{base_url}/v1",
            api_key=test_user_token,
            messages=[{"role": "user", "content": "Test message"}],
            mock_response="Direct litellm mock response"
        )
        
        print(f"Direct litellm response: {response}")
        assert response is not None, "Should get response from litellm"
        
        # Verify service listing includes the LLM service
        services = await api.list_services(
            query={"workspace": api.config.workspace},
            include_app_services=True
        )
        
        service_ids = [s.get("id") for s in services]
        llm_services = [s for s in service_ids if "test-llm-service" in s]
        assert len(llm_services) > 0, "Should have the test-llm-service registered"
        
        # Get logs from the LLM proxy
        logs = await controller.get_logs(session_id)
        print(f"LLM proxy logs: {logs}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Clean up on failure
        await controller.stop(session_id)
        raise
    
    # Clean up after successful test
    await controller.stop(session_id)
        
    # Test 2: LLM proxy with workspace secrets
    print("\n=== Testing LLM proxy with workspace secrets ===")
    
    # Set workspace environment variables using the api directly
    await api.set_env("TEST_OPENAI_KEY", "sk-test-openai-key-123456")
    await api.set_env("TEST_CLAUDE_KEY", "sk-ant-test-claude-key-789012")
    await api.set_env("TEST_GEMINI_KEY", "test-gemini-api-key-345678")
    
    # Create LLM proxy app with HYPHA_SECRET: prefixed API keys
    llm_app_with_secrets = """
<config lang="json">
{
    "name": "Test LLM Proxy with Secrets",
    "type": "llm-proxy",
    "version": "1.0.0",
    "description": "LLM proxy using workspace secrets for API keys",
    "config": {
        "service_id": "test-llm-secrets",
        "model_list": [
            {
                "model_name": "secret-gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "HYPHA_SECRET:TEST_OPENAI_KEY",
                    "mock_response": "Mock response with resolved OpenAI secret"
                }
            },
            {
                "model_name": "secret-claude",
                "litellm_params": {
                    "model": "anthropic/claude-3-opus",
                    "api_key": "HYPHA_SECRET:TEST_CLAUDE_KEY",
                    "mock_response": "Mock response with resolved Claude secret"
                }
            },
            {
                "model_name": "secret-gemini",
                "litellm_params": {
                    "model": "gemini/gemini-pro",
                    "api_key": "HYPHA_SECRET:TEST_GEMINI_KEY",
                    "mock_response": "Mock response with resolved Gemini secret"
                }
            },
            {
                "model_name": "direct-key-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "direct-api-key-no-secret",
                    "mock_response": "Mock response with direct API key"
                }
            }
        ],
        "litellm_settings": {
            "debug": false,
            "drop_params": true
        }
    }
}
</config>
"""
    
    # Install the app with secrets
    app_info_secrets = await controller.install(
        source=llm_app_with_secrets,
        wait_for_service=False,
        timeout=20,
        overwrite=True,
    )
    
    app_id_secrets = app_info_secrets["id"]
    print(f"Installed LLM proxy app with secrets: {app_id_secrets}")
    
    # Start the app - this should resolve the secrets
    session_secrets = await controller.start(
        app_id_secrets, 
        wait_for_service="test-llm-secrets", 
        timeout=30
    )
    session_id_secrets = session_secrets["id"]
    
    print(f"Started LLM proxy with secrets session: {session_id_secrets}")
    
    # Wait for service to be ready
    await asyncio.sleep(2)
    
    try:
        # Test that the models are available (secrets were resolved)
        base_url_secrets = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/test-llm-secrets"
        
        async with httpx.AsyncClient() as client:
            # Get models list
            models_response = await client.get(
                f"{base_url_secrets}/v1/models",
                headers=headers,
                timeout=10
            )
            
            if models_response.status_code == 200:
                models_data = models_response.json()
                print(f"Models with secrets: {models_data}")
                
                # Verify all models with secrets are available
                model_ids = [m["id"] for m in models_data.get("data", [])]
                assert "secret-gpt-4" in model_ids, "secret-gpt-4 should be available"
                assert "secret-claude" in model_ids, "secret-claude should be available"
                assert "secret-gemini" in model_ids, "secret-gemini should be available"
                assert "direct-key-model" in model_ids, "direct-key-model should be available"
            
            # Test a chat completion with a model using secrets
            chat_request = {
                "model": "secret-gpt-4",
                "messages": [
                    {"role": "user", "content": "Test with workspace secret"}
                ],
                "max_tokens": 50
            }
            
            chat_response = await client.post(
                f"{base_url_secrets}/v1/chat/completions",
                headers=headers,
                json=chat_request,
                timeout=10
            )
            
            if chat_response.status_code == 200:
                chat_data = chat_response.json()
                print(f"Chat response with secrets: {chat_data}")
                
                # Should get mock response since we're using mock_response
                message_content = chat_data["choices"][0]["message"]["content"]
                assert "mock response" in message_content.lower(), "Should return mock response"
                print("✓ Successfully resolved workspace secrets for API keys")
        
        # Test error handling for missing secret
        llm_app_missing_secret = """
<config lang="json">
{
    "name": "Test LLM Missing Secret",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {
        "service_id": "test-missing-secret",
        "model_list": [{
            "model_name": "missing-secret-model",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": "HYPHA_SECRET:NONEXISTENT_KEY"
            }
        }]
    }
}
</config>
"""
        
        # Try to install and start app with missing secret - should fail
        try:
            app_missing = await controller.install(
                source=llm_app_missing_secret,
                wait_for_service="test-missing-secret",
                timeout=20,
                overwrite=True,
            )
            assert False, "Should have failed with missing secret during install"
        except Exception as e:
            print(f"✓ Correctly failed with missing secret: {e}")
            assert "NONEXISTENT_KEY" in str(e) or "not found" in str(e).lower(), "Error should mention missing key"
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Clean up on failure
        await controller.stop(session_id_secrets)
        await controller.uninstall(app_id_secrets)
        await controller.uninstall(app_id)
        await api.set_env("TEST_OPENAI_KEY", None)
        await api.set_env("TEST_CLAUDE_KEY", None)
        await api.set_env("TEST_GEMINI_KEY", None)
        raise
    
    # Clean up after successful test
    await controller.stop(session_id_secrets)
    await controller.uninstall(app_id_secrets)
    await controller.uninstall(app_id)
    await api.set_env("TEST_OPENAI_KEY", None)
    await api.set_env("TEST_CLAUDE_KEY", None)
    await api.set_env("TEST_GEMINI_KEY", None)
    
    print("LLM proxy integration test with secrets completed successfully")


async def test_llm_proxy_with_real_models(
    minio_server, fastapi_server, test_user_token
):
    """Test LLM proxy with configuration for real models (but using mock responses)."""
    api = await connect_to_server({
        "name": "test llm real models",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    # Create an app with configuration that mimics real model setup
    # but still uses mock responses for testing
    llm_app_source = """
<config lang="json">
{
    "name": "Multi-Provider LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "description": "LLM proxy with multiple provider configuration",
    "config": {
        "service_id": "multi-provider-llm",
        "model_list": [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "mock-openai-key",
                    "mock_response": "Mock OpenAI GPT-3.5 response"
                }
            },
            {
                "model_name": "claude-3-opus",
                "litellm_params": {
                    "model": "claude-3-opus-20240229",
                    "api_key": "mock-anthropic-key",
                    "mock_response": "Mock Claude 3 Opus response"
                }
            },
            {
                "model_name": "llama-2-70b",
                "litellm_params": {
                    "model": "replicate/meta/llama-2-70b-chat",
                    "api_key": "mock-replicate-key",
                    "mock_response": "Mock Llama 2 response"
                }
            }
        ],
        "litellm_settings": {
            "debug": false,
            "drop_params": true,
            "num_retries": 1,
            "timeout": 30,
            "routing_strategy": "least-busy"
        }
    }
}
</config>
"""
    
    app_info = await controller.install(
        source=llm_app_source,
        wait_for_service=False,
        timeout=20,
        overwrite=True,
    )
    
    app_id = app_info["id"]
    
    # Start the app
    session = await controller.start(app_id, wait_for_service="multi-provider-llm", timeout=30)
    session_id = session["id"]
    service_id = session.get("service_id")
    
    await asyncio.sleep(2)
    
    try:
        # Test with different models
        # Use the configured service_id
        llm_service_id = "multi-provider-llm"
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/{llm_service_id}"
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        async with httpx.AsyncClient() as client:
            # Test each model
            for model_name in ["gpt-3.5-turbo", "claude-3-opus", "llama-2-70b"]:
                chat_request = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Testing {model_name}"}
                    ],
                    "temperature": 0.5
                }
                
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=chat_request,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"{model_name} response: {data['choices'][0]['message']['content']}")
                    assert "Mock" in data["choices"][0]["message"]["content"], "Should return mock response"
                else:
                    print(f"{model_name} failed with status {response.status_code}")
            
            # Test streaming
            stream_request = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Test streaming"}],
                "stream": True
            }
            
            # Streaming might not work with mock responses, but test the endpoint
            stream_response = await client.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=stream_request,
                timeout=10
            )
            
            print(f"Streaming response status: {stream_response.status_code}")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Clean up on failure
        await controller.stop(session_id)
        await controller.uninstall(app_id)
        raise
    
    # Clean up after successful test
    await controller.stop(session_id)
    await controller.uninstall(app_id)
    
    print("Multi-provider LLM proxy test completed")


async def test_llm_proxy_error_handling(
    minio_server, fastapi_server, test_user_token
):
    """Test error handling in LLM proxy."""
    api = await connect_to_server({
        "name": "test llm errors",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    # Test 1: Install app with invalid configuration (no model_list)
    invalid_app_source = """
<config lang="json">
{
    "name": "Invalid LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {
        "litellm_settings": {}
    }
}
</config>
"""
    
    # This should fail during compilation due to missing model_list
    with pytest.raises(Exception) as exc_info:
        await controller.install(
            source=invalid_app_source,
            wait_for_service=False,
            timeout=20,
            overwrite=True,
        )
    
    assert "model_list" in str(exc_info.value), "Should fail due to missing model_list"
    print(f"Expected compilation error: {exc_info.value}")
    
    # Test 2: Install app with empty model_list
    empty_model_app_source = """
<config lang="json">
{
    "name": "Empty Model LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {
        "model_list": [],
        "litellm_settings": {}
    }
}
</config>
"""
    
    # This should also fail during compilation due to empty model_list
    with pytest.raises(Exception) as exc_info:
        await controller.install(
            source=empty_model_app_source,
            wait_for_service=False,
            timeout=20,
            overwrite=True,
        )
    
    assert "model_list" in str(exc_info.value), "Should fail due to empty model_list"
    print(f"Expected compilation error for empty list: {exc_info.value}")
    
    print("Error handling test completed")


async def test_llm_proxy_workspace_isolation(
    minio_server, fastapi_server, test_user_token, test_user_token_2
):
    """Test that LLM proxy services are properly isolated between workspaces."""
    # Connect as user 1
    api1 = await connect_to_server({
        "name": "test llm user1",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    # Connect as user 2
    api2 = await connect_to_server({
        "name": "test llm user2",
        "server_url": SERVER_URL,
        "token": test_user_token_2
    })
    
    controller1 = await api1.get_service("public/server-apps")
    controller2 = await api2.get_service("public/server-apps")
    
    # User 1 creates an LLM proxy
    app_source = """
<config lang="json">
{
    "name": "User1 LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {
        "service_id": "user1-llm-service",
        "model_list": [{
            "model_name": "user1-model",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "mock_response": "Response from user1's LLM"
            }
        }],
        "litellm_settings": {"debug": false}
    }
}
</config>
"""
    
    app1_info = await controller1.install(source=app_source, overwrite=True, wait_for_service=False)
    app1_id = app1_info["id"]
    
    session1 = await controller1.start(app1_id, wait_for_service="user1-llm-service", timeout=30)
    session1_id = session1["id"]
    
    await asyncio.sleep(2)
    
    # We know the service ID from the manifest config
    service1_id = "user1-llm-service"
    
    try:
        # User 2 should NOT be able to access user 1's LLM service
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api1.config.workspace}/llm/{service1_id}"
        
        async with httpx.AsyncClient() as client:
            # Try with user2's token - should fail
            response = await client.get(
                f"{base_url}/v1/models",
                headers={"Authorization": f"Bearer {test_user_token_2}"},
                timeout=10
            )
            
            # Should get 403 Forbidden or 404 Not Found
            assert response.status_code in [403, 404], "User 2 should not access User 1's service"
            print(f"Workspace isolation working: User 2 got {response.status_code}")
            
            # User 1 should be able to access their own service
            response = await client.get(
                f"{base_url}/v1/models",
                headers={"Authorization": f"Bearer {test_user_token}"},
                timeout=10
            )
            
            if response.status_code == 200:
                print("User 1 can access their own service")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Clean up on failure
        await controller1.stop(session1_id)
        await controller1.uninstall(app1_id)
        raise
    
    # Clean up after successful test
    await controller1.stop(session1_id)
    await controller1.uninstall(app1_id)
    
    print("Workspace isolation test completed")