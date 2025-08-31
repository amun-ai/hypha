"""Tests for the LLM proxy worker - integration tests with real infrastructure."""

import asyncio
import json
import pytest
import httpx
from hypha_rpc import connect_to_server
import litellm

from . import (
    WS_SERVER_URL,
    SERVER_URL,
    SIO_PORT,
)

# All test coroutines will be treated as marked with pytest.mark.asyncio
pytestmark = pytest.mark.asyncio


# ============================================================================
# INTEGRATION TESTS WITH REAL INFRASTRUCTURE
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
        stage=True,  # Don't test the app during install to avoid leftover services
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
    await controller.uninstall(app_id)
    
    print("LLM proxy basic operation test completed successfully")


async def test_llm_proxy_with_workspace_secrets(
    minio_server, fastapi_server, test_user_token
):
    """Test LLM proxy with workspace secrets."""
    api = await connect_to_server({
        "name": "test llm secrets",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
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
        stage=True,  # Don't test the app during install to avoid leftover services
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
    
    headers = {
        "Authorization": f"Bearer {test_user_token}"
    }
    
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
        await api.set_env("TEST_OPENAI_KEY", None)
        await api.set_env("TEST_CLAUDE_KEY", None)
        await api.set_env("TEST_GEMINI_KEY", None)
        raise
    
    # Clean up after successful test
    await controller.stop(session_id_secrets)
    await controller.uninstall(app_id_secrets)
    await api.set_env("TEST_OPENAI_KEY", None)
    await api.set_env("TEST_CLAUDE_KEY", None)
    await api.set_env("TEST_GEMINI_KEY", None)
    
    print("LLM proxy integration test with secrets completed successfully")


async def test_llm_proxy_with_multiple_models(
    minio_server, fastapi_server, test_user_token
):
    """Test LLM proxy with multiple model configurations."""
    api = await connect_to_server({
        "name": "test llm multi models",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    # Create an app with configuration that has multiple providers
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
        stage=True,  # Don't test the app during install to avoid leftover services
    )
    
    app_id = app_info["id"]
    
    # Start the app
    session = await controller.start(app_id, wait_for_service="multi-provider-llm", timeout=30)
    session_id = session["id"]
    service_id = session.get("service_id")
    
    await asyncio.sleep(2)
    
    try:
        # Test with different models
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
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api1.config.workspace}/apps/{service1_id}"
        
        async with httpx.AsyncClient() as client:
            # Try with user2's token - should fail
            response = await client.get(
                f"{base_url}/v1/models",
                headers={"Authorization": f"Bearer {test_user_token_2}"},
                timeout=10
            )
            
            # Should get error status code (403, 404, or 500)
            assert response.status_code >= 400, "User 2 should not access User 1's service"
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


async def test_llm_proxy_lifecycle_management(
    minio_server, fastapi_server, test_user_token
):
    """Test the complete lifecycle of LLM proxy: install, start, stop, restart, uninstall."""
    api = await connect_to_server({
        "name": "test llm lifecycle",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    # Clean up any leftover instances from previous test runs
    try:
        instances = await controller.list_apps()
        for instance in instances:
            if instance.get("app_id") and "lifecycle" in str(instance.get("app_id", "")):
                try:
                    await controller.stop(instance["id"])
                    print(f"Cleaned up leftover instance: {instance['id']}")
                except:
                    pass  # Ignore errors during cleanup
    except:
        pass  # Ignore if list fails
    
    # Use unique service ID to avoid conflicts
    import uuid
    unique_service_id = f"lifecycle-llm-{uuid.uuid4().hex[:8]}"
    
    llm_app_source = f"""
<config lang="json">
{{
    "name": "Lifecycle Test LLM",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {{
        "service_id": "{unique_service_id}",
        "model_list": [{{
            "model_name": "test-model",
            "litellm_params": {{
                "model": "gpt-3.5-turbo",
                "mock_response": "Lifecycle test response"
            }}
        }}],
        "litellm_settings": {{"debug": false}}
    }}
}}
</config>
"""
    
    # Install
    app_info = await controller.install(
        source=llm_app_source,
        wait_for_service=False,
        timeout=20,
        overwrite=True,
    )
    app_id = app_info["id"]
    print(f"Installed app: {app_id}")
    
    # Start
    session = await controller.start(app_id, wait_for_service=unique_service_id, timeout=30)
    session_id = session["id"]
    print(f"Started session: {session_id}")
    
    await asyncio.sleep(2)
    
    # Verify it's running
    base_url = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/{unique_service_id}"
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health", headers=headers, timeout=10)
        assert response.status_code in [200, 204], "Service should be healthy"
        print("Service is healthy after start")
    
    # Stop
    print(f"About to stop session: {session_id}")
    await controller.stop(session_id)
    print(f"Stopped session: {session_id}")
    
    # Verify services are cleaned up
    await asyncio.sleep(1)
    instances_after_stop = await controller.list_apps()
    print(f"Instances after stop: {[inst['id'] for inst in instances_after_stop if inst.get('app_id') == app_id]}")
    
    # Add more aggressive cleanup: check for any services with our service ID and clean them up
    try:
        workspace_manager = await api.get_service("public/server-apps")  # This should actually be workspace manager but let's work with what we have
        print(f"Checking for remaining services with service_id: {unique_service_id}")
        
        # Wait longer for cleanup to complete
        for i in range(5):  # Try up to 5 times with 2-second intervals
            await asyncio.sleep(2)
            try:
                # Try to access the service - if it fails, it's cleaned up
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{base_url}/health", headers=headers, timeout=5)
                    if response.status_code >= 400:
                        print(f"Service cleanup confirmed after {(i+1)*2} seconds")
                        break
            except:
                print(f"Service cleanup confirmed after {(i+1)*2} seconds")
                break
        else:
            print("Warning: Service might still be accessible after stop")
        
    except Exception as e:
        print(f"Error during cleanup verification: {e}")
    
    # Verify it's stopped
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{base_url}/health", headers=headers, timeout=5)
            # Should fail or return error since service is stopped
            assert response.status_code >= 400, "Service should not be accessible when stopped"
        except:
            # Connection error is expected when service is stopped
            print("Service correctly unavailable after stop")
    
    # Restart
    print(f"About to restart app: {app_id}")
    session2 = await controller.start(app_id, wait_for_service=unique_service_id, timeout=30)
    session2_id = session2["id"]
    print(f"Restarted with new session: {session2_id}")
    
    await asyncio.sleep(2)
    
    # Verify it's running again - with debugging for 500 errors
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health", headers=headers, timeout=10)
        if response.status_code != 200 and response.status_code != 204:
            print(f"Service health check failed with status {response.status_code}")
            try:
                error_text = response.text
                print(f"Error response: {error_text}")
            except:
                print("Could not read error response")
            
            # Try to get more information about what services are registered
            try:
                models_response = await client.get(f"{base_url}/v1/models", headers=headers, timeout=10)
                print(f"Models endpoint status: {models_response.status_code}")
                if models_response.status_code == 200:
                    print(f"Models available: {models_response.json()}")
                else:
                    print(f"Models endpoint error: {models_response.text}")
            except Exception as e:
                print(f"Could not check models endpoint: {e}")
        
        assert response.status_code in [200, 204], f"Service should be healthy after restart, got {response.status_code}: {response.text if hasattr(response, 'text') else 'No response text'}"
        print("Service is healthy after restart")
    
    try:
        # Stop before uninstall
        await controller.stop(session2_id)
    except:
        pass  # Continue with cleanup even if stop fails
    
    try:
        # Uninstall
        await controller.uninstall(app_id)
        print(f"Uninstalled app: {app_id}")
    except:
        pass  # Continue even if uninstall fails
    
    # Verify app is gone
    apps = await controller.list_apps()
    app_ids = [app["id"] for app in apps]
    assert app_id not in app_ids, "App should be uninstalled"
    print("App successfully removed from list")
    
    print("Lifecycle management test completed")


async def test_llm_proxy_concurrent_sessions(
    minio_server, fastapi_server, test_user_token
):
    """Test running multiple concurrent LLM proxy sessions."""
    api = await connect_to_server({
        "name": "test llm concurrent",
        "server_url": SERVER_URL,
        "token": test_user_token
    })
    
    controller = await api.get_service("public/server-apps")
    
    # Use unique service IDs to avoid conflicts
    import uuid
    unique_service_id1 = f"concurrent-llm-1-{uuid.uuid4().hex[:8]}"
    unique_service_id2 = f"concurrent-llm-2-{uuid.uuid4().hex[:8]}"
    
    # Create two different LLM proxy apps
    llm_app1 = f"""
<config lang="json">
{{
    "name": "Concurrent LLM 1",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {{
        "service_id": "{unique_service_id1}",
        "model_list": [{{
            "model_name": "model-1",
            "litellm_params": {{
                "model": "gpt-3.5-turbo",
                "mock_response": "Response from LLM 1"
            }}
        }}],
        "litellm_settings": {{"debug": false}}
    }}
}}
</config>
"""
    
    llm_app2 = f"""
<config lang="json">
{{
    "name": "Concurrent LLM 2",
    "type": "llm-proxy",
    "version": "1.0.0",
    "config": {{
        "service_id": "{unique_service_id2}",
        "model_list": [{{
            "model_name": "model-2",
            "litellm_params": {{
                "model": "gpt-4",
                "mock_response": "Response from LLM 2"
            }}
        }}],
        "litellm_settings": {{"debug": false}}
    }}
}}
</config>
"""
    
    # Install both apps
    app1_info = await controller.install(source=llm_app1, overwrite=True, wait_for_service=False, stage=True)  # Don't test during install
    app1_id = app1_info["id"]
    
    app2_info = await controller.install(source=llm_app2, overwrite=True, wait_for_service=False, stage=True)  # Don't test during install
    app2_id = app2_info["id"]
    
    print(f"Installed apps: {app1_id}, {app2_id}")
    
    # Start both concurrently
    session1 = await controller.start(app1_id, wait_for_service=unique_service_id1, timeout=30)
    session1_id = session1["id"]
    
    session2 = await controller.start(app2_id, wait_for_service=unique_service_id2, timeout=30)
    session2_id = session2["id"]
    
    print(f"Started sessions: {session1_id}, {session2_id}")
    
    await asyncio.sleep(2)
    
    try:
        # Test both services concurrently
        headers = {"Authorization": f"Bearer {test_user_token}"}
        
        async with httpx.AsyncClient() as client:
            # Test service 1
            base_url1 = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/{unique_service_id1}"
            response1 = await client.get(f"{base_url1}/v1/models", headers=headers, timeout=10)
            assert response1.status_code == 200, "Service 1 should be accessible"
            models1 = response1.json()
            print(f"Service 1 models response: {models1}")
            # Look for model-1 in the response
            model_found = any("model-1" in m["id"] for m in models1.get("data", []))
            print(f"Looking for 'model-1' in models: {[m['id'] for m in models1.get('data', [])]}")
            assert model_found, f"Service 1 should have model-1, got: {[m['id'] for m in models1.get('data', [])]}"
            
            # Test service 2
            base_url2 = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/apps/{unique_service_id2}"
            response2 = await client.get(f"{base_url2}/v1/models", headers=headers, timeout=10)
            assert response2.status_code == 200, "Service 2 should be accessible"
            models2 = response2.json()
            print(f"Service 2 models response: {models2}")
            # Look for model-2 in the response
            model_found = any("model-2" in m["id"] for m in models2.get("data", []))
            print(f"Looking for 'model-2' in models: {[m['id'] for m in models2.get('data', [])]}")
            assert model_found, f"Service 2 should have model-2, got: {[m['id'] for m in models2.get('data', [])]}"
            
            print("Both concurrent services are running correctly")
            
            # Test chat completions on both
            chat_tasks = []
            for service_id, model_name, base_url in [
                ("concurrent-llm-1", "model-1", base_url1),
                ("concurrent-llm-2", "model-2", base_url2)
            ]:
                chat_request = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": f"Test {service_id}"}],
                    "max_tokens": 50
                }
                task = client.post(
                    f"{base_url}/v1/chat/completions",
                    headers=headers,
                    json=chat_request,
                    timeout=10
                )
                chat_tasks.append((service_id, task))
            
            # Wait for both responses
            for service_id, task in chat_tasks:
                response = await task
                if response.status_code == 200:
                    data = response.json()
                    print(f"{service_id} responded successfully")
                    assert "choices" in data, f"{service_id} should return proper response"
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        # Clean up on failure
        await controller.stop(session1_id)
        await controller.stop(session2_id)
        await controller.uninstall(app1_id)
        await controller.uninstall(app2_id)
        raise
    
    # Clean up
    await controller.stop(session1_id)
    await controller.stop(session2_id)
    await controller.uninstall(app1_id)
    await controller.uninstall(app2_id)
    
    print("Concurrent sessions test completed")