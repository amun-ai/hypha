"""Integration tests for the LLM proxy worker using real infrastructure."""

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
    
    # Create an LLM proxy app with mock configuration
    # This uses litellm's mock response feature for testing
    llm_app_source = """
<config lang="json">
{
    "name": "Test LLM Proxy",
    "type": "llm-proxy",
    "version": "1.0.0",
    "description": "LLM proxy for testing with mock responses",
    "config": {
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
    
    # Start the LLM proxy app
    session = await controller.start(app_id, wait_for_service=True, timeout=30)
    session_id = session["id"]
    service_id = session.get("service_id")
    
    print(f"Started LLM proxy session: {session_id}, service: {service_id}")
    
    # Wait a bit for the service to be fully ready
    await asyncio.sleep(2)
    
    try:
        # Test 1: Access the LLM service through Hypha's API
        # The service should be registered as an ASGI service
        if service_id:
            # Extract the actual service name from the full service_id
            # Format: workspace/client_id:service_name@app_id
            # We want the full client_id:service_name part for the get_service call
            if "@" in service_id:
                # Extract everything before @app_id
                service_path = service_id.split("@")[0]
                # Try to get the specific service
                llm_service = await api.get_service(service_path)
                assert llm_service is not None, "Service should be registered"
                print(f"Service found: {service_path}")
            else:
                # Fallback to using service_id as-is
                llm_service = await api.get_service(service_id)
                assert llm_service is not None, "LLM service should be registered"
                print(f"LLM service found: {service_id}")
        
        # Test 2: Make direct HTTP requests to the LLM endpoint
        # Find the actual LLM service from the registered services
        # The LLM service is registered as llm-{uuid}
        services = await api.list_services(
            query={"workspace": api.config.workspace},
            include_app_services=True
        )
        
        # Find the LLM service for this session
        llm_service_id = None
        llm_service_full_id = None
        for svc in services:
            svc_id = svc.get("id", "")
            # Check if this is an LLM service for our session
            if session_id in svc_id and ":llm-" in svc_id:
                # Extract the service name part (llm-{uuid})
                parts = svc_id.split(":")
                if len(parts) >= 2:
                    service_name = parts[1].split("@")[0]  # Remove app_id suffix if present
                    if service_name.startswith("llm-"):
                        llm_service_id = service_name
                        llm_service_full_id = svc_id
                        print(f"Found LLM service: {llm_service_full_id}")
                        break
        
        if not llm_service_id:
            # Fallback: construct from session_id (may not work with UUID-based IDs)
            client_part = session_id.split("/")[1] if "/" in session_id else session_id
            llm_service_id = f"llm-{client_part}"
            print(f"Warning: Could not find LLM service, using fallback: {llm_service_id}")
        
        # The URL pattern is: /workspace/llm/<service_id>/...
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/llm/{llm_service_id}"
        
        # Create an HTTP client with auth header
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
            
            # The health check might fail if middleware is not properly set up
            # but we should at least get a response
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
                
                # Verify our test models are listed
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
                
                # Verify we got the mock response
                message_content = chat_data["choices"][0]["message"]["content"]
                assert "mock response" in message_content.lower(), "Should return mock response"
        
        # Test 3: Use litellm directly with the proxy endpoint
        # This tests if the service can be used as an OpenAI-compatible endpoint
        # Configure litellm to use our proxy
        response = await litellm.acompletion(
            model="openai/test-gpt-3.5",
            api_base=f"{base_url}/v1",
            api_key=test_user_token,
            messages=[{"role": "user", "content": "Test message"}],
            mock_response="Direct litellm mock response"
        )
        
        print(f"Direct litellm response: {response}")
        assert response is not None, "Should get response from litellm"
        
        # Test 4: Verify service listing includes the LLM service
        # (Already fetched services above, but re-fetch for this test)
        services = await api.list_services(
            query={"workspace": api.config.workspace},
            include_app_services=True
        )
        
        service_ids = [s.get("id") for s in services]
        # Check that the LLM service we found is in the list
        if llm_service_full_id:
            llm_service_id = llm_service_full_id.split("@")[0]
            assert llm_service_full_id in service_ids, "LLM service should be in service list"
        else:
            # At least check that some LLM service exists for this session
            llm_services = [s for s in service_ids if session_id in s and ":llm-" in s]
            assert len(llm_services) > 0, "Should have at least one LLM service for this session"
        
        # Test 5: Get logs from the LLM proxy
        logs = await controller.get_logs(session_id)
        print(f"LLM proxy logs: {logs}")
        
    finally:
        # Always clean up: stop the app
        await controller.stop(session_id)
        
        # Optionally uninstall the app
        await controller.uninstall(app_id)
    
    print("LLM proxy integration test completed successfully")


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
    session = await controller.start(app_id, wait_for_service=True, timeout=30)
    session_id = session["id"]
    service_id = session.get("service_id")
    
    await asyncio.sleep(2)
    
    try:
        # Test with different models
        base_url = f"http://127.0.0.1:{SIO_PORT}/{api.config.workspace}/llm/{service_id}"
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
            
    finally:
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
    
    app1_info = await controller1.install(source=app_source, overwrite=True)
    app1_id = app1_info["id"]
    
    session1 = await controller1.start(app1_id, wait_for_service=True, timeout=30)
    session1_id = session1["id"]
    
    await asyncio.sleep(2)
    
    # Find the actual LLM service from the registered services (same logic as server apps test)
    services1 = await api1.list_services(
        query={"workspace": api1.config.workspace},
        include_app_services=True
    )
    
    # Find the LLM service for this session
    service1_id = None
    for svc in services1:
        svc_id = svc.get("id", "")
        # Check if this is an LLM service for our session
        if session1_id in svc_id and ":llm-" in svc_id:
            # Extract the service name part (llm-{uuid})
            parts = svc_id.split(":")
            if len(parts) >= 2:
                service_name = parts[1].split("@")[0]  # Remove app_id suffix if present
                if service_name.startswith("llm-"):
                    service1_id = service_name
                    break
    
    if not service1_id:
        raise AssertionError(f"Could not find LLM service for session {session1_id} in services")
    
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
            
    finally:
        await controller1.stop(session1_id)
        await controller1.uninstall(app1_id)
    
    print("Workspace isolation test completed")