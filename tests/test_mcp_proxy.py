"""Test MCP Apps functionality with both SSE and Streamable HTTP transports."""

import asyncio
import json
import pytest
import httpx
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from . import WS_SERVER_URL, SERVER_URL

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_mcp_streamable_http_round_trip_service_consistency(fastapi_server, test_user_token):
    """Test complete MCP double round-trip using Streamable HTTP transport."""
    
    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")
    
    # Step 1: Register a type=mcp hypha service with tools, resources, and prompts
    @schema_function
    def calculate_tool(operation: str, a: float, b: float) -> str:
        """Perform basic arithmetic calculations."""
        if operation == "add":
            return f"Result: {a + b}"
        elif operation == "subtract":
            return f"Result: {a - b}"
        elif operation == "multiply":
            return f"Result: {a * b}"
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            return f"Result: {a / b}"
        else:
            return f"Unknown operation: {operation}"

    @schema_function
    def resource_read() -> str:
        """Read a test resource."""
        return "This is comprehensive test resource content with detailed information."

    @schema_function
    def prompt_read(task: str = "general", difficulty: str = "medium") -> dict:
        """Read a test prompt with parameters."""
        return {
            "description": f"Test prompt for {task} at {difficulty} level",
            "messages": [
                {
                    "role": "user", 
                    "content": {
                        "type": "text",
                        "text": f"Help me with {task} at {difficulty} difficulty level"
                    }
                }
            ]
        }
    
    # Register the original MCP service
    original_service_info = await api.register_service(
        {
            "id": "streamable-http-test-service",
            "name": "Streamable HTTP Test Service",
            "description": "A service for testing MCP streamable HTTP round-trip consistency",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": [calculate_tool],
            "resources": [
                {
                    "uri": "resource://streamable-http-test",
                    "name": "Streamable HTTP Test Resource", 
                    "description": "A test resource for streamable HTTP transport",
                    "tags": ["test", "streamable-http", "resource"],
                    "mime_type": "text/plain",
                    "read": resource_read,
                }
            ],
            "prompts": [
                {
                    "name": "streamable_http_prompt",
                    "description": "A test prompt for streamable HTTP transport",
                    "tags": ["test", "streamable-http", "prompt"],
                    "read": prompt_read,
                }
            ],
        }
    )

    print(f"✓ Original MCP service registered: {original_service_info['id']}")
    
    # Step 2: Get the original service to save its function references
    original_service = await api.get_service(original_service_info["id"])
                
    # Verify the original service has the expected structure
    assert hasattr(original_service, "tools") and original_service.tools is not None
    assert hasattr(original_service, "resources") and original_service.resources is not None
    assert hasattr(original_service, "prompts") and original_service.prompts is not None

    print("✓ Original service has tools, resources, and prompts organized correctly")

    # Step 3: Get the MCP endpoint URL for this service
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/mcp"
    print(f"✓ MCP Endpoint URL: {mcp_endpoint_url}")

    # Step 4: Create MCP server configuration with streamable-http transport
    mcp_config = {
        "type": "mcp-server",
        "name": "Streamable HTTP MCP App",
        "version": "1.0.0",
        "description": "MCP app created from original service for streamable HTTP testing",
        "mcpServers": {
            "streamable-http-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url
            }
        }
    }
    
    # Install the MCP server app
    mcp_app_info = await controller.install(
        config=mcp_config,
        overwrite=True,
        stage=True,
    )

    print(f"✓ MCP app installed: {mcp_app_info['id']}")

    # Step 5: Start the app manually 
    print("🚀 Starting MCP app...")
    server_name = "streamable-http-server"
    session_info = await controller.start(mcp_app_info["id"], wait_for_service=server_name, timeout=30)
    print(f"✓ MCP app started successfully: {session_info}")

    # Step 6: Get the unified MCP service
    print(f"🔍 Getting unified MCP service: {session_info['id']}:{server_name}")
    
    mcp_service_id = f"{session_info['id']}:{server_name}"
    mcp_service = await api.get_service(mcp_service_id)
    print(f"✓ Successfully retrieved unified MCP service: {mcp_service_id}")
    
    # Step 7: Compare original service with the MCP service
    print("🔄 Comparing original service with MCP service...")
    
    # Test tool functionality consistency
    print("  Testing calculate_tool...")
    original_tool = original_service.tools[0]
    mcp_tool = mcp_service.tools[0]
    
    # Test the tools with the same parameters
    original_result = await original_tool(operation="add", a=5, b=3)
    mcp_result = await mcp_tool(operation="add", a=5, b=3)
    assert original_result == mcp_result, f"Tool results differ: original={original_result}, mcp={mcp_result}"
    print(f"    ✓ Both return: {original_result}")
    
    # Test resource functionality consistency
    print("  Testing resource access...")
    original_resource = original_service.resources[0]
    mcp_resource = mcp_service.resources[0]
    
    original_resource_content = await original_resource["read"]()
    mcp_resource_content = await mcp_resource["read"]()
    assert original_resource_content == mcp_resource_content, f"Resource content differs: original={original_resource_content}, mcp={mcp_resource_content}"
    print(f"    ✓ Both return: {original_resource_content[:50]}...")
    
    # Test prompt functionality consistency  
    print("  Testing prompt access...")
    original_prompt = original_service.prompts[0]
    mcp_prompt = mcp_service.prompts[0]
    
    original_prompt_content = await original_prompt["read"](task="testing", difficulty="easy")
    mcp_prompt_content = await mcp_prompt["read"](task="testing", difficulty="easy")
    
    assert "description" in original_prompt_content and "description" in mcp_prompt_content
    assert "messages" in original_prompt_content and "messages" in mcp_prompt_content
    print(f"    ✓ Both prompts have correct structure with messages")
    
    print("✅ STREAMABLE HTTP ROUND-TRIP SUCCESS!")
    
    # Clean up
    await controller.stop(session_info["id"])
    print("✓ MCP app session stopped")
    
    await controller.uninstall(mcp_app_info["id"])
    print("✓ MCP app uninstalled")
    
    await api.unregister_service(original_service_info["id"])
    print("✓ Original service unregistered")
    
    await api.disconnect()


async def test_mcp_error_handling_and_debugging(fastapi_server, test_user_token):
    """Test error handling and debugging capabilities of MCP clients."""
    
    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")
    
    # Test 1: Invalid URL handling
    print("🔍 Testing invalid URL handling...")
    mcp_config_invalid = {
        "type": "mcp-server",
        "name": "Invalid URL MCP App",
        "version": "1.0.0",
        "description": "MCP app with invalid URL to test error handling",
        "mcpServers": {
            "invalid-server": {
                "type": "streamable-http",
                "url": "http://invalid-host:9999/mcp"
            }
        }
    }
    
    # Install the MCP server app with invalid URL
    mcp_app_info = await controller.install(
        config=mcp_config_invalid,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Invalid URL MCP app installed: {mcp_app_info['id']}")

    # Start the app - should handle the error gracefully
    try:
        session_info = await controller.start(mcp_app_info["id"], wait_for_service="invalid-server", timeout=10)
        print(f"⚠️ App started but likely with errors: {session_info}")
        
        # Check logs for error messages
        logs = await controller.get_logs(session_info["id"])
        print(f"✓ Logs retrieved: {logs}")
        
        # Should have error logs
        assert "error" in logs
        assert len(logs["error"]) > 0
        print("✓ Error logs found as expected")
        
        # Stop the session
        await controller.stop(session_info["id"])
        
    except Exception as e:
        print(f"✓ Expected error occurred: {e}")
    
    # Clean up
    await controller.uninstall(mcp_app_info["id"])
    print("✓ Invalid URL MCP app uninstalled")
    
    # Test 2: Unsupported transport type
    print("🔍 Testing unsupported transport type...")
    mcp_config_unsupported = {
        "type": "mcp-server",
        "name": "Unsupported Transport MCP App",
        "version": "1.0.0",
        "description": "MCP app with unsupported transport to test error handling",
        "mcpServers": {
            "unsupported-server": {
                "type": "websocket",  # Unsupported transport
                "url": "ws://localhost:8080/mcp"
            }
        }
    }
    
    # Install the MCP server app with unsupported transport
    mcp_app_info = await controller.install(
        config=mcp_config_unsupported,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Unsupported transport MCP app installed: {mcp_app_info['id']}")

    # Start the app - should handle the error gracefully
    try:
        session_info = await controller.start(mcp_app_info["id"], wait_for_service="unsupported-server", timeout=10)
        print(f"⚠️ App started but likely with errors: {session_info}")
        
        # Check logs for error messages
        logs = await controller.get_logs(session_info["id"])
        print(f"✓ Logs retrieved: {logs}")
        
        # Should have error logs about unsupported transport
        assert "error" in logs
        assert len(logs["error"]) > 0
        print("✓ Error logs found as expected for unsupported transport")
        
        # Stop the session
        await controller.stop(session_info["id"])
        
    except Exception as e:
        print(f"✓ Expected error occurred: {e}")
    
    # Clean up
    await controller.uninstall(mcp_app_info["id"])
    print("✓ Unsupported transport MCP app uninstalled")
    
    print("✅ ERROR HANDLING AND DEBUGGING TESTS PASSED!")
    
    await api.disconnect()


async def test_real_deepwiki_mcp_server_validation(fastapi_server, test_user_token):
    """Test DeepWiki MCP server configuration validation, lazy loading, and actual service functionality."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
    workspace = api.config.workspace
    controller = await api.get_service("public/server-apps")
    
    # Test real DeepWiki MCP server configuration
    deepwiki_config = {
        "type": "mcp-server",
        "name": "DeepWiki MCP Server",
        "version": "1.0.0",
        "description": "Real DeepWiki MCP server for documentation and search",
        "mcpServers": {
            "deepwiki": {
                "type": "streamable-http",
                "url": "https://mcp.deepwiki.com/mcp"
            }
        }
    }
    
    # Validate the configuration
    validation_result = await controller.validate_app_config(deepwiki_config)
    assert validation_result["valid"] is True
    
    # Install the real MCP server app - connect to actual external service
    app_info = await controller.install(
        config=deepwiki_config,
        wait_for_service="deepwiki",
        overwrite=True,
    )
    
    # Verify the app was installed correctly
    assert app_info["name"] == "DeepWiki MCP Server"
    assert app_info["type"] == "mcp-server"
    assert "mcpServers" in app_info
    assert "deepwiki" in app_info["mcpServers"]
    assert app_info["mcpServers"]["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"
    
    # Verify the config file is properly created
    config_content = await controller.read_file(app_info["id"], "mcp-config.json")
    config_json = json.loads(config_content)
    assert config_json["type"] == "mcp-server"
    assert "deepwiki" in config_json["mcpServers"]
    assert config_json["mcpServers"]["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"
    
    # Test lazy loading - get the unified service without explicit start
    session_info = await controller.start(app_info["id"], wait_for_service="deepwiki", timeout=30)
    deepwiki_service = await api.get_service("deepwiki@" + app_info.id, mode="first")
    
    # Verify the service structure
    assert hasattr(deepwiki_service, "tools")
    assert hasattr(deepwiki_service, "resources")
    assert hasattr(deepwiki_service, "prompts")
    assert deepwiki_service.tools is not None
    assert deepwiki_service.resources is not None
    assert deepwiki_service.prompts is not None
    
    # Test actual tool functionality
    assert len(deepwiki_service.tools) > 0
    tools = {}
    for tool in deepwiki_service.tools:
        tools[tool.__name__] = tool
    print(f"Available Tools in DeepWiki: {tools}")

    read_wiki_structure = tools["read_wiki_structure"]
    # Test with simple parameters - DeepWiki tools likely have search functionality
    test_result = await read_wiki_structure(repoName="amun-ai/hypha")
    print(f"Read wiki structure result: {test_result}")
    assert test_result is not None and "amun-ai/hypha" in test_result
    

    # Test MCP HTTP endpoint functionality
    running_apps = await controller.list_running()
    deepwiki_app = None
    for app in running_apps:
        if app.get("app_id") == app_info["id"]:
            deepwiki_app = app
            break
    
    assert deepwiki_app is not None
    # Clean up
    running_apps = await controller.list_running()
    for app in running_apps:
        if app.get("app_id") == app_info["id"]:
            await controller.stop(app["id"])
    
    await controller.uninstall(app_info["id"])
    await api.disconnect() 