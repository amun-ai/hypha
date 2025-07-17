"""Test MCP Apps functionality."""

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


async def test_mcp_round_trip_service_consistency(fastapi_server, test_user_token):
    """Test complete MCP double round-trip: Original Service -> MCP Endpoint 1 -> MCP App -> Unified Service -> MCP Endpoint 2 -> Consistency verification."""
    
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
            "id": "round-trip-test-service",
            "name": "Round Trip Test Service",
            "description": "A service for testing MCP round-trip consistency",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": [calculate_tool],
            "resources": [
                {
                    "uri": "resource://comprehensive-test",
                    "name": "Comprehensive Test Resource", 
                    "description": "A comprehensive test resource with detailed content",
                    "tags": ["test", "comprehensive", "resource"],
                    "mime_type": "text/plain",
                    "read": resource_read,
                }
            ],
            "prompts": [
                {
                    "name": "comprehensive_prompt",
                    "description": "A comprehensive test prompt template with parameters",
                    "tags": ["test", "comprehensive", "prompt"],
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

    # Step 3: Get the FIRST MCP endpoint URL for this service
    mcp_endpoint_1_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/"
    print(f"✓ MCP Endpoint 1 URL: {mcp_endpoint_1_url}")

    # Step 4: Create MCP server configuration and install as MCP app
    mcp_config = {
        "type": "mcp-server",
        "name": "Round Trip MCP App",
        "version": "1.0.0",
        "description": "MCP app created from original service for round-trip testing",
        "mcpServers": {
            "round-trip-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_1_url
            }
        }
    }
    
    # Install the MCP server app
    mcp_app_info = await controller.install(
        config=mcp_config,
        overwrite=True,
        stage=True,  # Use stage mode first to avoid timeout issues
    )

    print(f"✓ MCP app installed: {mcp_app_info['id']}")

    # Step 5: Start the app manually 
    print("🚀 Starting MCP app...")
    server_name = "round-trip-server"  # This is the key from mcpServers config
    session_info = await controller.start(mcp_app_info["id"], wait_for_service=server_name, timeout=30)
    print(f"✓ MCP app started successfully: {session_info}")

    # Step 6: Get the unified MCP service directly using session ID and server name
    print(f"🔍 Getting unified MCP service directly: {session_info['id']}:{server_name}")
    
    mcp_service_id = f"{session_info['id']}:{server_name}"
    mcp_service = await api.get_service(mcp_service_id)
    print(f"✓ Successfully retrieved unified MCP service: {mcp_service_id}")
    
    # Step 7: Compare original service with the MCP service
    print("🔄 Comparing original service with MCP service...")
    
    # Test tool functionality consistency
    print("  Testing calculate_tool...")
    
    # Access tools properly - both services have tools in the tools array
    # Neither service exposes tools as direct attributes
    original_tool = original_service.tools[0]  # calculate_tool is the first (and only) tool
    mcp_tool = mcp_service.tools[0]  # MCP service also stores tools in the tools array
    
    # Test the tools with the same parameters
    original_result = await original_tool(operation="add", a=5, b=3)
    mcp_result = await mcp_tool(operation="add", a=5, b=3)
    assert original_result == mcp_result, f"Tool results differ: original={original_result}, mcp={mcp_result}"
    print(f"    ✓ Both return: {original_result}")
    
    # Test another calculation
    original_result2 = await original_tool(operation="multiply", a=10, b=7)
    mcp_result2 = await mcp_tool(operation="multiply", a=10, b=7)
    assert original_result2 == mcp_result2, f"Tool results differ: original={original_result2}, mcp={mcp_result2}"
    print(f"    ✓ Both return: {original_result2}")
    
    # Test resource functionality consistency
    print("  Testing resource access...")
    
    # Access resources properly - original service has resources in the resources array
    original_resource = original_service.resources[0]  # comprehensive-test resource
    mcp_resource = mcp_service.resources[0]  # MCP service also has resources array
    
    # Test reading the resource
    original_resource_content = await original_resource["read"]()
    mcp_resource_content = await mcp_resource["read"]()
    assert original_resource_content == mcp_resource_content, f"Resource content differs: original={original_resource_content}, mcp={mcp_resource_content}"
    print(f"    ✓ Both return: {original_resource_content[:50]}...")
    
    # Test prompt functionality consistency  
    print("  Testing prompt access...")
    
    # Access prompts properly - original service has prompts in the prompts array
    original_prompt = original_service.prompts[0]  # comprehensive_prompt
    mcp_prompt = mcp_service.prompts[0]  # MCP service also has prompts array
    
    # Test reading the prompt with parameters
    original_prompt_content = await original_prompt["read"](task="testing", difficulty="easy")
    mcp_prompt_content = await mcp_prompt["read"](task="testing", difficulty="easy")
    
    # Compare the basic structure
    assert "description" in original_prompt_content and "description" in mcp_prompt_content
    assert "messages" in original_prompt_content and "messages" in mcp_prompt_content
    print(f"    ✓ Both prompts have correct structure with messages")
    
    print("✅ FIRST ROUND-TRIP SUCCESS! Original service -> MCP service consistency verified!")
    
    # Step 8: CREATE SECOND MCP ENDPOINT from the unified service
    print("\n🔄 STARTING SECOND ROUND-TRIP: Creating MCP Endpoint 2 from unified service...")
    
    # Get the SECOND MCP endpoint URL from the unified service
    mcp_endpoint_2_url = f"{SERVER_URL}/{workspace}/mcp/{mcp_service_id.split('/')[-1]}/"
    print(f"✓ MCP Endpoint 2 URL: {mcp_endpoint_2_url}")
    
    # Step 9: TEST MCP ENDPOINT CONSISTENCY using direct HTTP calls
    print("🔍 Comparing MCP Endpoint 1 vs MCP Endpoint 2 for consistency...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test tools/list consistency
        print("  Testing tools/list consistency...")
        
        tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        # Call both endpoints
        endpoint1_tools_response = await client.post(mcp_endpoint_1_url, json=tools_request)
        endpoint2_tools_response = await client.post(mcp_endpoint_2_url, json=tools_request)
        
        endpoint1_tools = endpoint1_tools_response.json()
        endpoint2_tools = endpoint2_tools_response.json()
        
        # Compare tools lists
        assert endpoint1_tools["result"]["tools"] == endpoint2_tools["result"]["tools"]
        print(f"    ✓ Both endpoints return identical tools list: {len(endpoint1_tools['result']['tools'])} tools")
        
        # Test resources/list consistency
        print("  Testing resources/list consistency...")
        
        resources_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/list",
            "params": {}
        }
        
        endpoint1_resources_response = await client.post(mcp_endpoint_1_url, json=resources_request)
        endpoint2_resources_response = await client.post(mcp_endpoint_2_url, json=resources_request)
        
        endpoint1_resources = endpoint1_resources_response.json()
        endpoint2_resources = endpoint2_resources_response.json()
        
        # Compare resources lists
        assert endpoint1_resources["result"]["resources"] == endpoint2_resources["result"]["resources"]
        print(f"    ✓ Both endpoints return identical resources list: {len(endpoint1_resources['result']['resources'])} resources")
        
        # Test prompts/list consistency
        print("  Testing prompts/list consistency...")
        
        prompts_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "prompts/list",
            "params": {}
        }
        
        endpoint1_prompts_response = await client.post(mcp_endpoint_1_url, json=prompts_request)
        endpoint2_prompts_response = await client.post(mcp_endpoint_2_url, json=prompts_request)
        
        endpoint1_prompts = endpoint1_prompts_response.json()
        endpoint2_prompts = endpoint2_prompts_response.json()
        
        # Compare prompts lists
        assert endpoint1_prompts["result"]["prompts"] == endpoint2_prompts["result"]["prompts"]
        print(f"    ✓ Both endpoints return identical prompts list: {len(endpoint1_prompts['result']['prompts'])} prompts")
        
        # Test tool execution consistency
        print("  Testing tool execution consistency...")
        
        tool_call_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calculate_tool",
                "arguments": {
                    "operation": "add",
                    "a": 15,
                    "b": 25
                }
            }
        }
        
        endpoint1_tool_response = await client.post(mcp_endpoint_1_url, json=tool_call_request)
        endpoint2_tool_response = await client.post(mcp_endpoint_2_url, json=tool_call_request)
        
        endpoint1_tool_result = endpoint1_tool_response.json()
        endpoint2_tool_result = endpoint2_tool_response.json()
        
        # Compare tool execution results
        assert endpoint1_tool_result["result"] == endpoint2_tool_result["result"]
        print(f"    ✓ Both endpoints return identical tool results: {endpoint1_tool_result['result']['content'][0]['text']}")
        
        # Test resource reading consistency
        print("  Testing resource reading consistency...")
        
        resource_read_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/read",
            "params": {
                "uri": "resource://comprehensive-test"
            }
        }
        
        endpoint1_resource_response = await client.post(mcp_endpoint_1_url, json=resource_read_request)
        endpoint2_resource_response = await client.post(mcp_endpoint_2_url, json=resource_read_request)
        
        endpoint1_resource_result = endpoint1_resource_response.json()
        endpoint2_resource_result = endpoint2_resource_response.json()
        
        # Compare resource content
        assert endpoint1_resource_result["result"] == endpoint2_resource_result["result"]
        print(f"    ✓ Both endpoints return identical resource content: {endpoint1_resource_result['result']['contents'][0]['text'][:50]}...")
        
        # Test prompt reading consistency  
        print("  Testing prompt reading consistency...")
        
        prompt_get_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "prompts/get",
            "params": {
                "name": "comprehensive_prompt",
                "arguments": {
                    "task": "testing",
                    "difficulty": "medium"
                }
            }
        }
        
        endpoint1_prompt_response = await client.post(mcp_endpoint_1_url, json=prompt_get_request)
        endpoint2_prompt_response = await client.post(mcp_endpoint_2_url, json=prompt_get_request)
        
        endpoint1_prompt_result = endpoint1_prompt_response.json()
        endpoint2_prompt_result = endpoint2_prompt_response.json()
        
        # Compare prompt results
        assert endpoint1_prompt_result["result"] == endpoint2_prompt_result["result"]
        print(f"    ✓ Both endpoints return identical prompt responses with same description and messages")
    
    print("\n🎉 **COMPLETE DOUBLE ROUND-TRIP SUCCESS!** 🎉")
    print("✅ PERFECT MCP ECOSYSTEM DEMONSTRATED:")
    print("   1. Original Service -> MCP Endpoint 1: ✓ CONSISTENT")
    print("   2. MCP Endpoint 1 -> MCP App -> Unified Service: ✓ CONSISTENT") 
    print("   3. Unified Service -> MCP Endpoint 2: ✓ CONSISTENT")
    print("   4. MCP Endpoint 1 ≡ MCP Endpoint 2: ✓ IDENTICAL")
    print("\n   🔄 The complete MCP round-trip preserves perfect fidelity!")
    print("   🚀 Hypha's MCP implementation is a lossless, transparent proxy!")
    
    # Clean up - let errors explode!
    await controller.stop(session_info["id"])
    print("✓ MCP app session stopped")
    
    await controller.uninstall(mcp_app_info["id"])
    print("✓ MCP app uninstalled")
    
    await api.unregister_service(original_service_info["id"])
    print("✓ Original service unregistered")
    
    await api.disconnect()


async def test_real_deepwiki_mcp_server_validation(fastapi_server, test_user_token):
    """Test DeepWiki MCP server configuration validation and basic setup."""
    api = await connect_to_server(
        {
            "name": "test client",
            "server_url": WS_SERVER_URL,
            "method_timeout": 30,
            "token": test_user_token,
        }
    )
    
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
    print("✓ DeepWiki MCP server configuration passes validation")
    
    # Install the real MCP server app in stage mode to test configuration without connection
    app_info = await controller.install(
        config=deepwiki_config,
        overwrite=True,
        stage=True,  # Stage mode avoids connection timeout
    )
    
    print(f"✓ DeepWiki MCP app installed in stage mode: {app_info['id']}")
    
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
    
    print("✓ DeepWiki MCP configuration properly stored and validated")
    print("✓ Real MCP server integration infrastructure working correctly")
    
    # Clean up
    await controller.uninstall(app_info["id"])
    await api.disconnect() 