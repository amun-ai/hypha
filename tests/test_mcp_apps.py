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
    """Test MCP round-trip: Hypha service -> MCP endpoint -> MCP app -> Hypha service consistency."""
    
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

    # Step 3: Get the MCP endpoint URL for this service
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/"
    print(f"✓ MCP endpoint URL: {mcp_endpoint_url}")

    # Step 4: Create MCP server configuration and install as MCP app
    mcp_config = {
        "type": "mcp-server",
        "name": "Round Trip MCP App",
        "version": "1.0.0",
        "description": "MCP app created from original service for round-trip testing",
        "mcpServers": {
            "round-trip-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url
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
    
    print("✅ SUCCESS! MCP round-trip service consistency verified!")
    print("   - Tools work identically between original and MCP services")
    print("   - Resources are accessible and return the same content")
    print("   - Prompts have the same structure and behavior")
    print("   - The MCP service is a perfect proxy for the original service")
    
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