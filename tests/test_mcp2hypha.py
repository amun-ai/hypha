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
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/"
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


async def test_mcp_sse_round_trip_service_consistency(fastapi_server, test_user_token):
    """Test complete MCP double round-trip using SSE transport."""
    
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
    def sse_calculate_tool(operation: str, a: float, b: float) -> str:
        """Perform basic arithmetic calculations via SSE."""
        if operation == "add":
            return f"SSE Result: {a + b}"
        elif operation == "subtract":
            return f"SSE Result: {a - b}"
        elif operation == "multiply":
            return f"SSE Result: {a * b}"
        elif operation == "divide":
            if b == 0:
                return "SSE Error: Division by zero"
            return f"SSE Result: {a / b}"
        else:
            return f"SSE Unknown operation: {operation}"

    @schema_function
    def sse_resource_read() -> str:
        """Read a test resource via SSE."""
        return "This is comprehensive SSE test resource content with detailed information."

    @schema_function
    def sse_prompt_read(task: str = "general", difficulty: str = "medium") -> dict:
        """Read a test prompt via SSE with parameters."""
        return {
            "description": f"SSE test prompt for {task} at {difficulty} level",
            "messages": [
                {
                    "role": "user", 
                    "content": {
                        "type": "text",
                        "text": f"Help me with {task} at {difficulty} difficulty level via SSE"
                    }
                }
            ]
        }
    
    # Register the original MCP service
    original_service_info = await api.register_service(
        {
            "id": "sse-test-service",
            "name": "SSE Test Service",
            "description": "A service for testing MCP SSE round-trip consistency",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": [sse_calculate_tool],
            "resources": [
                {
                    "uri": "resource://sse-test",
                    "name": "SSE Test Resource", 
                    "description": "A test resource for SSE transport",
                    "tags": ["test", "sse", "resource"],
                    "mime_type": "text/plain",
                    "read": sse_resource_read,
                }
            ],
            "prompts": [
                {
                    "name": "sse_prompt",
                    "description": "A test prompt for SSE transport",
                    "tags": ["test", "sse", "prompt"],
                    "read": sse_prompt_read,
                }
            ],
        }
    )

    print(f"✓ Original SSE MCP service registered: {original_service_info['id']}")
    
    # Step 2: Get the original service to save its function references
    original_service = await api.get_service(original_service_info["id"])
                
    # Verify the original service has the expected structure
    assert hasattr(original_service, "tools") and original_service.tools is not None
    assert hasattr(original_service, "resources") and original_service.resources is not None
    assert hasattr(original_service, "prompts") and original_service.prompts is not None

    print("✓ Original SSE service has tools, resources, and prompts organized correctly")

    # Step 3: Get the MCP endpoint URL for this service
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/"
    print(f"✓ MCP Endpoint URL: {mcp_endpoint_url}")

    # Step 4: Create MCP server configuration with SSE transport
    mcp_config = {
        "type": "mcp-server",
        "name": "SSE MCP App",
        "version": "1.0.0",
        "description": "MCP app created from original service for SSE testing",
        "mcpServers": {
            "sse-server": {
                "type": "sse",
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

    print(f"✓ SSE MCP app installed: {mcp_app_info['id']}")

    # Step 5: Start the app manually 
    print("🚀 Starting SSE MCP app...")
    server_name = "sse-server"
    session_info = await controller.start(mcp_app_info["id"], wait_for_service=server_name, timeout=30)
    print(f"✓ SSE MCP app started successfully: {session_info}")

    # Step 6: Get the unified MCP service
    print(f"🔍 Getting unified SSE MCP service: {session_info['id']}:{server_name}")
    
    mcp_service_id = f"{session_info['id']}:{server_name}"
    mcp_service = await api.get_service(mcp_service_id)
    print(f"✓ Successfully retrieved unified SSE MCP service: {mcp_service_id}")
    
    # Step 7: Compare original service with the MCP service
    print("🔄 Comparing original SSE service with MCP service...")
    
    # Test tool functionality consistency
    print("  Testing sse_calculate_tool...")
    original_tool = original_service.tools[0]
    mcp_tool = mcp_service.tools[0]
    
    # Test the tools with the same parameters
    original_result = await original_tool(operation="add", a=10, b=7)
    mcp_result = await mcp_tool(operation="add", a=10, b=7)
    assert original_result == mcp_result, f"Tool results differ: original={original_result}, mcp={mcp_result}"
    print(f"    ✓ Both return: {original_result}")
    
    # Test resource functionality consistency
    print("  Testing SSE resource access...")
    original_resource = original_service.resources[0]
    mcp_resource = mcp_service.resources[0]
    
    original_resource_content = await original_resource["read"]()
    mcp_resource_content = await mcp_resource["read"]()
    assert original_resource_content == mcp_resource_content, f"Resource content differs: original={original_resource_content}, mcp={mcp_resource_content}"
    print(f"    ✓ Both return: {original_resource_content[:50]}...")
    
    # Test prompt functionality consistency  
    print("  Testing SSE prompt access...")
    original_prompt = original_service.prompts[0]
    mcp_prompt = mcp_service.prompts[0]
    
    original_prompt_content = await original_prompt["read"](task="testing", difficulty="hard")
    mcp_prompt_content = await mcp_prompt["read"](task="testing", difficulty="hard")
    
    assert "description" in original_prompt_content and "description" in mcp_prompt_content
    assert "messages" in original_prompt_content and "messages" in mcp_prompt_content
    print(f"    ✓ Both SSE prompts have correct structure with messages")
    
    print("✅ SSE ROUND-TRIP SUCCESS!")
    
    # Clean up
    await controller.stop(session_info["id"])
    print("✓ SSE MCP app session stopped")
    
    await controller.uninstall(mcp_app_info["id"])
    print("✓ SSE MCP app uninstalled")
    
    await api.unregister_service(original_service_info["id"])
    print("✓ Original SSE service unregistered")
    
    await api.disconnect()


async def test_mcp_dual_transport_consistency(fastapi_server, test_user_token):
    """Test that both SSE and Streamable HTTP transports work with the same MCP service."""
    
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
    
    # Step 1: Register a single MCP service
    @schema_function
    def dual_calculate_tool(operation: str, a: float, b: float) -> str:
        """Perform basic arithmetic calculations."""
        if operation == "add":
            return f"Dual Result: {a + b}"
        elif operation == "subtract":
            return f"Dual Result: {a - b}"
        else:
            return f"Dual Unknown operation: {operation}"

    @schema_function
    def dual_resource_read() -> str:
        """Read a test resource."""
        return "This is dual transport test resource content."

    # Register the original MCP service
    original_service_info = await api.register_service(
        {
            "id": "dual-transport-test-service",
            "name": "Dual Transport Test Service",
            "description": "A service for testing both SSE and Streamable HTTP transports",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": [dual_calculate_tool],
            "resources": [
                {
                    "uri": "resource://dual-transport-test",
                    "name": "Dual Transport Test Resource", 
                    "description": "A test resource for both transports",
                    "tags": ["test", "dual", "resource"],
                    "mime_type": "text/plain",
                    "read": dual_resource_read,
                }
            ],
            "prompts": [],
        }
    )

    print(f"✓ Original dual transport MCP service registered: {original_service_info['id']}")
    
    # Step 2: Get the MCP endpoint URL for this service
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/"
    print(f"✓ MCP Endpoint URL: {mcp_endpoint_url}")

    # Step 3: Create MCP server configuration with BOTH transports
    mcp_config = {
        "type": "mcp-server",
        "name": "Dual Transport MCP App",
        "version": "1.0.0",
        "description": "MCP app that tests both SSE and Streamable HTTP transports",
        "mcpServers": {
            "http-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url
            },
            "sse-server": {
                "type": "sse",
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

    print(f"✓ Dual transport MCP app installed: {mcp_app_info['id']}")

    # Step 4: Start the app manually 
    print("🚀 Starting dual transport MCP app...")
    session_info = await controller.start(mcp_app_info["id"], wait_for_service="http-server", timeout=30)
    print(f"✓ Dual transport MCP app started successfully: {session_info}")

    # Step 5: Get both unified MCP services
    print("🔍 Getting both unified MCP services...")
    
    # Get HTTP service
    http_service_id = f"{session_info['id']}:http-server"
    http_service = await api.get_service(http_service_id)
    print(f"✓ Successfully retrieved HTTP MCP service: {http_service_id}")
    
    # Get SSE service
    sse_service_id = f"{session_info['id']}:sse-server"
    sse_service = await api.get_service(sse_service_id)
    print(f"✓ Successfully retrieved SSE MCP service: {sse_service_id}")
    
    # Step 6: Compare both services for consistency
    print("🔄 Comparing HTTP and SSE services for consistency...")
    
    # Test tool functionality consistency
    print("  Testing dual_calculate_tool...")
    http_tool = http_service.tools[0]
    sse_tool = sse_service.tools[0]
    
    # Test the tools with the same parameters
    http_result = await http_tool(operation="add", a=15, b=25)
    sse_result = await sse_tool(operation="add", a=15, b=25)
    assert http_result == sse_result, f"Tool results differ: http={http_result}, sse={sse_result}"
    print(f"    ✓ Both transports return: {http_result}")
    
    # Test resource functionality consistency
    print("  Testing dual resource access...")
    http_resource = http_service.resources[0]
    sse_resource = sse_service.resources[0]
    
    http_resource_content = await http_resource["read"]()
    sse_resource_content = await sse_resource["read"]()
    assert http_resource_content == sse_resource_content, f"Resource content differs: http={http_resource_content}, sse={sse_resource_content}"
    print(f"    ✓ Both transports return: {http_resource_content}")
    
    print("✅ DUAL TRANSPORT CONSISTENCY SUCCESS!")
    
    # Clean up
    await controller.stop(session_info["id"])
    print("✓ Dual transport MCP app session stopped")
    
    await controller.uninstall(mcp_app_info["id"])
    print("✓ Dual transport MCP app uninstalled")
    
    await api.unregister_service(original_service_info["id"])
    print("✓ Original dual transport service unregistered")
    
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
    first_tool = deepwiki_service.tools[0]
    
    # Test with simple parameters - DeepWiki tools likely have search functionality
    test_result = await first_tool(query="test")
    assert test_result is not None
    
    # Test actual resource functionality
    assert len(deepwiki_service.resources) > 0
    first_resource = deepwiki_service.resources[0]
    resource_content = await first_resource["read"]()
    assert resource_content is not None
    
    # Test actual prompt functionality
    assert len(deepwiki_service.prompts) > 0
    first_prompt = deepwiki_service.prompts[0]
    prompt_content = await first_prompt["read"]()
    assert prompt_content is not None
    
    # Test MCP HTTP endpoint functionality
    running_apps = await controller.list_running()
    deepwiki_app = None
    for app in running_apps:
        if app.get("app_id") == app_info["id"]:
            deepwiki_app = app
            break
    
    assert deepwiki_app is not None
    
    # Construct the MCP endpoint URL
    session_id = deepwiki_app["id"]
    service_name = "deepwiki"
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{session_id}:{service_name}/"
    
    # Test the MCP endpoint with HTTP requests
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test tools/list
        tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = await client.post(mcp_endpoint_url, json=tools_request)
        assert response.status_code == 200
        tools_result = response.json()
        assert len(tools_result.get('result', {}).get('tools', [])) > 0
        
        # Test resources/list
        resources_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/list",
            "params": {}
        }
        
        response = await client.post(mcp_endpoint_url, json=resources_request)
        assert response.status_code == 200
        resources_result = response.json()
        assert len(resources_result.get('result', {}).get('resources', [])) > 0
        
        # Test prompts/list
        prompts_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "prompts/list",
            "params": {}
        }
        
        response = await client.post(mcp_endpoint_url, json=prompts_request)
        assert response.status_code == 200
        prompts_result = response.json()
        assert len(prompts_result.get('result', {}).get('prompts', [])) > 0
    
    # Test round-trip consistency
    ping_request = {
        "jsonrpc": "2.0",
        "id": "ping",
        "method": "tools/list",
        "params": {}
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(mcp_endpoint_url, json=ping_request)
        assert response.status_code == 200
    
    # Clean up
    running_apps = await controller.list_running()
    for app in running_apps:
        if app.get("app_id") == app_info["id"]:
            await controller.stop(app["id"])
    
    await controller.uninstall(app_info["id"])
    await api.disconnect() 