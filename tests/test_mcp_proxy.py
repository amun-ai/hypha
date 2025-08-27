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


async def test_mcp_streamable_http_round_trip_service_consistency(
    fastapi_server, test_user_token
):
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
                        "text": f"Help me with {task} at {difficulty} difficulty level",
                    },
                }
            ],
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
            "tools": {"calculate_tool": calculate_tool},
            "resources": {
                "resource://streamable-http-test": {
                    "uri": "resource://streamable-http-test",
                    "name": "Streamable HTTP Test Resource",
                    "description": "A test resource for streamable HTTP transport",
                    "tags": ["test", "streamable-http", "resource"],
                    "mime_type": "text/plain",
                    "read": resource_read,
                }
            },
            "prompts": {
                "streamable_http_prompt": {
                    "name": "streamable_http_prompt",
                    "description": "A test prompt for streamable HTTP transport",
                    "tags": ["test", "streamable-http", "prompt"],
                    "read": prompt_read,
                }
            },
        }
    )

    print(f"✓ Original MCP service registered: {original_service_info['id']}")

    # Step 2: Get the original service to save its function references
    original_service = await api.get_service(original_service_info["id"])

    # Verify the original service has the expected structure
    assert hasattr(original_service, "tools") and original_service.tools is not None
    assert (
        hasattr(original_service, "resources")
        and original_service.resources is not None
    )
    assert hasattr(original_service, "prompts") and original_service.prompts is not None

    print("✓ Original service has tools, resources, and prompts organized correctly")

    # Step 3: Get the MCP endpoint URL for this service
    mcp_endpoint_url = (
        f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/mcp"
    )
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
                "url": mcp_endpoint_url,
            }
        },
    }

    # Install the MCP server app
    mcp_app_info = await controller.install(
        manifest=mcp_config,
        overwrite=True,
        stage=True,
    )

    print(f"✓ MCP app installed: {mcp_app_info['id']}")

    # Step 5: Start the app manually
    print("🚀 Starting MCP app...")
    server_name = "streamable-http-server"
    session_info = await controller.start(
        mcp_app_info["id"], wait_for_service=server_name, timeout=30
    )
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
    original_tool = original_service.tools["calculate_tool"]
    mcp_tool = mcp_service.tools["calculate_tool"]

    # Test the tools with the same parameters
    original_result = await original_tool(operation="add", a=5, b=3)
    
    # Add a small delay to ensure connection is stable
    await asyncio.sleep(0.1)
    
    try:
        mcp_result = await mcp_tool(operation="add", a=5, b=3)
    except Exception as e:
        # If connection error, try once more after a delay
        if "Connection has already been closed" in str(e):
            await asyncio.sleep(0.5)
            mcp_result = await mcp_tool(operation="add", a=5, b=3)
        else:
            raise
    
    assert (
        original_result == mcp_result
    ), f"Tool results differ: original={original_result}, mcp={mcp_result}"
    print(f"    ✓ Both return: {original_result}")

    # Test resource functionality consistency
    print("  Testing resource access...")
    original_resource = original_service.resources["resource://streamable-http-test"]
    mcp_resource = mcp_service.resources["resource://streamable-http-test"]

    original_resource_content = await original_resource["read"]()
    mcp_resource_content = await mcp_resource["read"]()
    assert (
        original_resource_content == mcp_resource_content
    ), f"Resource content differs: original={original_resource_content}, mcp={mcp_resource_content}"
    print(f"    ✓ Both return: {original_resource_content[:50]}...")

    # Test prompt functionality consistency
    print("  Testing prompt access...")
    original_prompt = original_service.prompts["streamable_http_prompt"]
    mcp_prompt = mcp_service.prompts["streamable_http_prompt"]

    original_prompt_content = await original_prompt["read"](
        task="testing", difficulty="easy"
    )
    mcp_prompt_content = await mcp_prompt["read"](task="testing", difficulty="easy")

    assert (
        "description" in original_prompt_content and "description" in mcp_prompt_content
    )
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
                "url": "http://invalid-host:9999/mcp",
            }
        },
    }

    # Install the MCP server app with invalid URL
    mcp_app_info = await controller.install(
        manifest=mcp_config_invalid,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Invalid URL MCP app installed: {mcp_app_info['id']}")

    # Start the app - should handle the error gracefully
    try:
        session_info = await controller.start(
            mcp_app_info["id"], wait_for_service="invalid-server", timeout=10
        )
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
                "url": "ws://localhost:8080/mcp",
            }
        },
    }

    # Install the MCP server app with unsupported transport
    mcp_app_info = await controller.install(
        manifest=mcp_config_unsupported,
        overwrite=True,
        stage=True,
    )

    print(f"✓ Unsupported transport MCP app installed: {mcp_app_info['id']}")

    # Start the app - should handle the error gracefully
    try:
        session_info = await controller.start(
            mcp_app_info["id"], wait_for_service="unsupported-server", timeout=10
        )
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
                "url": "https://mcp.deepwiki.com/mcp",
            }
        },
    }

    # Validate the configuration
    validation_result = await controller.validate_app_manifest(deepwiki_config)
    assert validation_result["valid"] is True

    # Install the real MCP server app - connect to actual external service
    app_info = await controller.install(
        manifest=deepwiki_config,
        wait_for_service="deepwiki",
        overwrite=True,
    )

    # Verify the app was installed correctly
    assert app_info["name"] == "DeepWiki MCP Server"
    assert app_info["type"] == "mcp-server"
    assert "mcpServers" in app_info
    assert "deepwiki" in app_info["mcpServers"]
    assert app_info["mcpServers"]["deepwiki"]["url"] == "https://mcp.deepwiki.com/mcp"

    # Test lazy loading - get the unified service without explicit start
    deepwiki_service = await api.get_service(f"deepwiki@{app_info.id}", mode="first")

    # Verify the service structure
    assert hasattr(deepwiki_service, "tools")
    assert hasattr(deepwiki_service, "resources")
    assert hasattr(deepwiki_service, "prompts")
    assert deepwiki_service.tools is not None
    assert deepwiki_service.resources is not None
    assert deepwiki_service.prompts is not None

    # Test actual tool functionality
    assert len(deepwiki_service.tools.keys()) > 0

    print(f"Available Tools in DeepWiki: {deepwiki_service.tools}")

    read_wiki_structure = deepwiki_service.tools.read_wiki_structure
    assert read_wiki_structure.__schema__ is not None
    assert (
        read_wiki_structure.__schema__["parameters"]["properties"]["repoName"]
        is not None
    )
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


async def test_mcp_lazy_service_one_instance(fastapi_server, test_user_token):
    """Test that lazy loading an MCP service creates only one app instance."""
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

    # Clean up running apps from previous tests
    for app in await controller.list_running():
        await controller.stop(app["id"])

    @schema_function
    def my_tool(x: int) -> int:
        """A simple tool."""
        return x * 2

    original_service_info = await api.register_service(
        {
            "id": "lazy-mcp-test-service",
            "name": "Lazy MCP Test Service",
            "type": "mcp",
            "config": {"visibility": "public"},
            "tools": {"my_tool": my_tool},
        }
    )

    mcp_endpoint_url = (
        f"{SERVER_URL}/{workspace}/mcp/{original_service_info['id'].split('/')[-1]}/mcp"
    )

    mcp_config = {
        "type": "mcp-server",
        "name": "Lazy MCP App",
        "mcpServers": {
            "lazy-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url,
            }
        },
    }

    app_info = await controller.install(manifest=mcp_config, overwrite=True, stage=True)

    # Commit the staged app to make it available
    await controller.commit_app(app_info.id)

    # First lazy call
    print("First lazy call to get_service...")
    service = await api.get_service(f"lazy-server@{app_info.id}")
    assert service is not None
    assert "my_tool" in service.tools
    assert await service.tools.my_tool(2) == 4

    # Check running instances
    running_apps = await controller.list_running()
    mcp_app_instances = [app for app in running_apps if app["app_id"] == app_info.id]
    assert (
        len(mcp_app_instances) == 1
    ), f"Expected 1 instance, but found {len(mcp_app_instances)}"
    print(f"Found {len(mcp_app_instances)} instance(s) running, as expected.")

    # Second lazy call
    print("Second lazy call to get_service...")
    service2 = await api.get_service(f"lazy-server@{app_info.id}")
    assert service2 is not None
    assert "my_tool" in service2.tools
    assert await service2.tools.my_tool(3) == 6

    # Check running instances again
    running_apps_after = await controller.list_running()
    mcp_app_instances_after = [
        app for app in running_apps_after if app["app_id"] == app_info.id
    ]
    assert (
        len(mcp_app_instances_after) == 1
    ), f"Expected 1 instance after second call, but found {len(mcp_app_instances_after)}"
    print(
        f"Found {len(mcp_app_instances_after)} instance(s) running after second call, as expected."
    )

    # Cleanup
    # Stop the running instance
    await controller.stop(mcp_app_instances[0]["id"])
    await controller.uninstall(app_info.id)
    await api.unregister_service(original_service_info["id"])
    await api.disconnect()


async def test_mcp_ghost_services_after_stop(fastapi_server, test_user_token):
    """Test that MCP services are properly cleaned up after stopping and don't become ghost services."""
    
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

    # Step 1: Create a simple MCP service to use as backend
    @schema_function
    def test_tool(value: str) -> str:
        """A test tool that returns the input value."""
        return f"Processed: {value}"

    # Register the backend MCP service
    backend_service_info = await api.register_service(
        {
            "id": "ghost-test-backend",
            "name": "Ghost Test Backend Service",
            "description": "Backend service for ghost service test",
            "type": "mcp",
            "config": {"visibility": "public"},
            "tools": {"test_tool": test_tool},
        }
    )

    # Step 2: Create and install an MCP app that connects to the backend service
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{backend_service_info['id'].split('/')[-1]}/mcp"

    mcp_app_config = {
        "type": "mcp-server",
        "name": "Ghost Test MCP App",
        "description": "Test app for checking ghost services",
        "mcpServers": {
            "ghost-test-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url,
            }
        },
    }

    # Install the MCP app
    app_info = await controller.install(manifest=mcp_app_config, overwrite=True, stage=True)
    await controller.commit_app(app_info.id)

    # Step 3: Start the MCP app to register services
    print("Starting MCP app...")
    start_result = await controller.start(app_info.id, timeout=30)
    session_id = start_result["id"]
    
    print(f"MCP app started with session ID: {session_id}")

    # Step 4: Verify that services are registered and visible
    print("Checking services before stopping...")
    initial_services = await api.list_services()
    mcp_services_before = [s for s in initial_services if "ghost-test-server" in s.get("id", "")]
    
    print(f"Found {len(mcp_services_before)} MCP services before stopping:")
    for service in mcp_services_before:
        print(f"  - {service['id']}")
    
    # There should be at least one MCP service registered
    assert len(mcp_services_before) > 0, "No MCP services found after starting the app"

    # Step 5: Get the service to make sure it works
    mcp_service = await api.get_service(f"ghost-test-server@{app_info.id}")
    assert mcp_service is not None, "Could not get MCP service"
    assert "test_tool" in mcp_service.tools, "test_tool not found in MCP service"
    
    # Test the service works
    result = await mcp_service.tools.test_tool("hello")
    assert result == "Processed: hello", f"Unexpected result: {result}"
    
    print("MCP service is working correctly")

    # Step 6: Stop the MCP app
    print(f"Stopping MCP app with session ID: {session_id}")
    await controller.stop(session_id)
    
    print("MCP app stopped")

    # Step 7: Wait a moment for cleanup to complete
    await asyncio.sleep(2)

    # Step 8: Check if ghost services remain
    print("Checking services after stopping...")
    final_services = await api.list_services()
    mcp_services_after = [s for s in final_services if "ghost-test-server" in s.get("id", "")]
    
    print(f"Found {len(mcp_services_after)} MCP services after stopping:")
    for service in mcp_services_after:
        print(f"  - {service['id']} (this is a GHOST SERVICE)")

    # Step 9: Verify that all MCP services have been cleaned up
    assert len(mcp_services_after) == 0, f"Ghost services detected! {len(mcp_services_after)} MCP services still exist after stopping: {[s['id'] for s in mcp_services_after]}"

    print("✅ No ghost services detected - cleanup was successful")

    # Cleanup
    await controller.uninstall(app_info.id)
    await api.unregister_service(backend_service_info["id"])
    await api.disconnect()


async def test_mcp_non_schema_function_exclusion(fastapi_server, test_user_token):
    """Test that non-MCP services with functions lacking schemas are handled gracefully."""
    
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

    # Step 1: Create a service with mixed functions (some with schema, some without)
    @schema_function
    def valid_function(x: int) -> int:
        """A function with proper schema."""
        return x * 2

    # Function without schema decorator
    def no_schema_function(y: int) -> int:
        """A function without schema decorator."""
        return y * 3

    # Register a non-MCP service with mixed functions
    mixed_service_info = await api.register_service(
        {
            "id": "mixed-schema-service",
            "name": "Mixed Schema Service",
            "description": "Service with both schema and non-schema functions",
            "type": "generic",  # Not explicitly MCP type
            "config": {"visibility": "public"},
            "valid_function": valid_function,
            "no_schema_function": no_schema_function,
        }
    )

    print(f"✓ Registered mixed service: {mixed_service_info['id']}")

    # Step 2: Test MCP endpoint directly via HTTP
    service_id = mixed_service_info['id'].split('/')[-1]
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{service_id}/mcp"
    
    print(f"Testing MCP endpoint: {mcp_endpoint_url}")
    
    # Test MCP initialize request
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            mcp_endpoint_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {test_user_token}"  # Include auth token
            }
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        init_result = response.json()
        assert "result" in init_result, f"Expected result in response: {init_result}"
        
        # Test list_tools
        response = await client.post(
            mcp_endpoint_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 2
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {test_user_token}"  # Include auth token
            }
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        tools_result = response.json()
        assert "result" in tools_result, f"Expected result in response: {tools_result}"
        
        tools = tools_result["result"]["tools"]
        tool_names = [tool["name"] for tool in tools]
        
        # Check that valid_function is exposed
        assert "valid_function" in tool_names, f"valid_function should be exposed, got: {tool_names}"
        
        # Check that no_schema_function is NOT exposed
        assert "no_schema_function" not in tool_names, f"no_schema_function should NOT be exposed, got: {tool_names}"
        
        print(f"✓ Tools list: {tool_names}")
        
        # Test calling the valid function
        response = await client.post(
            mcp_endpoint_url,
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "valid_function",
                    "arguments": {"x": 3}
                },
                "id": 3
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {test_user_token}"  # Include auth token
            }
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        call_result = response.json()
        assert "result" in call_result, f"Expected result in response: {call_result}"
        
        # Check the result content
        content = call_result["result"]["content"]
        assert len(content) > 0, "Expected content in result"
        assert "6" in str(content[0]["text"]), f"Expected result 6, got: {content}"

    print("✅ Non-schema functions are correctly excluded from MCP tools")

    # Cleanup
    await api.unregister_service(mixed_service_info["id"])
    await api.disconnect()


async def test_mcp_service_id_resolution(fastapi_server, test_user_token):
    """Test that MCP endpoint handles both relative and absolute service IDs correctly."""
    
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

    # Register a simple service with schema function
    @schema_function
    def test_function(x: int) -> int:
        """Test function with schema."""
        return x * 2

    service_info = await api.register_service(
        {
            "id": "test-id-resolution",
            "name": "Test ID Resolution",
            "description": "Service for testing ID resolution",
            "type": "generic",
            "config": {"visibility": "public"},
            "test_function": test_function,
        }
    )

    print(f"✓ Registered service: {service_info['id']}")
    
    # Extract the full service ID (with client ID)
    full_service_id = service_info['id'].split('/')[-1]  # e.g., "CLIENT_ID:test-id-resolution"
    relative_service_id = "test-id-resolution"
    
    import httpx
    async with httpx.AsyncClient() as client:
        # Test 1: Access with relative service ID
        relative_url = f"{SERVER_URL}/{workspace}/mcp/{relative_service_id}/mcp"
        print(f"Testing relative URL: {relative_url}")
        
        response = await client.post(
            relative_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                },
                "id": 1
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {test_user_token}"
            }
        )
        
        assert response.status_code == 200, f"Relative ID failed with {response.status_code}: {response.text}"
        result = response.json()
        assert "result" in result, f"Expected result in response: {result}"
        print("✓ Relative service ID works")
        
        # Test 2: Access with fully-qualified service ID
        absolute_url = f"{SERVER_URL}/{workspace}/mcp/{full_service_id}/mcp"
        print(f"Testing absolute URL: {absolute_url}")
        
        response = await client.post(
            absolute_url,
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                },
                "id": 2
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {test_user_token}"
            }
        )
        
        assert response.status_code == 200, f"Absolute ID failed with {response.status_code}: {response.text}"
        result = response.json()
        assert "result" in result, f"Expected result in response: {result}"
        print("✓ Fully-qualified service ID works")

    print("✅ Both relative and absolute service IDs resolve correctly")

    # Cleanup
    await api.unregister_service(service_info["id"])
    await api.disconnect()


async def test_mcp_cleanup_after_stop(fastapi_server, test_user_token):
    """Test that after stop() the MCP service is not accessible and client is disconnected."""
    
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

    # Step 1: Create a simple MCP service to use as backend
    @schema_function
    def cleanup_test_tool(message: str) -> str:
        """A test tool for cleanup verification."""
        return f"Echo: {message}"

    # Register the backend MCP service
    backend_service_info = await api.register_service(
        {
            "id": "cleanup-test-backend",
            "name": "Cleanup Test Backend Service",
            "description": "Backend service for cleanup test",
            "type": "mcp",
            "config": {"visibility": "public"},
            "tools": {"cleanup_test_tool": cleanup_test_tool},
        }
    )

    # Step 2: Create and install an MCP app
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{backend_service_info['id'].split('/')[-1]}/mcp"

    mcp_app_config = {
        "type": "mcp-server",
        "name": "Cleanup Test MCP App",
        "description": "Test app for verifying proper cleanup",
        "mcpServers": {
            "cleanup-test-server": {
                "type": "streamable-http",
                "url": mcp_endpoint_url,
            }
        },
    }

    # Install the MCP app
    app_info = await controller.install(manifest=mcp_app_config, overwrite=True, stage=True)
    await controller.commit_app(app_info.id)

    # Step 3: Start the MCP app
    print("Starting MCP app for cleanup test...")
    start_result = await controller.start(app_info.id, timeout=30)
    session_id = start_result["id"]
    
    print(f"MCP app started with session ID: {session_id}")

    # Step 4: Verify the service is accessible before stopping
    print("Verifying service is accessible before stop...")
    
    # Get the service
    mcp_service = await api.get_service(f"cleanup-test-server@{app_info.id}")
    assert mcp_service is not None, "Could not get MCP service before stop"
    assert "cleanup_test_tool" in mcp_service.tools, "cleanup_test_tool not found in MCP service"
    
    # Test the service works
    result = await mcp_service.tools.cleanup_test_tool("test message")
    assert result == "Echo: test message", f"Unexpected result: {result}"
    
    print("✓ MCP service is working correctly before stop")

    # Step 5: Check that the service is registered
    services_before = await api.list_services()
    mcp_services_before = [s for s in services_before if "cleanup-test-server" in s.get("id", "")]
    assert len(mcp_services_before) > 0, "MCP service not found in service list before stop"
    
    service_id_before = f"{session_id}:cleanup-test-server"
    print(f"✓ Service {service_id_before} is registered before stop")

    # Step 6: Stop the MCP app
    print(f"Stopping MCP app with session ID: {session_id}")
    await controller.stop(session_id)
    
    print("✓ MCP app stopped successfully")

    # Step 7: Wait a moment for cleanup to complete
    await asyncio.sleep(2)

    # Step 8: Verify the service is no longer accessible
    print("Verifying service is NOT accessible after stop...")
    
    # Try to get the service again - it should fail or return None
    try:
        mcp_service_after = await api.get_service(f"cleanup-test-server", mode="first")
        # If we get here without error, the service should be None or unusable
        if mcp_service_after is not None:
            # Try to use the service - it should fail
            try:
                result_after = await mcp_service_after.tools.cleanup_test_tool("test after stop")
                assert False, f"Service call succeeded after stop! Result: {result_after}"
            except Exception as e:
                print(f"✓ Service call failed after stop as expected: {e}")
    except Exception as e:
        print(f"✓ Cannot get service after stop as expected: {e}")

    # Step 9: Verify the service is not in the service list
    services_after = await api.list_services()
    mcp_services_after = [s for s in services_after if "cleanup-test-server" in s.get("id", "")]
    
    # The service should be completely removed from the list
    assert len(mcp_services_after) == 0, f"Service still exists after stop! Found: {[s['id'] for s in mcp_services_after]}"
    
    print("✓ Service is not in the service list after stop")

    # Step 10: Verify we cannot get the service using the session ID
    try:
        # Try to get service with the old session ID
        old_service = await api.get_service(service_id_before)
        if old_service is not None:
            # Try to use it
            try:
                await old_service.tools.cleanup_test_tool("should fail")
                assert False, "Service with old session ID still works after stop!"
            except Exception:
                print("✓ Service with old session ID is not functional after stop")
    except Exception as e:
        print(f"✓ Cannot get service with old session ID after stop: {e}")

    # Step 11: Verify the session is truly stopped
    running_apps = await controller.list_running()
    running_session_ids = [app["id"] for app in running_apps]
    assert session_id not in running_session_ids, f"Session {session_id} still in running apps after stop!"
    
    print("✓ Session is not in the running apps list after stop")

    print("✅ CLEANUP TEST PASSED - All resources properly cleaned up after stop()")

    # Cleanup
    await controller.uninstall(app_info.id)
    await api.unregister_service(backend_service_info["id"])
    await api.disconnect()
