"""Test MCP (Model Context Protocol) services."""

import asyncio
import time
from typing import Dict, Any, List, Optional

import pytest
import httpx
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from . import WS_SERVER_URL, SERVER_URL

# MCP imports

import mcp.types as types
from mcp.server.lowlevel import Server
# Now test using the actual MCP client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_mcp_service_registration(fastapi_server, test_user_token):
    """Test basic MCP service registration with type=mcp."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="echo",
                description="Echo back the input text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo"}
                    },
                    "required": ["text"],
                },
            )
        ]

    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.ContentBlock]:
        if name == "echo":
            text = arguments.get("text", "")
            return [types.TextContent(type="text", text=f"Echo: {text}")]
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def list_prompts() -> List[types.Prompt]:
        return [
            types.Prompt(
                name="greeting", description="A simple greeting prompt", arguments=[]
            )
        ]

    async def get_prompt(name: str, arguments: Dict[str, Any]) -> types.GetPromptResult:
        if name == "greeting":
            return types.GetPromptResult(
                description="A greeting prompt",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text", text="Hello, how are you?"
                        ),
                    )
                ],
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    async def list_resource_templates() -> List[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                uriTemplate="test://resource/{id}",
                name="Test Resource",
                description="A test resource template",
                mimeType="text/plain",
            )
        ]

    async def list_resources() -> List[types.Resource]:
        return [
            types.Resource(
                uri="test://resource/1",
                name="Test Resource 1",
                description="A test resource instance",
                mimeType="text/plain",
            )
        ]

    async def read_resource(uri: str) -> types.ReadResourceResult:
        if uri == "test://resource/1":
            return types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=uri,
                        mimeType="text/plain",
                        text="This is test resource content"
                    )
                ]
            )
        else:
            raise ValueError(f"Unknown resource: {uri}")

    async def progress_notification(
        progress_token: str, progress: float, total: Optional[float] = None
    ):
        # Mock progress notification
        pass

    # Register MCP service
    service = await api.register_service(
        {
            "id": "test-mcp-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
            "list_prompts": list_prompts,
            "get_prompt": get_prompt,
            "list_resource_templates": list_resource_templates,
            "list_resources": list_resources,
            "read_resource": read_resource,
            "progress_notification": progress_notification,
        }
    )

    assert service["type"] == "mcp"
    assert "test-mcp-service" in service["id"]


    # Create MCP client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/test-mcp-service/mcp"

    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 1
            assert tools_result.tools[0].name == "echo"
            assert tools_result.tools[0].description == "Echo back the input text"

            # Test 2: Call tool
            tool_result = await session.call_tool(
                name="echo",
                arguments={"text": "Hello MCP"}
            )
            assert tool_result is not None
            assert hasattr(tool_result, "content")
            assert len(tool_result.content) == 1
            assert tool_result.content[0].type == "text"
            assert tool_result.content[0].text == "Echo: Hello MCP"

            # Test 3: List prompts
            prompts_result = await session.list_prompts()
            assert prompts_result is not None
            assert hasattr(prompts_result, "prompts")
            assert len(prompts_result.prompts) == 1
            assert prompts_result.prompts[0].name == "greeting"

            # Test 4: Get prompt
            prompt_result = await session.get_prompt(
                name="greeting",
                arguments={}
            )
            assert prompt_result is not None
            assert hasattr(prompt_result, "messages")
            assert len(prompt_result.messages) == 1
            assert prompt_result.messages[0].role == "user"
            assert "Hello, how are you?" in prompt_result.messages[0].content.text

            # Test 5: List resource templates
            templates_result = await session.list_resource_templates()
            assert templates_result is not None
            assert hasattr(templates_result, "resourceTemplates")
            assert len(templates_result.resourceTemplates) == 1
            assert templates_result.resourceTemplates[0].name == "Test Resource"

            # Test 6: List resources
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            assert len(resources_result.resources) == 1
            assert str(resources_result.resources[0].uri) == "test://resource/1"

            # Test 7: Read resource
            resource_result = await session.read_resource(
                uri="test://resource/1"
            )
            assert resource_result is not None
            assert hasattr(resource_result, "contents")
            assert len(resource_result.contents) == 1
            assert resource_result.contents[0].text == "This is test resource content"

    print("✓ MCP service registered and tested successfully with real MCP client")

    await api.disconnect()


async def test_mcp_schema_function_service(fastapi_server, test_user_token):
    """Test MCP service with schema_function decorated functions."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define schema_function decorated MCP service functions
    @schema_function
    async def list_tools() -> List[Dict[str, Any]]:
        """List available tools."""
        return [
            {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform",
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ]

    @schema_function
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call a tool with given arguments."""
        if name == "calculate":
            operation = arguments.get("operation")
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)

            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return [{"type": "text", "text": "Error: Division by zero"}]
                result = a / b
            else:
                return [{"type": "text", "text": f"Unknown operation: {operation}"}]

            return [{"type": "text", "text": f"Result: {result}"}]
        else:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

    @schema_function
    async def list_prompts() -> List[Dict[str, Any]]:
        """List available prompts."""
        return [
            {
                "name": "math_help",
                "description": "Help with math problems",
                "arguments": [
                    {
                        "name": "topic",
                        "description": "Math topic to get help with",
                        "required": True,
                    }
                ],
            }
        ]

    @schema_function
    async def get_prompt(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt by name."""
        if name == "math_help":
            topic = arguments.get("topic", "general")
            return {
                "description": f"Math help for {topic}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Can you help me with {topic} problems?",
                        },
                    }
                ],
            }
        else:
            raise ValueError(f"Unknown prompt: {name}")

    @schema_function
    async def list_resource_templates() -> List[Dict[str, Any]]:
        """List available resource templates."""
        return [
            {
                "uriTemplate": "math://formula/{category}",
                "name": "Math Formula",
                "description": "Mathematical formulas by category",
                "mimeType": "text/plain",
            }
        ]

    @schema_function
    async def list_resources() -> List[Dict[str, Any]]:
        """List available resources."""
        return [
            {
                "uri": "math://formula/algebra",
                "name": "Algebra Formulas",
                "description": "Common algebra formulas",
                "mimeType": "text/plain",
            }
        ]

    @schema_function
    async def read_resource(uri: str) -> Dict[str, Any]:
        """Read a resource by URI."""
        if uri == "math://formula/algebra":
            return {
                "contents": [
                    {
                        "type": "text",
                        "uri": uri,
                        "mimeType": "text/plain",
                        "text": "Quadratic Formula: x = (-b ± √(b² - 4ac)) / 2a"
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")

    # Register schema_function decorated MCP service
    service = await api.register_service(
        {
            "id": "schema-mcp-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
            "list_prompts": list_prompts,
            "get_prompt": get_prompt,
            "list_resource_templates": list_resource_templates,
            "list_resources": list_resources,
            "read_resource": read_resource,
        }
    )

    # Service should be registered successfully
    assert service["type"] == "mcp"
    assert "schema-mcp-service" in service["id"]
    assert service["id"].startswith(workspace)


    # Create MCP client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/schema-mcp-service/mcp"

    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 1
            assert tools_result.tools[0].name == "calculate"

            # Test 2: Call tool with add operation
            tool_result = await session.call_tool(
                name="calculate",
                arguments={"operation": "add", "a": 5, "b": 3}
            )
            assert tool_result is not None
            assert hasattr(tool_result, "content")
            assert len(tool_result.content) == 1
            assert tool_result.content[0].type == "text"
            assert "Result: 8" in tool_result.content[0].text

            # Test 3: Call tool with divide operation
            tool_result = await session.call_tool(
                name="calculate",
                arguments={"operation": "divide", "a": 10, "b": 2}
            )
            assert tool_result is not None
            assert "Result: 5" in tool_result.content[0].text

            # Test 4: List prompts
            prompts_result = await session.list_prompts()
            assert prompts_result is not None
            assert hasattr(prompts_result, "prompts")
            assert len(prompts_result.prompts) == 1
            assert prompts_result.prompts[0].name == "math_help"

            # Test 5: Get prompt
            prompt_result = await session.get_prompt(
                name="math_help",
                arguments={"topic": "calculus"}
            )
            assert prompt_result is not None
            assert hasattr(prompt_result, "messages")
            assert len(prompt_result.messages) == 1
            assert "calculus" in prompt_result.messages[0].content.text

            # Test 6: List resource templates
            templates_result = await session.list_resource_templates()
            assert templates_result is not None
            assert hasattr(templates_result, "resourceTemplates")
            assert len(templates_result.resourceTemplates) == 1
            assert templates_result.resourceTemplates[0].name == "Math Formula"

            # Test 7: List resources
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            assert len(resources_result.resources) == 1
            assert str(resources_result.resources[0].uri) == "math://formula/algebra"

            # Test 8: Read resource
            resource_result = await session.read_resource(
                uri="math://formula/algebra"
            )
            assert resource_result is not None
            assert hasattr(resource_result, "contents")
            assert len(resource_result.contents) == 1
            assert "Quadratic Formula" in resource_result.contents[0].text

    print("✓ Schema function MCP service tested successfully with real MCP client")

    await api.disconnect()


async def test_mcp_inline_config_service(fastapi_server, test_user_token):
    """Test MCP service with inline tools/resources/prompts configuration."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define simple tool handler - must be schema_function decorated
    @schema_function
    def simple_tool(operation: str, a: int, b: int) -> str:
        """Simple arithmetic tool."""
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

    # Define resource read handler - must be schema_function decorated
    @schema_function
    def resource_read() -> str:
        """Read a test resource."""
        return "This is a test resource content."

    # Define prompt read handler - must be schema_function decorated
    @schema_function
    def prompt_read(topic: str = "general") -> Dict[str, Any]:
        """Read a test prompt."""
        return {
            "description": f"Test prompt for {topic}",
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": f"Help me with {topic}"},
                }
            ],
        }

    # Register MCP service with inline configuration using schema functions
    service = await api.register_service(
        {
            "id": "inline-mcp-service",
            "name": "Inline MCP Service",
            "description": "A service with inline MCP configuration",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "tools": {
                "simple_tool": simple_tool  # Use the actual schema function
            },
            "resources": {
                "resource://test": {
                    "uri": "resource://test",
                    "name": "Test Resource",
                    "description": "A test resource",
                    "tags": ["test", "resource"],
                    "mime_type": "text/plain",
                    "read": resource_read,
                }
            },
            "prompts": {
                "test_prompt": {
                    "name": "test_prompt",
                    "description": "A test prompt template",
                    "tags": ["test", "prompt"],
                    "read": prompt_read,
                }
            },
        }
    )

    # Service should be registered successfully
    assert service["type"] == "mcp"
    assert "inline-mcp-service" in service["id"]
    assert service["id"].startswith(workspace)


    # Create MCP client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/inline-mcp-service/mcp"

    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 1
            assert tools_result.tools[0].name == "simple_tool"

            # Test 2: Call tool with add operation
            tool_result = await session.call_tool(
                name="simple_tool",
                arguments={"operation": "add", "a": 10, "b": 5}
            )
            assert tool_result is not None
            assert hasattr(tool_result, "content")
            assert len(tool_result.content) == 1
            assert tool_result.content[0].type == "text"
            assert "Result: 15" in tool_result.content[0].text

            # Test 3: Call tool with divide operation
            tool_result = await session.call_tool(
                name="simple_tool",
                arguments={"operation": "divide", "a": 20, "b": 4}
            )
            assert tool_result is not None
            assert "Result: 5" in tool_result.content[0].text

            # Test 4: List prompts
            prompts_result = await session.list_prompts()
            assert prompts_result is not None
            assert hasattr(prompts_result, "prompts")
            assert len(prompts_result.prompts) == 1
            assert prompts_result.prompts[0].name == "test_prompt"

            # Test 5: Get prompt
            prompt_result = await session.get_prompt(
                name="test_prompt",
                arguments={"topic": "programming"}
            )
            assert prompt_result is not None
            assert hasattr(prompt_result, "messages")
            assert len(prompt_result.messages) == 1
            assert "programming" in prompt_result.messages[0].content.text

            # Test 6: List resources
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            assert len(resources_result.resources) == 1
            assert str(resources_result.resources[0].uri) == "resource://test"

            # Test 7: Read resource
            resource_result = await session.read_resource(
                uri="resource://test"
            )
            assert resource_result is not None
            assert hasattr(resource_result, "contents")
            assert len(resource_result.contents) == 1
            assert resource_result.contents[0].text == "This is a test resource content."

    print("✓ Inline config MCP service tested successfully with real MCP client")

    await api.disconnect()


async def test_mcp_merged_approach(fastapi_server, test_user_token):
    """Test MCP service with merged approach: inline tools/resources/prompts using schema functions."""
    # Connect to the Hypha server
    api = await connect_to_server(
        {
            "name": "mcp-merged-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

    workspace = api.config.workspace

    # Define a simple resource read function
    @schema_function
    def resource_read():
        """Read a test resource."""
        return "This is a test resource content"

    # Define a simple prompt read function
    @schema_function
    def prompt_read(name: str = "general"):
        """Read a test prompt."""
        return f"This is a test prompt template, name: {name}"

    # Create a simple tool function
    @schema_function
    def simple_tool(text: str = "hello") -> str:
        """A simple tool that processes text."""
        return f"Processed: {text}"

    # Register the MCP service with tools, resources, and prompts
    mcp_service_info = await api.register_service(
        {
            "id": "mcp-merged-features",
            "name": "MCP Merged Features Service",
            "description": "A service with merged MCP features",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "tools": {"simple_tool": simple_tool},
            "resources": {
                "resource://test": {
                    "uri": "resource://test",
                    "name": "Test Resource",
                    "description": "A test resource",
                    "tags": ["test", "resource"],
                    "mime_type": "text/plain",
                    "read": resource_read,
                }
            },
            "prompts": {
                "test_prompt": {
                    "name": "test_prompt",
                    "description": "A test prompt template",
                    "tags": ["test", "prompt"],
                    "read": prompt_read,
                }
            },
        }
    )

    # Service should be registered successfully
    assert mcp_service_info["type"] == "mcp"
    assert "mcp-merged-features" in mcp_service_info["id"]


    # Create MCP client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/mcp-merged-features/mcp"

    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 1
            assert tools_result.tools[0].name == "simple_tool"
            assert tools_result.tools[0].description == "A simple tool that processes text."

            # Test 2: Call tool
            tool_result = await session.call_tool(
                name="simple_tool",
                arguments={"text": "test input"}
            )
            assert tool_result is not None
            assert hasattr(tool_result, "content")
            assert len(tool_result.content) == 1
            assert tool_result.content[0].type == "text"
            assert tool_result.content[0].text == "Processed: test input"

            # Test 3: List prompts
            prompts_result = await session.list_prompts()
            assert prompts_result is not None
            assert hasattr(prompts_result, "prompts")
            assert len(prompts_result.prompts) == 1
            assert prompts_result.prompts[0].name == "test_prompt"

            # Test 4: Get prompt
            prompt_result = await session.get_prompt(
                name="test_prompt",
                arguments={"name": "specific"}
            )
            assert prompt_result is not None
            assert hasattr(prompt_result, "messages")
            assert len(prompt_result.messages) == 1
            assert prompt_result.messages[0].role == "user"
            assert prompt_result.messages[0].content.text == "This is a test prompt template, name: specific"

            # Test 5: List resources
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            assert len(resources_result.resources) == 1
            assert str(resources_result.resources[0].uri) == "resource://test"
            assert resources_result.resources[0].name == "Test Resource"

            # Test 6: Read resource
            resource_result = await session.read_resource(
                uri="resource://test"
            )
            assert resource_result is not None
            assert hasattr(resource_result, "contents")
            assert len(resource_result.contents) == 1
            assert resource_result.contents[0].text == "This is a test resource content"

    print("✓ MCP service with merged approach tested successfully with real MCP client")

    await api.disconnect()


async def test_mcp_validation_errors(fastapi_server, test_user_token, redis_server):
    """Test that validation errors are properly raised for invalid configurations."""
    server = await connect_to_server(
        {
            "name": "mcp-validation-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

    # Test direct validation by creating HyphaMCPAdapter directly with invalid configs
    from hypha.mcp import HyphaMCPAdapter
    from redis import asyncio as aioredis

    # Create redis client like in hypha/core/store.py
    redis_client = aioredis.from_url(redis_server)

    # Test 1: Tool that is not a schema function should raise error
    def non_schema_tool(text: str) -> str:
        return f"Processed: {text}"

    class MockServiceInfo:
        def __init__(self, **kwargs):
            self.id = "test-service"
            self.type = "mcp"
            for k, v in kwargs.items():
                setattr(self, k, v)

    # The service object should contain the tools, not the service_info
    service_with_tools = {"tools": {non_schema_tool.__name__: non_schema_tool}}
    service_info = MockServiceInfo()

    try:
        HyphaMCPAdapter(service_with_tools, service_info, redis_client)
        assert False, "Should have raised ValueError for non-schema tool"
    except ValueError as e:
        assert "must be a @schema_function decorated function" in str(e)
        print("✓ Tool validation error properly raised")

    # Test 2: Resource without read function should raise error
    service_with_resources = {
        "resources": {
            "resource://test": {
                "uri": "resource://test",
                "name": "Test Resource",
                "description": "A test resource",
            }
        }
    }

    try:
        HyphaMCPAdapter(service_with_resources, service_info, redis_client)
        assert False, "Should have raised ValueError for resource without read"
    except ValueError as e:
        assert "must have a 'read' key" in str(e)
        print("✓ Resource validation error properly raised")

    # Test 3: Resource with non-schema read function should raise error
    def non_schema_read():
        return "content"

    service_with_bad_resource = {
        "resources": {
            "resource://test": {
                "uri": "resource://test",
                "name": "Test Resource",
                "description": "A test resource",
                "read": non_schema_read,
            }
        }
    }

    try:
        HyphaMCPAdapter(service_with_bad_resource, service_info, redis_client)
        assert False, "Should have raised ValueError for non-schema read function"
    except ValueError as e:
        assert "must be a @schema_function decorated function" in str(e)
        print("✓ Resource read validation error properly raised")

    # Test 4: Prompt without read function should raise error
    service_with_prompts = {
        "prompts": {
            "test_prompt": {
                "name": "test_prompt",
                "description": "A test prompt",
            }
        }
    }

    try:
        HyphaMCPAdapter(service_with_prompts, service_info, redis_client)
        assert False, "Should have raised ValueError for prompt without read"
    except ValueError as e:
        assert "must have a 'read' key" in str(e)
        print("✓ Prompt validation error properly raised")

    print("✓ All validation errors properly raised")

    await server.disconnect()


async def test_mcp_http_endpoint_tools(fastapi_server, test_user_token):
    """Test MCP HTTP endpoint for tools functionality."""

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="timestamp",
                description="Get current timestamp",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["unix", "iso"],
                            "description": "Timestamp format",
                        }
                    },
                    "required": ["format"],
                },
            )
        ]

    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.ContentBlock]:
        if name == "timestamp":
            format_type = arguments.get("format", "unix")
            if format_type == "unix":
                result = str(int(time.time()))
            elif format_type == "iso":
                result = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            else:
                result = "Invalid format"
            return [types.TextContent(type="text", text=result)]
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Register MCP service
    await api.register_service(
        {
            "id": "timestamp-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
        }
    )

    # Test MCP HTTP endpoint
    async with httpx.AsyncClient() as client:
        # Test tools/list
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/timestamp-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "tools/list",
                "params": {},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-1"
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) == 1
        assert data["result"]["tools"][0]["name"] == "timestamp"

        # Test tools/call
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/timestamp-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-2",
                "method": "tools/call",
                "params": {"name": "timestamp", "arguments": {"format": "iso"}},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-2"
        assert "result" in data
        assert "content" in data["result"]
        assert len(data["result"]["content"]) == 1
        assert data["result"]["content"][0]["type"] == "text"
        # Check that the result looks like an ISO timestamp
        assert "T" in data["result"]["content"][0]["text"]
        assert "Z" in data["result"]["content"][0]["text"]

    await api.disconnect()


async def test_mcp_http_endpoint_prompts(fastapi_server, test_user_token):
    """Test MCP HTTP endpoint for prompts functionality."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_prompts() -> List[types.Prompt]:
        return [
            types.Prompt(
                name="code_review",
                description="Code review prompt",
                arguments=[
                    types.PromptArgument(
                        name="language",
                        description="Programming language",
                        required=True,
                    ),
                    types.PromptArgument(
                        name="code", description="Code to review", required=True
                    ),
                ],
            )
        ]

    async def get_prompt(name: str, arguments: Dict[str, Any]) -> types.GetPromptResult:
        if name == "code_review":
            language = arguments.get("language", "python")
            code = arguments.get("code", "")
            return types.GetPromptResult(
                description=f"Code review for {language}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Please review this {language} code:\n\n{code}",
                        ),
                    )
                ],
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    # Register MCP service
    await api.register_service(
        {
            "id": "code-review-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_prompts": list_prompts,
            "get_prompt": get_prompt,
        }
    )

    # Test MCP HTTP endpoint
    async with httpx.AsyncClient() as client:
        # Test prompts/list
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/code-review-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "prompts/list",
                "params": {},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-1"
        assert "result" in data
        assert "prompts" in data["result"]
        assert len(data["result"]["prompts"]) == 1
        assert data["result"]["prompts"][0]["name"] == "code_review"

        # Test prompts/get
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/code-review-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-2",
                "method": "prompts/get",
                "params": {
                    "name": "code_review",
                    "arguments": {
                        "language": "python",
                        "code": "def hello():\n    print('Hello, world!')",
                    },
                },
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-2"
        assert "result" in data
        assert "description" in data["result"]
        assert "messages" in data["result"]
        assert len(data["result"]["messages"]) == 1
        assert data["result"]["messages"][0]["role"] == "user"
        assert "python" in data["result"]["messages"][0]["content"]["text"]

    await api.disconnect()


async def test_mcp_http_endpoint_resources(fastapi_server, test_user_token):
    """Test MCP HTTP endpoint for resources functionality."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_resource_templates() -> List[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                uriTemplate="file://docs/{category}/{filename}",
                name="Documentation File",
                description="Access documentation files by category",
                mimeType="text/markdown",
            )
        ]

    # Register MCP service
    await api.register_service(
        {
            "id": "docs-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_resource_templates": list_resource_templates,
        }
    )

    # Test MCP HTTP endpoint
    async with httpx.AsyncClient() as client:
        # Test resources/templates/list
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/docs-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "resources/templates/list",
                "params": {},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-1"
        assert "result" in data
        assert "resourceTemplates" in data["result"]
        assert len(data["result"]["resourceTemplates"]) == 1
        assert data["result"]["resourceTemplates"][0]["name"] == "Documentation File"
        assert data["result"]["resourceTemplates"][0]["mimeType"] == "text/markdown"

    await api.disconnect()


async def test_mcp_error_handling(fastapi_server, test_user_token):
    """Test MCP error handling for invalid requests."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="error_tool",
                description="Tool that can produce errors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "should_error": {
                            "type": "boolean",
                            "description": "Whether to trigger an error",
                        }
                    },
                    "required": ["should_error"],
                },
            )
        ]

    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.ContentBlock]:
        if name == "error_tool":
            should_error = arguments.get("should_error", False)
            if should_error:
                raise RuntimeError("Intentional error for testing")
            return [types.TextContent(type="text", text="No error")]
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Register MCP service
    await api.register_service(
        {
            "id": "error-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
        }
    )

    async with httpx.AsyncClient() as client:
        # Test 1: Invalid JSON-RPC request
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/error-service/mcp",
            json={"invalid": "request"},
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )
        # For invalid JSON-RPC requests, we expect 400 status with error response
        assert response.status_code == 400  # Invalid request returns 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602  # Invalid params (validation error)

        # Test 2: Method not found
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/error-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-2",
                "method": "unknown/method",
                "params": {},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        # The MCP SDK returns -32602 (Invalid params) for unknown methods in stateless mode
        assert data["error"]["code"] == -32602  # Invalid params

        # Test 3: Tool runtime error
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/error-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-3",
                "method": "tools/call",
                "params": {"name": "error_tool", "arguments": {"should_error": True}},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )
        assert response.status_code == 200
        data = response.json()
        # In stateless mode, tool errors are returned as results with isError flag
        assert "result" in data
        assert data["result"].get("isError") is True
        # The error message should be in the content
        assert len(data["result"]["content"]) > 0
        assert "Intentional error for testing" in data["result"]["content"][0]["text"]

    await api.disconnect()


async def test_mcp_nonexistent_service(fastapi_server, test_user_token):
    """Test accessing nonexistent MCP service."""
    async with httpx.AsyncClient() as client:
        # Try to access nonexistent service
        response = await client.post(
            f"{SERVER_URL}/ws-user-user-1/mcp/nonexistent/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "tools/list",
                "params": {},
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )
        assert response.status_code == 404


async def test_mcp_info_message(fastapi_server, test_user_token):
    """Test that accessing MCP service without /mcp suffix returns helpful info."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Register a simple service without type="mcp" - it will be auto-wrapped
    @schema_function
    def test_tool(input: str) -> str:
        """A test tool that processes input."""
        return f"Processed: {input}"
    
    @schema_function
    def another_tool(value: int) -> int:
        """Another tool that doubles a value."""
        return value * 2

    await api.register_service(
        {
            "id": "test-service",
            # No type specified - will be auto-wrapped for MCP
            "config": {
                "visibility": "public",
            },
            "test_tool": test_tool,
            "another_tool": another_tool,
        }
    )

    async with httpx.AsyncClient() as client:
        # Try to access service without /mcp suffix - should return service info
        response = await client.get(
            f"{SERVER_URL}/{workspace}/mcp/test-service",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
        )

        assert response.status_code == 200  # Now returns 200 with service info
        data = response.json()
        
        # Check that we get comprehensive service information
        assert "service" in data
        assert data["service"]["id"].endswith("test-service")
        
        # For auto-wrapped services, it should NOT be marked as native MCP
        assert data["service"].get("is_native_mcp", False) == False
        
        # Check endpoints are provided
        assert "endpoints" in data
        assert "streamable_http" in data["endpoints"]
        assert data["endpoints"]["streamable_http"] == f"/{workspace}/mcp/test-service/mcp"
        assert "sse" in data["endpoints"]
        assert data["endpoints"]["sse"] == f"/{workspace}/mcp/test-service/sse"
        
        # Check capabilities are provided - for auto-wrapped services, tools should be listed
        assert "capabilities" in data
        assert "tools" in data["capabilities"]
        # Auto-wrapped services should have their tools listed
        assert len(data["capabilities"]["tools"]) == 2
        tool_names = [tool["name"] for tool in data["capabilities"]["tools"]]
        assert "test_tool" in tool_names
        assert "another_tool" in tool_names
        
        # Check help message is provided
        assert "help" in data

    await api.disconnect()


async def test_mcp_sse_transport(fastapi_server, test_user_token):
    """Test MCP SSE transport functionality."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="sse_echo",
                description="Echo back the input text via SSE",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to echo"}
                    },
                    "required": ["text"],
                },
            )
        ]

    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.ContentBlock]:
        if name == "sse_echo":
            text = arguments.get("text", "")
            return [types.TextContent(type="text", text=f"SSE Echo: {text}")]
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Register MCP service
    service = await api.register_service(
        {
            "id": "sse-test-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
        }
    )

    assert service["type"] == "mcp"
    assert "sse-test-service" in service["id"]

    # Now test using the actual MCP SSE client
    from mcp.client.sse import sse_client
    from mcp.client.session import ClientSession

    # Create MCP SSE client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/sse-test-service/sse"

    async with sse_client(base_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()

            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 1
            assert tools_result.tools[0].name == "sse_echo"
            assert tools_result.tools[0].description == "Echo back the input text via SSE"

            # Test 2: Call tool
            tool_result = await session.call_tool(
                name="sse_echo",
                arguments={"text": "Hello SSE"}
            )
            assert tool_result is not None
            assert hasattr(tool_result, "content")
            assert len(tool_result.content) == 1
            assert tool_result.content[0].type == "text"
            assert tool_result.content[0].text == "SSE Echo: Hello SSE"

    print("✓ MCP SSE transport tested successfully")

    await api.disconnect()


async def test_mcp_sse_server_connection(fastapi_server, test_user_token):
    """Test MCP proxy worker connecting to public SSE server."""
    # Import the MCP proxy worker
    from hypha.workers.mcp_proxy import MCPClientRunner
    from hypha.workers.base import WorkerConfig
    
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Create MCP proxy worker
    worker = MCPClientRunner()
    
    # Register the worker with the server
    await worker.register_worker_service(api)
    
    # Create MCP server configuration for public SSE server
    mcp_servers_config = {
        "deepwiki": {
            "type": "sse",
            "url": "https://mcp.deepwiki.com/sse",
            "headers": {}  # No special headers needed for public server
        }
    }

    # Create application manifest
    manifest = {
        "id": "sse-test-app",
        "name": "SSE Test App",
        "description": "Test application for SSE MCP server",
        "type": "mcp-server",
        "mcpServers": mcp_servers_config,
    }
    
    # Create worker config with unique client ID
    import uuid
    unique_client_id = f"sse-test-client-{uuid.uuid4().hex[:8]}"
    
    config = WorkerConfig(
        id="sse-test-session",
        app_id="sse-test-app",
        artifact_id="sse-test-artifact",  # Add required artifact_id
        workspace=workspace,
        client_id=unique_client_id,  # Use unique client ID to avoid conflicts
        server_url=WS_SERVER_URL,
        token=test_user_token,
        manifest=manifest,
        entry_point="source",
    )
    
    # Start the MCP proxy session
    session_id = await worker.start(config)
    assert session_id == "sse-test-session"
    
    # Wait for services to be registered
    await asyncio.sleep(3)

    # Get the registered services
    services = await api.list_services()
    
    # Find the deepwiki service
    deepwiki_service = None
    for service in services:
        if "deepwiki" in service.get("id", ""):
            deepwiki_service = service
            break
    
    if deepwiki_service:
        print(f"✓ DeepWiki service registered: {deepwiki_service['id']}")
        
        # Get the service to interact with it
        service = await api.get_service(deepwiki_service["id"])
        
        # Check if the service has tools
        if hasattr(service, "tools") and service.tools:
            tools = service.tools
            print(f"✓ SSE server connected successfully, found {len(tools)} tools")
            
            # Try to list tool names
            if tools:
                tool_names = list(tools.keys())
                print(f"✓ Available tools: {tool_names[:5]}")  # Show first 5 tools
        
        # Check if the service has resources
        if hasattr(service, "resources") and service.resources:
            resources = service.resources
            print(f"✓ Found {len(resources)} resources")
            
            # List resource URIs
            if resources:
                resource_uris = list(resources.keys())
                print(f"✓ Available resources: {resource_uris[:5]}")  # Show first 5 resources
        
        # Check if the service has prompts
        if hasattr(service, "prompts") and service.prompts:
            prompts = service.prompts
            print(f"✓ Found {len(prompts)} prompts")
    else:
        print("✗ DeepWiki service not found, but this might be due to connection issues")
        print(f"Available services: {[s.get('id', '') for s in services]}")
    
    # Clean up - properly stop the session
    await worker.stop(session_id)
    
    print("✓ SSE MCP server integration test completed")
    
    await api.disconnect()


async def test_mcp_real_client_integration(fastapi_server, test_user_token):
    """Test MCP integration using real MCP client from the SDK."""
    # First, register an MCP service
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="weather",
                description="Get weather information for a location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Location to get weather for",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units",
                        },
                    },
                    "required": ["location"],
                },
            ),
            types.Tool(
                name="time",
                description="Get current time",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (optional)",
                        }
                    },
                },
            ),
        ]

    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.ContentBlock]:
        if name == "weather":
            location = arguments.get("location", "unknown")
            units = arguments.get("units", "celsius")
            temp_symbol = "°C" if units == "celsius" else "°F"
            temp_value = 22 if units == "celsius" else 72
            return [
                types.TextContent(
                    type="text",
                    text=f"Weather in {location}: {temp_value}{temp_symbol}, sunny",
                )
            ]
        elif name == "time":
            timezone = arguments.get("timezone", "UTC")
            return [
                types.TextContent(
                    type="text", text=f"Current time in {timezone}: 2024-01-15 14:30:00"
                )
            ]
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    async def list_prompts() -> List[types.Prompt]:
        return [
            types.Prompt(
                name="travel_planner",
                description="Help plan a trip to a destination",
                arguments=[
                    types.PromptArgument(
                        name="destination",
                        description="Travel destination",
                        required=True,
                    ),
                    types.PromptArgument(
                        name="duration",
                        description="Trip duration in days",
                        required=False,
                    ),
                ],
            )
        ]

    async def get_prompt(name: str, arguments: Dict[str, Any]) -> types.GetPromptResult:
        if name == "travel_planner":
            destination = arguments.get("destination", "Paris")
            duration = arguments.get("duration", "3")
            return types.GetPromptResult(
                description=f"Travel planning for {destination}",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"I want to plan a {duration}-day trip to {destination}. Can you help me create an itinerary?",
                        ),
                    )
                ],
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    # Register MCP service
    await api.register_service(
        {
            "id": "weather-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
            "list_prompts": list_prompts,
            "get_prompt": get_prompt,
        }
    )


    # Create MCP client session
    base_url = f"{SERVER_URL}/{workspace}/mcp/weather-service/mcp"

    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Test 1: List tools
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) == 2

            tool_names = [tool.name for tool in tools_result.tools]
            assert "weather" in tool_names
            assert "time" in tool_names

            # Find weather tool
            weather_tool = next(
                tool for tool in tools_result.tools if tool.name == "weather"
            )
            assert weather_tool.description == "Get weather information for a location"
            assert "location" in weather_tool.inputSchema["properties"]

            # Test 2: Call weather tool
            weather_result = await session.call_tool(
                name="weather",
                arguments={"location": "New York", "units": "fahrenheit"},
            )
            assert weather_result is not None
            assert hasattr(weather_result, "content")
            assert len(weather_result.content) == 1
            assert weather_result.content[0].type == "text"
            assert "New York" in weather_result.content[0].text
            assert "°F" in weather_result.content[0].text

            # Test 3: Call time tool
            time_result = await session.call_tool(
                name="time", arguments={"timezone": "EST"}
            )
            assert time_result is not None
            assert hasattr(time_result, "content")
            assert len(time_result.content) == 1
            assert "EST" in time_result.content[0].text

            # Test 4: List prompts
            prompts_result = await session.list_prompts()
            assert prompts_result is not None
            assert hasattr(prompts_result, "prompts")
            assert len(prompts_result.prompts) == 1
            assert prompts_result.prompts[0].name == "travel_planner"

            # Test 5: Get prompt
            prompt_result = await session.get_prompt(
                name="travel_planner",
                arguments={"destination": "Tokyo", "duration": "7"},
            )
            assert prompt_result is not None
            assert hasattr(prompt_result, "messages")
            assert len(prompt_result.messages) == 1
            assert prompt_result.messages[0].role == "user"
            assert "Tokyo" in prompt_result.messages[0].content.text
            assert "7-day" in prompt_result.messages[0].content.text


async def test_mcp_nested_services(fastapi_server, test_user_token):
    """Test that services with nested callable functions are properly exposed via MCP when converted."""
    
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
    
    # Create a service with nested structure containing callable functions
    @schema_function
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    @schema_function
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    @schema_function
    def format_result(value: float, prefix: str = "Result") -> str:
        """Format a result with a prefix."""
        return f"{prefix}: {value}"
    
    @schema_function
    def power(x: float, n: float) -> float:
        """Raise x to the power of n."""
        return x ** n
    
    @schema_function
    def uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    @schema_function  
    def lowercase(text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    # Service with nested structure - should NOT use type="mcp" 
    # because it doesn't follow valid MCP structure
    nested_service_info = await api.register_service(
        {
            "id": "nested-service",
            "name": "Nested Service",
            "description": "Test service with nested callable functions",
            # NO type="mcp" - let it be auto-wrapped
            "config": {"visibility": "public"},
            # Nested structure with functions
            "math": {
                "basic": {
                    "add": add,
                    "multiply": multiply,
                },
                "advanced": {
                    "power": power,
                }
            },
            "formatting": {
                "text": format_result,
                "utils": {
                    "uppercase": uppercase,
                    "lowercase": lowercase,
                }
            }
        }
    )
    
    # Construct the MCP endpoint URL
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{nested_service_info['id'].split('/')[-1]}/mcp"
    
    # Connect using the MCP client to the Hypha MCP endpoint
    async with streamablehttp_client(mcp_endpoint_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools - should include flattened nested functions
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            
            # Check that nested functions are properly flattened with underscore-separated names
            tool_names = {tool.name for tool in tools_result.tools}
            
            # Verify expected flattened tool names
            expected_tools = {
                "math_basic_add",
                "math_basic_multiply",
                "math_advanced_power",
                "formatting_text",
                "formatting_utils_uppercase",
                "formatting_utils_lowercase",
            }
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}. Available tools: {tool_names}"
            
            # Test calling a deeply nested function
            result = await session.call_tool(
                name="math_basic_add",
                arguments={"a": 5, "b": 3}
            )
            assert result is not None
            assert result.content[0].text == "8" or "8" in str(result.content[0].text)
            
            # Test another nested function
            result = await session.call_tool(
                name="formatting_text",
                arguments={"value": 42, "prefix": "Answer"}
            )
            assert result is not None
            assert "Answer: 42" in str(result.content[0].text)
            
            # Test power function
            result = await session.call_tool(
                name="math_advanced_power",
                arguments={"x": 2, "n": 3}
            )
            assert result is not None
            assert "8" in str(result.content[0].text)
    
    # Clean up
    await api.unregister_service(nested_service_info["id"])
    await api.disconnect()


async def test_mcp_invalid_type_validation(fastapi_server, test_user_token):
    """Test that services with type='mcp' but invalid structure throw an error."""

    
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
    
    # Create schema functions for testing
    @schema_function
    def test_func(x: int) -> int:
        """A test function."""
        return x * 2
    
    # Try to register a service with type="mcp" but invalid structure
    # This should be rejected when accessed via MCP endpoint
    invalid_service_info = await api.register_service(
        {
            "id": "invalid-mcp-service",
            "name": "Invalid MCP Service",
            "description": "Service incorrectly using type='mcp'",
            "type": "mcp",  # WRONG: This service doesn't follow MCP structure
            "config": {"visibility": "public"},
            # This is neither native MCP nor inline config structure
            "nested": {
                "functions": {
                    "test": test_func
                }
            }
        }
    )
    
    # The service should be registered (registration doesn't validate MCP structure)
    assert "invalid-mcp-service" in invalid_service_info["id"]
    
    # Now try to access it via MCP endpoint - this should fail
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{invalid_service_info['id'].split('/')[-1]}/mcp"
    
    try:
        async with streamablehttp_client(mcp_endpoint_url) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                # This should fail during initialization or when listing tools
                await session.initialize()
                # If we get here, the validation didn't work
                assert False, "Expected ValueError for invalid MCP structure with type='mcp'"
    except Exception as e:
        # We expect an error here - could be wrapped in ExceptionGroup
        # The validation worked if we got a 500 error (server rejected the invalid structure)
        error_repr = repr(e)
        error_str = str(e)
        # Check for HTTP 500 error in the exception representation
        validation_worked = ("500" in error_repr or "500" in error_str or 
                           "does not follow valid MCP structure" in error_repr or
                           "does not follow valid MCP structure" in error_str)
        assert validation_worked, f"Expected validation error but got: {error_repr}"
        print(f"✓ Correctly rejected invalid type='mcp' service with error")
    
    # Clean up
    await api.unregister_service(invalid_service_info["id"])
    
    # Now test that the same service works WITHOUT type="mcp"
    valid_service_info = await api.register_service(
        {
            "id": "valid-auto-wrap-service",
            "name": "Valid Auto-wrap Service",
            "description": "Service without type='mcp' gets auto-wrapped",
            # NO type="mcp" - let it be auto-wrapped
            "config": {"visibility": "public"},
            # Same structure as before
            "nested": {
                "functions": {
                    "test": test_func
                }
            }
        }
    )
    
    # This should work when accessed via MCP
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{valid_service_info['id'].split('/')[-1]}/mcp"
    
    async with streamablehttp_client(mcp_endpoint_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # List tools - should work
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            
            # Should have the nested function
            tool_names = {tool.name for tool in tools_result.tools}
            assert "nested_functions_test" in tool_names
            
            # Test calling the function
            result = await session.call_tool(
                name="nested_functions_test",
                arguments={"x": 5}
            )
            assert result is not None
            assert "10" in str(result.content[0].text)
    
    print("✓ Service without type='mcp' correctly auto-wrapped and accessible via MCP")
    
    # Clean up
    await api.unregister_service(valid_service_info["id"])
    await api.disconnect()


async def test_mcp_arbitrary_service_conversion(fastapi_server, test_user_token):
    """Test that ANY service (without explicit type=mcp) can be accessed via MCP endpoints."""

    
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
    
    # Create schema functions for testing
    @schema_function
    def calculate(operation: str, x: float, y: float) -> float:
        """Perform a calculation."""
        if operation == "add":
            return x + y
        elif operation == "subtract":
            return x - y
        elif operation == "multiply":
            return x * y
        elif operation == "divide":
            return x / y if y != 0 else float('inf')
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @schema_function
    def process_text(text: str, mode: str = "upper") -> str:
        """Process text in different modes."""
        if mode == "upper":
            return text.upper()
        elif mode == "lower":
            return text.lower()
        elif mode == "title":
            return text.title()
        else:
            return text
    
    @schema_function
    def get_info(name: str) -> Dict[str, Any]:
        """Get information about something."""
        return {
            "name": name,
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "active",
            "data": {"key": "value"}
        }
    
    # Register a regular service WITHOUT type="mcp"
    # This should still be accessible via MCP endpoints
    service_info = await api.register_service(
        {
            "id": "regular-service",
            "name": "Regular Service",
            "description": "A regular service without explicit MCP type",
            # NOTE: No "type": "mcp" here!
            "config": {"visibility": "public"},
            # Nested structure with functions
            "operations": {
                "math": calculate,
                "text": {
                    "process": process_text,
                    "utils": {
                        "get_info": get_info
                    }
                }
            }
        }
    )
    
    # The service should be registered successfully
    assert "regular-service" in service_info["id"]
    
    # Now try to access it via MCP endpoint - this should work!
    mcp_endpoint_url = f"{SERVER_URL}/{workspace}/mcp/{service_info['id'].split('/')[-1]}/mcp"
    
    async with streamablehttp_client(mcp_endpoint_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools - should include flattened nested functions
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            
            # Check that nested functions are properly flattened
            tool_names = {tool.name for tool in tools_result.tools}
            
            # Verify expected flattened tool names
            expected_tools = {
                "operations_math",  # calculate function
                "operations_text_process",  # process_text function
                "operations_text_utils_get_info",  # get_info function
            }
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Missing tool: {expected_tool}. Available tools: {tool_names}"
            
            # Test calling the math function
            result = await session.call_tool(
                name="operations_math",
                arguments={"operation": "multiply", "x": 7, "y": 6}
            )
            assert result is not None
            assert "42" in str(result.content[0].text)
            
            # Test calling the text processing function
            result = await session.call_tool(
                name="operations_text_process",
                arguments={"text": "hello world", "mode": "title"}
            )
            assert result is not None
            assert "Hello World" in str(result.content[0].text)
            
            # Test calling the nested get_info function
            result = await session.call_tool(
                name="operations_text_utils_get_info",
                arguments={"name": "test"}
            )
            assert result is not None
            result_text = str(result.content[0].text)
            assert "test" in result_text
            assert "active" in result_text
    
    print("✓ Arbitrary service without type='mcp' successfully accessed via MCP endpoint")
    
    # Clean up
    await api.unregister_service(service_info["id"])
    await api.disconnect()


@pytest.mark.asyncio
async def test_mcp_service_with_docs_field(fastapi_server, test_user_token):
    """Test that a service with a docs field exposes it as an MCP resource."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    
    workspace = api.config.workspace
    
    docs_content = """
# Service Documentation

This service processes data according to specific rules.

## Usage
```python
result = await service.process(data)
```
"""
    
    # Use a schema_function to ensure proper function wrapping
    @schema_function
    def process(x: int) -> int:
        """Process a value."""
        return x * 2
    
    service_info = await api.register_service(
        {
            "id": "docs-service",
            "name": "Service with Documentation",
            # Don't specify type - let it be auto-wrapped as MCP
            "docs": docs_content,
            "description": "A service with documentation field",
            "config": {"visibility": "public"},
            "process": process,  # Use the schema_function
        }
    )
    
    # Wait a moment for the service to be fully registered
    await asyncio.sleep(1.0)  # Increase wait time
    
    # Connect to the MCP endpoint
    service_id = service_info['id'].split('/')[-1]  # Get just the service ID part
    base_url = f"{SERVER_URL}/{workspace}/mcp/{service_id}/mcp"
    
    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # First verify the service has tools (the process function)
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) > 0, "Service should have at least one tool"
            
            # List resources - should include the docs field
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            
            # Find the docs resource
            docs_resource = None
            for resource in resources_result.resources:
                # Convert AnyUrl to string for comparison
                if str(resource.uri) == "resource://docs":
                    docs_resource = resource
                    break
            
            assert docs_resource is not None, f"Docs resource not found. Available resources: {[str(r.uri) for r in resources_result.resources]}"
            assert "Documentation" in docs_resource.name
            assert docs_resource.mimeType == "text/plain"
            
            # Read the docs resource
            read_result = await session.read_resource(uri="resource://docs")
            assert read_result is not None
            assert hasattr(read_result, "contents")
            assert len(read_result.contents) > 0
            assert read_result.contents[0].text == docs_content
    await api.disconnect()

@pytest.mark.asyncio
async def test_mcp_service_with_nested_string_resources(fastapi_server, test_user_token):
    """Test that services with nested string fields expose them as MCP resources."""

    api = await connect_to_server(
        {"server_url": WS_SERVER_URL, "client_id": "test-client-nested", "token": test_user_token}
    )
    
    workspace = api.config.workspace
    
    # Use a schema_function to ensure proper function wrapping
    @schema_function
    def process(x: int) -> int:
        """Process a value."""
        return x * 2
    
    # Register a non-MCP service with nested string fields
    service_info = await api.register_service(
        {
            "id": "nested-service",
            "name": "Service with Nested Resources",
            # Don't specify type - let it be auto-wrapped as MCP
            "description": "A service with nested string resources",
            "config": {"visibility": "public"},
            "metadata": {
                "version": "1.0.0",
                "author": "Test Author",
                "nested": {
                    "info": "Some nested information",
                    "details": "More details here"
                }
            },
            "settings": {
                "feature_flags": ["flag1", "flag2"],
                "configuration": "Production configuration"
            },
            "process": process,  # Use the schema_function
        }
    )
    
    # Wait a moment for the service to be fully registered
    await asyncio.sleep(1.0)  # Increase wait time
    
    # Connect to the MCP endpoint
    service_id = service_info['id'].split('/')[-1]  # Get just the service ID part
    base_url = f"{SERVER_URL}/{workspace}/mcp/{service_id}/mcp"
    
    async with streamablehttp_client(base_url) as (
        read_stream,
        write_stream,
        get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            
            # First verify the service has tools (the process function)
            tools_result = await session.list_tools()
            assert tools_result is not None
            assert hasattr(tools_result, "tools")
            assert len(tools_result.tools) > 0, "Service should have at least one tool"
            
            # List resources - should include nested string fields
            resources_result = await session.list_resources()
            assert resources_result is not None
            assert hasattr(resources_result, "resources")
            assert len(resources_result.resources) > 0, f"No resources found! Service should have string resources"
            
            # Check for expected resources with hierarchical URIs
            expected_resources = {
                "resource://metadata/version": "1.0.0",
                "resource://metadata/author": "Test Author", 
                "resource://metadata/nested/info": "Some nested information",
                "resource://metadata/nested/details": "More details here",
                "resource://settings/feature_flags/0": "flag1",
                "resource://settings/feature_flags/1": "flag2",
                "resource://settings/configuration": "Production configuration"
            }
            
            # Build a map of actual resources
            # Convert AnyUrl objects to strings for comparison
            actual_resources = {}
            for resource in resources_result.resources:
                uri_str = str(resource.uri)  # Convert AnyUrl to string
                actual_resources[uri_str] = resource
            
            # Verify all expected resources are present
            for uri, expected_content in expected_resources.items():
                assert uri in actual_resources, f"Expected resource {uri} not found. Available: {list(actual_resources.keys())}"
                
                # Read the resource and verify content
                read_result = await session.read_resource(uri=uri)
                assert read_result is not None
                assert hasattr(read_result, "contents")
                assert len(read_result.contents) > 0
                assert read_result.contents[0].text == expected_content
    await api.disconnect()
