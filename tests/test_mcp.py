"""Test MCP (Model Context Protocol) services."""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional

import pytest
import httpx
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from . import WS_SERVER_URL, SERVER_URL

# MCP imports
try:
    import mcp.types as types
    from mcp.server.lowlevel import Server

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

    # Mock types for tests when MCP is not available
    class MockTool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    class MockTextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text

    class MockPrompt:
        def __init__(self, name, description, arguments):
            self.name = name
            self.description = description
            self.arguments = arguments

    class MockPromptArgument:
        def __init__(self, name, description, required):
            self.name = name
            self.description = description
            self.required = required

    class MockGetPromptResult:
        def __init__(self, description, messages):
            self.description = description
            self.messages = messages

    class MockPromptMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class MockResourceTemplate:
        def __init__(self, uriTemplate, name, description, mimeType):
            self.uriTemplate = uriTemplate
            self.name = name
            self.description = description
            self.mimeType = mimeType

    # Create a mock types module
    class MockTypes:
        Tool = MockTool
        TextContent = MockTextContent
        ContentBlock = MockTextContent  # ContentBlock is an alias for TextContent
        Prompt = MockPrompt
        PromptArgument = MockPromptArgument
        GetPromptResult = MockGetPromptResult
        PromptMessage = MockPromptMessage
        ResourceTemplate = MockResourceTemplate

    types = MockTypes()

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
            "progress_notification": progress_notification,
        }
    )

    assert service["type"] == "mcp"
    assert "test-mcp-service" in service["id"]

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

    # Register schema_function decorated MCP service
    service = await api.register_service(
        {
            "id": "schema-mcp-service",
            "config": {
                "visibility": "public",
            },
            "list_tools": list_tools,
            "call_tool": call_tool,
            "list_prompts": list_prompts,
            "get_prompt": get_prompt,
            "list_resource_templates": list_resource_templates,
        }
    )

    # Service should be registered successfully
    assert "schema-mcp-service" in service["id"]
    assert service["id"].startswith(workspace)

    await api.disconnect()


async def test_mcp_inline_config_service(fastapi_server, test_user_token):
    """Test MCP service with inline tools/resources/prompts configuration."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Define simple tool handler
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

    # Define resource read handler
    def resource_read() -> str:
        """Read a test resource."""
        return "This is a test resource content."

    # Define prompt read handler
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

    # Register MCP service with inline configuration
    service = await api.register_service(
        {
            "id": "inline-mcp-service",
            "name": "Inline MCP Service",
            "description": "A service with inline MCP configuration",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": {
                "simple_tool": {
                    "description": "Simple arithmetic tool",
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
                    "handler": simple_tool,
                }
            },
            "resources": {
                "test_resource": {
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

    await api.disconnect()


async def test_mcp_merged_approach(fastapi_server, test_user_token):
    """Test MCP service with merged approach: inline tools/resources/prompts using schema functions."""
    # Connect to the Hypha server
    server = await connect_to_server(
        {
            "name": "mcp-merged-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

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
    mcp_service_info = await server.register_service(
        {
            "id": "mcp-merged-features",
            "name": "MCP Merged Features Service",
            "description": "A service with merged MCP features",
            "type": "mcp",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "tools": {"simple_tool": simple_tool},
            "resources": {
                "test_resource": {
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

    print("✓ MCP service with merged approach registered successfully")

    await server.disconnect()


async def test_mcp_validation_errors(fastapi_server, test_user_token):
    """Test that validation errors are properly raised for invalid configurations."""
    server = await connect_to_server(
        {
            "name": "mcp-validation-test",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        }
    )

    # Test direct validation by creating HyphaMCPServer directly with invalid configs
    from hypha.mcp import HyphaMCPServer

    # Test 1: Tool that is not a schema function should raise error
    def non_schema_tool(text: str) -> str:
        return f"Processed: {text}"

    class MockServiceInfo:
        def __init__(self, **kwargs):
            self.id = "test-service"
            self.type = "mcp"
            for k, v in kwargs.items():
                setattr(self, k, v)

    service_info = MockServiceInfo(tools={non_schema_tool.__name__: non_schema_tool})

    try:
        HyphaMCPServer({}, service_info)
        assert False, "Should have raised ValueError for non-schema tool"
    except ValueError as e:
        assert "must be a @schema_function decorated function" in str(e)
        print("✓ Tool validation error properly raised")

    # Test 2: Resource without read function should raise error
    service_info = MockServiceInfo(
        resources={
            "resource://test": {
                "uri": "resource://test",
                "name": "Test Resource",
                "description": "A test resource",
            }
        }
    )

    try:
        HyphaMCPServer({}, service_info)
        assert False, "Should have raised ValueError for resource without read"
    except ValueError as e:
        assert "must have a 'read' key" in str(e)
        print("✓ Resource validation error properly raised")

    # Test 3: Resource with non-schema read function should raise error
    def non_schema_read():
        return "content"

    service_info = MockServiceInfo(
        resources={
            "resource://test": {
                "uri": "resource://test",
                "name": "Test Resource",
                "description": "A test resource",
                "read": non_schema_read,
            }
        }
    )

    try:
        HyphaMCPServer({}, service_info)
        assert False, "Should have raised ValueError for non-schema read function"
    except ValueError as e:
        assert "must be a @schema_function decorated function" in str(e)
        print("✓ Resource read validation error properly raised")

    # Test 4: Prompt without read function should raise error
    service_info = MockServiceInfo(
        prompts={
            "test_prompt": {
                "name": "test_prompt",
                "description": "A test prompt",
            }
        }
    )

    try:
        HyphaMCPServer({}, service_info)
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
            headers={"Content-Type": "application/json"},
        )

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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
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
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200  # JSON-RPC errors are returned with 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid Request

        # Test 2: Method not found
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/error-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-2",
                "method": "unknown/method",
                "params": {},
            },
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found

        # Test 3: Tool runtime error
        response = await client.post(
            f"{SERVER_URL}/{workspace}/mcp/error-service/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "test-3",
                "method": "tools/call",
                "params": {"name": "error_tool", "arguments": {"should_error": True}},
            },
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603  # Internal error

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
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 404


async def test_mcp_helpful_404_message(fastapi_server, test_user_token):
    """Test that accessing MCP service without /mcp suffix returns helpful 404."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Register a simple MCP service
    await api.register_service(
        {
            "id": "test-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
        }
    )

    async with httpx.AsyncClient() as client:
        # Try to access service without /mcp suffix
        response = await client.get(
            f"{SERVER_URL}/{workspace}/mcp/test-service",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"] == "MCP endpoint not found"
        assert "available_endpoints" in data
        assert (
            data["available_endpoints"]["streamable_http"]
            == f"/{workspace}/mcp/test-service/mcp"
        )
        assert "sse" in data["available_endpoints"]["sse"]
        assert "help" in data

    await api.disconnect()


async def test_mcp_sse_not_implemented(fastapi_server, test_user_token):
    """Test that SSE endpoint returns not implemented message."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )

    workspace = api.config.workspace

    # Register a simple MCP service
    await api.register_service(
        {
            "id": "test-service",
            "type": "mcp",
            "config": {
                "visibility": "public",
            },
        }
    )

    async with httpx.AsyncClient() as client:
        # Try to access SSE endpoint
        response = await client.get(
            f"{SERVER_URL}/{workspace}/mcp/test-service/sse",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 501
        data = response.json()
        assert "error" in data
        assert "not yet implemented" in data["error"]

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

    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession

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

    await api.disconnect()
