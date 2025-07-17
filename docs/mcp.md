# Model Context Protocol (MCP) Support

Hypha provides comprehensive support for the Model Context Protocol (MCP), enabling seamless integration with AI tools and services. This implementation includes both server and client capabilities, supporting multiple transport protocols.

## Overview

The MCP implementation in Hypha consists of two main components:

1. **MCP Server** (`hypha/mcp.py`): Exposes Hypha services as MCP endpoints
2. **MCP Client Proxy** (`hypha/workers/mcp_proxy.py`): Connects to external MCP servers and integrates them as Hypha services

To enable MCP support, start Hypha server with `--enable-mcp`:

```bash
python -m hypha.server --enable-mcp
```

## Hypha Service as MCP Server

Hypha can act as an MCP server, exposing its services to external MCP clients like Cursor, Claude Desktop, and other AI tools.

### Registering Hypha Services for MCP

There are three main approaches to create Hypha services that can be exposed as MCP endpoints:

#### 1. Schema Function Approach (Recommended)

The most straightforward way using `@schema_function` decorators:

```python
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

async def main():
    api = await connect_to_server({
        "server_url": "ws://localhost:9527",
        "workspace": "my-workspace"
    })

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
        return "This is comprehensive resource content with detailed information."

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

    # Register MCP service
    service = await api.register_service({
        "id": "my-calculator-service",
        "name": "Calculator MCP Service",
        "description": "A calculator service with tools, resources, and prompts",
        "type": "mcp",
        "config": {
            "visibility": "public",
            "run_in_executor": True,
        },
        "tools": [calculate_tool],
        "resources": [
            {
                "uri": "resource://calculator",
                "name": "Calculator Resource", 
                "description": "Calculator help documentation",
                "tags": ["calculator", "help"],
                "mime_type": "text/plain",
                "read": resource_read,
            }
        ],
        "prompts": [
            {
                "name": "calculation_help",
                "description": "Help with calculations",
                "tags": ["calculator", "prompt"],
                "read": prompt_read,
            }
        ],
    })

    print(f"MCP service registered: {service['id']}")
    # MCP endpoint will be available at:
    # http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

#### 2. Direct MCP Function Approach

Using direct MCP functions (requires MCP SDK):

```python
from hypha_rpc import connect_to_server
import mcp.types as types

async def main():
    api = await connect_to_server({
        "server_url": "ws://localhost:9527",
        "workspace": "my-workspace"
    })

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
                            "description": "Timestamp format"
                        }
                    },
                    "required": ["format"]
                }
            )
        ]

    async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
        if name == "timestamp":
            format_type = arguments.get("format", "unix")
            import time
            if format_type == "unix":
                result = str(int(time.time()))
            elif format_type == "iso":
                result = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return [types.TextContent(type="text", text=result)]
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Register MCP service
    service = await api.register_service({
        "id": "timestamp-service",
        "type": "mcp",
        "config": {"visibility": "public"},
        "list_tools": list_tools,
        "call_tool": call_tool,
    })

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

#### 3. Inline Configuration Approach

Define tools inline with handlers:

```python
from hypha_rpc import connect_to_server

async def main():
    api = await connect_to_server({
        "server_url": "ws://localhost:9527",
        "workspace": "my-workspace"
    })

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

    def resource_read() -> str:
        """Read a test resource."""
        return "This is test resource content."

    # Register MCP service with inline configuration
    service = await api.register_service({
        "id": "inline-mcp-service",
        "name": "Inline MCP Service",
        "description": "A service with inline MCP configuration",
        "type": "mcp",
        "config": {
            "visibility": "public",
            "run_in_executor": True,
        },
        "tools": [
            {
                "name": "simple_tool",
                "description": "Simple arithmetic tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The operation to perform"
                        },
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["operation", "a", "b"]
                },
                "handler": simple_tool
            }
        ],
        "resources": [
            {
                "uri": "resource://test",
                "name": "Test Resource",
                "description": "A test resource",
                "tags": ["test", "resource"],
                "mime_type": "text/plain",
                "read": resource_read,
            }
        ],
    })

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### MCP HTTP Endpoints

When MCP is enabled (starting Hypha server with `--enable-mcp`), Hypha exposes MCP endpoints at:

```
/{workspace}/mcp/{service_id}/mcp        # Streamable HTTP transport
/{workspace}/mcp/{service_id}/sse        # Server-Sent Events transport (not yet implemented)
/{workspace}/mcp/{service_id}            # Info endpoint (returns helpful 404)
```

#### Supported Methods

- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/read` - Read a resource
- `prompts/list` - List available prompts
- `prompts/get` - Get a prompt

### Testing MCP Endpoints

You can test your MCP endpoints using curl:

```bash
# List tools
curl -X POST http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Call a tool
curl -X POST http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "calculate_tool",
      "arguments": {"operation": "add", "a": 5, "b": 3}
    },
    "id": 2
  }'
```

## Using Hypha MCP Endpoints with AI Tools

### Cursor IDE

1. Open Cursor Settings (`⇧+⌘+J` on Mac, `Ctrl+Shift+J` on Windows/Linux)
2. Navigate to "MCP Tools" and click "New MCP Server"
3. Add this configuration:

```json
{
  "mcpServers": {
    "Hypha Calculator": {
      "url": "http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp"
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "hypha-calculator": {
      "url": "http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp"
    }
  }
}
```

### OpenAI Python Library

```python
import openai
from openai import OpenAI

client = OpenAI()

# Use with OpenAI's new responses API
response = client.responses.create(
    model="gpt-4o",
    tools=[
        {
            "type": "mcp",
            "server_label": "hypha",
            "server_url": "http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp",
            "require_approval": "never"
        }
    ],
    input="Calculate 15 * 24 using the calculator tool",
    tool_choice="required"
)

print(response.content)
```

### LiteLLM Integration

Add to your LiteLLM configuration:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  hypha_calculator:
    url: "http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp"
    description: "Hypha calculator service"
```

Then use with the LiteLLM proxy:

```bash
curl -X POST http://localhost:4000/v1/responses \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer $LITELLM_API_KEY" \
  --data '{
    "model": "gpt-4o",
    "tools": [
      {
        "type": "mcp",
        "server_label": "litellm",
        "server_url": "litellm_proxy",
        "require_approval": "never",
        "headers": {
          "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
          "x-mcp-servers": "hypha_calculator"
        }
      }
    ],
    "input": "Calculate 100 / 4",
    "tool_choice": "required"
  }'
```

### Direct MCP Client Usage

Using the official MCP SDK:

```python
import asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

async def use_hypha_mcp():
    base_url = "http://localhost:9527/my-workspace/mcp/my-calculator-service/mcp"
    
    async with streamablehttp_client(base_url) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            # List tools
            tools_result = await session.list_tools()
            print(f"Available tools: {[tool.name for tool in tools_result.tools]}")
            
            # Call a tool
            result = await session.call_tool(
                name="calculate_tool",
                arguments={"operation": "multiply", "a": 7, "b": 6}
            )
            print(f"Result: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(use_hypha_mcp())
```

## MCP Client Proxy

The MCP Client Proxy allows Hypha to connect to external MCP servers and expose their tools, resources, and prompts as unified Hypha services.

### Supported Transports

- **Streamable HTTP** (`streamable-http`): Standard HTTP-based MCP communication
- **Server-Sent Events** (`sse`): Real-time streaming MCP communication

### Installing External MCP Servers

To use external MCP servers, create an app configuration with type `mcp-server`:

```json
{
  "type": "mcp-server",
  "name": "External MCP Services",
  "version": "1.0.0",
  "description": "Integration with external MCP servers",
  "mcpServers": {
    "deepwiki": {
      "type": "streamable-http",
      "url": "https://mcp.deepwiki.com/mcp"
    },
    "weather-server": {
      "type": "sse",
      "url": "https://weather.example.com/sse"
    },
    "filesystem-server": {
      "type": "streamable-http",
      "url": "https://files.example.com/mcp",
      "headers": {
        "Authorization": "Bearer your-token-here"
      }
    }
  }
}
```

### Installing and Using External MCP Servers

```python
from hypha_rpc import connect_to_server

async def install_mcp_server():
    api = await connect_to_server({
        "server_url": "ws://localhost:9527",
        "workspace": "my-workspace"
    })

    # Get the server apps controller
    controller = await api.get_service("public/server-apps")

    # Install DeepWiki MCP server
    app_info = await controller.install(
        config={
            "type": "mcp-server",
            "name": "DeepWiki Integration",
            "version": "1.0.0",
            "description": "DeepWiki knowledge base access",
            "mcpServers": {
                "deepwiki": {
                    "type": "streamable-http",
                    "url": "https://mcp.deepwiki.com/mcp"
                }
            }
        }
    )

    # Start the MCP server
    session_info = await controller.start(app_info["id"], wait_for_service="deepwiki", timeout=30)

    # Use the unified MCP service
    deepwiki_service = await api.get_service(f"{session_info['id']}:deepwiki")

    # Access tools, resources, and prompts
    tools = deepwiki_service.tools
    resources = deepwiki_service.resources
    prompts = deepwiki_service.prompts

    print(f"Available tools: {len(tools)}")
    print(f"Available resources: {len(resources)}")
    print(f"Available prompts: {len(prompts)}")

    # Call a tool (example)
    if tools:
        result = await tools[0](query="Python programming")
        print(f"Tool result: {result}")

    # Read a resource (example)
    if resources:
        content = await resources[0]["read"]()
        print(f"Resource content: {content[:100]}...")

    # Get a prompt (example)
    if prompts:
        prompt_content = await prompts[0]["read"](topic="AI development")
        print(f"Prompt: {prompt_content}")

    await api.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(install_mcp_server())
```

### Popular MCP Server Examples

#### DeepWiki (Knowledge Base)
```json
{
  "type": "mcp-server",
  "name": "DeepWiki Knowledge Base",
  "version": "1.0.0",
  "description": "Access to DeepWiki knowledge base",
  "mcpServers": {
    "deepwiki": {
      "type": "streamable-http",
      "url": "https://mcp.deepwiki.com/mcp"
    }
  }
}
```

#### Filesystem Server
```json
{
  "type": "mcp-server",
  "name": "Filesystem Operations",
  "version": "1.0.0",
  "description": "Cross-platform file system operations",
  "mcpServers": {
    "filesystem": {
      "type": "streamable-http",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    }
  }
}
```

#### Puppeteer Automation
```json
{
  "type": "mcp-server",
  "name": "Browser Automation",
  "version": "1.0.0",
  "description": "Browser automation via Puppeteer",
  "mcpServers": {
    "puppeteer": {
      "type": "streamable-http",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
    }
  }
}
```

## Error Handling and Debugging

### Logging Levels

Set the logging level using the `HYPHA_LOGLEVEL` environment variable:

```bash
export HYPHA_LOGLEVEL=DEBUG  # For detailed debugging
export HYPHA_LOGLEVEL=INFO   # For general information
export HYPHA_LOGLEVEL=ERROR  # For errors only
```

### Common Issues and Solutions

#### 1. MCP SDK Not Available
**Error**: `MCP SDK not available`
**Solution**: Install the MCP SDK:
```bash
pip install mcp
```

#### 2. SSE Client Not Available
**Error**: `SSE client not available`
**Solution**: Install MCP with SSE support:
```bash
pip install mcp[sse]
```

#### 3. Connection Failures
**Error**: `Failed to connect to MCP server`
**Solutions**:
- Verify the server URL is correct
- Check that the Hypha server is running with `--enable-mcp`
- Ensure the transport type matches the server capabilities
- Review the error logs for specific details

#### 4. Service Registration Issues
**Error**: `Schema function validation failed`
**Solution**: Ensure all tool functions are decorated with `@schema_function`:
```python
from hypha_rpc.utils.schema import schema_function

@schema_function
def my_tool(param: str) -> str:
    """Tool description."""
    return f"Processed: {param}"
```

### Debugging Tips

1. **Enable Debug Logging**: Set `HYPHA_LOGLEVEL=DEBUG` to see detailed connection attempts
2. **Test Endpoints**: Use curl to test your MCP endpoints directly
3. **Verify Service Registration**: Check that your service appears in the Hypha service list
4. **Check Function Schemas**: Ensure your `@schema_function` decorators are properly applied

#### Testing MCP Endpoints

```bash
# Test if MCP is enabled and endpoint exists
curl -X GET http://localhost:9527/my-workspace/mcp/my-service

# Test tools listing
curl -X POST http://localhost:9527/my-workspace/mcp/my-service/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Test tool calling
curl -X POST http://localhost:9527/my-workspace/mcp/my-service/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "my_tool",
      "arguments": {"param": "value"}
    },
    "id": 2
  }'
```

## Best Practices

1. **Use Schema Functions**: Always use `@schema_function` decorators for better tool discovery and validation
2. **Proper Error Handling**: Handle errors gracefully in your tool implementations
3. **Resource Management**: Use context managers when connecting to external services
4. **Security**: Validate inputs and sanitize outputs in your tools
5. **Documentation**: Provide clear descriptions for all tools, resources, and prompts
6. **Testing**: Test your MCP endpoints with curl before integrating with AI tools

## Examples Repository

For more examples and templates, see the Hypha examples directory which includes:
- Complete MCP service implementations
- Integration examples with popular AI tools
- Best practices and patterns
- Performance optimization techniques