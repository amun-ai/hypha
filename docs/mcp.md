# Model Context Protocol (MCP) Support

Hypha provides comprehensive support for the Model Context Protocol (MCP), enabling seamless integration with AI tools and services. This implementation includes both server and client capabilities, supporting multiple transport protocols.

## Overview

The MCP implementation in Hypha consists of two main components:

1. **MCP Server** (`hypha/mcp.py`): Exposes Hypha services as MCP endpoints
2. **MCP Client Proxy** (`hypha/runner/mcp_proxy.py`): Connects to external MCP servers and integrates them as Hypha services

## MCP Client Proxy

The MCP Client Runner allows Hypha to connect to external MCP servers and expose their tools, resources, and prompts as unified Hypha services.

### Supported Transports

#### Streamable HTTP
- **Type**: `streamable-http`
- **Usage**: Standard HTTP-based MCP communication
- **Configuration**:
  ```json
  {
    "type": "streamable-http",
    "url": "http://localhost:8000/mcp"
  }
  ```


#### mcp-remote Bridge
- **Type**: `stdio-bridge`
- **Usage**: Bridge to remote servers via mcp-remote package
- **Configuration**:
  ```json
  {
    "type": "stdio-bridge",
    "command": "npx",
    "args": [
      "mcp-remote",
      "https://remote.server.com/sse",
      "--transport", "sse-first",
      "--debug"
    ],
    "env": {
      "AUTH_TOKEN": "your-token-here"
    }
  }
  ```

#### mcp-remote Transport Strategies
- **`http-first`** (default): Tries HTTP first, falls back to SSE
- **`sse-first`**: Tries SSE first, falls back to HTTP
- **`http-only`**: Only uses HTTP transport
- **`sse-only`**: Only uses SSE transport

#### mcp-remote Authentication
```json
{
  "type": "stdio-bridge",
  "command": "npx",
  "args": [
    "mcp-remote",
    "https://api.example.com/sse",
    "--header", "Authorization:Bearer ${API_TOKEN}",
    "--header", "X-Custom-Header:${CUSTOM_VALUE}"
  ],
  "env": {
    "API_TOKEN": "your-api-token",
    "CUSTOM_VALUE": "custom-header-value"
  }
}
```

### MCP Server App Configuration

To use external MCP servers, create an app configuration with type `mcp-server`:

```json
{
  "type": "mcp-server",
  "name": "My MCP Server",
  "version": "1.0.0",
  "description": "Integration with external MCP server",
  "mcpServers": {
    "github-server": {
      "type": "streamable-http",
      "url": "https://api.github.com/mcp"
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

### Popular MCP Server Configurations

#### GitHub Server
```json
{
  "type": "mcp-server",
  "name": "GitHub Integration",
  "version": "1.0.0",
  "description": "GitHub repository and issue management",
  "mcpServers": {
    "github": {
      "type": "streamable-http",
      "url": "https://api.github.com/mcp",
      "headers": {
        "Authorization": "Bearer github_pat_your_token_here"
      }
    }
  }
}
```

#### Weather Services
```json
{
  "type": "mcp-server", 
  "name": "Weather Services",
  "version": "1.0.0",
  "description": "Weather data and forecasts",
  "mcpServers": {
    "openweather": {
      "type": "streamable-http",
      "url": "https://api.openweathermap.org/mcp",
      "headers": {
        "X-API-Key": "your_openweather_api_key"
      }
    },
    "national-weather": {
      "type": "sse",
      "url": "https://api.weather.gov/sse"
    }
  }
}
```

#### Filesystem Server
```json
{
  "type": "mcp-server",
  "name": "File Operations",
  "version": "1.0.0", 
  "description": "Cross-platform file system operations",
  "mcpServers": {
    "filesystem": {
      "type": "streamable-http",
      "url": "https://files.myserver.com/mcp"
    }
  }
}
```

#### Remote Server Bridge (mcp-remote)
For servers that don't directly support HTTP/SSE, use the `mcp-remote` bridge:

```json
{
  "type": "mcp-server",
  "name": "Remote Bridge Server",
  "version": "1.0.0",
  "description": "Bridge to remote MCP servers via mcp-remote",
  "mcpServers": {
    "remote-server": {
      "type": "stdio-bridge",
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://remote.mcp.server/sse",
        "--transport", "sse-first",
        "--debug"
      ],
      "env": {
        "AUTH_TOKEN": "your_auth_token_here"
      }
    }
  }
}
```

### MCP Server Providers

#### Popular Remote MCP Servers

| Provider | Service | Transport | Endpoint | Authentication |
|----------|---------|-----------|----------|----------------|
| **GitHub** | Repository Management | HTTP | `https://api.github.com/mcp` | Bearer Token |
| **OpenWeatherMap** | Weather Data | HTTP | `https://api.openweathermap.org/mcp` | API Key |
| **National Weather Service** | US Weather | SSE | `https://api.weather.gov/sse` | None |
| **Filesystem** | File Operations | HTTP | `https://files.example.com/mcp` | Optional |
| **DeepWiki** | Knowledge Base | HTTP | `https://mcp.deepwiki.com/mcp` | API Key |
| **Cursor** | Code Editor | SSE | `https://cursor.example.com/sse` | OAuth |

#### Community MCP Servers

Many MCP servers from the community can be accessed via `mcp-remote`:

```json
{
  "type": "mcp-server",
  "name": "Community Servers",
  "version": "1.0.0",
  "description": "Community-hosted MCP servers",
  "mcpServers": {
    "playwright": {
      "type": "stdio-bridge",
      "command": "npx",
      "args": ["mcp-remote", "https://playwright.example.com/sse"]
    },
    "notes": {
      "type": "stdio-bridge", 
      "command": "npx",
      "args": ["mcp-remote", "https://notes.example.com/sse"]
    },
    "amazon-fresh": {
      "type": "stdio-bridge",
      "command": "npx", 
      "args": ["mcp-remote", "https://amazon-fresh.example.com/sse"]
    }
  }
}
```

### Features

- **Unified Service Registration**: Each MCP server is registered as a single Hypha service containing all tools, resources, and prompts
- **Transport Flexibility**: Supports both streamable HTTP and SSE transports
- **Error Handling**: Comprehensive error handling with detailed logging
- **Session Management**: Proper session lifecycle management with cleanup
- **Debugging Support**: Extensive logging at different levels (INFO, DEBUG, ERROR)

### Usage Examples

#### Basic Remote HTTP Server
```python
from hypha_rpc import connect_to_server

# Connect to Hypha
api = await connect_to_server({
    "server_url": "ws://localhost:9527",
    "workspace": "my-workspace",
    "token": "your-token"
})

# Get the server apps controller
controller = await api.get_service("public/server-apps")

# Install GitHub MCP server app
app_info = await controller.install(
    config={
        "type": "mcp-server",
        "name": "GitHub Integration",
        "version": "1.0.0",
        "description": "GitHub repository management",
        "mcpServers": {
            "github": {
                "type": "streamable-http",
                "url": "https://api.github.com/mcp",
                "headers": {
                    "Authorization": "Bearer github_pat_your_token_here"
                }
            }
        }
    }
)

# Start the MCP server
session_info = await controller.start(app_info["id"])

# Use the unified MCP service
service_id = f"{session_info['id']}:github"
github_service = await api.get_service(service_id)

# Access tools, resources, and prompts
tools = github_service.tools
resources = github_service.resources
prompts = github_service.prompts

# Call a GitHub tool
issues = await tools[0](repo="owner/repo", state="open")

# Read a repository resource
repo_info = await resources[0]["read"]()

# Get a code review prompt
review_prompt = await prompts[0]["read"](code="def hello(): pass")
```

#### Using mcp-remote Bridge
```python
# Install server with mcp-remote bridge
app_info = await controller.install(
    config={
        "type": "mcp-server",
        "name": "Remote Weather Server",
        "version": "1.0.0",
        "description": "Weather data via mcp-remote",
        "mcpServers": {
            "weather": {
                "type": "stdio-bridge",
                "command": "npx",
                "args": [
                    "mcp-remote",
                    "https://weather.example.com/sse",
                    "--transport", "sse-first",
                    "--header", "Authorization:Bearer ${API_KEY}"
                ],
                "env": {
                    "API_KEY": "your_weather_api_key"
                }
            }
        }
    }
)

# Start and use the bridged server
session_info = await controller.start(app_info["id"])
weather_service = await api.get_service(f"{session_info['id']}:weather")

# Get weather forecast
forecast = await weather_service.tools[0](location="New York", days=7)
```

#### Multiple Transport Types
```python
# Configure multiple servers with different transports
app_info = await controller.install(
    config={
        "type": "mcp-server",
        "name": "Multi-Transport Server",
        "version": "1.0.0",
        "description": "Multiple MCP servers with different transports",
        "mcpServers": {
            "github": {
                "type": "streamable-http",
                "url": "https://api.github.com/mcp",
                "headers": {
                    "Authorization": "Bearer github_pat_token"
                }
            },
            "weather": {
                "type": "sse",
                "url": "https://weather.example.com/sse"
            },
            "filesystem": {
                "type": "streamable-http",
                "url": "https://files.example.com/mcp"
            }
        }
    }
)

# Start and access all services
session_info = await controller.start(app_info["id"])

# Each server becomes a separate service
github_service = await api.get_service(f"{session_info['id']}:github")
weather_service = await api.get_service(f"{session_info['id']}:weather")
files_service = await api.get_service(f"{session_info['id']}:filesystem")

# Use tools from different services
repos = await github_service.tools[0](query="python")
forecast = await weather_service.tools[0](location="San Francisco")
files = await files_service.tools[0](path="/home/user/projects")
```

## MCP Server (Hypha as MCP Provider)

Hypha can also act as an MCP server, exposing its services to external MCP clients.

### HTTP Endpoints

When MCP is enabled (`--enable-mcp`), Hypha exposes MCP endpoints at:

```
/{workspace}/mcp/{service_id}/mcp        # Streamable HTTP transport
/{workspace}/mcp/{service_id}/sse        # Server-Sent Events transport (future)
/{workspace}/mcp/{service_id}            # Info endpoint (returns helpful 404)
```

#### Endpoint Structure

- **Streamable HTTP**: `/{workspace}/mcp/{service_id}/mcp`
  - Used for standard MCP JSON-RPC communication
  - Supports all MCP methods (tools/list, tools/call, prompts/list, etc.)
  - Content-Type: `application/json`

- **Server-Sent Events**: `/{workspace}/mcp/{service_id}/sse` *(future)*
  - Will be used for real-time streaming communication
  - Content-Type: `text/event-stream`
  - Currently returns 501 Not Implemented

- **Info Endpoint**: `/{workspace}/mcp/{service_id}`
  - Returns helpful 404 with available endpoint information
  - Helps users discover the correct endpoint URLs
  - Content-Type: `application/json`

### Supported Methods

- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/read` - Read a resource
- `prompts/list` - List available prompts
- `prompts/get` - Get a prompt

### Example Usage

```bash
# List tools
curl -X POST http://localhost:9527/my-workspace/mcp/my-service/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Call a tool
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

# Get helpful endpoint information
curl -X GET http://localhost:9527/my-workspace/mcp/my-service \
  -H "Content-Type: application/json"
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
- Check that the server is running
- Ensure the transport type matches the server
- Review the error logs for specific details

#### 4. Session Initialization Issues
**Error**: `unhandled errors in a TaskGroup`
**Solution**: This was a known issue that has been fixed by adding proper `await session.initialize()` calls

### Debugging Tips

1. **Enable Debug Logging**: Set `HYPHA_LOGLEVEL=DEBUG` to see detailed connection attempts
2. **Check Session Logs**: Use `controller.get_logs(session_id)` to retrieve session-specific logs
3. **Verify Transport Type**: Ensure the transport type in configuration matches the server capabilities
4. **Test Connectivity**: Use curl or similar tools to test the MCP server endpoint directly

#### Remote Server Debugging

For remote HTTP/SSE servers:
```bash
# Test streamable HTTP endpoint
curl -X POST https://api.example.com/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Test SSE endpoint
curl -N https://api.example.com/sse \
  -H "Accept: text/event-stream" \
  -H "Authorization: Bearer your-token"
```

For Hypha MCP endpoints:
```bash
# Test local MCP service streamable HTTP endpoint
curl -X POST http://localhost:9527/my-workspace/mcp/my-service/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Get endpoint information
curl -X GET http://localhost:9527/my-workspace/mcp/my-service

# Test SSE endpoint (when implemented)
curl -N http://localhost:9527/my-workspace/mcp/my-service/sse \
  -H "Accept: text/event-stream"
```