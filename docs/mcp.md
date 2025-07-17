# MCP (Model Context Protocol) Services

## Introduction

Hypha supports the Model Context Protocol (MCP), an open standard designed to enable AI applications to connect to external data sources and tools. MCP provides a standardized way for AI models to interact with resources, tools, and prompts through a structured interface.

With Hypha's MCP service support, you can:
- Register services that comply with the MCP protocol specification
- Automatically expose services through standardized HTTP endpoints
- Enable tool and resource discovery through MCP protocol
- Support various interaction patterns including streaming responses
- Integrate with MCP-compatible clients like Cursor, Claude Desktop, and Windsurf

## Prerequisites

- **Hypha Server**: Ensure you have Hypha installed and configured
- **MCP SDK**: Install the MCP SDK dependency: `pip install mcp`
- **Python 3.9+**: The MCP implementation requires Python 3.9 or higher

## Installation and Setup

### Installing MCP SDK

```bash
pip install mcp
```

### Enabling MCP Support in Hypha

Start your Hypha server with MCP support enabled:

```bash
# Enable MCP support
python -m hypha.server --enable-mcp

# Enable both A2A and MCP
python -m hypha.server --enable-a2a --enable-mcp
```

Or set the environment variable:

```bash
export HYPHA_ENABLE_MCP=true
python -m hypha.server --from-env
```

## MCP Service Architecture

When you register a service with `type="mcp"` or with MCP-compatible functions, Hypha automatically:
1. Routes requests to the MCP URL namespace (`/{workspace}/mcp/{service_id}/`)
2. Validates the MCP service implementation
3. Creates a `HyphaMCPServer` adapter from your service
4. Sets up proper MCP protocol handling with JSON-RPC 2.0
5. Provides HTTP endpoints for MCP clients

## Registering MCP Services

Hypha supports multiple approaches for registering MCP services, allowing you to choose the method that best fits your needs:

### Approach 1: Standard MCP Functions (`type="mcp"`)

Services can be registered with `type="mcp"` and implement MCP protocol functions:

```python
import asyncio
from typing import Dict, Any, List, Optional
from hypha_rpc import connect_to_server

# Import MCP types
try:
    import mcp.types as types
except ImportError:
    print("Please install mcp: pip install mcp")
    exit(1)

async def calculator_service():
    """Example MCP service with explicit type declaration."""
    
    # Define MCP service functions
    async def list_tools() -> List[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name="calculate",
                description="Perform basic arithmetic calculations",
                inputSchema={
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
                }
            ),
            types.Tool(
                name="factorial",
                description="Calculate factorial of a number",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 0, "description": "Number to calculate factorial"}
                    },
                    "required": ["n"]
                }
            )
        ]

    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
        """Execute a tool with given arguments."""
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
                    return [types.TextContent(type="text", text="Error: Division by zero")]
                result = a / b
            else:
                return [types.TextContent(type="text", text=f"Unknown operation: {operation}")]
            
            return [types.TextContent(type="text", text=f"Result: {result}")]
        
        elif name == "factorial":
            n = arguments.get("n", 0)
            if n < 0:
                return [types.TextContent(type="text", text="Error: Factorial is not defined for negative numbers")]
            
            result = 1
            for i in range(1, n + 1):
                result *= i
            
            return [types.TextContent(type="text", text=f"Factorial of {n} is {result}")]
        
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    async def list_prompts() -> List[types.Prompt]:
        """List available prompts."""
        return [
            types.Prompt(
                name="math_tutor",
                description="A helpful math tutor prompt",
                arguments=[
                    types.PromptArgument(
                        name="topic",
                        description="Math topic to get help with",
                        required=True
                    ),
                    types.PromptArgument(
                        name="level",
                        description="Difficulty level (beginner/intermediate/advanced)",
                        required=False
                    )
                ]
            )
        ]

    async def get_prompt(name: str, arguments: Dict[str, Any]) -> types.GetPromptResult:
        """Get a specific prompt."""
        if name == "math_tutor":
            topic = arguments.get("topic", "general math")
            level = arguments.get("level", "beginner")
            
            return types.GetPromptResult(
                description=f"Math tutor for {topic} at {level} level",
                messages=[
                    types.PromptMessage(
                        role="system",
                        content=types.TextContent(
                            type="text",
                            text=f"You are a helpful math tutor specializing in {topic}. "
                                 f"Explain concepts at a {level} level with clear examples."
                        )
                    ),
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Can you help me understand {topic}?"
                        )
                    )
                ]
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    async def list_resource_templates() -> List[types.ResourceTemplate]:
        """List available resource templates."""
        return [
            types.ResourceTemplate(
                uriTemplate="math://formula/{category}",
                name="Math Formula",
                description="Mathematical formulas by category",
                mimeType="text/plain"
            ),
            types.ResourceTemplate(
                uriTemplate="math://constants/{name}",
                name="Math Constants",
                description="Mathematical constants",
                mimeType="application/json"
            )
        ]

    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Register MCP service
    service_info = await server.register_service({
        "id": "calculator-mcp",
        "name": "Calculator MCP Service",
        "type": "mcp",
        "config": {
            "visibility": "public"
        },
        "list_tools": list_tools,
        "call_tool": call_tool,
        "list_prompts": list_prompts,
        "get_prompt": get_prompt,
        "list_resource_templates": list_resource_templates,
    })
    
    print(f"MCP service registered: {service_info['id']}")
    print(f"Access at: https://hypha.aicell.io/{server.config.workspace}/mcp/{service_info['id']}/")
    
    # Keep the service running
    await server.serve()

# Run the service
asyncio.run(calculator_service())
```

### Approach 2: Inline Configuration (`type="mcp"`)

Services can be registered with `type="mcp"` and provide inline tools, resources, and prompts configuration:

```python
import asyncio
from typing import Dict, Any
from hypha_rpc import connect_to_server

async def mcp_service_with_inline_config():
    """Example MCP service with inline configuration."""
    
    # Define tool handlers
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
                    "content": {
                        "type": "text",
                        "text": f"Help me with {topic}"
                    }
                }
            ]
        }

    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Register MCP service with inline configuration
    service_info = await server.register_service({
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
        "prompts": [
            {
                "name": "test_prompt",
                "description": "A test prompt template",
                "tags": ["test", "prompt"],
                "read": prompt_read,
            }
        ],
    })
    
    print(f"MCP service registered: {service_info['id']}")
    print(f"Access at: https://hypha.aicell.io/{server.config.workspace}/mcp/{service_info['id']}/")
    
    # Keep the service running
    await server.serve()

# Run the service
asyncio.run(mcp_service_with_inline_config())
```

### Approach 3: Schema Function Services (Automatic Tool Extraction)

Hypha services with `@schema_function` decorated functions are automatically detected and converted to MCP tools. The system automatically extracts the input schema from the function's `__schema__` attribute and allows direct function calls without requiring `list_tools` or `call_tool` methods.

Here's a simple example:

```python
import asyncio
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

# Define a simple tool with @schema_function decorator
@schema_function
def add(
    a: int = Field(..., description="First integer to add"),
    b: int = Field(..., description="Second integer to add"),
) -> int:
    """Adds two integers together."""
    return a + b

async def main():
    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Register MCP service - Hypha automatically detects schema functions
    service_info = await server.register_service({
        "id": "simple-calculator",
        "type": "mcp",
        "config": {"visibility": "public"},
        "add": add,  # Just add the function - no list_tools/call_tool needed!
    })
    
    print(f"MCP service registered: {service_info['id']}")
    print(f"Tool 'add' available at: https://hypha.aicell.io/{server.config.workspace}/mcp/{service_info['id']}/")
    
    await server.serve()

asyncio.run(main())
```

For more complex scenarios:

```python
import asyncio
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function


async def schema_function_mcp_service():
    """Example MCP service with automatic schema function extraction."""
    
    # Define tools as @schema_function decorated functions
    @schema_function
    def analyze_text(text: str, analysis_type: str = "summary") -> str:
        """Analyze text content and provide insights.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform (sentiment, keywords, summary)
        """
        if analysis_type == "sentiment":
            # Simple sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
            negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return f"Sentiment analysis result: {sentiment}\nPositive indicators: {positive_count}\nNegative indicators: {negative_count}"
        
        elif analysis_type == "keywords":
            # Simple keyword extraction
            words = text.split()
            word_freq = {}
            for word in words:
                word_clean = word.lower().strip('.,!?;:"')
                if len(word_clean) > 3:  # Only words longer than 3 characters
                    word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
            
            # Get top 5 most frequent words
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            keywords_text = ", ".join([f"{word} ({count})" for word, count in top_keywords])
            
            return f"Top keywords: {keywords_text}"
        
        elif analysis_type == "summary":
            # Simple summary (first and last sentences)
            sentences = text.split('. ')
            if len(sentences) > 2:
                summary = f"{sentences[0]}. ... {sentences[-1]}"
            else:
                summary = text
            
            return f"Summary: {summary}"
        
        else:
            return f"Unknown analysis type: {analysis_type}"

    @schema_function  
    def count_words(text: str) -> str:
        """Count words, characters, and lines in text.
        
        Args:
            text: Text to count
        """
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        return f"Text statistics:\nWords: {word_count}\nCharacters: {char_count}\nLines: {line_count}"

    @schema_function
    def calculate(operation: str, a: float, b: float) -> str:
        """Perform basic arithmetic calculations.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
        """
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

    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Register MCP service with automatic schema function detection
    # The server will automatically detect @schema_function decorated functions
    # and convert them to MCP tools without requiring list_tools/call_tool methods
    service_info = await server.register_service({
        "id": "schema-auto-mcp",
        "name": "Schema Function MCP Service",
        "type": "mcp",
        "config": {
            "visibility": "public"
        },
        # Register the schema functions as service methods
        # Hypha will automatically extract inputSchema from the __schema__ attribute
        "analyze_text": analyze_text,
        "count_words": count_words,
        "calculate": calculate,
    })
    
    print(f"Schema function MCP service registered: {service_info['id']}")
    print(f"Access at: https://hypha.aicell.io/{server.config.workspace}/mcp/{service_info['id']}/")
    print("Tools automatically extracted from @schema_function decorated functions:")
    print("- analyze_text: Analyze text content and provide insights")
    print("- count_words: Count words, characters, and lines in text") 
    print("- calculate: Perform basic arithmetic calculations")
    
    # Keep the service running
    await server.serve()

# Run the service
asyncio.run(schema_function_mcp_service())
```

## Choosing the Right Approach

### When to Use Each Approach

**Approach 1 (Standard MCP Functions)**: Best for complex services that need:
- Dynamic tool/resource/prompt discovery
- Runtime generation of MCP objects
- Complex business logic in MCP functions
- Integration with external APIs or databases

**Approach 2 (Inline Configuration)**: Best for simple services that have:
- Static, predefined tools/resources/prompts
- Simple handler functions
- Minimal configuration requirements
- Easy deployment and testing

**Approach 3 (Schema Function Services)**: Best for:
- Existing Hypha services that need MCP compatibility
- Services that already use `@schema_function` decorators
- Automatic tool extraction from function signatures
- Zero-config MCP tool registration

### Key Differences

| Feature | Approach 1 | Approach 2 | Approach 3 |
|---------|------------|------------|------------|
| **Configuration** | Dynamic via functions | Static in service config | Automatic from decorators |
| **Tools Definition** | `list_tools()` function | `tools` array in config | Automatic extraction from `@schema_function` |
| **Tool Execution** | `call_tool()` function | `handler` functions | Direct function calls |
| **Schema Extraction** | Manual | Manual | Automatic from `__schema__` attribute |
| **Complexity** | High (full MCP protocol) | Low (simple handlers) | Very Low (zero config) |
| **Type Safety** | MCP types required | Dict-based | Schema validation + type hints |
| **Best For** | Complex services | Simple services | Auto-generated tools |

### Validation Rules

- **Cannot mix approaches**: You cannot provide both `list_tools` function and `tools` config
- **Schema functions are exclusive**: Services with `@schema_function` decorated functions cannot mix with other approaches
- **Type consistency**: All tools/resources/prompts must use the same approach
- **MCP compatibility**: Services must have `type="mcp"`, provide MCP functions, or have `@schema_function` decorated functions to be detected

## MCP Protocol Functions

The MCP implementation supports all standard protocol functions:

### Tools
- `list_tools()`: List available tools
- `call_tool(name, arguments)`: Execute a tool with given arguments

### Prompts
- `list_prompts()`: List available prompts
- `get_prompt(name, arguments)`: Get a prompt by name with arguments

### Resources
- `list_resource_templates()`: List available resource templates
- `list_resources()`: List available resources
- `read_resource(uri)`: Read a resource by URI

### Notifications
- `progress_notification(token, progress, total)`: Send progress updates

## HTTP Endpoints

When MCP is enabled, services are accessible via HTTP endpoints following the JSON-RPC 2.0 specification:

### Base Endpoint
```
POST /{workspace}/mcp/{service_id}/
Content-Type: application/json
```

### Examples

**List Tools:**
```bash
curl -X POST http://localhost:9527/my-workspace/mcp/calculator-mcp/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/list",
    "params": {}
  }'
```

**Call Tool:**
```bash
curl -X POST http://localhost:9527/my-workspace/mcp/calculator-mcp/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "2",
    "method": "tools/call",
    "params": {
      "name": "calculate",
      "arguments": {
        "operation": "add",
        "a": 5,
        "b": 3
      }
    }
  }'
```

**List Prompts:**
```bash
curl -X POST http://localhost:9527/my-workspace/mcp/calculator-mcp/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "3",
    "method": "prompts/list",
    "params": {}
  }'
```

**Get Prompt:**
```bash
curl -X POST http://localhost:9527/my-workspace/mcp/calculator-mcp/ \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": "4",
    "method": "prompts/get",
    "params": {
      "name": "math_tutor",
      "arguments": {
        "topic": "algebra",
        "level": "intermediate"
      }
    }
  }'
```

## Client Configuration

### Using MCP HTTP Client

First, install the MCP SDK with HTTP support:

```bash
pip install mcp
```

Then use the HTTP client to connect to your Hypha MCP service:

```python
import asyncio
import httpx
from mcp.client.streamable_http import StreamableHTTPClientSession

async def test_mcp_client():
    # Your Hypha MCP service URL
    base_url = "https://hypha.aicell.io/my-workspace/mcp/calculator-mcp"
    
    async with httpx.AsyncClient() as httpx_client:
        session = StreamableHTTPClientSession(
            httpx_client=httpx_client,
            base_url=base_url
        )
        
        async with session.connect() as connection:
            # List available tools
            tools_result = await connection.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"- {tool.name}: {tool.description}")
            
            # Call a tool
            result = await connection.call_tool(
                name="calculate",
                arguments={"operation": "multiply", "a": 7, "b": 6}
            )
            print(f"Tool result: {result.content[0].text}")
            
            # List available prompts
            prompts_result = await connection.list_prompts()
            print("\nAvailable prompts:")
            for prompt in prompts_result.prompts:
                print(f"- {prompt.name}: {prompt.description}")
            
            # Get a prompt
            prompt_result = await connection.get_prompt(
                name="math_tutor",
                arguments={"topic": "calculus", "level": "advanced"}
            )
            print(f"Prompt: {prompt_result.messages[0].content.text}")

# Run the test
asyncio.run(test_mcp_client())
```

### Cursor Configuration

Add your Hypha MCP service to Cursor by editing `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hypha-calculator": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://hypha.aicell.io/my-workspace/mcp/calculator-mcp"
      ]
    }
  }
}
```

Or for local development:

```json
{
  "mcpServers": {
    "hypha-calculator-local": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:9527/my-workspace/mcp/calculator-mcp"
      ]
    }
  }
}
```

### Claude Desktop Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS or `%APPDATA%\Claude\claude_desktop_config.json` on Windows:

```json
{
  "mcpServers": {
    "hypha-calculator": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://hypha.aicell.io/my-workspace/mcp/calculator-mcp"
      ]
    },
    "hypha-file-processor": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://hypha.aicell.io/my-workspace/mcp/file-processor-mcp"
      ]
    }
  }
}
```

### Windsurf Configuration

Edit `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "hypha-calculator": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://hypha.aicell.io/my-workspace/mcp/calculator-mcp"
      ]
    }
  }
}
```

### Custom Headers and Authentication

For authenticated services, add custom headers:

```json
{
  "mcpServers": {
    "hypha-secure-service": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "https://hypha.aicell.io/my-workspace/mcp/secure-service",
        "--header",
        "Authorization: Bearer ${HYPHA_TOKEN}"
      ],
      "env": {
        "HYPHA_TOKEN": "your-hypha-token-here"
      }
    }
  }
}
```

## Advanced Features

### Error Handling

Implement comprehensive error handling:

```python
async def robust_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
    """Tool with robust error handling."""
    try:
        # Validate arguments
        if not arguments:
            raise ValueError("No arguments provided")
        
        # Tool-specific validation
        if name == "calculate":
            required_args = ["operation", "a", "b"]
            for arg in required_args:
                if arg not in arguments:
                    raise ValueError(f"Missing required argument: {arg}")
            
            # Perform calculation
            result = perform_calculation(arguments)
            return [types.TextContent(type="text", text=f"Result: {result}")]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except ValueError as e:
        # Client errors
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    except Exception as e:
        # Server errors
        logger.exception(f"Unexpected error in tool {name}: {e}")
        return [types.TextContent(type="text", text="Internal server error")]
```

## Testing and Debugging

### Testing MCP Services

```python
import asyncio
import pytest
from hypha_rpc import connect_to_server

async def test_mcp_service():
    """Test MCP service functionality."""
    server = await connect_to_server({"server_url": "http://localhost:9527"})
    
    # Register test service
    service_info = await server.register_service({
        "id": "test-mcp",
        "type": "mcp",
        "config": {"visibility": "public"},
        "list_tools": lambda: [
            {"name": "echo", "description": "Echo back input"}
        ],
        "call_tool": lambda name, args: [
            {"type": "text", "text": f"Echo: {args.get('text', '')}"}
        ]
    })
    
    print(f"Test service registered: {service_info['id']}")
    
    # Test HTTP endpoint
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:9527/{server.config.workspace}/mcp/test-mcp/",
            json={
                "jsonrpc": "2.0",
                "id": "test",
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"text": "Hello MCP!"}
                }
            }
        )
        
        result = response.json()
        assert result["result"]["content"][0]["text"] == "Echo: Hello MCP!"
        print("Test passed!")

# Run test
asyncio.run(test_mcp_service())
```

### Debug Logging

Enable debug logging for MCP services:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("hypha.mcp")

# Your MCP service code here
```

### MCP Inspector

Use the MCP Inspector to test your services:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run inspector
npx @modelcontextprotocol/inspector
```

Then connect to your Hypha MCP service at:
`https://hypha.aicell.io/my-workspace/mcp/your-service-id`

## Best Practices

### Service Design

1. **Clear Tool Descriptions**: Provide comprehensive descriptions for all tools
2. **Input Validation**: Validate all tool arguments thoroughly
3. **Error Handling**: Return meaningful error messages
4. **Resource Management**: Implement proper cleanup for resources
5. **Documentation**: Document all available functions and their parameters

### Performance Optimization

```python
import asyncio
from functools import lru_cache

class OptimizedMCPService:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=128)
    async def cached_operation(self, input_data: str) -> str:
        """Cached expensive operation."""
        # Simulate expensive operation
        await asyncio.sleep(0.1)
        return f"Processed: {input_data}"
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.ContentBlock]:
        """Optimized tool with caching."""
        if name == "expensive_operation":
            input_data = arguments.get("data", "")
            
            # Use cache for repeated requests
            if input_data in self.cache:
                result = self.cache[input_data]
            else:
                result = await self.cached_operation(input_data)
                self.cache[input_data] = result
            
            return [types.TextContent(type="text", text=result)]
```

### Security Considerations

```python
import re
from typing import Dict, Any

def validate_arguments(arguments: Dict[str, Any], name: str) -> None:
    """Validate tool arguments for security."""
    # Check for dangerous patterns
    dangerous_patterns = [
        r'__import__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'os\.system',
        r'subprocess\.',
    ]
    
    for arg_name, arg_value in arguments.items():
        if isinstance(arg_value, str):
            for pattern in dangerous_patterns:
                if re.search(pattern, arg_value):
                    raise ValueError(f"Dangerous pattern detected in {arg_name}")
    
    # Tool-specific validation
    if name == "file_operation":
        filepath = arguments.get("filepath", "")
        if ".." in filepath or filepath.startswith("/"):
            raise ValueError("Invalid file path")
```

## Troubleshooting

### Common Issues

1. **Service Not Found (404)**
   - Check that MCP is enabled: `--enable-mcp`
   - Verify service ID and workspace name
   - Ensure service is registered correctly

2. **JSON-RPC Errors**
   - Check request format matches JSON-RPC 2.0 spec
   - Verify method names (e.g., `tools/list`, `tools/call`)
   - Validate parameter structure

3. **Tool Execution Errors**
   - Check tool implementation for exceptions
   - Verify argument validation
   - Review server logs for detailed error messages

### Debug Steps

1. **Check Service Registration**:
   ```bash
   curl -X GET "https://hypha.aicell.io/my-workspace/services"
   ```

2. **Test MCP Endpoint**:
   ```bash
   curl -X POST "https://hypha.aicell.io/my-workspace/mcp/my-service/" \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc": "2.0", "id": "test", "method": "tools/list", "params": {}}'
   ```

3. **Enable Debug Logging**:
   ```bash
   python -m hypha.server --enable-mcp --log-level DEBUG
   ```

### Client Configuration Issues

1. **Cursor Not Connecting**:
   - Check `~/.cursor/mcp.json` syntax
   - Verify URL accessibility
   - Restart Cursor after configuration changes

2. **Claude Desktop Errors**:
   - Restart Claude Desktop app
   - Check configuration file permissions
   - Verify mcp-remote package is installed

3. **Network Issues**:
   - Check firewall settings
   - Verify SSL certificates for HTTPS
   - Test with local HTTP for development

## Conclusion

Hypha's MCP service support provides a powerful way to expose your services to MCP-compatible clients. By following the MCP protocol specification and using Hypha's simplified registration process, you can create tools, prompts, and resources that integrate seamlessly with popular AI applications like Cursor, Claude Desktop, and Windsurf.

The combination of Hypha's infrastructure capabilities with MCP's standardized protocol creates a robust platform for building and deploying AI-accessible services that can be discovered and used by a wide range of client applications. 