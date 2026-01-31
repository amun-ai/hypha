# Getting Started

Welcome to Hypha! This guide will walk you through the basics of installing, running, and using Hypha - a powerful platform for building distributed applications and services.

## What is Hypha?

Hypha is a server platform that enables you to:
- **Create and share services**: Build reusable functions that can be called from anywhere
- **Connect different applications**: Bridge Python, JavaScript, and other languages seamlessly  
- **Build distributed systems**: Services can run on different machines and communicate easily
- **Manage workspaces**: Organize your services and collaborate with others

Key concepts:
- **Services**: Collections of functions that you can register and call remotely
- **Workspaces**: Isolated environments where services are organized
- **Clients**: Applications that connect to Hypha to register or use services

## Installation

To install the Hypha package, run:

```bash
pip install -U hypha
```

## Starting Your First Hypha Server

Start a basic Hypha server with:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527
```

You should see output indicating the server is running. Visit [http://localhost:9527](http://localhost:9527) to confirm it's working - you'll see the Hypha server interface.

> **Need more advanced server configuration?** Check out the [Configuration Guide](configurations.md) for options like built-in S3 storage, server apps, static file serving, and production deployment.

## Your First Service

Let's create a simple "Hello World" service to understand how Hypha works.

### Step 1: Register a Service

A service is a collection of functions that you make available to other applications. Here's how to create and register your first service:

<!-- tabs:start -->
#### ** Python (Async) **

```python
import asyncio
from hypha_rpc import connect_to_server

async def start_server(server_url):
    # Connect to the Hypha server
    async with connect_to_server({"server_url": server_url}) as server:
        # Define a simple function
        def hello(name):
            print(f"Hello {name}!")
            return f"Hello {name}!"

        # Register the function as a service
        service = await server.register_service({
            "name": "Hello World Service",
            "id": "hello-world",
            "config": {
                "visibility": "public"  # Anyone can use this service
            },
            "hello": hello  # Make our function available
        })
        
        print(f"✅ Service registered!")
        print(f"Service ID: {service.id}")
        print(f"Workspace: {server.config.workspace}")
        
        # Keep the service running
        await server.serve()

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    asyncio.run(start_server(server_url))
```

#### ** Python (Sync) **

```python
from hypha_rpc.sync import connect_to_server

def start_server(server_url):
    # Connect to the Hypha server  
    with connect_to_server({"server_url": server_url}) as server:
        # Define a simple function
        def hello(name):
            print(f"Hello {name}!")
            return f"Hello {name}!"

        # Register the function as a service
        service = server.register_service({
            "name": "Hello World Service", 
            "id": "hello-world",
            "config": {
                "visibility": "public"  # Anyone can use this service
            },
            "hello": hello  # Make our function available
        })
        
        print(f"✅ Service registered!")
        print(f"Service ID: {service.id}")
        print(f"Workspace: {server.config.workspace}")

        # Keep the service running (blocks here)
        input("Press Enter to stop the service...")

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    start_server(server_url)
```

#### ** JavaScript **

First, install the JavaScript client:
```bash
npm install hypha-rpc
```

Or include it in your HTML:
```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.93/dist/hypha-rpc-websocket.min.js"></script>
```

```javascript
async function startServer(serverUrl) {
    // Connect to the Hypha server
  const server = await hyphaWebsocketClient.connectToServer({
        server_url: serverUrl
    });

    // Define a simple function
    function hello(name) {
        console.log(`Hello ${name}!`);
        return `Hello ${name}!`;
    }

    // Register the function as a service
    const service = await server.registerService({
        name: "Hello World Service",
    id: "hello-world",
    config: {
            visibility: "public"  // Anyone can use this service
        },
        hello: hello  // Make our function available
    });
    
    console.log("✅ Service registered!");
    console.log(`Service ID: ${service.id}`);
    console.log(`Workspace: ${server.config.workspace}`);
    
    // Service will keep running until page is closed
}

// Start the service
const serverUrl = "http://localhost:9527";
startServer(serverUrl);
```
<!-- tabs:end -->

**What's happening here?**
1. We connect to the Hypha server
2. We define a simple `hello()` function 
3. We register it as a service with ID "hello-world"
4. The service becomes available for other clients to use
5. We keep the service running so others can call it

Save this code and run it. You should see the service ID printed - something like `ws-user-amazing-butterfly-12345/ABC123:hello-world`. This unique ID allows other clients to find and use your service.

> Tip: You can run the client from another machine. Just change `server_url` to the server's IP or domain.

### Step 2: Use the Service

Now let's create a client that uses the service we just registered:

<!-- tabs:start -->
#### ** Python (Async) **

```python
import asyncio
from hypha_rpc import connect_to_server

async def use_service():
    # Connect to the same Hypha server
    async with connect_to_server({"server_url": "http://localhost:9527"}) as server:
        # Get the service by its ID (replace with your actual service ID)
        service = await server.get_service("REPLACE_WITH_YOUR_SERVICE_ID")
        
        # Call the hello function
        result = await service.hello("World")
        print(f"Service returned: {result}")

if __name__ == "__main__":
    asyncio.run(use_service())
```

#### ** Python (Sync) **

```python
from hypha_rpc.sync import connect_to_server

def use_service():
    # Connect to the same Hypha server
    with connect_to_server({"server_url": "http://localhost:9527"}) as server:
        # Get the service by its ID (replace with your actual service ID)
        service = server.get_service("REPLACE_WITH_YOUR_SERVICE_ID")
        
        # Call the hello function
        result = service.hello("World")
        print(f"Service returned: {result}")

if __name__ == "__main__":
    use_service()
```

#### ** JavaScript **

```javascript
async function useService() {
    // Connect to the same Hypha server
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "http://localhost:9527"
    });
    
    // Get the service by its ID (replace with your actual service ID)
    const service = await server.getService("REPLACE_WITH_YOUR_SERVICE_ID");
    
    // Call the hello function
    const result = await service.hello("World");
    console.log(`Service returned: ${result}`);
    
    // Clean up
    server.disconnect();
}

// Use the service
useService();
```
<!-- tabs:end -->

**Important:** Replace `REPLACE_WITH_YOUR_SERVICE_ID` with the actual service ID that was printed when you registered the service.

> Note: Prefer async APIs when possible. If you need sync APIs, the examples above show both. See the hypha-rpc "Synchronous Wrapper" docs for details.

**What's happening here?**
1. We connect to the Hypha server as a client
2. We get a reference to the service using its ID
3. We call the `hello()` function remotely
4. The function executes on the service provider and returns the result

## HTTP Endpoints: Every Service is a Web API

**Automatic HTTP Exposure**: Every Hypha service is automatically exposed as HTTP endpoints! You can call your service functions directly via HTTP requests without any additional configuration.

### HTTP URL Pattern

When you register a service, it becomes available at:

```text
http://your-server:port/{workspace}/services/{service-id}/{function-name}
```

For our "Hello World" service example above, if your service ID is `ws-user-amazing-butterfly-12345/ABC123:hello-world`, you can call it via HTTP:

```bash
# GET request with query parameters
curl "http://localhost:9527/ws-user-amazing-butterfly-12345/services/ABC123:hello-world/hello?name=World"

# POST request with JSON body
curl -X POST http://localhost:9527/ws-user-amazing-butterfly-12345/services/ABC123:hello-world/hello \
  -H "Content-Type: application/json" \
  -d '{"name": "World"}'
```

Or simply open this URL in your browser:

```text
http://localhost:9527/ws-user-amazing-butterfly-12345/services/ABC123:hello-world/hello?name=World
```

### List All Services

You can also list all available services in a workspace:
```bash
curl http://localhost:9527/ws-user-amazing-butterfly-12345/services
```

**Key Benefits:**

- **Zero Configuration**: No need to set up REST APIs or web frameworks
- **Multiple Formats**: Supports JSON, MessagePack, and query parameters
- **CORS Enabled**: Works directly from web browsers
- **Streaming Support**: Generator functions return Server-Sent Events

## Understanding Workspaces

When you connect to Hypha, you're automatically assigned to a **workspace** - think of it as your personal area where your services live. Each workspace has a unique name like `ws-user-amazing-butterfly-12345`.

- Services registered in your workspace are private by default
- You can make services public so anyone can use them
- You can share your workspace with others for collaboration

## User Authentication

For more advanced features, you might want to authenticate with the server:

<!-- tabs:start -->
#### ** Python (Async) **

```python
from hypha_rpc import login, connect_to_server

async def authenticated_connection():
    # Login and get a token
    token = await login({"server_url": "http://localhost:9527"})
    # A URL will be printed for you to open and login
    
    # Connect with the token
    async with connect_to_server({
        "server_url": "http://localhost:9527", 
        "token": token
    }) as server:
        print(f"Connected as authenticated user!")
        print(f"Workspace: {server.config.workspace}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(authenticated_connection())
```

#### ** Python (Sync) **

```python
from hypha_rpc.sync import login, connect_to_server

def authenticated_connection():
    # Login and get a token
    token = login({"server_url": "http://localhost:9527"})
    # A URL will be printed for you to open and login
    
    # Connect with the token
    with connect_to_server({
        "server_url": "http://localhost:9527",
        "token": token
    }) as server:
        print(f"Connected as authenticated user!")
        print(f"Workspace: {server.config.workspace}")

if __name__ == "__main__":
    authenticated_connection()
```

#### ** JavaScript **

```javascript
async function authenticatedConnection() {
    // Login and get a token
    const token = await hyphaWebsocketClient.login({
        server_url: "http://localhost:9527",
        login_callback: (context) => {
            console.log(`Please open: ${context.login_url}`);
        }
    });
    
    // Connect with the token
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "http://localhost:9527",
        token: token
    });
    
    console.log("Connected as authenticated user!");
    console.log(`Workspace: ${server.config.workspace}`);
}

authenticatedConnection();
```
<!-- tabs:end -->

## MCP Tools: AI Assistant Integration

**Model Context Protocol (MCP)**: Every Hypha service with properly annotated functions automatically becomes available as MCP tools for AI agents like Claude Code, Cursor, and others!

### Enhancing Functions for MCP

To make your functions available as MCP tools, use the `@schema_function` decorator. Let's enhance our Hello World service:

<!-- tabs:start -->
#### ** Python (Enhanced with MCP) **

```python
import asyncio
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function
from pydantic import Field

async def start_enhanced_server(server_url):
    async with connect_to_server({"server_url": server_url}) as server:
        # Enhanced hello function with schema annotation
        @schema_function
        def hello(
            name: str = Field(..., description="The name of the person to greet"),
            greeting: str = Field("Hello", description="The greeting word to use")
        ) -> str:
            """Generate a personalized greeting message.
            
            This function creates a customized greeting by combining
            a greeting word with a person's name.
            """
            message = f"{greeting} {name}!"
            print(message)
            return message

        # Register the enhanced service
        service = await server.register_service({
            "name": "Enhanced Hello World Service",
            "id": "enhanced-hello-world",
            "config": {
                "visibility": "public"
            },
            "hello": hello
        })
        
        print(f"✅ Enhanced service registered!")
        print(f"Service ID: {service.id}")
        print(f"MCP endpoint: /{server.config.workspace}/mcp/enhanced-hello-world/mcp")
        
        await server.serve()

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    asyncio.run(start_enhanced_server(server_url))
```
<!-- tabs:end -->

### MCP URL Pattern

When you register a service with `@schema_function` decorated functions, it becomes available as MCP tools at:

```text
http://your-server:port/{workspace}/mcp/{service-id}/mcp
```

For our enhanced service:

```text
http://localhost:9527/ws-user-amazing-butterfly-12345/mcp/enhanced-hello-world/mcp
```

### Using with AI Coding Assistants

1. **Claude Desktop**: Add the MCP server configuration:
   ```json
   {
     "mcpServers": {
       "hypha-hello": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-fetch", "http://localhost:9527/ws-user-amazing-butterfly-12345/mcp/enhanced-hello-world/mcp"]
       }
     }
   }
   ```

2. **Direct MCP Connection**: Use any MCP client to connect to your service endpoint

3. **Tool Discovery**: AI assistants will automatically discover your `hello` function and can call it with proper parameters

### What @schema_function Provides

- **Type Annotations**: Converts Python type hints to JSON Schema
- **Parameter Descriptions**: Use `Field()` to add detailed parameter descriptions
- **Validation**: Automatic input validation with Pydantic models
- **Documentation**: Function docstrings become tool descriptions
- **Default Values**: Support for default parameters and optional fields
- **MCP Compatibility**: Makes functions discoverable by AI assistants

**Key Components:**
- `Field(...)`: Required parameter (no default value)
- `Field("default", description="...")`: Optional parameter with default
- `Field(None, description="...")`: Optional parameter that can be None

**Advanced Example:**
```python
@schema_function
def process_data(
    data: str = Field(..., description="Input data to process"),
    format: str = Field("json", description="Output format: json, csv, or xml"),
    include_metadata: bool = Field(False, description="Whether to include metadata"),
    max_items: Optional[int] = Field(None, description="Maximum number of items to process")
) -> Dict[str, Any]:
    """Process data with various formatting options."""
    # Your processing logic here
    return {"processed": data, "format": format}
```

**Example AI Assistant Usage:**

```
User: "Use the hello tool to greet Alice with a custom greeting"
AI Assistant: I'll use the hello function to greet Alice.
[Calls hello(name="Alice", greeting="Hi there")]
Result: "Hi there Alice!"
```


### Advanced: Live Service Updates

Service can be updated by calling `server.register_service` again with `{"overwrite": True}`:

For example:
```python
service = await server.register_service({
    "name": "Hello World Service",
    "id": "hello-world",
    "config": {
        "visibility": "public"  # Anyone can use this service
    },
    "hello2": hello2  # Make our function available
}, {"overwrite": True})
```

On the client side, you can create a service proxy that automatically refreshes:

```python
import asyncio
import random

async def create_live_service_proxy(server, sid, config=None, interval=5.0, jitter=0.0):
    """
    Return a dict-like service proxy with dot-attribute access.
    It refreshes itself at the given interval and supports await proxy._dispose().
    
    Args:
        server: The Hypha server connection
        sid: Service ID to connect to
        config: Service configuration (optional)
        interval: Refresh interval in seconds
        jitter: Random jitter to add to interval (prevents thundering herd)
    """
    proxy = await server.get_service(sid, config)

    async def refresher():
        try:
            while True:
                await asyncio.sleep(interval + random.random() * jitter)
                latest = await server.get_service(sid, config)
                proxy.clear()
                proxy.update(latest)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(refresher())

    async def _dispose():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    proxy._dispose = _dispose
    return proxy

# Usage example
async def use_live_service():
    async with connect_to_server({"server_url": "http://localhost:9527"}) as server:
        # Create a live service proxy that updates every 5 seconds with 1 second jitter
        svc = await create_live_service_proxy(
            server, "hello-world", interval=5.0, jitter=1.0
        )
        
        # Use the service normally - it will auto-refresh in the background
        result = await svc.hello("World")
        print(f"Result: {result}")

        # You can access svc.hello2() as well after the service is updated
        
        # Clean up when done
        await svc._dispose()
```

**When to use live updates:**
- Services that might restart or update their functions
- Dynamic service discovery scenarios  
- Long-running clients that need to stay current with service changes
- Load balancing across multiple service instances


## Next Steps

> Compatibility note: If you are using Hypha older than 0.15.x, use `imjoy-rpc` instead of `hypha-rpc`.

Congratulations! You've learned the basics of Hypha:

- ✅ Started a Hypha server
- ✅ Registered your first service
- ✅ Connected clients to use services
- ✅ Set up live service updates for dynamic scenarios
- ✅ Used services via HTTP endpoints
- ✅ Enhanced functions for MCP tool integration
- ✅ Understood workspaces and authentication

### Ready for More?

Now that you understand the basics, you can explore more advanced features:

- **[Configuration Guide](configurations.md)** - Advanced server configuration, storage options, custom workers, and production deployment
- **[Server Apps](apps.md)** - Build web applications that run on the server
- **[Artifact Management](artifact-manager.md)** - Store and version your data and models
- **[Vector Search](vector-search.md)** - Add semantic search capabilities to your applications

### Common Patterns

Here are some common ways people use Hypha:

1. **Microservices**: Break your application into small, reusable services
2. **HTTP APIs**: Every service automatically becomes a REST API
3. **AI Tool Integration**: Use `@schema_function` to make functions available to AI assistants
4. **Dynamic Service Discovery**: Use live service updates for services that change frequently
5. **Distributed Computing**: Run compute-intensive tasks across multiple machines
6. **Real-time Communication**: Build chat applications, collaborative tools, etc.
7. **AI/ML Workflows**: Chain together different AI models and data processing steps

### Getting Help

- **Documentation**: Explore the other guides in this documentation
- **Examples**: Check the `examples/` directory in the Hypha repository
- **Community**: Join our discussions on GitHub

Happy building with Hypha! 🚀