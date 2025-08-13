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
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.66/dist/hypha-rpc-websocket.min.js"></script>
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

## Next Steps

> Compatibility note: If you are using Hypha older than 0.15.x, use `imjoy-rpc` instead of `hypha-rpc`.

Congratulations! You've learned the basics of Hypha:
- ✅ Started a Hypha server
- ✅ Registered your first service
- ✅ Connected clients to use services
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
2. **API Gateway**: Expose your services through HTTP endpoints
3. **Distributed Computing**: Run compute-intensive tasks across multiple machines
4. **Real-time Communication**: Build chat applications, collaborative tools, etc.
5. **AI/ML Workflows**: Chain together different AI models and data processing steps

### Getting Help

- **Documentation**: Explore the other guides in this documentation
- **Examples**: Check the `examples/` directory in the Hypha repository
- **Community**: Join our discussions on GitHub

Happy building with Hypha! 🚀