# Hypha Quick Start Guide

This guide provides quick examples for creating and using Hypha services in both Python and JavaScript.

## Installation

```bash
pip install -U hypha
```

## Starting the Server

```bash
python -m hypha.server --host=0.0.0.0 --port=9527
```

## Registering a Service

### Python

```python
# Async version
import asyncio
from hypha_rpc import connect_to_server

async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})
    
    def hello(name):
        print("Hello " + name)
        return "Hello " + name

    svc = await server.register_service({
        "name": "Hello World",
        "id": "hello-world",
        "config": {"visibility": "public"},
        "hello": hello
    })
    
    print(f"Service registered with ID: {svc.id}")
    await server.serve()  # Keep the server running

if __name__ == "__main__":
    server_url = "http://localhost:9527"
    asyncio.run(start_server(server_url))
```

### JavaScript

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.51/dist/hypha-rpc-websocket.min.js"></script>
<script>
async function registerService() {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "http://localhost:9527"
    });
    
    // Define service function
    function hello(name) {
        console.log("Hello " + name);
        return "Hello " + name;
    }
    
    // Register service
    const svc = await server.register_service({
        name: "Hello World",
        id: "hello-world",
        config: {
            visibility: "public"
        },
        hello: hello
    });
    
    console.log(`Service registered with ID: ${svc.id}`);
    
    // For services that need to run continuously
    await server.serve();
}

// Call the function to register your service
registerService().catch(console.error);
</script>
```

## Using a Service

### Python Client

```python
# Async client
import asyncio
from hypha_rpc import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "http://localhost:9527"})
    
    # Replace with your actual service ID from registration output
    svc = await server.get_service("your-workspace/your-service-id:hello-world")
    result = await svc.hello("World")
    print(result)  # Outputs: Hello World

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript Client

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.51/dist/hypha-rpc-websocket.min.js"></script>
<script>
async function useService() {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "http://localhost:9527"
    });
    
    // Replace with your actual service ID from registration output
    const svc = await server.get_service("your-workspace/your-service-id:hello-world");
    const result = await svc.hello("World");
    console.log(result);  // Outputs: Hello World
}

// Call the function to use the service
useService().catch(console.error);
</script>
```

## Authentication for Private Services

For services marked with `visibility: "private"`, you'll need to authenticate:

### Python

```python
import asyncio
from hypha_rpc import login, connect_to_server

async def authenticate_and_use():
    token = await login({"server_url": "http://localhost:9527"})
    server = await connect_to_server({
        "server_url": "http://localhost:9527", 
        "token": token
    })
    
    # Now you can use private services
    svc = await server.get_service("your-workspace/your-service-id:hello-world")
    # ...

if __name__ == "__main__":
    asyncio.run(authenticate_and_use())
```

### JavaScript

```javascript
async function authenticateAndUse() {
    const token = await hyphaWebsocketClient.login({
        server_url: "http://localhost:9527"
    });
    
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "http://localhost:9527",
        token: token
    });
    
    // Now you can use private services
    const svc = await server.get_service("your-workspace/your-service-id:hello-world");
    // ...
}
```

For more detailed explanations, see the [full documentation](./getting-started.md). 