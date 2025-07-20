# Serverless Functions

Hypha provides **two main approaches** for building HTTP endpoints and web applications:

## 🚀 **Quick Comparison: Functions vs ASGI**

| Feature | **Serverless Functions** | **ASGI Applications** |
|---------|---------------------------|----------------------|
| **Best for** | Simple HTTP endpoints, APIs, microservices | Full web applications, complex routing |
| **Setup complexity** | ✅ Simple - just register functions | ⚠️  More setup - FastAPI/Django apps |
| **Routing** | ✅ Automatic URL mapping | 🔧 Manual route definitions |
| **Performance** | ✅ Lightweight, fast startup | ⚠️  Heavier, slower startup |
| **Nested paths** | ✅ Built-in support (`api/v1/users`) | 🔧 Manual implementation |
| **Fallback handling** | ✅ Built-in default function | 🔧 Manual 404 handling |
| **Framework support** | 🔧 Custom functions only | ✅ FastAPI, Django, Starlette, etc. |
| **Use cases** | REST APIs, webhooks, simple services | Full websites, complex web apps |

**👉 Choose Functions for:** Simple APIs, microservices, webhooks, quick prototypes  
**👉 Choose ASGI for:** Full web applications, existing FastAPI/Django projects, complex routing needs

---

## Serverless Functions

You can register serverless function services which will be served as HTTP endpoints.

A serverless function can be defined in Javascript or Python:

```javascript
async function(event, context) {
    // your server-side functionality
}
```

It takes two arguments:
 * `event`: request event similar to the scope defined in the [ASGI spec](https://github.com/django/asgiref/blob/main/specs/www.rst#http-connection-scope). The types are the same as the `scope` defined in the ASGI spec, except the following fields:
    - `query_string`: a string (instead of bytes) with the query string
    - `raw_path`: a string (instead of bytes) with the raw path
    - `headers`: a dictionary (instead of an iterable) with the headers
    - `body`: the request body (bytes or arrayBuffer), will be None if empty
    - `context`: Contains user and environment related information.

To register the functions, call `api.register_service` with `type="functions"`.

```javascript
await api.register_service({
    "id": "hello-functions",
    "type": "functions",
    "config": {
        "visibility": "public",
    },
    "hello-world": async function(event) {
        return {
            status: 200,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message: "Hello World"})
        };
    },
    "index": async function(event) {
        return {
            status: 200,
            body: "Home page"
        };
    }
})
```

In the above example, `hello-world` is a function served under `/{workspace}/apps/hello-functions/hello-world/`.
You can define multiple functions in a single service. 

Specifically for the index or home page, you can define a function named `index`, and it will be served under `/{workspace}/apps/hello-functions/` (note the trailing slash) and `/{workspace}/apps/hello-functions/index`.

## Nested Functions

You can organize functions in a nested structure for better organization:

```javascript
await api.register_service({
    "id": "nested-functions",
    "type": "functions",
    "config": {
        "visibility": "public",
    },
    "api": {
        "v1": {
            "users": async function(event) {
                return {
                    status: 200,
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({users: ["alice", "bob"]})
                };
            },
            "posts": async function(event) {
                return {
                    status: 200,
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({posts: []})
                };
            }
        },
        "v2": {
            "users": async function(event) {
                return {
                    status: 200,
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({users: [], version: "v2"})
                };
            }
        }
    }
})
```

In this example, the functions would be accessible at:
- `/{workspace}/apps/nested-functions/api/v1/users/`
- `/{workspace}/apps/nested-functions/api/v1/posts/`  
- `/{workspace}/apps/nested-functions/api/v2/users/`

## Default Fallback Function

You can define a `default` function that will handle all requests that don't match any specific function:

```javascript
await api.register_service({
    "id": "fallback-functions",
    "type": "functions",
    "config": {
        "visibility": "public",
    },
    "hello": async function(event) {
        return {
            status: 200,
            body: "Hello from specific function"
        };
    },
    "default": async function(event) {
        const path = event.function_path || "/";
        return {
            status: 200,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: "Default handler", 
                path: path,
                method: event.method
            })
        };
    }
})
```

In this example:
- `/{workspace}/apps/fallback-functions/hello/` will call the `hello` function
- `/{workspace}/apps/fallback-functions/anything-else/` will call the `default` function
- The `default` function receives the requested path in `event.function_path`

## Serverless Functions Example: Creating a Data Store Service

While the Hypha platform excels at providing services that can be easily shared and utilized via Hypha clients, there is a growing need in the technology landscape to interface seamlessly with web applications. This requirement is particularly pronounced when dealing with outputs from various computational or data-driven services that need to be integrated directly into web environments—for example, displaying generated image files within a Markdown page or embedding JSON data into a web application for dynamic content generation.

To demonstrate how serverless functions can be used in this case, we provide an example implementation of `HyphaDataStore` for storing, managing, and retrieving data through the Hypha serverless functions. See the [Hypha Data Store](https://github.com/amun-ai/hypha/blob/main/docs/hypha_data_store.py) implementation for more details. This service allows users to share files and JSON-serializable objects with others by generating accessible HTTP URLs.

## 🎯 **Choosing Between Functions and ASGI**

### When to Use Serverless Functions

**✅ Perfect for:**
- **REST APIs** with simple CRUD operations
- **Webhooks** and event handlers
- **Microservices** with focused functionality  
- **Quick prototypes** and MVPs
- **Data processing endpoints** with clear input/output
- **Integration APIs** connecting different systems

**Example Use Cases:**
```javascript
// Simple REST API
api.register_service({
    "id": "user-api",
    "type": "functions",
    "users": {
        "list": async (event) => ({ users: await getUsers() }),
        "create": async (event) => await createUser(JSON.parse(event.body)),
        "delete": async (event) => await deleteUser(event.path.split('/')[3])
    }
})
```

### When to Use ASGI Applications

**✅ Perfect for:**
- **Full web applications** with complex UI
- **Existing FastAPI/Django projects** you want to deploy
- **Advanced routing** with middleware, authentication
- **File uploads/downloads** with streaming
- **WebSocket connections** for real-time features
- **Complex business logic** requiring framework features

**Example: FastAPI Integration**
```python
from fastapi import FastAPI, UploadFile
from hypha_rpc import connect_to_server

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile):
    return {"filename": file.filename}

# Deploy as ASGI service
api = await connect_to_server({"server_url": "..."})
await api.register_service({
    "id": "file-service",
    "type": "asgi", 
    "serve": app
})
```

### 🔗 **Using Both Together**

You can combine both approaches in the same application:

```javascript
// Functions for simple APIs
await api.register_service({
    "id": "simple-api",
    "type": "functions",
    "health": async (event) => ({ status: "ok" }),
    "version": async (event) => ({ version: "1.0.0" })
})

// ASGI for complex web interface  
await api.register_service({
    "id": "admin-panel",
    "type": "asgi",
    "serve": fastapi_admin_app
})
```

**URL Structure:**
- Functions: `/{workspace}/apps/simple-api/health`
- ASGI: `/{workspace}/apps/admin-panel/dashboard`

### 📊 **Performance Considerations**

**Serverless Functions:**
- ⚡ **Cold start**: ~10-50ms  
- ⚡ **Memory usage**: Low
- ⚡ **Concurrent requests**: High throughput

**ASGI Applications:**
- 🐌 **Cold start**: ~100-500ms (depending on framework)
- 📈 **Memory usage**: Higher  
- 📈 **Concurrent requests**: Framework dependent

### 🛠️ **Development Workflow**

1. **Start with Functions** for simple endpoints
2. **Migrate to ASGI** when you need complex features
3. **Use both** for different parts of your application
4. **Deploy everything** through Hypha's unified system

### 📚 **Related Documentation**

- **[ASGI Applications](asgi-apps.md)** - Deploy FastAPI, Django, and other ASGI frameworks
- **[Server Apps](apps.md)** - Overall application management and deployment
- **[Autoscaling](autoscaling.md)** - Scale both functions and ASGI apps automatically

---

**💡 Pro Tip:** Start with serverless functions for rapid development, then migrate specific endpoints to ASGI applications as your complexity grows. Hypha's unified deployment system makes this transition seamless!
