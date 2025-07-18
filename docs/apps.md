# Server Apps

The **Server Apps** service is a core Hypha service that enables serverless computing and dynamic application management within the Hypha ecosystem. It provides a powerful platform for running various types of applications in a serverless fashion - meaning applications are only started on-demand and automatically scaled based on usage.

> **Related Documentation:**
> - [ASGI Applications](asgi-apps.md) - Serve FastAPI, Django, and other ASGI applications
> - [Serverless Functions](serverless-functions.md) - Deploy HTTP endpoint functions
> - [Autoscaling](autoscaling.md) - Automatic scaling based on load

## Two Types of Applications

The Server Apps service supports two distinct approaches for creating and deploying applications:

### 1. Hypha Apps (Built-in Templates + Config)

**Hypha Apps** use built-in templates combined with configuration to create standardized applications. This approach provides:

- **Built-in Templates**: Pre-configured templates for common app types
- **Supported Types**: `web-worker`, `web-python`, `iframe`, `window`
- **Configuration-Based**: Apps are defined through configuration objects
- **Template Rendering**: Source code is automatically generated from templates using Jinja2
- **Simplified Development**: No need to write HTML boilerplate

**Example Hypha App:**
```python
# Simple web-worker app using built-in template
app_info = await controller.install(
    source='''
    api.export({
        async add(a, b) { return a + b; },
        async multiply(a, b) { return a * b; }
    });
    ''',
    config={
        "name": "Calculator",
        "type": "web-worker",  # Built-in template type
        "version": "1.0.0"
    }
)
```

### 2. Generic Apps (Custom Source + Manifest)

**Generic Apps** allow complete control over the application structure and support custom application types through custom server-app-workers:

- **Custom Source**: Full control over HTML, CSS, and JavaScript
- **Custom Manifest**: Direct manifest specification without conversion
- **Custom Types**: Support for any application type via custom workers
- **Raw HTML**: Can include complete HTML documents
- **Extensible**: New app types can be added through custom workers

**Example Generic App:**
```python
# Custom HTML app with manifest
app_info = await controller.install(
    source='''
    <!DOCTYPE html>
    <html>
    <head><title>Custom App</title></head>
    <body>
        <h1>Custom Application</h1>
        <script>
            // Custom application logic
        </script>
    </body>
    </html>
    ''',
    manifest={
        "name": "Custom App",
        "type": "custom-html",  # Custom type
        "version": "1.0.0",
        "entry_point": "index.html"
    }
)
```

## How It Works

The Server Apps service operates by running a full-featured web browser (Chromium) on the server side for built-in types, or delegating to custom workers for specialized types. This architecture provides:

- **Full Browser Environment**: Standard web APIs, DOM manipulation, modern JavaScript features
- **Sandboxed Execution**: Isolated execution contexts for security
- **Server-Side Rendering**: Processing without client-side overhead
- **Custom Workers**: Extensible support for specialized application types
- **Automatic Scaling**: Dynamic resource allocation based on demand

**Key Features:**
- **Dynamic App Installation**: Install apps from source code, URLs, or local files
- **Multiple App Types**: Support for built-in types and custom types via workers
- **Lazy Loading**: Apps start automatically when accessed
- **Startup Configuration**: Configure default startup parameters
- **HTTP Access**: Direct HTTP access to web applications
- **Lifecycle Management**: Complete control over installation, starting, stopping, and uninstalling
- **Autoscaling**: Automatic scaling based on load (see [Autoscaling documentation](autoscaling.md))

For web applications that need HTTP endpoints, consider using [ASGI Applications](asgi-apps.md) for full web frameworks like FastAPI, or [Serverless Functions](serverless-functions.md) for simple HTTP endpoints.

---

## Hypha Apps (Built-in Templates)

Hypha Apps use built-in templates and configuration to create standardized applications. The system automatically generates the necessary HTML, CSS, and JavaScript based on your configuration and source code.

### Supported Built-in Types

#### Web Worker Apps
Run JavaScript in a web worker environment, suitable for CPU-intensive tasks:

```python
web_worker_source = '''
api.export({
    async processData(data) {
        // CPU-intensive processing
        return data.map(x => x * 2);
    },
    async fibonacci(n) {
        if (n <= 1) return n;
        return await this.fibonacci(n - 1) + await this.fibonacci(n - 2);
    }
});
'''

app_info = await controller.install(
    source=web_worker_source,
    config={
        "name": "Data Processor",
        "type": "web-worker",
        "version": "1.0.0"
    }
)
```

#### Web Python Apps
Run Python code in a web environment using Pyodide:

```python
web_python_source = '''
<config lang="yaml">
name: Python Data Processor
type: web-python
version: 1.0.0
</config>

<script lang="python">
import numpy as np
from hypha_rpc import api

def process_array(arr):
    return np.array(arr) ** 2

def analyze_data(data):
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

api.export({
    "process_array": process_array,
    "analyze_data": analyze_data
})
</script>
'''

app_info = await controller.install(
    source=web_python_source,
    config={"type": "web-python"}
)
```

#### Window Apps
Run applications in a window context with full DOM access:

```python
window_app_source = '''
<config lang="yaml">
name: Interactive Dashboard
type: window
version: 1.0.0
</config>

<script lang="javascript">
class Dashboard {
    constructor() {
        this.setupUI();
    }
    
    setupUI() {
        document.body.innerHTML = `
            <h1>Interactive Dashboard</h1>
            <button id="updateBtn">Update Data</button>
            <div id="dataDisplay"></div>
        `;
        
        document.getElementById('updateBtn').onclick = () => {
            this.updateData();
        };
    }
    
    updateData() {
        const data = Math.random() * 100;
        document.getElementById('dataDisplay').innerHTML = `Data: ${data.toFixed(2)}`;
    }
}

api.export({
    dashboard: new Dashboard()
});
</script>
'''

app_info = await controller.install(
    source=window_app_source,
    config={"type": "window"}
)
```

### Template Configuration

Built-in templates use Jinja2 for rendering and support various configuration options:

```python
app_info = await controller.install(
    source=app_source,
    config={
        "name": "My App",
        "type": "web-worker",
        "version": "1.0.0",
        "description": "A sample web worker application",
        "requirements": ["numpy", "pandas"],  # For web-python apps
        "entry_point": "index.html",  # Custom entry point
        "startup_config": {
            "timeout": 30,
            "wait_for_service": "default",
            "stop_after_inactive": 300
        },
        "autoscaling": {  # Enable autoscaling (see autoscaling.md)
            "enabled": True,
            "min_instances": 1,
            "max_instances": 5,
            "target_requests_per_instance": 100
        }
    }
)
```

> **Note:** For autoscaling configuration and advanced scaling options, see the [Autoscaling documentation](autoscaling.md).

---

## Generic Apps (Custom Source + Manifest)

Generic Apps provide complete control over the application structure and support custom application types through custom server-app-workers.

### Custom HTML Applications

You can install complete HTML applications with full control over structure:

```python
custom_html_app = '''
<!DOCTYPE html>
<html>
<head>
    <title>Custom Application</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        button { padding: 10px 20px; margin: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Custom Application</h1>
        <button onclick="performAction()">Click Me</button>
        <div id="result"></div>
    </div>
    
    <script>
        function performAction() {
            document.getElementById('result').innerHTML = 
                `Action performed at ${new Date().toLocaleTimeString()}`;
        }
    </script>
</body>
</html>
'''

app_info = await controller.install(
    source=custom_html_app,
    manifest={
        "name": "Custom HTML App",
        "type": "window",
        "version": "1.0.0",
        "entry_point": "index.html",
        "description": "A custom HTML application with full control"
    }
)
```

### Custom Application Types

With custom server-app-workers, you can support entirely new application types:

```python
# This would work with a custom worker that supports "python-eval" type
app_info = await controller.install(
    source='''
    import os
    import time
    
    print(f"Hello from Python eval at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workspace: {os.environ.get('HYPHA_WORKSPACE', 'unknown')}")
    print(f"App ID: {os.environ.get('HYPHA_APP_ID', 'unknown')}")
    
    # This is a simple Python script that runs and exits
    result = 42 * 2
    print(f"Calculation result: {result}")
    ''',
    manifest={
        "name": "Python Eval Example",
        "type": "python-eval",  # Custom type supported by python-eval worker
        "version": "1.0.0",
        "entry_point": "main.py"
    }
)
```

---

## Application Types and Alternatives

Depending on your use case, you might want to consider different approaches:

### For Web Applications with HTTP Endpoints
- **[ASGI Applications](asgi-apps.md)**: Full web frameworks like FastAPI, Django, Flask, and other ASGI web applications with full HTTP support, streaming responses, and utility functions
- **[Serverless Functions](serverless-functions.md)**: Create simple HTTP endpoint functions for lightweight web services and API endpoints
- **Server Apps**: For rich interactive applications with complex UI

### For Scalable Applications
- **[Autoscaling](autoscaling.md)**: Configure automatic scaling based on load for any application type, with detailed examples and best practices
- **Load Balancing**: Distribute requests across multiple instances
- **Service Selection**: Control how requests are routed to instances

### For Different Programming Languages
- **Web Python**: Python applications using Pyodide
- **Web Worker**: JavaScript applications in worker context
- **Custom Workers**: Support for any language or runtime

---

## Custom Server-App-Workers

You can extend the Server Apps service with custom workers to support new application types. Custom workers handle the execution of specific application types and integrate seamlessly with the Server Apps system.

> **Detailed Worker Documentation:**
> For comprehensive documentation on the worker API, interface specifications, and implementation examples, see the [Worker API Documentation](workers.md).

### Creating a Custom Worker

Creating a custom worker is simple - you just need to implement a few functions and register them as a service. No classes or complex inheritance needed!

**Simple Function-Based Worker:**

```python
# custom_worker.py
import asyncio
from hypha_rpc import connect_to_server

# Simple storage for demo (use proper storage in production)
worker_sessions = {}

async def start(config):
    """Start a new worker session."""
    session_id = f"{config['workspace']}/{config['client_id']}"
    
    print(f"Starting custom session {session_id} for app {config['app_id']}")
    
    # Your custom application startup logic here
    # This is where you would:
    # 1. Fetch the application source code
    # 2. Set up the execution environment  
    # 3. Start the application
    
    # Example implementation:
    app_source = await fetch_app_source(config)
    execution_env = await create_environment(config)
    result = await execute_app(app_source, execution_env)
    
    # Store session data
    session_data = {
        "session_id": session_id,
        "app_id": config["app_id"],
        "workspace": config["workspace"],
        "client_id": config["client_id"],
        "status": "running",
        "app_type": config.get("app_type", "unknown"),
        "created_at": "2024-01-01T00:00:00Z",
        "logs": [f"Custom session {session_id} started"],
        "execution_env": execution_env,
        "result": result
    }
    
    worker_sessions[session_id] = session_data
    return session_data

async def stop(session_id):
    """Stop a custom worker session."""
    print(f"Stopping custom session {session_id}")
    
    if session_id in worker_sessions:
        session_data = worker_sessions[session_id]
        execution_env = session_data.get("execution_env")
        
        # Clean up the execution environment
        if execution_env:
            await cleanup_environment(execution_env)
        
        session_data["status"] = "stopped"
        session_data["logs"].append(f"Session {session_id} stopped")
        print(f"Custom session {session_id} stopped")
    else:
        print(f"Session {session_id} not found")

async def list_sessions(workspace):
    """List all sessions for a workspace."""
    workspace_sessions = []
    for session_id, session_data in worker_sessions.items():
        if session_data["workspace"] == workspace:
            workspace_sessions.append(session_data)
    return workspace_sessions

async def get_logs(session_id, log_type=None, offset=0, limit=None):
    """Get logs for a session."""
    if session_id not in worker_sessions:
        raise Exception(f"Session {session_id} not found")
    
    logs = worker_sessions[session_id]["logs"]
    
    # Apply offset and limit
    if limit:
        logs = logs[offset:offset+limit]
    else:
        logs = logs[offset:]
    
    if log_type:
        return logs
    else:
        return {"log": logs, "error": []}

async def get_session_info(session_id):
    """Get information about a session."""
    if session_id not in worker_sessions:
        raise Exception(f"Session {session_id} not found")
    return worker_sessions[session_id]

async def prepare_workspace(workspace):
    """Prepare workspace for worker operations."""
    print(f"Preparing workspace {workspace} for custom worker")

async def close_workspace(workspace):
    """Close workspace and cleanup sessions."""
    print(f"Closing workspace {workspace} for custom worker")
    
    # Stop all sessions for this workspace
    sessions_to_stop = [
        session_id for session_id, session_data in worker_sessions.items()
        if session_data["workspace"] == workspace
    ]
    
    for session_id in sessions_to_stop:
        await stop(session_id)

# Helper functions for your custom logic
async def fetch_app_source(config):
    """Fetch application source code."""
    # Implementation depends on your storage backend
    # This is a placeholder
    return "# Custom app source code"

async def create_environment(config):
    """Create an execution environment."""
    return {
        "workspace": config["workspace"], 
        "app_id": config["app_id"],
        "env_vars": {}
    }

async def execute_app(source, env):
    """Execute application source code."""
    # Your custom execution logic here
    return {"status": "executed", "result": "success"}

async def cleanup_environment(env):
    """Clean up an execution environment."""
    # Clean up any resources
    pass

# Register the worker as a startup function
async def hypha_startup(server):
    """Initialize the custom worker as a startup function."""
    
    # Register the worker service
    service = await server.register_service({
        "name": "Custom Worker",
        "description": "A custom worker for specialized application types",
        "type": "server-app-worker",
        "config": {
            "visibility": "protected",
            "run_in_executor": True,
        },
        "supported_types": ["my-custom-type", "another-type"],
        "start": start,
        "stop": stop,
        "list_sessions": list_sessions,
        "get_logs": get_logs,
        "get_session_info": get_session_info,
        "prepare_workspace": prepare_workspace,
        "close_workspace": close_workspace,
    })
    
    print(f"Custom worker registered: {service.id}")
```



### Real Example: Python Eval Worker

The `python_eval.py` file provides a complete example of a custom worker that supports Python script execution:

```python
class PythonEvalRunner:
    """Python evaluation worker for simple Python code execution."""
    
    def __init__(self, server):
        self.server = server
        self._eval_sessions: Dict[str, Dict[str, Any]] = {}
        self.controller_id = str(PythonEvalRunner.instance_counter)
        PythonEvalRunner.instance_counter += 1
    
    async def start(self, client_id: str, app_id: str, **kwargs):
        """Start a Python eval session."""
        # Fetch Python code from entry point
        get_url = await self.artifact_manager.get_file(
            f"{workspace}/{app_id}", file_path=entry_point, version=version
        )
        
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            python_code = response.text

        # Set up execution environment
        logs = []
        execution_globals = {
            "__name__": "__main__",
            "print": lambda *args: logs.append(" ".join(str(arg) for arg in args)),
        }
        
        # Execute Python code
        try:
            exec(python_code, execution_globals)
            status = "completed"
        except Exception as e:
            status = "error"
            logs.append(f"Error: {traceback.format_exc()}")
        
        # Store session data
        self._eval_sessions[full_session_id] = {
            "session_id": full_session_id,
            "status": status,
            "logs": logs,
            "metadata": metadata,
        }
        
        return {"session_id": full_session_id, "status": status}

    def get_service(self):
        """Get the service definition."""
        return {
            "id": f"python-eval-worker-{self.controller_id}",
            "type": "server-app-worker",
            "name": "Python Eval Worker",
            "description": "A worker for running Python evaluation apps",
            "config": {"visibility": "protected"},
            "supported_types": ["python-eval"],  # Supports python-eval type
            "start": self.start,
            "stop": self.stop,
            "list": self.list,
            "get_logs": self.get_logs,
            "shutdown": self.shutdown,
            "close_workspace": self.close_workspace,
            "prepare_workspace": self.prepare_workspace,
        }

# Register as startup function
async def hypha_startup(server):
    """Initialize the Python eval worker as a startup function."""
    python_eval_runner = PythonEvalRunner(server)
    await python_eval_runner.initialize()
```

### Registering Custom Workers

To register your custom worker, you need to:

1. **Create the worker class** implementing the required interface
2. **Add a startup function** that registers the worker with the server
3. **Configure the server** to load your startup function

**Method 1: Startup Function (Recommended)**

Add your worker as a startup function in your Hypha server configuration:

```python
# In your server startup configuration
async def hypha_startup(server):
    """Initialize custom workers."""
    custom_worker = CustomWorker(server)
    await custom_worker.initialize()
```

**Method 2: Direct Registration**

Register the worker directly after server startup:

```python
# After server is running
server = await hypha_server.start()
custom_worker = CustomWorker(server)
await custom_worker.initialize()
```

### Starting the Server with Startup Functions

When you have created a `hypha_startup` function (either for custom workers or any other server initialization), you need to start the Hypha server with the `--startup-functions` option to load your custom code.

**Using a Python File:**

If your `hypha_startup` function is in a local Python file (e.g., `my_custom_worker.py`):

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --startup-functions=./my_custom_worker.py:hypha_startup
```

**Using an Installed Python Module:**

If your `hypha_startup` function is in an installed Python package:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --startup-functions=my_package.custom_worker:hypha_startup
```

**Multiple Startup Functions:**

You can specify multiple startup functions by providing additional `--startup-functions` arguments:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --startup-functions=./my_custom_worker.py:hypha_startup \
    --startup-functions=./another_service.py:hypha_startup
```

**Example with Server Apps and S3:**

For a complete server setup with Server Apps, built-in S3 storage, and custom workers:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --start-minio-server \
    --enable-server-apps \
    --startup-functions=./my_custom_worker.py:hypha_startup
```

> **Note:** The `--startup-functions` option expects the format `<python_module_or_file>:<function_name>`, where the function should be named `hypha_startup` and accept a single `server` parameter.

For more details on server startup options, see the [Getting Started documentation](getting-started.md#custom-initialization-and-service-integration-with-hypha-server).

### Worker Interface Requirements

All custom workers must implement these simple functions:

- `start(config)`: Start an application session
- `stop(session_id)`: Stop a session  
- `list_sessions(workspace)`: List sessions in a workspace
- `get_logs(session_id, ...)`: Get session logs
- `get_session_info(session_id)`: Get session information
- `close_workspace(workspace)`: Clean up workspace sessions
- `prepare_workspace(workspace)`: Prepare workspace for use

That's it! Just implement these functions and register them as a service with:
- `supported_types`: List of application types your worker supports
- `type`: Must be `"server-app-worker"`
- `visibility`: Usually `"protected"`

### Using Custom Application Types

Once a custom worker is registered, you can install applications of its supported types:

```python
# Install app with custom type
app_info = await controller.install(
    source=your_custom_source,
    manifest={
        "name": "My Custom App",
        "type": "my-custom-type",  # Must match worker's supported_types
        "version": "1.0.0",
        "entry_point": "main.ext"
    }
)

# Start the app - will automatically use your custom worker
session = await controller.start(app_info.id)
```

---

## Getting Started

### Step 1: Connecting to the Server Apps Service

To use the Server Apps service, start by connecting to the Hypha server and accessing the service.

```python
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"  # Replace with your server URL

# Connect to the server
api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})

# Get the server apps controller
controller = await api.get_service("public/server-apps")
```

### Step 2: Installing Your First App

Let's install a simple web worker application that provides basic functionality.

```python
# Define a simple web worker app
app_source = '''
api.export({
    async setup(){
        console.log("App initialized");
    },
    async add(a, b){
        return a + b;
    },
    async echo(message){
        return message;
    }
})
'''

# Install the app
app_info = await controller.install(
    source=app_source,
    config={"type": "web-worker"},
    overwrite=True,
)

print(f"App installed with ID: {app_info.id}")
```

### Step 3: Starting and Using the App

After installation, you can start the app and use its services.

```python
# Start the app
started_app = await controller.start(
    app_info.id,
    wait_for_service="default",  # Wait for the default service to be available
    timeout=30
)

print(f"App started with session ID: {started_app.id}")

# Get the app service and use it
app_service = await api.get_service(f"default@{app_info.id}")
result = await app_service.add(5, 3)
print(f"5 + 3 = {result}")

echo_result = await app_service.echo("Hello, World!")
print(f"Echo result: {echo_result}")
```

### Step 4: Managing Apps

You can list, stop, and uninstall apps as needed.

```python
# List all installed apps
apps = await controller.list_apps()
print(f"Total apps: {len(apps)}")

# List running apps
running_apps = await controller.list_running()
print(f"Running apps: {len(running_apps)}")

# Stop the app
await controller.stop(started_app.id)

# Uninstall the app
await controller.uninstall(app_info.id)
```

---

## Related Documentation

For more information on related Hypha features, see:

- **[ASGI Applications](asgi-apps.md)**: Deploy FastAPI, Django, Flask, and other ASGI web applications with full HTTP support, streaming responses, and utility functions
- **[Serverless Functions](serverless-functions.md)**: Create simple HTTP endpoint functions for lightweight web services and API endpoints
- **[Autoscaling](autoscaling.md)**: Configure automatic scaling based on load for any application type, with detailed examples and best practices

The Server Apps service provides a powerful and flexible platform for running various types of applications in the Hypha ecosystem. With support for both built-in templates and custom workers, it enables developers to build and deploy sophisticated applications with ease while maintaining extensibility for specialized use cases.