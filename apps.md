# Server Apps

The **Server Apps** service is a core Hypha service that enables serverless computing and dynamic application management within the Hypha ecosystem. It provides a powerful platform for running various types of applications including web workers, web Python apps, ASGI web applications, and daemon services in a serverless fashion - meaning applications are only started on-demand and automatically scaled based on usage. The service supports features like lazy loading, automatic startup configuration, and HTTP access to applications, making it ideal for building modern serverless architectures.

## How It Works

The Server Apps service operates by running a full-featured web browser (Chromium) on the server side, enabling it to execute web applications and JavaScript code in a controlled server environment. This unique architecture provides several key advantages:

- **Full Browser Environment**: Applications have access to standard web APIs, DOM manipulation, and modern JavaScript features just like in a regular browser
- **Sandboxed Execution**: Each app runs in an isolated browser context for security and resource management
- **Server-Side Rendering**: Web applications can perform server-side rendering and processing without client-side overhead
- **WebWorker Support**: Dedicated web workers can run CPU-intensive tasks in parallel
- **Native Performance**: Direct access to server resources while maintaining browser compatibility

The service manages a pool of browser instances and automatically scales them based on demand. When a request comes in:

1. The service checks if a browser instance is available for the requested app
2. If needed, it spins up a new browser instance and loads the application
3. The request is routed to the appropriate browser context
4. Results are returned through the Hypha API layer

This architecture enables true serverless operation - applications only consume resources when actively being used, while providing the full capabilities of a modern web browser runtime.


**Key Features:**
- **Dynamic App Installation**: Install apps from source code, URLs, or local files
- **Multiple App Types**: Support for web workers, web Python, ASGI, daemon apps, and more
- **Lazy Loading**: Apps can be started automatically when accessed
- **Startup Configuration**: Configure default startup parameters for consistent behavior
- **HTTP Access**: Direct HTTP access to web applications and services
- **Lifecycle Management**: Complete control over app installation, starting, stopping, and uninstalling

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

## Full Example: Web Python App with Startup Configuration

Here's a complete example showing how to install and run a web Python app with startup configuration.

```python
import asyncio
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the server
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    controller = await api.get_service("public/server-apps")
    
    # Define a web Python app
    python_app_source = '''
<config lang="yaml">
name: Python Calculator
type: web-python
version: 0.1.0
</config>

<script lang="python">
from hypha_rpc import api

class PythonCalculator:
    def __init__(self):
        self.history = []
    
    async def calculate(self, expression):
        try:
            result = eval(expression)
            self.history.append(f"{expression} = {result}")
            return result
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def get_history(self):
        return self.history
    
    async def clear_history(self):
        self.history = []
        return "History cleared"

# Export the service
calculator = PythonCalculator()
api.export({
    "calculate": calculator.calculate,
    "get_history": calculator.get_history,
    "clear_history": calculator.clear_history,
})
</script>
'''
    
    # Install app with startup configuration
    app_info = await controller.install(
        source=python_app_source,
        config={
            "startup_config": {
                "timeout": 45,  # Startup timeout
                "wait_for_service": "default",  # Service to wait for
                "stop_after_inactive": 300,  # Auto-stop after 5 minutes of inactivity
            }
        },
        overwrite=True
    )
    
    print(f"Python app installed: {app_info.id}")
    
    # Start the app (uses startup_config defaults)
    started_app = await controller.start(app_info.id)
    print(f"App started: {started_app.id}")
    
    # Use the calculator
    calc_service = await api.get_service(f"default@{app_info.id}")
    
    result1 = await calc_service.calculate("2 + 3 * 4")
    print(f"Calculation result: {result1}")
    
    result2 = await calc_service.calculate("10 / 2")
    print(f"Calculation result: {result2}")
    
    history = await calc_service.get_history()
    print(f"Calculation history: {history}")
    
    # Clean up
    await controller.stop(started_app.id)
    await controller.uninstall(app_info.id)

asyncio.run(main())
```

---

## Advanced Features

### Lazy Loading

The Server Apps service supports lazy loading, where apps are automatically started when their services are accessed, even if they haven't been explicitly started.

```python
# Install an app without starting it
app_info = await controller.install(
    source=app_source,
    config={
        "type": "web-worker",
        "startup_config": {
            "timeout": 30,
            "wait_for_service": "echo",
            "stop_after_inactive": 120,  # Auto-stop after 2 minutes
        }
    },
    overwrite=True
)

# Access the service directly - this will trigger lazy loading
service = await api.get_service(f"echo@{app_info.id}")
result = await service.echo("Hello from lazy loaded app!")
print(f"Lazy loading result: {result}")
```

### Daemon Apps

Daemon apps are persistent applications that continue running even after the client disconnects.

```python
# Install a daemon app
daemon_app_source = '''
api.export({
    async setup(){
        console.log("Daemon app started");
        // Set up periodic tasks, monitoring, etc.
    },
    async get_status(){
        return {
            uptime: Date.now(),
            status: "running"
        };
    }
})
'''

app_info = await controller.install(
    source=daemon_app_source,
    config={
        "type": "web-worker",
        "daemon": True,  # Mark as daemon
        "singleton": True,  # Ensure only one instance
    },
    overwrite=True
)

# Start the daemon app
daemon_app = await controller.start(app_info.id)
print(f"Daemon app started: {daemon_app.id}")

# The daemon will continue running even after client disconnects
```

### ASGI Web Applications

You can run full web applications using ASGI frameworks like FastAPI.

```python
# Define a FastAPI app
fastapi_app_source = '''
<config lang="yaml">
name: FastAPI Hello World
type: web-python
version: 0.1.0
</config>

<script lang="python">
from hypha_rpc import api
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create FastAPI app
app = FastAPI(title="Hello World API")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Export as ASGI service
api.export({
    "type": "asgi",
    "app": app,
    "serve": lambda: app,  # Function to get the ASGI app
})
</script>
'''

# Install the ASGI app
app_info = await controller.install(
    source=fastapi_app_source,
    config={
        "startup_config": {
            "timeout": 60,
            "wait_for_service": "serve",
            "stop_after_inactive": 600,  # 10 minutes
        }
    },
    overwrite=True
)

print(f"FastAPI app installed: {app_info.id}")
print(f"Access it at: {SERVER_URL}/{api.config['workspace']}/apps/serve@{app_info.id}/")
```

### Startup Configuration

The `startup_config` parameter allows you to set default configurations that will be used when starting apps.

**Configuration Options:**
You can configure startup behavior in two ways:

1. **In the app's HTML `<config>` section:**
```html
<config lang="json">
{
    "name": "My App",
    "type": "web-python",
    "startup_config": {
        "timeout": 60,
        "wait_for_service": "default",
        "stop_after_inactive": 300
    }
}
</config>
```

2. **As part of the `config` parameter in `install()`:**

```python
# Install app with comprehensive startup configuration
app_info = await controller.install(
    source=app_source,
    config={
        "startup_config": {
            "timeout": 45,  # Maximum time to wait for app startup
            "wait_for_service": "default",  # Service to wait for during startup
            "stop_after_inactive": 300,  # Auto-stop after 5 minutes of inactivity
        }
    },
    overwrite=True
)

```

**Priority Order:**
When the same startup parameter is specified in multiple places, the priority order is:
1. **Explicit parameters in `start()` method** (highest priority)
2. **`startup_config` in `config` parameter of `install()` method** (medium priority)  
3. **`startup_config` in HTML `<config>` section** (lowest priority)

```python
# Start app - will use the startup_config defaults
started_app = await controller.start(app_info.id)

# You can still override defaults when starting
started_app = await controller.start(
    app_info.id,
    timeout=60,  # Override the default timeout
    stop_after_inactive=600  # Override the default inactivity timeout
)
```

---

## API Reference

### `install(source: str = None, source_hash: str = None, config: Dict[str, Any] = None, workspace: str = None, overwrite: bool = False, timeout: float = 60, version: str = None, context: dict = None) -> AppInfo`

Installs an application from source code, URL, or configuration.

**Parameters:**

- `source`: The source code of the application, URL to fetch the source, or None if using config
- `source_hash`: Optional hash of the source for verification
- `config`: Application configuration dictionary containing app metadata and settings. Can include `startup_config` with default startup parameters:
  - `startup_config.timeout`: Default timeout for starting the app
  - `startup_config.wait_for_service`: Default service to wait for during startup
  - `startup_config.stop_after_inactive`: Default inactivity timeout (seconds) for auto-stopping
- `workspace`: Target workspace for installation (defaults to current workspace)
- `overwrite`: Whether to overwrite existing app with same name
- `timeout`: Maximum time to wait for installation completion
- `version`: Version identifier for the app
- `context`: Additional context information

**Returns:** `AppInfo` object containing app details including `id`, `name`, `type`, etc.

**Example:**

```python
app_info = await controller.install(
    source=app_source,
    config={
        "type": "web-worker", 
        "name": "My App",
        "startup_config": {
            "timeout": 30,
            "wait_for_service": "default",
            "stop_after_inactive": 180
        }
    },
    overwrite=True
)
```

---

### `start(app_id: str, client_id: str = None, timeout: float = None, version: str = None, wait_for_service: Union[str, bool] = None, stop_after_inactive: float = None, context: dict = None) -> AppSessionInfo`

Starts an installed application.

**Parameters:**

- `app_id`: The ID of the app to start
- `client_id`: Optional client identifier for the app session
- `timeout`: Maximum time to wait for app startup (uses startup_config default if not specified)
- `version`: Version of the app to start
- `wait_for_service`: Service to wait for during startup (uses startup_config default if not specified)
- `stop_after_inactive`: Seconds of inactivity before auto-stopping (uses startup_config default if not specified)
- `context`: Additional context information

**Returns:** `AppSessionInfo` object with session details including `id`, `app_id`, `service_id`, etc.

**Example:**

```python
# Start with default startup_config
session = await controller.start(app_info.id)

# Start with custom parameters (overrides defaults)
session = await controller.start(
    app_info.id,
    timeout=45,
    wait_for_service="custom-service",
    stop_after_inactive=300
)
```

---

### `stop(session_id: str, context: dict = None) -> None`

Stops a running application session.

**Parameters:**

- `session_id`: The session ID of the running app to stop
- `context`: Additional context information

**Example:**

```python
await controller.stop(session.id)
```

---

### `list_apps(workspace: str = None, context: dict = None) -> List[AppInfo]`

Lists all installed applications.

**Parameters:**

- `workspace`: Workspace to list apps from (defaults to current workspace)
- `context`: Additional context information

**Returns:** List of `AppInfo` objects

**Example:**

```python
apps = await controller.list_apps()
for app in apps:
    print(f"App: {app.name} (ID: {app.id})")
```

---

### `list_running(workspace: str = None, context: dict = None) -> List[AppSessionInfo]`

Lists all currently running application sessions.

**Parameters:**

- `workspace`: Workspace to list running apps from (defaults to current workspace)
- `context`: Additional context information

**Returns:** List of `AppSessionInfo` objects

**Example:**

```python
running_apps = await controller.list_running()
for session in running_apps:
    print(f"Running: {session.app_id} (Session: {session.id})")
```

---

### `uninstall(app_id: str, context: dict = None) -> None`

Uninstalls an application and all its resources.

**Parameters:**

- `app_id`: The ID of the app to uninstall
- `context`: Additional context information

**Example:**

```python
await controller.uninstall(app_info.id)
```

---

### `get_log(session_id: str, type: str = None, offset: int = 0, limit: int = 100, context: dict = None) -> Union[Dict[str, List], List]`

Retrieves logs from a running application session.

**Parameters:**

- `session_id`: The session ID to get logs from
- `type`: Log type filter ("log", "error", "warn", etc.)
- `offset`: Number of log entries to skip
- `limit`: Maximum number of log entries to return
- `context`: Additional context information

**Returns:** Log entries as a list or dictionary grouped by type

**Example:**

```python
# Get all logs
logs = await controller.get_log(session.id)

# Get only error logs
errors = await controller.get_log(session.id, type="error")

# Get recent logs with pagination
recent_logs = await controller.get_log(session.id, offset=0, limit=50)
```

---

### `launch(source: str = None, config: Dict[str, Any] = None, wait_for_service: Union[str, bool] = None, timeout: float = 60, overwrite: bool = False, context: dict = None) -> AppSessionInfo`

Installs and immediately starts an application in one operation.

**Parameters:**

- `source`: The source code of the application or URL
- `config`: Application configuration dictionary
- `wait_for_service`: Service to wait for during startup
- `timeout`: Maximum time to wait for launch completion
- `overwrite`: Whether to overwrite existing app
- `context`: Additional context information

**Returns:** `AppSessionInfo` object for the running app session

**Example:**

```python
# Launch an app directly
session = await controller.launch(
    source=app_source,
    config={"type": "web-worker"},
    wait_for_service="default",
    timeout=30
)

# Use the app immediately
service = await api.get_service(session.service_id)
result = await service.echo("Hello!")
```

---

## HTTP API for Accessing Apps

The Server Apps service provides HTTP endpoints for accessing web applications directly through the browser.

### App Access Endpoints

#### Direct App Access

- **`/{workspace}/apps/{service_name}@{app_id}/`**: Access the main endpoint of a web application
- **`/{workspace}/apps/{service_name}@{app_id}/{path:path}`**: Access specific paths within a web application

#### Service Function Access

- **`/{workspace}/services/{service_name}@{app_id}/{function_name}`**: Call specific service functions via HTTP

### Lazy Loading via HTTP

Apps can be started automatically when accessed via HTTP endpoints, even if they haven't been explicitly started.

```python
import requests

# Install an app without starting it
app_info = await controller.install(
    source=fastapi_app_source,
    startup_config={
        "timeout": 30,
        "wait_for_service": "serve",
        "stop_after_inactive": 300
    },
    overwrite=True
)

# Access the app via HTTP - this will trigger lazy loading
workspace = api.config["workspace"]
app_url = f"{SERVER_URL}/{workspace}/apps/serve@{app_info.id}/"

# The app will be started automatically on first access
response = requests.get(app_url)
print(f"App response: {response.json()}")
```

### Authentication

For private workspaces, include authentication headers:

```python
# Generate an access token
token = await api.generate_token()

# Access private app with authentication
response = requests.get(
    app_url,
    headers={"Authorization": f"Bearer {token}"}
)
```

---

## Application Types

### Web Worker Apps

Web Worker apps run JavaScript in a web worker environment and are suitable for CPU-intensive tasks.

```python
web_worker_source = '''
// Web Worker App
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
    config={"type": "web-worker"}
)
```

### Web Python Apps

Web Python apps run Python code in a web environment using Pyodide.

```python
web_python_source = '''
<config lang="yaml">
name: Python Data Processor
type: web-python
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

### ASGI Web Apps

ASGI apps are full web applications that can handle HTTP requests and WebSocket connections.

```python
asgi_app_source = '''
<config lang="yaml">
name: FastAPI Service
type: web-python
</config>

<script lang="python">
from hypha_rpc import api
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Data Service")

class DataItem(BaseModel):
    name: str
    value: float

data_store = {}

@app.get("/")
async def root():
    return {"message": "Data Service API"}

@app.post("/data/{item_id}")
async def create_data(item_id: str, item: DataItem):
    data_store[item_id] = item.dict()
    return {"message": f"Data {item_id} created"}

@app.get("/data/{item_id}")
async def get_data(item_id: str):
    if item_id not in data_store:
        raise HTTPException(status_code=404, detail="Item not found")
    return data_store[item_id]

# Export as ASGI service
api.export({
    "type": "asgi",
    "serve": lambda: app
})
</script>
'''

app_info = await controller.install(
    source=asgi_app_source,
    startup_config={
        "wait_for_service": "serve",
        "stop_after_inactive": 600
    }
)
```

---

## Best Practices

### 1. Use Startup Configuration

Always define startup configuration for consistent app behavior:

```python
app_info = await controller.install(
    source=app_source,
    config={
        "startup_config": {
            "timeout": 45,  # Allow sufficient time for startup
            "wait_for_service": "default",  # Wait for main service
            "stop_after_inactive": 300  # Auto-cleanup after 5 minutes
        }
    }
)
```

### 2. Handle App Lifecycle

Properly manage app lifecycle with try-finally blocks:

```python
app_info = await controller.install(source=app_source)
session = None
try:
    session = await controller.start(app_info.id)
    # Use the app
    service = await api.get_service(session.service_id)
    result = await service.process_data(data)
finally:
    if session:
        await controller.stop(session.id)
    await controller.uninstall(app_info.id)
```

### 3. Use Appropriate App Types

Choose the right app type for your use case:

- **Web Worker**: For CPU-intensive JavaScript tasks
- **Web Python**: For Python data processing and analysis
- **ASGI**: For full web applications with HTTP/WebSocket support
- **Daemon**: For persistent background services

### 4. Implement Error Handling

Always implement proper error handling:

```python
try:
    app_info = await controller.install(source=app_source)
    session = await controller.start(app_info.id, timeout=30)
except Exception as e:
    print(f"Failed to start app: {e}")
    # Handle the error appropriately
```

### 5. Use Lazy Loading for On-Demand Apps

For apps that are accessed infrequently, use lazy loading:

```python
# Install without starting
app_info = await controller.install(
    source=app_source,
    config={
        "startup_config": {
            "stop_after_inactive": 60,  # Quick cleanup for on-demand apps
            "timeout": 30
        }
    }
)

# App will start automatically when accessed
service = await api.get_service(f"default@{app_info.id}")
```

---

## Examples

### Example 1: Machine Learning Model Server

```python
ml_model_source = '''
<config lang="yaml">
name: ML Model Server
type: web-python
</config>

<script lang="python">
import numpy as np
from hypha_rpc import api
from sklearn.linear_model import LinearRegression
import joblib

class MLModelServer:
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    async def train(self, X, y):
        """Train a simple linear regression model"""
        self.model = LinearRegression()
        self.model.fit(np.array(X), np.array(y))
        self.is_trained = True
        return {"status": "trained", "coefficients": self.model.coef_.tolist()}
    
    async def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        predictions = self.model.predict(np.array(X))
        return predictions.tolist()
    
    async def get_model_info(self):
        """Get model information"""
        if not self.is_trained:
            return {"status": "not_trained"}
        return {
            "status": "trained",
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_)
        }

ml_server = MLModelServer()
api.export({
    "train": ml_server.train,
    "predict": ml_server.predict,
    "get_model_info": ml_server.get_model_info
})
</script>
'''

# Install and start the ML model server
app_info = await controller.install(
    source=ml_model_source,
    config={
        "startup_config": {
            "timeout": 60,  # ML models may take longer to start
            "wait_for_service": "default",
            "stop_after_inactive": 1800  # 30 minutes for ML workloads
        }
    }
)

session = await controller.start(app_info.id)
ml_service = await api.get_service(session.service_id)

# Train the model
training_data = [[1], [2], [3], [4], [5]]
labels = [2, 4, 6, 8, 10]
result = await ml_service.train(training_data, labels)
print(f"Training result: {result}")

# Make predictions
predictions = await ml_service.predict([[6], [7], [8]])
print(f"Predictions: {predictions}")
```

### Example 2: Real-time Data Dashboard

```python
dashboard_source = '''
<config lang="yaml">
name: Real-time Dashboard
type: web-python
</config>

<script lang="python">
from hypha_rpc import api
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import asyncio
import random
from datetime import datetime

app = FastAPI(title="Real-time Dashboard")

# Store active WebSocket connections
active_connections = []

@app.get("/")
async def dashboard():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Real-time Data Dashboard</h1>
        <canvas id="chart" width="400" height="200"></canvas>
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            const ctx = document.getElementById('chart').getContext('2d');
            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Real-time Data',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                chart.data.labels.push(data.time);
                chart.data.datasets[0].data.push(data.value);
                
                // Keep only last 20 data points
                if (chart.data.labels.length > 20) {
                    chart.data.labels.shift();
                    chart.data.datasets[0].data.shift();
                }
                
                chart.update();
            };
        </script>
    </body>
    </html>
    """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def generate_data():
    """Generate random data and send to all connected clients"""
    while True:
        if active_connections:
            data = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "value": random.uniform(0, 100)
            }
            # Send to all connected clients
            for connection in active_connections.copy():
                try:
                    await connection.send_text(json.dumps(data))
                except:
                    active_connections.remove(connection)
        await asyncio.sleep(1)

# Start data generation task
asyncio.create_task(generate_data())

# Export as ASGI service
api.export({
    "type": "asgi",
    "serve": lambda: app
})
</script>
'''

app_info = await controller.install(
    source=dashboard_source,
    config={
        "startup_config": {
            "wait_for_service": "serve",
            "stop_after_inactive": 3600  # 1 hour for dashboard
        }
    }
)

print(f"Dashboard available at: {SERVER_URL}/{api.config['workspace']}/apps/serve@{app_info.id}/")
```

### Example 3: File Processing Service

```python
file_processor_source = '''
<config lang="yaml">
name: File Processor
type: web-python
</config>

<script lang="python">
from hypha_rpc import api
import csv
import json
import io
import base64

class FileProcessor:
    def __init__(self):
        self.processed_files = {}
    
    async def process_csv(self, file_content, filename):
        """Process CSV file content"""
        try:
            # Decode base64 content
            content = base64.b64decode(file_content).decode('utf-8')
            
            # Parse CSV
            reader = csv.DictReader(io.StringIO(content))
            data = list(reader)
            
            # Store result
            self.processed_files[filename] = {
                "type": "csv",
                "rows": len(data),
                "columns": list(data[0].keys()) if data else [],
                "data": data[:10]  # Store first 10 rows
            }
            
            return {
                "status": "success",
                "rows": len(data),
                "columns": list(data[0].keys()) if data else []
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def process_json(self, file_content, filename):
        """Process JSON file content"""
        try:
            # Decode base64 content
            content = base64.b64decode(file_content).decode('utf-8')
            
            # Parse JSON
            data = json.loads(content)
            
            # Store result
            self.processed_files[filename] = {
                "type": "json",
                "data": data
            }
            
            return {"status": "success", "type": type(data).__name__}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_processed_files(self):
        """Get list of processed files"""
        return list(self.processed_files.keys())
    
    async def get_file_info(self, filename):
        """Get information about a processed file"""
        if filename not in self.processed_files:
            return {"status": "error", "message": "File not found"}
        return self.processed_files[filename]

processor = FileProcessor()
api.export({
    "process_csv": processor.process_csv,
    "process_json": processor.process_json,
    "get_processed_files": processor.get_processed_files,
    "get_file_info": processor.get_file_info
})
</script>
'''

app_info = await controller.install(
    source=file_processor_source,
    config={
        "startup_config": {
            "timeout": 30,
            "wait_for_service": "default",
            "stop_after_inactive": 600
        }
    }
)

session = await controller.start(app_info.id)
processor_service = await api.get_service(session.service_id)

# Example usage
csv_content = base64.b64encode(b"name,age,city\nJohn,30,NYC\nJane,25,LA").decode()
result = await processor_service.process_csv(csv_content, "example.csv")
print(f"CSV processing result: {result}")

files = await processor_service.get_processed_files()
print(f"Processed files: {files}")
```

The Server Apps service provides a powerful and flexible platform for running various types of applications in the Hypha ecosystem. With features like lazy loading, startup configuration, and comprehensive lifecycle management, it enables developers to build and deploy sophisticated applications with ease. 