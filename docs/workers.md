# Worker API Documentation

## Overview

Hypha workers are responsible for executing different types of applications within isolated environments. Creating a worker is simple - you just need to implement a few functions and register them as a service.

## Simple Worker Functions

A worker needs to implement these basic functions:

- `start(config)` - Start a new session and return the session_id
- `stop(session_id)` - Stop a session  
- `list_sessions(workspace)` - List sessions for a workspace
- `get_logs(session_id)` - Get logs for a session
- `get_session_info(session_id)` - Get detailed session information

That's it! No classes or complex inheritance needed.

## Worker Configuration

When the `start(config)` function is called, the `config` parameter contains:

```python
{
    "id": "workspace/client_id",           # Full session ID
    "app_id": "my-app",                    # Application ID
    "workspace": "my-workspace",           # Workspace name
    "client_id": "unique-client-id",       # Client ID
    "server_url": "http://localhost:9527", # Server URL
    "token": "auth-token",                 # Authentication token
    "entry_point": "index.html",           # Entry point file
    "artifact_id": "workspace/app_id",     # Artifact ID for file access
    "manifest": {                          # Full application manifest
        "type": "python-eval",             # Application type
        "name": "My App",                  # Application name
        "version": "1.0.0",                # Application version
        "entry_point": "index.html",       # Entry point file
        "description": "My application",   # Application description
        # ... other manifest fields
    }
}
```

## Accessing Application Files

Workers can access application files using the `artifact_id` and connecting to the artifact manager:

```python
async def start(config):
    # Connect to the artifact manager
    server = await connect_to_server({"server_url": config["server_url"]})
    artifact_manager = await server.get_service("public/artifact-manager")
    
    # Read a file from the application
    get_url = await artifact_manager.get_file(
        config["artifact_id"], 
        file_path="my-script.py"
    )
    
    # Download the file content
    async with httpx.AsyncClient() as client:
        response = await client.get(get_url)
        if response.status_code == 200:
            file_content = response.text
            # Use the file content...
        else:
            raise Exception(f"Failed to download file: {response.status_code}")
```

## Reading App Type and Settings

Workers should read the application type and settings from the manifest:

```python
async def start(config):
    # Get app type from manifest
    app_type = config["manifest"]["type"]
    
    # Validate the app type
    if app_type not in ["python-eval", "web-python", "window"]:
        raise Exception(f"Unsupported app type: {app_type}")
    
    # Get app-specific settings from manifest
    app_name = config["manifest"]["name"]
    app_version = config["manifest"]["version"]
    
    # For MCP servers, get MCP configuration
    if app_type == "mcp-server":
        mcp_servers = config["manifest"].get("mcpServers", {})
        # Process MCP servers...
    
    # For A2A agents, get A2A configuration  
    if app_type == "a2a-agent":
        a2a_agents = config["manifest"].get("a2aAgents", {})
        # Process A2A agents...
```

## Python Worker Example

Here's a complete Python worker using only functions and dictionaries:

```python
"""Simple Python worker example."""
import asyncio
import httpx
from hypha_rpc import connect_to_server

# Simple storage for demo (use proper storage in production)
sessions = {}

async def start(config):
    """Start a new worker session."""
    session_id = config["id"]  # Already includes workspace/client_id
    
    print(f"Starting session {session_id} for app {config['app_id']}")
    
    # Get app type from manifest
    app_type = config["manifest"]["type"]
    
    # Connect to artifact manager to access files
    server = await connect_to_server({"server_url": config["server_url"]})
    artifact_manager = await server.get_service("public/artifact-manager")
    
    # Example: Read a configuration file from the application
    try:
        get_url = await artifact_manager.get_file(
            config["artifact_id"], 
            file_path="config.json"
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(get_url)
            if response.status_code == 200:
                app_config = response.json()
                print(f"Loaded app config: {app_config}")
            else:
                print(f"No config file found, using defaults")
                app_config = {}
    except Exception as e:
        print(f"Error loading config: {e}")
        app_config = {}
    
    # Your custom session startup logic here
    session_data = {
        "session_id": session_id,
        "app_id": config["app_id"],
        "workspace": config["workspace"],
        "client_id": config["client_id"],
        "status": "running",
        "app_type": app_type,
        "created_at": "2024-01-01T00:00:00Z",
        "logs": [f"Session {session_id} started"],
        "manifest": config["manifest"],
        "app_config": app_config
    }
    
    # Store session data
    sessions[session_id] = session_data
    
    # Return just the session_id - other data can be retrieved via get_session_info
    return session_id

async def stop(session_id):
    """Stop a worker session."""
    print(f"Stopping session {session_id}")
    
    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"
        sessions[session_id]["logs"].append(f"Session {session_id} stopped")
        print(f"Session {session_id} stopped")
    else:
        print(f"Session {session_id} not found")

async def list_sessions(workspace):
    """List all sessions for a workspace."""
    workspace_sessions = []
    for session_id, session_data in sessions.items():
        if session_data["workspace"] == workspace:
            workspace_sessions.append(session_data)
    return workspace_sessions

async def get_logs(session_id, log_type=None, offset=0, limit=None):
    """Get logs for a session."""
    if session_id not in sessions:
        raise Exception(f"Session {session_id} not found")
    
    logs = sessions[session_id]["logs"]
    
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
    if session_id not in sessions:
        raise Exception(f"Session {session_id} not found")
    return sessions[session_id]

async def prepare_workspace(workspace):
    """Prepare workspace for worker operations."""
    print(f"Preparing workspace {workspace}")

async def close_workspace(workspace):
    """Close workspace and cleanup sessions."""
    print(f"Closing workspace {workspace}")
    
    # Stop all sessions for this workspace
    sessions_to_stop = [
        session_id for session_id, session_data in sessions.items()
        if session_data["workspace"] == workspace
    ]
    
    for session_id in sessions_to_stop:
        await stop(session_id)

# Register the worker
async def register_worker():
    """Register the worker as a service."""
    server = await connect_to_server({
        "server_url": "http://localhost:9527"
    })
    
    # Register the worker service
    service = await server.register_service({
        "name": "Simple Python Worker",
        "description": "A simple worker implemented with functions",
        "type": "server-app-worker",
        "config": {
            "visibility": "protected",
            "run_in_executor": True,
        },
        "supported_types": ["python-custom", "simple-task"],
        "start": start,
        "stop": stop,
        "list_sessions": list_sessions,
        "get_logs": get_logs,
        "get_session_info": get_session_info,
        "prepare_workspace": prepare_workspace,
        "close_workspace": close_workspace,
    })
    
    print(f"Python worker registered: {service.id}")
    return server

# Start the worker
if __name__ == "__main__":
    async def main():
        server = await register_worker()
        print("Python worker is running...")
        # Keep the connection alive
        await server.serve()
    
    asyncio.run(main())
```

## JavaScript Worker Example

Here's a complete JavaScript worker using only functions and objects:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.66/dist/hypha-rpc-websocket.min.js"></script>
```
    
```javascript
// simple-worker.js
const { connectToServer } = hyphaWebsocketClient;

// Simple storage for demo (use proper storage in production)
const sessions = {};

async function start(config) {
    const sessionId = config.id;  // Already includes workspace/client_id
    
    console.log(`Starting session ${sessionId} for app ${config.app_id}`);
    
    // Get app type from manifest
    const appType = config.manifest.type;
    
    // Connect to artifact manager to access files
    const server = await connectToServer({server_url: config.server_url});
    const artifactManager = await server.getService("public/artifact-manager");
    
    // Example: Read a configuration file from the application
    let appConfig = {};
    try {
        const getUrl = await artifactManager.getFile(
            config.artifact_id, 
            {file_path: "config.json"}
        );
        const response = await fetch(getUrl);
        if (response.ok) {
            appConfig = await response.json();
            console.log(`Loaded app config:`, appConfig);
        } else {
            console.log(`No config file found, using defaults`);
        }
    } catch (error) {
        console.log(`Error loading config:`, error);
    }
    
    // Your custom session startup logic here
    const sessionData = {
        session_id: sessionId,
        app_id: config.app_id,
        workspace: config.workspace,
        client_id: config.client_id,
        status: "running",
        app_type: appType,
        created_at: new Date().toISOString(),
        logs: [`Session ${sessionId} started`],
        manifest: config.manifest,
        app_config: appConfig
    };
    
    // Store session data
    sessions[sessionId] = sessionData;
    
    // Return just the session_id - other data can be retrieved via get_session_info
    return sessionId;
}

async function stop(sessionId) {
    console.log(`Stopping session ${sessionId}`);
    
    if (sessions[sessionId]) {
        sessions[sessionId].status = "stopped";
        sessions[sessionId].logs.push(`Session ${sessionId} stopped`);
        console.log(`Session ${sessionId} stopped`);
    } else {
        console.log(`Session ${sessionId} not found`);
    }
}

async function listSessions(workspace) {
    const workspaceSessions = [];
    for (const [sessionId, sessionData] of Object.entries(sessions)) {
        if (sessionData.workspace === workspace) {
            workspaceSessions.push(sessionData);
        }
    }
    return workspaceSessions;
}

async function getLogs(sessionId, logType = null, offset = 0, limit = null) {
    if (!sessions[sessionId]) {
        throw new Error(`Session ${sessionId} not found`);
    }
    
    let logs = sessions[sessionId].logs;
    
    // Apply offset and limit
    if (limit) {
        logs = logs.slice(offset, offset + limit);
    } else {
        logs = logs.slice(offset);
    }
    
    if (logType) {
        return logs;
    } else {
        return { log: logs, error: [] };
    }
}

async function getSessionInfo(sessionId) {
    if (!sessions[sessionId]) {
        throw new Error(`Session ${sessionId} not found`);
    }
    return sessions[sessionId];
}

async function prepareWorkspace(workspace) {
    console.log(`Preparing workspace ${workspace}`);
}

async function closeWorkspace(workspace) {
    console.log(`Closing workspace ${workspace}`);
    
    // Stop all sessions for this workspace
    const sessionsToStop = Object.keys(sessions)
        .filter(sessionId => sessions[sessionId].workspace === workspace);
    
    for (const sessionId of sessionsToStop) {
        await stop(sessionId);
    }
}

// Register the worker
async function registerWorker() {
    try {
        const server = await connectToServer({
            server_url: "http://localhost:9527"
        });
        
        // Register the worker service
        const service = await server.registerService({
            name: "Simple JavaScript Worker",
            description: "A simple worker implemented with functions",
            type: "server-app-worker",
            config: {
                visibility: "protected",
                run_in_executor: true,
            },
            supported_types: ["javascript-custom", "simple-task"],
            start: start,
            stop: stop,
            list_sessions: listSessions,
            get_logs: getLogs,
            get_session_info: getSessionInfo,
            prepare_workspace: prepareWorkspace,
            close_workspace: closeWorkspace,
        });
        
        console.log(`JavaScript worker registered: ${service.id}`);
        return server;
        
    } catch (error) {
        console.error("Failed to register worker:", error);
        throw error;
    }
}

// Start the worker
registerWorker().then(server => {
    console.log("JavaScript worker is running...");
    // Keep the process running
}).catch(error => {
    console.error("Worker startup failed:", error);
    process.exit(1);
});
```

## Running the Workers

### Python Worker

1. Save the code as `simple-worker.py`
2. Install dependencies: `pip install hypha-rpc`
3. Run: `python simple-worker.py`

### JavaScript Worker

1. Save the code as `simple-worker.js`
2. Install dependencies: `npm install hypha-rpc`
3. Run: `node simple-worker.js`

## How It Works

1. **Connect to server**: Use `connect_to_server()` to connect to Hypha
2. **Define functions**: Implement the required worker functions
3. **Register as service**: Call `register_service()` with your functions
4. **That's it!** Your worker is now available to handle app requests

## Worker Workflow

When a worker's `start()` function is called:

1. **Parse configuration**: Extract app type and settings from the manifest
2. **Connect to artifact manager**: Access the application's files using the artifact_id
3. **Download required files**: Use the artifact manager to get configuration files, scripts, or other assets
4. **Initialize session**: Set up the application environment based on the manifest and downloaded files
5. **Start the app**: Launch the application process (browser, Python script, proxy, etc.)
6. **Return session data**: Provide session information back to the controller

This workflow ensures that workers have access to all application files and can configure themselves based on the application's manifest rather than hard-coded app types.

## Return Values and Session Management

The worker interface is designed to be simple and consistent:

- `start(config)` returns just the `session_id` (string)
- To get detailed session information, use `get_session_info(session_id)`
- This separation keeps the interface clean and allows for lazy loading of session data

Example:
```python
# Start a session
session_id = await start(config)

# Later, get detailed session information when needed
session_info = await get_session_info(session_id)
```

This design allows workers to:
1. Start sessions quickly with minimal data transfer
2. Retrieve detailed session information only when needed
3. Maintain a consistent interface across all worker types

## Legacy Configuration Reference

For reference, the old `config` parameter structure was:

```python
{
    "client_id": "unique-client-id",
    "app_id": "my-app", 
    "workspace": "my-workspace",
    "server_url": "http://localhost:9527",
    "public_base_url": "http://localhost:9527",
    "local_base_url": "http://localhost:9527",
    "app_type": "my-app-type",
    "entry_point": "index.html",
    "metadata": {"key": "value"},
    "token": "auth-token",
    "version": "1.0.0"
}
```

The new structure is simpler and more focused on the essential information workers need.

## Session Data

Return session information from `start()` like this:

```python
{
    "session_id": "workspace/client-id",
    "app_id": "my-app",
    "workspace": "my-workspace", 
    "client_id": "client-id",
    "status": "running",
    "app_type": "my-app-type",
    "created_at": "2024-01-01T00:00:00Z",
    "logs": ["Session started"],
    "metadata": {"key": "value"}
}
```

That's all you need to create a custom worker! No complex classes or inheritance required.
