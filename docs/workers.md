# Hypha Worker API Documentation

## Overview

Hypha workers are responsible for executing different types of applications within isolated environments. The worker interface has been simplified to focus on essential functions, making it easy to create custom workers in both Python and JavaScript.

### Built-in Workers

Hypha includes several built-in workers for common use cases:

- **[Browser Worker](browser-worker.md)**: Runs web applications in isolated browser environments via Playwright, supporting JavaScript, Python (web-python), web workers, and external web apps
- **[Conda Environment Worker](conda-worker.md)**: Executes Python code in isolated conda environments with automatic mamba/conda detection for faster dependency management

For detailed information about each worker, including installation, configuration, and usage examples, see:
- [Browser Worker Documentation](browser-worker.md)
- [Conda Worker Documentation](conda-worker.md)

## Hypha Worker Interface

All workers must implement these core methods:

- `start(config, context=None)` - Start a new session and return session_id
- `stop(session_id, context=None)` - Stop a session
- `list_sessions(workspace, context=None)` - List sessions for a workspace
- `get_session_info(session_id, context=None)` - Get detailed session information
- `get_logs(session_id, type, offset, limit, context=None)` - Get logs for a session
- `prepare_workspace(workspace, context=None)` - Prepare workspace for operations
- `close_workspace(workspace, context=None)` - Close workspace and cleanup sessions

Optional methods:
- `compile(manifest, files, config, context=None)` - Compile application files (for build-time processing)

## Worker Properties

All workers must define these properties:

- `supported_types` - List of supported application types
- `name` - Worker display name
- `description` - Worker description
- `visibility` - Worker visibility ("public", "protected", or "private", defaults to "protected")
- `run_in_executor` - Whether worker should run in executor (defaults to False)
- `require_context` - Whether worker requires context parameter (defaults to True)
- `service_id` - Unique service identifier (auto-generated from name and instance)

### Context Parameter

Workers with `require_context = True` (default) receive a `context` parameter in all their methods containing workspace and user information. This allows the server to inject security context and workspace details into worker operations.

If `require_context = False`, the context parameter will still be present but may be None depending on the calling context.

## Hypha Worker Configuration

When `start(config, context)` is called, the config parameter contains:

```python
{
    "id": "workspace/client_id",           # Full session ID
    "app_id": "my-app",                    # Application ID
    "workspace": "my-workspace",           # Workspace name
    "client_id": "unique-client-id",       # Client ID
    "server_url": "http://localhost:9527", # Server URL
    "token": "auth-token",                 # Authentication token (optional)
    "entry_point": "index.html",           # Entry point file (optional)
    "artifact_id": "workspace/app_id",     # Artifact ID for file access
    "manifest": {                          # Full application manifest
        "type": "python-eval",             # Application type
        "name": "My App",                  # Application name
        "version": "1.0.0",                # Version
        "description": "My application",   # Description
        # ... other manifest fields
    },
    "timeout": 30,                         # Optional timeout in seconds
    "progress_callback": callable          # Optional progress callback
}
```

The `context` parameter contains workspace and user information when available:

```python
{
    "ws": "workspace-name",               # Workspace name
    "user": {                             # User information
        "id": "user-id",
        "email": "user@example.com",
        "roles": ["user"],
        # ... other user properties
    },
    "from": "client-id"                   # Optional client ID
}
```

## Progress Callbacks

The `progress_callback` parameter in the worker configuration allows workers to provide real-time feedback during long-running operations like environment setup, dependency installation, or application compilation. Implementing progress callbacks significantly improves user experience by keeping users informed about the current status.

### Default Progress Callback Implementation

Here's a recommended default progress callback implementation that workers can use:

```python
def progress_callback(info: Dict[str, Any]):
    """Default progress callback with emoji indicators."""
    emoji = {
        "info": "ℹ️",
        "success": "✅", 
        "error": "❌",
        "warning": "⚠️",
    }.get(info.get("type", ""), "🔸")
    print(f"{emoji} {info.get('message', '')}")
```

### Using Progress Callbacks in Workers

Workers should call the progress callback during key phases of the `start()` method:

```python
async def start(self, config: WorkerConfig) -> str:
    """Start a worker session with progress reporting."""
    progress_callback = config.get("progress_callback")
    
    if progress_callback:
        progress_callback({
            "type": "info", 
            "message": "Starting worker session..."
        })
    
    try:
        # Phase 1: Environment setup
        if progress_callback:
            progress_callback({
                "type": "info",
                "message": "Setting up environment..."
            })
        
        # Your environment setup code here
        await setup_environment()
        
        if progress_callback:
            progress_callback({
                "type": "success",
                "message": "Environment setup completed"
            })
        
        # Phase 2: Installing dependencies
        if progress_callback:
            progress_callback({
                "type": "info", 
                "message": "Installing dependencies..."
            })
        
        # Your dependency installation code here
        await install_dependencies()
        
        if progress_callback:
            progress_callback({
                "type": "success",
                "message": "Dependencies installed successfully"
            })
        
        # Phase 3: Final initialization
        if progress_callback:
            progress_callback({
                "type": "info",
                "message": "Finalizing session initialization..."
            })
        
        # Your final initialization code here
        session_id = await finalize_session()
        
        if progress_callback:
            progress_callback({
                "type": "success",
                "message": f"Session {session_id} started successfully"
            })
        
        return session_id
        
    except Exception as e:
        if progress_callback:
            progress_callback({
                "type": "error",
                "message": f"Failed to start session: {str(e)}"
            })
        raise
```

### Progress Message Types

Use these standard message types for consistency:

- **`info`**: General information messages (🔸 or ℹ️)
- **`success`**: Successful completion of operations (✅)
- **`error`**: Error messages (❌) 
- **`warning`**: Warning messages (⚠️)

### JavaScript Example

```javascript
function progressCallback(info) {
    const emoji = {
        "info": "ℹ️",
        "success": "✅",
        "error": "❌", 
        "warning": "⚠️",
    }[info.type] || "🔸";
    
    console.log(`${emoji} ${info.message || ''}`);
}

 async function start(config) {
     const progressCallback = config.progress_callback;
     
     if (progressCallback) {
         progressCallback({
             type: "info",
             message: "Starting JavaScript worker session..."
         });
     }
     
     // Your implementation with progress reporting...
 }
 ```

### Recommendation

**It is highly recommended that all workers implement progress callbacks** to provide users with real-time feedback during session startup. This significantly improves the user experience, especially for workers that perform time-consuming operations like:

- Environment setup and dependency installation (conda environments, Docker containers)
- Large file downloads or processing
- Compilation or build processes
- Database connections and initialization

Workers that don't implement progress callbacks may appear unresponsive to users during startup, leading to confusion about whether the operation is still in progress or has failed.

## Hypha Worker Implementation in Python

### Method 1: Using BaseWorker Class (Recommended)

```python
"""Example worker using BaseWorker class."""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
from hypha_rpc import connect_to_server

from hypha.workers.base import (
    BaseWorker, WorkerConfig, SessionInfo, SessionStatus,
    SessionNotFoundError, WorkerError
)

class MyCustomWorker(BaseWorker):
    """Custom worker implementation."""
    
    def __init__(self):
        super().__init__()
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}
    
    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["my-custom-type", "another-type"]
    
    @property
    def name(self) -> str:
        """Return the worker name."""
        return "My Custom Worker"
    
    @property
    def description(self) -> str:
        """Return the worker description."""
        return "A custom worker for specialized tasks"
    
    @property
    def visibility(self) -> str:
        """Return the worker visibility."""
        return "protected"  # or "public" if needed
    
    @property
    def run_in_executor(self) -> bool:
        """Return whether the worker should run in an executor."""
        return True  # Set to True for CPU-intensive workers
    
    async def start(self, config: Union[WorkerConfig, Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        """Start a new worker session."""
        # Handle both dict and WorkerConfig input
        if isinstance(config, dict):
            config = WorkerConfig(**config)
        
        session_id = config.id
        progress_callback = config.get("progress_callback")
        
        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")
        
        # Report initial progress
        if progress_callback:
            progress_callback({
                "type": "info",
                "message": f"Starting session {session_id} for app {config.app_id}"
            })
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            app_id=config.app_id,
            workspace=config.workspace,
            client_id=config.client_id,
            status=SessionStatus.STARTING,
            app_type=config.manifest.get("type", "unknown"),
            entry_point=config.entry_point,
            created_at=datetime.now().isoformat(),
            metadata=config.manifest
        )
        
        self._sessions[session_id] = session_info
        
        try:
            # Phase 1: Server connection
            if progress_callback:
                progress_callback({
                    "type": "info",
                    "message": "Connecting to server and artifact manager..."
                })
            
            server = await connect_to_server({"server_url": config.server_url})
            artifact_manager = await server.get_service("public/artifact-manager")
            
            if progress_callback:
                progress_callback({
                    "type": "success",
                    "message": "Server connection established"
                })
            
            # Phase 2: Application file processing
            if progress_callback:
                progress_callback({
                    "type": "info",
                    "message": "Processing application files..."
                })
            
            # Example: Read application files
            try:
                file_url = await artifact_manager.get_file(
                    config.artifact_id, 
                    file_path="config.json"
                )
                # Download and process file...
                print(f"Found config file: {file_url}")
                
                if progress_callback:
                    progress_callback({
                        "type": "success",
                        "message": "Application files processed successfully"
                    })
            except Exception as e:
                print(f"No config file found: {e}")
                if progress_callback:
                    progress_callback({
                        "type": "warning",
                        "message": "No configuration file found, using defaults"
                    })
            
            # Phase 3: Session finalization
            if progress_callback:
                progress_callback({
                    "type": "info",
                    "message": "Finalizing session setup..."
                })
            
            # Store session data
            session_data = {
                "server": server,
                "artifact_manager": artifact_manager,
                "logs": [f"Session {session_id} started"],
                "custom_data": {"key": "value"}
            }
            self._session_data[session_id] = session_data
            
            # Update session status
            session_info.status = SessionStatus.RUNNING
            
            if progress_callback:
                progress_callback({
                    "type": "success",
                    "message": f"Session {session_id} started successfully"
                })
            
            print(f"Session {session_id} started successfully")
            return session_id
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            
            if progress_callback:
                progress_callback({
                    "type": "error",
                    "message": f"Failed to start session: {str(e)}"
                })
            
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise
    
    async def stop(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Stop a worker session."""
        if session_id not in self._sessions:
            return  # Session not found, already stopped
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        try:
            # Your custom cleanup logic here
            print(f"Stopping session {session_id}")
            
            session_data = self._session_data.get(session_id)
            if session_data:
                # Clean up resources
                if "server" in session_data:
                    await session_data["server"].disconnect()
            
            session_info.status = SessionStatus.STOPPED
            print(f"Session {session_id} stopped successfully")
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            raise
        finally:
            # Always clean up
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)
    
    async def list_sessions(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> List[SessionInfo]:
        """List all sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]
    
    async def get_session_info(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> SessionInfo:
        """Get information about a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return self._sessions[session_id]
    
    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session {session_id} not found")
        
        session_data = self._session_data.get(session_id, {})
        logs = session_data.get("logs", [])
        
        # Apply offset and limit
        if limit:
            logs = logs[offset:offset + limit]
        else:
            logs = logs[offset:]
        
        if type:
            return logs
        else:
            return {"log": logs, "error": []}
    
    async def prepare_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Prepare workspace for worker operations."""
        print(f"Preparing workspace {workspace}")
    
    async def close_workspace(self, workspace: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Close workspace and cleanup sessions."""
        print(f"Closing workspace {workspace}")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id, context=context)
            except Exception as e:
                print(f"Failed to stop session {session_id}: {e}")

# Register and start the worker
async def main():
    worker = MyCustomWorker()
    
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": "http://localhost:9527"
    })
    
    # Register worker as a service (Method 1: using get_worker_service)
    service_config = worker.get_worker_service()
    service = await server.register_service(service_config)
    
    # Alternative (Method 2: using register_worker_service helper)
    # service = await worker.register_worker_service(server)
    
    print(f"Worker registered: {service.id}")
    print("Worker is running...")
    
    # Keep running
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

### Method 2: Simple Functions (Minimal Approach)

```python
"""Simple worker using functions only."""

import asyncio
from hypha_rpc import connect_to_server

# Simple storage for sessions
sessions = {}
session_data = {}

async def start(config, context=None):
    """Start a new worker session."""
    session_id = config["id"]
    
    print(f"Starting session {session_id} for app {config['app_id']}")
    
    # Get app type from manifest
    app_type = config["manifest"]["type"]
    
    # Store session information
    session_info = {
        "session_id": session_id,
        "app_id": config["app_id"],
        "workspace": config["workspace"],
        "client_id": config["client_id"],
        "status": "running",
        "app_type": app_type,
        "created_at": "2024-01-01T00:00:00Z",
        "entry_point": config.get("entry_point"),
        "metadata": config["manifest"]
    }
    
    sessions[session_id] = session_info
    
    # Connect to artifact manager to access files
    server = await connect_to_server({"server_url": config["server_url"]})
    artifact_manager = await server.get_service("public/artifact-manager")
    
    # Store session data
    session_data[session_id] = {
        "server": server,
        "artifact_manager": artifact_manager,
        "logs": [f"Session {session_id} started"]
    }
    
    return session_id

async def stop(session_id, context=None):
    """Stop a worker session."""
    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"
        # Clean up resources
        if session_id in session_data:
            data = session_data[session_id]
            if "server" in data:
                await data["server"].disconnect()
        session_data.pop(session_id, None)
        print(f"Session {session_id} stopped")

async def list_sessions(workspace, context=None):
    """List all sessions for a workspace."""
    return [
        session for session in sessions.values()
        if session["workspace"] == workspace
    ]

async def get_session_info(session_id, context=None):
    """Get information about a session."""
    if session_id not in sessions:
        raise Exception(f"Session {session_id} not found")
    return sessions[session_id]

async def get_logs(session_id, type=None, offset=0, limit=None, context=None):
    """Get logs for a session."""
    if session_id not in session_data:
        return [] if type else {"log": [], "error": []}
    
    logs = session_data[session_id].get("logs", [])
    
    # Apply offset and limit
    if limit:
        logs = logs[offset:offset + limit]
    else:
        logs = logs[offset:]
    
    return logs if type else {"log": logs, "error": []}

async def prepare_workspace(workspace, context=None):
    """Prepare workspace for worker operations."""
    print(f"Preparing workspace {workspace}")

async def close_workspace(workspace, context=None):
    """Close workspace and cleanup sessions."""
    print(f"Closing workspace {workspace}")
    sessions_to_stop = [
        session_id for session_id, session in sessions.items()
        if session["workspace"] == workspace
    ]
    for session_id in sessions_to_stop:
        await stop(session_id, context=context)

# Register the worker
async def main():
    server = await connect_to_server({
        "server_url": "http://localhost:9527"
    })
    
    # Register the worker service
    service = await server.register_service({
        "id": f"simple-worker-{id(start)}",
        "name": "Simple Worker",
        "description": "A simple worker implemented with functions",
        "type": "server-app-worker",
        "config": {
            "visibility": "protected",
            "run_in_executor": True,
        },
        "supported_types": ["simple-task", "custom-type"],
        "start": start,
        "stop": stop,
        "list_sessions": list_sessions,
        "get_logs": get_logs,
        "get_session_info": get_session_info,
        "prepare_workspace": prepare_workspace,
        "close_workspace": close_workspace,
    })
    
    print(f"Simple worker registered: {service.id}")
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

## Hypha Worker Implementation in Javascript

### Method 1: Using Class Structure

```javascript
// my-worker.js
const { connectToServer } = require('hypha-rpc');

class MyCustomWorker {
    constructor() {
        this.sessions = new Map();
        this.sessionData = new Map();
    }

    get supportedTypes() {
        return ["javascript-custom", "my-js-type"];
    }

    get name() {
        return "My JavaScript Worker";
    }

    get description() {
        return "A custom JavaScript worker";
    }

    get visibility() {
        return "protected";
    }

    get runInExecutor() {
        return true;
    }

    get requireContext() {
        return true;
    }

    get serviceId() {
        return `${this.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`;
    }

    async start(config, context = null) {
        const sessionId = config.id;
        const progressCallback = config.progress_callback;
        
        if (this.sessions.has(sessionId)) {
            throw new Error(`Session ${sessionId} already exists`);
        }

        // Report initial progress
        if (progressCallback) {
            progressCallback({
                type: "info",
                message: `Starting session ${sessionId} for app ${config.app_id}`
            });
        }

        // Create session info
        const sessionInfo = {
            session_id: sessionId,
            app_id: config.app_id,
            workspace: config.workspace,
            client_id: config.client_id,
            status: "starting",
            app_type: config.manifest.type,
            entry_point: config.entry_point,
            created_at: new Date().toISOString(),
            metadata: config.manifest
        };

        this.sessions.set(sessionId, sessionInfo);

        try {
            // Phase 1: Server connection
            if (progressCallback) {
                progressCallback({
                    type: "info",
                    message: "Connecting to server and artifact manager..."
                });
            }

            const server = await connectToServer({server_url: config.server_url});
            const artifactManager = await server.getService("public/artifact-manager");

            if (progressCallback) {
                progressCallback({
                    type: "success",
                    message: "Server connection established"
                });
            }

            // Phase 2: Application file processing
            if (progressCallback) {
                progressCallback({
                    type: "info",
                    message: "Processing application files..."
                });
            }

            // Example: Access application files
            try {
                const fileUrl = await artifactManager.getFile(
                    config.artifact_id,
                    {file_path: "package.json"}
                );
                console.log(`Found package.json: ${fileUrl}`);
                
                if (progressCallback) {
                    progressCallback({
                        type: "success",
                        message: "Application files processed successfully"
                    });
                }
            } catch (error) {
                console.log(`No package.json found: ${error.message}`);
                if (progressCallback) {
                    progressCallback({
                        type: "warning",
                        message: "No package.json found, using defaults"
                    });
                }
            }

            // Phase 3: Session finalization
            if (progressCallback) {
                progressCallback({
                    type: "info",
                    message: "Finalizing session setup..."
                });
            }

            // Store session data
            const sessionData = {
                server: server,
                artifactManager: artifactManager,
                logs: [`Session ${sessionId} started`],
                customData: {key: "value"}
            };
            this.sessionData.set(sessionId, sessionData);

            // Update status
            sessionInfo.status = "running";
            
            if (progressCallback) {
                progressCallback({
                    type: "success",
                    message: `Session ${sessionId} started successfully`
                });
            }
            
            console.log(`Session ${sessionId} started successfully`);
            return sessionId;

        } catch (error) {
            sessionInfo.status = "failed";
            sessionInfo.error = error.message;
            
            if (progressCallback) {
                progressCallback({
                    type: "error",
                    message: `Failed to start session: ${error.message}`
                });
            }
            
            this.sessions.delete(sessionId);
            throw error;
        }
    }

    async stop(sessionId, context = null) {
        if (!this.sessions.has(sessionId)) {
            return; // Already stopped
        }

        const sessionInfo = this.sessions.get(sessionId);
        sessionInfo.status = "stopping";

        try {
            console.log(`Stopping session ${sessionId}`);

            const sessionData = this.sessionData.get(sessionId);
            if (sessionData && sessionData.server) {
                await sessionData.server.disconnect();
            }

            sessionInfo.status = "stopped";
            console.log(`Session ${sessionId} stopped successfully`);

        } catch (error) {
            sessionInfo.status = "failed";
            sessionInfo.error = error.message;
            throw error;
        } finally {
            this.sessions.delete(sessionId);
            this.sessionData.delete(sessionId);
        }
    }

    async listSessions(workspace, context = null) {
        const workspaceSessions = [];
        for (const [sessionId, sessionInfo] of this.sessions) {
            if (sessionInfo.workspace === workspace) {
                workspaceSessions.push(sessionInfo);
            }
        }
        return workspaceSessions;
    }

    async getSessionInfo(sessionId, context = null) {
        if (!this.sessions.has(sessionId)) {
            throw new Error(`Session ${sessionId} not found`);
        }
        return this.sessions.get(sessionId);
    }

    async getLogs(sessionId, type = null, offset = 0, limit = null, context = null) {
        if (!this.sessions.has(sessionId)) {
            throw new Error(`Session ${sessionId} not found`);
        }

        const sessionData = this.sessionData.get(sessionId) || {};
        let logs = sessionData.logs || [];

        // Apply offset and limit
        if (limit) {
            logs = logs.slice(offset, offset + limit);
        } else {
            logs = logs.slice(offset);
        }

        return type ? logs : {log: logs, error: []};
    }

    async prepareWorkspace(workspace, context = null) {
        console.log(`Preparing workspace ${workspace}`);
    }

    async closeWorkspace(workspace, context = null) {
        console.log(`Closing workspace ${workspace}`);
        
        const sessionsToStop = [];
        for (const [sessionId, sessionInfo] of this.sessions) {
            if (sessionInfo.workspace === workspace) {
                sessionsToStop.push(sessionId);
            }
        }

        for (const sessionId of sessionsToStop) {
            try {
                await this.stop(sessionId, context);
            } catch (error) {
                console.log(`Failed to stop session ${sessionId}: ${error.message}`);
            }
        }
    }

    getWorkerService() {
        return {
            id: this.serviceId,
            name: this.name,
            description: this.description,
            type: "server-app-worker",
            config: {
                visibility: this.visibility,
                run_in_executor: this.runInExecutor,
                require_context: this.requireContext,
            },
            supported_types: this.supportedTypes,
            start: this.start.bind(this),
            stop: this.stop.bind(this),
            list_sessions: this.listSessions.bind(this),
            get_logs: this.getLogs.bind(this),
            get_session_info: this.getSessionInfo.bind(this),
            prepare_workspace: this.prepareWorkspace.bind(this),
            close_workspace: this.closeWorkspace.bind(this),
        };
    }
}

// Register and start the worker
async function main() {
    try {
        const worker = new MyCustomWorker();
        
        const server = await connectToServer({
            server_url: "http://localhost:9527"
        });

        const serviceConfig = worker.getWorkerService();
        const service = await server.registerService(serviceConfig);

        console.log(`Worker registered: ${service.id}`);
        console.log("Worker is running...");

        // Keep the process running
        process.on('SIGINT', async () => {
            console.log('Shutting down...');
            await server.disconnect();
            process.exit(0);
        });

    } catch (error) {
        console.error("Failed to start worker:", error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = MyCustomWorker;
```

### Method 2: Simple Functions

```javascript
// simple-worker.js
const { connectToServer } = require('hypha-rpc');

// Simple storage for sessions
const sessions = new Map();
const sessionData = new Map();

async function start(config, context = null) {
    const sessionId = config.id;
    
    console.log(`Starting session ${sessionId} for app ${config.app_id}`);
    
    // Store session information
    const sessionInfo = {
        session_id: sessionId,
        app_id: config.app_id,
        workspace: config.workspace,
        client_id: config.client_id,
        status: "running",
        app_type: config.manifest.type,
        created_at: new Date().toISOString(),
        entry_point: config.entry_point,
        metadata: config.manifest
    };
    
    sessions.set(sessionId, sessionInfo);
    
    // Connect to artifact manager
    const server = await connectToServer({server_url: config.server_url});
    const artifactManager = await server.getService("public/artifact-manager");
    
    // Store session data
    sessionData.set(sessionId, {
        server: server,
        artifactManager: artifactManager,
        logs: [`Session ${sessionId} started`]
    });
    
    return sessionId;
}

async function stop(sessionId, context = null) {
    if (sessions.has(sessionId)) {
        const sessionInfo = sessions.get(sessionId);
        sessionInfo.status = "stopped";
        
        // Clean up resources
        const data = sessionData.get(sessionId);
        if (data && data.server) {
            await data.server.disconnect();
        }
        
        sessionData.delete(sessionId);
        console.log(`Session ${sessionId} stopped`);
    }
}

async function listSessions(workspace, context = null) {
    const workspaceSessions = [];
    for (const [sessionId, sessionInfo] of sessions) {
        if (sessionInfo.workspace === workspace) {
            workspaceSessions.push(sessionInfo);
        }
    }
    return workspaceSessions;
}

async function getSessionInfo(sessionId, context = null) {
    if (!sessions.has(sessionId)) {
        throw new Error(`Session ${sessionId} not found`);
    }
    return sessions.get(sessionId);
}

async function getLogs(sessionId, type = null, offset = 0, limit = null, context = null) {
    if (!sessionData.has(sessionId)) {
        return type ? [] : {log: [], error: []};
    }
    
    let logs = sessionData.get(sessionId).logs || [];
    
    // Apply offset and limit
    if (limit) {
        logs = logs.slice(offset, offset + limit);
    } else {
        logs = logs.slice(offset);
    }
    
    return type ? logs : {log: logs, error: []};
}

async function prepareWorkspace(workspace, context = null) {
    console.log(`Preparing workspace ${workspace}`);
}

async function closeWorkspace(workspace, context = null) {
    console.log(`Closing workspace ${workspace}`);
    
    const sessionsToStop = [];
    for (const [sessionId, sessionInfo] of sessions) {
        if (sessionInfo.workspace === workspace) {
            sessionsToStop.push(sessionId);
        }
    }
    
    for (const sessionId of sessionsToStop) {
        await stop(sessionId, context);
    }
}

// Register the worker
async function main() {
    try {
        const server = await connectToServer({
            server_url: "http://localhost:9527"
        });
        
        // Register the worker service
        const service = await server.registerService({
            id: `simple-js-worker-${Date.now()}`,
            name: "Simple JavaScript Worker",
            description: "A simple worker implemented with functions",
            type: "server-app-worker",
            config: {
                visibility: "protected",
                run_in_executor: true,
            },
            supported_types: ["javascript-simple", "custom-js-type"],
            start: start,
            stop: stop,
            list_sessions: listSessions,
            get_logs: getLogs,
            get_session_info: getSessionInfo,
            prepare_workspace: prepareWorkspace,
            close_workspace: closeWorkspace,
        });
        
        console.log(`Simple worker registered: ${service.id}`);
        console.log("Worker is running...");
        
        // Keep the process running
        process.on('SIGINT', async () => {
            console.log('Shutting down...');
            await server.disconnect();
            process.exit(0);
        });
        
    } catch (error) {
        console.error("Failed to start worker:", error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}
```

## Accessing Application Files

Workers can access application files using the artifact manager:

```python
# Python example
server = await connect_to_server({"server_url": config["server_url"]})
artifact_manager = await server.get_service("public/artifact-manager")

# Get file URL
file_url = await artifact_manager.get_file(
    config["artifact_id"], 
    file_path="config.json"
)

# Download file content
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get(file_url)
    if response.status_code == 200:
        content = response.text
```

```javascript
// JavaScript example
const server = await connectToServer({server_url: config.server_url});
const artifactManager = await server.getService("public/artifact-manager");

// Get file URL
const fileUrl = await artifactManager.getFile(
    config.artifact_id,
    {file_path: "config.json"}
);

// Download file content
const response = await fetch(fileUrl);
if (response.ok) {
    const content = await response.text();
}
```

## Session Management

### Session States

Sessions follow these status transitions:
- `starting` → `running` → `stopping` → `stopped`
- `starting` → `failed` (if startup fails)
- `running` → `failed` (if runtime error occurs)

### Session Information

The `get_session_info()` method returns:

```python
{
    "session_id": "workspace/client_id",
    "app_id": "my-app",
    "workspace": "workspace",
    "client_id": "client_id",
    "status": "running",  # SessionStatus enum value
    "app_type": "python-eval",
    "entry_point": "main.py",
    "created_at": "2024-01-01T00:00:00Z",
    "metadata": {...},  # Application manifest
    "error": "error message"  # Only if status is "failed"
}
```

## Compilation Support

Workers can optionally implement a `compile()` method to process application files at install time:

```python
async def compile(self, manifest: Dict[str, Any], files: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compile application manifest and files."""
    
    # Process files and manifest as needed
    processed_manifest = manifest.copy()
    processed_files = []
    
    for file in files:
        if file["name"].endswith(".js"):
            # Example: Minify JavaScript files
            processed_content = minify_js(file["content"])
            processed_files.append({
                "path": file["name"],
                "content": processed_content,
                "format": file.get("format", "text")
            })
        else:
            processed_files.append(file)
    
    # Update manifest if needed
    processed_manifest["compiled"] = True
    
    return processed_manifest, processed_files
```

## Running Your Hypha Workers

### Python
```bash
pip install hypha-rpc
python my-worker.py
```

### JavaScript
```bash
npm install hypha-rpc
node my-worker.js
```

## Key Changes from Previous API

1. **Simplified Interface**: No complex base classes required - implement functions directly
2. **Standardized Config**: All workers receive the same `WorkerConfig` structure
3. **Session Management**: Built-in session tracking with `SessionInfo` objects
4. **Error Handling**: Standardized exceptions (`SessionNotFoundError`, `WorkerError`)
5. **Type Safety**: Optional Pydantic models for Python workers
6. **Consistent Returns**: `start()` returns only `session_id`, detailed info via `get_session_info()`

The new worker API is designed to be simple, consistent, and easy to implement while providing all the necessary functionality for managing application sessions.