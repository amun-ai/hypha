**Hypha** is a generative AI-powered application framework designed for large-scale data management, AI model serving, and real-time communication. Hypha allows the creation of computational platforms consisting of both computational and user interface components.

## Project Overview

The Hypha project aims to provide a serverless application framework. The core piece is a FastAPI-based server (`hypha/server.py`) that provides WebSocket and HTTP transport. On top of the WebSocket transport, Hypha builds a Remote Procedure Call (RPC) system that enables distributed computing and service management.

## Architecture

### Core Module and Event Bus System

**Location**: `hypha/core/__init__.py`

The core module contains the essential init function which establishes the Redis-based event bus system. This event bus system built on top of Redis enables horizontal scaling - allowing multiple Hypha instances to connect to the same Redis server, enabling the system to scale and handle more concurrent WebSocket clients.

The event bus system is implemented in two ways:
1. **Real Redis server** (`RedisEventBus` class): For production deployments supporting horizontal scaling across multiple Hypha instances
   - Uses Redis pub/sub with prefix-based routing (`broadcast:` and `targeted:` prefixes)
   - Supports dynamic pattern subscriptions for client-specific messages
   - Includes health checking and circuit breaker patterns
   - Handles automatic reconnection with exponential backoff
2. **In-memory fake Redis implementation** (via `fakeredis.aioredis.FakeRedis`): For single Hypha server deployments that don't require horizontal scaling
   - Used when no Redis URI is provided
   - Provides same API interface for seamless switching

Key components in `hypha/core/__init__.py`:
- `RedisEventBus`: Main event bus implementation with local client tracking
- `RedisRPCConnection`: RPC connection management over Redis
- Data models: `ServiceInfo`, `ClientInfo`, `WorkspaceInfo`, `UserInfo`, `TokenConfig`

### RPC and WebSocket Architecture

**WebSocket Server Location**: `hypha/websocket.py`

The `WebsocketServer` class handles:
- Client authentication and connection management
- Message routing between clients and the event bus
- Graceful disconnection handling
- Connection health monitoring

The RPC system is integrated with the WebSocket server and uses the `hypha-rpc` library for:
- Remote procedure call semantics
- Service registration and discovery
- Method invocation with JSON Schema validation
- Bidirectional communication with callback support

### Authentication Systems

**Location**: `hypha/core/auth.py`

#### External Authentication (Auth0)
- Uses Auth0 as the external authentication provider
- JWT token validation using RSA keys from JWKS endpoint
- Environment variables: `AUTH0_CLIENT_ID`, `AUTH0_DOMAIN`, `AUTH0_AUDIENCE`, `AUTH0_ISSUER`
- Functions:
  - `valid_token()`: Validates Auth0 JWT tokens
  - `get_rsa_key()`: Retrieves RSA keys for token verification
  - `get_user_info()`: Extracts user information from tokens

#### Internal Authentication System
- JWT-based internal token generation using `JWT_SECRET`
- Functions:
  - `generate_auth_token()`: Creates internal auth tokens
  - `parse_auth_token()`: Parses and validates internal tokens
  - `create_scope()`: Creates permission scopes for fine-grained access control
  - `generate_anonymous_user()`: Creates anonymous user tokens

### Store Module and Workspace Management

**Store Module Location**: `hypha/core/store.py`

The `RedisStore` class serves as the central orchestrator that stitches together:
- Authentication systems
- Workspace management
- Service registry
- Artifact manager
- S3 storage controller
- WebSocket server

Key features:
- Manages both Redis and SQL database connections
- Handles workspace interfaces through `WorkspaceInterfaceContextManager`
- Supports both production Redis and in-memory FakeRedis
- Integrates with OpenAI and Ollama for AI capabilities

**Workspace Manager Location**: `hypha/core/workspace.py`

The `WorkspaceManager` class provides:
- Multi-tenant workspace isolation
- Service registration and discovery within workspaces
- Client management and activity tracking
- Event logging to SQL database (`EventLog` model)
- Workspace lifecycle management with activity-based cleanup

Key components:
- `WorkspaceActivityManager`: Intelligent cleanup of inactive workspaces
- Protected system workspaces: `public`, `ws-user-root`, `ws-anonymous`
- Service visibility modes: `protected` (default) and `public`

### Storage System and Databases

**Artifact Manager Location**: `hypha/artifact.py`

The artifact management system provides:
- S3-compatible object storage integration
- SQL database for metadata (SQLAlchemy with async support)
- Hybrid storage model combining SQL and S3

Key classes:
- `ArtifactModel`: SQLAlchemy model for artifact metadata
- `CollectionArtifact`: Special artifact type for folder-like structures
- Support for:
  - File versioning with Git-like commit model
  - Multipart uploads for large files
  - Zip download functionality
  - Vector search capabilities (when enabled)

**Vector Search Location**: `hypha/vectors.py`

The `VectorSearchEngine` class provides:
- Embedding generation using sentence transformers
- Vector similarity search
- Hybrid search combining vector and metadata

### Serverless Applications and Workers

**Application Controller Location**: `hypha/apps.py`

The `ServerAppController` class manages:
- Application lifecycle (install, start, stop, uninstall)
- Worker pool management with autoscaling
- Dynamic worker allocation based on demand
- Application manifest validation

Key components:
- `AutoscalingManager`: Handles worker scaling policies
- `WorkerSelectionConfig`: Configures worker selection strategies
- Support for multiple worker types based on application requirements

**Worker Implementations Location**: `hypha/workers/`

1. **Browser Worker** (`hypha/workers/browser.py`)
   - Playwright-based browser automation
   - Sandboxed JavaScript/Python execution
   - WebSocket communication with Hypha server

2. **Terminal Worker** (`hypha/workers/terminal.py`)
   - Command execution in isolated environments
   - Shell script support

3. **Conda Worker** (`hypha/workers/conda.py`, `hypha/workers/conda_executor/`)
   - Native Python execution with conda environments
   - GPU support for ML workloads
   - Jupyter kernel integration
   - Shared memory support for efficient data transfer

4. **Kubernetes Worker** (`hypha/workers/k8s.py`)
   - Pod and service management in Kubernetes
   - Dynamic resource allocation

5. **MCP Proxy Worker** (`hypha/workers/mcp_proxy.py`)
   - Converts MCP servers to Hypha services
   - Bidirectional protocol translation

6. **A2A Proxy Worker** (`hypha/workers/a2a_proxy.py`)
   - Integrates A2A-compatible servers
   - Protocol adaptation layer

## Extension Modules & Protocol Converters

### MCP Integration

**Location**: `hypha/mcp.py`

Key components:
- `HyphaMCPAdapter`: Converts Hypha services to MCP-compatible endpoints
- `MCPSessionWrapper`: Manages MCP session lifecycle
- `MCPRoutingMiddleware`: Routes MCP requests to appropriate Hypha services
- `create_mcp_app_from_service()`: Creates FastAPI app for MCP endpoints
- Support for HTTP JSON-RPC protocol used by Claude, Gemini CLI, etc.

### A2A Integration

**Location**: `hypha/a2a.py`

Provides bidirectional conversion between Hypha services and A2A protocol endpoints.

### Transport Layers

**HTTP Module Location**: `hypha/http.py`

Provides RESTful HTTP endpoints for:
- Service invocation
- File upload/download
- Artifact management
- Static file serving
- View endpoints for artifact rendering

**ASGI Service Support** (`hypha/http.py::ASGIRoutingMiddleware`)
- Register services with a `serve` function that handles ASGI calls
- Mount frameworks like FastAPI as Hypha services
- Hypha server receives HTTP requests and proxies them to connected ASGI services
- ASGI services can run inside browser workers, enabling serverless web servers
- Route pattern: `/{workspace}/apps/{service_id}/{path:path}`

**Functions Server Type** (`hypha/http.py::handle_function_service`)
- More generic than ASGI - define Python or JavaScript functions to handle HTTP requests
- Functions receive HTTP request scope and return response dictionaries
- Support for nested function paths and default fallback functions
- Automatic CORS header injection for cross-origin requests
- Can be used for lightweight HTTP endpoints without full ASGI complexity

**ASGI Proxy Utilities Location**: `hypha/utils/asgi_proxy.py`
- `ProxyConfig` and `ProxyContext` for configuring proxy behavior
- Support for streaming responses based on content size thresholds
- Helper functions for converting between proxy and user responses

**WebSocket Module Location**: `hypha/websocket.py`

Enhanced capabilities over HTTP:
- Bidirectional real-time communication
- Remote callback function support
- Streaming and lazy data transmission
- Generator and async pattern support
- Lower latency for interactive applications

### Admin and Developer Features

**Interactive Module Location**: `hypha/interactive.py`

Provides CLI interface for:
- Token generation and management
- Service debugging and testing
- Workspace administration
- System cleanup operations
- Only activated when no external Auth0 server is configured

**Startup Module Location**: `hypha/startup.py`

Enables custom initialization:
- Loading additional Python modules at server start
- Registering custom authentication providers
- Creating default services in public workspaces
- System-wide configuration overrides

**Server Configuration Location**: `hypha/server.py`

Main FastAPI application setup with:
- Command-line argument parsing
- Environment variable configuration
- Route registration
- Middleware setup
- Static file mounting options

### Additional Features

**Queue System Location**: `hypha/queue.py`

Provides task queue functionality for asynchronous job processing.

**S3 Integration Location**: `hypha/s3.py`

Handles S3-compatible storage operations:
- Presigned URL generation
- Multipart upload coordination
- Access control integration

**MinIO Support Location**: `hypha/minio.py`

Provides MinIO-specific storage features when using MinIO as S3 backend.

**Local Authentication Location**: `hypha/local_auth.py`

Alternative authentication system for deployments without Auth0.

## Service & Application Lifecycle Flow

1. Client connects via WebSocket (`hypha/websocket.py`); token validated via Auth0 or internal auth (`hypha/core/auth.py`)
2. Client registers a service in a workspace (managed by `hypha/core/workspace.py`)
3. Worker selection based on application type (`hypha/apps.py`)
4. Worker executes application script (`hypha/workers/` implementations)
5. Services registered in workspace become available to other clients
6. Event bus (`hypha/core/__init__.py::RedisEventBus`) routes messages across instances
7. Autoscaling manager monitors load and adjusts worker pool (`hypha/apps.py::AutoscalingManager`)

## Installation Options

### Helm Chart
- **Basic Hypha server**: Minimal deployment with core Hypha server
- **Hypha server kit**: Full deployment including:
  - Postgres database for scalability
  - Redis server for event bus and caching
  - S3-compatible storage (optional)
  - Triton inference server support (optional)

### Docker Compose
- Docker-based deployment configurations
- Suitable for development and smaller deployments
- Includes optional services like Redis, Postgres, MinIO

## Key Design Goals

- **Multi-tenant isolation** through workspace system with activity-based cleanup
- **Fine-grained authorization** using JWT scopes and context injection
- **Dynamic serverless runtime** with intelligent autoscaling and worker lifecycle management
- **Horizontal scaling** via Redis-backed event bus with prefix-based routing
- **Protocol interoperability** (MCP, A2A, HTTP, WebSocket, gRPC via Triton)
- **Hybrid storage** combining SQL metadata with S3 object storage
- **Resilient architecture** with circuit breakers, health checks, and automatic reconnection

## Implementation Considerations

- Event bus uses prefix-based routing (`broadcast:` and `targeted:`) for efficient message distribution
- WebSocket connections tracked per-client for optimized local routing
- Activity tracking enables automatic cleanup of inactive resources
- SQL models use SQLAlchemy with async support for non-blocking operations
- Worker implementations use asyncio for concurrent task handling
- Support for both synchronous (HTTP) and asynchronous (WebSocket) communication patterns
- Comprehensive metrics and monitoring via Prometheus-compatible endpoints


## Service Documentation and Type Annotations for LLM Agents

Type annotations in Hypha services are crucial for enabling LLM agents to autonomously discover and use services. Hypha uses `@schema_method` and `@schema_function` decorators with Pydantic Field descriptions to provide rich context for AI agents.

### Key Principles

1. **Use @schema_method with Field descriptions** - Every parameter should have a clear, detailed description
2. **Document limitations and constraints** - Include rate limits, size restrictions, supported formats
3. **Provide actionable error messages** - Help agents recover from failures
4. **Keep descriptions concise but complete** - Balance detail with token efficiency

### Hypha Schema Method Pattern

```python
from hypha_rpc.utils.schema import schema_method
from pydantic import Field
from typing import Optional, List, Dict

@schema_method
async def install(
    self,
    source: Optional[str] = Field(
        None,
        description="The source code of the application, URL to fetch the source, or None if using config. Can be raw HTML content, ImJoy/Hypha plugin code, or a URL to download the source from. URLs must be HTTPS or localhost/127.0.0.1.",
    ),
    app_id: Optional[str] = Field(
        None,
        description="The unique identifier of the application to install. This is typically the alias of the application.",
    ),
    overwrite: Optional[bool] = Field(
        False,
        description="Whether to overwrite existing app with same name. Set to True to replace existing installations.",
    ),
    timeout: Optional[float] = Field(
        None,
        description="Maximum time to wait for installation completion in seconds. Increase for complex apps that take longer to start.",
    ),
    context: Optional[dict] = Field(
        None,
        description="Additional context information including user and workspace details. Usually provided automatically by the system.",
    ),
) -> str:
    """Install a server application.
    
    This method installs an application from source code or manifest:
    1. Validates user permissions in the workspace
    2. Compiles the app using appropriate worker type
    3. Creates an artifact in the applications collection
    4. Optionally tests the app can start successfully
    
    Returns:
        Application manifest dictionary with installed app details
        
    Raises:
        ValueError: If app_id contains invalid characters or manifest is malformed
        Exception: If no suitable worker found or installation fails
    """
```

### Best Practices from Hypha Codebase

1. **Parameter Documentation Pattern**
   - Start with what the parameter represents
   - Include examples of valid values
   - Specify format requirements (e.g., "workspace/alias")
   - Note if parameter is auto-provided by system

2. **Complex Type Documentation**
   ```python
   class WorkerSelectionConfig(BaseModel):
       mode: Optional[str] = Field(
           None,
           description="Mode for selecting the worker. Can be 'random', 'first', 'last', 'exact', 'min_load', or 'select:criteria:function' format (e.g., 'select:min:get_load', 'select:max:get_cpu_usage')",
       )
       timeout: Optional[float] = Field(
           10.0,
           description="The timeout duration in seconds for fetching the worker. This determines how long the function will wait for a worker to respond before considering it a timeout.",
       )
   ```

3. **Method Documentation Structure**
   - Brief one-line summary
   - Detailed explanation of what happens step-by-step
   - Returns section with type and meaning
   - Raises section for error conditions
   - Examples when behavior is complex

4. **Service Registration Pattern**
   ```python
   def get_service_api(self):
       return {
           "id": "server-apps",
           "name": "Server Apps Controller",
           "description": "Manage server applications lifecycle",
           "config": {
               "visibility": "public",
               "require_context": True
           },
           "install": self.install,
           "start": self.start,
           "stop": self.stop,
           "list": self.list_apps,
       }
   ```

### Schema Validation Benefits

- **Automatic validation** - Pydantic validates types at runtime
- **Auto-generated documentation** - Schema can be exported for API docs
- **IDE support** - Type hints enable autocomplete and type checking
- **LLM understanding** - Structured descriptions help AI agents use services correctly

### Monitoring and Improvement

Track service usage patterns to refine descriptions:
- Which services have high error rates (unclear documentation)
- Which parameters are frequently misused (ambiguous descriptions)
- Which services are underutilized (poor discoverability)

## Development Guidelines

We need to follow the TDD development approach where every feature should be translated into a set of tests.

After changes to the repo, try to run the tests, and ensure it pass locally.


When you create tests, never cheat, no mockup, always make real tests, introduce new fixture in tests/conftest.py file, and always aim for full coverage.

DO NOT use defensive programming pattern, NEVER do try and finally to let the error pass silently, NEVER convert except to warning, just for the sake of passing tests, you should be honest on failures, unable to fix, report the honest results and failure to the user.

Never ignore errors, or warnings, either it's relate to the current tests goal or not, you should always at least warn the user about the failing tests, never ignore them, always aim for full test passes. Otherwise fix them.

If instructed, you can commit the changes (only the files related to your current task) and push to remote, and then use the github-ci mcp tools to check for the ci status, we need to wait for the CI tests to pass, if any error occurs, read the error, and try to fix the failed tests. You should always wait for the CI to pass so we can merge the PR, that the main goal of each branch.

In case of error, use git diff tool to compare passing branch to compare differences. ALWAYS aim to find root cause of bugs or issues, never do shallow fix to just let the test pass, or adding try except, or add pytest skip, never do this, these are cheating, will facing serious consequences.

If you failed to run the tests, warn the user the failure, never conclude to the user that you completed the task if you haven't actually run and pass the tests.

When user provide description about error, provide CI error logs, or any other information, the first thing to do is to try to create a test to reproduce the issue, never try to fix it in the first place, always try to reproduce the issue and do a root cause analysis, once you clearly understand the issue, move on to the actual fix.
