"""Agent Skills module for Hypha.

This module provides Agent Skills documentation endpoints that dynamically
generate skills content tailored to each workspace. It follows the Agent Skills
specification (https://agentskills.io/specification.md) to enable AI agents
to discover and use Hypha services.

The skills endpoint provides:
- SKILL.md: Main skill instructions
- REFERENCE.md: Comprehensive API reference (generated from service schemas)
- EXAMPLES.md: Code examples for common operations
- WORKSPACE_CONTEXT.md: Workspace-specific context and configuration
"""

import io
import json
import logging
import os
import sys
import inspect
import re
import time
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from starlette.types import ASGIApp

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("skills")
logger.setLevel(LOGLEVEL)

# Skill metadata constants
SKILL_NAME = "hypha"
SKILL_VERSION = "1.0.0"

# Documentation directory
DOCS_DIR = Path(__file__).parent.parent / "docs"

# Cache TTL in seconds for enabled services detection
ENABLED_SERVICES_CACHE_TTL = 60

# Mapping of service IDs to their implementing classes
# Used for source code endpoints and API documentation generation
SERVICE_CLASS_MAP = {
    "workspace-manager": ("hypha.core.workspace", "WorkspaceManager"),
    "artifact-manager": ("hypha.artifact", "ArtifactController"),
    "server-apps": ("hypha.apps", "ServerAppController"),
    "s3-storage": ("hypha.s3", "S3Controller"),
    "queue": ("hypha.queue", "QueueService"),
    "triton-client": ("hypha.triton", "TritonProxy"),
}


def extract_schema_from_callable(func: Callable) -> Optional[Dict[str, Any]]:
    """Extract JSON schema from a callable if it has __schema__ attribute."""
    if hasattr(func, "__schema__"):
        return func.__schema__
    return None


def extract_docstring(func: Callable) -> str:
    """Extract and clean docstring from a callable."""
    doc = inspect.getdoc(func)
    if doc:
        # Clean up the docstring
        return doc.strip()
    return ""


def _service_has_schema(service_dict: dict) -> bool:
    """Check if a service has meaningful service_schema with documented methods.

    Returns True only if at least one method has a non-empty description
    or non-empty parameters. This filters out auto-generated schemas for
    undecorated functions (which have empty descriptions and empty properties).
    """
    schema = service_dict.get("service_schema")
    if not schema or not isinstance(schema, dict):
        return False
    for method_name, method_def in schema.items():
        if not isinstance(method_def, dict):
            continue
        if method_def.get("type") != "function":
            continue
        function_def = method_def.get("function")
        if not function_def or not isinstance(function_def, dict):
            continue
        # Check for meaningful schema: non-empty description or parameters
        has_description = bool(function_def.get("description", "").strip())
        params = function_def.get("parameters", {})
        has_params = bool(params.get("properties", {}))
        if has_description or has_params:
            return True
    return False


def _normalize_service_slug(service_id: str) -> str:
    """Convert a full service ID to a URL-safe slug for REFERENCE paths.

    Examples:
        "public/client_id:my-service" -> "my-service"
        "workspace/client:svc@app" -> "svc"
        "simple-service" -> "simple-service"
    """
    # Remove workspace prefix
    if "/" in service_id:
        service_id = service_id.split("/", 1)[1]
    # Extract service name (after colon)
    if ":" in service_id:
        service_id = service_id.split(":", 1)[1]
    # Remove @app-id suffix
    if "@" in service_id:
        service_id = service_id.split("@", 1)[0]
    return service_id


def format_schema_as_markdown(schema: Dict[str, Any], method_name: str, docstring: str = "") -> str:
    """Format a JSON schema as markdown documentation."""
    lines = [f"### {method_name}\n"]

    # Add description from schema or docstring
    description = schema.get("description") or docstring.split("\n")[0] if docstring else ""
    if description:
        lines.append(f"{description}\n")

    # Add detailed docstring if available
    if docstring and "\n" in docstring:
        # Get everything after the first line
        details = "\n".join(docstring.split("\n")[1:]).strip()
        if details:
            lines.append(f"\n{details}\n")

    # Add parameters
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    # Filter out server-injected parameters that users should not pass
    SERVER_INJECTED_PARAMS = {"context"}
    user_properties = {k: v for k, v in properties.items() if k not in SERVER_INJECTED_PARAMS}

    if user_properties:
        lines.append("\n**Parameters:**\n")
        for param_name, param_info in user_properties.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            is_required = param_name in required
            req_marker = " *(required)*" if is_required else ""

            # Handle enum values
            if "enum" in param_info:
                enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                param_desc += f" Options: {enum_values}"

            # Handle default values
            if "default" in param_info:
                default = param_info["default"]
                if isinstance(default, str):
                    param_desc += f" Default: `\"{default}\"`"
                elif default is not None:
                    param_desc += f" Default: `{default}`"

            lines.append(f"- `{param_name}` ({param_type}){req_marker}: {param_desc}\n")

    # Add return type if available
    returns = schema.get("returns", {})
    if returns:
        return_type = returns.get("type", "any")
        return_desc = returns.get("description", "")
        lines.append(f"\n**Returns:** `{return_type}` - {return_desc}\n")

    return "".join(lines)


def serialize_service_for_docs(service_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a service dictionary, extracting schemas from callable methods."""
    result = {
        "id": service_dict.get("id"),
        "name": service_dict.get("name"),
        "description": service_dict.get("description"),
        "type": service_dict.get("type"),
        "methods": {}
    }

    # Skip non-method keys
    skip_keys = {"id", "name", "description", "type", "config", "_intf"}

    for key, value in service_dict.items():
        if key in skip_keys or key.startswith("_"):
            continue

        if callable(value):
            method_info = {"name": key}

            # Extract schema
            schema = extract_schema_from_callable(value)
            if schema:
                method_info["schema"] = schema

            # Extract docstring
            docstring = extract_docstring(value)
            if docstring:
                method_info["docstring"] = docstring

            result["methods"][key] = method_info

    return result


def load_documentation_file(filename: str) -> Optional[str]:
    """Load a documentation file from the docs directory."""
    filepath = DOCS_DIR / filename
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return None


def extract_section_from_doc(doc_content: str, section_title: str) -> Optional[str]:
    """Extract a specific section from markdown documentation."""
    if not doc_content:
        return None

    # Find section by header
    pattern = rf"^##?\s+{re.escape(section_title)}\s*\n(.*?)(?=^##?\s+|\Z)"
    match = re.search(pattern, doc_content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class DynamicDocGenerator:
    """Generates documentation dynamically from service APIs and doc files."""

    def __init__(self, store, server_url: str):
        self.store = store
        self.server_url = server_url
        self._doc_cache = {}
        self._enabled_services_cache = None
        self._enabled_services_cache_time = 0

    def _get_doc_content(self, filename: str) -> Optional[str]:
        """Get cached documentation content."""
        if filename not in self._doc_cache:
            self._doc_cache[filename] = load_documentation_file(filename)
        return self._doc_cache[filename]

    def get_enabled_services(self) -> List[str]:
        """Get list of enabled public services from the store.

        Returns a list of service IDs that are currently registered.
        Uses a TTL cache to avoid stale results when services change.
        """
        now = time.time()
        if (
            self._enabled_services_cache is not None
            and (now - self._enabled_services_cache_time) < ENABLED_SERVICES_CACHE_TTL
        ):
            return self._enabled_services_cache

        enabled = []
        # Always include workspace-manager as it's the core service
        enabled.append("workspace-manager")

        # Check registered public services
        if hasattr(self.store, '_public_services'):
            for svc in self.store._public_services:
                svc_id = svc.id if hasattr(svc, 'id') else str(svc)
                # Normalize service ID (remove workspace prefix if present)
                if '/' in svc_id:
                    svc_id = svc_id.split('/')[-1]
                if ':' in svc_id:
                    svc_id = svc_id.split(':')[-1]
                enabled.append(svc_id)

        self._enabled_services_cache = enabled
        self._enabled_services_cache_time = now
        return enabled

    def is_service_enabled(self, service_name: str) -> bool:
        """Check if a service is enabled."""
        enabled = self.get_enabled_services()
        # Check various forms of the service name
        normalized = service_name.lower().replace('_', '-')
        return any(
            normalized in s.lower() or s.lower() in normalized
            for s in enabled
        )

    async def get_service_api(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get a service API definition dynamically.

        Only returns API for services that are enabled on this server.
        """
        # Map service names to their extractors and check functions
        service_map = {
            "workspace-manager": (self._get_workspace_manager_api, lambda: True),  # Always enabled
            "artifact-manager": (self._get_artifact_manager_api, self._is_artifact_manager_enabled),
            "server-apps": (self._get_server_apps_api, self._is_server_apps_enabled),
            "s3-storage": (self._get_s3_storage_api, self._is_s3_enabled),
            "queue": (self._get_queue_api, self._is_queue_enabled),
            "triton-client": (self._get_triton_api, self._is_triton_enabled),
        }

        if service_name not in service_map:
            return None

        getter, is_enabled = service_map[service_name]
        if not is_enabled():
            return None

        return await getter()

    def _is_artifact_manager_enabled(self) -> bool:
        """Check if artifact manager is enabled."""
        return hasattr(self.store, '_artifact_manager') and self.store._artifact_manager is not None

    def _is_server_apps_enabled(self) -> bool:
        """Check if server apps are enabled."""
        return self.is_service_enabled("server-apps")

    def _is_s3_enabled(self) -> bool:
        """Check if S3 storage is enabled."""
        return hasattr(self.store, '_s3_controller') and self.store._s3_controller is not None

    def _is_queue_enabled(self) -> bool:
        """Check if queue service is enabled."""
        return self.is_service_enabled("queue")

    def _is_triton_enabled(self) -> bool:
        """Check if Triton inference is enabled."""
        return self.is_service_enabled("triton")

    async def get_all_enabled_services(self) -> List[Dict[str, Any]]:
        """Get API documentation for all enabled services."""
        services = []
        service_names = [
            "workspace-manager",
            "artifact-manager",
            "server-apps",
            "s3-storage",
            "queue",
            "triton-client",
        ]

        for name in service_names:
            api = await self.get_service_api(name)
            if api:
                services.append(api)

        return services

    def _extract_methods_from_class(self, cls) -> Dict[str, Any]:
        """Extract schema methods from a class."""
        methods = {}
        for name in dir(cls):
            if name.startswith("_"):
                continue
            attr = getattr(cls, name, None)
            if callable(attr) and hasattr(attr, "__schema__"):
                methods[name] = {
                    "name": name,
                    "schema": attr.__schema__,
                    "docstring": extract_docstring(attr)
                }
        return methods

    async def _get_workspace_manager_api(self) -> Dict[str, Any]:
        """Get workspace manager API definition."""
        from hypha.core.workspace import WorkspaceManager

        return {
            "id": "workspace-manager",
            "name": "Workspace Manager",
            "description": "Core workspace operations including service management, authentication, and workspace lifecycle. Access via `server.get_service('~')` or use workspace manager methods directly on the server object.",
            "methods": self._extract_methods_from_class(WorkspaceManager)
        }

    async def _get_artifact_manager_api(self) -> Dict[str, Any]:
        """Get artifact manager API definition."""
        from hypha.artifact import ArtifactController

        return {
            "id": "artifact-manager",
            "name": "Artifact Manager",
            "description": "Manage artifacts including files, datasets, models, and collections with version control. Access via `server.get_service('public/artifact-manager')`.",
            "methods": self._extract_methods_from_class(ArtifactController)
        }

    async def _get_server_apps_api(self) -> Dict[str, Any]:
        """Get server apps API definition."""
        from hypha.apps import ServerAppController

        return {
            "id": "server-apps",
            "name": "Server Apps",
            "description": "Deploy and manage serverless applications and workers. Access via `server.get_service('public/server-apps')`.",
            "methods": self._extract_methods_from_class(ServerAppController)
        }

    async def _get_s3_storage_api(self) -> Dict[str, Any]:
        """Get S3 storage API definition."""
        try:
            from hypha.s3 import S3Controller
            return {
                "id": "s3-storage",
                "name": "S3 Storage",
                "description": "Direct S3-compatible object storage operations. Access via `server.get_service('public/s3-storage')`. Note: For most use cases, prefer the Artifact Manager which provides higher-level file management.",
                "methods": self._extract_methods_from_class(S3Controller)
            }
        except ImportError:
            return None

    async def _get_queue_api(self) -> Dict[str, Any]:
        """Get queue service API definition."""
        try:
            from hypha.queue import QueueService
            return {
                "id": "queue",
                "name": "Task Queue",
                "description": "Distributed task queue for asynchronous job processing. Access via `server.get_service('public/queue')`.",
                "methods": self._extract_methods_from_class(QueueService)
            }
        except ImportError:
            return None

    async def _get_triton_api(self) -> Dict[str, Any]:
        """Get Triton inference API definition."""
        try:
            from hypha.triton import TritonProxy
            return {
                "id": "triton-client",
                "name": "Triton Inference Client",
                "description": "ML model inference via NVIDIA Triton Inference Server. Access via `server.get_service('public/triton-client')`.",
                "methods": self._extract_methods_from_class(TritonProxy)
            }
        except ImportError:
            return None

    def get_source_code(self, module_path: str, symbol_name: str = None) -> Optional[str]:
        """Get source code for a module or specific symbol.

        Args:
            module_path: Python module path (e.g., 'hypha.core.workspace')
            symbol_name: Optional symbol name within the module (e.g., 'WorkspaceManager')

        Returns:
            Source code as string, or None if not found
        """
        try:
            import importlib

            # Only allow access to hypha modules for security
            if not module_path.startswith('hypha.'):
                return None

            module = importlib.import_module(module_path)

            if symbol_name:
                # Get specific symbol
                if hasattr(module, symbol_name):
                    obj = getattr(module, symbol_name)
                    return inspect.getsource(obj)
            else:
                # Get entire module source
                return inspect.getsource(module)
        except Exception as e:
            logger.warning(f"Could not get source code for {module_path}: {e}")
            return None

    def get_method_source(self, service_class: str, method_name: str) -> Optional[str]:
        """Get source code for a specific method.

        Args:
            service_class: Service class name (e.g., 'WorkspaceManager', 'ArtifactController')
            method_name: Method name to get source for

        Returns:
            Source code as string, or None if not found
        """
        # Build class name -> module path mapping from SERVICE_CLASS_MAP
        class_map = {
            class_name: (module_path, class_name)
            for module_path, class_name in SERVICE_CLASS_MAP.values()
        }

        if service_class not in class_map:
            return None

        module_path, class_name = class_map[service_class]

        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            method = getattr(cls, method_name, None)
            if method:
                return inspect.getsource(method)
        except Exception as e:
            logger.warning(f"Could not get source for {service_class}.{method_name}: {e}")

        return None

    def generate_api_reference(self, service_api: Dict[str, Any]) -> str:
        """Generate markdown API reference from service API definition."""
        lines = [f"## {service_api['name']}\n"]

        if service_api.get("description"):
            lines.append(f"{service_api['description']}\n")

        lines.append(f"\n**Service ID:** `{service_api['id']}`\n")

        methods = service_api.get("methods", {})
        if methods:
            for method_name, method_info in sorted(methods.items()):
                schema = method_info.get("schema", {})
                docstring = method_info.get("docstring", "")
                lines.append("\n---\n")
                lines.append(format_schema_as_markdown(schema, method_name, docstring))

        return "".join(lines)


def get_skill_md(workspace: str, server_url: str, workspace_info: dict = None, services: list = None, public_services: list = None) -> str:
    """Generate the main SKILL.md content for a workspace."""
    workspace_info = workspace_info or {}
    services = services or []
    public_services = public_services or []
    workspace_name = workspace_info.get("name", workspace)
    workspace_desc = workspace_info.get("description", "A Hypha workspace")
    is_persistent = workspace_info.get("persistent", False)

    # Load quick start documentation
    quick_start_doc = load_documentation_file("quick-start.md") or ""

    # Format available services (with reference links for those with schemas)
    service_list = ""
    for svc in services[:20]:  # Limit to 20 services for readability
        svc_id = svc.get("id", "unknown")
        svc_name = svc.get("name", "")
        svc_desc = (svc.get("description") or "")[:100]
        svc_slug = _normalize_service_slug(svc_id)
        service_list += f"- `{svc_id}`: {svc_name}"
        if svc_desc:
            service_list += f" - {svc_desc}"
        if _service_has_schema(svc):
            service_list += f" | [Reference](REFERENCE/{svc_slug}.md)"
        service_list += "\n"

    # Format public services (only those with schemas, accessible from any workspace)
    public_service_list = ""
    if public_services:
        schema_services = [s for s in public_services if _service_has_schema(s)]
        for svc in schema_services[:20]:
            svc_slug = _normalize_service_slug(svc.get("id", ""))
            svc_name = svc.get("name") or svc_slug
            svc_desc = (svc.get("description") or "")[:100]
            public_service_list += f"- `{svc_slug}`: {svc_name}"
            if svc_desc:
                public_service_list += f" - {svc_desc}"
            public_service_list += f" | [Reference](REFERENCE/{svc_slug}.md)\n"

    return f"""---
name: {SKILL_NAME}
description: Access and manage Hypha workspace services, artifacts, and applications. Use this skill when users need to interact with the Hypha distributed computing platform, manage files and data, deploy serverless applications, or access remote services.
license: BSD-3-Clause
compatibility: Requires network access to Hypha server. Works with Python 3.8+ or modern JavaScript environments.
metadata:
  version: {SKILL_VERSION}
  workspace: {workspace}
  server_url: {server_url}
---

# Hypha Workspace Manager

This skill enables you to interact with the Hypha distributed computing platform for workspace **{workspace_name}**.

## Workspace Context

- **Workspace ID**: `{workspace}`
- **Description**: {workspace_desc}
- **Persistent Storage**: {"Yes" if is_persistent else "No (temporary workspace)"}
- **Server URL**: `{server_url}`

## Step 1: Install Dependencies

### Python

```bash
pip install hypha-rpc
```

This installs the `hypha-rpc` package which provides `connect_to_server`, `login`, and other utilities.

### JavaScript / Node.js

```bash
npm install hypha-rpc
```

### HTTP Only (No Library Needed)

You can use Hypha entirely via HTTP REST endpoints with `curl`, `fetch`, or any HTTP client. No library installation required. See the HTTP API section below.

## Step 2: Authentication

Hypha supports three authentication modes. Choose the one that fits your situation:

### Option A: Anonymous Access (No Token Needed)

Connect without a token to get a temporary anonymous workspace. Ideal for quick testing and public services.

```python
import asyncio
from hypha_rpc import connect_to_server

async def main():
    async with connect_to_server({{
        "server_url": "{server_url}"
    }}) as server:
        # You are connected with an auto-generated anonymous workspace
        print(f"Connected to: {{server.config.workspace}}")
        services = await server.list_services()
        print(f"Found {{len(services)}} services")

asyncio.run(main())
```

```bash
# HTTP: Anonymous access to public workspace services
curl "{server_url}/public/services"
```

### Option B: Token-Based Access (Recommended for Agents)

If you already have a workspace token (e.g., provided by a user or generated via the API), use it directly:

```python
async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}",
    "token": "YOUR_TOKEN"
}}) as server:
    # Authenticated access with the token's permission level
    status = await server.check_status()
    print(f"Permission: {{status.get('user_permission')}}")
```

```bash
# HTTP: Authenticated access with Bearer token
curl -H "Authorization: Bearer YOUR_TOKEN" \\
  "{server_url}/{workspace}/services"
```

### Option C: Interactive Login (Browser-Based OAuth)

For interactive sessions where the user can open a browser:

```python
from hypha_rpc import login, connect_to_server

async def main():
    # Opens a browser window for OAuth login
    token = await login({{"server_url": "{server_url}"}})

    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": token
    }}) as server:
        # Now authenticated as the logged-in user
        # Generate a reusable token for future sessions
        workspace_token = await server.generate_token({{
            "permission": "read_write",
            "expires_in": 86400  # 24 hours
        }})
        print(f"Save this token for later: {{workspace_token}}")

asyncio.run(main())
```

### Generating Tokens for Programmatic Access

Once authenticated, generate tokens for use by scripts or AI agents:

```python
# Generate a read-only token (safe to share)
read_token = await server.generate_token({{
    "permission": "read",
    "expires_in": 86400  # 24 hours
}})

# Generate a read-write token (for data modification)
rw_token = await server.generate_token({{
    "permission": "read_write",
    "expires_in": 3600  # 1 hour
}})
```

Permission levels: `read` (view only), `read_write` (modify data), `admin` (full control including workspace deletion and token generation).

### How Authorization Works (Context Injection)

Hypha uses a **context injection** mechanism for authorization. When a service sets `require_context: True` in its config, the Hypha server automatically injects a `context` dictionary into every service method call. This context contains the caller's identity, workspace, and permissions — derived from the authentication token used during connection.

**Key points for API consumers:**
- You do **NOT** pass `context` when calling service methods — the server injects it automatically based on your token.
- The `context` parameter is intentionally omitted from the API reference documentation for this reason.
- Each service method receives the caller's identity and can enforce fine-grained permission checks server-side.

**What happens internally:**
1. You connect with a token (via WebSocket SDK or HTTP `Authorization: Bearer` header).
2. The server validates the token and associates your connection with a `UserInfo` (identity, roles, permissions).
3. When you call a service method, the server appends `context` (containing `ws`, `from`, `user`) before forwarding to the service.
4. The service method can check `context["user"]` permissions to authorize the operation.

This means **the permission level of your token determines what you can do** — you don't need to handle authorization yourself when calling services, but you may get `PermissionError` if your token lacks the required permission.

## Step 3: Connect and Use Services

### Python (Recommended)

```python
from hypha_rpc import connect_to_server

async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}"
}}) as server:
    # List available services
    services = await server.list_services()

    # Get a specific service
    svc = await server.get_service("service-id")

    # Call a service function
    result = await svc.some_function(param1="value")
```

### JavaScript

```javascript
const {{ connectToServer }} = require('hypha-rpc');

const server = await connectToServer({{
    server_url: "{server_url}",
    workspace: "{workspace}"
}});

const services = await server.listServices();
const svc = await server.getService("service-id");
const result = await svc.someFunction({{ param1: "value" }});
```

### HTTP API (curl / fetch / any HTTP client)

```bash
# List all services
curl "{server_url}/{workspace}/services"

# Get service details
curl "{server_url}/{workspace}/services/{{service_id}}"

# Call a service function (GET for simple params)
curl "{server_url}/{workspace}/services/{{service_id}}/{{function_name}}?param1=value"

# Call a service function (POST for complex params)
curl -X POST "{server_url}/{workspace}/services/{{service_id}}/{{function_name}}" \\
  -H "Content-Type: application/json" \\
  -d '{{"param1": "value1", "param2": 42}}'

# With authentication
curl -H "Authorization: Bearer YOUR_TOKEN" \\
  "{server_url}/{workspace}/services/{{service_id}}/{{function_name}}"
```

## Core Capabilities

### 1. Service Management
- List, discover, and call remote services
- Register new services with custom functions
- Search services using semantic similarity (vector search)

### 2. Token & Permission Management
- Generate workspace access tokens (`read`, `read_write`, `admin`)
- Parse and validate tokens for permission checking
- Revoke tokens and manage access scopes

### 3. Artifact Management
{"- Create, read, edit, and delete artifacts (files, datasets, models)" if is_persistent else "- Limited artifact operations (workspace is not persistent)"}
{"- Version control with Git-like staging and commit" if is_persistent else ""}
{"- Vector search collections for semantic queries" if is_persistent else ""}
{"- File upload/download with presigned URLs" if is_persistent else ""}

### 4. S3 Storage (if enabled)
- Direct S3-compatible object storage operations
- Upload/download files with presigned URLs
- HTTP proxy endpoint: `{server_url}/{workspace}/files/`

### 5. Server Applications
- Install and manage serverless applications (web-worker, web-python, window, iframe)
- Deploy HTTP endpoints via ASGI (FastAPI) or serverless functions
- Access apps at: `{server_url}/{workspace}/apps/{{app_id}}/`

### 6. MCP Integration (if enabled)
- Expose Hypha services as MCP endpoints for AI tools (Claude, Cursor, etc.)
- MCP endpoint: `{server_url}/{workspace}/mcp/{{service_id}}/mcp`

### 7. A2A Protocol (if enabled)
- Agent-to-Agent protocol for multi-agent communication
- A2A endpoint: `{server_url}/{workspace}/a2a/{{service_id}}`

### 8. Vector Search
- Semantic search by text query or vector
- Search services by natural language description

## Available Services

{service_list if service_list else "Use `list_services()` or `GET /{workspace}/services` to discover all services."}

{"## Public Services (Available from Any Workspace)" + chr(10) + chr(10) + public_service_list if public_service_list else ""}

Key built-in services:

| Service | ID | Reference |
|---------|-----|-----------|
| Workspace Manager | `~` or `default` | [REFERENCE/workspace-manager.md](REFERENCE/workspace-manager.md) |
| Artifact Manager | `public/artifact-manager` | [REFERENCE/artifact-manager.md](REFERENCE/artifact-manager.md) |
| Server Apps | `public/server-apps` | [REFERENCE/server-apps.md](REFERENCE/server-apps.md) |
| S3 Storage | `public/s3-storage` | [REFERENCE/s3-storage.md](REFERENCE/s3-storage.md) |

## Instructions for AI Agents

1. **Bootstrap first** - Install `hypha-rpc` (`pip install hypha-rpc`), then connect
2. **Check workspace status** - `await server.check_status()` to verify workspace is ready
3. **Discover services** - `await server.list_services()` to see what's available
4. **Read service docs** - Check `REFERENCE/{{service-id}}.md` for the specific service you need
5. **Choose transport** - Use Python/JS SDK for bidirectional real-time calls; use HTTP for simple stateless calls
6. **Handle permissions** - Check user has required permissions before operations
7. **HTTP fallback** - If you can't install libraries, all services work via HTTP REST endpoints

## Error Handling

Common errors and solutions:

- `PermissionError`: User lacks required permission level. Generate a new token with higher permission.
- `KeyError: Workspace not found`: Workspace doesn't exist or was unloaded due to inactivity.
- `KeyError: Service not found`: Service not registered. Try `list_services()` first, or start the app.
- `TimeoutError`: Service didn't respond in time. Increase timeout or check if the service is running.

## Documentation URLs

### Generic Documentation (No Auth Required)

These URLs provide general Hypha documentation accessible without authentication:

- **Overview**: `{server_url}/ws/agent-skills/SKILL.md`
- **API Reference**: `{server_url}/ws/agent-skills/REFERENCE.md`
- **Per-Service Reference**: `{server_url}/ws/agent-skills/REFERENCE/{{service-id}}.md`
- **Examples**: `{server_url}/ws/agent-skills/EXAMPLES.md`

### Workspace-Specific Documentation (Auth Required)

For documentation tailored to a specific workspace (includes workspace services, context, and capabilities):

- **Overview**: `{server_url}/{{workspace}}/agent-skills/SKILL.md`
- **API Reference**: `{server_url}/{{workspace}}/agent-skills/REFERENCE.md`
- **Per-Service Reference**: `{server_url}/{{workspace}}/agent-skills/REFERENCE/{{service-id}}.md`
- **Examples**: `{server_url}/{{workspace}}/agent-skills/EXAMPLES.md`
- **Workspace Context**: `{server_url}/{{workspace}}/agent-skills/WORKSPACE_CONTEXT.md`
- **Source Code**: `{server_url}/{{workspace}}/agent-skills/SOURCE/{{service-id}}/{{method-name}}`

Replace `{{workspace}}` with the target workspace ID and provide a Bearer token via the `Authorization` header.

## Reference Documentation

For detailed API documentation, see:
- [REFERENCE.md](REFERENCE.md) - API reference overview with links to per-service docs
- [REFERENCE/workspace-manager.md](REFERENCE/workspace-manager.md) - Workspace Manager full API
- [REFERENCE/artifact-manager.md](REFERENCE/artifact-manager.md) - Artifact Manager full API
- [REFERENCE/server-apps.md](REFERENCE/server-apps.md) - Server Apps full API
- [REFERENCE/s3-storage.md](REFERENCE/s3-storage.md) - S3 Storage full API
- [EXAMPLES.md](EXAMPLES.md) - Code examples for every feature
- [WORKSPACE_CONTEXT.md](WORKSPACE_CONTEXT.md) - This workspace's specific configuration
"""


async def get_service_reference_md(
    service_id: str, workspace: str, server_url: str, doc_generator: DynamicDocGenerator
) -> Optional[str]:
    """Generate a standalone reference document for a single service.

    Returns None if the service is not enabled.
    """
    service_api = await doc_generator.get_service_api(service_id)
    if not service_api:
        return None

    methods = service_api.get("methods", {})
    method_toc = "\n".join(f"- [`{name}`](#{name})" for name in sorted(methods.keys()))

    lines = [f"""# {service_api['name']} API Reference

**Service ID:** `{service_api['id']}`
"""]

    if service_api.get("description"):
        lines.append(f"\n{service_api['description']}\n")

    lines.append(f"""
## Quick Access

```python
from hypha_rpc import connect_to_server

async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}",
    "token": "YOUR_TOKEN"
}}) as server:
    svc = await server.get_service("{service_api['id']}")
    # Call methods on svc...
```

```bash
# HTTP: Call a method
curl -X POST "{server_url}/{workspace}/services/{service_api['id']}/METHOD_NAME" \\\\
  -H "Content-Type: application/json" \\\\
  -H "Authorization: Bearer YOUR_TOKEN" \\\\
  -d '{{"param1": "value"}}'
```

## Methods

{method_toc}

---

""")

    for method_name, method_info in sorted(methods.items()):
        schema = method_info.get("schema", {})
        docstring = method_info.get("docstring", "")
        lines.append(format_schema_as_markdown(schema, method_name, docstring))
        lines.append("\n---\n\n")

    lines.append(f"""
## Source Code

To inspect the implementation of any method, use:

```
GET {server_url}/{workspace}/agent-skills/SOURCE/{service_id}/METHOD_NAME
```

[Back to API Reference Overview](REFERENCE.md)
""")

    return "".join(lines)


def generate_reference_from_service_schema(
    service_dict: dict, workspace: str, server_url: str
) -> Optional[str]:
    """Generate reference documentation from a service's service_schema.

    This handles dynamically registered services that have service_schema
    but are not built-in services managed by DynamicDocGenerator.

    Args:
        service_dict: Service dict from list_services() containing service_schema
        workspace: Current workspace ID
        server_url: Server URL for code examples

    Returns:
        Markdown string, or None if service has no valid schema
    """
    service_schema = service_dict.get("service_schema")
    if not service_schema or not isinstance(service_schema, dict):
        return None

    service_id = service_dict.get("id", "unknown")
    service_name = service_dict.get("name") or service_id
    service_desc = service_dict.get("description") or ""
    slug = _normalize_service_slug(service_id)

    # Extract methods from service_schema format
    # Format: {"method": {"type": "function", "function": {"description": ..., "parameters": ...}}}
    # Skip methods without meaningful schemas (no description and no parameters)
    methods = {}
    for method_name, method_def in service_schema.items():
        if not isinstance(method_def, dict):
            continue
        if method_def.get("type") != "function":
            continue
        function_def = method_def.get("function")
        if not function_def or not isinstance(function_def, dict):
            continue
        has_description = bool(function_def.get("description", "").strip())
        has_params = bool(function_def.get("parameters", {}).get("properties", {}))
        if not has_description and not has_params:
            continue
        # Convert to format expected by format_schema_as_markdown
        schema = {
            "description": function_def.get("description", ""),
            "parameters": function_def.get("parameters", {}),
        }
        methods[method_name] = {
            "name": method_name,
            "schema": schema,
            "docstring": function_def.get("description", ""),
        }

    if not methods:
        return None

    method_toc = "\n".join(
        f"- [`{name}`](#{name})" for name in sorted(methods.keys())
    )

    lines = [
        f"# {service_name} API Reference\n\n**Service ID:** `{service_id}`\n"
    ]

    if service_desc:
        lines.append(f"\n{service_desc}\n")

    lines.append(f"""
## Quick Access

```python
from hypha_rpc import connect_to_server

async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}",
    "token": "YOUR_TOKEN"
}}) as server:
    svc = await server.get_service("{slug}")
    # Call methods on svc...
```

```bash
# HTTP: Call a method
curl -X POST "{server_url}/{workspace}/services/{slug}/METHOD_NAME" \\\\
  -H "Content-Type: application/json" \\\\
  -H "Authorization: Bearer YOUR_TOKEN" \\\\
  -d '{{"param1": "value"}}'
```

## Methods

{method_toc}

---

""")

    for method_name, method_info in sorted(methods.items()):
        schema = method_info.get("schema", {})
        docstring = method_info.get("docstring", "")
        lines.append(format_schema_as_markdown(schema, method_name, docstring))
        lines.append("\n---\n\n")

    lines.append(f"\n[Back to API Reference Overview](REFERENCE.md)\n")

    return "".join(lines)


async def get_reference_md(
    workspace: str, server_url: str, doc_generator: DynamicDocGenerator,
    services: list = None, public_services: list = None,
) -> str:
    """Generate API reference overview with links to per-service reference docs.

    This function generates a concise overview that links to individual service
    reference documents at REFERENCE/<service-id>.md.
    """
    services = services or []
    public_services = public_services or []

    # Get all enabled built-in services dynamically
    enabled_services = await doc_generator.get_all_enabled_services()

    # Build service table with links to per-service docs
    service_rows = []
    for svc in enabled_services:
        svc_id = svc["id"]
        svc_name = svc["name"]
        svc_desc = (svc.get("description") or "")[:80]
        method_count = len(svc.get("methods", {}))
        service_rows.append(
            f"| [{svc_name}](REFERENCE/{svc_id}.md) | `{svc_id}` | {svc_desc} | {method_count} |"
        )
    service_table = "\n".join(service_rows)

    # Build dynamic services tables (workspace + public)
    builtin_ids = {svc["id"] for svc in enabled_services}
    ws_service_rows = []
    pub_service_rows = []
    for svc_list, target_rows, source_label in [
        (services, ws_service_rows, "workspace"),
        (public_services, pub_service_rows, "public"),
    ]:
        for svc in svc_list:
            if not _service_has_schema(svc):
                continue
            slug = _normalize_service_slug(svc.get("id", ""))
            if slug in builtin_ids:
                continue
            svc_name = svc.get("name") or slug
            svc_desc = (svc.get("description") or "")[:80]
            schema = svc.get("service_schema", {})
            method_count = sum(
                1 for m in schema.values()
                if isinstance(m, dict) and m.get("type") == "function"
            )
            target_rows.append(
                f"| [{svc_name}](REFERENCE/{slug}.md) | `{slug}` | {svc_desc} | {method_count} |"
            )

    enabled_list = "\n".join(f"- {svc['name']} (`{svc['id']}`)" for svc in enabled_services)

    lines = [f"""# Hypha API Reference

Complete API reference for workspace `{workspace}` at `{server_url}`.

This documentation is automatically generated from the service API schemas.
Each service has its own detailed reference document with full method signatures and descriptions.

## Built-in Services

| Service | ID | Description | Methods |
|---------|-----|-------------|---------|
{service_table}

Click on any service name above to view its full API reference.

"""]

    # Add workspace services section if any have schemas
    if ws_service_rows:
        ws_table = "\n".join(ws_service_rows)
        lines.append(f"""## Workspace Services

| Service | ID | Description | Methods |
|---------|-----|-------------|---------|
{ws_table}

""")

    # Add public services section if any have schemas
    if pub_service_rows:
        pub_table = "\n".join(pub_service_rows)
        lines.append(f"""## Public Services (Available from Any Workspace)

| Service | ID | Description | Methods |
|---------|-----|-------------|---------|
{pub_table}

""")

    lines.append(f"""## Connection Methods

### Python (Async)

```python
from hypha_rpc import connect_to_server

async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}",
    "token": "YOUR_TOKEN"  # Optional for authenticated access
}}) as server:
    # Use server API here
    pass
```

### JavaScript

```javascript
const {{ connectToServer }} = require('hypha-rpc');

const server = await connectToServer({{
    server_url: "{server_url}",
    workspace: "{workspace}"
}});
```

### HTTP API

Base URL: `{server_url}/{workspace}`

---

""")

    # Generate a brief summary for each built-in service
    for service_api in enabled_services:
        svc_id = service_api["id"]
        svc_name = service_api["name"]
        svc_desc = service_api.get("description", "")
        methods = service_api.get("methods", {})
        method_list = ", ".join(f"`{m}`" for m in sorted(methods.keys())[:10])
        if len(methods) > 10:
            method_list += f", ... ({len(methods)} total)"
        lines.append(f"""## {svc_name}

**Service ID:** `{svc_id}` | **[Full Reference](REFERENCE/{svc_id}.md)**

{svc_desc}

**Key methods:** {method_list}

""")

    # Add HTTP endpoints reference
    lines.append("""---

## HTTP Endpoints Reference

### Workspace Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/{workspace}/info` | GET | Workspace information |
| `/{workspace}/services` | GET | List all services |
| `/{workspace}/services/{service_id}` | GET | Service details |
| `/{workspace}/services/{service_id}/{function}` | GET/POST | Call function |

### App Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/{workspace}/apps/{service_id}/...` | * | ASGI/Functions app |
| `/{workspace}/mcp/{service_id}/mcp` | POST | MCP protocol endpoint |

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Full health check |
| `/health/readiness` | GET | Readiness probe |
| `/health/liveness` | GET | Liveness probe |
| `/metrics` | GET | Prometheus metrics |

---

## Permission Levels

- **read**: View services and data
- **read_write**: Modify services and data
- **admin**: Full control including workspace deletion

---

## Error Codes

| Error | Description | Solution |
|-------|-------------|----------|
| `PermissionError` | Insufficient permissions | Request higher permission level |
| `KeyError` | Resource not found | Check ID exists |
| `TimeoutError` | Operation timed out | Increase timeout |
| `ValueError` | Invalid parameter | Check parameter format |
""")

    return "".join(lines)


def get_examples_md(workspace: str, server_url: str) -> str:
    """Generate code examples for common operations."""
    # Load examples from documentation files
    getting_started = load_documentation_file("getting-started.md") or ""
    artifact_doc = load_documentation_file("artifact-manager.md") or ""
    apps_doc = load_documentation_file("apps.md") or ""

    return f"""# Hypha Code Examples

Comprehensive examples for workspace `{workspace}`.

## Table of Contents

0. [Setup & Installation](#setup--installation)
1. [Connection & Authentication](#connection-examples)
2. [Service Management](#service-examples)
3. [Token & Permission Management](#token--permission-management)
4. [Artifact Management](#artifact-examples)
5. [Git Storage & Pull Requests](#git-storage--pull-request-examples)
6. [S3 Storage Operations](#s3-storage-examples)
7. [Server Applications](#server-app-examples)
8. [MCP Integration](#mcp-integration-examples)
9. [A2A Protocol](#a2a-agent-to-agent-examples)
10. [Vector Search](#vector-search-examples)
11. [Error Handling](#error-handling-examples)

---

## Setup & Installation

### Python

```bash
# Install the hypha-rpc client library
pip install hypha-rpc

# Verify installation
python -c "from hypha_rpc import connect_to_server; print('hypha-rpc installed successfully')"
```

### JavaScript / Node.js

```bash
npm install hypha-rpc
```

### HTTP Only (No Installation)

All Hypha services are accessible via REST HTTP endpoints. You can use `curl`, `fetch()`, `httpx`, `requests`, or any HTTP client:

```bash
# Test connection to the server
curl {server_url}/health/liveness

# List services in a workspace (no auth needed for public workspace)
curl {server_url}/public/services
```

---

## Connection Examples

### Anonymous Connection (No Token)

```python
import asyncio
from hypha_rpc import connect_to_server

async def main():
    # Connect without a token - gets a temporary anonymous workspace
    async with connect_to_server({{
        "server_url": "{server_url}"
    }}) as server:
        print(f"Connected to workspace: {{server.config.workspace}}")

        # List services
        services = await server.list_services()
        for svc in services:
            print(f"  - {{svc['id']}}: {{svc.get('name', 'unnamed')}}")

asyncio.run(main())
```

### Authenticated Connection

```python
from hypha_rpc import connect_to_server, login

async def main():
    # Login to get token
    token = await login({{"server_url": "{server_url}"}})

    async with connect_to_server({{
        "server_url": "{server_url}",
        "workspace": "{workspace}",
        "token": token
    }}) as server:
        # Generate workspace token for later use
        ws_token = await server.generate_token({{
            "permission": "read_write",
            "expires_in": 86400  # 24 hours
        }})
        print(f"Workspace token: {{ws_token}}")

asyncio.run(main())
```

---

## Service Examples

### Register a Service

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Define service functions
        def add(a, b):
            return a + b

        async def process_data(data):
            return {{"processed": data, "status": "success"}}

        # Register service
        service = await server.register_service({{
            "id": "my-calculator",
            "name": "Calculator Service",
            "config": {{
                "visibility": "public"  # or "protected" for workspace-only
            }},
            "add": add,
            "process": process_data
        }})

        print(f"Service registered: {{service.id}}")
        await server.serve()  # Keep running
```

### Use a Service

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Get service by ID
        calc = await server.get_service("my-calculator")

        # Call functions
        result = await calc.add(5, 3)
        print(f"5 + 3 = {{result}}")

        processed = await calc.process({{"key": "value"}})
        print(f"Processed: {{processed}}")
```

### Call Service via HTTP

```bash
# GET request
curl "{server_url}/{workspace}/services/my-calculator/add?a=5&b=3"

# POST request
curl -X POST "{server_url}/{workspace}/services/my-calculator/process" \\
  -H "Content-Type: application/json" \\
  -d '{{"data": {{"key": "value"}}}}'
```

---

## Token & Permission Management

### Generate Workspace Token

```python
async def main():
    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": "YOUR_AUTH_TOKEN"  # Must have admin permission
    }}) as server:
        # Generate a read-only token (expires in 24 hours)
        read_token = await server.generate_token({{
            "permission": "read",          # read, read_write, or admin
            "expires_in": 86400,           # 24 hours in seconds
            "description": "Read-only access for monitoring"
        }})
        print(f"Read token: {{read_token}}")

        # Generate a read-write token for service operations
        rw_token = await server.generate_token({{
            "permission": "read_write",
            "expires_in": 3600,            # 1 hour
            "extra_scopes": [              # Optional: limit to specific operations
                {{"workspace": "{workspace}", "resource": "artifact:my-dataset:*"}}
            ]
        }})
        print(f"RW token: {{rw_token}}")

        # Generate admin token for workspace management
        admin_token = await server.generate_token({{
            "permission": "admin",
            "expires_in": 600,             # Short-lived for security
            "description": "Temporary admin access"
        }})
```

### Parse and Validate Tokens

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Parse a token to inspect its permissions
        token_info = await server.parse_token("eyJ...")

        print(f"User ID: {{token_info.get('id')}}")
        print(f"Workspace: {{token_info.get('workspace')}}")
        print(f"Permission: {{token_info.get('permission')}}")
        print(f"Expires: {{token_info.get('expires_at')}}")
        print(f"Scopes: {{token_info.get('scopes')}}")

        # Check if token has specific permission
        if token_info.get("permission") in ["read_write", "admin"]:
            print("Token can modify data")
```

### Revoke Tokens

```python
async def main():
    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": "ADMIN_TOKEN"
    }}) as server:
        # Revoke a specific token
        await server.revoke_token("token_to_revoke")

        # Revoke all tokens for the workspace
        await server.revoke_all_tokens()
```

### Login Flow (Interactive)

```python
from hypha_rpc import login, connect_to_server

async def main():
    # Interactive login - opens browser for OAuth
    token = await login({{
        "server_url": "{server_url}",
        "login_timeout": 60  # Wait up to 60 seconds for user
    }})

    # Use the token for authenticated access
    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": token
    }}) as server:
        # Now have authenticated access with user's permissions
        info = await server.get_workspace_info()
        print(f"Logged in to: {{info.get('name')}}")
```

### Check User Permissions

```python
async def main():
    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": "YOUR_TOKEN"
    }}) as server:
        # Get current user info
        user = await server.get_user_info()
        print(f"User ID: {{user.get('id')}}")
        print(f"Email: {{user.get('email')}}")
        print(f"Roles: {{user.get('roles')}}")

        # Check workspace status and permissions
        status = await server.check_status()
        print(f"Workspace status: {{status.get('status')}}")
        print(f"User permission: {{status.get('user_permission')}}")
```

---

## Artifact Examples

### Create and Upload Files

```python
import httpx

async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Create a dataset artifact
        dataset = await artifact_manager.create(
            type="dataset",
            alias="my-dataset",
            manifest={{
                "name": "Training Data",
                "description": "Image training dataset",
                "tags": ["images", "training"]
            }}
        )
        print(f"Created artifact: {{dataset['id']}}")

        # Upload a file
        put_info = await artifact_manager.put_file(
            artifact_id=dataset["id"],
            file_path="images/sample.jpg"
        )

        # Actually upload the file
        async with httpx.AsyncClient() as client:
            with open("local_sample.jpg", "rb") as f:
                await client.put(put_info["put_url"], content=f.read())

        # Commit the changes
        await artifact_manager.commit(
            artifact_id=dataset["id"],
            message="Added sample image"
        )
```

### Download Files

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Get download URL
        file_info = await artifact_manager.get_file(
            artifact_id="{workspace}/my-dataset",
            file_path="images/sample.jpg"
        )

        # Download the file
        async with httpx.AsyncClient() as client:
            response = await client.get(file_info["get_url"])
            with open("downloaded_sample.jpg", "wb") as f:
                f.write(response.content)
```

### Read and Write Small Files Directly

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Write a small file directly
        await artifact_manager.write_file(
            artifact_id="{workspace}/my-dataset",
            path="config.json",
            content='{{"setting": "value"}}'
        )

        # Read the file content directly
        content = await artifact_manager.read_file(
            artifact_id="{workspace}/my-dataset",
            path="config.json"
        )
        print(f"Config: {{content}}")
```

---

## Git Storage & Pull Request Examples

Git-storage artifacts provide full Git repository functionality backed by S3. The PR system enables
multiple AI agents to collaborate on shared codebases through branches, diffs, reviews, and merges.

### Create a Git Repository and Work with Branches

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        svc = await server.get_service("public/artifact-manager")

        # Create a git-storage artifact
        await svc.create(
            alias="my-repo",
            manifest={{"name": "My Project"}},
            config={{"storage": "git"}},
        )

        # Push initial files
        await svc.edit("my-repo", stage=True)
        await svc.write_file("my-repo", "README.md", "# My Project\\n")
        await svc.write_file("my-repo", "src/main.py", "print('hello')\\n")
        await svc.commit("my-repo", comment="Initial commit")

        # Create a feature branch
        await svc.create_branch("my-repo", branch="feature-auth")

        # Write files to the feature branch
        await svc.edit("my-repo", stage=True, branch="feature-auth")
        await svc.write_file("my-repo", "src/auth.py", "import jwt\\n...")
        await svc.commit("my-repo", comment="Add auth module", branch="feature-auth")

        # List branches
        branches = await svc.list_branches("my-repo")
        for b in branches:
            print(f"  {{b['name']}} ({{b['sha'][:8]}}) {{'[default]' if b.get('default') else ''}}")
```

### Pull Request Workflow (Create, Diff, Review, Merge)

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        svc = await server.get_service("public/artifact-manager")

        # Create a PR from feature branch to main
        pr = await svc.create_pr(
            "my-repo",
            title="Add authentication module",
            source_branch="feature-auth",
            target_branch="main",
            description="JWT-based auth with refresh tokens.",
        )
        print(f"PR #{{pr['pr_number']}} created")

        # Get the structured diff
        diff = await svc.get_diff("my-repo", base="main", head="feature-auth", include_content=True)
        print(f"Files changed: {{diff['stats']['files_changed']}}")
        for f in diff["files"]:
            print(f"  {{f['status']:10s}} {{f['path']}}")

        # Submit a review
        await svc.submit_review("my-repo", pr["pr_number"], verdict="approve", comment="LGTM")

        # Merge the PR (auto strategy: fast-forward if possible, else 3-way merge)
        result = await svc.merge_pr("my-repo", pr["pr_number"])
        print(f"Merged via {{result['merge_type']}}: {{result['merge_sha']}}")
```

### Handle Merge Conflicts

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        svc = await server.get_service("public/artifact-manager")

        # Attempt merge
        result = await svc.merge_pr("my-repo", pr_number=3)

        if result["status"] == "conflict":
            for conflict in result["conflicts"]:
                # Read both versions
                ours = await svc.read_file("my-repo", conflict["path"], version="main")
                theirs = await svc.read_file("my-repo", conflict["path"], version="feature-x")

                # Resolve and push to source branch
                resolved = resolve_conflict(ours["content"], theirs["content"])
                await svc.edit("my-repo", stage=True, branch="feature-x")
                await svc.write_file("my-repo", conflict["path"], resolved)
                await svc.commit("my-repo", comment="Resolve conflict", branch="feature-x")

            # Retry merge
            result = await svc.merge_pr("my-repo", pr_number=3)
```

---

## Server App Examples

Server Apps enable serverless computing with applications that start on-demand and auto-scale.

### Install and Run a Web Worker App

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Install a simple calculator app (JavaScript web worker)
        app_source = '''
        api.export({{
            async add(a, b) {{ return a + b; }},
            async multiply(a, b) {{ return a * b; }}
        }});
        '''

        app_info = await controller.install(
            source=app_source,
            app_id="my-calculator",  # Optional custom ID
            manifest={{
                "name": "Calculator",
                "type": "web-worker",
                "version": "1.0.0"
            }},
            overwrite=True
        )

        # Start the app
        session = await controller.start(
            app_info["id"],
            wait_for_service="default",
            timeout=30
        )

        # Use the app service
        calc = await server.get_service(f"default@{{app_info['id']}}")
        result = await calc.add(10, 20)
        print(f"10 + 20 = {{result}}")

        # Stop when done
        await controller.stop(session["id"])
```

### Deploy a FastAPI/ASGI Application

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Deploy a FastAPI app as serverless ASGI service
        fastapi_source = '''
from hypha_rpc import api
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import datetime

def create_app():
    app = FastAPI(title="My API")

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return f"<h1>Hello! Server time: {{datetime.now()}}</h1>"

    @app.get("/api/add/{{a}}/{{b}}")
    async def add(a: int, b: int):
        return {{"result": a + b, "timestamp": datetime.now().isoformat()}}

    @app.get("/api/status")
    async def status():
        return {{"status": "healthy"}}

    return app

async def setup():
    fastapi_app = create_app()

    async def serve(args):
        await fastapi_app(args["scope"], args["receive"], args["send"])

    await api.register_service({{
        "id": "my-api",
        "type": "asgi",
        "serve": serve,
        "config": {{"visibility": "public"}}
    }}, {{"overwrite": True}})

api.export({{"setup": setup}})
'''

        app_info = await controller.install(
            source=fastapi_source,
            app_id="fastapi-demo",
            manifest={{
                "type": "web-python",
                "name": "FastAPI Demo",
                "version": "1.0.0",
                "requirements": ["fastapi"]
            }},
            overwrite=True
        )
        print(f"App installed: {{app_info['id']}}")

        # The app is now accessible at:
        # {server_url}/{workspace}/apps/my-api/
        # {server_url}/{workspace}/apps/my-api/api/add/5/3
```

### Create Serverless HTTP Functions

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Deploy serverless functions with automatic routing
        functions_source = '''
api.register_service({{
    "id": "user-api",
    "type": "functions",
    "users": {{
        "list": async (event) => {{
            return {{ users: ["alice", "bob", "charlie"] }};
        }},
        "get": async (event) => {{
            const id = event.path_params?.id || event.query_params?.id;
            return {{ user: id, name: "User " + id }};
        }},
        "create": async (event) => {{
            const data = event.body;
            return {{ message: "User created", data: data }};
        }}
    }},
    "health": async (event) => {{ return {{ status: "ok" }}; }},
    "default": async (event) => {{
        return {{ error: "Not found", path: event.function_path }};
    }}
}});
'''

        await controller.install(
            source=functions_source,
            app_id="user-api",
            manifest={{
                "type": "web-worker",
                "name": "User API",
                "version": "1.0.0"
            }},
            overwrite=True
        )

        # Access via HTTP:
        # GET {server_url}/{workspace}/apps/user-api/users/list
        # GET {server_url}/{workspace}/apps/user-api/users/get?id=123
        # POST {server_url}/{workspace}/apps/user-api/users/create
        # GET {server_url}/{workspace}/apps/user-api/health
```

### Manage App Lifecycle

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # List all installed apps
        apps = await controller.list_apps()
        print(f"Installed apps: {{len(apps)}}")
        for app in apps:
            print(f"  - {{app.get('id')}}: {{app.get('name')}}")

        # List running app sessions
        running = await controller.list_running()
        print(f"Running sessions: {{len(running)}}")
        for session in running:
            print(f"  - {{session.get('id')}} (app: {{session.get('app_id')}})")

        # Stop a running session
        if running:
            await controller.stop(running[0]["id"])

        # Uninstall an app (removes all files and metadata)
        # await controller.uninstall("my-old-app")
```

### Web Python App with Artifact Access

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Web Python app that accesses its artifact files
        python_app = '''
from hypha_rpc import api
import os
from hypha_artifact import AsyncHyphaArtifact

async def setup():
    # Access environment variables provided by Hypha
    artifact = AsyncHyphaArtifact(
        server_url=os.environ.get("HYPHA_SERVER_URL"),
        artifact_id=os.environ.get("HYPHA_APP_ID"),
        workspace=os.environ.get("HYPHA_WORKSPACE"),
        token=os.environ.get("HYPHA_TOKEN")
    )

    # List files in the app's artifact
    files = await artifact.ls()
    print(f"App files: {{files}}")

async def process_data(data):
    import numpy as np
    arr = np.array(data)
    return {{
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sum": float(np.sum(arr))
    }}

api.export({{"setup": setup, "process_data": process_data}})
'''

        app_info = await controller.install(
            source=python_app,
            app_id="data-processor",
            manifest={{
                "type": "web-python",
                "name": "Data Processor",
                "version": "1.0.0",
                "requirements": ["numpy", "hypha-artifact"]
            }},
            overwrite=True
        )

        # Start and use the app
        session = await controller.start(app_info["id"])
        svc = await server.get_service(f"default@{{app_info['id']}}")
        result = await svc.process_data([1, 2, 3, 4, 5])
        print(f"Statistics: {{result}}")
```

### Manage App Files

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # List files in an installed app
        files = await controller.list_files("my-app")
        print(f"App files: {{files}}")

        # Edit a file in the app (creates new staged version)
        await controller.edit_file(
            app_id="my-app",
            file_path="config.json",
            content='{{"debug": true, "version": "2.0"}}'
        )

        # Commit changes (creates new version)
        await controller.commit_app(
            app_id="my-app",
            message="Updated configuration"
        )
```

### Application Types Reference

| Type | Description | Use Case |
|------|-------------|----------|
| `web-worker` | JavaScript in Web Worker | CPU-intensive JS tasks |
| `web-python` | Python via Pyodide | Data processing, ML |
| `window` | Full browser context | DOM manipulation |
| `iframe` | Isolated iframe | Legacy/untrusted apps |
| Custom types | Via custom workers | Specialized runtimes |

---

## S3 Storage Examples

S3 storage provides direct object storage operations. Enable with `--enable-s3` or `--start-minio-server`.

### Upload and Download Files

```python
import httpx

async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        s3 = await server.get_service("public/s3-storage")

        # Upload a file - get presigned URL
        put_info = await s3.put_file(
            workspace="{workspace}",
            path="data/sample.json",
            options={{"content_type": "application/json"}}
        )

        # Upload using the presigned URL
        async with httpx.AsyncClient() as client:
            content = '{{"key": "value", "data": [1, 2, 3]}}'
            await client.put(put_info["put_url"], content=content)

        print(f"Uploaded to: {{put_info['path']}}")

        # Download a file - get presigned URL
        get_info = await s3.get_file(
            workspace="{workspace}",
            path="data/sample.json"
        )

        # Download using the presigned URL
        async with httpx.AsyncClient() as client:
            response = await client.get(get_info["get_url"])
            print(f"Downloaded: {{response.text}}")
```

### List and Remove Files

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        s3 = await server.get_service("public/s3-storage")

        # List files in a directory
        files = await s3.list_files(
            workspace="{workspace}",
            path="data/",
            max_keys=100
        )

        print(f"Found {{len(files.get('items', []))}} files:")
        for item in files.get("items", []):
            print(f"  - {{item['name']}} ({{item.get('size', 0)}} bytes)")

        # Remove a file
        await s3.remove_file(
            workspace="{workspace}",
            path="data/old-file.json"
        )
```

### Multipart Upload for Large Files

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        s3 = await server.get_service("public/s3-storage")

        # Start multipart upload
        upload_info = await s3.put_file_start_multipart(
            workspace="{workspace}",
            path="data/large-file.bin",
            options={{"content_type": "application/octet-stream"}}
        )

        upload_id = upload_info["upload_id"]
        parts = []

        # Upload parts (5MB+ each, except last)
        async with httpx.AsyncClient() as client:
            for part_number in range(1, 4):
                # Get presigned URL for this part
                part_info = await s3.put_file_upload_part(
                    workspace="{workspace}",
                    path="data/large-file.bin",
                    upload_id=upload_id,
                    part_number=part_number
                )

                # Upload the part
                chunk = b"x" * (5 * 1024 * 1024)  # 5MB chunk
                response = await client.put(part_info["put_url"], content=chunk)

                # Record the ETag for completion
                parts.append({{
                    "PartNumber": part_number,
                    "ETag": response.headers["ETag"]
                }})

        # Complete the upload
        await s3.put_file_complete_multipart(
            workspace="{workspace}",
            path="data/large-file.bin",
            upload_id=upload_id,
            parts=parts
        )
```

### HTTP Proxy Access

```bash
# Files are also accessible via HTTP proxy endpoint:
# Download
curl "{server_url}/{workspace}/files/data/sample.json"

# Upload with token
curl -X PUT "{server_url}/{workspace}/files/data/new-file.json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"key": "value"}}'
```

---

## MCP Integration Examples

MCP (Model Context Protocol) allows exposing Hypha services as MCP endpoints for AI tools. Enable with `--enable-mcp`.

### Register an MCP Service

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Register a service as MCP endpoint
        await server.register_service({{
            "id": "my-tools",
            "name": "My AI Tools",
            "type": "mcp",  # This enables MCP protocol
            "config": {{"visibility": "public"}},

            # Tools - callable functions for AI
            "tools": {{
                "search_documents": {{
                    "description": "Search through documents",
                    "parameters": {{
                        "type": "object",
                        "properties": {{
                            "query": {{"type": "string", "description": "Search query"}},
                            "limit": {{"type": "integer", "default": 10}}
                        }},
                        "required": ["query"]
                    }},
                    "handler": lambda query, limit=10: search_docs(query, limit)
                }},
                "get_weather": {{
                    "description": "Get current weather for a location",
                    "parameters": {{
                        "type": "object",
                        "properties": {{
                            "city": {{"type": "string", "description": "City name"}}
                        }},
                        "required": ["city"]
                    }},
                    "handler": lambda city: get_weather(city)
                }}
            }},

            # Resources - data sources AI can read
            "resources": {{
                "config": {{
                    "uri": "config://settings",
                    "name": "Configuration",
                    "description": "Current system configuration",
                    "handler": lambda: {{"version": "1.0", "debug": False}}
                }},
                "user-profile": {{
                    "uri": "user://profile",
                    "name": "User Profile",
                    "mimeType": "application/json",
                    "handler": lambda: get_user_profile()
                }}
            }},

            # Prompts - predefined prompt templates
            "prompts": {{
                "summarize": {{
                    "name": "Summarize Text",
                    "description": "Summarize the given text",
                    "arguments": [
                        {{"name": "text", "description": "Text to summarize", "required": True}}
                    ],
                    "handler": lambda text: f"Please summarize: {{text}}"
                }}
            }}
        }})

        print("MCP service registered!")
        print(f"Endpoint: {server_url}/{workspace}/mcp/my-tools/mcp")
        await server.serve()
```

### Use MCP Endpoint with Claude/Cursor

```bash
# Claude Desktop config (~/.config/claude/claude_desktop_config.json)
{{
  "mcpServers": {{
    "hypha-tools": {{
      "command": "npx",
      "args": ["-y", "mcp-remote", "{server_url}/{workspace}/mcp/my-tools/mcp"]
    }}
  }}
}}

# Cursor MCP config
{{
  "mcpServers": {{
    "hypha-tools": {{
      "url": "{server_url}/{workspace}/mcp/my-tools/mcp"
    }}
  }}
}}
```

### Connect to External MCP Servers (MCP Proxy)

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Connect to an external MCP server like DeepWiki
        app_info = await controller.install(
            source="https://mcp.deepwiki.com/mcp",  # External MCP URL
            app_id="deepwiki",
            manifest={{
                "type": "mcp-server",  # MCP proxy type
                "name": "DeepWiki MCP"
            }},
            overwrite=True
        )

        # Start the MCP proxy
        session = await controller.start(app_info["id"])

        # Use the proxied MCP tools as a Hypha service
        deepwiki = await server.get_service(f"default@{{app_info['id']}}")

        # Call MCP tools through Hypha
        result = await deepwiki.call_tool(
            tool_name="ask_question",
            arguments={{
                "repoName": "facebook/react",
                "question": "How does reconciliation work?"
            }}
        )
        print(result)
```

### MCP Service via HTTP

```bash
# Call MCP endpoint directly
curl -X POST "{server_url}/{workspace}/mcp/my-tools/mcp" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {{
      "name": "search_documents",
      "arguments": {{"query": "machine learning"}}
    }},
    "id": 1
  }}'
```

---

## A2A (Agent-to-Agent) Examples

A2A protocol enables agent-to-agent communication. Enable with `--enable-a2a`.

### Register an A2A Agent

```python
from pydantic import Field
from hypha_rpc.utils.schema import schema_method

async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Define agent capabilities
        @schema_method
        async def process_request(
            message: str = Field(..., description="User message to process"),
            context: dict = Field(default={{}}, description="Additional context")
        ) -> str:
            \"\"\"Process a user request and return a response.\"\"\"
            # Your agent logic here
            return f"Processed: {{message}}"

        @schema_method
        async def stream_response(
            message: str = Field(..., description="Message to stream")
        ):
            \"\"\"Stream a response word by word.\"\"\"
            for word in message.split():
                yield word + " "

        # Register as A2A agent
        await server.register_service({{
            "id": "my-agent",
            "name": "My AI Agent",
            "type": "a2a",  # Enable A2A protocol
            "config": {{"visibility": "public"}},

            # Agent Card metadata (required for A2A)
            "agent_card": {{
                "name": "My AI Agent",
                "description": "An intelligent assistant",
                "url": f"{server_url}/{workspace}/a2a/my-agent",
                "version": "1.0.0",
                "capabilities": {{
                    "streaming": True,
                    "pushNotifications": False
                }},
                "skills": [
                    {{"name": "text-processing", "description": "Process text"}}
                ]
            }},

            "process": process_request,
            "stream": stream_response,
        }})

        print("A2A agent registered!")
        print(f"Agent Card: {server_url}/{workspace}/a2a/my-agent/agent.json")
        print(f"Endpoint: {server_url}/{workspace}/a2a/my-agent")
        await server.serve()
```

### Call A2A Agent via HTTP

```bash
# Get agent card (discovery)
curl "{server_url}/{workspace}/a2a/my-agent/agent.json"

# Send task to agent
curl -X POST "{server_url}/{workspace}/a2a/my-agent" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "jsonrpc": "2.0",
    "method": "tasks/send",
    "params": {{
      "id": "task-123",
      "message": {{
        "role": "user",
        "parts": [{{"type": "text", "text": "Hello, agent!"}}]
      }}
    }},
    "id": 1
  }}'

# Stream response from agent
curl -X POST "{server_url}/{workspace}/a2a/my-agent" \\
  -H "Content-Type: application/json" \\
  -H "Accept: text/event-stream" \\
  -d '{{
    "jsonrpc": "2.0",
    "method": "tasks/sendSubscribe",
    "params": {{
      "id": "task-456",
      "message": {{
        "role": "user",
        "parts": [{{"type": "text", "text": "Stream your response"}}]
      }}
    }},
    "id": 1
  }}'
```

### Connect to External A2A Agents (A2A Proxy)

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        controller = await server.get_service("public/server-apps")

        # Connect to an external A2A agent
        app_info = await controller.install(
            source="https://external-agent.example.com/agent.json",
            app_id="external-agent",
            manifest={{
                "type": "a2a-agent",  # A2A proxy type
                "name": "External Agent Proxy"
            }},
            overwrite=True
        )

        # Start the A2A proxy
        session = await controller.start(app_info["id"])

        # Use the external agent through Hypha
        agent = await server.get_service(f"default@{{app_info['id']}}")
        response = await agent.send_task({{
            "message": {{"role": "user", "parts": [{{"type": "text", "text": "Hello!"}}]}}
        }})
        print(response)
```

### A2A Streaming in Python

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        agent = await server.get_service("{workspace}/my-agent")

        # Non-streaming call
        result = await agent.process("What is 2+2?")
        print(result)

        # Streaming call
        async for chunk in agent.stream("Tell me a story"):
            print(chunk, end="", flush=True)
```

---

## Vector Search Examples

### Create Vector Collection

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Create a vector collection with custom settings
        collection = await artifact_manager.create(
            alias="knowledge-base",
            type="vector-collection",
            config={{
                "dimension": 768,           # Must match embedding model
                "distance_metric": "cosine" # or "euclidean", "dot_product"
            }},
            manifest={{
                "name": "Knowledge Base",
                "description": "Searchable documentation",
                "tags": ["docs", "knowledge"]
            }}
        )
        print(f"Created collection: {{collection['id']}}")
```

### Add Documents with Embeddings

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Add documents - embeddings can be auto-generated or provided
        await artifact_manager.add_vectors(
            artifact_id="{workspace}/knowledge-base",
            vectors=[
                {{
                    "id": "doc-001",
                    "text": "Hypha is a serverless application framework",
                    # "vector": [...],  # Optional: provide custom embedding
                    "metadata": {{"category": "overview", "source": "docs"}}
                }},
                {{
                    "id": "doc-002",
                    "text": "Connect to server using connect_to_server()",
                    "metadata": {{"category": "api", "source": "reference"}}
                }},
                {{
                    "id": "doc-003",
                    "text": "Artifacts store files, datasets and models",
                    "metadata": {{"category": "storage", "source": "docs"}}
                }}
            ]
        )

        # Commit the changes
        await artifact_manager.commit(
            artifact_id="{workspace}/knowledge-base",
            message="Added documentation entries"
        )
```

### Search by Text Query

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Semantic search - query is auto-embedded
        results = await artifact_manager.search_vectors(
            artifact_id="{workspace}/knowledge-base",
            query="how to connect to the server",
            limit=5,
            filters={{"category": "api"}}  # Optional metadata filter
        )

        print(f"Found {{len(results['items'])}} results:")
        for item in results["items"]:
            print(f"  Score: {{item.get('score', 0):.4f}}")
            print(f"  Text: {{item['text'][:100]}}...")
            print(f"  ID: {{item['id']}}")
            print()
```

### Search by Vector

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # If you have your own embedding
        query_vector = [0.1, 0.2, ...]  # 768-dimensional vector

        results = await artifact_manager.search_vectors(
            artifact_id="{workspace}/knowledge-base",
            vector=query_vector,  # Use pre-computed vector
            limit=10
        )

        for item in results["items"]:
            print(f"{{item['id']}}: {{item['text'][:50]}}... ({{item.get('score', 0):.4f}})")
```

### Search Services by Similarity

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Search for services using natural language
        services = await server.search_services(
            query="image processing model inference",
            limit=5
        )

        print("Matching services:")
        for svc in services:
            print(f"  {{svc['id']}}: {{svc.get('name', 'unnamed')}}")
            print(f"    Score: {{svc.get('score', 0):.4f}}")
            print(f"    Desc: {{svc.get('description', '')[:100]}}")
```

### Manage Vector Collections

```python
async def main():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        artifact_manager = await server.get_service("public/artifact-manager")

        # Get collection info
        info = await artifact_manager.read(
            artifact_id="{workspace}/knowledge-base"
        )
        print(f"Collection: {{info['manifest']['name']}}")
        print(f"Vector count: {{info.get('vector_count', 0)}}")

        # Update vectors
        await artifact_manager.update_vectors(
            artifact_id="{workspace}/knowledge-base",
            vectors=[
                {{"id": "doc-001", "text": "Updated content here"}}
            ]
        )

        # Remove specific vectors
        await artifact_manager.remove_vectors(
            artifact_id="{workspace}/knowledge-base",
            ids=["doc-003"]
        )

        # Delete entire collection
        # await artifact_manager.delete(artifact_id="{workspace}/knowledge-base")
```

---

## Error Handling Examples

```python
async def safe_operation():
    async with connect_to_server({{"server_url": "{server_url}"}}) as server:
        # Check workspace status
        status = await server.check_status()
        if status["status"] != "ready":
            raise RuntimeError(f"Workspace not ready: {{status}}")

        # Handle service not found
        try:
            service = await server.get_service("maybe-missing")
        except KeyError as e:
            print(f"Service not found: {{e}}")
            # Maybe start the app first
            controller = await server.get_service("public/server-apps")
            await controller.start("my-app", wait_for_service="maybe-missing")
            service = await server.get_service("maybe-missing")

        # Handle permission errors
        try:
            await server.delete_workspace("protected-workspace")
        except PermissionError as e:
            print(f"Permission denied: {{e}}")

        # Handle timeouts
        try:
            result = await server.get_service(
                "slow-service",
                {{"timeout": 5.0}}
            )
        except TimeoutError:
            print("Service didn't respond in time")
```

---

## Quick Reference

### Service Types

| Type | Description | Protocol |
|------|-------------|----------|
| `default` | Standard Hypha RPC service | WebSocket/HTTP |
| `asgi` | ASGI web application (FastAPI, etc.) | HTTP |
| `functions` | Serverless HTTP functions | HTTP |
| `mcp` | Model Context Protocol | MCP JSON-RPC |
| `a2a` | Agent-to-Agent protocol | A2A JSON-RPC |

### Permission Levels

| Level | Description |
|-------|-------------|
| `read` | View services, artifacts, files |
| `read_write` | Create, modify, delete data |
| `admin` | Full control, generate tokens |

### HTTP Endpoints

| Endpoint | Description |
|----------|-------------|
| `/{workspace}/services` | List services |
| `/{workspace}/services/{{id}}/{{fn}}` | Call function |
| `/{workspace}/apps/{{id}}/...` | ASGI/Functions app |
| `/{workspace}/files/...` | S3 file proxy |
| `/{workspace}/mcp/{{id}}/mcp` | MCP endpoint |
| `/{workspace}/a2a/{{id}}` | A2A endpoint |
"""


def get_workspace_context_md(workspace: str, server_url: str, workspace_info: dict = None, services: list = None) -> str:
    """Generate workspace-specific context documentation."""
    workspace_info = workspace_info or {}
    services = services or []

    workspace_name = workspace_info.get("name", workspace)
    workspace_desc = workspace_info.get("description", "No description")
    is_persistent = workspace_info.get("persistent", False)
    is_read_only = workspace_info.get("read_only", False)
    owners = workspace_info.get("owners", [])

    # Format services list with schemas
    service_list = ""
    for svc in services[:50]:  # Limit to 50 services
        svc_id = svc.get("id", "unknown")
        svc_name = svc.get("name", "unnamed")
        svc_type = svc.get("type", "unknown")
        svc_desc = (svc.get("description") or "")[:100]
        service_list += f"\n### {svc_name or svc_id}\n- **ID**: `{svc_id}`\n- **Type**: {svc_type}\n"
        if svc_desc:
            service_list += f"- **Description**: {svc_desc}\n"

    if not service_list:
        service_list = "\nNo services currently registered.\n"

    return f"""# Workspace Context: {workspace}

This document contains the current state and configuration of workspace `{workspace}`.

## Workspace Information

| Property | Value |
|----------|-------|
| **ID** | `{workspace}` |
| **Name** | {workspace_name} |
| **Description** | {workspace_desc} |
| **Persistent** | {"Yes" if is_persistent else "No"} |
| **Read-Only** | {"Yes" if is_read_only else "No"} |
| **Owners** | {", ".join(owners) if owners else "None specified"} |
| **Server URL** | `{server_url}` |

## Capabilities

Based on workspace configuration:

{"### Storage (Persistent)" if is_persistent else "### Storage (Temporary)"}
{'''- ✅ Artifact creation and management
- ✅ File upload/download
- ✅ Version control (Git-like commits)
- ✅ Vector search collections''' if is_persistent else '''- ⚠️ Limited storage capabilities
- ⚠️ Data will be lost when workspace unloads
- ⚠️ No persistent file storage'''}

{"### Access Mode" if not is_read_only else "### Access Mode (Read-Only)"}
{'''- ✅ Can create services
- ✅ Can modify artifacts
- ✅ Can install applications''' if not is_read_only else '''- ❌ Cannot create new artifacts
- ❌ Cannot modify existing data
- ✅ Can read and use existing services'''}

## Registered Services

Current services available in this workspace:
{service_list}

## Quick Access URLs

| Resource | URL |
|----------|-----|
| Workspace Info | `{server_url}/{workspace}/info` |
| Service List | `{server_url}/{workspace}/services` |
| Health Check | `{server_url}/health` |

## Connection Code

```python
from hypha_rpc import connect_to_server

async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "{workspace}"
}}) as server:
    # Your code here
    services = await server.list_services()
```

## Notes

- This context was generated dynamically and reflects the current workspace state
- Service availability may change as applications start/stop
- Use `list_services()` for the most current service information
"""


def create_agent_skills_service(store) -> dict:
    """Create agent skills service for registration as a public service.

    This creates a service of type 'functions' that handles requests to:
    /{workspace}/agent-skills/

    Files served:
    - SKILL.md: Main skill instructions
    - REFERENCE.md: API reference documentation (dynamically generated from service schemas)
    - EXAMPLES.md: Code examples
    - WORKSPACE_CONTEXT.md: Workspace-specific context

    Args:
        store: The RedisStore instance

    Returns:
        Service dictionary ready for registration via store.register_public_service()
    """
    server_info = store.get_server_info()
    server_url = server_info.get("public_base_url", "http://localhost:9527")

    # Create a lazy-loaded doc generator
    _doc_generator = None

    def get_doc_generator() -> DynamicDocGenerator:
        nonlocal _doc_generator
        if _doc_generator is None:
            _doc_generator = DynamicDocGenerator(store, server_url)
        return _doc_generator

    async def get_workspace_info_safe(workspace: str, user_info) -> dict:
        """Get workspace information safely."""
        if not user_info:
            return {}
        try:
            async with store.get_workspace_interface(user_info, workspace) as manager:
                workspace_info = await manager.get_workspace_info(workspace)
                return workspace_info
        except Exception as e:
            logger.warning(f"Could not get workspace info: {e}")
            return {}

    async def get_services_safe(workspace: str, user_info) -> list:
        """Get list of services in workspace."""
        if not user_info:
            return []
        try:
            async with store.get_workspace_interface(user_info, workspace) as manager:
                services = await manager.list_services()
                return services
        except Exception as e:
            logger.warning(f"Could not get services: {e}")
            return []

    def extract_workspace(scope: dict) -> Optional[str]:
        """Extract workspace from scope/request context."""
        # Try to get from scope directly
        if "workspace" in scope:
            return scope["workspace"]

        # Try to extract from path
        path = scope.get("raw_path", scope.get("path", ""))
        if isinstance(path, bytes):
            path = path.decode("latin-1")

        # Path format: /{workspace}/agent-skills/...
        parts = path.strip("/").split("/")
        if len(parts) >= 1 and parts[0] and not parts[0].startswith("apps"):
            return parts[0]

        return None

    async def get_user_from_scope(scope: dict):
        """Get user info from request scope.

        Returns authenticated user info if a valid token is provided,
        otherwise returns None (indicating anonymous/unauthenticated access).
        """
        try:
            # Try to get token from headers
            headers = scope.get("headers", {})
            if isinstance(headers, dict):
                auth_header = headers.get("authorization", "")
            else:
                # Headers might be a list of tuples
                auth_header = ""
                for k, v in headers:
                    if k.lower() == b"authorization":
                        auth_header = v.decode("utf-8") if isinstance(v, bytes) else v
                        break

            token = None
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]

            # Try query string
            if not token:
                qs = scope.get("query_string", b"")
                if isinstance(qs, bytes):
                    qs = qs.decode("utf-8")
                for part in qs.split("&"):
                    if part.startswith("token="):
                        token = part[6:]
                        break

            if token:
                # Parse token to get user info
                user_info = await store.parse_user_token(token)
                return user_info
        except Exception as e:
            logger.debug(f"Could not get user from scope: {e}")

        # Return None for anonymous users - the service will provide
        # basic documentation without workspace-specific details
        return None

    async def handle_index(
            scope: dict, services: list = None, public_services: list = None,
        ) -> dict:
        """Handle index request - return skill directory listing."""
        services = services or []
        public_services = public_services or []
        doc_generator = get_doc_generator()
        enabled_services = await doc_generator.get_all_enabled_services()

        # Build file list including per-service reference docs
        files = [
            "SKILL.md",
            "REFERENCE.md",
            "EXAMPLES.md",
            "WORKSPACE_CONTEXT.md",
        ]
        reference_files = []
        for svc in enabled_services:
            ref_file = f"REFERENCE/{svc['id']}.md"
            files.append(ref_file)
            reference_files.append(ref_file)

        # Add dynamic service references (workspace + public)
        builtin_ids = {svc["id"] for svc in enabled_services}
        dynamic_slugs = []
        for svc in services + public_services:
            if _service_has_schema(svc):
                slug = _normalize_service_slug(svc.get("id", ""))
                if slug not in builtin_ids and slug not in dynamic_slugs:
                    dynamic_slugs.append(slug)
                    ref_file = f"REFERENCE/{slug}.md"
                    files.append(ref_file)
                    reference_files.append(ref_file)

        return {
            "status": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "name": SKILL_NAME,
                "version": SKILL_VERSION,
                "description": "Hypha workspace management skill for AI agents",
                "files": files,
                "reference_files": reference_files,
                "download": {
                    "zip": "create-zip-file",
                    "description": "Download all skills documentation as a zip file"
                },
                "source_endpoints": [
                    f"SOURCE/{svc['id']}" for svc in enabled_services
                ],
                "enabled_services": [svc['id'] for svc in enabled_services],
                "dynamic_services": dynamic_slugs,
            })
        }

    async def handle_source_request(scope: dict, path: str) -> dict:
        """Handle source code requests.

        Paths:
        - SOURCE/{service_id}: Get all method source for a service
        - SOURCE/{service_id}/{method_name}: Get specific method source
        """
        doc_generator = get_doc_generator()

        # Parse path: SOURCE/{service_id}/{method_name}
        parts = path.split("/")
        if len(parts) < 2:
            return {
                "status": 400,
                "headers": {"Content-Type": "text/plain"},
                "body": "Invalid source path. Use SOURCE/{service_id} or SOURCE/{service_id}/{method_name}"
            }

        service_id = parts[1]
        method_name = parts[2] if len(parts) > 2 else None

        # Derive service_id -> class_name from the module-level constant
        service_id_to_class = {
            svc_id: class_name
            for svc_id, (_, class_name) in SERVICE_CLASS_MAP.items()
        }

        if service_id not in service_id_to_class:
            # List available services
            available = ", ".join(service_id_to_class.keys())
            return {
                "status": 404,
                "headers": {"Content-Type": "text/plain"},
                "body": f"Unknown service: {service_id}. Available services: {available}"
            }

        class_name = service_id_to_class[service_id]

        if method_name:
            # Get specific method source
            source = doc_generator.get_method_source(class_name, method_name)
            if source:
                return {
                    "status": 200,
                    "headers": {"Content-Type": "text/markdown; charset=utf-8"},
                    "body": f"# Source code for {class_name}.{method_name}\n\n```python\n{source}\n```"
                }
            else:
                return {
                    "status": 404,
                    "headers": {"Content-Type": "text/plain"},
                    "body": f"Method not found: {class_name}.{method_name}"
                }
        else:
            # Get service API with method listing
            service_api = await doc_generator.get_service_api(service_id)
            if not service_api:
                return {
                    "status": 404,
                    "headers": {"Content-Type": "text/plain"},
                    "body": f"Service {service_id} is not enabled on this server"
                }

            # Build a summary with links to each method's source
            methods = service_api.get("methods", {})
            method_list = "\n".join(f"- `{name}`: SOURCE/{service_id}/{name}" for name in sorted(methods.keys()))

            content = f"""# Source Code Reference: {service_api['name']}

**Service ID:** `{service_id}`
**Class:** `{class_name}`

## Available Methods

{method_list}

## How to Use

To get the source code for a specific method, use:

```
GET /{'{workspace}'}/agent-skills/SOURCE/{service_id}/{'{method_name}'}
```

For example:
- `SOURCE/{service_id}/list_services` - Get source for list_services method
- `SOURCE/{service_id}/create` - Get source for create method

## Note

Source code is provided for reference and implementation details.
For high-level usage, refer to REFERENCE.md and EXAMPLES.md.
"""
            return {
                "status": 200,
                "headers": {"Content-Type": "text/markdown; charset=utf-8"},
                "body": content
            }

    async def handle_create_zip(
            scope: dict, services: list = None, public_services: list = None,
        ) -> dict:
        """Create a zip file containing all skills documentation.

        Returns a zip archive with:
        - SKILL.md
        - REFERENCE.md
        - EXAMPLES.md
        - WORKSPACE_CONTEXT.md
        - REFERENCE/<service-id>.md for each service with schema
        - SOURCE/ directory with all service source code
        """
        services = services or []
        public_services = public_services or []
        workspace = extract_workspace(scope)
        user_info = await get_user_from_scope(scope)
        workspace_info = {}

        ws = workspace or "public"

        if workspace and user_info:
            workspace_info = await get_workspace_info_safe(workspace, user_info)
            # Use passed services if available, otherwise fetch
            if not services:
                services = await get_services_safe(workspace, user_info)

        doc_generator = get_doc_generator()

        # Create in-memory zip file
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add main documentation files
            zf.writestr("SKILL.md", get_skill_md(
                ws, server_url, workspace_info, services,
                public_services=public_services,
            ))
            zf.writestr("REFERENCE.md", await get_reference_md(
                ws, server_url, doc_generator,
                services=services, public_services=public_services,
            ))

            # Add per-service reference docs for built-in services
            enabled_services_list = await doc_generator.get_all_enabled_services()
            builtin_ids = set()
            for svc_api in enabled_services_list:
                svc_ref = await get_service_reference_md(
                    svc_api["id"], ws, server_url, doc_generator
                )
                if svc_ref:
                    zf.writestr(f"REFERENCE/{svc_api['id']}.md", svc_ref)
                    builtin_ids.add(svc_api["id"])

            # Add per-service reference docs for dynamic services
            for svc in services + public_services:
                if not _service_has_schema(svc):
                    continue
                slug = _normalize_service_slug(svc.get("id", ""))
                if slug in builtin_ids:
                    continue
                svc_ref = generate_reference_from_service_schema(svc, ws, server_url)
                if svc_ref:
                    zf.writestr(f"REFERENCE/{slug}.md", svc_ref)
                    builtin_ids.add(slug)

            zf.writestr("EXAMPLES.md", get_examples_md(ws, server_url))
            zf.writestr("WORKSPACE_CONTEXT.md", get_workspace_context_md(ws, server_url, workspace_info, services))

            # Add source code for each enabled built-in service
            enabled_services = await doc_generator.get_all_enabled_services()
            # Derive service_id -> class_name from the module-level constant
            svc_id_to_class = {
                svc_id: class_name
                for svc_id, (_, class_name) in SERVICE_CLASS_MAP.items()
            }

            for service_api in enabled_services:
                service_id = service_api["id"]
                if service_id not in svc_id_to_class:
                    continue

                class_name = svc_id_to_class[service_id]
                methods = service_api.get("methods", {})

                # Create a README for this service
                method_list = "\n".join(f"- `{name}`: {name}.py" for name in sorted(methods.keys()))
                readme = f"""# Source Code: {service_api['name']}

**Service ID:** `{service_id}`
**Class:** `{class_name}`

## Methods

{method_list}
"""
                zf.writestr(f"SOURCE/{service_id}/README.md", readme)

                # Add source code for each method
                for method_name in methods.keys():
                    source = doc_generator.get_method_source(class_name, method_name)
                    if source:
                        zf.writestr(f"SOURCE/{service_id}/{method_name}.py", source)

        # Get the zip content
        zip_buffer.seek(0)
        zip_content = zip_buffer.getvalue()

        return {
            "status": 200,
            "headers": {
                "Content-Type": "application/zip",
                "Content-Disposition": f"attachment; filename=hypha-skills-{ws}.zip",
                "Content-Length": str(len(zip_content)),
            },
            "body": zip_content,
        }

    async def handle_request(scope: dict) -> dict:
        """Handle requests for skill files."""
        path = scope.get("function_path", scope.get("path", "/")).strip("/")
        workspace = extract_workspace(scope)

        # Get user info for authenticated requests
        user_info = await get_user_from_scope(scope)
        workspace_info = {}
        services = []
        public_services = []

        ws = workspace or "public"

        # For public workspace, create anonymous user if not authenticated
        # so we can discover public services
        if ws == "public" and user_info is None:
            from hypha.core.auth import create_scope as _create_scope, generate_anonymous_user
            user_info = generate_anonymous_user(
                scope=_create_scope("ws-anonymous#r", current_workspace="ws-anonymous")
            )

        # Require authentication for non-public workspaces
        if ws != "public" and user_info is None:
            return {
                "status": 401,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Authentication required",
                    "message": f"Access to workspace '{ws}' skills requires a valid token. "
                    "Use Authorization: Bearer <token> header or ?token=<token> query parameter or cookies. "
                    "For generic Hypha documentation without authentication, use the global endpoint.",
                    "global_url": f"{server_url}/ws/agent-skills/",
                    "public_url": f"{server_url}/public/agent-skills/"
                })
            }

        if workspace and user_info:
            workspace_info = await get_workspace_info_safe(workspace, user_info)
            services = await get_services_safe(workspace, user_info)

        # For non-public workspaces, also fetch public services
        # (they are accessible from any workspace)
        if ws != "public" and user_info:
            public_services = await get_services_safe("public", user_info)

        doc_generator = get_doc_generator()

        # Route to appropriate handler
        if path in ["", "index", "index.html"]:
            return await handle_index(
                scope, services=services, public_services=public_services
            )
        elif path == "SKILL.md":
            content = get_skill_md(
                ws, server_url, workspace_info, services,
                public_services=public_services,
            )
        elif path == "REFERENCE.md":
            content = await get_reference_md(
                ws, server_url, doc_generator,
                services=services, public_services=public_services,
            )
        elif path.startswith("REFERENCE/") and path.endswith(".md"):
            # Per-service reference docs: REFERENCE/<service-id>.md
            service_slug = path[len("REFERENCE/"):-len(".md")]
            # 1. Try built-in services first (richer docs from source code)
            content = await get_service_reference_md(
                service_slug, ws, server_url, doc_generator
            )
            # 2. If not found, try dynamic services from workspace + public
            if content is None:
                all_dynamic = services + public_services
                matching_service = None
                for svc in all_dynamic:
                    if _normalize_service_slug(svc.get("id", "")) == service_slug:
                        matching_service = svc
                        break
                if matching_service and _service_has_schema(matching_service):
                    content = generate_reference_from_service_schema(
                        matching_service, ws, server_url
                    )
            # 3. If still not found, return 404
            if content is None:
                enabled_services = await doc_generator.get_all_enabled_services()
                available = [svc["id"] for svc in enabled_services]
                for svc in services + public_services:
                    slug = _normalize_service_slug(svc.get("id", ""))
                    if _service_has_schema(svc) and slug not in available:
                        available.append(slug)
                return {
                    "status": 404,
                    "headers": {"Content-Type": "text/plain"},
                    "body": f"Service not found or not enabled: {service_slug}. "
                    f"Available services: {', '.join(available)}"
                }
        elif path == "EXAMPLES.md":
            content = get_examples_md(ws, server_url)
        elif path == "WORKSPACE_CONTEXT.md":
            content = get_workspace_context_md(ws, server_url, workspace_info, services)
        elif path.startswith("SOURCE"):
            # Handle source code requests
            return await handle_source_request(scope, path)
        elif path in ["create-zip-file", "download.zip", "skills.zip"]:
            # Create and return zip file with all documentation
            return await handle_create_zip(
                scope, services=services, public_services=public_services
            )
        else:
            return {
                "status": 404,
                "headers": {"Content-Type": "text/plain"},
                "body": f"File not found: {path}"
            }

        return {
            "status": 200,
            "headers": {"Content-Type": "text/markdown; charset=utf-8"},
            "body": content
        }

    # Return service dictionary for registration
    return {
        "id": "agent-skills",
        "name": "Agent Skills Provider",
        "description": "Provides Agent Skills documentation for AI agents following the Agent Skills specification (https://agentskills.io)",
        "type": "functions",
        "config": {
            "visibility": "public",
            "require_context": False,
        },
        "default": handle_request,
    }


# Keep these for backwards compatibility and testing
class AgentSkillsService:
    """Service that provides Agent Skills documentation for AI agents.

    Note: This class is kept for testing purposes. For production use,
    use create_agent_skills_service() with store.register_public_service().
    """

    def __init__(self, store, server_info: dict):
        self.store = store
        self.server_info = server_info
        self.server_url = server_info.get("public_base_url", "http://localhost:9527")
        self._doc_generator = None

    def _get_doc_generator(self) -> DynamicDocGenerator:
        """Lazy initialize the documentation generator."""
        if self._doc_generator is None:
            self._doc_generator = DynamicDocGenerator(self.store, self.server_url)
        return self._doc_generator


class AgentSkillsMiddleware:
    """ASGI middleware for serving agent skills.

    Note: This is deprecated. Use create_agent_skills_service() with
    store.register_public_service() instead.
    """

    def __init__(self, app: ASGIApp, base_path: str = None, store=None):
        self.app = app
        logger.warning(
            "AgentSkillsMiddleware is deprecated. "
            "Use create_agent_skills_service() with store.register_public_service() instead."
        )

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


def setup_agent_skills(store) -> dict:
    """Set up agent skills as a public service.

    This creates a 'functions' type service that serves agent skills documentation at:
    /{workspace}/agent-skills/

    Files served:
    - SKILL.md: Main skill instructions
    - REFERENCE.md: API reference documentation (dynamically generated from service schemas)
    - EXAMPLES.md: Code examples
    - WORKSPACE_CONTEXT.md: Workspace-specific context

    Args:
        store: The RedisStore instance

    Returns:
        Service dictionary that should be registered via store.register_public_service()
    """
    logger.info("Setting up Agent Skills service")
    return create_agent_skills_service(store)


def get_global_skill_md(server_url: str) -> str:
    """Generate the global SKILL.md for unauthenticated access.

    This provides generic instructions on how to use Hypha: connecting,
    obtaining tokens, discovering workspaces, and accessing workspace-specific skills.
    """
    return f"""---
name: {SKILL_NAME}
description: Connect to the Hypha distributed computing platform - obtain tokens, discover workspaces, and access services. This is the global entry point for AI agents.
license: BSD-3-Clause
compatibility: Requires network access to Hypha server. Works with Python 3.8+ or modern JavaScript environments.
metadata:
  version: {SKILL_VERSION}
  server_url: {server_url}
---

# Hypha Platform Guide

This is the **global** agent skills endpoint for the Hypha server at `{server_url}`.
It provides instructions on how to connect, authenticate, and access workspace-specific services.

## Quick Start

### 1. Install Dependencies

**Python:**
```bash
pip install hypha-rpc
```

**JavaScript / Node.js:**
```bash
npm install hypha-rpc
```

**HTTP Only (No Library Needed):**
All Hypha services are accessible via HTTP REST endpoints with `curl`, `fetch`, or any HTTP client.

### 2. Connect Anonymously

```python
import asyncio
from hypha_rpc import connect_to_server

async def main():
    async with connect_to_server({{
        "server_url": "{server_url}"
    }}) as server:
        print(f"Connected to workspace: {{server.config.workspace}}")
        services = await server.list_services()
        print(f"Found {{len(services)}} services")

asyncio.run(main())
```

```bash
# HTTP: List public services (no auth needed)
curl "{server_url}/public/services"
```

### 3. Authenticate with a Token

If you have a workspace token (provided by a user or generated via the API):

```python
async with connect_to_server({{
    "server_url": "{server_url}",
    "workspace": "YOUR_WORKSPACE",
    "token": "YOUR_TOKEN"
}}) as server:
    status = await server.check_status()
    print(f"Permission: {{status.get('user_permission')}}")
```

```bash
# HTTP: Authenticated access
curl -H "Authorization: Bearer YOUR_TOKEN" \\
  "{server_url}/YOUR_WORKSPACE/services"
```

### 4. Interactive Login (Browser-Based OAuth)

For interactive sessions where a user can open a browser:

```python
from hypha_rpc import login, connect_to_server

async def main():
    token = await login({{"server_url": "{server_url}"}})
    async with connect_to_server({{
        "server_url": "{server_url}",
        "token": token
    }}) as server:
        # Generate a reusable token for future sessions
        workspace_token = await server.generate_token({{
            "permission": "read_write",
            "expires_in": 86400  # 24 hours
        }})
        print(f"Workspace: {{server.config.workspace}}")
        print(f"Save this token: {{workspace_token}}")

asyncio.run(main())
```

## Token & Permission Management

### Permission Levels
- **`read`**: View workspace resources and services
- **`read_write`**: Create/modify services, upload data, manage artifacts
- **`admin`**: Full workspace control including deletion and token generation

### Generating Tokens

Once authenticated, generate tokens for programmatic access:

```python
# Read-only token (safe to share)
read_token = await server.generate_token({{
    "permission": "read",
    "expires_in": 86400  # 24 hours
}})

# Read-write token
rw_token = await server.generate_token({{
    "permission": "read_write",
    "expires_in": 3600  # 1 hour
}})
```

## Discovering Workspaces and Services

### List Services in a Workspace

```python
# List all services
services = await server.list_services()

# Get a specific service
svc = await server.get_service("service-id")

# Call a service function
result = await svc.some_function(param1="value")
```

```bash
# HTTP: List services
curl "{server_url}/YOUR_WORKSPACE/services"

# HTTP: Call a service function
curl -X POST "{server_url}/YOUR_WORKSPACE/services/SERVICE_ID/FUNCTION" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{{"param1": "value"}}'
```

## Core Platform Capabilities

### 1. Service Management
Register, discover, and call remote services via RPC or HTTP. Every service exposes methods that can be called via WebSocket (real-time, bidirectional) or HTTP (stateless REST).
- **Reference**: [REFERENCE/workspace-manager.md](REFERENCE/workspace-manager.md)

### 2. Artifact Management
Store and manage files, datasets, models, and applications with Git-like versioning. Supports vector search for semantic queries over collections.
- Create collections and items with metadata and file attachments
- Stage files, commit changes, and browse version history
- Upload/download via presigned S3 URLs
- **Reference**: [REFERENCE/artifact-manager.md](REFERENCE/artifact-manager.md)

### 3. Server Applications
Deploy serverless applications that run in browser workers, Python sandboxes, or custom runtimes. Applications expose HTTP endpoints via ASGI (e.g., FastAPI) or serverless functions.
- Install apps from source code or URLs
- Start/stop/scale worker instances
- Access apps at: `{server_url}/WORKSPACE/apps/{{app_id}}/`
- **Reference**: [REFERENCE/server-apps.md](REFERENCE/server-apps.md)

### 4. S3 Storage
Direct S3-compatible object storage with presigned URLs, multipart uploads, and an HTTP proxy for file access.
- Upload/download files with presigned URLs
- HTTP proxy endpoint: `{server_url}/WORKSPACE/files/`
- **Reference**: [REFERENCE/s3-storage.md](REFERENCE/s3-storage.md)

### 5. MCP Integration
Expose Hypha services as MCP (Model Context Protocol) endpoints for AI tools like Claude, Cursor, Gemini CLI, etc.
- MCP endpoint: `{server_url}/WORKSPACE/mcp/{{service_id}}/mcp`
- Supports HTTP JSON-RPC transport

### 6. A2A Protocol
Agent-to-Agent protocol for multi-agent communication following Google's A2A specification.
- A2A endpoint: `{server_url}/WORKSPACE/a2a/{{service_id}}`

### 7. Vector Search
Semantic search by text query or vector embedding. Search services by natural language description.

## Key Built-in Services

| Service | ID | Reference |
|---------|-----|-----------|
| Workspace Manager | `~` or `default` | [Full API](REFERENCE/workspace-manager.md) |
| Artifact Manager | `public/artifact-manager` | [Full API](REFERENCE/artifact-manager.md) |
| Server Apps | `public/server-apps` | [Full API](REFERENCE/server-apps.md) |
| S3 Storage | `public/s3-storage` | [Full API](REFERENCE/s3-storage.md) |

## Instructions for AI Agents

1. **Start here** - Read this global SKILL.md to understand the platform
2. **Get a token** - Use anonymous access or request a token from the user
3. **Read service docs** - Check `REFERENCE/{{service-id}}.md` for the service API you need
4. **Access workspace skills** - Use `{server_url}/WORKSPACE/agent-skills/SKILL.md` for workspace-specific docs
5. **Discover services** - `GET {server_url}/WORKSPACE/services` to list available services
6. **Use HTTP or SDK** - HTTP REST for simple calls; Python/JS SDK for real-time bidirectional communication
7. **Handle errors** - Check permissions, verify workspace exists, handle timeouts

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `PermissionError` | Insufficient permissions | Generate token with higher permission level |
| `Workspace not found` | Workspace doesn't exist or was unloaded | Create workspace or check workspace ID |
| `Service not found` | Service not registered | Use `list_services()` to discover available services |
| `TimeoutError` | Service didn't respond | Increase timeout or check service status |
| `401 Unauthorized` | Missing or invalid token | Provide valid token via Bearer header or cookie |

## Documentation URLs

### Generic Documentation (No Auth Required)

These URLs serve general Hypha documentation accessible without authentication:

| Document | URL |
|----------|-----|
| Platform Guide | `{server_url}/ws/agent-skills/SKILL.md` |
| API Reference Overview | `{server_url}/ws/agent-skills/REFERENCE.md` |
| Workspace Manager API | `{server_url}/ws/agent-skills/REFERENCE/workspace-manager.md` |
| Artifact Manager API | `{server_url}/ws/agent-skills/REFERENCE/artifact-manager.md` |
| Server Apps API | `{server_url}/ws/agent-skills/REFERENCE/server-apps.md` |
| S3 Storage API | `{server_url}/ws/agent-skills/REFERENCE/s3-storage.md` |
| Code Examples | `{server_url}/ws/agent-skills/EXAMPLES.md` |

### Workspace-Specific Documentation (Auth Required)

For documentation tailored to a specific workspace (includes workspace services, context, and capabilities):

| Document | URL |
|----------|-----|
| Workspace Guide | `{server_url}/{{workspace}}/agent-skills/SKILL.md` |
| API Reference Overview | `{server_url}/{{workspace}}/agent-skills/REFERENCE.md` |
| Per-Service API | `{server_url}/{{workspace}}/agent-skills/REFERENCE/{{service-id}}.md` |
| Code Examples | `{server_url}/{{workspace}}/agent-skills/EXAMPLES.md` |
| Workspace Context | `{server_url}/{{workspace}}/agent-skills/WORKSPACE_CONTEXT.md` |
| Source Code | `{server_url}/{{workspace}}/agent-skills/SOURCE/{{service-id}}/{{method}}` |
| Download All (ZIP) | `{server_url}/{{workspace}}/agent-skills/create-zip-file` |

Replace `{{workspace}}` with the target workspace ID. Provide a Bearer token via the `Authorization` header or `?token=` query parameter.

### Public Workspace (No Auth Required)

The `public` workspace skills are always accessible without authentication:

- `{server_url}/public/agent-skills/SKILL.md`
- `{server_url}/public/agent-skills/REFERENCE/workspace-manager.md`
"""


def setup_global_agent_skills_routes(app, store, base_path: str = ""):
    """Set up global agent-skills routes at /ws/agent-skills/.

    These routes serve generic Hypha documentation without requiring
    authentication, providing instructions on how to connect, obtain
    tokens, and access workspace-specific skills.

    Args:
        app: The FastAPI application
        store: The RedisStore instance
        base_path: URL base path prefix
    """
    from fastapi import Request
    from fastapi.responses import Response, JSONResponse

    server_info = store.get_server_info()
    server_url = server_info.get("public_base_url", "http://localhost:9527")

    _doc_generator = None

    def get_doc_generator() -> DynamicDocGenerator:
        nonlocal _doc_generator
        if _doc_generator is None:
            _doc_generator = DynamicDocGenerator(store, server_url)
        return _doc_generator

    def norm_url(path: str) -> str:
        return (base_path.rstrip("/") + path) if base_path else path

    @app.get(norm_url("/ws/agent-skills/"))
    @app.get(norm_url("/ws/agent-skills/{path:path}"))
    async def global_agent_skills(request: Request, path: str = ""):
        """Serve global agent skills documentation (no auth required)."""
        path = path.strip("/")

        cors_headers = {
            "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
        }

        if request.method == "OPTIONS":
            return Response(status_code=204, headers=cors_headers)

        if path in ["", "index", "index.html"]:
            doc_generator = get_doc_generator()
            enabled_services = await doc_generator.get_all_enabled_services()
            files = ["SKILL.md", "REFERENCE.md", "EXAMPLES.md"]
            reference_files = []
            for svc in enabled_services:
                ref_file = f"REFERENCE/{svc['id']}.md"
                files.append(ref_file)
                reference_files.append(ref_file)
            data = {
                "name": SKILL_NAME,
                "version": SKILL_VERSION,
                "description": "Hypha platform guide for AI agents - global entry point",
                "type": "global",
                "files": files,
                "reference_files": reference_files,
                "workspace_skills_pattern": f"{server_url}/{{workspace}}/agent-skills/",
                "enabled_services": [svc["id"] for svc in enabled_services],
            }
            return JSONResponse(content=data, headers=cors_headers)

        if path == "SKILL.md":
            content = get_global_skill_md(server_url)
            return Response(
                content=content,
                media_type="text/markdown; charset=utf-8",
                headers=cors_headers,
            )

        if path == "REFERENCE.md":
            doc_generator = get_doc_generator()
            content = await get_reference_md("public", server_url, doc_generator)
            return Response(
                content=content,
                media_type="text/markdown; charset=utf-8",
                headers=cors_headers,
            )

        if path == "EXAMPLES.md":
            content = get_examples_md("public", server_url)
            return Response(
                content=content,
                media_type="text/markdown; charset=utf-8",
                headers=cors_headers,
            )

        if path.startswith("REFERENCE/") and path.endswith(".md"):
            service_id = path[len("REFERENCE/"):-len(".md")]
            doc_generator = get_doc_generator()
            content = await get_service_reference_md(
                service_id, "public", server_url, doc_generator
            )
            if content:
                return Response(
                    content=content,
                    media_type="text/markdown; charset=utf-8",
                    headers=cors_headers,
                )
            return JSONResponse(
                status_code=404,
                content={"error": f"Service not found or not enabled: {service_id}"},
                headers=cors_headers,
            )

        return JSONResponse(
            status_code=404,
            content={"error": f"File not found: {path}"},
            headers=cors_headers,
        )

    logger.info("Global agent skills routes registered at /ws/agent-skills/")
