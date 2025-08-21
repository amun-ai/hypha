"""MCP (Model Context Protocol) service support for Hypha.

This module provides integration with the MCP protocol using the official mcp-sdk.
It allows registering MCP services as Hypha services and automatically exposes them
via HTTP endpoints with proper MCP protocol handling.
"""

import inspect
import json
import logging
import asyncio
from typing import Any, Dict, Optional, List, Callable, Union
from collections import deque
from dataclasses import dataclass
from uuid import uuid4
import time

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.types import ASGIApp, Scope, Receive, Send
from starlette.requests import Request
from pydantic import AnyUrl

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.streamable_http import (
    EventCallback,
    EventId,
    EventMessage,
    EventStore,
    StreamId,
)
from mcp.types import JSONRPCMessage


def unwrap_object_proxy(obj):
    """Recursively unwrap ObjectProxy objects from Hypha RPC."""
    if hasattr(obj, "_rvalue"):
        return unwrap_object_proxy(obj._rvalue)
    elif isinstance(obj, dict):
        return {k: unwrap_object_proxy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unwrap_object_proxy(item) for item in obj]
    else:
        return obj


@dataclass
class EventEntry:
    """Represents an event entry in the event store."""
    event_id: EventId
    stream_id: StreamId
    message: JSONRPCMessage
    timestamp: float


class RedisEventStore(EventStore):
    """Redis-based event store for MCP session resumability.
    
    This implementation uses Redis to store events with expiration times,
    enabling horizontal scaling and session persistence across multiple
    Hypha instances.
    """
    
    def __init__(
        self,
        redis_client,
        max_events_per_stream: int = 100,
        expiration_time: int = 3600,
    ):
        """Initialize the Redis event store.
        
        Args:
            redis_client: Redis client instance
            max_events_per_stream: Maximum number of events to keep per stream
            expiration_time: Event expiration time in seconds
        """
        self.redis = redis_client
        self.max_events_per_stream = max_events_per_stream
        self.expiration_time = expiration_time
        self.key_prefix = "mcp:events:"
        self.stream_key_prefix = "mcp:streams:"
    
    def _get_stream_key(self, stream_id: StreamId) -> str:
        """Get Redis key for a stream."""
        return f"{self.stream_key_prefix}{stream_id}"
    
    def _get_event_key(self, event_id: EventId) -> str:
        """Get Redis key for an event."""
        return f"{self.key_prefix}{event_id}"
    
    async def store_event(
        self, stream_id: StreamId, message: JSONRPCMessage
    ) -> EventId:
        """Store an event with a generated event ID."""
        event_id = str(uuid4())
        timestamp = time.time()
        
        event_entry = EventEntry(
            event_id=event_id, stream_id=stream_id, message=message, timestamp=timestamp
        )
        
        # Store the event with expiration
        # Convert JSONRPCMessage to dict if needed
        if hasattr(message, 'model_dump'):
            message_data = message.model_dump()
        elif hasattr(message, 'dict'):
            message_data = message.dict()
        else:
            message_data = dict(message)
        
        event_data = {
            "event_id": event_id,
            "stream_id": stream_id,
            "message": json.dumps(message_data),
            "timestamp": timestamp,
        }
        
        await self.redis.hset(self._get_event_key(event_id), mapping=event_data)
        await self.redis.expire(self._get_event_key(event_id), self.expiration_time)
        
        # Add to stream's event list
        stream_key = self._get_stream_key(stream_id)
        await self.redis.lpush(stream_key, event_id)
        
        # Trim stream to max events
        await self.redis.ltrim(stream_key, 0, self.max_events_per_stream - 1)
        
        # Set expiration for stream
        await self.redis.expire(stream_key, self.expiration_time)
        
        return event_id
    
    async def replay_events_after(
        self,
        last_event_id: EventId,
        send_callback: EventCallback,
    ) -> StreamId | None:
        """Replay events that occurred after the specified event ID."""
        # Get event info
        event_key = self._get_event_key(last_event_id)
        event_data = await self.redis.hgetall(event_key)
        
        if not event_data:
            logger.warning(f"Event ID {last_event_id} not found in store")
            return None
        
        stream_id = event_data[b"stream_id"].decode()
        last_timestamp = float(event_data[b"timestamp"])
        
        # Get all events in stream
        stream_key = self._get_stream_key(stream_id)
        event_ids = await self.redis.lrange(stream_key, 0, -1)
        
        # Events are stored in reverse chronological order (latest first)
        # We need to reverse to get chronological order
        event_ids.reverse()
        
        # Find events after the last event ID
        replay_events = []
        found_last = False
        
        for eid in event_ids:
            event_id_str = eid.decode()
            if found_last:
                # Get event data
                event_key = self._get_event_key(event_id_str)
                event_data = await self.redis.hgetall(event_key)
                if event_data:
                    message = json.loads(event_data[b"message"].decode())
                    replay_events.append((event_id_str, message))
            elif event_id_str == last_event_id:
                found_last = True
        
        # Send replay events
        for event_id_str, message in replay_events:
            await send_callback(EventMessage(message, event_id_str))
        
        return stream_id
    
    async def cleanup_expired_events(self):
        """Clean up expired events (called periodically)."""
        # This is handled automatically by Redis TTL
        pass



class HyphaMCPAdapter:
    """Adapter that bridges Hypha services to MCP protocol using the official SDK."""
    
    def __init__(self, service, service_info, redis_client=None):
        """Initialize the MCP adapter.
        
        Args:
            service: Hypha service object
            service_info: Service information from Hypha
            redis_client: Redis client for event store (optional)
        """
        # Unwrap the service object to ensure we have local access to functions
        self.service = unwrap_object_proxy(service) if service else {}
        self.service_info = service_info
        self.server = Server(name=service_info.id)
        
        # Store tool/resource/prompt mappings for handler lookup
        self.tool_handlers = {}
        self.resource_handlers = {}
        self.prompt_handlers = {}
        
        # Setup handlers based on service configuration
        self._setup_handlers()
        
        # Create event store
        if redis_client:
            self.event_store = RedisEventStore(redis_client)
        else:
            raise ValueError("Redis client is required for MCP event store")
        
        # Create session manager - use stateless mode for simplicity
        # In stateless mode, each request is independent and doesn't require session tracking
        self.session_manager = StreamableHTTPSessionManager(
            app=self.server,
            event_store=self.event_store,
            stateless=True,
            json_response=True  # Enable JSON responses for better compatibility
        )
        
        # Create SSE transport for SSE endpoint support
        # The endpoint path will be relative to the SSE endpoint itself
        self.sse_transport = SseServerTransport("/messages/")
        
        # Track if session manager is running
        self._session_manager_context = None
    
    def _is_mcp_native_service(self) -> bool:
        """Check if service has native MCP protocol functions."""
        return any(func in self.service for func in [
            "list_tools", "call_tool",
            "list_prompts", "get_prompt", 
            "list_resources", "list_resource_templates", "read_resource"
        ])
    
    def _setup_handlers(self):
        """Set up MCP protocol handlers based on service configuration."""
        service_type = getattr(self.service_info, "type", None)
        
        logger.debug(f"Setting up handlers for service type: {service_type}")
        logger.debug(f"Is native MCP: {self._is_mcp_native_service()}")
        logger.debug(f"Has inline config: {self._has_inline_config()}")
        logger.debug(f"Service keys: {list(self.service.keys()) if isinstance(self.service, dict) else 'Not a dict'}")
        
        # If service explicitly declares type="mcp", validate it follows MCP structure
        if service_type == "mcp":
            if not (self._is_mcp_native_service() or self._has_inline_config()):
                raise ValueError(
                    f"Service '{self.service_info.id}' has type='mcp' but does not follow valid MCP structure. "
                    "Services with type='mcp' must either:\n"
                    "1. Have MCP protocol functions (list_tools, call_tool, etc.), OR\n"
                    "2. Have inline configuration with 'tools', 'resources', or 'prompts' dictionaries.\n"
                    "If your service has arbitrary nested functions, do not specify type='mcp'."
                )
        
        # Check if it's a native MCP service (has MCP protocol functions)
        if self._is_mcp_native_service():
            # Native MCP service - use provided protocol functions
            logger.info("Setting up native MCP handlers")
            self._setup_native_mcp_handlers()
        elif self._has_inline_config():
            # Service with inline configuration (tools/resources/prompts dicts)
            logger.info("Setting up inline config handlers")
            self._setup_inline_config_handlers()
        else:
            # Auto-wrap regular Hypha service functions as MCP tools
            # This handles any service type, including nested structures
            logger.info("Setting up auto-wrapped handlers")
            self._setup_auto_wrapped_handlers()
    
    def _has_inline_config(self) -> bool:
        """Check if service has inline tools/resources/prompts configuration."""
        # Check if the service object has inline configuration
        # Handle both dict and object attribute access
        if isinstance(self.service, dict):
            # Only consider it inline config if tools/resources/prompts are dicts
            tools = self.service.get("tools")
            resources = self.service.get("resources")
            prompts = self.service.get("prompts")
            return any([
                tools and isinstance(tools, dict),
                resources and isinstance(resources, dict),
                prompts and isinstance(prompts, dict)
            ])
        else:
            # For object attributes, check if they are dicts
            tools = getattr(self.service, "tools", None)
            resources = getattr(self.service, "resources", None)
            prompts = getattr(self.service, "prompts", None)
            return any([
                tools and isinstance(tools, dict),
                resources and isinstance(resources, dict),
                prompts and isinstance(prompts, dict)
            ])
    
    def _setup_native_mcp_handlers(self):
        """Setup handlers for native MCP protocol functions."""
        
        # Tools handlers
        if "list_tools" in self.service:
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                result = await self._call_service_function("list_tools", {})
                if isinstance(result, list):
                    return [self._ensure_tool(tool) for tool in result]
                return []
        
        if "call_tool" in self.service:
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                # Let exceptions propagate so MCP SDK can convert them to JSON-RPC errors
                result = await self._call_service_function(
                    "call_tool", {"name": name, "arguments": arguments}
                )
                return self._ensure_content_blocks(result)
        
        # Prompts handlers
        if "list_prompts" in self.service:
            @self.server.list_prompts()
            async def list_prompts() -> List[types.Prompt]:
                result = await self._call_service_function("list_prompts", {})
                if isinstance(result, list):
                    return [self._ensure_prompt(prompt) for prompt in result]
                return []
        
        if "get_prompt" in self.service:
            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                result = await self._call_service_function(
                    "get_prompt", {"name": name, "arguments": arguments}
                )
                return self._ensure_prompt_result(result)
        
        # Resources handlers
        if "list_resource_templates" in self.service:
            @self.server.list_resource_templates()
            async def list_resource_templates() -> List[types.ResourceTemplate]:
                result = await self._call_service_function("list_resource_templates", {})
                if isinstance(result, list):
                    return [self._ensure_resource_template(template) for template in result]
                return []
        
        if "list_resources" in self.service:
            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                result = await self._call_service_function("list_resources", {})
                if isinstance(result, list):
                    return [self._ensure_resource(resource) for resource in result]
                return []
        
        if "read_resource" in self.service:
            # Import the helper type used by MCP SDK
            from mcp.server.lowlevel.helper_types import ReadResourceContents
            
            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> List[ReadResourceContents]:
                result = await self._call_service_function(
                    "read_resource", {"uri": str(uri)}
                )
                # The MCP SDK expects us to return Iterable[ReadResourceContents]
                # Not a ReadResourceResult directly - it will create that for us
                resource_result = self._ensure_read_resource_result(result, uri)
                
                # Convert the ReadResourceResult contents to ReadResourceContents helper objects
                contents = []
                for content in resource_result.contents:
                    if hasattr(content, 'text'):
                        # Text content
                        contents.append(ReadResourceContents(
                            content=content.text,
                            mime_type=getattr(content, 'mimeType', 'text/plain')
                        ))
                    elif hasattr(content, 'blob'):
                        # Binary content - decode base64 and return as bytes
                        import base64
                        contents.append(ReadResourceContents(
                            content=base64.b64decode(content.blob),
                            mime_type=getattr(content, 'mimeType', 'application/octet-stream')
                        ))
                
                return contents
    
    def _setup_inline_config_handlers(self):
        """Setup handlers for inline tools/resources/prompts configuration."""
        
        # Tools handlers
        tools = self.service.get("tools") if isinstance(self.service, dict) else getattr(self.service, "tools", None)
        if tools:
            # Store tool handlers - unwrap ObjectProxy objects and validate them
            for key, tool_func in tools.items():
                unwrapped_func = unwrap_object_proxy(tool_func)
                # Validate the tool function immediately to make errors explode early
                self._extract_tool_from_function(key, unwrapped_func)
                self.tool_handlers[key] = unwrapped_func
            
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                tools = []
                for key, tool_func in self.tool_handlers.items():
                    tool_def = self._extract_tool_from_function(key, tool_func)
                    tools.append(types.Tool(**tool_def))
                return tools
            
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                if name not in self.tool_handlers:
                    raise ValueError(f"Tool '{name}' not found")
                
                tool_func = self.tool_handlers[name]
                result = await self._call_function(tool_func, arguments)
                return [types.TextContent(type="text", text=str(result))]
        
        # Resources handlers
        resources = self.service.get("resources") if isinstance(self.service, dict) else getattr(self.service, "resources", None)
        if resources:
            # Store resource handlers - unwrap ObjectProxy objects and validate them
            for key, resource_config in resources.items():
                unwrapped_config = unwrap_object_proxy(resource_config)
                # Validate the resource configuration immediately to make errors explode early
                if not isinstance(unwrapped_config, dict):
                    raise ValueError(f"Resource '{key}' configuration must be a dictionary")
                if "read" not in unwrapped_config:
                    raise ValueError(f"Resource '{key}' must have a 'read' key with a callable function")
                if not callable(unwrapped_config["read"]):
                    raise ValueError(f"Resource '{key}' 'read' must be a callable function")
                # Validate that the read function is a schema function
                read_func = unwrapped_config["read"]
                if not hasattr(read_func, "__schema__"):
                    raise ValueError(f"Resource '{key}' read function must be a @schema_function decorated function. "
                                   f"Use the @schema_function decorator from hypha_rpc.utils.schema to define the function schema.")
                self.resource_handlers[key] = unwrapped_config
            
            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                resources = []
                for key, config in self.resource_handlers.items():
                    resources.append(types.Resource(
                        uri=config.get("uri", key),
                        name=config.get("name", key),
                        description=config.get("description", ""),
                        mimeType=config.get("mime_type", "text/plain")
                    ))
                return resources
            
            # Import the helper type used by MCP SDK
            from mcp.server.lowlevel.helper_types import ReadResourceContents
            
            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> List[ReadResourceContents]:
                uri_str = str(uri)
                
                # Find resource by URI
                resource_config = None
                for key, config in self.resource_handlers.items():
                    if config.get("uri", key) == uri_str:
                        resource_config = config
                        break
                
                if not resource_config:
                    raise ValueError(f"Resource '{uri_str}' not found")
                
                read_func = resource_config.get("read")
                if not read_func:
                    raise ValueError(f"Resource '{uri_str}' has no read function")
                
                result = await self._call_function(read_func, {})
                # Return as ReadResourceContents helper object
                # The SDK will convert this to TextResourceContents in a ReadResourceResult
                return [ReadResourceContents(
                    content=str(result),
                    mime_type=resource_config.get("mime_type", "text/plain")
                )]
        
        # Prompts handlers
        prompts = self.service.get("prompts") if isinstance(self.service, dict) else getattr(self.service, "prompts", None)
        if prompts:
            # Store prompt handlers - unwrap ObjectProxy objects and validate them
            for key, prompt_config in prompts.items():
                unwrapped_config = unwrap_object_proxy(prompt_config)
                # Validate the prompt configuration immediately to make errors explode early
                if not isinstance(unwrapped_config, dict):
                    raise ValueError(f"Prompt '{key}' configuration must be a dictionary")
                if "read" not in unwrapped_config:
                    raise ValueError(f"Prompt '{key}' must have a 'read' key with a callable function")
                if not callable(unwrapped_config["read"]):
                    raise ValueError(f"Prompt '{key}' 'read' must be a callable function")
                # Validate that the read function is a schema function
                read_func = unwrapped_config["read"]
                if not hasattr(read_func, "__schema__"):
                    raise ValueError(f"Prompt '{key}' read function must be a @schema_function decorated function. "
                                   f"Use the @schema_function decorator from hypha_rpc.utils.schema to define the function schema.")
                self.prompt_handlers[key] = unwrapped_config
            
            @self.server.list_prompts()
            async def list_prompts() -> List[types.Prompt]:
                prompts = []
                for key, config in self.prompt_handlers.items():
                    # Extract arguments from read function if available
                    arguments = []
                    read_func = config.get("read")
                    if read_func and hasattr(read_func, "__schema__"):
                        schema = getattr(read_func, "__schema__", {})
                        parameters = schema.get("parameters", {})
                        properties = parameters.get("properties", {})
                        required = parameters.get("required", [])
                        
                        for prop_name, prop_info in properties.items():
                            arguments.append(types.PromptArgument(
                                name=prop_name,
                                description=prop_info.get("description", ""),
                                required=prop_name in required
                            ))
                    
                    prompts.append(types.Prompt(
                        name=config.get("name", key),
                        description=config.get("description", ""),
                        arguments=arguments
                    ))
                return prompts
            
            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                if name not in self.prompt_handlers:
                    raise ValueError(f"Prompt '{name}' not found")
                
                config = self.prompt_handlers[name]
                read_func = config.get("read")
                if not read_func:
                    raise ValueError(f"Prompt '{name}' has no read function")
                
                result = await self._call_function(read_func, arguments)
                
                # Convert result to GetPromptResult
                if isinstance(result, dict) and "messages" in result:
                    return types.GetPromptResult(**result)
                else:
                    # Simple string result
                    return types.GetPromptResult(
                        description=f"Prompt for {name}",
                        messages=[
                            types.PromptMessage(
                                role="user",
                                content=types.TextContent(type="text", text=str(result))
                            )
                        ]
                    )
    
    def _setup_auto_wrapped_handlers(self):
        """Auto-wrap all service functions as MCP tools and string fields as resources."""
        
        def collect_tools(obj, prefix=""):
            """Recursively collect callable functions from nested structures."""
            collected = {}
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.startswith("_"):
                        continue
                    
                    new_key = f"{prefix}_{key}" if prefix else key
                    
                    if callable(value):
                        # Direct callable at this level
                        collected[new_key] = value
                    elif isinstance(value, (dict, list)):
                        # Nested structure - recurse
                        nested = collect_tools(value, new_key)
                        collected.update(nested)
                    # Skip non-callable, non-container values
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}_{i}" if prefix else str(i)
                    
                    if callable(item):
                        collected[new_key] = item
                    elif isinstance(item, (dict, list)):
                        # Nested structure - recurse
                        nested = collect_tools(item, new_key)
                        collected.update(nested)
                    # Skip non-callable, non-container values
                    
            return collected
        
        def collect_string_resources(obj, prefix="", skip_keys=None):
            """Recursively collect string fields as resources from nested structures."""
            if skip_keys is None:
                skip_keys = {'docs', 'id', 'name', 'description', 'type', 'config'}  # Skip metadata fields
            
            collected = {}
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Skip private keys, metadata, and callables
                    if key.startswith("_") or key in skip_keys or callable(value):
                        continue
                    
                    # Build the resource path
                    new_key = f"{prefix}/{key}" if prefix else key
                    
                    if isinstance(value, str):
                        # Direct string value - add as resource
                        collected[new_key] = value
                    elif isinstance(value, (dict, list)):
                        # Nested structure - recurse
                        nested = collect_string_resources(value, new_key, skip_keys)
                        collected.update(nested)
                    # Skip other types (numbers, booleans, etc.)
                    
            elif isinstance(obj, list):
                # For lists, use index as part of the path
                for i, item in enumerate(obj):
                    new_key = f"{prefix}/{i}" if prefix else str(i)
                    
                    if isinstance(item, str):
                        collected[new_key] = item
                    elif isinstance(item, (dict, list)):
                        # Nested structure - recurse
                        nested = collect_string_resources(item, new_key, skip_keys)
                        collected.update(nested)
                    # Skip other types
                    
            return collected
        
        # Collect all callable functions from the service (including nested)
        tools = collect_tools(self.service)
        
        if tools:
            self.tool_handlers = tools
            
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                tool_list = []
                for key, func in self.tool_handlers.items():
                    tool_def = self._extract_tool_from_function(key, func)
                    tool_list.append(types.Tool(**tool_def))
                return tool_list
            
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                if name not in self.tool_handlers:
                    raise ValueError(f"Tool '{name}' not found")
                
                func = self.tool_handlers[name]
                result = await self._call_function(func, arguments)
                
                # Convert result to content blocks
                if isinstance(result, list):
                    return self._ensure_content_blocks(result)
                else:
                    return [types.TextContent(type="text", text=str(result))]
        
        # Collect string resources (including docs and other string fields)
        string_resources = collect_string_resources(self.service)
        
        # Special handling for docs field - always include if present
        docs = self.service.get("docs") if isinstance(self.service, dict) else getattr(self.service, "docs", None)
        if docs and isinstance(docs, str):
            string_resources['docs'] = docs
        
        if string_resources:
            # Store the string resources for later access
            self.string_resources = string_resources
            
            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                service_name = getattr(self.service_info, 'name', self.service_info.id)
                resources = []
                
                for resource_key, resource_value in self.string_resources.items():
                    # Format the URI with resource:// prefix
                    uri = f"resource://{resource_key}"
                    
                    # Generate descriptive name based on the key
                    if resource_key == 'docs':
                        name = f"Documentation for {service_name}"
                        description = f"Service documentation content for '{service_name}'"
                    else:
                        # Create human-readable name from the key path
                        key_parts = resource_key.split('/')
                        name = ' '.join(part.replace('_', ' ').title() for part in key_parts)
                        description = f"String resource at path '{resource_key}' in service '{service_name}'"
                    
                    resources.append(types.Resource(
                        uri=uri,
                        name=name,
                        description=description,
                        mimeType="text/plain"
                    ))
                
                return resources
            
            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> List[ReadResourceContents]:
                uri_str = str(uri)
                
                # Remove resource:// prefix if present
                if uri_str.startswith("resource://"):
                    resource_key = uri_str[len("resource://"):]
                else:
                    resource_key = uri_str
                
                # Try direct match first
                if resource_key in self.string_resources:
                    content = unwrap_object_proxy(self.string_resources[resource_key])
                    # Return a list of ReadResourceContents with proper fields
                    return [ReadResourceContents(
                        content=str(content),
                        mime_type="text/plain"
                    )]
                
                # Try with slashes replaced by underscores (for compatibility)
                resource_key_with_underscores = resource_key.replace('/', '_')
                if resource_key_with_underscores in self.string_resources:
                    content = unwrap_object_proxy(self.string_resources[resource_key_with_underscores])
                    # Return a list of ReadResourceContents with proper fields
                    return [ReadResourceContents(
                        content=str(content),
                        mime_type="text/plain"
                    )]
                
                raise ValueError(f"Resource '{uri_str}' not found")
    
    def _extract_tool_from_function(self, name: str, func: Callable) -> Dict[str, Any]:
        """Extract MCP tool definition from a function."""
        # Validate that the function is a schema function
        if not hasattr(func, "__schema__"):
            raise ValueError(f"Tool function '{name}' must be a @schema_function decorated function. "
                           f"Use the @schema_function decorator from hypha_rpc.utils.schema to define the function schema.")
        
        schema = getattr(func, "__schema__", {})
        tool_def = {
            "name": name,
            "description": schema.get("description", func.__doc__ or f"Function {name}"),
            "inputSchema": schema.get("parameters", {
                "type": "object",
                "properties": {},
                "required": []
            })
        }
        
        return tool_def
    
    async def _call_service_function(self, function_name: str, kwargs: Dict[str, Any]) -> Any:
        """Call a service function and handle async/sync appropriately."""
        if function_name not in self.service:
            raise ValueError(f"Function {function_name} not found in service")
        
        func = self.service[function_name]
        return await self._call_function(func, kwargs)
    
    async def _call_function(self, func: Callable, kwargs: Dict[str, Any]) -> Any:
        """Call a function (async or sync) and return the result."""
        # Unwrap ObjectProxy if needed
        func = unwrap_object_proxy(func)
        kwargs = unwrap_object_proxy(kwargs)
        
        # Handle ObjectProxy that might still be wrapped
        # Check if it's still an ObjectProxy after unwrapping (happens with nested proxies)
        if hasattr(func, '_rvalue'):
            # Try to call it directly through RPC - let exceptions propagate
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
                if asyncio.isfuture(result) or inspect.iscoroutine(result):
                    result = await result
        else:
            # Normal function call
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
                if asyncio.isfuture(result) or inspect.iscoroutine(result):
                    result = await result
        
        return unwrap_object_proxy(result)
    
    def _ensure_tool(self, tool: Any) -> types.Tool:
        """Ensure the tool is a proper MCP Tool object."""
        tool = unwrap_object_proxy(tool)
        if isinstance(tool, types.Tool):
            return tool
        elif isinstance(tool, dict):
            return types.Tool(**tool)
        else:
            raise ValueError(f"Invalid tool format: {tool}")
    
    def _ensure_prompt(self, prompt: Any) -> types.Prompt:
        """Ensure the prompt is a proper MCP Prompt object."""
        prompt = unwrap_object_proxy(prompt)
        if isinstance(prompt, types.Prompt):
            return prompt
        elif isinstance(prompt, dict):
            return types.Prompt(**prompt)
        else:
            raise ValueError(f"Invalid prompt format: {prompt}")
    
    def _ensure_resource(self, resource: Any) -> types.Resource:
        """Ensure the resource is a proper MCP Resource object."""
        resource = unwrap_object_proxy(resource)
        if isinstance(resource, types.Resource):
            return resource
        elif isinstance(resource, dict):
            return types.Resource(**resource)
        else:
            raise ValueError(f"Invalid resource format: {resource}")
    
    def _ensure_resource_template(self, template: Any) -> types.ResourceTemplate:
        """Ensure the template is a proper MCP ResourceTemplate object."""
        template = unwrap_object_proxy(template)
        if isinstance(template, types.ResourceTemplate):
            return template
        elif isinstance(template, dict):
            return types.ResourceTemplate(**template)
        else:
            raise ValueError(f"Invalid resource template format: {template}")
    
    def _ensure_content_blocks(self, content: Any) -> List[types.ContentBlock]:
        """Ensure the content is a list of proper MCP ContentBlock objects."""
        content = unwrap_object_proxy(content)
        
        if not isinstance(content, list):
            content = [content]
        
        blocks = []
        for item in content:
            if isinstance(item, types.ContentBlock):
                blocks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    blocks.append(types.TextContent(**item))
                else:
                    blocks.append(types.TextContent(type="text", text=str(item)))
            else:
                blocks.append(types.TextContent(type="text", text=str(item)))
        
        return blocks
    
    def _ensure_prompt_result(self, result: Any) -> types.GetPromptResult:
        """Ensure the result is a proper MCP GetPromptResult object."""
        result = unwrap_object_proxy(result)
        
        if isinstance(result, types.GetPromptResult):
            return result
        elif isinstance(result, dict):
            # Convert messages if needed
            if "messages" in result:
                messages = []
                for msg in result["messages"]:
                    if isinstance(msg, types.PromptMessage):
                        messages.append(msg)
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, dict):
                            if content.get("type") == "text":
                                content_obj = types.TextContent(**content)
                            else:
                                content_obj = types.TextContent(type="text", text=str(content))
                        else:
                            content_obj = types.TextContent(type="text", text=str(content))
                        
                        messages.append(types.PromptMessage(
                            role=msg.get("role", "user"),
                            content=content_obj
                        ))
                result["messages"] = messages
            return types.GetPromptResult(**result)
        else:
            # Convert simple string to GetPromptResult
            return types.GetPromptResult(
                description="Prompt result",
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=str(result))
                    )
                ]
            )
    
    def _ensure_read_resource_result(self, result: Any, uri: AnyUrl) -> types.ReadResourceResult:
        """Ensure the result is a proper MCP ReadResourceResult object."""
        result = unwrap_object_proxy(result)
        
        # Log for debugging
        logger.debug(f"_ensure_read_resource_result: type={type(result)}, result={result}")
        
        if isinstance(result, types.ReadResourceResult):
            return result
        elif isinstance(result, (dict, list, tuple)):
            # Handle case where result might be serialized as tuple/list
            if isinstance(result, (list, tuple)):
                # Try to convert to dict if it looks like a serialized object
                if len(result) > 0 and isinstance(result[0], dict) and "contents" in result[0]:
                    result = result[0]
                elif len(result) > 0 and isinstance(result[0], str):
                    # It's a tuple of strings, treat as text content
                    result = {
                        "contents": [
                            {
                                "uri": str(uri),
                                "mimeType": "text/plain",
                                "text": str(result[0]) if len(result) == 1 else str(result)
                            }
                        ]
                    }
                else:
                    # Assume it's just a content list
                    result = {"contents": result}
            
            if isinstance(result, dict) and "contents" in result:
                contents = []
                for content in result["contents"]:
                    if isinstance(content, dict):
                        # Ensure uri is a string, not AnyUrl object
                        if "uri" in content and not isinstance(content["uri"], str):
                            content["uri"] = str(content["uri"])
                        # Remove None/meta values that might cause issues
                        cleaned_content = {
                            "uri": content.get("uri"),
                            "mimeType": content.get("mimeType", "text/plain"),
                            "text": content.get("text", "")
                        }
                        # Only add meta if it's not None
                        if content.get("meta") is not None:
                            cleaned_content["meta"] = content["meta"]
                        contents.append(types.TextResourceContents(**cleaned_content))
                result["contents"] = contents
                # Remove meta from result if it's None
                if result.get("meta") is None:
                    result = {"contents": result["contents"]}
            return types.ReadResourceResult(**result)
        else:
            # Convert simple string to ReadResourceResult
            return types.ReadResourceResult(
                contents=[
                    types.TextResourceContents(
                        uri=uri,
                        mimeType="text/plain",
                        text=str(result)
                    )
                ]
            )
    
    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle incoming HTTP request using the session manager."""
        # The session manager should be running continuously, not per request
        # Just handle the request directly
        await self.session_manager.handle_request(scope, receive, send)
    
    async def handle_sse_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle incoming SSE request."""
        # Connect SSE and run the server with the provided streams
        async with self.sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            # Run the MCP server with the SSE streams
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )
    
    async def handle_sse_message(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle incoming SSE message POST request."""
        await self.sse_transport.handle_post_message(scope, receive, send)
    
    def get_asgi_app(self) -> ASGIApp:
        """Get ASGI application for this MCP service."""
        async def app(scope: Scope, receive: Receive, send: Send) -> None:
            await self.handle_request(scope, receive, send)
        return app


class MCPSessionWrapper:
    """Wrapper that manages the MCP session manager lifecycle."""
    
    def __init__(self, adapter: HyphaMCPAdapter):
        self.adapter = adapter
        self.session_context = None
        self.session_task = None
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle request, ensuring session manager is running."""
        # Start session manager if not already running
        if self.session_context is None:
            self.session_context = self.adapter.session_manager.run()
            self.session_task = asyncio.create_task(self.session_context.__aenter__())
            # Wait a bit for it to start
            await asyncio.sleep(0.01)
        
        # Handle the request
        try:
            await self.adapter.handle_request(scope, receive, send)
        except Exception as e:
            # If session manager is not running, try to restart it
            if "Task group is not initialized" in str(e):
                # Clean up old context
                if self.session_context:
                    try:
                        await self.session_context.__aexit__(None, None, None)
                    except:
                        pass
                    self.session_context = None
                    self.session_task = None
                
                # Start new context
                self.session_context = self.adapter.session_manager.run()
                self.session_task = asyncio.create_task(self.session_context.__aenter__())
                await asyncio.sleep(0.01)
                
                # Retry the request
                await self.adapter.handle_request(scope, receive, send)
            else:
                raise


async def create_mcp_app_from_service(service, service_info, redis_client):
    """Create MCP ASGI application from a Hypha service.
    
    Args:
        service: Hypha service object
        service_info: Service information from Hypha
        redis_client: Redis client for event store
    
    Returns:
        ASGI application that can be called with (scope, receive, send)
    """
    logger.info(f"Creating MCP app for service: {service_info.id}")
    
    # Create MCP adapter
    adapter = HyphaMCPAdapter(service, service_info, redis_client)
    
    # Return wrapped ASGI app that manages session lifecycle
    return MCPSessionWrapper(adapter)


def is_mcp_compatible_service(service) -> bool:
    """Check if a service is MCP compatible.
    
    Any service can be MCP compatible - we'll auto-wrap its functions as MCP tools.
    Services with explicit type="mcp" or MCP protocol functions get special handling.
    """
    # All services are MCP compatible - we can auto-wrap any service
    return True


class MCPRoutingMiddleware:
    """Middleware specifically for routing MCP services."""
    
    def __init__(self, app: ASGIApp, base_path=None, store=None):
        self.app = app
        self.base_path = base_path.rstrip("/") if base_path else ""
        self.store = store
        
        # Cache MCP adapters and their API connections
        self._mcp_cache = {}  # cache_key -> (mcp_app, api_context, api_task)
        
        # MCP route patterns
        # Main route for streamable HTTP: /{workspace}/mcp/{service_id}/mcp
        mcp_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}/mcp"
        )
        self.mcp_route = Route(mcp_route, endpoint=None)
        
        # Info route for helpful 404: /{workspace}/mcp/{service_id}
        info_route = self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}"
        self.info_route = Route(info_route, endpoint=None)
        
        # SSE routes: /{workspace}/mcp/{service_id}/sse and /sse/messages/
        sse_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}/sse"
        )
        self.sse_route = Route(sse_route, endpoint=None)
        
        # SSE message route for POST messages
        sse_message_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}/sse/messages/"
        )
        self.sse_message_route = Route(sse_message_route, endpoint=None)
        
        logger.info(f"MCP middleware initialized with route patterns:")
        logger.info(f"  - Streamable HTTP: {self.mcp_route.path}")
        logger.info(f"  - SSE: {self.sse_route.path}")
    
    def _get_authorization_header(self, scope):
        """Extract the Authorization header from the scope."""
        for key, value in scope.get("headers", []):
            if key.decode("utf-8").lower() == "authorization":
                return value.decode("utf-8")
        return None
    
    def _get_access_token_from_cookies(self, scope):
        """Extract the access_token cookie from the scope."""
        try:
            cookies = None
            for key, value in scope.get("headers", []):
                if key.decode("utf-8").lower() == "cookie":
                    cookies = value.decode("utf-8")
                    break
            if not cookies:
                return None
            # Split cookies and find the access_token
            cookie_dict = {
                k.strip(): v.strip()
                for k, v in (cookie.split("=") for cookie in cookies.split(";"))
            }
            return cookie_dict.get("access_token")
        except Exception:
            return None
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            logger.debug(f"MCP Middleware: Processing HTTP request to {path}")
            
            # Check if the current request path matches any MCP route
            from starlette.routing import Match
            
            # Check SSE message route first (/{workspace}/mcp/{service_id}/sse/messages/) - most specific
            sse_message_match, sse_message_params = self.sse_message_route.matches(scope)
            if sse_message_match == Match.FULL:
                path_params = sse_message_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                
                logger.debug(
                    f"MCP Middleware: SSE message route match - workspace='{workspace}', service_id='{service_id}'"
                )
                
                # Handle SSE message POST request
                await self._handle_sse_message_request(
                    scope, receive, send, workspace, service_id
                )
                return
            
            # Check SSE route (/{workspace}/mcp/{service_id}/sse) - specific
            sse_match, sse_params = self.sse_route.matches(scope)
            if sse_match == Match.FULL:
                path_params = sse_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                
                logger.debug(
                    f"MCP Middleware: SSE route match - workspace='{workspace}', service_id='{service_id}'"
                )
                
                # Handle SSE request
                await self._handle_sse_request(
                    scope, receive, send, workspace, service_id
                )
                return
            
            # Check MCP route (/{workspace}/mcp/{service_id}/mcp) - specific
            mcp_match, mcp_params = self.mcp_route.matches(scope)
            if mcp_match == Match.FULL:
                path_params = mcp_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                
                logger.debug(
                    f"MCP Middleware: MCP route match - workspace='{workspace}', service_id='{service_id}'"
                )
                
                # Handle MCP streamable HTTP request
                await self._handle_mcp_request(
                    scope, receive, send, workspace, service_id
                )
                return
            
            # Check info route last (/{workspace}/mcp/{service_id}) - most general
            info_match, info_params = self.info_route.matches(scope)
            if info_match == Match.FULL:
                path_params = info_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]
                
                logger.debug(
                    f"MCP Middleware: Info route match - workspace='{workspace}', service_id='{service_id}'"
                )
                
                # Return helpful 404 message
                await self._send_helpful_404(send, workspace, service_id)
                return
        
        # Continue to next middleware if not an MCP route
        await self.app(scope, receive, send)
    
    async def _send_helpful_404(self, send, workspace, service_id):
        """Send a helpful 404 message with endpoint information."""
        message = {
            "error": "MCP endpoint not found",
            "message": f"The MCP service '{service_id}' endpoint was not found at this path.",
            "available_endpoints": {
                "streamable_http": f"/{workspace}/mcp/{service_id}/mcp",
                "sse": f"/{workspace}/mcp/{service_id}/sse",
            },
            "help": "Use the streamable HTTP endpoint for MCP communication.",
        }
        response_body = json.dumps(message, indent=2).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(response_body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )
    
    async def _handle_mcp_request(self, scope, receive, send, workspace, service_id):
        """Handle MCP streamable HTTP requests."""
        try:
            # Prepare scope for the MCP service
            scope = {
                k: scope[k]
                for k in scope
                if isinstance(
                    scope[k],
                    (str, int, float, bool, tuple, list, dict, bytes),
                )
            }
            # Set path to root for MCP service
            scope["path"] = "/"
            scope["raw_path"] = b"/"
            
            # Get _mode from query string
            query = scope.get("query_string", b"").decode("utf-8")
            if query:
                _mode = dict([q.split("=") for q in query.split("&")]).get("_mode")
            else:
                _mode = None
            
            # Get authentication info
            # Create a mock Request object with the scope
            request = Request(scope, receive=receive, send=send)
            
            # Login and get user info
            user_info = await self.store.login_optional(request)
            
            # Check if we have a cached MCP app for this service
            cache_key = f"{workspace}/{service_id}"
            
            if cache_key in self._mcp_cache:
                # Use cached MCP app
                mcp_app, api_context, api_task = self._mcp_cache[cache_key]
                try:
                    # Handle the request with the cached MCP app
                    await mcp_app(scope, receive, send)
                except Exception as e:
                    # If the cached connection failed, clean it up and retry
                    logger.warning(f"Cached MCP app failed, recreating: {e}")
                    
                    # Clean up old context
                    try:
                        if api_context:
                            await api_context.__aexit__(None, None, None)
                    except:
                        pass
                    del self._mcp_cache[cache_key]
                    
                    # Retry with fresh connection
                    await self._handle_mcp_request(scope, receive, send, workspace, service_id)
            else:
                # Create new MCP app with persistent API connection
                api_context_manager = self.store.get_workspace_interface(user_info, workspace)
                api_context = api_context_manager.__aenter__()
                api = await api_context
                
                try:
                    service_info = await api.get_service_info(
                        service_id, {"mode": _mode}
                    )
                    logger.debug(
                        f"MCP Middleware: Found service '{service_id}' of type '{service_info.type}'"
                    )
                except (KeyError, Exception) as e:
                    # Clean up on error
                    await api_context_manager.__aexit__(None, None, None)
                    logger.error(f"MCP Middleware: Service lookup failed: {e}")
                    if "Service not found" in str(e):
                        await self._send_error_response(
                            send, 404, f"Service {service_id} not found"
                        )
                        return
                    else:
                        raise
                
                # No longer require explicit type="mcp" - any service can be exposed as MCP
                # Services with type="mcp" get special handling, others are auto-wrapped
                
                service = await api.get_service(service_info.id)
                
                # Check if it's MCP compatible
                if not is_mcp_compatible_service(service):
                    await api_context_manager.__aexit__(None, None, None)
                    await self._send_error_response(
                        send, 400, f"Service {service_id} is not MCP compatible"
                    )
                    return
                
                # Create MCP app with the service
                mcp_app = await create_mcp_app_from_service(
                    service, service_info, self.store.get_redis()
                )
                
                # Cache the MCP app and API connection
                self._mcp_cache[cache_key] = (mcp_app, api_context_manager, api)
                
                # Handle the request with the MCP app
                await mcp_app(scope, receive, send)
        
        except Exception as exp:
            logger.exception(f"Error in MCP service: {exp}")
            await self._send_error_response(send, 500, f"Internal Server Error: {exp}")
    
    async def _send_error_response(self, send, status_code, message):
        """Send an error response."""
        response_body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(response_body)).encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )
    
    async def _handle_sse_request(self, scope, receive, send, workspace, service_id):
        """Handle MCP SSE requests."""
        try:
            # Prepare scope for the MCP service
            scope = {
                k: scope[k]
                for k in scope
                if isinstance(
                    scope[k],
                    (str, int, float, bool, tuple, list, dict, bytes),
                )
            }
            # Set path to root for MCP SSE service
            scope["path"] = "/"
            scope["raw_path"] = b"/"
            # Set root_path to include the full SSE endpoint prefix
            # This allows the SSE transport to correctly construct the message endpoint
            scope["root_path"] = f"/{workspace}/mcp/{service_id}/sse"
            
            # Get authentication info
            # Create a mock Request object with the scope
            request = Request(scope, receive=receive, send=send)
            
            # Login and get user info
            user_info = await self.store.login_optional(request)
            
            # Check if we have a cached MCP app for this service
            cache_key = f"{workspace}/{service_id}"
            
            if cache_key in self._mcp_cache:
                # Use cached MCP app
                mcp_app, api_context, api_task = self._mcp_cache[cache_key]
                adapter = mcp_app.adapter if hasattr(mcp_app, 'adapter') else mcp_app
                try:
                    # Handle the SSE request with the cached MCP app
                    await adapter.handle_sse_request(scope, receive, send)
                except Exception as e:
                    # If the cached connection failed, clean it up and retry
                    logger.warning(f"Cached MCP app failed for SSE, recreating: {e}")
                    
                    # Clean up old context
                    try:
                        if api_context:
                            await api_context.__aexit__(None, None, None)
                    except:
                        pass
                    del self._mcp_cache[cache_key]
                    
                    # Retry with fresh connection
                    await self._handle_sse_request(scope, receive, send, workspace, service_id)
            else:
                # Create new MCP app with persistent API connection
                api_context_manager = self.store.get_workspace_interface(user_info, workspace)
                api_context = api_context_manager.__aenter__()
                api = await api_context
                
                try:
                    service_info = await api.get_service_info(
                        service_id, {"mode": None}
                    )
                    logger.debug(
                        f"MCP Middleware: Found service '{service_id}' of type '{service_info.type}'"
                    )
                except (KeyError, Exception) as e:
                    # Clean up on error
                    await api_context_manager.__aexit__(None, None, None)
                    logger.error(f"MCP Middleware: Service lookup failed: {e}")
                    if "Service not found" in str(e):
                        await self._send_error_response(
                            send, 404, f"Service {service_id} not found"
                        )
                        return
                    else:
                        raise
                
                # No longer require explicit type="mcp" - any service can be exposed as MCP
                # Services with type="mcp" get special handling, others are auto-wrapped
                
                service = await api.get_service(service_info.id)
                
                # Check if it's MCP compatible
                if not is_mcp_compatible_service(service):
                    await api_context_manager.__aexit__(None, None, None)
                    await self._send_error_response(
                        send, 400, f"Service {service_id} is not MCP compatible"
                    )
                    return
                
                # Create MCP app with the service
                mcp_app = await create_mcp_app_from_service(
                    service, service_info, self.store.get_redis()
                )
                
                # Cache the MCP app and API connection
                self._mcp_cache[cache_key] = (mcp_app, api_context_manager, api)
                
                # Handle the SSE request with the MCP app
                adapter = mcp_app.adapter if hasattr(mcp_app, 'adapter') else mcp_app
                await adapter.handle_sse_request(scope, receive, send)
        
        except Exception as exp:
            logger.exception(f"Error in MCP SSE service: {exp}")
            await self._send_error_response(send, 500, f"Internal Server Error: {exp}")
    
    async def _handle_sse_message_request(self, scope, receive, send, workspace, service_id):
        """Handle MCP SSE message POST requests."""
        try:
            # Get authentication info
            # Create a mock Request object with the scope
            request = Request(scope, receive=receive, send=send)
            
            # Login and get user info
            user_info = await self.store.login_optional(request)
            
            # Check if we have a cached MCP app for this service
            cache_key = f"{workspace}/{service_id}"
            
            if cache_key not in self._mcp_cache:
                await self._send_error_response(
                    send, 404, f"SSE session not found for service {service_id}"
                )
                return
            
            # Use cached MCP app
            mcp_app, api_context, api_task = self._mcp_cache[cache_key]
            adapter = mcp_app.adapter if hasattr(mcp_app, 'adapter') else mcp_app
            
            try:
                # Handle the SSE message POST request
                await adapter.handle_sse_message(scope, receive, send)
            except Exception as e:
                logger.error(f"Error handling SSE message: {e}")
                await self._send_error_response(send, 500, f"Error processing message: {e}")
        
        except Exception as exp:
            logger.exception(f"Error in MCP SSE message service: {exp}")
            await self._send_error_response(send, 500, f"Internal Server Error: {exp}")