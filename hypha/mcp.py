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
import anyio
from dataclasses import dataclass
from uuid import uuid4
import time

from starlette.routing import Route, Match
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
    ) -> Optional[StreamId]:
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
        
        # Track excluded functions and warnings
        self.excluded_functions = []
        self.warnings = []
        
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
        
        # Track SSE connections for proper cleanup
        self._sse_contexts = {}
    
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
            # For MCP services, strict validation - let exceptions propagate
            if getattr(self.service_info, "type", None) == "mcp":
                self.tool_handlers = {}
                for key, func in tools.items():
                    # This will raise an exception if the function doesn't have proper schema
                    tool_def = self._extract_tool_from_function(key, func)
                    if tool_def:
                        self.tool_handlers[key] = func
            else:
                # For non-MCP services (auto-wrapped), filter out tools without valid schemas
                valid_tools = {}
                skipped_tools = []
                for key, func in tools.items():
                    try:
                        tool_def = self._extract_tool_from_function(key, func)
                        if tool_def:
                            valid_tools[key] = func
                        else:
                            skipped_tools.append(key)
                            self.excluded_functions.append({
                                "name": key,
                                "reason": "No schema information available. Functions must be decorated with @schema_function or have schema in service_schema."
                            })
                    except ValueError:
                        # This shouldn't happen for non-MCP services, but handle it gracefully
                        skipped_tools.append(key)
                        self.excluded_functions.append({
                            "name": key,
                            "reason": "Failed to extract schema information."
                        })
                
                if skipped_tools:
                    logger.debug(f"Skipped {len(skipped_tools)} functions without schemas: {skipped_tools}")
                    self.warnings.append(f"Excluded {len(skipped_tools)} functions without proper schemas. Use @schema_function decorator to expose them via MCP.")
                
                self.tool_handlers = valid_tools
            
            if self.tool_handlers:
                @self.server.list_tools()
                async def list_tools() -> List[types.Tool]:
                    tool_list = []
                    for key, func in self.tool_handlers.items():
                        tool_def = self._extract_tool_from_function(key, func)
                        if tool_def:  # Double-check in case something changed
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
        # Check if function has schema (from @schema_function decorator)
        if hasattr(func, "__schema__"):
            schema = func.__schema__
            # For services explicitly marked as MCP type, enforce strict validation
            if getattr(self.service_info, "type", None) == "mcp" and not schema:
                raise ValueError(f"Tool function '{name}' must be a @schema_function decorated function. "
                               f"Use the @schema_function decorator from hypha_rpc.utils.schema to define the function schema.")
            
            if schema:  # Only process if schema is not None
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
        
        # For MCP services, functions without __schema__ are invalid
        if getattr(self.service_info, "type", None) == "mcp":
            raise ValueError(f"Tool function '{name}' must be a @schema_function decorated function. "
                           f"Use the @schema_function decorator from hypha_rpc.utils.schema to define the function schema.")
        
        # For non-MCP services (auto-wrapped), try to extract from service_schema if available
        # This handles services that have schema information but not as decorators
        if isinstance(self.service, dict) and "service_schema" in self.service:
            service_schema = self.service.get("service_schema", {})
            if name in service_schema and service_schema[name]:
                func_schema = service_schema[name]
                # Check if func_schema is a dict before trying to access its properties
                if isinstance(func_schema, dict) and func_schema.get("type") == "function":
                    function_def = func_schema.get("function", {})
                    if function_def and isinstance(function_def, dict):
                        tool_def = {
                            "name": name,
                            "description": function_def.get("description", func.__doc__ or f"Function {name}"),
                            "inputSchema": function_def.get("parameters", {
                                "type": "object",
                                "properties": {},
                                "required": []
                            })
                        }
                        return tool_def
        
        # If no schema available for non-MCP services, return None to indicate this function should be skipped
        return None
    
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
    
    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get a summary of the MCP adapter including warnings and excluded functions."""
        summary = {
            "service_id": self.service_info.id,
            "service_type": getattr(self.service_info, "type", "unknown"),
            "tools_count": len(self.tool_handlers),
            "resources_count": len(getattr(self, "string_resources", {})),
            "prompts_count": len(self.prompt_handlers),
        }
        
        if self.warnings:
            summary["warnings"] = self.warnings
        
        if self.excluded_functions:
            summary["excluded_functions"] = self.excluded_functions
            
        return summary
    
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
        # In stateless mode, each request is handled independently
        # The session manager will create its own context internally for each request
        try:
            async with self.session_manager.run():
                await self.session_manager.handle_request(scope, receive, send)
        except anyio.ClosedResourceError:
            # This can happen if the client disconnects early - ignore
            pass
        except Exception as e:
            raise
    
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
        
        # Note: We don't call server.run() here because SSE works differently:
        # 1. The GET request establishes the SSE connection and returns headers
        # 2. Subsequent POST requests to /messages/ handle the actual communication
        # 3. The transport manages the event stream lifecycle
    
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
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle request independently for stateless mode."""
        # In stateless mode, we handle each request independently
        # The session manager will handle its own context internally
        await self.adapter.handle_request(scope, receive, send)


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
        
        # Cache MCP apps by full service ID to maintain SSE sessions
        self.mcp_app_cache = {}
        # Track active SSE connections for cleanup
        self.active_sse_connections = {}
        
        # MCP route patterns
        # Redirect route for base MCP path without trailing slash: /{workspace}/mcp
        mcp_base_redirect_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp"
        )
        self.mcp_base_redirect_route = Route(mcp_base_redirect_route, endpoint=None)
        
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
    
    async def _get_or_create_mcp_app(self, service_id: str, workspace: str, user_info, mode=None, is_sse=False):
        """Get cached MCP app or create new one.
        
        Args:
            service_id: Service identifier
            workspace: Workspace name
            user_info: User information
            mode: Optional mode parameter
            is_sse: If True, keeps workspace interface alive for SSE sessions
        """
        cache_key = f"{workspace}/{service_id}"
        
        if cache_key in self.mcp_app_cache:
            logger.debug(f"Using cached MCP app for {cache_key}")
            cache_entry = self.mcp_app_cache[cache_key]
            return cache_entry["app"] if isinstance(cache_entry, dict) else cache_entry
        
        logger.debug(f"Creating new MCP app for {cache_key}")
        
        if is_sse:
            # For SSE, keep workspace interface alive
            api = await self.store.get_workspace_interface(
                user_info, user_info.scope.current_workspace
            ).__aenter__()
            
            try:
                # Build full service ID
                if "/" in service_id:
                    full_service_id = service_id
                elif ":" in service_id:
                    full_service_id = f"{workspace}/{service_id}"
                else:
                    full_service_id = f"{workspace}/{service_id}"
                
                service_info = await api.get_service_info(full_service_id, {"mode": mode})
                service = await api.get_service(full_service_id)
                
                if not is_mcp_compatible_service(service):
                    await api.__aexit__(None, None, None)
                    raise ValueError(f"Service {service_id} is not MCP compatible")
                
                mcp_app = await create_mcp_app_from_service(
                    service, service_info, self.store.get_redis()
                )
                
                # Cache the app AND the API connection for SSE
                self.mcp_app_cache[cache_key] = {
                    "app": mcp_app,
                    "api": api  # Keep the API connection alive for SSE
                }
                return mcp_app
            except Exception:
                await api.__aexit__(None, None, None)
                raise
        else:
            # For non-SSE, use normal context manager
            async with self.store.get_workspace_interface(
                user_info, user_info.scope.current_workspace
            ) as api:
                # Build full service ID
                if "/" in service_id:
                    full_service_id = service_id
                elif ":" in service_id:
                    full_service_id = f"{workspace}/{service_id}"
                else:
                    full_service_id = f"{workspace}/{service_id}"
                
                service_info = await api.get_service_info(full_service_id, {"mode": mode})
                service = await api.get_service(full_service_id)
                
                if not is_mcp_compatible_service(service):
                    raise ValueError(f"Service {service_id} is not MCP compatible")
                
                mcp_app = await create_mcp_app_from_service(
                    service, service_info, self.store.get_redis()
                )
                
                # Cache only the MCP app for non-SSE
                self.mcp_app_cache[cache_key] = mcp_app
                return mcp_app
    
    async def _cleanup_mcp_app(self, cache_key: str):
        """Clean up cached MCP app."""
        if cache_key in self.mcp_app_cache:
            logger.debug(f"Cleaning up cached MCP app for {cache_key}")
            cache_entry = self.mcp_app_cache[cache_key]
            
            # If it's a dict with an API connection (SSE case), clean it up
            if isinstance(cache_entry, dict) and "api" in cache_entry:
                try:
                    await cache_entry["api"].__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error cleaning up API connection: {e}")
            
            del self.mcp_app_cache[cache_key]
    
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
            
            # Check for base MCP path redirect (/{workspace}/mcp -> /{workspace}/mcp/)
            redirect_match, redirect_params = self.mcp_base_redirect_route.matches(scope)
            if redirect_match == Match.FULL:
                path_params = redirect_params.get("path_params", {})
                workspace = path_params["workspace"]
                
                logger.debug(
                    f"MCP Middleware: Redirecting from /{workspace}/mcp to /{workspace}/mcp/"
                )
                
                # Send 301 redirect to the path with trailing slash
                redirect_url = f"/{workspace}/mcp/"
                await self._send_redirect(send, redirect_url)
                return
            
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
                
                # Return comprehensive service information
                await self._send_service_info(send, workspace, service_id)
                return
        
        # Continue to next middleware if not an MCP route
        await self.app(scope, receive, send)
    
    async def _send_redirect(self, send, redirect_url):
        """Send a 301 redirect response."""
        await send(
            {
                "type": "http.response.start",
                "status": 301,
                "headers": [
                    [b"location", redirect_url.encode()],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"",
            }
        )

    async def _send_service_info(self, send, workspace, service_id):
        """Send comprehensive service information including available tools, resources, and prompts."""
        # Handle empty service_id case
        if not service_id:
            message = {
                "error": "MCP service ID required",
                "message": "No service ID was provided in the URL.",
                "help": "Please specify a service ID. URL format: /{workspace}/mcp/{service_id}/mcp",
                "example": f"/{workspace}/mcp/your-service-id/mcp"
            }
            status_code = 404
        else:
            try:
                # Get authentication info
                from starlette.requests import Request
                
                # Create a minimal scope for auth check
                scope = {
                    "type": "http",
                    "method": "GET",
                    "headers": [],
                    "query_string": b"",
                }
                
                # Login and get user info (we need to get auth from somewhere)
                # For now, we'll try to get the service info without full auth
                # This is safe because we're only showing metadata, not executing anything
                
                # Try to get the service and create an adapter to extract capabilities
                try:
                    # Import required modules
                    from starlette.requests import Request
                    
                    # Create a mock request for auth
                    request = Request(scope, receive=None, send=None)
                    user_info = await self.store.login_optional(request)
                    
                    # Get service through workspace interface
                    async with self.store.get_workspace_interface(
                        user_info, user_info.scope.current_workspace
                    ) as api:
                        # Build full service ID
                        if "/" in service_id:
                            full_service_id = service_id
                        elif ":" in service_id:
                            full_service_id = f"{workspace}/{service_id}"
                        else:
                            full_service_id = f"{workspace}/{service_id}"
                        
                        # Try to get service info
                        try:
                            service_info = await api.get_service_info(full_service_id, {})
                            service = await api.get_service(full_service_id)
                            
                            # Create a temporary adapter to extract capabilities
                            adapter = HyphaMCPAdapter(service, service_info, self.store.get_redis())
                            
                            # Get the adapter summary
                            adapter_summary = adapter.get_adapter_summary()
                            
                            # Extract detailed tool information
                            tools = []
                            for tool_name, tool_func in adapter.tool_handlers.items():
                                tool_def = adapter._extract_tool_from_function(tool_name, tool_func)
                                if tool_def:
                                    tools.append({
                                        "name": tool_def["name"],
                                        "description": tool_def["description"],
                                        "inputSchema": tool_def.get("inputSchema", {})
                                    })
                            
                            # Extract resource information
                            resources = []
                            if hasattr(adapter, 'string_resources'):
                                for resource_key in adapter.string_resources.keys():
                                    uri = f"resource://{resource_key}"
                                    if resource_key == 'docs':
                                        name = f"Documentation for {service_info.name or service_info.id}"
                                        description = f"Service documentation content"
                                    else:
                                        key_parts = resource_key.split('/')
                                        name = ' '.join(part.replace('_', ' ').title() for part in key_parts)
                                        description = f"String resource at path '{resource_key}'"
                                    
                                    resources.append({
                                        "uri": uri,
                                        "name": name,
                                        "description": description,
                                        "mimeType": "text/plain"
                                    })
                            
                            # Extract prompt information
                            prompts = []
                            for prompt_name, prompt_config in adapter.prompt_handlers.items():
                                prompt_info = {
                                    "name": prompt_config.get("name", prompt_name),
                                    "description": prompt_config.get("description", ""),
                                }
                                
                                # Extract arguments if available
                                read_func = prompt_config.get("read")
                                if read_func and hasattr(read_func, "__schema__"):
                                    schema = getattr(read_func, "__schema__", {})
                                    parameters = schema.get("parameters", {})
                                    properties = parameters.get("properties", {})
                                    required = parameters.get("required", [])
                                    
                                    arguments = []
                                    for prop_name, prop_info in properties.items():
                                        arguments.append({
                                            "name": prop_name,
                                            "description": prop_info.get("description", ""),
                                            "required": prop_name in required
                                        })
                                    if arguments:
                                        prompt_info["arguments"] = arguments
                                
                                prompts.append(prompt_info)
                            
                            # Build comprehensive response
                            message = {
                                "service": {
                                    "id": service_info.id,
                                    "name": getattr(service_info, 'name', service_info.id),
                                    "type": getattr(service_info, 'type', 'auto-wrapped'),
                                    "description": getattr(service_info, 'description', 'Hypha service exposed via MCP'),
                                },
                                "endpoints": {
                                    "streamable_http": f"/{workspace}/mcp/{service_id}/mcp",
                                    "sse": f"/{workspace}/mcp/{service_id}/sse",
                                },
                                "capabilities": {
                                    "tools": tools,
                                    "resources": resources,
                                    "prompts": prompts,
                                },
                                "summary": adapter_summary,
                                "help": "Use the streamable HTTP endpoint for MCP communication or SSE for server-sent events.",
                            }
                            status_code = 200
                            
                        except Exception as e:
                            if "not found" in str(e).lower() or "permission denied" in str(e).lower():
                                # Service not found or not accessible
                                message = {
                                    "error": "Service not found or not accessible",
                                    "message": f"The service '{service_id}' was not found in workspace '{workspace}' or you don't have permission to access it.",
                                    "help": "Ensure the service exists and you have permission to access it.",
                                }
                                status_code = 404
                            else:
                                raise
                            
                except Exception as e:
                    logger.warning(f"Failed to get service info for {service_id}: {e}")
                    # Fallback to basic info without service details
                    message = {
                        "service": {
                            "id": service_id,
                            "workspace": workspace,
                            "status": "unknown",
                        },
                        "endpoints": {
                            "streamable_http": f"/{workspace}/mcp/{service_id}/mcp",
                            "sse": f"/{workspace}/mcp/{service_id}/sse",
                        },
                        "help": "Use the streamable HTTP endpoint for MCP communication.",
                        "note": "Could not retrieve detailed service information. The service may not be running or accessible.",
                    }
                    status_code = 200
                    
            except Exception as e:
                logger.error(f"Error getting service info: {e}")
                message = {
                    "error": "Failed to retrieve service information",
                    "message": str(e),
                    "endpoints": {
                        "streamable_http": f"/{workspace}/mcp/{service_id}/mcp",
                        "sse": f"/{workspace}/mcp/{service_id}/sse",
                    },
                }
                status_code = 500
        
        response_body = json.dumps(message, indent=2).encode()
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
    
    async def _handle_mcp_request(self, scope, receive, send, workspace, service_id):
        """Handle regular MCP HTTP requests (JSON-RPC)."""
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
            
            # Create new MCP app for each request
            # Use the user's own workspace for the interface
            # This allows us to access services in other workspaces if they're public/accessible
            async with self.store.get_workspace_interface(
                user_info, user_info.scope.current_workspace
            ) as api:
                try:
                    # Build the full service ID based on what's provided
                    # Service IDs can be in several formats:
                    # 1. Simple service name: "my-service"
                    # 2. Client-qualified: "client-id:service-name"
                    # 3. Workspace-qualified: "workspace/service-name" or "workspace/client-id:service-name"
                    
                    if "/" in service_id:
                        # Already has workspace prefix, use as-is
                        full_service_id = service_id
                    elif ":" in service_id:
                        # Has client ID but no workspace, prepend workspace
                        full_service_id = f"{workspace}/{service_id}"
                    else:
                        # Simple service name, prepend workspace
                        full_service_id = f"{workspace}/{service_id}"
                    
                    service_info = await api.get_service_info(
                        full_service_id, {"mode": _mode}
                    )
                    logger.debug(
                        f"MCP Middleware: Found service '{service_id}' of type '{service_info.type}'"
                    )
                except (KeyError, Exception) as e:
                    logger.error(f"MCP Middleware: Service lookup failed: {e}")
                    if "Service not found" in str(e) or "Permission denied" in str(e):
                        await self._send_error_response(
                            send, 404, f"Service {service_id} not found or not accessible"
                        )
                        return
                    else:
                        raise
                
                # No longer require explicit type="mcp" - any service can be exposed as MCP
                # Services with type="mcp" get special handling, others are auto-wrapped
                
                service = await api.get_service(full_service_id)
                
                # Check if it's MCP compatible
                if not is_mcp_compatible_service(service):
                    await self._send_error_response(
                        send, 400, f"Service {service_id} is not MCP compatible"
                    )
                    return
                
                # Create MCP app with the service
                mcp_app = await create_mcp_app_from_service(
                    service, service_info, self.store.get_redis()
                )
                
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
        cache_key = f"{workspace}/{service_id}"
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
            
            # Get or create cached MCP app (with is_sse=True to keep workspace interface alive)
            try:
                mcp_app = await self._get_or_create_mcp_app(
                    service_id, workspace, user_info, mode=None, is_sse=True
                )
            except (KeyError, Exception) as e:
                logger.error(f"MCP Middleware: Service lookup failed: {e}")
                if "Service not found" in str(e) or "Permission denied" in str(e):
                    await self._send_error_response(
                        send, 404, f"Service {service_id} not found or not accessible"
                    )
                    return
                elif "not MCP compatible" in str(e):
                    await self._send_error_response(
                        send, 400, f"Service {service_id} is not MCP compatible"
                    )
                    return
                else:
                    raise
            
            # Track active connection
            self.active_sse_connections[cache_key] = self.active_sse_connections.get(cache_key, 0) + 1
            logger.debug(f"SSE connection opened for {cache_key}, active connections: {self.active_sse_connections[cache_key]}")
            
            try:
                # Handle the SSE request with the cached MCP app
                adapter = mcp_app.adapter if hasattr(mcp_app, 'adapter') else mcp_app
                await adapter.handle_sse_request(scope, receive, send)
            except Exception as e:
                logger.error(f"Error during SSE handling for {cache_key}: {e}")
                raise
            finally:
                # Decrement connection count
                if cache_key in self.active_sse_connections:
                    self.active_sse_connections[cache_key] -= 1
                    logger.debug(f"SSE connection closed for {cache_key}, remaining connections: {self.active_sse_connections[cache_key]}")
                    # Clean up if no active connections
                    if self.active_sse_connections[cache_key] <= 0:
                        del self.active_sse_connections[cache_key]
                        # Clean up the cached app immediately to prevent memory leaks
                        # This ensures workspace interface is closed when all SSE connections are done
                        await self._cleanup_mcp_app(cache_key)
        
        except Exception as exp:
            logger.exception(f"Error in MCP SSE service: {exp}")
            await self._send_error_response(send, 500, f"Internal Server Error: {exp}")
    
    async def _handle_sse_message_request(self, scope, receive, send, workspace, service_id):
        """Handle MCP SSE message POST requests."""
        try:
            # Use cached MCP app
            cache_key = f"{workspace}/{service_id}"
            if cache_key not in self.mcp_app_cache:
                # Connection was lost or app not found, return 404
                logger.warning(f"SSE session not found for {cache_key}")
                await self._send_error_response(
                    send, 404, f"SSE session not found. Please reconnect."
                )
                return
            
            cache_entry = self.mcp_app_cache[cache_key]
            mcp_app = cache_entry["app"] if isinstance(cache_entry, dict) else cache_entry
            adapter = mcp_app.adapter if hasattr(mcp_app, 'adapter') else mcp_app
            
            # Handle the SSE message POST request
            await adapter.handle_sse_message(scope, receive, send)
        
        except Exception as exp:
            logger.exception(f"Error in MCP SSE message service: {exp}")
            await self._send_error_response(send, 500, f"Internal Server Error: {exp}")