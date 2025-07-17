"""MCP (Model Context Protocol) service support for Hypha.

This module provides integration with the MCP protocol using the official mcp-sdk.
It allows registering MCP services as Hypha services and automatically exposes them
via HTTP endpoints with proper MCP protocol handling.
"""

import inspect
import json
import logging
import os
import asyncio
import time
from typing import Any, Dict, Optional, List, Callable, Union
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.types import ASGIApp, Scope, Receive, Send
from collections import deque
from dataclasses import dataclass
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def unwrap_object_proxy(obj):
    """Recursively unwrap ObjectProxy objects from Hypha RPC."""
    if hasattr(obj, '_rvalue'):
        return unwrap_object_proxy(obj._rvalue)
    elif isinstance(obj, dict):
        return {k: unwrap_object_proxy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [unwrap_object_proxy(item) for item in obj]
    else:
        return obj

try:
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.streamable_http import (
        EventCallback,
        EventId,
        EventMessage,
        EventStore,
        StreamId,
    )
    from mcp.types import JSONRPCMessage
    from pydantic import AnyUrl

    MCP_SDK_AVAILABLE = True
except ImportError:
    logger.warning("MCP SDK not available. Install with: pip install mcp")
    MCP_SDK_AVAILABLE = False

    # Define dummy classes for type hints
    class Server:
        pass

    class StreamableHTTPSessionManager:
        pass

    class EventStore:
        pass

    class JSONRPCMessage:
        pass

    class EventCallback:
        pass

    class EventId:
        pass

    class EventMessage:
        pass

    class StreamId:
        pass


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

    def __init__(self, redis_client, max_events_per_stream: int = 100, 
                 expiration_time: int = 3600):
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

    async def store_event(self, stream_id: StreamId, message: JSONRPCMessage) -> EventId:
        """Store an event with a generated event ID."""
        event_id = str(uuid4())
        timestamp = time.time()
        
        event_entry = EventEntry(
            event_id=event_id,
            stream_id=stream_id,
            message=message,
            timestamp=timestamp
        )
        
        # Store the event with expiration
        event_data = {
            "event_id": event_id,
            "stream_id": stream_id,
            "message": json.dumps(message),
            "timestamp": timestamp
        }
        
        await self.redis.hset(
            self._get_event_key(event_id),
            mapping=event_data
        )
        await self.redis.expire(
            self._get_event_key(event_id),
            self.expiration_time
        )
        
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


class HyphaMCPServer:
    """MCP Server that adapts Hypha services to MCP protocol."""
    
    def __init__(self, service, service_info):
        """Initialize with Hypha service.
        
        Args:
            service: Hypha service object
            service_info: Service information from Hypha
        """
        self.service = service
        self.service_info = service_info
        self.server = Server(name=service_info.id)
        
        # Validate service configuration
        self._validate_service_config()
        
        # Setup handlers based on service type
        self._setup_handlers()
    
    def _has_schema_functions(self) -> bool:
        """Check if service has @schema_function decorated functions."""
        if not isinstance(self.service, dict):
            return False
        
        for key, func in self.service.items():
            if callable(func) and hasattr(func, '__schema__'):
                return True
        return False
    
    def _extract_schema_functions(self) -> List[Dict[str, Any]]:
        """Extract MCP tools from @schema_function decorated functions."""
        tools = []
        
        if not isinstance(self.service, dict):
            return tools
        
        for key, func in self.service.items():
            if callable(func) and hasattr(func, '__schema__'):
                schema = getattr(func, '__schema__', {}) or {}
                
                # Create tool from schema function
                # The schema contains the JSON schema for the function
                tool_dict = {
                    "name": key,
                    "description": schema.get('description', func.__doc__ or f"Function {key}"),
                    "inputSchema": schema.get('parameters', schema)  # Use the full schema as inputSchema
                }
                
                # Ensure inputSchema has the correct structure
                if not isinstance(tool_dict["inputSchema"], dict) or "type" not in tool_dict["inputSchema"]:
                    tool_dict["inputSchema"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                
                tools.append(tool_dict)
        
        return tools
    
    def _validate_service_config(self):
        """Validate MCP service configuration to prevent conflicts."""
        # Check if service has inline tools/resources/prompts in config
        has_inline_tools = hasattr(self.service_info, 'tools') and self.service_info.tools
        has_inline_resources = hasattr(self.service_info, 'resources') and self.service_info.resources
        has_inline_prompts = hasattr(self.service_info, 'prompts') and self.service_info.prompts
        
        # Check if service has MCP function methods
        has_list_tools = "list_tools" in self.service
        has_call_tool = "call_tool" in self.service
        has_list_prompts = "list_prompts" in self.service
        has_get_prompt = "get_prompt" in self.service
        has_list_resources = "list_resources" in self.service
        has_list_resource_templates = "list_resource_templates" in self.service
        has_read_resource = "read_resource" in self.service
        
        # Check if service has schema functions (decorated with @schema_function)
        has_schema_functions = self._has_schema_functions()
        
        # Validate tools configuration
        if has_inline_tools and has_list_tools:
            raise ValueError("Cannot provide both 'tools' config and 'list_tools' function. Use one approach.")
        
        if has_schema_functions and (has_inline_tools or has_list_tools):
            raise ValueError("Cannot mix schema_function decorated functions with other MCP approaches.")
        
        # Validate resources configuration
        if has_inline_resources and (has_list_resources or has_list_resource_templates or has_read_resource):
            raise ValueError("Cannot provide both 'resources' config and resource functions. Use one approach.")
        
        # Validate prompts configuration
        if has_inline_prompts and (has_list_prompts or has_get_prompt):
            raise ValueError("Cannot provide both 'prompts' config and prompt functions. Use one approach.")
        
    def _setup_handlers(self):
        """Set up MCP protocol handlers."""
        
        # Tools handlers - approach 1: using list_tools/call_tool functions
        if "list_tools" in self.service:
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                try:
                    result = await self._call_service_function("list_tools", {})
                    # Convert to MCP Tool objects if needed
                    if isinstance(result, list) and result:
                        if isinstance(result[0], dict):
                            return [types.Tool(**tool) for tool in result]
                        return result
                    return []
                except Exception as e:
                    logger.exception("Error in list_tools")
                    return []
        
        # Tools handlers - approach 2: using inline tools config
        elif hasattr(self.service_info, 'tools') and self.service_info.tools:
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                try:
                    tools = []
                    for tool_config in self.service_info.tools:
                        if isinstance(tool_config, dict):
                            tools.append(types.Tool(**tool_config))
                        else:
                            # Assume it's already a Tool object
                            tools.append(tool_config)
                    return tools
                except Exception as e:
                    logger.exception("Error in inline tools list")
                    return []
        
        # Tools handlers - approach 3: using schema functions
        elif self._has_schema_functions():
            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                try:
                    schema_tools = self._extract_schema_functions()
                    tools = []
                    for tool_dict in schema_tools:
                        tools.append(types.Tool(**tool_dict))
                    return tools
                except Exception as e:
                    logger.exception("Error in schema function tools list")
                    return []
        
        if "call_tool" in self.service:
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                try:
                    result = await self._call_service_function("call_tool", {
                        "name": name,
                        "arguments": arguments
                    })
                    # Convert to MCP ContentBlock objects if needed
                    if isinstance(result, list) and result:
                        content_blocks = []
                        for item in result:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    content_blocks.append(types.TextContent(**item))
                                else:
                                    content_blocks.append(types.TextContent(
                                        type="text",
                                        text=str(item)
                                    ))
                            else:
                                content_blocks.append(types.TextContent(
                                    type="text",
                                    text=str(item)
                                ))
                        return content_blocks
                    return [types.TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.exception("Error in call_tool")
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        
        # Tool call handler - approach 2: using inline tools config
        elif hasattr(self.service_info, 'tools') and self.service_info.tools:
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                try:
                    # Find the tool by name
                    tool_handler = None
                    for tool_config in self.service_info.tools:
                        if isinstance(tool_config, dict) and tool_config.get('name') == name:
                            tool_handler = tool_config.get('handler')
                            break
                        elif hasattr(tool_config, 'name') and tool_config.name == name:
                            tool_handler = getattr(tool_config, 'handler', None)
                            break
                    
                    if not tool_handler:
                        return [types.TextContent(type="text", text=f"Tool '{name}' not found")]
                    
                    # Call the tool handler
                    if inspect.iscoroutinefunction(tool_handler):
                        result = await tool_handler(**arguments)
                    else:
                        result = tool_handler(**arguments)
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result
                    
                    # Convert result to ContentBlock
                    if isinstance(result, list) and result:
                        content_blocks = []
                        for item in result:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    content_blocks.append(types.TextContent(**item))
                                else:
                                    content_blocks.append(types.TextContent(
                                        type="text",
                                        text=str(item)
                                    ))
                            else:
                                content_blocks.append(types.TextContent(
                                    type="text",
                                    text=str(item)
                                ))
                        return content_blocks
                    return [types.TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.exception(f"Error in inline tool call for {name}")
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        
        # Tool call handler - approach 3: using schema functions
        elif self._has_schema_functions():
            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                try:
                    # Find the schema function by name
                    if name not in self.service:
                        return [types.TextContent(type="text", text=f"Tool '{name}' not found")]
                    
                    func = self.service[name]
                    if not (callable(func) and hasattr(func, '__schema__')):
                        return [types.TextContent(type="text", text=f"Function '{name}' is not a schema function")]
                    
                    # Call the schema function directly with unpacked arguments
                    if inspect.iscoroutinefunction(func):
                        result = await func(**arguments)
                    else:
                        result = func(**arguments)
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result
                    
                    # Convert result to ContentBlock
                    # For schema functions, always wrap the result as text content
                    return [types.TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.exception(f"Error in schema function call for {name}")
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]
        
        # Prompts handlers - approach 1: using list_prompts/get_prompt functions
        if "list_prompts" in self.service:
            @self.server.list_prompts()
            async def list_prompts() -> List[types.Prompt]:
                try:
                    result = await self._call_service_function("list_prompts", {})
                    # Convert to MCP Prompt objects if needed
                    if isinstance(result, list) and result:
                        if isinstance(result[0], dict):
                            return [types.Prompt(**prompt) for prompt in result]
                        return result
                    return []
                except Exception as e:
                    logger.exception("Error in list_prompts")
                    return []
        
        # Prompts handlers - approach 2: using inline prompts config
        elif hasattr(self.service_info, 'prompts') and self.service_info.prompts:
            @self.server.list_prompts()
            async def list_prompts() -> List[types.Prompt]:
                try:
                    prompts = []
                    for prompt_config in self.service_info.prompts:
                        if isinstance(prompt_config, dict):
                            # Extract prompt metadata without handler
                            prompt_dict = {k: v for k, v in prompt_config.items() if k != 'read'}
                            prompts.append(types.Prompt(**prompt_dict))
                        else:
                            prompts.append(prompt_config)
                    return prompts
                except Exception as e:
                    logger.exception("Error in inline prompts list")
                    return []
        
        if "get_prompt" in self.service:
            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                try:
                    result = await self._call_service_function("get_prompt", {
                        "name": name,
                        "arguments": arguments
                    })
                    # Convert to MCP GetPromptResult if needed
                    if isinstance(result, dict):
                        # Convert messages if needed
                        if "messages" in result:
                            messages = []
                            for msg in result["messages"]:
                                if isinstance(msg, dict):
                                    # Convert content
                                    content = msg.get("content", {})
                                    if isinstance(content, dict):
                                        if content.get("type") == "text":
                                            content_obj = types.TextContent(**content)
                                        else:
                                            content_obj = types.TextContent(
                                                type="text",
                                                text=str(content)
                                            )
                                    else:
                                        content_obj = types.TextContent(
                                            type="text",
                                            text=str(content)
                                        )
                                    
                                    messages.append(types.PromptMessage(
                                        role=msg.get("role", "user"),
                                        content=content_obj
                                    ))
                            result["messages"] = messages
                        return types.GetPromptResult(**result)
                    return types.GetPromptResult(
                        description="No description",
                        messages=[types.PromptMessage(
                            role="user",
                            content=types.TextContent(type="text", text=str(result))
                        )]
                    )
                except Exception as e:
                    logger.exception("Error in get_prompt")
                    return types.GetPromptResult(
                        description="Error",
                        messages=[types.PromptMessage(
                            role="user",
                            content=types.TextContent(type="text", text=f"Error: {str(e)}")
                        )]
                    )
        
        # Prompt handler - approach 2: using inline prompts config
        elif hasattr(self.service_info, 'prompts') and self.service_info.prompts:
            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                try:
                    # Find the prompt by name
                    prompt_handler = None
                    for prompt_config in self.service_info.prompts:
                        if isinstance(prompt_config, dict) and prompt_config.get('name') == name:
                            prompt_handler = prompt_config.get('read')
                            break
                        elif hasattr(prompt_config, 'name') and prompt_config.name == name:
                            prompt_handler = getattr(prompt_config, 'read', None)
                            break
                    
                    if not prompt_handler:
                        return types.GetPromptResult(
                            description="Prompt not found",
                            messages=[types.PromptMessage(
                                role="user",
                                content=types.TextContent(type="text", text=f"Prompt '{name}' not found")
                            )]
                        )
                    
                    # Call the prompt handler
                    if inspect.iscoroutinefunction(prompt_handler):
                        result = await prompt_handler(**arguments)
                    else:
                        result = prompt_handler(**arguments)
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result
                    
                    # Convert result to GetPromptResult
                    if isinstance(result, dict):
                        # Convert messages if needed
                        if "messages" in result:
                            messages = []
                            for msg in result["messages"]:
                                if isinstance(msg, dict):
                                    # Convert content
                                    content = msg.get("content", {})
                                    if isinstance(content, dict):
                                        if content.get("type") == "text":
                                            content_obj = types.TextContent(**content)
                                        else:
                                            content_obj = types.TextContent(
                                                type="text",
                                                text=str(content)
                                            )
                                    else:
                                        content_obj = types.TextContent(
                                            type="text",
                                            text=str(content)
                                        )
                                    
                                    messages.append(types.PromptMessage(
                                        role=msg.get("role", "user"),
                                        content=content_obj
                                    ))
                            result["messages"] = messages
                        return types.GetPromptResult(**result)
                    return types.GetPromptResult(
                        description="No description",
                        messages=[types.PromptMessage(
                            role="user",
                            content=types.TextContent(type="text", text=str(result))
                        )]
                    )
                except Exception as e:
                    logger.exception(f"Error in inline prompt get for {name}")
                    return types.GetPromptResult(
                        description="Error",
                        messages=[types.PromptMessage(
                            role="user",
                            content=types.TextContent(type="text", text=f"Error: {str(e)}")
                        )]
                    )
        
        # Resources handlers - approach 1: using resource functions
        if "list_resource_templates" in self.service:
            @self.server.list_resource_templates()
            async def list_resource_templates() -> List[types.ResourceTemplate]:
                try:
                    result = await self._call_service_function("list_resource_templates", {})
                    # Convert to MCP ResourceTemplate objects if needed
                    if isinstance(result, list) and result:
                        if isinstance(result[0], dict):
                            return [types.ResourceTemplate(**template) for template in result]
                        return result
                    return []
                except Exception as e:
                    logger.exception("Error in list_resource_templates")
                    return []
        
        # Resources handlers - approach 2: using inline resources config
        elif hasattr(self.service_info, 'resources') and self.service_info.resources:
            @self.server.list_resource_templates()
            async def list_resource_templates() -> List[types.ResourceTemplate]:
                try:
                    templates = []
                    for resource_config in self.service_info.resources:
                        if isinstance(resource_config, dict):
                            # Create ResourceTemplate from resource config
                            template_dict = {
                                "uriTemplate": resource_config.get("uri", ""),
                                "name": resource_config.get("name", ""),
                                "description": resource_config.get("description", ""),
                                "mimeType": resource_config.get("mime_type", "text/plain")
                            }
                            templates.append(types.ResourceTemplate(**template_dict))
                        else:
                            templates.append(resource_config)
                    return templates
                except Exception as e:
                    logger.exception("Error in inline resources list_resource_templates")
                    return []
        
        if "list_resources" in self.service:
            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                try:
                    result = await self._call_service_function("list_resources", {})
                    # Convert to MCP Resource objects if needed
                    if isinstance(result, list) and result:
                        if isinstance(result[0], dict):
                            return [types.Resource(**resource) for resource in result]
                        return result
                    return []
                except Exception as e:
                    logger.exception("Error in list_resources")
                    return []
        
        # Resources list handler - approach 2: using inline resources config
        elif hasattr(self.service_info, 'resources') and self.service_info.resources:
            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                try:
                    resources = []
                    for resource_config in self.service_info.resources:
                        if isinstance(resource_config, dict):
                            # Create Resource from resource config
                            resource_dict = {
                                "uri": resource_config.get("uri", ""),
                                "name": resource_config.get("name", ""),
                                "description": resource_config.get("description", ""),
                                "mimeType": resource_config.get("mime_type", "text/plain")
                            }
                            resources.append(types.Resource(**resource_dict))
                        else:
                            resources.append(resource_config)
                    return resources
                except Exception as e:
                    logger.exception("Error in inline resources list_resources")
                    return []
        
        if "read_resource" in self.service:
            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> types.ReadResourceResult:
                try:
                    result = await self._call_service_function("read_resource", {
                        "uri": str(uri)
                    })
                    # Convert to MCP ReadResourceResult if needed
                    if isinstance(result, dict):
                        return types.ReadResourceResult(**result)
                    return types.ReadResourceResult(
                        contents=[types.TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text=str(result)
                        )]
                    )
                except Exception as e:
                    logger.exception("Error in read_resource")
                    return types.ReadResourceResult(
                        contents=[types.TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text=f"Error: {str(e)}"
                        )]
                    )
        
        # Resource read handler - approach 2: using inline resources config
        elif hasattr(self.service_info, 'resources') and self.service_info.resources:
            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> types.ReadResourceResult:
                try:
                    # Find the resource by URI
                    resource_handler = None
                    for resource_config in self.service_info.resources:
                        if isinstance(resource_config, dict) and resource_config.get('uri') == str(uri):
                            resource_handler = resource_config.get('read')
                            break
                        elif hasattr(resource_config, 'uri') and resource_config.uri == str(uri):
                            resource_handler = getattr(resource_config, 'read', None)
                            break
                    
                    if not resource_handler:
                        return types.ReadResourceResult(
                            contents=[types.TextResourceContents(
                                uri=uri,
                                mimeType="text/plain",
                                text=f"Resource '{uri}' not found"
                            )]
                        )
                    
                    # Call the resource handler
                    if inspect.iscoroutinefunction(resource_handler):
                        result = await resource_handler()
                    else:
                        result = resource_handler()
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result
                    
                    # Convert result to ReadResourceResult
                    if isinstance(result, dict):
                        return types.ReadResourceResult(**result)
                    return types.ReadResourceResult(
                        contents=[types.TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text=str(result)
                        )]
                    )
                except Exception as e:
                    logger.exception(f"Error in inline resource read for {uri}")
                    return types.ReadResourceResult(
                        contents=[types.TextResourceContents(
                            uri=uri,
                            mimeType="text/plain",
                            text=f"Error: {str(e)}"
                        )]
                    )
        
        # Progress notifications
        if "progress_notification" in self.service:
            # This is handled by the session manager
            pass
    
    async def _call_service_function(self, function_name: str, kwargs: Dict[str, Any]) -> Any:
        """Call a service function."""
        if function_name not in self.service:
            raise ValueError(f"Function {function_name} not found in service")
        
        func = self.service[function_name]
        
        if not callable(func):
            raise ValueError(f"Function {function_name} is not callable")
        
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
                # Handle case where sync function returns a Future (Hypha wrapped async functions)
                if asyncio.isfuture(result) or inspect.iscoroutine(result):
                    result = await result
            
            return result
        except Exception as e:
            logger.exception(f"Error calling service function {function_name}")
            raise
    
    def get_server(self) -> Server:
        """Get the MCP server instance."""
        return self.server


async def create_mcp_app_from_service(service, service_info, redis_client):
    """Create MCP Starlette application from a Hypha service.
    
    Args:
        service: Hypha service object
        service_info: Service information from Hypha
        redis_client: Redis client for event store
        
    Returns:
        ASGI application that can be called with (scope, receive, send)
    """
    logger.info(f"Creating MCP app for service: {service_info.id}")
    
    # Create MCP server
    mcp_server = HyphaMCPServer(service, service_info)
    server = mcp_server.get_server()
    
    # Simple JSON-RPC handler for testing (no SSE for now)
    async def handle_mcp_request(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await send({"type": "http.response.start", "status": 405})
            await send({"type": "http.response.body", "body": b"Method Not Allowed"})
            return
            
        # Read request body
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
        
        try:
            # Parse JSON-RPC request
            request_data = json.loads(body.decode("utf-8"))
            
            # Handle JSON-RPC request
            if request_data.get("method") == "tools/list":
                if "list_tools" in service:
                    try:
                        tools = await mcp_server._call_service_function("list_tools", {})
                        logger.debug(f"Raw tools result: {tools}")
                        logger.debug(f"Tools type: {type(tools)}")
                        if tools:
                            logger.debug(f"First tool type: {type(tools[0])}")
                            logger.debug(f"First tool: {tools[0]}")
                        
                        # Convert to proper JSON format
                        if isinstance(tools, list) and tools:
                            tools_data = []
                            for tool in tools:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                tool_data = unwrap_object_proxy(tool)
                                
                                # Convert to dict if it's an MCP object
                                if hasattr(tool_data, 'model_dump'):
                                    tool_dict = tool_data.model_dump()
                                elif isinstance(tool_data, dict):
                                    tool_dict = tool_data
                                else:
                                    tool_dict = {"name": str(tool_data), "description": "", "inputSchema": {}}
                                
                                tools_data.append(tool_dict)
                        else:
                            tools_data = []
                        
                        logger.debug(f"Final tools_data: {tools_data}")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"tools": tools_data}
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                elif mcp_server._has_schema_functions():
                    try:
                        # Extract tools from schema functions
                        schema_tools = mcp_server._extract_schema_functions()
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"tools": schema_tools}
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"tools": []}
                    }
            
            elif request_data.get("method") == "tools/call":
                if "call_tool" in service:
                    try:
                        params = request_data.get("params", {})
                        result = await mcp_server._call_service_function("call_tool", params)
                        # Convert to proper JSON format
                        if isinstance(result, list) and result:
                            content_data = []
                            for item in result:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                item_data = unwrap_object_proxy(item)
                                
                                # Convert to dict if it's an MCP object
                                if hasattr(item_data, 'model_dump'):
                                    item_dict = item_data.model_dump()
                                elif isinstance(item_data, dict):
                                    item_dict = item_data
                                else:
                                    item_dict = {"type": "text", "text": str(item_data)}
                                
                                content_data.append(item_dict)
                        else:
                            content_data = [{"type": "text", "text": str(result)}]
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"content": content_data}
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                elif mcp_server._has_schema_functions():
                    try:
                        params = request_data.get("params", {})
                        name = params.get("name")
                        arguments = params.get("arguments", {})
                        
                        # Find the schema function by name
                        if name not in service:
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_data.get("id"),
                                "error": {"code": -32602, "message": f"Tool '{name}' not found"}
                            }
                        else:
                            func = service[name]
                            if not (callable(func) and hasattr(func, '__schema__')):
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_data.get("id"),
                                    "error": {"code": -32602, "message": f"Function '{name}' is not a schema function"}
                                }
                            else:
                                # Call the schema function directly
                                if inspect.iscoroutinefunction(func):
                                    result = await func(**arguments)
                                else:
                                    result = func(**arguments)
                                    if asyncio.isfuture(result) or inspect.iscoroutine(result):
                                        result = await result
                                
                                # Convert result to proper format
                                content_data = [{"type": "text", "text": str(result)}]
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_data.get("id"),
                                    "result": {"content": content_data}
                                }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "error": {"code": -32601, "message": "Method not found"}
                    }
            
            elif request_data.get("method") == "prompts/list":
                if "list_prompts" in service:
                    try:
                        prompts = await mcp_server._call_service_function("list_prompts", {})
                        # Convert to proper JSON format
                        if isinstance(prompts, list) and prompts:
                            prompts_data = []
                            for prompt in prompts:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                prompt_data = unwrap_object_proxy(prompt)
                                
                                # Convert to dict if it's an MCP object
                                if hasattr(prompt_data, 'model_dump'):
                                    prompt_dict = prompt_data.model_dump()
                                elif isinstance(prompt_data, dict):
                                    prompt_dict = prompt_data
                                else:
                                    prompt_dict = {"name": str(prompt_data), "description": "", "arguments": []}
                                
                                prompts_data.append(prompt_dict)
                        else:
                            prompts_data = []
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"prompts": prompts_data}
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"prompts": []}
                    }
            
            elif request_data.get("method") == "prompts/get":
                if "get_prompt" in service:
                    try:
                        params = request_data.get("params", {})
                        result = await mcp_server._call_service_function("get_prompt", params)
                        # Convert to proper JSON format
                        # Handle ObjectProxy wrapping from Hypha RPC
                        result_data = unwrap_object_proxy(result)
                        
                        if isinstance(result_data, dict):
                            if hasattr(result_data, 'model_dump'):
                                result_data = result_data.model_dump()
                        else:
                            result_data = {"description": "No description", "messages": []}
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": result_data
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "error": {"code": -32601, "message": "Method not found"}
                    }
            
            elif request_data.get("method") == "resources/templates/list":
                if "list_resource_templates" in service:
                    try:
                        templates = await mcp_server._call_service_function("list_resource_templates", {})
                        # Convert to proper JSON format
                        if isinstance(templates, list) and templates:
                            templates_data = []
                            for template in templates:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                template_data = unwrap_object_proxy(template)
                                
                                # Convert to dict if it's an MCP object
                                if hasattr(template_data, 'model_dump'):
                                    template_dict = template_data.model_dump()
                                elif isinstance(template_data, dict):
                                    template_dict = template_data
                                else:
                                    template_dict = {"uriTemplate": str(template_data), "name": str(template_data)}
                                
                                templates_data.append(template_dict)
                        else:
                            templates_data = []
                        
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"resourceTemplates": templates_data}
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"resourceTemplates": []}
                    }
            
            elif "method" not in request_data:
                logger.debug(f"Invalid request: {request_data}")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "error": {"code": -32600, "message": "Invalid request"}
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}
                }
        
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"}
            }
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "id": request_data.get("id") if "request_data" in locals() else None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
        
        # Send response
        response_body = json.dumps(response).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(response_body)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": response_body,
        })
    
    # Create Starlette app
    app = Starlette(
        routes=[
            Mount("/", app=handle_mcp_request),
        ]
    )
    
    return app


def is_mcp_compatible_service(service) -> bool:
    """Check if a service is MCP compatible.
    
    A service is MCP compatible if:
    1. It has type="mcp", OR
    2. It has MCP protocol functions (list_tools, call_tool, etc.), OR
    3. It has inline tools/resources/prompts config (via service_info), OR
    4. It has @schema_function decorated functions
    """
    # Check for explicit MCP type in service
    if hasattr(service, "type") and service.type == "mcp":
        return True
    
    # Check for MCP protocol functions
    mcp_functions = [
        "list_tools", "call_tool", "list_prompts", "get_prompt",
        "list_resource_templates", "list_resources", "read_resource"
    ]
    
    if isinstance(service, dict):
        # Check if any MCP function exists
        has_mcp_function = any(func in service for func in mcp_functions)
        
        # For dict services, also check if they have a type field
        service_type = service.get("type", "")
        if service_type == "mcp":
            return True
        
        return has_mcp_function
    
    # For object services, check attributes
    has_mcp_functions = any(hasattr(service, func) for func in mcp_functions)
    
    # Check for schema functions (decorated with @schema_function)
    has_schema_functions = False
    if isinstance(service, dict):
        for key, func in service.items():
            if callable(func) and hasattr(func, '__schema__'):
                has_schema_functions = True
                break
    
    return has_mcp_functions or has_schema_functions


class MCPRoutingMiddleware:
    """Middleware specifically for routing MCP services."""
    
    def __init__(self, app: ASGIApp, base_path=None, store=None):
        self.app = app
        self.base_path = base_path.rstrip("/") if base_path else ""
        self.store = store
        
        # MCP route pattern: /{workspace}/mcp/{service_id}/{path:path}
        route = self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}"
        self.route = Route(route, endpoint=None)
        logger.info(f"MCP middleware initialized with route pattern: {self.route.path}")
    
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
            
            # Check if the current request path matches the MCP route
            from starlette.routing import Match
            
            match, params = self.route.matches(scope)
            path_params = params.get("path_params", {})
            
            if match == Match.FULL:
                # Extract workspace and service_id from the matched path
                workspace = path_params["workspace"]
                service_id_path = path_params["service_id"]
                
                # Parse service_id and path from service_id_path
                service_id_parts = service_id_path.split("/", 1)
                service_id = service_id_parts[0]
                path = "/" + service_id_parts[1] if len(service_id_parts) > 1 else "/"
                
                logger.debug(f"MCP Middleware: Parsed - workspace='{workspace}', service_id='{service_id}', path='{path}'")
                
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
                    if not path.startswith("/"):
                        path = "/" + path
                    scope["path"] = path
                    scope["raw_path"] = path.encode("latin-1")
                    
                    # Get _mode from query string
                    query = scope.get("query_string", b"").decode("utf-8")
                    if query:
                        _mode = dict([q.split("=") for q in query.split("&")]).get("_mode")
                    else:
                        _mode = None
                    
                    # Get authentication info
                    access_token = self._get_access_token_from_cookies(scope)
                    authorization = self._get_authorization_header(scope)
                    
                    # Login and get user info
                    user_info = await self.store.login_optional(
                        authorization=authorization, access_token=access_token
                    )
                    
                    # Get the MCP service
                    async with self.store.get_workspace_interface(
                        user_info, workspace
                    ) as api:
                        try:
                            service_info = await api.get_service_info(
                                service_id, {"mode": _mode}
                            )
                            logger.debug(f"MCP Middleware: Found service '{service_id}' of type '{service_info.type}'")
                        except (KeyError, Exception) as e:
                            logger.error(f"MCP Middleware: Service lookup failed: {e}")
                            if "Service not found" in str(e):
                                await self._send_error_response(
                                    send, 404, f"Service {service_id} not found"
                                )
                                return
                            else:
                                raise
                        
                        if service_info.type != "mcp":
                            await self._send_error_response(
                                send, 400, f"Service {service_id} is not MCP compatible"
                            )
                            return
                        
                        service = await api.get_service(service_info.id)
                        
                        # Check if it's MCP compatible
                        if not is_mcp_compatible_service(service):
                            await self._send_error_response(
                                send, 400, f"Service {service_id} is not MCP compatible"
                            )
                            return
                        
                        mcp_app = await create_mcp_app_from_service(
                            service, service_info, self.store.get_redis()
                        )
                        
                        # Handle the request with the MCP app
                        await mcp_app(scope, receive, send)
                        return
                
                except Exception as exp:
                    logger.exception(f"Error in MCP service: {exp}")
                    await self._send_error_response(
                        send, 500, f"Internal Server Error: {exp}"
                    )
                    return
        
        # Continue to next middleware if not an MCP route
        await self.app(scope, receive, send)
    
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