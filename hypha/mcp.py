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
from pydantic import AnyUrl

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        event_data = {
            "event_id": event_id,
            "stream_id": stream_id,
            "message": json.dumps(message),
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

    def _extract_tool_from_schema_function(self, func) -> Dict[str, Any]:
        """Extract MCP tool definition from @schema_function decorated function."""
        if not (callable(func) and hasattr(func, "__schema__")):
            raise ValueError(f"Function {func} is not a schema function")

        schema = getattr(func, "__schema__", {}) or {}

        # Create tool from schema function
        # The schema contains the JSON schema for the function
        tool_dict = {
            "name": func.__name__,
            "description": schema.get(
                "description", func.__doc__ or f"Function {func.__name__}"
            ),
            "inputSchema": schema.get(
                "parameters", schema
            ),  # Use the full schema as inputSchema
        }

        # Ensure inputSchema has the correct structure
        if (
            not isinstance(tool_dict["inputSchema"], dict)
            or "type" not in tool_dict["inputSchema"]
        ):
            tool_dict["inputSchema"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

        return tool_dict

    def _validate_service_config(self):
        """Validate MCP service configuration to prevent conflicts."""
        # Check if service has inline tools/resources/prompts in config
        has_inline_tools = (
            hasattr(self.service_info, "tools") and self.service_info.tools
        )
        has_inline_resources = (
            hasattr(self.service_info, "resources") and self.service_info.resources
        )
        has_inline_prompts = (
            hasattr(self.service_info, "prompts") and self.service_info.prompts
        )

        # Check if service has MCP function methods
        has_list_tools = "list_tools" in self.service
        has_call_tool = "call_tool" in self.service
        has_list_prompts = "list_prompts" in self.service
        has_get_prompt = "get_prompt" in self.service
        has_list_resources = "list_resources" in self.service
        has_list_resource_templates = "list_resource_templates" in self.service
        has_read_resource = "read_resource" in self.service

        # Validate tools configuration
        if has_inline_tools and has_list_tools:
            raise ValueError(
                "Cannot provide both 'tools' config and 'list_tools' function. Use one approach."
            )

        # Validate resources configuration
        if has_inline_resources and (
            has_list_resources or has_list_resource_templates or has_read_resource
        ):
            raise ValueError(
                "Cannot provide both 'resources' config and resource functions. Use one approach."
            )

        # Validate prompts configuration
        if has_inline_prompts and (has_list_prompts or has_get_prompt):
            raise ValueError(
                "Cannot provide both 'prompts' config and prompt functions. Use one approach."
            )

        # Validate inline tools are schema functions
        if has_inline_tools:
            for key, tool in self.service_info.tools.items():
                if not (callable(tool) and hasattr(tool, "__schema__")):
                    raise ValueError(
                        f"Tool {key} must be a @schema_function decorated function"
                    )

        # Validate inline resources have schema function readers
        if has_inline_resources:
            for key, resource in self.service_info.resources.items():
                if not isinstance(resource, dict):
                    raise ValueError(f"Resource {resource} must be a dictionary")
                if "read" not in resource:
                    raise ValueError(f"Resource {resource} must have a 'read' key")
                read_func = resource["read"]
                if not (callable(read_func) and hasattr(read_func, "__schema__")):
                    raise ValueError(
                        f"Resource read function {read_func} must be a @schema_function decorated function"
                    )

        # Validate inline prompts have schema function readers
        if has_inline_prompts:
            for key, prompt in self.service_info.prompts.items():
                if not isinstance(prompt, dict):
                    raise ValueError(f"Prompt {prompt} must be a dictionary")
                if "read" not in prompt:
                    raise ValueError(f"Prompt {prompt} must have a 'read' key")
                read_func = prompt["read"]
                if not (callable(read_func) and hasattr(read_func, "__schema__")):
                    raise ValueError(
                        f"Prompt read function {read_func} must be a @schema_function decorated function"
                    )

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

        # Tools handlers - approach 2: using inline tools config (schema functions)
        elif hasattr(self.service_info, "tools") and self.service_info.tools:

            @self.server.list_tools()
            async def list_tools() -> List[types.Tool]:
                try:
                    tools = []
                    for key, tool_func in self.service_info.tools.items():
                        # Extract tool definition from schema function
                        tool_dict = self._extract_tool_from_schema_function(tool_func)
                        tool_dict["name"] = key
                        tools.append(types.Tool(**tool_dict))
                    return tools
                except Exception as e:
                    logger.exception("Error in inline tools list")
                    return []

        if "call_tool" in self.service:

            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                try:
                    result = await self._call_service_function(
                        "call_tool", {"name": name, "arguments": arguments}
                    )
                    # Convert to MCP ContentBlock objects if needed
                    if isinstance(result, list) and result:
                        content_blocks = []
                        for item in result:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    content_blocks.append(types.TextContent(**item))
                                else:
                                    content_blocks.append(
                                        types.TextContent(type="text", text=str(item))
                                    )
                            else:
                                content_blocks.append(
                                    types.TextContent(type="text", text=str(item))
                                )
                        return content_blocks
                    return [types.TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.exception("Error in call_tool")
                    return [types.TextContent(type="text", text=f"Error: {str(e)}")]

        # Tool call handler - approach 2: using inline tools config (schema functions)
        elif hasattr(self.service_info, "tools") and self.service_info.tools:

            @self.server.call_tool()
            async def call_tool(name: str, arguments: dict) -> List[types.ContentBlock]:
                try:
                    # Find the tool by name
                    tool_func = None
                    for key, func in self.service_info.tools:
                        if func.__name__ == name:
                            tool_func = func
                            break

                    if not tool_func:
                        return [
                            types.TextContent(
                                type="text", text=f"Tool '{name}' not found"
                            )
                        ]

                    # Call the schema function directly with unpacked arguments
                    if inspect.iscoroutinefunction(tool_func):
                        result = await tool_func(**arguments)
                    else:
                        result = tool_func(**arguments)
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result

                    # Convert result to ContentBlock
                    # For schema functions, always wrap the result as text content
                    return [types.TextContent(type="text", text=str(result))]
                except Exception as e:
                    logger.exception(f"Error in inline tool call for {name}")
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

        # Prompts handlers - approach 2: using inline prompts config (schema functions)
        elif hasattr(self.service_info, "prompts") and self.service_info.prompts:

            @self.server.list_prompts()
            async def list_prompts() -> List[types.Prompt]:
                try:
                    prompts = []
                    for key, prompt_config in self.service_info.prompts.items():
                        # Extract prompt metadata without handler
                        prompt_dict = {
                            k: v for k, v in prompt_config.items() if k != "read"
                        }
                        assert (
                            key == prompt_dict["name"]
                        ), f"Prompt name mismatch: {key} != {prompt_dict['name']}"

                        # Add arguments from the schema function if available
                        if "read" in prompt_config:
                            read_func = prompt_config["read"]
                            if hasattr(read_func, "__schema__"):
                                schema = getattr(read_func, "__schema__", {}) or {}
                                parameters = schema.get("parameters", {})
                                properties = parameters.get("properties", {})
                                required = parameters.get("required", [])

                                # Convert schema properties to prompt arguments
                                arguments = []
                                for prop_name, prop_info in properties.items():
                                    arguments.append(
                                        types.PromptArgument(
                                            name=prop_name,
                                            description=prop_info.get(
                                                "description", ""
                                            ),
                                            required=prop_name in required,
                                        )
                                    )

                                prompt_dict["arguments"] = arguments

                        prompts.append(types.Prompt(**prompt_dict))
                    return prompts
                except Exception as e:
                    logger.exception("Error in inline prompts list")
                    return []

        if "get_prompt" in self.service:

            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                try:
                    result = await self._call_service_function(
                        "get_prompt", {"name": name, "arguments": arguments}
                    )
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
                                                type="text", text=str(content)
                                            )
                                    else:
                                        content_obj = types.TextContent(
                                            type="text", text=str(content)
                                        )

                                    messages.append(
                                        types.PromptMessage(
                                            role=msg.get("role", "user"),
                                            content=content_obj,
                                        )
                                    )
                            result["messages"] = messages
                        return types.GetPromptResult(**result)
                    return types.GetPromptResult(
                        description="No description",
                        messages=[
                            types.PromptMessage(
                                role="user",
                                content=types.TextContent(
                                    type="text", text=str(result)
                                ),
                            )
                        ],
                    )
                except Exception as e:
                    logger.exception("Error in get_prompt")
                    return types.GetPromptResult(
                        description="Error",
                        messages=[
                            types.PromptMessage(
                                role="user",
                                content=types.TextContent(
                                    type="text", text=f"Error: {str(e)}"
                                ),
                            )
                        ],
                    )

        # Prompt handler - approach 2: using inline prompts config (schema functions)
        elif hasattr(self.service_info, "prompts") and self.service_info.prompts:

            @self.server.get_prompt()
            async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
                try:
                    # Find the prompt by name
                    prompt_handler = None
                    for key, prompt_config in self.service_info.prompts.items():
                        if key == name:
                            prompt_handler = prompt_config.get("read")
                            break

                    if not prompt_handler:
                        return types.GetPromptResult(
                            description="Prompt not found",
                            messages=[
                                types.PromptMessage(
                                    role="user",
                                    content=types.TextContent(
                                        type="text", text=f"Prompt '{name}' not found"
                                    ),
                                )
                            ],
                        )

                    # Call the schema function directly
                    if inspect.iscoroutinefunction(prompt_handler):
                        result = await prompt_handler(**arguments)
                    else:
                        result = prompt_handler(**arguments)
                        if asyncio.isfuture(result) or inspect.iscoroutine(result):
                            result = await result

                    # For schema functions, assume result is a simple string to be converted to a prompt
                    # The schema function should return either a string or a properly structured dict
                    if isinstance(result, dict):
                        # If result is already a dict, try to use it as-is
                        if "messages" in result:
                            # Convert messages if needed
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
                                                type="text", text=str(content)
                                            )
                                    else:
                                        content_obj = types.TextContent(
                                            type="text", text=str(content)
                                        )

                                    messages.append(
                                        types.PromptMessage(
                                            role=msg.get("role", "user"),
                                            content=content_obj,
                                        )
                                    )
                            result["messages"] = messages
                        return types.GetPromptResult(**result)
                    else:
                        # If result is a string, create a simple prompt
                        return types.GetPromptResult(
                            description=f"Prompt template for {name}",
                            messages=[
                                types.PromptMessage(
                                    role="user",
                                    content=types.TextContent(
                                        type="text", text=str(result)
                                    ),
                                )
                            ],
                        )
                except Exception as e:
                    logger.exception(f"Error in inline prompt get for {name}")
                    return types.GetPromptResult(
                        description="Error",
                        messages=[
                            types.PromptMessage(
                                role="user",
                                content=types.TextContent(
                                    type="text", text=f"Error: {str(e)}"
                                ),
                            )
                        ],
                    )

        # Resources handlers - approach 1: using resource functions
        if "list_resource_templates" in self.service:

            @self.server.list_resource_templates()
            async def list_resource_templates() -> List[types.ResourceTemplate]:
                try:
                    result = await self._call_service_function(
                        "list_resource_templates", {}
                    )
                    # Convert to MCP ResourceTemplate objects if needed
                    if isinstance(result, list) and result:
                        if isinstance(result[0], dict):
                            return [
                                types.ResourceTemplate(**template)
                                for template in result
                            ]
                        return result
                    return []
                except Exception as e:
                    logger.exception("Error in list_resource_templates")
                    return []

        # Resources handlers - approach 2: using inline resources config (schema functions)
        elif hasattr(self.service_info, "resources") and self.service_info.resources:

            @self.server.list_resource_templates()
            async def list_resource_templates() -> List[types.ResourceTemplate]:
                try:
                    templates = []
                    for key, resource_config in self.service_info.resources.items():
                        # Create ResourceTemplate from resource config
                        template_dict = {
                            "uriTemplate": resource_config.get("uri", ""),
                            "name": resource_config.get("name", ""),
                            "description": resource_config.get("description", ""),
                            "mimeType": resource_config.get("mime_type", "text/plain"),
                        }
                        templates.append(types.ResourceTemplate(**template_dict))
                    return templates
                except Exception as e:
                    logger.exception(
                        "Error in inline resources list_resource_templates"
                    )
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

        # Resources list handler - approach 2: using inline resources config (schema functions)
        elif hasattr(self.service_info, "resources") and self.service_info.resources:

            @self.server.list_resources()
            async def list_resources() -> List[types.Resource]:
                try:
                    resources = []
                    for key, resource_config in self.service_info.resources.items():
                        # Create Resource from resource config
                        resource_dict = {
                            "uri": resource_config.get("uri", ""),
                            "name": resource_config.get("name", ""),
                            "description": resource_config.get("description", ""),
                            "mimeType": resource_config.get("mime_type", "text/plain"),
                        }
                        resources.append(types.Resource(**resource_dict))
                    return resources
                except Exception as e:
                    logger.exception("Error in inline resources list_resources")
                    return []

        if "read_resource" in self.service:

            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> types.ReadResourceResult:
                try:
                    result = await self._call_service_function(
                        "read_resource", {"uri": str(uri)}
                    )
                    # Convert to MCP ReadResourceResult if needed
                    if isinstance(result, dict):
                        return types.ReadResourceResult(**result)
                    return types.ReadResourceResult(
                        contents=[
                            types.TextResourceContents(
                                uri=uri, mimeType="text/plain", text=str(result)
                            )
                        ]
                    )
                except Exception as e:
                    logger.exception("Error in read_resource")
                    return types.ReadResourceResult(
                        contents=[
                            types.TextResourceContents(
                                uri=uri, mimeType="text/plain", text=f"Error: {str(e)}"
                            )
                        ]
                    )

        # Resource read handler - approach 2: using inline resources config (schema functions)
        elif hasattr(self.service_info, "resources") and self.service_info.resources:

            @self.server.read_resource()
            async def read_resource(uri: AnyUrl) -> types.ReadResourceResult:
                try:
                    # Find the resource by URI
                    resource_handler = None
                    for key, resource_config in self.service_info.resources.items():
                        if resource_config.get("uri") == str(uri):
                            resource_handler = resource_config.get("read")
                            break

                    if not resource_handler:
                        return types.ReadResourceResult(
                            contents=[
                                types.TextResourceContents(
                                    uri=uri,
                                    mimeType="text/plain",
                                    text=f"Resource '{uri}' not found",
                                )
                            ]
                        )

                    # Call the schema function directly
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
                        contents=[
                            types.TextResourceContents(
                                uri=uri, mimeType="text/plain", text=str(result)
                            )
                        ]
                    )
                except Exception as e:
                    logger.exception(f"Error in inline resource read for {uri}")
                    return types.ReadResourceResult(
                        contents=[
                            types.TextResourceContents(
                                uri=uri, mimeType="text/plain", text=f"Error: {str(e)}"
                            )
                        ]
                    )

        # Progress notifications
        if "progress_notification" in self.service:
            # This is handled by the session manager
            pass

    async def _call_service_function(
        self, function_name: str, kwargs: Dict[str, Any]
    ) -> Any:
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

    # Get the actual service object to access tool functions
    # The 'service' parameter is already the actual service object passed from the middleware

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
            logger.debug(f"Parsed JSON-RPC request: {request_data}")

            # Handle JSON-RPC request
            if request_data.get("method") == "initialize":
                logger.debug(f"Received initialize request: {request_data}")
                # Return server capabilities and information
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                            "logging": {},
                        },
                        "serverInfo": {"name": "Hypha MCP Server", "version": "1.0.0"},
                    },
                }

            elif request_data.get("method") == "tools/list":
                logger.debug(f"Received tools/list request: {request_data}")
                # Check if service has list_tools function OR if MCP server has tools configured
                has_list_tools = "list_tools" in service
                # Check for tools in service.tools (actual functions) and service_info.service_schema.tools (schema)
                has_inline_tools = (
                    hasattr(service, "tools")
                    and service.tools
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "tools")
                    and service_info.service_schema.tools
                    and len(service.tools) > 0
                )
                has_tools = has_list_tools or has_inline_tools

                if has_tools:
                    try:
                        # Use MCP server's list_tools handler which handles both approaches
                        if "list_tools" in service:
                            logger.debug("Using list_tools function")
                            # Direct function call
                            tools = await mcp_server._call_service_function(
                                "list_tools", {}
                            )
                        else:
                            logger.debug("Using inline tools extraction")
                            # Use the actual tool functions from service.tools
                            tools = []
                            logger.debug(f"About to process {len(service.tools)} tools")
                            # Iterate through both the actual functions and their schema
                            for key, tool_func in service.tools.items():
                                assert (
                                    key == tool_func.__name__
                                ), f"Tool name mismatch: {key} != {tool_func.__name__}"
                                logger.debug(
                                    f"Processing tool function {key}: {tool_func}"
                                )
                                try:
                                    # Extract tool schema using the MCP server's method
                                    tool_dict = (
                                        mcp_server._extract_tool_from_schema_function(
                                            tool_func
                                        )
                                    )
                                    logger.debug(f"Extracted tool dict: {tool_dict}")
                                    tools.append(tool_dict)
                                except Exception as e:
                                    logger.exception(
                                        f"Error extracting tool schema for function {key}: {e}"
                                    )
                                    # Fallback: use schema from service_info if available
                                    if key in service_info.service_schema.tools:
                                        tool_schema = service_info.service_schema.tools[
                                            key
                                        ]
                                        if hasattr(tool_schema, "function"):
                                            func_info = tool_schema.function
                                            tool_dict = {
                                                "name": func_info.name,
                                                "description": func_info.description,
                                                "inputSchema": func_info.parameters,
                                            }
                                            logger.debug(
                                                f"Fallback tool dict: {tool_dict}"
                                            )
                                            tools.append(tool_dict)
                            logger.debug(
                                f"Finished processing tools, got {len(tools)} tools"
                            )

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
                                if hasattr(tool_data, "model_dump"):
                                    tool_dict = tool_data.model_dump()
                                elif isinstance(tool_data, dict):
                                    tool_dict = tool_data
                                else:
                                    tool_dict = {
                                        "name": str(tool_data),
                                        "description": "",
                                        "inputSchema": {},
                                    }

                                tools_data.append(tool_dict)
                        else:
                            tools_data = []

                        logger.debug(f"Final tools_data: {tools_data}")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"tools": tools_data},
                        }
                    except Exception as e:
                        logger.exception("Error in tools/list handler")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    logger.debug(f"No tools detected, returning empty list")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"tools": []},
                    }

            elif request_data.get("method") == "tools/call":
                # Check if service has call_tool function OR if MCP server has tools configured
                has_call_tool = "call_tool" in service or (
                    hasattr(service, "tools")
                    and service.tools
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "tools")
                    and service_info.service_schema.tools
                    and len(service.tools) > 0
                )

                if has_call_tool:
                    try:
                        params = request_data.get("params", {})

                        # Use appropriate call mechanism
                        if "call_tool" in service:
                            # Direct function call
                            result = await mcp_server._call_service_function(
                                "call_tool", params
                            )
                        else:
                            # Use inline tools - find and call the tool function directly
                            tool_name = params.get("name")
                            arguments = params.get("arguments", {})

                            # Find the matching tool schema and function
                            target_tool_function = None
                            target_tool_schema = None

                            # Search through service tools
                            if hasattr(service, "tools") and service.tools:
                                for key, tool_function in service.tools.items():
                                    assert (
                                        key == tool_function.__name__
                                    ), f"Tool name mismatch: {key} != {tool_function.__name__}"
                                    # Get corresponding schema from service_info if available
                                    tool_schema = None
                                    if (
                                        hasattr(service_info, "service_schema")
                                        and hasattr(
                                            service_info.service_schema, "tools"
                                        )
                                        and service_info.service_schema.tools
                                    ):
                                        tool_schema = service_info.service_schema.tools[
                                            key
                                        ]

                                    # Extract tool name from schema - handle different structures
                                    schema_tool_name = None
                                    if tool_schema:
                                        try:
                                            # Try original service structure: tool_schema.function.name
                                            if hasattr(
                                                tool_schema, "function"
                                            ) and hasattr(tool_schema.function, "name"):
                                                schema_tool_name = (
                                                    tool_schema.function.name
                                                )
                                            # Try unified service structure: tool_schema.function has direct access
                                            elif hasattr(tool_schema, "function"):
                                                # Check if function has a name attribute or if it's a dict
                                                func_obj = tool_schema.function
                                                if hasattr(func_obj, "name"):
                                                    schema_tool_name = func_obj.name
                                                elif (
                                                    isinstance(func_obj, dict)
                                                    and "name" in func_obj
                                                ):
                                                    schema_tool_name = func_obj["name"]
                                                # Try to get from parameters description or other fields
                                                elif hasattr(func_obj, "description"):
                                                    # Extract name from function object itself if available
                                                    if hasattr(
                                                        tool_function, "__name__"
                                                    ):
                                                        schema_tool_name = (
                                                            tool_function.__name__
                                                        )
                                                    elif hasattr(
                                                        tool_function, "__schema__"
                                                    ) and hasattr(
                                                        tool_function.__schema__,
                                                        "function",
                                                    ):
                                                        schema_func = (
                                                            tool_function.__schema__.function
                                                        )
                                                        if hasattr(schema_func, "name"):
                                                            schema_tool_name = (
                                                                schema_func.name
                                                            )
                                                        elif (
                                                            isinstance(
                                                                schema_func, dict
                                                            )
                                                            and "name" in schema_func
                                                        ):
                                                            schema_tool_name = (
                                                                schema_func["name"]
                                                            )
                                        except (AttributeError, KeyError, TypeError):
                                            # Fallback: try to extract from function itself
                                            if hasattr(tool_function, "__name__"):
                                                schema_tool_name = (
                                                    tool_function.__name__
                                                )
                                            elif hasattr(
                                                tool_function, "__schema__"
                                            ) and hasattr(
                                                tool_function.__schema__, "function"
                                            ):
                                                try:
                                                    schema_func = (
                                                        tool_function.__schema__.function
                                                    )
                                                    if hasattr(schema_func, "name"):
                                                        schema_tool_name = (
                                                            schema_func.name
                                                        )
                                                    elif (
                                                        isinstance(schema_func, dict)
                                                        and "name" in schema_func
                                                    ):
                                                        schema_tool_name = schema_func[
                                                            "name"
                                                        ]
                                                except (
                                                    AttributeError,
                                                    KeyError,
                                                    TypeError,
                                                ):
                                                    pass

                                    # If we found a matching tool name, use this tool
                                    if schema_tool_name == tool_name:
                                        target_tool_function = tool_function
                                        target_tool_schema = tool_schema
                                        break

                            if target_tool_function:
                                # Call the schema function directly with unpacked arguments
                                if inspect.iscoroutinefunction(target_tool_function):
                                    result = await target_tool_function(**arguments)
                                else:
                                    result = target_tool_function(**arguments)
                                    if asyncio.isfuture(result) or inspect.iscoroutine(
                                        result
                                    ):
                                        result = await result
                            else:
                                result = f"Tool '{tool_name}' not found"

                        # Convert to proper JSON format
                        if isinstance(result, list) and result:
                            content_data = []
                            for item in result:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                item_data = unwrap_object_proxy(item)

                                # Convert to dict if it's an MCP object
                                if hasattr(item_data, "model_dump"):
                                    item_dict = item_data.model_dump()
                                elif isinstance(item_data, dict):
                                    item_dict = item_data
                                else:
                                    item_dict = {"type": "text", "text": str(item_data)}

                                content_data.append(item_dict)
                        else:
                            # Don't JSON encode the result, just convert to string
                            content_data = [{"type": "text", "text": str(result)}]

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"content": content_data},
                        }
                    except Exception as e:
                        logger.exception("Error in tools/call handler")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "error": {"code": -32601, "message": "Method not found"},
                    }

            elif request_data.get("method") == "prompts/list":
                # Check if service has list_prompts function OR if service has prompts configured
                has_list_prompts = "list_prompts" in service
                has_inline_prompts = (
                    hasattr(service, "prompts")
                    and service.prompts
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "prompts")
                    and service_info.service_schema.prompts
                    and len(service.prompts) > 0
                )
                has_prompts = has_list_prompts or has_inline_prompts

                if has_prompts:
                    try:
                        if has_list_prompts:
                            prompts = await mcp_server._call_service_function(
                                "list_prompts", {}
                            )
                        else:
                            # Extract prompts from service.prompts
                            prompts = []
                            logger.debug(
                                f"Service has {len(service.prompts)} prompt functions"
                            )
                            logger.debug(
                                f"Service schema has {len(service_info.service_schema.prompts)} prompt schemas"
                            )

                            # Extract prompt metadata directly from service.prompts
                            if hasattr(service, "prompts") and service.prompts:
                                logger.debug(f"Service.prompts: {service.prompts}")
                                # Extract metadata from the actual prompt objects
                                for key, prompt_obj in service.prompts.items():
                                    logger.debug(
                                        f"Processing prompt object {key}: {prompt_obj}"
                                    )

                                    # Get metadata from the prompt object
                                    if hasattr(prompt_obj, "name"):
                                        name = prompt_obj.name
                                    elif hasattr(prompt_obj, "get"):
                                        name = prompt_obj.get("name", f"prompt_{key}")
                                    else:
                                        name = f"prompt_{key}"

                                    if hasattr(prompt_obj, "description"):
                                        description = prompt_obj.description
                                    elif hasattr(prompt_obj, "get"):
                                        description = prompt_obj.get(
                                            "description", f"Prompt {key}"
                                        )
                                    else:
                                        description = f"Prompt {key}"

                                    # Get arguments from the schema function if available
                                    arguments = []
                                    if hasattr(prompt_obj, "read") or (
                                        hasattr(prompt_obj, "get")
                                        and prompt_obj.get("read")
                                    ):
                                        read_func = (
                                            prompt_obj.read
                                            if hasattr(prompt_obj, "read")
                                            else prompt_obj.get("read")
                                        )
                                        if hasattr(read_func, "__schema__"):
                                            schema = (
                                                getattr(read_func, "__schema__", {})
                                                or {}
                                            )
                                            parameters = schema.get("parameters", {})
                                            properties = parameters.get(
                                                "properties", {}
                                            )
                                            required = parameters.get("required", [])

                                            # Convert schema properties to prompt arguments
                                            for (
                                                prop_name,
                                                prop_info,
                                            ) in properties.items():
                                                arguments.append(
                                                    {
                                                        "name": prop_name,
                                                        "description": prop_info.get(
                                                            "description", ""
                                                        ),
                                                        "required": prop_name
                                                        in required,
                                                    }
                                                )

                                    prompt_dict = {
                                        "name": name,
                                        "description": description,
                                        "arguments": arguments,
                                    }
                                    logger.debug(
                                        f"Extracted prompt template: {prompt_dict}"
                                    )
                                    prompts.append(prompt_dict)
                            else:
                                logger.debug("No service.prompts found")

                        # Convert to proper JSON format
                        if isinstance(prompts, list) and prompts:
                            prompts_data = []
                            for prompt in prompts:
                                # Handle ObjectProxy wrapping from Hypha RPC
                                prompt_data = unwrap_object_proxy(prompt)

                                # Convert to dict if it's an MCP object
                                if hasattr(prompt_data, "model_dump"):
                                    prompt_dict = prompt_data.model_dump()
                                elif isinstance(prompt_data, dict):
                                    prompt_dict = prompt_data
                                else:
                                    prompt_dict = {
                                        "name": str(prompt_data),
                                        "description": "",
                                        "arguments": [],
                                    }

                                prompts_data.append(prompt_dict)
                        else:
                            prompts_data = []

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"prompts": prompts_data},
                        }
                    except Exception as e:
                        logger.exception("Error in prompts/list handler")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"prompts": []},
                    }

            elif request_data.get("method") == "prompts/get":
                # Check if service has get_prompt function OR if service has inline prompts
                has_get_prompt = "get_prompt" in service
                has_inline_prompts = (
                    hasattr(service, "prompts")
                    and service.prompts
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "prompts")
                    and service_info.service_schema.prompts
                    and len(service.prompts) > 0
                )

                if has_get_prompt:
                    try:
                        params = request_data.get("params", {})
                        result = await mcp_server._call_service_function(
                            "get_prompt", params
                        )
                        # Convert to proper JSON format
                        # Handle ObjectProxy wrapping from Hypha RPC
                        result_data = unwrap_object_proxy(result)

                        if isinstance(result_data, dict):
                            if hasattr(result_data, "model_dump"):
                                result_data = result_data.model_dump()
                        else:
                            result_data = {
                                "description": "No description",
                                "messages": [],
                            }

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": result_data,
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                elif has_inline_prompts:
                    try:
                        params = request_data.get("params", {})
                        prompt_name = params.get("name")
                        arguments = params.get("arguments", {})

                        # Find the prompt by name
                        prompt_handler = None
                        for key, prompt_obj in service.prompts.items():
                            assert (
                                key == prompt_obj.name
                            ), f"Prompt name mismatch: {key} != {prompt_obj.name}"
                            # Get name from prompt object
                            name = prompt_obj.name

                            if name == prompt_name:
                                prompt_handler = (
                                    prompt_obj.read
                                    if hasattr(prompt_obj, "read")
                                    else prompt_obj.get("read")
                                )
                                break

                        if not prompt_handler:
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_data.get("id"),
                                "error": {
                                    "code": -32602,
                                    "message": f"Prompt '{prompt_name}' not found",
                                },
                            }
                        else:
                            # Call the schema function directly
                            if inspect.iscoroutinefunction(prompt_handler):
                                result = await prompt_handler(**arguments)
                            else:
                                result = prompt_handler(**arguments)
                                if asyncio.isfuture(result) or inspect.iscoroutine(
                                    result
                                ):
                                    result = await result

                            # Convert result to proper MCP format
                            if isinstance(result, dict):
                                # If result is already a dict with messages, use it as-is
                                if "messages" in result:
                                    response = {
                                        "jsonrpc": "2.0",
                                        "id": request_data.get("id"),
                                        "result": result,
                                    }
                                else:
                                    # If result doesn't have messages, wrap it
                                    response = {
                                        "jsonrpc": "2.0",
                                        "id": request_data.get("id"),
                                        "result": {
                                            "description": result.get(
                                                "description",
                                                f"Prompt template for {prompt_name}",
                                            ),
                                            "messages": result.get("messages", []),
                                        },
                                    }
                            else:
                                # If result is a string, create a simple prompt response
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_data.get("id"),
                                    "result": {
                                        "description": f"Prompt template for {prompt_name}",
                                        "messages": [
                                            {
                                                "role": "user",
                                                "content": {
                                                    "type": "text",
                                                    "text": str(result),
                                                },
                                            }
                                        ],
                                    },
                                }
                    except Exception as e:
                        logger.exception(
                            f"Error in inline prompts/get handler for {prompt_name}"
                        )
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "error": {"code": -32601, "message": "Method not found"},
                    }

            elif request_data.get("method") == "resources/templates/list":
                # Check if service has list_resource_templates function OR if service has resources configured
                has_list_resource_templates = "list_resource_templates" in service
                has_inline_resources = (
                    hasattr(service, "resources")
                    and service.resources
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "resources")
                    and service_info.service_schema.resources
                    and len(service.resources) > 0
                )
                has_resources = has_list_resource_templates or has_inline_resources

                if has_resources:
                    try:
                        if has_list_resource_templates:
                            templates = await mcp_server._call_service_function(
                                "list_resource_templates", {}
                            )
                        else:
                            # Use MCP server's list_resource_templates method for inline resources
                            templates = []
                            # Extract resource templates from service.resources and service_info.service_schema.resources
                            # Similar to how tools work
                            logger.debug(
                                f"Service has {len(service.resources)} resource functions"
                            )
                            logger.debug(
                                f"Service schema has {len(service_info.service_schema.resources)} resource schemas"
                            )
                            logger.debug(
                                f"Service_info attributes: {[attr for attr in dir(service_info) if not attr.startswith('_')]}"
                            )

                            # Check if service_info has the original resource configuration
                            if (
                                hasattr(service_info, "resources")
                                and service_info.resources
                            ):
                                logger.debug(
                                    f"Found service_info.resources: {service_info.resources}"
                                )
                                # Use the original resource configuration
                                for (
                                    key,
                                    resource_config,
                                ) in service_info.resources.items():
                                    template_dict = {
                                        "uriTemplate": resource_config.get(
                                            "uri", "unknown"
                                        ),
                                        "name": resource_config.get("name", "unknown"),
                                        "description": resource_config.get(
                                            "description", "unknown"
                                        ),
                                        "mimeType": resource_config.get(
                                            "mime_type", "text/plain"
                                        ),
                                    }
                                    logger.debug(
                                        f"Template from service_info.resources: {template_dict}"
                                    )
                                    templates.append(template_dict)
                            else:
                                logger.debug(
                                    "No service_info.resources found, using fallback method"
                                )

                                # Try to access the original resource configuration from various sources
                                logger.debug(
                                    f"MCP server attributes: {[attr for attr in dir(mcp_server) if not attr.startswith('_')]}"
                                )

                                # Check if service has original resource config as metadata
                                if hasattr(service, "_original_resources"):
                                    logger.debug(
                                        f"Service has _original_resources: {service._original_resources}"
                                    )

                                # Check if the service has resource metadata stored somewhere
                                service_attrs = [
                                    attr
                                    for attr in dir(service)
                                    if not attr.startswith("_")
                                ]
                                logger.debug(f"Service attributes: {service_attrs}")

                                # Fallback: Create template from resource functions and schema
                                if (
                                    hasattr(service, "resources")
                                    and service.resources
                                    and hasattr(service_info, "service_schema")
                                    and service_info.service_schema.resources
                                ):
                                    logger.debug(
                                        "Using fallback: service.resources and service_info.service_schema.resources"
                                    )

                                    # service.resources should contain resource objects
                                    # service_info.service_schema.resources should contain schema info
                                    for key, resource_obj in service.resources.items():
                                        logger.debug(
                                            f"Processing resource object {key}: {resource_obj}"
                                        )

                                        # Get corresponding schema if available
                                        resource_schema = None
                                        resource_schema = (
                                            service_info.service_schema.resources[key]
                                        )
                                        logger.debug(
                                            f"Resource schema {key}: {resource_schema}"
                                        )

                                        # Extract metadata from resource object or schema
                                        try:
                                            # Try to get attributes from resource object
                                            uri = getattr(
                                                resource_obj, "uri", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "uri", "resource://unknown"
                                            )
                                            name = getattr(
                                                resource_obj, "name", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "name", f"Resource {i}"
                                            )
                                            description = getattr(
                                                resource_obj, "description", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "description", "A resource"
                                            )
                                            mime_type = getattr(
                                                resource_obj, "mime_type", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "mime_type", "text/plain"
                                            )
                                        except Exception as e:
                                            logger.debug(
                                                f"Failed to extract metadata from resource object: {e}"
                                            )
                                            uri = "resource://unknown"
                                            name = f"Resource {i}"
                                            description = "A resource"
                                            mime_type = "text/plain"

                                        template_dict = {
                                            "uriTemplate": uri,
                                            "name": name,
                                            "description": description,
                                            "mimeType": mime_type,
                                        }
                                        logger.debug(
                                            f"Extracted template dict: {template_dict}"
                                        )
                                        templates.append(template_dict)

                        # Convert to final format
                        templates_data = []
                        if templates:
                            logger.debug(f"Templates type: {type(templates)}")
                            logger.debug(f"First template type: {type(templates[0])}")
                            logger.debug(f"First template: {templates[0]}")

                            for template_data in templates:
                                # Unwrap ObjectProxy first
                                unwrapped_template = unwrap_object_proxy(template_data)

                                # Convert to dict if it's an MCP object
                                if hasattr(unwrapped_template, "model_dump"):
                                    template_dict = unwrapped_template.model_dump()
                                elif isinstance(unwrapped_template, dict):
                                    template_dict = unwrapped_template
                                else:
                                    template_dict = {
                                        "uriTemplate": str(unwrapped_template),
                                        "name": str(unwrapped_template),
                                    }

                                templates_data.append(template_dict)
                        else:
                            templates_data = []

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"resourceTemplates": templates_data},
                        }
                    except Exception as e:
                        logger.exception("Error in resources/templates/list handler")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"resourceTemplates": []},
                    }

            elif request_data.get("method") == "resources/list":
                # Handle resources/list method - this is what many MCP clients call
                # We can reuse the same logic as resources/templates/list but return "resources" instead of "resourceTemplates"
                has_list_resource_templates = "list_resource_templates" in service
                has_inline_resources = (
                    hasattr(service, "resources")
                    and service.resources
                    and hasattr(service_info, "service_schema")
                    and hasattr(service_info.service_schema, "resources")
                    and service_info.service_schema.resources
                    and len(service.resources) > 0
                )
                has_resources = has_list_resource_templates or has_inline_resources

                if has_resources:
                    try:
                        if has_list_resource_templates:
                            resources = await mcp_server._call_service_function(
                                "list_resource_templates", {}
                            )
                        else:
                            # Extract resources from service.resources and service_info.service_schema.resources
                            resources = []
                            logger.debug(
                                f"Service has {len(service.resources)} resource functions"
                            )
                            logger.debug(
                                f"Service schema has {len(service_info.service_schema.resources)} resource schemas"
                            )

                            # Check if service_info has the original resource configuration
                            if (
                                hasattr(service_info, "resources")
                                and service_info.resources
                            ):
                                logger.debug(
                                    f"Found service_info.resources: {service_info.resources}"
                                )
                                # Use the original resource configuration
                                for (
                                    key,
                                    resource_config,
                                ) in service_info.resources.items():
                                    resource_dict = {
                                        "uri": resource_config.get("uri", "unknown"),
                                        "name": resource_config.get("name", "unknown"),
                                        "description": resource_config.get(
                                            "description", "unknown"
                                        ),
                                        "mimeType": resource_config.get(
                                            "mime_type", "text/plain"
                                        ),
                                    }
                                    assert (
                                        resource_dict["uri"] == key
                                    ), f"Resource URI mismatch: {resource_dict['uri']} != {key}"
                                    logger.debug(
                                        f"Resource from service_info.resources: {resource_dict}"
                                    )
                                    resources.append(resource_dict)
                            else:
                                # Fallback: Create resource from resource functions and schema
                                if (
                                    hasattr(service, "resources")
                                    and service.resources
                                    and hasattr(service_info, "service_schema")
                                    and service_info.service_schema.resources
                                ):
                                    logger.debug(
                                        "Using fallback: service.resources and service_info.service_schema.resources"
                                    )

                                    for key, resource_obj in service.resources.items():
                                        logger.debug(
                                            f"Processing resource object {key}: {resource_obj}"
                                        )

                                        # Extract metadata from resource object
                                        try:
                                            # Try to get attributes from resource object
                                            uri = getattr(
                                                resource_obj, "uri", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "uri", "resource://unknown"
                                            )
                                            name = getattr(
                                                resource_obj, "name", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "name", f"Resource {key}"
                                            )
                                            description = getattr(
                                                resource_obj, "description", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "description", "A resource"
                                            )
                                            mime_type = getattr(
                                                resource_obj, "mime_type", None
                                            ) or getattr(
                                                resource_obj, "get", lambda k, d: d
                                            )(
                                                "mime_type", "text/plain"
                                            )
                                        except Exception as e:
                                            logger.debug(
                                                f"Failed to extract metadata from resource object: {e}"
                                            )
                                            uri = "resource://unknown"
                                            name = f"Resource {key}"
                                            description = "A resource"
                                            mime_type = "text/plain"

                                        resource_dict = {
                                            "uri": uri,
                                            "name": name,
                                            "description": description,
                                            "mimeType": mime_type,
                                        }
                                        logger.debug(
                                            f"Extracted resource dict: {resource_dict}"
                                        )
                                        resources.append(resource_dict)

                        # Convert to final format
                        resources_data = []
                        if resources:
                            logger.debug(f"Resources type: {type(resources)}")
                            logger.debug(f"First resource type: {type(resources[0])}")
                            logger.debug(f"First resource: {resources[0]}")

                            for resource_data in resources:
                                # Convert to dict if it's an MCP object
                                if hasattr(resource_data, "model_dump"):
                                    resource_dict = resource_data.model_dump()
                                elif isinstance(resource_data, dict):
                                    resource_dict = resource_data
                                else:
                                    resource_dict = {
                                        "uri": str(resource_data),
                                        "name": str(resource_data),
                                    }

                                resources_data.append(resource_dict)
                        else:
                            resources_data = []

                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "result": {"resources": resources_data},
                        }
                    except Exception as e:
                        logger.exception("Error in resources/list handler")
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "result": {"resources": []},
                    }

            elif request_data.get("method") == "resources/read":
                # Handle reading a specific resource
                params = request_data.get("params", {})
                uri = params.get("uri")

                if not uri:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id"),
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: uri is required",
                        },
                    }
                else:
                    logger.debug(f"Reading resource with URI: {uri}")

                    # Find the resource by URI
                    resource_to_read = None
                    for key, resource in service.resources.items():
                        if resource.get("uri") == uri:
                            resource_to_read = resource
                            assert key == uri, f"Resource URI mismatch: {key} != {uri}"
                            break

                    if not resource_to_read:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id"),
                            "error": {
                                "code": -32602,
                                "message": f"Resource not found: {uri}",
                            },
                        }
                    else:
                        # Call the resource's read function
                        read_function = resource_to_read.get("read")
                        if not read_function:
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_data.get("id"),
                                "error": {
                                    "code": -32601,
                                    "message": f"Resource {uri} has no read function",
                                },
                            }
                        else:
                            try:
                                content = await read_function()
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_data.get("id"),
                                    "result": {
                                        "contents": [
                                            {
                                                "type": "text",
                                                "text": str(content),
                                                "uri": uri,  # Add the required uri field
                                            }
                                        ]
                                    },
                                }
                                logger.debug(f"Final JSON-RPC response: {response}")
                            except Exception as e:
                                logger.error(f"Error reading resource {uri}: {e}")
                                response = {
                                    "jsonrpc": "2.0",
                                    "id": request_data.get("id"),
                                    "error": {
                                        "code": -32603,
                                        "message": f"Internal error reading resource: {str(e)}",
                                    },
                                }

            elif "method" not in request_data:
                logger.debug(f"Invalid request: {request_data}")
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "error": {"code": -32600, "message": "Invalid request"},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_data.get("id"),
                    "error": {"code": -32601, "message": "Method not found"},
                }

        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "id": request_data.get("id") if "request_data" in locals() else None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

        # Send response
        logger.debug(f"Final JSON-RPC response: {response}")
        response_body = json.dumps(response).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
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
    3. It has inline tools/resources/prompts config (via service_info)
    """
    # Check for explicit MCP type in service
    if hasattr(service, "type") and service.type == "mcp":
        return True

    # Check for MCP protocol functions
    mcp_functions = [
        "list_tools",
        "call_tool",
        "list_prompts",
        "get_prompt",
        "list_resource_templates",
        "list_resources",
        "read_resource",
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

    return has_mcp_functions


class MCPRoutingMiddleware:
    """Middleware specifically for routing MCP services."""

    def __init__(self, app: ASGIApp, base_path=None, store=None):
        self.app = app
        self.base_path = base_path.rstrip("/") if base_path else ""
        self.store = store

        # MCP route patterns
        # Main route for streamable HTTP: /{workspace}/mcp/{service_id}/mcp
        mcp_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}/mcp"
        )
        self.mcp_route = Route(mcp_route, endpoint=None)

        # Info route for helpful 404: /{workspace}/mcp/{service_id}
        info_route = self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}"
        self.info_route = Route(info_route, endpoint=None)

        # Future SSE route: /{workspace}/mcp/{service_id}/sse
        sse_route = (
            self.base_path.rstrip("/") + "/{workspace}/mcp/{service_id:path}/sse"
        )
        self.sse_route = Route(sse_route, endpoint=None)

        logger.info(f"MCP middleware initialized with route patterns:")
        logger.info(f"  - Streamable HTTP: {self.mcp_route.path}")

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

            # Check SSE route first (/{workspace}/mcp/{service_id}/sse) - most specific
            sse_match, sse_params = self.sse_route.matches(scope)
            if sse_match == Match.FULL:
                path_params = sse_params.get("path_params", {})
                workspace = path_params["workspace"]
                service_id = path_params["service_id"]

                logger.debug(
                    f"MCP Middleware: SSE route match - workspace='{workspace}', service_id='{service_id}'"
                )

                # Return not implemented for now
                await self._send_error_response(
                    send, 501, "SSE transport not yet implemented"
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
                "sse": f"/{workspace}/mcp/{service_id}/sse (not yet implemented)",
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
            access_token = self._get_access_token_from_cookies(scope)
            authorization = self._get_authorization_header(scope)

            # Login and get user info
            user_info = await self.store.login_optional(
                authorization=authorization, access_token=access_token
            )

            # Get the MCP service
            async with self.store.get_workspace_interface(user_info, workspace) as api:
                try:
                    service_info = await api.get_service_info(
                        service_id, {"mode": _mode}
                    )
                    logger.debug(
                        f"MCP Middleware: Found service '{service_id}' of type '{service_info.type}'"
                    )
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
