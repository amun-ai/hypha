"""Provide an MCP client worker."""

import asyncio
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Union

from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("mcp_client")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048

# Try to import MCP SDK
try:
    import mcp.types as mcp_types
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.session import ClientSession
    MCP_SDK_AVAILABLE = True
    
    # Try to import SSE client
    try:
        from mcp.client.sse import sse_client
        SSE_CLIENT_AVAILABLE = True
    except ImportError:
        logger.warning("SSE client not available. Install with: pip install mcp[sse]")
        SSE_CLIENT_AVAILABLE = False
        
except ImportError:
    logger.warning("MCP SDK not available. Install with: pip install mcp")
    MCP_SDK_AVAILABLE = False
    SSE_CLIENT_AVAILABLE = False


class MCPClientRunner:
    """MCP client worker that connects to MCP servers and exposes tools as Hypha services."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the MCP client worker."""
        self.server = server
        self.initialized = False
        self._mcp_sessions: Dict[str, Dict[str, Any]] = {}
        self.controller_id = str(MCPClientRunner.instance_counter)
        MCPClientRunner.instance_counter += 1

    async def initialize(self) -> None:
        """Initialize the MCP client worker."""
        if not self.initialized:
            await self.server.register_service(self.get_service())
            self.initialized = True

    async def start(
        self,
        client_id: str,
        app_id: str,
        server_url: str,
        public_base_url: str,
        local_base_url: str,
        workspace: str,
        version: str = None,
        token: str = None,
        entry_point: str = None,
        app_type: str = None,
        metadata: dict = None,
    ) -> dict:
        """Start an MCP client session."""
        if not MCP_SDK_AVAILABLE:
            raise RuntimeError("MCP SDK not available. Install with: pip install mcp")

        full_client_id = workspace + "/" + client_id
        if full_client_id in self._mcp_sessions:
            logger.warning(f"MCP client session already exists: {full_client_id}")
            return self._mcp_sessions[full_client_id]

        # Get the MCP servers configuration from metadata
        mcp_servers = metadata.get("mcp_servers", {})
        if not mcp_servers:
            raise ValueError("No MCP servers configuration found in metadata")

        # Create user API connection for service registration in user workspace
        user_api = await connect_to_server({
            "server_url": server_url,
            "client_id": client_id,
            "workspace": workspace,
            "token": token,
            "method_timeout": 30,
        })
        
        # Store user_api separately to avoid serialization issues
        session_info = {
            "id": full_client_id,
            "app_id": app_id,
            "workspace": workspace,
            "client_id": client_id,
            "server_url": server_url,
            "public_base_url": public_base_url,
            "local_base_url": local_base_url,
            "version": version,
            "token": token,
            "app_type": app_type,
            "entry_point": entry_point,
            "metadata": metadata,
            "mcp_servers": mcp_servers,
            "logs": {"log": [], "error": []},
            "mcp_clients": {},
            "registered_services": []
        }

        # Store user_api separately to avoid serialization issues
        session_info["_internal"] = {
            "user_api": user_api
        }

        # Connect to each MCP server and register unified services
        for server_name, server_config in mcp_servers.items():
            await self._connect_and_register_mcp_service(session_info, server_name, server_config)

        # Register the session
        self._mcp_sessions[full_client_id] = session_info
        
        # Log successful startup
        session_info["logs"]["log"].append(f"MCP client session started with {len(mcp_servers)} servers")
        
        # Return session info without internal objects that can't be serialized
        return_info = session_info.copy()
        if "_internal" in return_info:
            del return_info["_internal"]
        
        return return_info

    async def _connect_and_register_mcp_service(self, session_info: dict, server_name: str, server_config: dict):
        """Connect to MCP server and register a single unified service."""
        try:
            server_type = server_config.get("type", "streamable-http")
            server_url = server_config.get("url")
            
            if not server_url:
                raise ValueError(f"Missing URL for MCP server: {server_name}")

            logger.info(f"Connecting to MCP server {server_name} ({server_type}) at {server_url}")
            
            # Support both streamable-http and sse transport types
            if server_type == "streamable-http":
                await self._connect_streamable_http(session_info, server_name, server_config)
            elif server_type == "sse":
                await self._connect_sse(session_info, server_name, server_config)
            else:
                raise ValueError(f"Unsupported MCP server type: {server_type}")
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}", exc_info=True)
            session_info["logs"]["error"].append(f"Failed to connect to {server_name}: {e}")

    async def _connect_streamable_http(self, session_info: dict, server_name: str, server_config: dict):
        """Connect to MCP server using streamable HTTP transport."""
        server_url = server_config.get("url")
        headers = server_config.get("headers", {})
        
        logger.debug(f"Connecting to streamable HTTP server {server_name} at {server_url}")
        
        # Note: Custom headers are not currently supported by streamablehttp_client
        if headers:
            logger.warning(f"Custom headers specified for {server_name} but not supported by streamablehttp_client: {headers}")
        
        # Connect to MCP server and collect tools, resources, prompts
        try:
            async with streamablehttp_client(server_url) as (read_stream, write_stream, get_session_id):
                async with ClientSession(read_stream, write_stream) as mcp_session:
                    # CRITICAL: Initialize the session
                    logger.debug(f"Initializing MCP session for {server_name}")
                    await mcp_session.initialize()
                    
                    # Store session info
                    session_info["mcp_clients"][server_name] = {
                        "config": server_config,
                        "session_id": get_session_id(),
                        "url": server_url,
                        "transport": "streamable-http"
                    }
                    
                    await self._collect_and_register_capabilities(session_info, server_name, mcp_session)
                    
        except Exception as e:
            logger.error(f"Failed to connect to streamable HTTP server {server_name}: {e}", exc_info=True)
            raise

    async def _connect_sse(self, session_info: dict, server_name: str, server_config: dict):
        """Connect to MCP server using SSE transport."""
        if not SSE_CLIENT_AVAILABLE:
            raise RuntimeError("SSE client not available. Install with: pip install mcp[sse]")
            
        server_url = server_config.get("url")
        headers = server_config.get("headers", {})
        
        logger.debug(f"Connecting to SSE server {server_name} at {server_url}")
        
        # Connect to MCP server and collect tools, resources, prompts
        try:
            async with sse_client(url=server_url) as streams:
                async with ClientSession(*streams) as mcp_session:
                    # CRITICAL: Initialize the session
                    logger.debug(f"Initializing MCP SSE session for {server_name}")
                    await mcp_session.initialize()
                    
                    # Store session info
                    session_info["mcp_clients"][server_name] = {
                        "config": server_config,
                        "session_id": f"sse-{uuid.uuid4().hex[:8]}",
                        "url": server_url,
                        "transport": "sse"
                    }
                    
                    await self._collect_and_register_capabilities(session_info, server_name, mcp_session)
                    
        except Exception as e:
            logger.error(f"Failed to connect to SSE server {server_name}: {e}", exc_info=True)
            raise

    async def _collect_and_register_capabilities(self, session_info: dict, server_name: str, mcp_session):
        """Collect tools, resources, and prompts from MCP session and register unified service."""
        # Collect tools
        tools = []
        try:
            tools_result = await mcp_session.list_tools()
            if tools_result and hasattr(tools_result, 'tools'):
                tools = await self._create_tool_wrappers(session_info, server_name, tools_result.tools)
                logger.info(f"Found {len(tools)} tools from {server_name}")
        except Exception as e:
            logger.debug(f"No tools found in MCP server {server_name}: {e}", exc_info=True)
            session_info["logs"]["log"].append(f"No tools found in {server_name}: {e}")

        # Collect resources  
        resources = []
        try:
            resources_result = await mcp_session.list_resources()
            if resources_result and hasattr(resources_result, 'resources'):
                resources = await self._create_resource_wrappers(session_info, server_name, resources_result.resources)
                logger.info(f"Found {len(resources)} resources from {server_name}")
        except Exception as e:
            logger.debug(f"No resources found in MCP server {server_name}: {e}", exc_info=True)
            session_info["logs"]["log"].append(f"No resources found in {server_name}: {e}")

        # Collect prompts
        prompts = []
        try:
            prompts_result = await mcp_session.list_prompts()
            if prompts_result and hasattr(prompts_result, 'prompts'):
                prompts = await self._create_prompt_wrappers(session_info, server_name, prompts_result.prompts)
                logger.info(f"Found {len(prompts)} prompts from {server_name}")
        except Exception as e:
            logger.debug(f"No prompts found in MCP server {server_name}: {e}", exc_info=True)
            session_info["logs"]["log"].append(f"No prompts found in {server_name}: {e}")

        # Register the unified MCP service
        await self._register_unified_mcp_service(session_info, server_name, tools, resources, prompts)

    def _create_mcp_client_context(self, session_info: dict, server_name: str):
        """Create appropriate MCP client context based on transport type."""
        server_config = session_info["mcp_clients"][server_name]["config"]
        server_url = server_config.get("url")
        transport = session_info["mcp_clients"][server_name]["transport"]
        
        if transport == "streamable-http":
            return streamablehttp_client(server_url)
        elif transport == "sse":
            if not SSE_CLIENT_AVAILABLE:
                raise RuntimeError("SSE client not available. Install with: pip install mcp[sse]")
            return sse_client(url=server_url)
        else:
            raise ValueError(f"Unsupported transport type: {transport}")

    def _wrap_tool(self, session_info: dict, server_name: str, tool_name: str, tool_description: str, tool_schema: dict):
        """Create a tool wrapper function with proper closure."""
        async def tool_wrapper(**kwargs):
            """Wrapper function to call MCP tool."""
            logger.debug(f"Calling MCP tool {server_name}.{tool_name} with args: {kwargs}")
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(session_info, server_name)
                transport = session_info["mcp_clients"][server_name]["transport"]
                
                # Connect and call tool
                if transport == "streamable-http":
                    async with client_context as (read_stream, write_stream, get_session_id):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(name=tool_name, arguments=kwargs)
                elif transport == "sse":
                    async with client_context as streams:
                        async with ClientSession(*streams) as session:
                            await session.initialize()
                            result = await session.call_tool(name=tool_name, arguments=kwargs)
                else:
                    raise ValueError(f"Unsupported transport: {transport}")
                
                # Extract content from MCP result
                if hasattr(result, 'content') and result.content:
                    if len(result.content) == 1:
                        return result.content[0].text
                    else:
                        return [content.text for content in result.content]
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error calling MCP tool {server_name}.{tool_name}: {e}", exc_info=True)
                raise

        # Set function name and schema
        tool_wrapper.__name__ = tool_name
        tool_wrapper.__schema__ = {
            "name": tool_name,
            "description": tool_description,
            "parameters": tool_schema
        }
        
        return tool_wrapper

    def _wrap_resource(self, session_info: dict, server_name: str, resource_uri: str, resource_name: str, resource_description: str, resource_mime_type: str):
        """Create a resource wrapper function with proper closure."""
        async def resource_read():
            """Wrapper function to read MCP resource."""
            logger.debug(f"Reading MCP resource {server_name}.{resource_uri}")
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(session_info, server_name)
                transport = session_info["mcp_clients"][server_name]["transport"]
                
                # Connect and read resource
                if transport == "streamable-http":
                    async with client_context as (read_stream, write_stream, get_session_id):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.read_resource(uri=resource_uri)
                elif transport == "sse":
                    async with client_context as streams:
                        async with ClientSession(*streams) as session:
                            await session.initialize()
                            result = await session.read_resource(uri=resource_uri)
                else:
                    raise ValueError(f"Unsupported transport: {transport}")
                
                # Extract content from MCP result
                if hasattr(result, 'contents') and result.contents:
                    if len(result.contents) == 1:
                        return result.contents[0].text
                    else:
                        return [content.text for content in result.contents]
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error reading MCP resource {server_name}.{resource_uri}: {e}", exc_info=True)
                raise

        # Set function name and schema
        resource_read.__name__ = f"read_{resource_name.replace(' ', '_').replace('-', '_')}"
        resource_read.__schema__ = {
            "name": resource_name,
            "description": f"Read {resource_description}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        return {
            "uri": str(resource_uri),  # Convert to string to avoid Pydantic AnyUrl serialization issues
            "name": str(resource_name),  # Ensure it's a string
            "description": str(resource_description),  # Ensure it's a string
            "tags": ["mcp", server_name],
            "mime_type": str(resource_mime_type),  # Ensure it's a string
            "read": resource_read,
        }

    def _wrap_prompt(self, session_info: dict, server_name: str, prompt_name: str, prompt_description: str, prompt_args: list):
        """Create a prompt wrapper function with proper closure."""
        # Build parameters schema from prompt arguments
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for arg in prompt_args:
            # Convert MCP argument objects to basic types to avoid Pydantic serialization issues
            arg_name = str(arg.name)
            arg_description = str(getattr(arg, 'description', ''))
            is_required = bool(getattr(arg, 'required', False))
            
            parameters["properties"][arg_name] = {
                "type": "string",
                "description": arg_description
            }
            
            if is_required:
                parameters["required"].append(arg_name)

        async def prompt_read(**kwargs):
            """Wrapper function to get MCP prompt."""
            logger.debug(f"Reading MCP prompt {server_name}.{prompt_name} with args: {kwargs}")
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(session_info, server_name)
                transport = session_info["mcp_clients"][server_name]["transport"]
                
                # Connect and get prompt
                if transport == "streamable-http":
                    async with client_context as (read_stream, write_stream, get_session_id):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.get_prompt(name=prompt_name, arguments=kwargs)
                elif transport == "sse":
                    async with client_context as streams:
                        async with ClientSession(*streams) as session:
                            await session.initialize()
                            result = await session.get_prompt(name=prompt_name, arguments=kwargs)
                else:
                    raise ValueError(f"Unsupported transport: {transport}")
                
                # Extract content from MCP result
                if hasattr(result, 'messages') and result.messages:
                    return {
                        "description": str(result.description),
                        "messages": [
                            {
                                "role": str(msg.role),
                                "content": {
                                    "type": "text",
                                    "text": str(msg.content.text)
                                }
                            }
                            for msg in result.messages
                        ]
                    }
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error getting MCP prompt {server_name}.{prompt_name}: {e}", exc_info=True)
                raise

        # Set function name and schema
        prompt_read.__name__ = f"read_{prompt_name.replace(' ', '_').replace('-', '_')}"
        prompt_read.__schema__ = {
            "name": prompt_name,
            "description": prompt_description,
            "parameters": parameters
        }

        return {
            "name": str(prompt_name),  # Ensure it's a string
            "description": str(prompt_description),  # Ensure it's a string
            "tags": ["mcp", server_name],
            "read": prompt_read,
        }

    async def _create_tool_wrappers(self, session_info: dict, server_name: str, tools: List) -> List:
        """Create tool wrapper functions with __schema__ property."""
        tool_wrappers = []
        
        for tool in tools:
            try:
                # Convert MCP objects to basic Python types to avoid Pydantic serialization issues
                tool_name = str(tool.name)
                tool_description = str(tool.description)
                tool_schema = tool.inputSchema
                
                # Convert tool_schema from Pydantic to dict if needed
                if hasattr(tool_schema, 'model_dump'):
                    tool_schema = tool_schema.model_dump()
                elif hasattr(tool_schema, 'dict'):
                    tool_schema = tool_schema.dict()

                # Create wrapper function using dedicated method
                tool_wrapper = self._wrap_tool(session_info, server_name, tool_name, tool_description, tool_schema)
                tool_wrappers.append(tool_wrapper)
                
                logger.info(f"Created tool wrapper: {server_name}.{tool_name}")
                session_info["logs"]["log"].append(f"Created tool wrapper: {server_name}.{tool_name}")

            except Exception as e:
                logger.error(f"Failed to create tool wrapper for {tool.name}: {e}", exc_info=True)
                session_info["logs"]["error"].append(f"Failed to create tool wrapper for {tool.name}: {e}")
        
        return tool_wrappers

    async def _create_resource_wrappers(self, session_info: dict, server_name: str, resources: List) -> List:
        """Create resource wrapper configurations with read functions."""
        resource_configs = []
        
        for resource in resources:
            try:
                # Convert MCP objects to basic Python types to avoid Pydantic serialization issues
                resource_uri = str(resource.uri)
                resource_name = str(resource.name)
                resource_description = str(resource.description)
                resource_mime_type = str(getattr(resource, 'mimeType', 'text/plain'))

                # Create resource config using dedicated method
                resource_config = self._wrap_resource(session_info, server_name, resource_uri, resource_name, resource_description, resource_mime_type)
                resource_configs.append(resource_config)
                
                logger.info(f"Created resource wrapper: {server_name}.{resource_name}")
                session_info["logs"]["log"].append(f"Created resource wrapper: {server_name}.{resource_name}")

            except Exception as e:
                logger.error(f"Failed to create resource wrapper for {resource.name}: {e}", exc_info=True)
                session_info["logs"]["error"].append(f"Failed to create resource wrapper for {resource.name}: {e}")
        
        return resource_configs

    async def _create_prompt_wrappers(self, session_info: dict, server_name: str, prompts: List) -> List:
        """Create prompt wrapper configurations with read functions."""
        prompt_configs = []
        
        for prompt in prompts:
            try:
                # Convert MCP objects to basic Python types to avoid Pydantic serialization issues
                prompt_name = str(prompt.name)
                prompt_description = str(prompt.description)
                prompt_args = getattr(prompt, 'arguments', [])

                # Create prompt config using dedicated method
                prompt_config = self._wrap_prompt(session_info, server_name, prompt_name, prompt_description, prompt_args)
                prompt_configs.append(prompt_config)
                
                logger.info(f"Created prompt wrapper: {server_name}.{prompt_name}")
                session_info["logs"]["log"].append(f"Created prompt wrapper: {server_name}.{prompt_name}")

            except Exception as e:
                logger.error(f"Failed to create prompt wrapper for {prompt.name}: {e}", exc_info=True)
                session_info["logs"]["error"].append(f"Failed to create prompt wrapper for {prompt.name}: {e}")
        
        return prompt_configs

    async def _register_unified_mcp_service(self, session_info: dict, server_name: str, tools: List, resources: List, prompts: List):
        """Register a unified MCP service with tools, resources, and prompts."""
        try:
            # Create service configuration
            service_info = {
                "id": server_name,  # Use server name as service ID
                "name": f"MCP {server_name.title()} Service",
                "description": f"MCP service for {server_name} server with tools, resources, and prompts",
                "type": "mcp",
                "config": {
                    "visibility": "public",
                    "run_in_executor": True,
                },
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
            }

            # Register the service
            registered_service = await session_info["_internal"]["user_api"].register_service(service_info)
            session_info["registered_services"].append(registered_service)
            
            logger.info(f"Registered unified MCP service: {server_name}")
            logger.info(f"  - Tools: {len(tools)}")
            logger.info(f"  - Resources: {len(resources)}")
            logger.info(f"  - Prompts: {len(prompts)}")
            
            session_info["logs"]["log"].append(f"Registered unified MCP service: {server_name}")
            session_info["logs"]["log"].append(f"  - Tools: {len(tools)}, Resources: {len(resources)}, Prompts: {len(prompts)}")
            await session_info["_internal"]["user_api"].export({
                "setup": lambda: logger.info(f"MCP server {server_name} setup complete")
            })

        except Exception as e:
            logger.error(f"Failed to register unified MCP service for {server_name}: {e}", exc_info=True)
            session_info["logs"]["error"].append(f"Failed to register unified MCP service for {server_name}: {e}")

    async def stop(self, session_id: str) -> None:
        """Stop an MCP client session."""
        if session_id in self._mcp_sessions:
            session_info = self._mcp_sessions.pop(session_id)
            
            # Unregister services using the user API
            user_api = session_info.get("_internal", {}).get("user_api")
            for service in session_info.get("registered_services", []):
                try:
                    if user_api:
                        await user_api.unregister_service(service["id"])
                    else:
                        # Fallback to server instance
                        await self.server.unregister_service(service["id"])
                except Exception as e:
                    logger.warning(f"Failed to unregister service {service['id']}: {e}")
            
            # Disconnect the user API
            if user_api:
                try:
                    await user_api.disconnect()
                except Exception as e:
                    logger.warning(f"Failed to disconnect user API for session {session_id}: {e}")
            
            logger.info(f"Stopped MCP client session: {session_id}")
        else:
            logger.warning(f"MCP client session not found: {session_id}")

    async def get_logs(
        self,
        session_id: str,
        type: str = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for an MCP client session."""
        if session_id not in self._mcp_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_info = self._mcp_sessions[session_id]
        logs = session_info["logs"]
        
        if type:
            # Return specific log type
            target_logs = logs.get(type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            # Return all logs
            result = {}
            for log_type, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type] = log_entries[offset:end_idx]
            return result

    async def list_sessions(self) -> List[dict]:
        """List all active MCP client sessions."""
        return [
            {
                "id": session_id,
                "app_id": session_info["app_id"],
                "workspace": session_info["workspace"],
                "mcp_servers": list(session_info["mcp_servers"].keys()),
                "registered_services": len(session_info["registered_services"])
            }
            for session_id, session_info in self._mcp_sessions.items()
        ]

    def get_service(self) -> dict:
        """Get the service definition for the MCP proxy worker."""
        return {
            "id": f"mcp-proxy-worker-{self.controller_id}",
            "name": "MCP Proxy Worker",
            "description": "MCP proxy worker for connecting to MCP servers",
            "type": "server-app-worker",
            "config": {
                "visibility": "public",
                "run_in_executor": True,
            },
            "supported_types": ["mcp-server"],
            "start": self.start,
            "stop": self.stop,
            "get_logs": self.get_logs,
            "list_sessions": self.list_sessions,
            "prepare_workspace": self.prepare_workspace,
            "close_workspace": self.close_workspace,
        }

    async def prepare_workspace(self, workspace_id: str) -> None:
        """Prepare workspace for MCP client operations."""
        logger.info(f"Preparing workspace {workspace_id} for MCP client operations")

    async def close_workspace(self, workspace_id: str) -> None:
        """Close workspace and cleanup MCP client sessions."""
        logger.info(f"Closing workspace {workspace_id} and cleaning up MCP client sessions")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._mcp_sessions.items()
            if session_info["workspace"] == workspace_id
        ]
        
        for session_id in sessions_to_stop:
            await self.stop(session_id)

async def hypha_startup(server):
    """Initialize the MCP client worker as a startup function."""
    mcp_client_runner = MCPClientRunner(server)
    await mcp_client_runner.initialize()
    logger.info("MCP client worker registered as startup function")