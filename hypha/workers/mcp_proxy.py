"""MCP (Model Context Protocol) Proxy Worker for connecting to MCP servers and exposing tools as Hypha services."""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from hypha.core import UserInfo
from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("mcp_proxy")
logger.setLevel(LOGLEVEL)

try:
    import mcp
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.types import CallToolRequest, ListToolsRequest, ReadResourceRequest, GetPromptRequest
    MCP_AVAILABLE = True
except ImportError:
    logger.warning("MCP not available, MCP proxy worker will not function")
    MCP_AVAILABLE = False

if not MCP_AVAILABLE:
    @asynccontextmanager
    async def asyncio_timeout(delay):
        """Fallback timeout manager when MCP is not available."""
        yield
else:
    from asyncio import timeout as asyncio_timeout


# MCP-specific exceptions and schemas
class StreamableHTTPTransport:
    """A transport for HTTP-based streamable MCP servers."""
    
    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {}
        self.client = httpx.AsyncClient()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server."""
        response = await self.client.post(
            self.url,
            json=request_data,
            headers={**self.headers, "Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()


class MCPClientRunner(BaseWorker):
    """MCP client worker that connects to MCP servers and exposes tools as Hypha services."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the MCP client runner."""
        super().__init__(server)
        self.controller_id = str(MCPClientRunner.instance_counter)
        MCPClientRunner.instance_counter += 1
        
        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["mcp-server"]

    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "MCP Proxy Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "MCP proxy worker for connecting to MCP servers"

    async def compile(self, manifest: dict, files: list, config: dict = None) -> tuple[dict, list]:
        """Compile MCP server manifest and files.
        
        This method processes MCP server configuration:
        1. Looks for 'source' file containing JSON configuration
        2. Extracts mcpServers from the source or manifest
        3. Updates manifest with proper MCP server settings
        4. Generates source file with final configuration
        """
        # For MCP servers, check if we need to generate source from manifest
        if manifest.get("type") == "mcp-server":
            # Extract MCP servers configuration
            mcp_servers = manifest.get("mcpServers", {})
            
            # Look for source file that might contain additional config
            source_file = None
            for file_info in files:
                if file_info.get("name") == "source":
                    source_file = file_info
                    break
            
            # If source file exists, try to parse it as JSON to extract mcpServers
            if source_file:
                try:
                    source_content = source_file.get("content", "{}")
                    source_json = json.loads(source_content) if source_content.strip() else {}
                    
                    # Merge servers from source with manifest
                    if "mcpServers" in source_json:
                        source_servers = source_json["mcpServers"]
                        # Manifest takes precedence, but use source as fallback
                        merged_servers = {**source_servers, **mcp_servers}
                        mcp_servers = merged_servers
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse source as JSON: {e}")
            
            # Create final configuration
            final_config = {
                "type": "mcp-server",
                "mcpServers": mcp_servers,
                "name": manifest.get("name", "MCP Server"),
                "description": manifest.get("description", "MCP server application"),
                "version": manifest.get("version", "1.0.0"),
            }
            
            # Merge any additional manifest fields
            for key, value in manifest.items():
                if key not in final_config:
                    final_config[key] = value
            
            # Generate source file with final configuration
            source_content = json.dumps(final_config, indent=2)
            
            # Create new files list, replacing or adding source
            new_files = [f for f in files if f.get("name") != "source"]
            new_files.append({
                "name": "source", 
                "content": source_content,
                "format": "text"
            })
            
            return final_config, new_files
        
        # Not an MCP server type, return unchanged
        return manifest, files

    async def start(self, config: Union[WorkerConfig, Dict[str, Any]]) -> str:
        """Start a new MCP client session."""
        if not MCP_AVAILABLE:
            raise WorkerError("MCP library not available")
        
        # Handle both pydantic model and dict input for RPC compatibility
        if isinstance(config, dict):
            config = WorkerConfig(**config)
        
        session_id = config.id
        
        if session_id in self._sessions:
            raise WorkerError(f"Session {session_id} already exists")
        
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
            session_data = await self._start_mcp_session(config)
            self._session_data[session_id] = session_data
            
            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started MCP session {session_id}")
            
            return session_id
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start MCP session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            raise

    async def _start_mcp_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start MCP client session and connect to configured servers."""
        if not MCP_AVAILABLE:
            raise Exception("MCP library not available")

        # Extract MCP servers configuration from manifest
        manifest = config.manifest
        mcp_servers = manifest.get("mcpServers", {})
        
        if not mcp_servers:
            raise Exception("No MCP servers configured in manifest")

        session_data = {
            "connections": {},
            "logs": {"info": [], "error": [], "debug": []},
            "services": []
        }

        # Connect to each MCP server
        for server_name, server_config in mcp_servers.items():
            try:
                logger.info(f"Connecting to MCP server: {server_name}")
                await self._connect_and_register_mcp_service(session_data, server_name, server_config)
                
                session_data["logs"]["info"].append(f"Successfully connected to MCP server: {server_name}")
                logger.info(f"Successfully connected to MCP server: {server_name}")
                
            except Exception as e:
                error_msg = f"Failed to connect to MCP server {server_name}: {e}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                # Continue with other servers even if one fails
                continue

        if not session_data["connections"]:
            raise Exception("Failed to connect to any MCP servers")

        return session_data

    async def _connect_and_register_mcp_service(self, session_data: dict, server_name: str, server_config: dict):
        """Connect to an MCP server and register its capabilities as Hypha services."""
        server_type = server_config.get("type", "stdio")
        
        if server_type == "streamable-http":
            await self._connect_streamable_http(session_data, server_name, server_config)
        elif server_type == "sse":
            await self._connect_sse(session_data, server_name, server_config)
        else:
            raise Exception(f"Unsupported MCP server type: {server_type}")

    async def _connect_streamable_http(self, session_data: dict, server_name: str, server_config: dict):
        """Connect to a streamable HTTP MCP server."""
        url = server_config.get("url")
        if not url:
            raise Exception(f"No URL specified for streamable-http server {server_name}")
        
        headers = server_config.get("headers", {})
        
        # Create transport
        transport = StreamableHTTPTransport(url, headers)
        
        # For streamable HTTP, we don't have persistent sessions like stdio/sse
        # Instead, we make direct HTTP calls for each operation
        session_data["connections"][server_name] = {
            "transport": transport,
            "type": "streamable-http",
            "config": server_config
        }
        
        # Collect and register server capabilities
        await self._collect_and_register_capabilities(session_data, server_name, None)

    async def _connect_sse(self, session_data: dict, server_name: str, server_config: dict):
        """Connect to an SSE (Server-Sent Events) MCP server."""
        url = server_config.get("url")
        if not url:
            raise Exception(f"No URL specified for SSE server {server_name}")
        
        headers = server_config.get("headers", {})
        
        try:
            # Use the SSE client from MCP library
            async with sse_client(url, headers=headers) as (read, write):
                mcp_session = ClientSession(read, write)
                await mcp_session.initialize()
                
                # Store connection info
                session_data["connections"][server_name] = {
                    "session": mcp_session,
                    "read": read,
                    "write": write,
                    "type": "sse",
                    "config": server_config
                }
                
                # Collect and register server capabilities
                await self._collect_and_register_capabilities(session_data, server_name, mcp_session)
                
        except Exception as e:
            logger.error(f"Failed to connect to SSE server {server_name}: {e}")
            raise

    async def _collect_and_register_capabilities(self, session_data: dict, server_name: str, mcp_session):
        """Collect server capabilities and register them as Hypha services."""
        connection_info = session_data["connections"][server_name]
        server_type = connection_info["type"]
        
        tools = []
        resources = []
        prompts = []
        
        try:
            if server_type == "streamable-http":
                # For streamable HTTP, make direct HTTP calls
                transport = connection_info["transport"]
                
                # List tools
                try:
                    tools_response = await transport.send_request({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list"
                    })
                    tools = tools_response.get("result", {}).get("tools", [])
                except Exception as e:
                    logger.warning(f"Failed to list tools from {server_name}: {e}")
                
                # List resources
                try:
                    resources_response = await transport.send_request({
                        "jsonrpc": "2.0", 
                        "id": 2,
                        "method": "resources/list"
                    })
                    resources = resources_response.get("result", {}).get("resources", [])
                except Exception as e:
                    logger.warning(f"Failed to list resources from {server_name}: {e}")
                
                # List prompts
                try:
                    prompts_response = await transport.send_request({
                        "jsonrpc": "2.0",
                        "id": 3, 
                        "method": "prompts/list"
                    })
                    prompts = prompts_response.get("result", {}).get("prompts", [])
                except Exception as e:
                    logger.warning(f"Failed to list prompts from {server_name}: {e}")
                    
            else:
                # For other types (SSE, stdio), use MCP session
                if mcp_session:
                    # List tools
                    try:
                        tools_result = await mcp_session.list_tools()
                        tools = tools_result.tools if hasattr(tools_result, 'tools') else []
                    except Exception as e:
                        logger.warning(f"Failed to list tools from {server_name}: {e}")
                    
                    # List resources  
                    try:
                        resources_result = await mcp_session.list_resources()
                        resources = resources_result.resources if hasattr(resources_result, 'resources') else []
                    except Exception as e:
                        logger.warning(f"Failed to list resources from {server_name}: {e}")
                    
                    # List prompts
                    try:
                        prompts_result = await mcp_session.list_prompts()
                        prompts = prompts_result.prompts if hasattr(prompts_result, 'prompts') else []
                    except Exception as e:
                        logger.warning(f"Failed to list prompts from {server_name}: {e}")
            
            logger.info(f"Server {server_name}: Found {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")
            
            # Register unified service for this server
            await self._register_unified_mcp_service(session_data, server_name, tools, resources, prompts)
            
        except Exception as e:
            logger.error(f"Failed to collect capabilities from {server_name}: {e}")
            raise

    def _create_mcp_client_context(self, session_data: dict, server_name: str):
        """Create context object that MCP client functions can access."""
        return {
            "session_data": session_data,
            "server_name": server_name,
            "server": self.server,
            "logger": logger
        }

    def _wrap_tool(self, session_data: dict, server_name: str, tool_name: str, tool_description: str, tool_schema: dict):
        """Create a wrapper function for an MCP tool."""
        async def tool_wrapper(**kwargs):
            try:
                connection_info = session_data["connections"][server_name]
                server_type = connection_info["type"]
                
                if server_type == "streamable-http":
                    # Use HTTP transport
                    transport = connection_info["transport"]
                    response = await transport.send_request({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": kwargs
                        }
                    })
                    
                    if "error" in response:
                        raise Exception(f"Tool call failed: {response['error']}")
                    
                    return response.get("result", {})
                    
                else:
                    # Use MCP session
                    mcp_session = connection_info["session"]
                    
                    async with asyncio_timeout(30):
                        # Create tool call request
                        call_request = CallToolRequest(
                            method="tools/call",
                            params={
                                "name": tool_name,
                                "arguments": kwargs
                            }
                        )
                        
                        result = await mcp_session.call_tool(call_request)
                        
                        if hasattr(result, 'content') and result.content:
                            # Extract content from MCP response
                            content_items = []
                            for item in result.content:
                                if hasattr(item, 'text'):
                                    content_items.append(item.text)
                                elif hasattr(item, 'type') and item.type == "text":
                                    content_items.append(getattr(item, 'text', str(item)))
                                else:
                                    content_items.append(str(item))
                            
                            return {
                                "content": content_items,
                                "isError": getattr(result, 'isError', False)
                            }
                        else:
                            return {"content": [], "isError": False}
                        
            except Exception as e:
                error_msg = f"Error calling tool {tool_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
        
        return tool_wrapper

    def _wrap_resource(self, session_data: dict, server_name: str, resource_uri: str, resource_name: str, resource_description: str, resource_mime_type: str):
        """Create a wrapper function for an MCP resource."""
        async def resource_read():
            try:
                connection_info = session_data["connections"][server_name]
                server_type = connection_info["type"]
                
                if server_type == "streamable-http":
                    # Use HTTP transport
                    transport = connection_info["transport"]
                    response = await transport.send_request({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "resources/read",
                        "params": {
                            "uri": resource_uri
                        }
                    })
                    
                    if "error" in response:
                        raise Exception(f"Resource read failed: {response['error']}")
                    
                    return response.get("result", {})
                    
                else:
                    # Use MCP session
                    mcp_session = connection_info["session"]
                    
                    async with asyncio_timeout(30):
                        # Create resource read request
                        read_request = ReadResourceRequest(
                            method="resources/read",
                            params={"uri": resource_uri}
                        )
                        
                        result = await mcp_session.read_resource(read_request)
                        
                        if hasattr(result, 'contents') and result.contents:
                            # Extract contents from MCP response
                            content_items = []
                            for item in result.contents:
                                if hasattr(item, 'text'):
                                    content_items.append({
                                        "type": "text",
                                        "text": item.text,
                                        "uri": resource_uri
                                    })
                                elif hasattr(item, 'blob'):
                                    content_items.append({
                                        "type": "blob", 
                                        "blob": item.blob,
                                        "uri": resource_uri,
                                        "mimeType": resource_mime_type
                                    })
                                else:
                                    content_items.append({
                                        "type": "text",
                                        "text": str(item),
                                        "uri": resource_uri
                                    })
                            
                            return {
                                "contents": content_items,
                                "uri": resource_uri,
                                "mimeType": resource_mime_type
                            }
                        else:
                            return {
                                "contents": [],
                                "uri": resource_uri, 
                                "mimeType": resource_mime_type
                            }
                            
            except Exception as e:
                error_msg = f"Error reading resource {resource_uri}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
        
        return resource_read

    def _wrap_prompt(self, session_data: dict, server_name: str, prompt_name: str, prompt_description: str, prompt_args: list):
        """Create a wrapper function for an MCP prompt."""
        # Extract argument names and create function signature
        arg_names = []
        for arg in prompt_args:
            if isinstance(arg, dict):
                arg_names.append(arg.get('name', 'arg'))
            else:
                # Fallback for different argument formats
                arg_names.append(getattr(arg, 'name', 'arg'))
        
        async def prompt_read(**kwargs):
            try:
                connection_info = session_data["connections"][server_name]
                server_type = connection_info["type"]
                
                if server_type == "streamable-http":
                    # Use HTTP transport
                    transport = connection_info["transport"]
                    response = await transport.send_request({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "prompts/get",
                        "params": {
                            "name": prompt_name,
                            "arguments": kwargs
                        }
                    })
                    
                    if "error" in response:
                        raise Exception(f"Prompt get failed: {response['error']}")
                    
                    return response.get("result", {})
                    
                else:
                    # Use MCP session  
                    mcp_session = connection_info["session"]
                    
                    async with asyncio_timeout(30):
                        # Create prompt get request
                        get_request = GetPromptRequest(
                            method="prompts/get",
                            params={
                                "name": prompt_name,
                                "arguments": kwargs
                            }
                        )
                        
                        result = await mcp_session.get_prompt(get_request)
                        
                        if hasattr(result, 'messages') and result.messages:
                            # Extract messages from MCP response
                            messages = []
                            for msg in result.messages:
                                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                                    content_items = []
                                    if hasattr(msg.content, '__iter__'):
                                        for item in msg.content:
                                            if hasattr(item, 'text'):
                                                content_items.append({
                                                    "type": "text", 
                                                    "text": item.text
                                                })
                                            else:
                                                content_items.append({
                                                    "type": "text",
                                                    "text": str(item)
                                                })
                                    else:
                                        content_items.append({
                                            "type": "text",
                                            "text": str(msg.content)
                                        })
                                    
                                    messages.append({
                                        "role": msg.role,
                                        "content": content_items
                                    })
                                else:
                                    # Fallback for different message formats
                                    messages.append({
                                        "role": getattr(msg, 'role', 'user'),
                                        "content": [{"type": "text", "text": str(msg)}]
                                    })
                            
                            return {
                                "messages": messages,
                                "description": prompt_description
                            }
                        else:
                            return {
                                "messages": [],
                                "description": prompt_description
                            }
                            
            except Exception as e:
                error_msg = f"Error getting prompt {prompt_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
        
        return prompt_read

    async def _create_tool_wrappers(self, session_data: dict, server_name: str, tools: List) -> List:
        """Create wrapper functions for all tools."""
        tool_wrappers = []
        
        for tool in tools:
            try:
                tool_name = tool.get('name') if isinstance(tool, dict) else getattr(tool, 'name', 'unknown')
                tool_description = tool.get('description') if isinstance(tool, dict) else getattr(tool, 'description', '')
                tool_schema = tool.get('inputSchema') if isinstance(tool, dict) else getattr(tool, 'inputSchema', {})
                
                wrapper = self._wrap_tool(session_data, server_name, tool_name, tool_description, tool_schema)
                
                tool_wrappers.append({
                    "name": tool_name,
                    "description": tool_description,
                    "schema": tool_schema,
                    "function": wrapper
                })
                
            except Exception as e:
                logger.error(f"Failed to create wrapper for tool {tool}: {e}")
                continue
        
        return tool_wrappers

    async def _create_resource_wrappers(self, session_data: dict, server_name: str, resources: List) -> List:
        """Create wrapper functions for all resources."""
        resource_wrappers = []
        
        for resource in resources:
            try:
                resource_uri = resource.get('uri') if isinstance(resource, dict) else getattr(resource, 'uri', 'unknown')
                resource_name = resource.get('name') if isinstance(resource, dict) else getattr(resource, 'name', resource_uri)
                resource_description = resource.get('description') if isinstance(resource, dict) else getattr(resource, 'description', '')
                resource_mime_type = resource.get('mimeType') if isinstance(resource, dict) else getattr(resource, 'mimeType', 'text/plain')
                
                wrapper = self._wrap_resource(session_data, server_name, resource_uri, resource_name, resource_description, resource_mime_type)
                
                resource_wrappers.append({
                    "name": resource_name,
                    "uri": resource_uri,
                    "description": resource_description,
                    "mimeType": resource_mime_type,
                    "function": wrapper
                })
                
            except Exception as e:
                logger.error(f"Failed to create wrapper for resource {resource}: {e}")
                continue
        
        return resource_wrappers

    async def _create_prompt_wrappers(self, session_data: dict, server_name: str, prompts: List) -> List:
        """Create wrapper functions for all prompts."""
        prompt_wrappers = []
        
        for prompt in prompts:
            try:
                prompt_name = prompt.get('name') if isinstance(prompt, dict) else getattr(prompt, 'name', 'unknown')
                prompt_description = prompt.get('description') if isinstance(prompt, dict) else getattr(prompt, 'description', '')
                prompt_args = prompt.get('arguments') if isinstance(prompt, dict) else getattr(prompt, 'arguments', [])
                
                wrapper = self._wrap_prompt(session_data, server_name, prompt_name, prompt_description, prompt_args)
                
                prompt_wrappers.append({
                    "name": prompt_name,
                    "description": prompt_description,
                    "arguments": prompt_args,
                    "function": wrapper
                })
                
            except Exception as e:
                logger.error(f"Failed to create wrapper for prompt {prompt}: {e}")
                continue
        
        return prompt_wrappers

    async def _register_unified_mcp_service(self, session_data: dict, server_name: str, tools: List, resources: List, prompts: List):
        """Register a unified service that exposes all MCP server capabilities."""
        # Create wrappers for all capabilities
        tool_wrappers = await self._create_tool_wrappers(session_data, server_name, tools)
        resource_wrappers = await self._create_resource_wrappers(session_data, server_name, resources)  
        prompt_wrappers = await self._create_prompt_wrappers(session_data, server_name, prompts)
        
        # Create the service definition
        service_def = {
            "name": f"MCP Server: {server_name}",
            "description": f"Unified service exposing tools, resources, and prompts from MCP server {server_name}",
            "type": "mcp-server-proxy",
            "config": {"visibility": "public"},
        }
        
        # Add tool functions
        for tool_wrapper in tool_wrappers:
            service_def[f"tool_{tool_wrapper['name']}"] = tool_wrapper["function"]
            
        # Add resource functions  
        for resource_wrapper in resource_wrappers:
            service_def[f"resource_{resource_wrapper['name']}"] = resource_wrapper["function"]
            
        # Add prompt functions
        for prompt_wrapper in prompt_wrappers:
            service_def[f"prompt_{prompt_wrapper['name']}"] = prompt_wrapper["function"]
        
        # Register the service
        if self.server:
            await self.server.register_service(service_def)
            session_data["services"].append(service_def["name"])
            logger.info(f"Registered unified MCP service: {service_def['name']}")

    async def stop(self, session_id: str) -> None:
        """Stop an MCP session."""
        if session_id not in self._sessions:
            logger.warning(f"MCP session {session_id} not found for stopping, may have already been cleaned up")
            return
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        try:
            session_data = self._session_data.get(session_id)
            if session_data:
                # Close all MCP connections
                connections = session_data.get("connections", {})
                for server_name, connection_info in connections.items():
                    try:
                        server_type = connection_info.get("type")
                        if server_type == "streamable-http":
                            # Close HTTP transport
                            transport = connection_info.get("transport")
                            if transport:
                                await transport.__aexit__(None, None, None)
                        elif "session" in connection_info:
                            # Close MCP session
                            mcp_session = connection_info["session"]
                            if hasattr(mcp_session, 'close'):
                                await mcp_session.close()
                        
                        logger.info(f"Closed connection to MCP server: {server_name}")
                    except Exception as e:
                        logger.warning(f"Error closing connection to {server_name}: {e}")
            
            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped MCP session {session_id}")
            
        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop MCP session {session_id}: {e}")
            raise
        finally:
            # Cleanup
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)

    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all MCP sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]

    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about an MCP session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"MCP session {session_id} not found")
        return self._sessions[session_id]

    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for an MCP session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"MCP session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if type is None else []

        logs = session_data.get("logs", {})
        
        if type:
            target_logs = logs.get(type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type_key] = log_entries[offset:end_idx]
            return result

    async def prepare_workspace(self, workspace: str) -> None:
        """Prepare workspace for MCP operations."""
        logger.info(f"Preparing workspace {workspace} for MCP proxy worker")
        pass

    async def close_workspace(self, workspace: str) -> None:
        """Close all MCP sessions for a workspace."""
        logger.info(f"Closing workspace {workspace} for MCP proxy worker")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop MCP session {session_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the MCP proxy worker."""
        logger.info("Shutting down MCP proxy worker...")
        
        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop MCP session {session_id}: {e}")
        
        logger.info("MCP proxy worker shutdown complete")

    def get_service(self) -> dict:
        """Get the service configuration."""
        return self.get_service_config()


async def hypha_startup(server):
    """Hypha startup function to initialize MCP client."""
    if MCP_AVAILABLE:
        worker = MCPClientRunner(server)
        await server.register_service(worker.get_service_config())
        logger.info("MCP client worker initialized and registered")
    else:
        logger.warning("MCP library not available, skipping MCP client worker")


async def start_worker(server_url, workspace, token):
    """Start MCP worker standalone."""
    from hypha_rpc import connect
    
    if not MCP_AVAILABLE:
        logger.error("MCP library not available")
        return
    
    server = await connect(server_url, workspace=workspace, token=token)
    worker = MCPClientRunner(server.rpc)
    logger.info(f"MCP worker started, server: {server_url}, workspace: {workspace}")
    
    return worker