"""MCP (Model Context Protocol) Proxy Worker for connecting to MCP servers and exposing tools as Hypha services."""

import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from hypha_rpc import connect_to_server
from hypha.core import UserInfo
from hypha.workers.base import (
    BaseWorker,
    WorkerConfig,
    SessionStatus,
    SessionInfo,
    SessionNotFoundError,
    WorkerError,
    safe_call_callback,
)

# Compatibility for asyncio.timeout (Python 3.11+)
try:
    from asyncio import timeout as asyncio_timeout
except ImportError:
    # Fallback for Python < 3.11
    import asyncio

    @asynccontextmanager
    async def asyncio_timeout(delay):
        """Fallback timeout implementation for Python < 3.11."""
        task = asyncio.current_task()
        handle = asyncio.get_event_loop().call_later(delay, task.cancel)
        try:
            yield
        except asyncio.CancelledError:
            if not handle.cancelled():
                handle.cancel()
                raise asyncio.TimeoutError() from None
        finally:
            handle.cancel()


LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("mcp_proxy")
logger.setLevel(LOGLEVEL)

# Import MCP SDK - let it fail directly if not available
import mcp.types as mcp_types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


class MCPClientRunner(BaseWorker):
    """MCP client worker that connects to MCP servers and exposes tools as Hypha services."""

    instance_counter: int = 0

    def __init__(self):
        """Initialize the MCP client worker."""
        super().__init__()
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
    def name(self) -> str:
        """Return the worker name."""
        return "MCP Proxy Worker"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return "MCP proxy worker for connecting to MCP servers"

    @property
    def require_context(self) -> bool:
        """Return whether the worker requires a context."""
        return True

    @property
    def use_local_url(self) -> bool:
        """Return whether the worker should use local URLs."""
        return True  # Built-in worker runs in same cluster/host

    async def compile(
        self,
        manifest: dict,
        files: list,
        config: dict = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[dict, list]:
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
                    source_json = (
                        json.loads(source_content) if source_content.strip() else {}
                    )

                    # Merge servers from source with manifest
                    if "mcpServers" in source_json:
                        source_servers = source_json["mcpServers"]
                        # Manifest takes precedence, but use source as fallback
                        merged_servers = {**source_servers, **mcp_servers}
                        mcp_servers = merged_servers

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse source as JSON: {e}")

            # Create final configuration
            final_manifest = {
                "type": "mcp-server",
                "mcpServers": mcp_servers,
                "name": manifest.get("name", "MCP Server"),
                "description": manifest.get("description", "MCP server application"),
                "version": manifest.get("version", "1.0.0"),
            }

            # Merge any additional manifest fields
            for key, value in manifest.items():
                if key not in final_manifest:
                    final_manifest[key] = value

            # Generate source file with final configuration
            source_content = json.dumps(final_manifest, indent=2)

            # Create new files list, replacing or adding source
            new_files = [f for f in files if f.get("name") != "source"]
            new_files.append(
                {"path": "source", "content": source_content, "format": "text"}
            )
            final_manifest["wait_for_service"] = "default"
            return final_manifest, new_files

        # Not an MCP server type, return unchanged
        return manifest, files

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new MCP client session."""
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
            metadata=config.manifest,
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
        # Call progress callback if provided
        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Initializing MCP proxy worker..."}
        )

        # Extract MCP servers configuration from manifest
        manifest = config.manifest
        mcp_servers = manifest.get("mcpServers", {})

        if not mcp_servers:
            raise Exception("No MCP servers configured in manifest")

        session_data = {
            "mcp_servers": mcp_servers,
            "logs": {"info": [], "error": [], "debug": []},
            "mcp_clients": {},
            "services": [],
        }

        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Connecting to Hypha server..."}
        )

        client = await connect_to_server(
            {
                "server_url": config.server_url,
                "client_id": config.client_id,
                "token": config.token,
                "workspace": config.workspace,
            }
        )

        await safe_call_callback(config.progress_callback,
            {
                "type": "info",
                "message": f"Connecting to {len(mcp_servers)} MCP server(s)...",
            }
        )

        # Connect to each MCP server
        for server_name, server_config in mcp_servers.items():
            try:
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Connecting to MCP server: {server_name}...",
                    }
                )
                logger.info(f"Connecting to MCP server: {server_name}")

                await self._connect_and_register_mcp_service(
                    session_data, server_name, server_config, client, config
                )

                session_data["logs"]["info"].append(
                    f"Successfully connected to MCP server: {server_name}"
                )
                logger.info(f"Successfully connected to MCP server: {server_name}")
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Successfully connected to MCP server: {server_name}",
                    }
                )

            except Exception as e:
                error_msg = f"Failed to connect to MCP server {server_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                # For single server configs, fail fast instead of continuing
                if len(mcp_servers) == 1:
                    raise Exception(error_msg)
                # Continue with other servers even if one fails
                continue

        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Registering default service..."}
        )

        # register the default service
        await client.register_service(
            {
                "id": "default",
                "name": "default",
                "description": "Default service",
                "setup": lambda: None,
                "app_id": config.app_id,  # Add app_id to default service too
            }
        )

        if not session_data["mcp_clients"]:
            # Provide more specific error message
            all_errors = session_data["logs"]["error"]
            if all_errors:
                last_error = all_errors[-1]
                error_msg = (
                    f"Failed to connect to any MCP servers. Last error: {last_error}"
                )
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                raise Exception(error_msg)
            else:
                error_msg = "Failed to connect to any MCP servers"
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                raise Exception(error_msg)

        await safe_call_callback(config.progress_callback,
            {
                "type": "success",
                "message": f"MCP proxy initialized successfully with {len(session_data['mcp_clients'])} server(s)",
            }
        )

        session_data["client"] = client
        return session_data

    async def _connect_and_register_mcp_service(
        self,
        session_data: dict,
        server_name: str,
        server_config: dict,
        client,
        config: WorkerConfig = None,
    ):
        """Connect to MCP server and register a single unified service."""
        try:
            server_type = server_config.get("type", "streamable-http")
            server_url = server_config.get("url")

            if not server_url:
                raise ValueError(f"Missing URL for MCP server: {server_name}")

            logger.info(
                f"Connecting to MCP server {server_name} ({server_type}) at {server_url}"
            )

            # Support both streamable-http and sse transport types
            if server_type == "streamable-http":
                await self._connect_streamable_http(
                    session_data, server_name, server_config, client, config
                )
            elif server_type == "sse":
                await self._connect_sse(
                    session_data, server_name, server_config, client, config
                )
            else:
                raise ValueError(f"Unsupported MCP server type: {server_type}")

        except Exception as e:
            logger.error(
                f"Failed to connect to MCP server {server_name}: {e}", exc_info=True
            )
            session_data["logs"]["error"].append(
                f"Failed to connect to {server_name}: {e}"
            )
            raise

    async def _connect_streamable_http(
        self,
        session_data: dict,
        server_name: str,
        server_config: dict,
        client,
        config: WorkerConfig = None,
    ):
        """Connect to MCP server using streamable HTTP transport."""
        server_url = server_config.get("url")
        headers = server_config.get("headers", {})

        logger.debug(
            f"Connecting to streamable HTTP server {server_name} at {server_url}"
        )

        # Note: Custom headers are not currently supported by streamablehttp_client
        if headers:
            logger.warning(
                f"Custom headers specified for {server_name} but not supported by streamablehttp_client: {headers}"
            )

        # Store session info for later use
        session_data["mcp_clients"][server_name] = {
            "config": server_config,
            "session_id": f"streamable-http-{uuid.uuid4().hex[:8]}",
            "url": server_url,
            "transport": "streamable-http",
        }

        # Collect and register server capabilities
        await self._collect_and_register_capabilities(
            session_data, server_name, client, config
        )

    async def _connect_sse(
        self,
        session_data: dict,
        server_name: str,
        server_config: dict,
        client,
        config: WorkerConfig = None,
    ):
        """Connect to MCP server using SSE transport."""
        server_url = server_config.get("url")
        headers = server_config.get("headers", {})

        logger.debug(f"Connecting to SSE server {server_name} at {server_url}")
        if headers:
            logger.debug(f"Using custom headers for {server_name}: {list(headers.keys())}")

        # Store session info for later use
        session_data["mcp_clients"][server_name] = {
            "config": server_config,
            "session_id": f"sse-{uuid.uuid4().hex[:8]}",
            "url": server_url,
            "transport": "sse",
            "headers": headers,  # Store headers for later use
        }

        # Collect and register server capabilities
        await self._collect_and_register_capabilities(
            session_data, server_name, client, config
        )

    async def _collect_and_register_capabilities(
        self, session_data: dict, server_name: str, client, config: WorkerConfig = None
    ):
        """Collect tools, resources, and prompts from MCP session and register unified service."""
        server_config = session_data["mcp_clients"][server_name]["config"]
        server_url = server_config.get("url")
        transport = session_data["mcp_clients"][server_name]["transport"]

        if config:
            await safe_call_callback(config.progress_callback,
                {
                    "type": "info",
                    "message": f"Discovering capabilities from {server_name}...",
                }
            )

        tools = {}
        resources = {}
        prompts = {}

        try:
            # Get the appropriate client context
            if transport == "streamable-http":
                client_context = streamablehttp_client(server_url)
            elif transport == "sse":
                client_context = sse_client(url=server_url)
            else:
                raise ValueError(f"Unsupported transport type: {transport}")

            # Connect and collect capabilities
            if transport == "streamable-http":
                async with client_context as (
                    read_stream,
                    write_stream,
                    get_session_id,
                ):
                    async with ClientSession(read_stream, write_stream) as mcp_session:
                        # CRITICAL: Initialize the session
                        logger.debug(f"Initializing MCP session for {server_name}")
                        await mcp_session.initialize()

                        # Collect tools
                        try:
                            tools_result = await mcp_session.list_tools()
                            if tools_result and hasattr(tools_result, "tools"):
                                tools = await self._create_tool_wrappers(
                                    session_data, server_name, tools_result.tools
                                )
                                logger.info(
                                    f"Found {len(tools.keys())} tools from {server_name}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"No tools found in MCP server {server_name}: {e}"
                            )
                            session_data["logs"]["info"].append(
                                f"No tools found in {server_name}: {e}"
                            )

                        # Collect resources
                        try:
                            resources_result = await mcp_session.list_resources()
                            if resources_result and hasattr(
                                resources_result, "resources"
                            ):
                                resources = await self._create_resource_wrappers(
                                    session_data,
                                    server_name,
                                    resources_result.resources,
                                )
                                logger.info(
                                    f"Found {len(resources)} resources from {server_name}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"No resources found in MCP server {server_name}: {e}"
                            )
                            session_data["logs"]["info"].append(
                                f"No resources found in {server_name}: {e}"
                            )

                        # Collect prompts
                        try:
                            prompts_result = await mcp_session.list_prompts()
                            if prompts_result and hasattr(prompts_result, "prompts"):
                                prompts = await self._create_prompt_wrappers(
                                    session_data, server_name, prompts_result.prompts
                                )
                                logger.info(
                                    f"Found {len(prompts)} prompts from {server_name}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"No prompts found in MCP server {server_name}: {e}"
                            )
                            session_data["logs"]["info"].append(
                                f"No prompts found in {server_name}: {e}"
                            )

            elif transport == "sse":
                # Add timeout to prevent hanging on connection issues
                try:
                    async with asyncio_timeout(10):  # 10 second timeout
                        async with client_context as streams:
                            async with ClientSession(*streams) as mcp_session:
                                # CRITICAL: Initialize the session
                                logger.debug(
                                    f"Initializing MCP SSE session for {server_name}"
                                )
                                await mcp_session.initialize()

                                # Collect tools
                                try:
                                    tools_result = await mcp_session.list_tools()
                                    if tools_result and hasattr(tools_result, "tools"):
                                        tools = await self._create_tool_wrappers(
                                            session_data,
                                            server_name,
                                            tools_result.tools,
                                        )
                                        logger.info(
                                            f"Found {len(tools)} tools from {server_name}"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"No tools found in MCP server {server_name}: {e}"
                                    )
                                    session_data["logs"]["info"].append(
                                        f"No tools found in {server_name}: {e}"
                                    )

                                # Collect resources
                                try:
                                    resources_result = (
                                        await mcp_session.list_resources()
                                    )
                                    if resources_result and hasattr(
                                        resources_result, "resources"
                                    ):
                                        resources = (
                                            await self._create_resource_wrappers(
                                                session_data,
                                                server_name,
                                                resources_result.resources,
                                            )
                                        )
                                        logger.info(
                                            f"Found {len(resources)} resources from {server_name}"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"No resources found in MCP server {server_name}: {e}"
                                    )
                                    session_data["logs"]["info"].append(
                                        f"No resources found in {server_name}: {e}"
                                    )

                                # Collect prompts
                                try:
                                    prompts_result = await mcp_session.list_prompts()
                                    if prompts_result and hasattr(
                                        prompts_result, "prompts"
                                    ):
                                        prompts = await self._create_prompt_wrappers(
                                            session_data,
                                            server_name,
                                            prompts_result.prompts,
                                        )
                                        logger.info(
                                            f"Found {len(prompts)} prompts from {server_name}"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"No prompts found in MCP server {server_name}: {e}"
                                    )
                                    session_data["logs"]["info"].append(
                                        f"No prompts found in {server_name}: {e}"
                                    )

                except asyncio.TimeoutError:
                    logger.error(
                        f"Timeout connecting to SSE server {server_name} at {server_url}"
                    )
                    raise RuntimeError(
                        f"Connection to SSE server {server_name} timed out after 10 seconds"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to connect to SSE server {server_name}: {e}",
                        exc_info=True,
                    )
                    # Check if this is a 501 error or similar HTTP error
                    if "501" in str(e) or "not implemented" in str(e).lower():
                        raise RuntimeError(
                            f"SSE transport not supported by server at {server_url}"
                        )
                    raise

            logger.info(
                f"Server {server_name}: Found {len(tools.keys())} tools, {len(resources.keys())} resources, {len(prompts.keys())} prompts"
            )

            if config:
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Found {len(tools.keys())} tools, {len(resources.keys())} resources, {len(prompts.keys())} prompts from {server_name}",
                    }
                )
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Registering services for {server_name}...",
                    }
                )

            # Register unified service for this server
            await self._register_unified_mcp_service(
                session_data, server_name, tools, resources, prompts, client, config
            )

        except Exception as e:
            logger.error(f"Failed to collect capabilities from {server_name}: {e}")
            raise

    def _create_mcp_client_context(self, session_data: dict, server_name: str):
        """Create appropriate MCP client context based on transport type."""
        server_config = session_data["mcp_clients"][server_name]["config"]
        server_url = server_config.get("url")
        transport = session_data["mcp_clients"][server_name]["transport"]
        headers = session_data["mcp_clients"][server_name].get("headers", {})

        if transport == "streamable-http":
            return streamablehttp_client(server_url)
        elif transport == "sse":
            # Pass headers to SSE client for authentication
            return sse_client(url=server_url, headers=headers if headers else None)
        else:
            raise ValueError(f"Unsupported transport type: {transport}")

    def _wrap_tool(
        self,
        session_data: dict,
        server_name: str,
        tool_name: str,
        tool_description: str,
        tool_schema: dict,
    ):
        """Create a tool wrapper function with proper closure."""

        async def tool_wrapper(*args, **kwargs):
            """Wrapper function to call MCP tool."""
            # Convert positional arguments to keyword arguments based on schema
            if args:
                param_names = []
                if tool_schema and "properties" in tool_schema:
                    param_names = list(tool_schema["properties"].keys())

                # Map positional args to parameter names
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        param_name = param_names[i]
                        if param_name not in kwargs:  # Don't override explicit kwargs
                            kwargs[param_name] = arg
            logger.debug(
                f"Calling MCP tool {server_name}.{tool_name} with args: {kwargs}"
            )
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(
                    session_data, server_name
                )
                transport = session_data["mcp_clients"][server_name]["transport"]

                # Connect and call tool
                if transport == "streamable-http":
                    async with client_context as (
                        read_stream,
                        write_stream,
                        get_session_id,
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(
                                name=tool_name, arguments=kwargs
                            )
                elif transport == "sse":
                    # Add timeout to prevent hanging on SSE connection issues
                    try:
                        async with asyncio_timeout(10):  # 10 second timeout
                            async with client_context as streams:
                                async with ClientSession(*streams) as session:
                                    await session.initialize()
                                    result = await session.call_tool(
                                        name=tool_name, arguments=kwargs
                                    )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout calling SSE tool {server_name}.{tool_name}"
                        )
                        raise RuntimeError(f"SSE tool call timed out after 10 seconds")
                else:
                    raise ValueError(f"Unsupported transport: {transport}")

                # Extract content from MCP result
                if hasattr(result, "content") and result.content:
                    if len(result.content) == 1:
                        text_result = result.content[0].text
                        # Try to convert back to original type based on schema
                        return self._convert_result_type(text_result, tool_schema)
                    else:
                        return [
                            self._convert_result_type(content.text, tool_schema)
                            for content in result.content
                        ]
                else:
                    return str(result)

            except Exception as e:
                logger.error(
                    f"Error calling MCP tool {server_name}.{tool_name}: {e}",
                    exc_info=True,
                )
                raise

        tool_wrapper.__name__ = tool_name
        tool_wrapper.__doc__ = tool_description or ""
        tool_wrapper.__schema__ = {
            "name": tool_name,
            "description": tool_description or "",
            "parameters": tool_schema,
        }
        return tool_wrapper

    def _convert_result_type(self, text_result: str, tool_schema: dict) -> any:
        """Convert string result back to appropriate type based on return type inference."""
        try:
            # First try to parse as JSON to handle structured data
            import json

            try:
                # Try JSON parsing first
                parsed = json.loads(text_result)
                return parsed
            except (json.JSONDecodeError, ValueError):
                pass

            # Try to infer type from the text content
            # Check for boolean values
            if text_result.lower() in ("true", "false"):
                return text_result.lower() == "true"

            # Check for None/null
            if text_result.lower() in ("none", "null"):
                return None

            # Try integer conversion
            try:
                if "." not in text_result and "e" not in text_result.lower():
                    return int(text_result)
            except ValueError:
                pass

            # Try float conversion
            try:
                return float(text_result)
            except ValueError:
                pass

            # Return as string if no conversion is possible
            return text_result

        except Exception:
            # If any conversion fails, return the original text
            return text_result

    def _wrap_resource(
        self,
        session_data: dict,
        server_name: str,
        resource_uri: str,
        resource_name: str,
        resource_description: str,
        resource_mime_type: str,
    ):
        """Create a resource wrapper function with proper closure."""

        async def resource_read():
            """Wrapper function to read MCP resource."""
            logger.debug(f"Reading MCP resource {server_name}.{resource_uri}")
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(
                    session_data, server_name
                )
                transport = session_data["mcp_clients"][server_name]["transport"]

                # Connect and read resource
                if transport == "streamable-http":
                    async with client_context as (
                        read_stream,
                        write_stream,
                        get_session_id,
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.read_resource(uri=resource_uri)
                elif transport == "sse":
                    # Add timeout to prevent hanging on SSE connection issues
                    try:
                        async with asyncio_timeout(10):  # 10 second timeout
                            async with client_context as streams:
                                async with ClientSession(*streams) as session:
                                    await session.initialize()
                                    result = await session.read_resource(
                                        uri=resource_uri
                                    )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout reading SSE resource {server_name}.{resource_uri}"
                        )
                        raise RuntimeError(
                            f"SSE resource read timed out after 10 seconds"
                        )
                else:
                    raise ValueError(f"Unsupported transport: {transport}")

                # Extract content from MCP result
                if hasattr(result, "contents") and result.contents:
                    if len(result.contents) == 1:
                        return result.contents[0].text
                    else:
                        return [content.text for content in result.contents]
                else:
                    return str(result)

            except Exception as e:
                logger.error(
                    f"Error reading MCP resource {server_name}.{resource_uri}: {e}",
                    exc_info=True,
                )
                raise

        return {
            "uri": str(resource_uri),
            "name": str(resource_name),
            "description": str(resource_description),
            "mimeType": str(resource_mime_type),
            "read": resource_read,
        }

    def _wrap_prompt(
        self,
        session_data: dict,
        server_name: str,
        prompt_name: str,
        prompt_description: str,
        prompt_args: list,
    ):
        """Create a prompt wrapper function with proper closure."""

        async def prompt_read(**kwargs):
            """Wrapper function to get MCP prompt."""
            logger.debug(
                f"Reading MCP prompt {server_name}.{prompt_name} with args: {kwargs}"
            )
            try:
                # Get the appropriate client context
                client_context = self._create_mcp_client_context(
                    session_data, server_name
                )
                transport = session_data["mcp_clients"][server_name]["transport"]

                # Connect and get prompt
                if transport == "streamable-http":
                    async with client_context as (
                        read_stream,
                        write_stream,
                        get_session_id,
                    ):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.get_prompt(
                                name=prompt_name, arguments=kwargs
                            )
                elif transport == "sse":
                    # Add timeout to prevent hanging on SSE connection issues
                    try:
                        async with asyncio_timeout(10):  # 10 second timeout
                            async with client_context as streams:
                                async with ClientSession(*streams) as session:
                                    await session.initialize()
                                    result = await session.get_prompt(
                                        name=prompt_name, arguments=kwargs
                                    )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout reading SSE prompt {server_name}.{prompt_name}"
                        )
                        raise RuntimeError(
                            f"SSE prompt read timed out after 10 seconds"
                        )
                else:
                    raise ValueError(f"Unsupported transport: {transport}")

                # Extract content from MCP result
                if hasattr(result, "messages") and result.messages:
                    return {
                        "description": str(result.description),
                        "messages": [
                            {
                                "role": str(msg.role),
                                "content": {
                                    "type": "text",
                                    "text": str(msg.content.text),
                                },
                            }
                            for msg in result.messages
                        ],
                    }
                else:
                    return str(result)

            except Exception as e:
                logger.error(
                    f"Error getting MCP prompt {server_name}.{prompt_name}: {e}",
                    exc_info=True,
                )
                raise

        return {
            "name": str(prompt_name),
            "description": str(prompt_description),
            "read": prompt_read,
        }

    async def _create_tool_wrappers(
        self, session_data: dict, server_name: str, tools: list
    ) -> List:
        """Create tool wrapper functions."""
        tool_wrappers = {}

        for tool in tools:
            try:
                # Convert MCP objects to basic Python types
                tool_name = str(tool.name)
                tool_description = str(tool.description)
                tool_schema = tool.inputSchema

                # Convert tool_schema from Pydantic to dict if needed
                if hasattr(tool_schema, "model_dump"):
                    tool_schema = tool_schema.model_dump()
                elif hasattr(tool_schema, "dict"):
                    tool_schema = tool_schema.dict()

                # Create wrapper function
                tool_wrapper = self._wrap_tool(
                    session_data, server_name, tool_name, tool_description, tool_schema
                )
                tool_wrappers[tool_name] = tool_wrapper

                logger.info(f"Created tool wrapper: {server_name}.{tool_name}")
                session_data["logs"]["info"].append(
                    f"Created tool wrapper: {server_name}.{tool_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create tool wrapper for {tool.name}: {e}", exc_info=True
                )
                session_data["logs"]["error"].append(
                    f"Failed to create tool wrapper for {tool.name}: {e}"
                )

        return tool_wrappers

    async def _create_resource_wrappers(
        self, session_data: dict, server_name: str, resources: list
    ) -> Dict[str, Any]:
        """Create resource wrapper configurations with read functions."""
        resource_configs = {}

        for resource in resources:
            try:
                # Convert MCP objects to basic Python types
                resource_uri = str(resource.uri)
                resource_name = str(resource.name)
                resource_description = str(resource.description)
                resource_mime_type = str(getattr(resource, "mimeType", "text/plain"))

                # Create resource config
                resource_config = self._wrap_resource(
                    session_data,
                    server_name,
                    resource_uri,
                    resource_name,
                    resource_description,
                    resource_mime_type,
                )
                resource_configs[resource_uri] = resource_config

                logger.info(f"Created resource wrapper: {server_name}.{resource_name}")
                session_data["logs"]["info"].append(
                    f"Created resource wrapper: {server_name}.{resource_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create resource wrapper for {resource.name}: {e}",
                    exc_info=True,
                )
                session_data["logs"]["error"].append(
                    f"Failed to create resource wrapper for {resource.name}: {e}"
                )

        return resource_configs

    async def _create_prompt_wrappers(
        self, session_data: dict, server_name: str, prompts: list
    ) -> List:
        """Create prompt wrapper configurations with read functions."""
        prompt_configs = {}

        for prompt in prompts:
            try:
                # Convert MCP objects to basic Python types
                prompt_name = str(prompt.name)
                prompt_description = str(prompt.description)
                prompt_args = getattr(prompt, "arguments", [])

                # Create prompt config
                prompt_config = self._wrap_prompt(
                    session_data,
                    server_name,
                    prompt_name,
                    prompt_description,
                    prompt_args,
                )
                prompt_configs[prompt_name] = prompt_config

                logger.info(f"Created prompt wrapper: {server_name}.{prompt_name}")
                session_data["logs"]["info"].append(
                    f"Created prompt wrapper: {server_name}.{prompt_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create prompt wrapper for {prompt.name}: {e}",
                    exc_info=True,
                )
                session_data["logs"]["error"].append(
                    f"Failed to create prompt wrapper for {prompt.name}: {e}"
                )

        return prompt_configs

    async def _register_unified_mcp_service(
        self,
        session_data: dict,
        server_name: str,
        tools: Dict[str, Any],
        resources: Dict[str, Any],
        prompts: Dict[str, Any],
        client,
        config: WorkerConfig = None,
    ):
        """Register a unified service that exposes all MCP server capabilities."""
        try:
            # Register the service with correct structure
            service_def = {
                "id": server_name,
                "name": f"MCP Server for {server_name}",
                "description": f"Tools, resources, and prompts from MCP server {server_name}",
                "type": "mcp",
                "config": {"visibility": "public"},
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
            }

            # Set app_id from config if available
            if config and config.app_id:
                service_def["app_id"] = config.app_id
                logger.info(
                    f"Setting app_id to {config.app_id} for service {server_name}"
                )
            else:
                logger.warning(
                    f"No config or app_id available for service {server_name} - config: {config}"
                )

            # Register the service
            service_info = await client.register_service(service_def, overwrite=True)
            # Store the actual service ID that was returned by register_service
            session_data["services"].append(service_info.id)
            logger.info(
                f"Registered unified MCP service: {service_info.id} with app_id: {service_def.get('app_id', 'NOT_SET')}"
            )

        except Exception as e:
            error_msg = f"Failed to register MCP service for {server_name}: {e}"
            session_data["logs"]["error"].append(error_msg)
            logger.error(error_msg)
            raise Exception(error_msg)

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop an MCP session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"MCP session {session_id} not found for stopping")

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            session_data = self._session_data.get(session_id)
            if not session_data:
                raise RuntimeError(f"No session data found for session {session_id}")
            
            # Get the client from session data
            client = session_data.get("client")
            if not client:
                raise RuntimeError(f"No client found for session {session_id}")
            
            # Unregister services
            services = session_data.get("services", [])
            for service_id in services:
                await client.unregister_service(service_id)
                logger.info(f"Unregistered service: {service_id}")

            # Disconnect the Hypha client to prevent memory leak
            await client.disconnect()
            logger.info(f"Disconnected Hypha client for session {session_id}")
            
            # Log cleanup
            logger.info(f"Cleaning up MCP connections for session {session_id}")

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

    

    async def get_logs(
        self,
        session_id: str,
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get logs for an MCP session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"MCP session {session_id} not found")

        session_data = self._session_data.get(session_id)
        if not session_data:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}

        logs = session_data.get("logs", {})
        
        # Convert logs to items format
        all_items = []
        for log_type, log_entries in logs.items():
            for entry in log_entries:
                all_items.append({"type": log_type, "content": entry})
        
        # Filter by type if specified
        if type:
            filtered_items = [item for item in all_items if item["type"] == type]
        else:
            filtered_items = all_items
        
        total = len(filtered_items)
        
        # Apply pagination
        if limit is None:
            paginated_items = filtered_items[offset:]
        else:
            paginated_items = filtered_items[offset:offset + limit]
        
        return {
            "items": paginated_items,
            "total": total,
            "offset": offset,
            "limit": limit
        }

    

    async def shutdown(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Shutdown the MCP proxy worker."""
        logger.info("Shutting down MCP proxy worker...")

        # Stop all sessions - any failure should be propagated
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.stop(session_id)

        logger.info("MCP proxy worker shutdown complete")


async def hypha_startup(server):
    """Hypha startup function to initialize MCP client."""
    worker = MCPClientRunner()
    await worker.register_worker_service(server)
    logger.info("MCP client worker initialized and registered")


async def start_worker(server_url, workspace, token):
    """Start MCP worker standalone."""
    from hypha_rpc import connect

    await connect(server_url, workspace=workspace, token=token)
    worker = MCPClientRunner()
    logger.info(f"MCP worker started, server: {server_url}, workspace: {workspace}")

    return worker
