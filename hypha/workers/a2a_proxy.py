"""A2A (Agent to Agent) Proxy Worker for connecting to A2A agents and exposing them as Hypha services."""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
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

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("a2a_proxy")
logger.setLevel(LOGLEVEL)

# Try to import A2A SDK
try:
    from a2a.types import AgentCard
    from a2a.client import A2AClient, A2ACardResolver
    from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart, Role

    A2A_SDK_AVAILABLE = True
except ImportError:
    logger.warning("A2A SDK not available. Install with: pip install a2a")
    A2A_SDK_AVAILABLE = False


class A2AClientRunner(BaseWorker):
    """A2A client worker that connects to A2A agents and exposes them as Hypha services."""

    instance_counter: int = 0

    def __init__(self):
        """Initialize the A2A client worker."""
        super().__init__()
        self.controller_id = str(A2AClientRunner.instance_counter)
        A2AClientRunner.instance_counter += 1

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}
        self._session_data: Dict[str, Dict[str, Any]] = {}

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["a2a-agent"]

    @property
    def name(self) -> str:
        """Return the worker name."""
        return "A2A Proxy Worker"

    @property
    def description(self) -> str:
        """Return the worker description."""
        return "A2A proxy worker for connecting to A2A agents"

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
        """Compile A2A agent manifest and files.

        This method processes A2A agent configuration:
        1. Looks for 'source' file containing JSON configuration
        2. Extracts a2aAgents from the source or manifest
        3. Updates manifest with proper A2A agent settings
        4. Generates source file with final configuration
        """
        # For A2A agents, check if we need to generate source from manifest
        if manifest.get("type") == "a2a-agent":
            # Extract A2A agents configuration
            a2a_agents = manifest.get("a2aAgents", {})

            # Look for source file that might contain additional config
            source_file = None
            for file_info in files:
                if file_info.get("name") == "source":
                    source_file = file_info
                    break

            # If source file exists, try to parse it as JSON to extract a2aAgents
            if source_file:
                try:
                    source_content = source_file.get("content", "{}")
                    source_json = (
                        json.loads(source_content) if source_content.strip() else {}
                    )

                    # Merge agents from source with manifest
                    if "a2aAgents" in source_json:
                        source_agents = source_json["a2aAgents"]
                        # Manifest takes precedence, but use source as fallback
                        merged_agents = {**source_agents, **a2a_agents}
                        a2a_agents = merged_agents

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse source as JSON: {e}")

            assert a2a_agents, "a2aAgents configuration is required"
            # Create final configuration
            final_manifest = {
                "type": "a2a-agent",
                "a2aAgents": a2a_agents,
                "name": manifest.get("name", "A2A Agent"),
                "description": manifest.get("description", "A2A agent application"),
                "version": manifest.get("version", "1.0.0"),
            }

            # Merge any additional manifest fields
            for key, value in manifest.items():
                if key not in final_manifest:
                    final_manifest[key] = value

            # Generate source file with final configuration
            source_content = json.dumps(final_manifest, indent=2)

            # Create new files list, replacing or adding source
            new_files = [f for f in files if f.get("path") != "source"]
            new_files.append(
                {"path": "source", "content": source_content, "format": "text"}
            )
            final_manifest["wait_for_service"] = "default"
            return final_manifest, new_files

        # Not an A2A agent type, return unchanged
        return manifest, files

    async def start(
        self,
        config: Union[WorkerConfig, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new A2A client session."""
        if not A2A_SDK_AVAILABLE:
            raise WorkerError("A2A SDK not available. Install with: pip install a2a")

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
            session_data = await self._start_a2a_session(config)
            self._session_data[session_id] = session_data

            # Update session status
            session_info.status = SessionStatus.RUNNING
            logger.info(f"Started A2A session {session_id}")

            return session_id

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to start A2A session {session_id}: {e}")
            # Clean up failed session
            self._sessions.pop(session_id, None)
            self._session_data.pop(session_id, None)  # Also clean up session data if it exists
            raise

    async def _start_a2a_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start A2A client session and connect to configured agents."""
        if not A2A_SDK_AVAILABLE:
            raise Exception("A2A SDK not available. Install with: pip install a2a")

        # Call progress callback if provided
        await safe_call_callback(config.progress_callback,
            {"type": "info", "message": "Initializing A2A proxy worker..."}
        )

        # Extract A2A agents configuration from manifest
        manifest = config.manifest
        a2a_agents = manifest.get("a2aAgents", {})

        if not a2a_agents:
            raise Exception("No A2A agents configured in manifest")

        session_data = {
            "a2a_agents": a2a_agents,
            "logs": {"info": [], "error": [], "debug": []},
            "a2a_clients": {},
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
                "message": f"Connecting to {len(a2a_agents)} A2A agent(s)...",
            }
        )

        # Connect to each A2A agent
        for agent_name, agent_config in a2a_agents.items():
            try:
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Connecting to A2A agent: {agent_name}...",
                    }
                )
                logger.info(f"Connecting to A2A agent: {agent_name}")

                await self._connect_and_register_a2a_service(
                    session_data, agent_name, agent_config, client, config
                )

                session_data["logs"]["info"].append(
                    f"Successfully connected to A2A agent: {agent_name}"
                )
                logger.info(f"Successfully connected to A2A agent: {agent_name}")
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Successfully connected to A2A agent: {agent_name}",
                    }
                )

            except Exception as e:
                error_msg = f"Failed to connect to A2A agent {agent_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                # For single agent configs, fail fast instead of continuing
                if len(a2a_agents) == 1:
                    raise Exception(error_msg)
                # Continue with other agents even if one fails
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
            }
        )

        if not session_data["a2a_clients"]:
            # Provide more specific error message
            all_errors = session_data["logs"]["error"]
            if all_errors:
                last_error = all_errors[-1]
                error_msg = (
                    f"Failed to connect to any A2A agents. Last error: {last_error}"
                )
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                raise Exception(error_msg)
            else:
                error_msg = "Failed to connect to any A2A agents"
                await safe_call_callback(config.progress_callback, {"type": "error", "message": error_msg})
                raise Exception(error_msg)

        await safe_call_callback(config.progress_callback,
            {
                "type": "success",
                "message": f"A2A proxy initialized successfully with {len(session_data['a2a_clients'])} agent(s)",
            }
        )

        # Store the client in session_data for use in stop method
        session_data["client"] = client
        return session_data

    async def _connect_and_register_a2a_service(
        self,
        session_data: dict,
        agent_name: str,
        agent_config: dict,
        client,
        config: WorkerConfig = None,
    ):
        """Connect to A2A agent and register a unified service."""
        try:
            agent_url = agent_config.get("url")
            headers = agent_config.get("headers", {})

            if not agent_url:
                raise ValueError(f"Missing URL for A2A agent: {agent_name}")

            logger.info(f"Connecting to A2A agent {agent_name} at {agent_url}")

            if config:
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Discovering capabilities from {agent_name}...",
                    }
                )

            # Create HTTP client for this agent
            http_client = httpx.AsyncClient(headers=headers, timeout=30.0)

            # Store agent info with persistent client
            session_data["a2a_clients"][agent_name] = {
                "config": agent_config,
                "url": agent_url,
                "headers": headers,
                "client": http_client,
            }

            # Resolve agent card and create A2A client
            agent_card_url = f"{agent_url}/.well-known/agent-card.json"
            resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url=agent_url,
                agent_card_path="/.well-known/agent-card.json",
            )

            agent_card: AgentCard = await resolver.get_agent_card()
            if not agent_card:
                raise ValueError(f"Failed to resolve agent card for {agent_name}")

            # Store the agent card for later use
            session_data["a2a_clients"][agent_name]["agent_card"] = agent_card

            # Create skills as callable functions
            skills = await self._create_skill_wrappers(
                session_data, agent_name, agent_card.skills if agent_card.skills else []
            )

            logger.info(f"Agent {agent_name}: Found {len(skills)} skills")

            if config:
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Found {len(skills)} skills from {agent_name}",
                    }
                )
                await safe_call_callback(config.progress_callback,
                    {
                        "type": "info",
                        "message": f"Registering services for {agent_name}...",
                    }
                )

            # Register the unified A2A service
            await self._register_unified_a2a_service(
                session_data, agent_name, agent_card, skills, client
            )

        except Exception as e:
            logger.error(
                f"Failed to connect to A2A agent {agent_name}: {e}", exc_info=True
            )
            session_data["logs"]["error"].append(
                f"Failed to connect to {agent_name}: {e}"
            )
            raise

    def _wrap_skill(
        self,
        session_data: dict,
        agent_name: str,
        skill_name: str,
        skill_description: str,
    ):
        """Create a skill wrapper function with proper closure."""

        async def skill_wrapper(text: str = "", **kwargs):
            """Wrapper function to call A2A skill."""
            logger.debug(
                f"Calling A2A skill {agent_name}.{skill_name} with text: {text}"
            )
            try:
                # Get the persistent client and agent card from session data
                agent_info = session_data["a2a_clients"][agent_name]
                http_client = agent_info["client"]
                agent_card = agent_info["agent_card"]

                # Create A2A client
                a2a_client = A2AClient(httpx_client=http_client, agent_card=agent_card)

                # Create message request
                message = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[TextPart(kind="text", text=text)],
                )

                params = MessageSendParams(message=message)
                request = SendMessageRequest(id=str(uuid.uuid4()), params=params)

                # Send message
                response = await a2a_client.send_message(request)

                # Extract response text
                if hasattr(response, "root") and hasattr(response.root, "result"):
                    result = response.root.result

                    if hasattr(result, "kind") and result.kind == "task":
                        # Handle Task response
                        if result.status.state == "completed" and result.artifacts:
                            text_parts = []
                            for artifact in result.artifacts:
                                for part in artifact.parts:
                                    if hasattr(part, "text"):
                                        text_parts.append(part.text)
                                    elif hasattr(part, "root") and hasattr(
                                        part.root, "text"
                                    ):
                                        text_parts.append(part.root.text)
                            return " ".join(text_parts)
                        else:
                            return f"Task {result.id} status: {result.status.state}"
                    elif hasattr(result, "parts"):
                        # Handle Message response
                        text_parts = []
                        for part in result.parts:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
                            elif hasattr(part, "root") and hasattr(part.root, "text"):
                                text_parts.append(part.root.text)
                        return " ".join(text_parts)

                return str(response)

            except Exception as e:
                logger.error(
                    f"Error calling A2A skill {agent_name}.{skill_name}: {e}",
                    exc_info=True,
                )
                raise

        skill_wrapper.__name__ = skill_name
        skill_wrapper.__doc__ = skill_description or ""

        return skill_wrapper

    async def _create_skill_wrappers(
        self, session_data: dict, agent_name: str, skills: List
    ) -> List:
        """Create skill wrapper functions from A2A agent skills."""
        skill_wrappers = []

        for skill in skills:
            try:
                skill_name = str(skill.name)
                skill_description = str(skill.description)

                # Create wrapper function using dedicated method
                skill_wrapper = self._wrap_skill(
                    session_data, agent_name, skill_name, skill_description
                )
                skill_wrappers.append(skill_wrapper)

                logger.info(f"Created skill wrapper: {agent_name}.{skill_name}")
                session_data["logs"]["info"].append(
                    f"Created skill wrapper: {agent_name}.{skill_name}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create skill wrapper for {skill.name}: {e}",
                    exc_info=True,
                )
                session_data["logs"]["error"].append(
                    f"Failed to create skill wrapper for {skill.name}: {e}"
                )

        return skill_wrappers

    async def _register_unified_a2a_service(
        self, session_data: dict, agent_name: str, agent_card: AgentCard, skills: List, client
    ):
        """Register a unified A2A service with skills."""
        try:
            # Create a run function that handles A2A protocol messages
            async def a2a_run(message, context=None):
                """Default A2A run function."""
                # Extract text from message parts
                text_content = ""
                for part in message.get("parts", []):
                    if part.get("kind") == "text":
                        text_content += part.get("text", "")

                # Use the first skill if available, otherwise return a default response
                if skills:
                    return await skills[0](text=text_content)
                else:
                    return f"A2A agent {agent_name} processed: {text_content}"

            # Create service configuration
            service_def = {
                "id": agent_name,  # Use agent name as service ID
                "name": f"A2A Agent: {agent_name}",
                "description": f"A2A agent service for {agent_name}",
                "type": "a2a-agent-proxy",
                "config": {"visibility": "public"},
                "skills": skills,
                "run": a2a_run,  # Add the run function
                "agent_card": {
                    "name": str(agent_card.name),
                    "description": str(agent_card.description),
                    "version": str(agent_card.version),
                    "capabilities": (
                        {
                            "streaming": bool(agent_card.capabilities.streaming),
                                                    "push_notifications": bool(
                            agent_card.capabilities.push_notifications
                            ),
                        }
                        if agent_card.capabilities
                        else {}
                    ),
                                    "default_input_modes": (
                    [str(mode) for mode in agent_card.default_input_modes]
                    if agent_card.default_input_modes
                        else []
                    ),
                    "default_output_modes": (
                        [str(mode) for mode in agent_card.default_output_modes]
                        if agent_card.default_output_modes
                        else []
                    ),
                },
            }

            # Register the service
            service_info = await client.register_service(service_def, overwrite=True)
            # Store the actual service ID that was returned by register_service
            session_data["services"].append(service_info.id)
            logger.info(
                f"Registered unified A2A service: {service_def['name']} with ID: {agent_name}"
            )

        except Exception as e:
            error_msg = f"Failed to register A2A service for {agent_name}: {e}"
            session_data["logs"]["error"].append(error_msg)
            logger.error(error_msg)
            raise Exception(error_msg)

    async def stop(
        self, session_id: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stop an A2A session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"A2A session {session_id} not found for stopping")

        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING

        try:
            session_data = self._session_data.get(session_id)
            if session_data:
                # Get the client from session data
                client = session_data.get("client")
                if client:
                    # Unregister services
                    services = session_data.get("services", [])
                    for service_name in services:
                        try:
                            await client.unregister_service(service_name)
                            logger.info(f"Unregistered service: {service_name}")
                        except Exception as e:
                            logger.warning(f"Failed to unregister service {service_name}: {e}")

                    # Disconnect the Hypha client to prevent memory leak
                    try:
                        await client.disconnect()
                        logger.info(f"Disconnected Hypha client for session {session_id}")
                    except Exception as e:
                        logger.warning(f"Failed to disconnect client for session {session_id}: {e}")

                # Close all HTTP clients
                a2a_clients = session_data.get("a2a_clients", {})
                for agent_name, agent_info in a2a_clients.items():
                    if "client" in agent_info:
                        try:
                            await agent_info["client"].aclose()
                            logger.info(f"Closed HTTP client for agent: {agent_name}")
                        except Exception as e:
                            logger.warning(f"Failed to close HTTP client for {agent_name}: {e}")
            else:
                logger.warning(f"No session data found for session {session_id}, skipping cleanup")

            session_info.status = SessionStatus.STOPPED
            logger.info(f"Stopped A2A session {session_id}")

        except Exception as e:
            session_info.status = SessionStatus.FAILED
            session_info.error = str(e)
            logger.error(f"Failed to stop A2A session {session_id}: {e}")
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
        """Get logs for an A2A session.
        
        Returns a dictionary with:
        - items: List of log events, each with 'type' and 'content' fields
        - total: Total number of log items (before filtering/pagination)
        - offset: The offset used for pagination
        - limit: The limit used for pagination
        """
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"A2A session {session_id} not found")

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
        """Shutdown the A2A proxy worker."""
        logger.info("Shutting down A2A proxy worker...")

        # Stop all sessions - any failure should be propagated
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.stop(session_id)

        logger.info("A2A proxy worker shutdown complete")


async def hypha_startup(server):
    """Hypha startup function to initialize A2A client."""
    if A2A_SDK_AVAILABLE:
        worker = A2AClientRunner()
        await server.register_service(worker.get_worker_service())
        logger.info("A2A client worker initialized and registered")
    else:
        logger.warning("A2A library not available, skipping A2A client worker")


async def start_worker(server_url, workspace, token):
    """Start A2A worker standalone."""
    from hypha_rpc import connect

    if not A2A_SDK_AVAILABLE:
        logger.error("A2A library not available")
        return

    server = await connect(server_url, workspace=workspace, token=token)
    worker = A2AClientRunner(server.rpc)
    logger.info(f"A2A worker started, server: {server_url}, workspace: {workspace}")

    return worker
