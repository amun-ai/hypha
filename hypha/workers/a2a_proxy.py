"""Provide an A2A client worker."""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Union

import httpx
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function
from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("a2a_client")
logger.setLevel(LOGLEVEL)

MAXIMUM_LOG_ENTRIES = 2048

# Try to import A2A SDK
try:
    from a2a.client.client import A2AClient, A2ACardResolver
    from a2a.types import SendMessageRequest, MessageSendParams, Message, TextPart, Role
    A2A_SDK_AVAILABLE = True
except ImportError:
    logger.warning("A2A SDK not available. Install with: pip install a2a")
    A2A_SDK_AVAILABLE = False


class A2AClientRunner(BaseWorker):
    """A2A client worker that connects to A2A agents and exposes them as Hypha services."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the A2A client worker."""
        super().__init__(server)
        self.controller_id = str(A2AClientRunner.instance_counter)
        A2AClientRunner.instance_counter += 1

    @property
    def supported_types(self) -> List[str]:
        """Return list of supported application types."""
        return ["a2a-agent"]

    @property
    def worker_name(self) -> str:
        """Return the worker name."""
        return "A2A Proxy Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A2A proxy worker for connecting to A2A agents"

    async def _initialize_worker(self) -> None:
        """Initialize the A2A client worker."""
        pass

    async def initialize(self) -> None:
        """Initialize the A2A client worker."""
        if not self.initialized:
            await self.server.register_service(self.get_service())
            self.initialized = True

    async def _start_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start an A2A client session."""
        if not A2A_SDK_AVAILABLE:
            raise RuntimeError("A2A SDK not available. Install with: pip install a2a")

        # Get the A2A agents configuration from metadata
        a2a_agents = config.metadata.get("a2a_agents", {})
        if not a2a_agents:
            raise ValueError("No A2A agents configuration found in metadata")

        # Create user API connection for service registration in user workspace
        user_api = await connect_to_server({
            "server_url": config.server_url,
            "client_id": config.client_id,
            "workspace": config.workspace,
            "token": config.token,
            "method_timeout": 30,
        })
        
        # Store session data
        session_data = {
            "a2a_agents": a2a_agents,
            "logs": {"log": [], "error": []},
            "a2a_clients": {},
            "registered_services": [],
            "_internal": {
                "user_api": user_api
            }
        }

        # Connect to each A2A agent and register unified services
        for agent_name, agent_config in a2a_agents.items():
            await self._connect_and_register_a2a_service(session_data, agent_name, agent_config)

        # Log successful startup
        session_data["logs"]["log"].append(f"A2A client session started with {len(a2a_agents)} agents")
        
        return session_data

    async def _connect_and_register_a2a_service(self, session_data: dict, agent_name: str, agent_config: dict):
        """Connect to A2A agent and register a unified service."""
        try:
            agent_url = agent_config.get("url")
            headers = agent_config.get("headers", {})
            
            if not agent_url:
                raise ValueError(f"Missing URL for A2A agent: {agent_name}")

            logger.info(f"Connecting to A2A agent {agent_name} at {agent_url}")
            
            # Create HTTP client for this agent
            client = httpx.AsyncClient(headers=headers, timeout=30.0)
            
            # Store agent info with persistent client
            session_data["a2a_clients"][agent_name] = {
                "config": agent_config,
                "url": agent_url,
                "headers": headers,
                "client": client
            }
            
            # Resolve agent card and create A2A client
            agent_card_url = f"{agent_url}/.well-known/agent.json"
            resolver = A2ACardResolver(
                httpx_client=client,
                base_url=agent_url,
                agent_card_path="/.well-known/agent.json"
            )
            
            agent_card = await resolver.get_agent_card()
            if not agent_card:
                raise ValueError(f"Failed to resolve agent card for {agent_name}")
            
            # Store the agent card for later use
            session_data["a2a_clients"][agent_name]["agent_card"] = agent_card
            
            # Create skills as callable functions
            skills = await self._create_skill_wrappers(session_data, agent_name, agent_card.skills)
            
            # Register the unified A2A service
            await self._register_unified_a2a_service(session_data, agent_name, agent_card, skills)
            
            logger.info(f"Successfully connected to A2A agent {agent_name}")
            session_data["logs"]["log"].append(f"Connected to A2A agent {agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to A2A agent {agent_name}: {e}", exc_info=True)
            session_data["logs"]["error"].append(f"Failed to connect to {agent_name}: {e}")



    def _wrap_skill(self, session_data: dict, agent_name: str, skill_name: str, skill_description: str):
        """Create a skill wrapper function with proper closure."""
        async def skill_wrapper(text: str = "", **kwargs):
            """Wrapper function to call A2A skill."""
            logger.debug(f"Calling A2A skill {agent_name}.{skill_name} with text: {text}")
            try:
                # Get the persistent client and agent card from session data
                agent_info = session_data["a2a_clients"][agent_name]
                client = agent_info["client"]
                agent_card = agent_info["agent_card"]
                
                # Create A2A client
                a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)
                
                # Create message request
                message = Message(
                    messageId=str(uuid.uuid4()),
                    role=Role.user,
                    parts=[TextPart(kind="text", text=text)]
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
                                    elif hasattr(part, "root") and hasattr(part.root, "text"):
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
                logger.error(f"Error calling A2A skill {agent_name}.{skill_name}: {e}", exc_info=True)
                raise

        # Set function name and schema
        skill_wrapper.__name__ = skill_name
        skill_wrapper.__schema__ = {
            "name": skill_name,
            "description": skill_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text message to send to the A2A agent"
                    }
                },
                "required": ["text"]
            }
        }
        
        return skill_wrapper

    async def _create_skill_wrappers(self, session_data: dict, agent_name: str, skills: List) -> List:
        """Create skill wrapper functions from A2A agent skills."""
        skill_wrappers = []
        
        for skill in skills:
            try:
                skill_name = str(skill.name)
                skill_description = str(skill.description)
                
                # Create wrapper function using dedicated method
                skill_wrapper = self._wrap_skill(session_data, agent_name, skill_name, skill_description)
                skill_wrappers.append(skill_wrapper)
                
                logger.info(f"Created skill wrapper: {agent_name}.{skill_name}")
                session_data["logs"]["log"].append(f"Created skill wrapper: {agent_name}.{skill_name}")

            except Exception as e:
                logger.error(f"Failed to create skill wrapper for {skill.name}: {e}", exc_info=True)
                session_data["logs"]["error"].append(f"Failed to create skill wrapper for {skill.name}: {e}")
        
        return skill_wrappers

    async def _register_unified_a2a_service(self, session_data: dict, agent_name: str, agent_card, skills: List):
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
            service_info = {
                "id": agent_name,  # Use agent name as service ID
                "name": f"A2A {agent_name.title()} Agent",
                "description": f"A2A agent for {agent_name}: {agent_card.description}",
                "type": "a2a",
                "config": {
                    "visibility": "public",
                    "run_in_executor": True,
                },
                "skills": skills,
                "run": a2a_run,  # Add the run function
                "agent_card": {
                    "name": str(agent_card.name),
                    "description": str(agent_card.description),
                    "version": str(agent_card.version),
                    "capabilities": {
                        "streaming": bool(agent_card.capabilities.streaming),
                        "pushNotifications": bool(agent_card.capabilities.pushNotifications)
                    } if agent_card.capabilities else {},
                    "defaultInputModes": [str(mode) for mode in agent_card.defaultInputModes] if agent_card.defaultInputModes else [],
                    "defaultOutputModes": [str(mode) for mode in agent_card.defaultOutputModes] if agent_card.defaultOutputModes else [],
                    "skills": [
                        {
                            "id": str(skill.id),
                            "name": str(skill.name),
                            "description": str(skill.description),
                            "tags": [str(tag) for tag in skill.tags] if skill.tags else [],
                            "examples": [str(example) for example in skill.examples] if skill.examples else []
                        }
                        for skill in agent_card.skills
                    ] if agent_card.skills else []
                }
            }

            # Register the service
            registered_service = await session_data["_internal"]["user_api"].register_service(service_info)
            session_data["registered_services"].append(registered_service)
            
            logger.info(f"Registered unified A2A service: {agent_name}")
            logger.info(f"  - Skills: {len(skills)}")
            
            session_data["logs"]["log"].append(f"Registered unified A2A service: {agent_name}")
            session_data["logs"]["log"].append(f"  - Skills: {len(skills)}")
            
            await session_data["_internal"]["user_api"].export({
                "setup": lambda: logger.info(f"A2A agent {agent_name} setup complete")
            })

        except Exception as e:
            logger.error(f"Failed to register unified A2A service for {agent_name}: {e}", exc_info=True)
            session_data["logs"]["error"].append(f"Failed to register unified A2A service for {agent_name}: {e}")

    async def _stop_session(self, session_id: str) -> None:
        """Stop an A2A client session."""
        session_data = self._session_data.get(session_id)
        if not session_data:
            return
            
        # Close all HTTP clients
        for agent_name, agent_info in session_data.get("a2a_clients", {}).items():
            try:
                if "client" in agent_info:
                    await agent_info["client"].aclose()
                    logger.info(f"Closed HTTP client for agent: {agent_name}")
            except Exception as e:
                logger.warning(f"Failed to close HTTP client for agent {agent_name}: {e}")
        
        # Unregister services using the user API
        user_api = session_data.get("_internal", {}).get("user_api")
        for service in session_data.get("registered_services", []):
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
        
        logger.info(f"Stopped A2A client session: {session_id}")

    async def _get_session_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for an A2A client session."""
        session_data = self._session_data.get(session_id)
        if not session_data:
            return {} if log_type is None else []

        logs = session_data.get("logs", {"log": [], "error": []})
        
        if log_type:
            # Return specific log type
            target_logs = logs.get(log_type, [])
            end_idx = len(target_logs) if limit is None else min(offset + limit, len(target_logs))
            return target_logs[offset:end_idx]
        else:
            # Return all logs
            result = {}
            for log_type_key, log_entries in logs.items():
                end_idx = len(log_entries) if limit is None else min(offset + limit, len(log_entries))
                result[log_type_key] = log_entries[offset:end_idx]
            return result

    # list_sessions method is now inherited from BaseWorker

    def get_service(self) -> dict:
        """Get the service definition for the A2A proxy worker."""
        return self.get_service_config()

    async def _prepare_workspace(self, workspace_id: str) -> None:
        """Prepare workspace for A2A client operations."""
        logger.info(f"Preparing workspace {workspace_id} for A2A client operations")

    async def _close_workspace(self, workspace_id: str) -> None:
        """Close workspace and cleanup A2A client sessions."""
        logger.info(f"Closing workspace {workspace_id} and cleaning up A2A client sessions")
        # This is now handled by the base class

async def hypha_startup(server):
    """Initialize the A2A client worker as a startup function."""
    a2a_client_runner = A2AClientRunner(server)
    await a2a_client_runner.initialize()
    logger.info("A2A client worker registered as startup function") 


async def start_worker(server_url, workspace, token):
    """Start the A2A client worker."""
    from hypha_rpc import connect_to_server
    async with connect_to_server(server_url) as server:
        a2a_client_runner = A2AClientRunner(server)
        await a2a_client_runner.initialize()
        logger.info("A2A client worker registered as startup function")
        await server.serve()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(start_worker(args.server_url, args.workspace, args.token))