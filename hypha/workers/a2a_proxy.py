"""A2A (Agent to Agent) Proxy Worker for connecting to A2A agents and exposing them as Hypha services."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from hypha_rpc import connect_to_server
from hypha.core import UserInfo
from hypha.workers.base import BaseWorker, WorkerConfig, SessionStatus, SessionInfo, SessionNotFoundError, WorkerError

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("a2a_proxy")
logger.setLevel(LOGLEVEL)


# A2A-specific transport implementation
class A2ATransport:
    """A transport for HTTP-based A2A agents."""
    
    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {}
        self.client = httpx.AsyncClient()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the A2A agent."""
        response = await self.client.post(
            self.url,
            json=request_data,
            headers={**self.headers, "Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        response = await self.client.get(
            self.url,
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()


class A2AClientRunner(BaseWorker):
    """A2A client worker that connects to A2A agents and exposes them as Hypha services."""

    instance_counter: int = 0

    def __init__(self, server):
        """Initialize the A2A client runner."""
        super().__init__(server)
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
    def worker_name(self) -> str:
        """Return the worker name."""
        return "A2A Proxy Worker"

    @property
    def worker_description(self) -> str:
        """Return the worker description."""
        return "A2A proxy worker for connecting to A2A agents"

    async def compile(self, manifest: dict, files: list, config: dict = None) -> tuple[dict, list]:
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
                    source_json = json.loads(source_content) if source_content and source_content.strip() else {}
                    
                    # Merge agents from source with manifest
                    if "a2aAgents" in source_json:
                        source_agents = source_json["a2aAgents"]
                        # Manifest takes precedence, but use source as fallback
                        merged_agents = {**source_agents, **a2a_agents}
                        a2a_agents = merged_agents
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse source as JSON: {e}")
            
            # Create final configuration
            final_config = {
                "type": "a2a-agent",
                "a2aAgents": a2a_agents,
                "name": manifest.get("name", "A2A Agent"),
                "description": manifest.get("description", "A2A agent application"),
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
        
        # Not an A2A agent type, return unchanged
        return manifest, files

    async def start(self, config: Union[WorkerConfig, Dict[str, Any]]) -> str:
        """Start a new A2A client session."""
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
            raise

    async def _start_a2a_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start A2A client session and connect to configured agents."""
        # Call progress callback if provided
        config.progress_callback({"type": "info", "message": "Initializing A2A proxy worker..."})

        # Extract A2A agents configuration from manifest
        manifest = config.manifest
        a2a_agents = manifest.get("a2aAgents", {})
        
        if not a2a_agents:
            raise Exception("No A2A agents configured in manifest")

        session_data = {
            "connections": {},
            "logs": {"info": [], "error": [], "debug": []},
            "services": []
        }
        
        config.progress_callback({"type": "info", "message": "Connecting to Hypha server..."})
        
        client = await connect_to_server({
            "server_url": config.server_url,
            "client_id": config.client_id,
            "token": config.token,
            "workspace": config.workspace,
        })

        config.progress_callback({"type": "info", "message": f"Connecting to {len(a2a_agents)} A2A agent(s)..."})

        # Connect to each A2A agent
        for agent_name, agent_config in a2a_agents.items():
            try:
                config.progress_callback({"type": "info", "message": f"Connecting to A2A agent: {agent_name}..."})
                logger.info(f"Connecting to A2A agent: {agent_name}")
                await self._connect_and_register_a2a_service(session_data, agent_name, agent_config, client, config)
                
                session_data["logs"]["info"].append(f"Successfully connected to A2A agent: {agent_name}")
                logger.info(f"Successfully connected to A2A agent: {agent_name}")
                config.progress_callback({"type": "info", "message": f"Successfully connected to A2A agent: {agent_name}"})
                
            except Exception as e:
                error_msg = f"Failed to connect to A2A agent {agent_name}: {e}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                config.progress_callback({"type": "error", "message": error_msg})
                # For single agent configs, fail fast instead of continuing
                if len(a2a_agents) == 1:
                    raise Exception(error_msg)
                # Continue with other agents even if one fails
                continue
        
        config.progress_callback({"type": "info", "message": "Registering default service..."})

        # register the default service
        await client.register_service(
            {
                "id": "default",
                "name": "default",
                "description": "Default service",
                "setup": lambda: None,
            }
        )

        if not session_data["connections"]:
            # Provide more specific error message
            all_errors = session_data["logs"]["error"]
            if all_errors:
                last_error = all_errors[-1]
                error_msg = f"Failed to connect to any A2A agents. Last error: {last_error}"
                config.progress_callback({"type": "error", "message": error_msg})
                raise Exception(error_msg)
            else:
                error_msg = "Failed to connect to any A2A agents"
                config.progress_callback({"type": "error", "message": error_msg})
                raise Exception(error_msg)

        config.progress_callback({"type": "success", "message": f"A2A proxy initialized successfully with {len(session_data['connections'])} agent(s)"})
        return session_data

    async def _connect_and_register_a2a_service(self, session_data: dict, agent_name: str, agent_config: dict, client, config: WorkerConfig = None):
        """Connect to an A2A agent and register its capabilities as Hypha services."""
        url = agent_config.get("url")
        if not url:
            raise Exception(f"No URL specified for A2A agent {agent_name}")
        
        headers = agent_config.get("headers", {})
        
        if config:
            config.progress_callback({"type": "info", "message": f"Creating transport for A2A agent: {agent_name}..."})
        
        # Create transport
        transport = A2ATransport(url, headers)
        
        # Store connection info
        session_data["connections"][agent_name] = {
            "transport": transport,
            "config": agent_config
        }
        
        # Get agent information and skills
        try:
            if config:
                config.progress_callback({"type": "info", "message": f"Discovering agent info and skills from {agent_name}..."})
                
            agent_info = await transport.get_agent_info()
            skills = agent_info.get("skills", [])
            
            logger.info(f"Agent {agent_name}: Found {len(skills)} skills")
            
            if config:
                config.progress_callback({"type": "info", "message": f"Found {len(skills)} skills from {agent_name}"})
                config.progress_callback({"type": "info", "message": f"Registering services for {agent_name}..."})
            
            # Register unified service for this agent
            await self._register_unified_a2a_service(session_data, agent_name, agent_info, skills, client)
            
        except Exception as e:
            logger.error(f"Failed to get agent info from {agent_name}: {e}")
            raise

    def _wrap_skill(self, session_data: dict, agent_name: str, skill_name: str, skill_description: str):
        """Create a wrapper function for an A2A skill."""
        async def skill_wrapper(text: str = "", **kwargs):
            try:
                connection_info = session_data["connections"][agent_name]
                transport = connection_info["transport"]
                
                # Create skill execution request
                request_data = {
                    "skill": skill_name,
                    "text": text,
                    "parameters": kwargs
                }
                
                response = await transport.send_request(request_data)
                
                if "error" in response:
                    raise Exception(f"Skill execution failed: {response['error']}")
                
                return response.get("result", response)
                
            except Exception as e:
                error_msg = f"Error executing skill {skill_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
        
        return skill_wrapper

    async def _create_skill_wrappers(self, session_data: dict, agent_name: str, skills: List) -> List:
        """Create wrapper functions for all skills."""
        skill_wrappers = []
        
        for skill in skills:
            try:
                skill_name = skill.get('name') if isinstance(skill, dict) else getattr(skill, 'name', 'unknown')
                skill_description = skill.get('description') if isinstance(skill, dict) else getattr(skill, 'description', '')
                
                wrapper = self._wrap_skill(session_data, agent_name, skill_name, skill_description)
                
                skill_wrappers.append({
                    "name": skill_name,
                    "description": skill_description,
                    "function": wrapper
                })
                
            except Exception as e:
                logger.error(f"Failed to create wrapper for skill {skill}: {e}")
                continue
        
        return skill_wrappers

    async def _register_unified_a2a_service(self, session_data: dict, agent_name: str, agent_card, skills: List, client):
        """Register a unified service that exposes all A2A agent capabilities."""
        # Create wrappers for all skills
        skill_wrappers = await self._create_skill_wrappers(session_data, agent_name, skills)
        
        # Create the service definition
        async def a2a_run(message, context=None):
            """Run the agent with a message."""
            try:
                connection_info = session_data["connections"][agent_name]
                transport = connection_info["transport"]
                
                # Create agent run request
                request_data = {
                    "message": message,
                    "context": context or {}
                }
                
                response = await transport.send_request(request_data)
                
                if "error" in response:
                    raise Exception(f"Agent run failed: {response['error']}")
                
                return response.get("result", response)
                
            except Exception as e:
                error_msg = f"Error running agent {agent_name}: {str(e)}"
                session_data["logs"]["error"].append(error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
        
        service_def = {
            "id": agent_name,
            "name": f"A2A Agent: {agent_name}",
            "description": f"A2A agent service for {agent_name}",
            "type": "a2a-agent-proxy",
            "config": {"visibility": "public"},
            "run": a2a_run,
            "agent_card": agent_card,
        }
        
        # Add skill functions
        for skill_wrapper in skill_wrappers:
            service_def[f"skill_{skill_wrapper['name']}"] = skill_wrapper["function"]
        
        # Register the service
        try:
            await client.register_service(service_def, overwrite=True)
            session_data["services"].append(service_def["name"])
            logger.info(f"Registered unified A2A service: {service_def['name']} with ID: {agent_name}")
        except Exception as e:
            error_msg = f"Failed to register A2A service for {agent_name}: {e}"
            session_data["logs"]["error"].append(error_msg)
            logger.error(error_msg)
            raise Exception(error_msg)

    async def stop(self, session_id: str) -> None:
        """Stop an A2A session."""
        if session_id not in self._sessions:
            logger.warning(f"A2A session {session_id} not found for stopping, may have already been cleaned up")
            return
        
        session_info = self._sessions[session_id]
        session_info.status = SessionStatus.STOPPING
        
        try:
            session_data = self._session_data.get(session_id)
            if session_data:
                # Close all A2A connections
                connections = session_data.get("connections", {})
                for agent_name, connection_info in connections.items():
                    try:
                        # Close HTTP transport
                        transport = connection_info.get("transport")
                        if transport:
                            await transport.__aexit__(None, None, None)
                        
                        logger.info(f"Closed connection to A2A agent: {agent_name}")
                    except Exception as e:
                        logger.warning(f"Error closing connection to {agent_name}: {e}")
            
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

    async def list_sessions(self, workspace: str) -> List[SessionInfo]:
        """List all A2A sessions for a workspace."""
        return [
            session_info for session_info in self._sessions.values()
            if session_info.workspace == workspace
        ]

    async def get_session_info(self, session_id: str) -> SessionInfo:
        """Get information about an A2A session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"A2A session {session_id} not found")
        return self._sessions[session_id]

    async def get_logs(
        self, 
        session_id: str, 
        type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for an A2A session."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"A2A session {session_id} not found")

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
        """Prepare workspace for A2A operations."""
        logger.info(f"Preparing workspace {workspace} for A2A proxy worker")
        pass

    async def close_workspace(self, workspace: str) -> None:
        """Close all A2A sessions for a workspace."""
        logger.info(f"Closing workspace {workspace} for A2A proxy worker")
        
        # Stop all sessions for this workspace
        sessions_to_stop = [
            session_id for session_id, session_info in self._sessions.items()
            if session_info.workspace == workspace
        ]
        
        for session_id in sessions_to_stop:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop A2A session {session_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the A2A proxy worker."""
        logger.info("Shutting down A2A proxy worker...")
        
        # Stop all sessions
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            try:
                await self.stop(session_id)
            except Exception as e:
                logger.warning(f"Failed to stop A2A session {session_id}: {e}")
        
        logger.info("A2A proxy worker shutdown complete")

    def get_service(self) -> dict:
        """Get the service configuration."""
        return self.get_service_config()


async def hypha_startup(server):
    """Hypha startup function to initialize A2A client."""
    worker = A2AClientRunner(server)
    await server.register_service(worker.get_service_config())
    logger.info("A2A client worker initialized and registered")


async def start_worker(server_url, workspace, token):
    """Start A2A worker standalone."""
    from hypha_rpc import connect
    
    server = await connect(server_url, workspace=workspace, token=token)
    worker = A2AClientRunner(server.rpc)
    logger.info(f"A2A worker started, server: {server_url}, workspace: {workspace}")
    
    return worker

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(start_worker(args.server_url, args.workspace, args.token))