"""LLM Proxy Application for Hypha using Conda Worker.

This application runs the LiteLLM proxy server in an isolated conda environment,
allowing it to have its own dependencies without conflicts with the main Hypha server.
"""

import json
from typing import Optional, Dict, Any

# The manifest for the LLM proxy conda application
LLM_PROXY_MANIFEST = {
    "name": "LLM Proxy",
    "id": "llm-proxy",
    "type": "conda-jupyter-kernel",
    "version": "0.1.0",
    "description": "LiteLLM proxy server for unified LLM API access",
    "dependencies": [
        "python=3.11",
        "pip",
        {
            "pip": [
                "litellm[proxy]>=1.52.0",
                "hypha-rpc>=0.20.78",
                "pydantic>=2.5.0",
                "httpx>=0.23.0",
            ]
        }
    ],
    "channels": ["conda-forge"],
    "config": {
        "visibility": "public",
        "require_context": False,
    }
}

# The main script that will run in the conda environment
LLM_PROXY_SCRIPT = '''
import os
import sys
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from hypha_rpc import connect_to_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProxyService:
    """Service that runs LiteLLM proxy in conda environment."""
    
    def __init__(self):
        self.proxy_process = None
        self.proxy_url = None
        
    async def start_proxy(self, config: Optional[Dict[str, Any]] = None):
        """Start the LiteLLM proxy server."""
        import subprocess
        import socket
        
        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        
        # Get configuration from environment or passed config
        model_config = config or {}
        
        # Build command to start litellm proxy
        cmd = [
            sys.executable, "-m", "litellm",
            "--port", str(port),
            "--host", "0.0.0.0"
        ]
        
        # Add model configuration if provided
        if "model" in model_config:
            cmd.extend(["--model", model_config["model"]])
        
        # Add API keys from environment or config
        env = os.environ.copy()
        if "api_keys" in model_config:
            for key, value in model_config["api_keys"].items():
                env[key] = value
        
        logger.info(f"Starting LiteLLM proxy on port {port}")
        self.proxy_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for the server to start
        await asyncio.sleep(3)
        
        self.proxy_url = f"http://localhost:{port}"
        logger.info(f"LiteLLM proxy started at {self.proxy_url}")
        
        return {
            "status": "running",
            "url": self.proxy_url,
            "port": port,
            "pid": self.proxy_process.pid
        }
    
    async def stop_proxy(self):
        """Stop the LiteLLM proxy server."""
        if self.proxy_process:
            self.proxy_process.terminate()
            self.proxy_process.wait(timeout=5)
            self.proxy_process = None
            self.proxy_url = None
            logger.info("LiteLLM proxy stopped")
            return {"status": "stopped"}
        return {"status": "not_running"}
    
    async def get_status(self):
        """Get the status of the proxy server."""
        if self.proxy_process and self.proxy_process.poll() is None:
            return {
                "status": "running",
                "url": self.proxy_url,
                "pid": self.proxy_process.pid
            }
        return {"status": "stopped"}
    
    async def chat_completion(self, messages: list, model: str = None, **kwargs):
        """Proxy a chat completion request to LiteLLM."""
        if not self.proxy_url:
            raise RuntimeError("Proxy server is not running")
        
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.proxy_url}/v1/chat/completions",
                json={
                    "model": model or "gpt-3.5-turbo",
                    "messages": messages,
                    **kwargs
                }
            )
            response.raise_for_status()
            return response.json()

async def main():
    """Main entry point for the LLM proxy service."""
    # Get configuration from environment
    server_url = os.environ.get("HYPHA_SERVER_URL", "ws://localhost:9000/ws")
    workspace = os.environ.get("HYPHA_WORKSPACE", "public")
    token = os.environ.get("HYPHA_TOKEN")
    
    logger.info(f"Connecting to Hypha server at {server_url}")
    
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": server_url,
        "workspace": workspace,
        "token": token,
        "client_id": "llm-proxy-service"
    })
    
    # Create service instance
    service = LLMProxyService()
    
    # Register the service
    await server.register_service({
        "id": "llm-proxy",
        "name": "LLM Proxy Service",
        "description": "LiteLLM proxy for unified LLM API access",
        "config": {
            "visibility": "public",
            "require_context": False
        },
        "start_proxy": service.start_proxy,
        "stop_proxy": service.stop_proxy,
        "get_status": service.get_status,
        "chat_completion": service.chat_completion,
    })
    
    logger.info("LLM Proxy Service registered successfully")
    
    # Auto-start the proxy with default configuration
    try:
        # Get configuration from environment or use defaults
        config = {}
        if os.environ.get("OPENAI_API_KEY"):
            config["api_keys"] = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
        
        await service.start_proxy(config)
    except Exception as e:
        logger.error(f"Failed to auto-start proxy: {e}")
    
    # Keep the service running
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
'''


async def create_llm_proxy_app() -> Dict[str, Any]:
    """Create the LLM proxy application manifest and files.
    
    Returns:
        A dictionary containing the manifest and script files.
    """
    return {
        "manifest": LLM_PROXY_MANIFEST,
        "files": {
            "main.py": LLM_PROXY_SCRIPT
        }
    }


async def install_llm_proxy(
    app_controller, 
    workspace: str = "public", 
    context: Optional[dict] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Install the LLM proxy application in the specified workspace.
    
    Args:
        app_controller: The ServerAppController instance
        workspace: The workspace to install the app in (default: "public")
        context: Optional context dictionary for authorization
        config: Optional configuration to pass to the LLM proxy
    
    Returns:
        The app ID of the installed application
    """
    app_info = await create_llm_proxy_app()
    
    # Create the application source with manifest and files
    app_source = f"""
<config>
{json.dumps(app_info["manifest"], indent=2)}
</config>

<script lang="python">
{app_info["files"]["main.py"]}
</script>
"""
    
    # Install the application
    result = await app_controller.install(
        source=app_source,
        app_id="llm-proxy",
        overwrite=True,
        context=context
    )
    
    return result