"""Test the LLM Proxy conda application."""

import pytest
import json
from hypha.apps.llm_proxy import (
    create_llm_proxy_app,
    LLM_PROXY_MANIFEST,
    LLM_PROXY_SCRIPT,
)


@pytest.mark.asyncio
async def test_create_llm_proxy_app():
    """Test creating the LLM proxy application."""
    app_info = await create_llm_proxy_app()
    
    # Check manifest
    assert app_info["manifest"]["name"] == "LLM Proxy"
    assert app_info["manifest"]["id"] == "llm-proxy"
    assert app_info["manifest"]["type"] == "conda-jupyter-kernel"
    assert "python=3.11" in app_info["manifest"]["dependencies"]
    
    # Check that litellm is in pip dependencies
    pip_deps = None
    for dep in app_info["manifest"]["dependencies"]:
        if isinstance(dep, dict) and "pip" in dep:
            pip_deps = dep["pip"]
            break
    
    assert pip_deps is not None
    assert any("litellm" in dep for dep in pip_deps)
    
    # Check files
    assert "main.py" in app_info["files"]
    assert "class LLMProxyService" in app_info["files"]["main.py"]
    assert "start_proxy" in app_info["files"]["main.py"]
    assert "chat_completion" in app_info["files"]["main.py"]


def test_llm_proxy_manifest():
    """Test that the LLM proxy manifest is properly configured."""
    assert LLM_PROXY_MANIFEST["name"] == "LLM Proxy"
    assert LLM_PROXY_MANIFEST["type"] == "conda-jupyter-kernel"
    assert LLM_PROXY_MANIFEST["version"] == "0.1.0"
    
    # Check dependencies
    deps = LLM_PROXY_MANIFEST["dependencies"]
    assert "python=3.11" in deps
    assert "pip" in deps
    
    # Find pip dependencies
    pip_deps = None
    for dep in deps:
        if isinstance(dep, dict) and "pip" in dep:
            pip_deps = dep["pip"]
            break
    
    assert pip_deps is not None
    assert any("litellm[proxy]" in dep for dep in pip_deps)
    assert any("hypha-rpc" in dep for dep in pip_deps)
    
    # Check config
    assert LLM_PROXY_MANIFEST["config"]["visibility"] == "public"
    assert LLM_PROXY_MANIFEST["config"]["require_context"] is False


def test_llm_proxy_script():
    """Test that the LLM proxy script contains necessary components."""
    # Check imports
    assert "import asyncio" in LLM_PROXY_SCRIPT
    assert "from hypha_rpc import connect_to_server" in LLM_PROXY_SCRIPT
    
    # Check class definition
    assert "class LLMProxyService:" in LLM_PROXY_SCRIPT
    
    # Check service methods
    assert "async def start_proxy" in LLM_PROXY_SCRIPT
    assert "async def stop_proxy" in LLM_PROXY_SCRIPT
    assert "async def get_status" in LLM_PROXY_SCRIPT
    assert "async def chat_completion" in LLM_PROXY_SCRIPT
    
    # Check main function
    assert "async def main():" in LLM_PROXY_SCRIPT
    assert 'server.register_service' in LLM_PROXY_SCRIPT
    assert '"llm-proxy"' in LLM_PROXY_SCRIPT
    
    # Check that it uses litellm
    assert "litellm" in LLM_PROXY_SCRIPT
    assert "subprocess.Popen" in LLM_PROXY_SCRIPT  # For starting the proxy server