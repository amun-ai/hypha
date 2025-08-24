"""Hypha internal applications module.

This module contains internal applications that can be automatically
installed and configured as part of the Hypha server startup.

Available apps:
- llm_proxy: LiteLLM proxy server running in conda environment
"""

# Import startup functions from internal apps
try:
    from .llm_proxy import hypha_startup as llm_proxy_startup
    __all__ = ["llm_proxy_startup"]
except ImportError:
    # Apps might not be available in all installations
    __all__ = []