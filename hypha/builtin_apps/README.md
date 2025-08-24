# Hypha Internal Applications

This directory contains internal applications that can be automatically installed and configured as part of the Hypha server.

## Structure

Each internal app should be in its own subdirectory with the following structure:

```
builtin_apps/
├── app_name/
│   ├── __init__.py          # Package init with exports
│   ├── app_name_app.py      # Main app definition
│   └── startup.py           # Startup function for auto-installation
└── README.md
```

## Available Apps

### LLM Proxy (`llm_proxy/`)

A conda-based LiteLLM proxy server that provides unified access to various LLM APIs.

**Features:**
- Runs in isolated conda environment
- Supports OpenAI, Anthropic, and other LLM providers
- Auto-configures from environment variables
- Can be enabled with `--enable-llm-proxy` flag

**Environment Variables:**
- `HYPHA_ENABLE_LLM_PROXY=true` - Enable the LLM proxy
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `HYPHA_LLM_PROXY_CONFIG` - JSON string with custom configuration

## Creating a New Internal App

To add a new internal app:

1. Create a new subdirectory under `builtin_apps/`
2. Create the app module with manifest and script
3. Create a `startup.py` with a `hypha_startup` function:

```python
async def hypha_startup(server):
    """Startup function for your app.
    
    Args:
        server: The Hypha server instance
    """
    # Your app installation logic here
```

4. Export the necessary functions in `__init__.py`
5. Optionally add CLI flags in `hypha/server.py` for easy enabling

## Startup Integration

Internal apps can be integrated with Hypha's startup system in several ways:

1. **Via CLI flags**: Add a flag like `--enable-app-name` in `server.py`
2. **Via environment variables**: Check for `HYPHA_ENABLE_APP_NAME` in startup
3. **Via startup modules**: Load as a startup module using `--startup-functions`
4. **Programmatically**: Call the app's install function directly

## Best Practices

1. **Isolation**: Use conda/docker for apps with complex dependencies
2. **Configuration**: Support environment variables for configuration
3. **Logging**: Use proper logging for debugging
4. **Error Handling**: Gracefully handle missing dependencies or services
5. **Documentation**: Document all environment variables and configuration options