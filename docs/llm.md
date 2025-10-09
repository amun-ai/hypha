# LLM Providers via LiteLLM Proxy

Hypha provides built-in support for creating LLM proxy services using [litellm](https://github.com/BerriAI/litellm), enabling you to create unified API endpoints that work with multiple LLM providers like OpenAI, Claude, Gemini, and more.

## Overview

The LLM proxy feature allows you to:
- Create OpenAI-compatible API endpoints for multiple LLM providers
- Configure multiple models with load balancing and routing strategies
- Use the same API interface regardless of the underlying LLM provider
- Deploy scalable LLM services with automatic session management
- Access services via both HTTP REST API and Hypha's WebSocket RPC
- Automatically discover and expose 35+ litellm endpoints including audio, images, fine-tuning, and more

## Server Setup

To enable LLM proxy support in your Hypha server, start it with the `--enable-llm-proxy` flag:

```bash
python -m hypha.server --enable-llm-proxy --enable-s3 --enable-server-apps
```

This automatically registers the LLM proxy worker, making it available for creating LLM proxy applications.

## Quick Start

This guide walks you through installing, starting, and accessing an LLM proxy service.

### Step 1: Install the App

The fastest path is through the Hypha workspace UI—no SDK required.

1. **Create a new Server App**
   - Open your workspace, go to `Apps`, and choose **New App → Server App → Create from source**.
2. **Paste the manifest below** into the source editor. The `<manifest>` block is the only required file for a proxy worker; you can keep the mock responses to test without any provider keys.

```html
<manifest lang="json">
{
  "name": "Workspace LLM Proxy",
  "type": "llm-proxy",
  "version": "1.0.0",
  "description": "LLM proxy for workspace testing",
  "startup_config": {
    "wait_for_service": "test-llm"
  },
  "config": {
    "service_id": "test-llm",
    "model_list": [
      {
        "model_name": "test-gpt-3.5",
        "litellm_params": {
          "model": "gpt-3.5-turbo",
          "mock_response": "Mock response from the workspace proxy"
        }
      },
      {
        "model_name": "test-claude",
        "litellm_params": {
          "model": "anthropic/claude-3-sonnet-20240229",
          "mock_response": "Mock response from the Claude slot"
        }
      }
    ],
    "litellm_settings": {
      "routing_strategy": "simple-shuffle",
      "drop_params": true
    }
  }
}
</manifest>
```

3. **Save and stage** the app. This installs the app and gives it a unique `app_id` (visible in the app list).

### Step 2: Start the App

Click **Start** in the UI to launch the service. The session panel shows the generated `service_id`, `base_url`, `master_key`, and discovered endpoints once the proxy is running.

### Step 3: Access the LLM API

Once started, you can access the LLM proxy via HTTP using OpenAI-compatible endpoints:

**When explicitly started** (with a running session):

```bash
# List available models
curl http://127.0.0.1:9527/ws-user-github%7C478667/apps/test-llm/v1/models

# Chat completions
curl -X POST http://127.0.0.1:9527/ws-user-github%7C478667/apps/test-llm/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-gpt-3.5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**When not explicitly started** (on-demand via app ID):

```bash
# Access via app_id - this will start the app automatically if not running
curl http://127.0.0.1:9527/ws-user-github%7C478667/apps/test-llm@app-abc123/v1/models
```

Replace `ws-user-github%7C478667` with your workspace name and `app-abc123` with your actual app ID from the installation.

**Notes:**

- The URL pattern is: `http://<server>/<workspace>/apps/<service_id>/v1/<endpoint>`
- For on-demand access: `http://<server>/<workspace>/apps/<service_id>@<app_id>/v1/<endpoint>`
- Workspace names with special characters are URL-encoded (e.g., `|` becomes `%7C`)
- Authentication may be required depending on your server configuration

### Programmatic Install (optional)

Prefer automation? You can drive the same flow from the `public/server-apps` service:

```javascript
const controller = await api.get_service("public/server-apps")
const source = `
<manifest lang="json">
{
  "name": "My LLM Proxy",
  "type": "llm-proxy",
  "version": "1.0.0",
  "startup_config": {
    "wait_for_service": "my-llm-service"
  },
  "config": {
    "service_id": "my-llm-service",
    "model_list": [
      {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
          "model": "gpt-3.5-turbo",
          "api_key": "HYPHA_SECRET:OPENAI_API_KEY"
        }
      }
    ],
    "litellm_settings": {
      "routing_strategy": "simple-shuffle"
    }
  }
}
</manifest>
`

// Step 1: Install
const appInfo = await controller.install({ source, overwrite: true, stage: true })
console.log("App installed with ID:", appInfo.id)

// Step 2: Start (the startup_config ensures it waits for the service to be ready)
const session = await controller.start(appInfo.id)
console.log("Service started at:", session.outputs.base_url)

// Step 3: Access
// Use session.outputs.base_url to construct your API endpoints
const apiUrl = `https://your-server/${api.config.workspace}${session.outputs.base_url}/v1/models`
```

### Swap in Real Providers

Once you've verified the setup works with mock responses, replace them with real API keys:

```json
{
  "model_name": "gpt-4",
  "litellm_params": {
    "model": "gpt-4",
    "api_key": "HYPHA_SECRET:OPENAI_API_KEY"
  }
}
```

Use `HYPHA_SECRET:YOUR_ENV_KEY` for API keys so the worker resolves them from workspace secrets at start-up.

## Configuration

### App Manifest Structure

LLM proxy apps use the following manifest structure:

```json
{
  "name": "LLM Proxy Service",
  "type": "llm-proxy", 
  "version": "1.0.0",
  "description": "Description of your LLM proxy",
  "config": {
    "service_id": "custom-llm-id",
    "model_list": [...],
    "litellm_settings": {...}
  }
}
```

### Service ID Configuration

The `service_id` determines how your service is accessed:
- **Custom ID**: `"service_id": "my-llm"` → Access at `/workspace/apps/my-llm/v1/`
- **Auto-generated**: If not specified, uses `llm-{session_id}` pattern

### Model Configuration

Each model in the `model_list` requires:

```json
{
  "model_name": "friendly-name",
  "litellm_params": {
    "model": "provider-specific-model-name",
    "api_key": "your-api-key",
    // Additional provider-specific parameters
  }
}
```

#### Supported Providers

**OpenAI:**
```json
{
  "model_name": "gpt-4",
  "litellm_params": {
    "model": "gpt-4",
    "api_key": "sk-..."
  }
}
```

**Anthropic Claude:**
```json
{
  "model_name": "claude-3-opus",
  "litellm_params": {
    "model": "claude-3-opus-20240229",
    "api_key": "sk-ant-..."
  }
}
```

**Google Gemini:**
```json
{
  "model_name": "gemini-pro",
  "litellm_params": {
    "model": "gemini-pro",
    "api_key": "your-google-api-key"
  }
}
```

**AWS Bedrock:**
```json
{
  "model_name": "bedrock-claude",
  "litellm_params": {
    "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "aws_access_key_id": "your-access-key",
    "aws_secret_access_key": "your-secret-key",
    "aws_region_name": "us-east-1"
  }
}
```

### LiteLLM Settings

Configure routing and behavior with `litellm_settings`:

```json
{
  "litellm_settings": {
    "debug": false,
    "drop_params": true,
    "routing_strategy": "simple-shuffle",
    "num_retries": 3,
    "timeout": 30,
    "max_budget": 100.0
  }
}
```

**Available Options:**
- `routing_strategy`: How to distribute requests across models
  - `"simple-shuffle"`: Random selection (default)
  - `"least-busy"`: Route to least busy model
  - `"usage-based-routing"`: Route based on usage patterns
- `debug`: Enable detailed logging
- `drop_params`: Remove unsupported parameters for each provider
- `num_retries`: Number of retry attempts on failure
- `timeout`: Request timeout in seconds
- `max_budget`: Maximum spend limit

### Session Outputs

When the proxy starts successfully, the returned session object includes:

- `service_id`: The identifier you can wait on (`wait_for_service`) and use in URLs
- `base_url`: Workspace-relative path for all OpenAI-compatible endpoints (e.g. `/apps/my-llm-service`)
- `endpoints`: Map of common endpoint names to fully qualified paths
- `master_key`: A session-scoped token you can use for direct HTTP requests if you don't want to reuse your Hypha user token

The same information is visible in the workspace UI under the session's **Outputs** tab. The `base_url` is relative to your workspace root, so the full REST root becomes `https://your-server/<workspace>${base_url}`. Persist the `service_id` or `base_url` if you need to reconnect later; the worker also keeps a rolling log accessible through `controller.get_logs(sessionId)`.

## Using the LLM Proxy

### HTTP REST API

Use the `base_url` from the session outputs (or the UI) and pass either your Hypha workspace token or the generated `master_key`:

```bash
# Chat completions
curl -X POST "https://your-server/${WORKSPACE}${BASE_URL}/v1/chat/completions" \
  -H "Authorization: Bearer ${MASTER_KEY_OR_HYPHA_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# List models
curl "https://your-server/${WORKSPACE}${BASE_URL}/v1/models" \
  -H "Authorization: Bearer ${MASTER_KEY_OR_HYPHA_TOKEN}"

# Text completions
curl -X POST "https://your-server/${WORKSPACE}${BASE_URL}/v1/completions" \
  -H "Authorization: Bearer ${MASTER_KEY_OR_HYPHA_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

Where:
- `WORKSPACE` is your workspace slug (for example `demo`).
- `BASE_URL` is the `base_url` value returned by the session (for example `/apps/test-llm`).
- `MASTER_KEY_OR_HYPHA_TOKEN` can be the generated `master_key` or any bearer token with access to the workspace.

### Available Endpoints

The LLM proxy automatically discovers and exposes all litellm endpoints. Common endpoints include:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (OpenAI format)
- `POST /v1/messages` - Claude/Anthropic messages format
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/audio/speech` - Text-to-speech generation
- `POST /v1/audio/transcriptions` - Speech-to-text transcription
- `POST /v1/images/generations` - Image generation
- `POST /v1/images/edits` - Image editing
- `POST /v1/fine_tuning/jobs` - Fine-tuning management
- `POST /v1/batches` - Batch processing
- `POST /v1/moderations` - Content moderation
- `GET /v1/vector_stores` - Vector store operations
- `GET /health` - Health check endpoint

And 20+ more endpoints for assistants, threads, MCP integration, and advanced features.

### Using with OpenAI SDK

```python
import openai

# session is the dict returned by controller.start(...); you can also copy the
# values from the workspace UI after the app starts.
workspace_slug = "my-workspace"  # set to your workspace slug or use api.config.workspace
BASE_URL = f"https://your-server/{workspace_slug}{session['outputs']['base_url']}/v1"
API_KEY = session["outputs"]["master_key"] or "your-hypha-token"

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

response = client.chat.completions.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Streaming Responses

The LLM proxy supports streaming for real-time responses:

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Advanced Configuration

### Load Balancing Multiple Instances

Configure multiple instances of the same model for load balancing:

```json
{
  "model_list": [
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "gpt-3.5-turbo",
        "api_key": "key-1"
      }
    },
    {
      "model_name": "gpt-3.5-turbo",
      "litellm_params": {
        "model": "gpt-3.5-turbo", 
        "api_key": "key-2"
      }
    }
  ],
  "litellm_settings": {
    "routing_strategy": "least-busy"
  }
}
```

### Workspace Secrets

Store sensitive API keys as workspace environment variables and reference them securely:

```json
{
  "model_name": "gpt-4",
  "litellm_params": {
    "model": "gpt-4",
    "api_key": "HYPHA_SECRET:OPENAI_API_KEY"
  }
}
```

To use workspace secrets:

1. **Set workspace environment variables:**
```javascript
// Set the secret in your workspace
await api.set_env("OPENAI_API_KEY", "sk-your-actual-api-key")
await api.set_env("CLAUDE_API_KEY", "sk-ant-your-actual-key")
```

2. **Reference in your model configuration:**
```json
{
  "model_list": [
    {
      "model_name": "gpt-4",
      "litellm_params": {
        "model": "gpt-4",
        "api_key": "HYPHA_SECRET:OPENAI_API_KEY"
      }
    },
    {
      "model_name": "claude-3-opus",
      "litellm_params": {
        "model": "anthropic/claude-3-opus",
        "api_key": "HYPHA_SECRET:CLAUDE_API_KEY"
      }
    }
  ]
}
```

The secrets are resolved automatically when the LLM proxy starts. If a referenced secret is not found, the proxy will fail to start with a clear error message.

### System Environment Variables

You can also use system environment variables (set on the server):

```json
{
  "model_name": "gpt-4",
  "litellm_params": {
    "model": "gpt-4",
    "api_key": "os.environ/OPENAI_API_KEY"
  }
}
```

### Custom Model Aliases

Create friendly names for complex model identifiers:

```json
{
  "model_name": "my-custom-claude",
  "litellm_params": {
    "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "aws_access_key_id": "...",
    "aws_secret_access_key": "...",
    "aws_region_name": "us-west-2"
  }
}
```

## Testing and Development

### Mock Responses

For testing, use litellm's mock response feature:

```json
{
  "model_name": "test-model",
  "litellm_params": {
    "model": "gpt-3.5-turbo",
    "mock_response": "This is a test response"
  }
}
```

### Development Mode

Enable debug logging for development:

```json
{
  "litellm_settings": {
    "debug": true,
    "detailed_debug": true
  }
}
```

## Monitoring and Management

### Session Management

Get session logs and status:

```javascript
const controller = await api.get_service("public/server-apps")
const logs = await controller.get_logs(sessionId)
console.log("LLM proxy logs:", logs)
```

### Health Monitoring

Check service health:

```bash
curl "https://your-server/workspace/apps/my-llm-service/health" \
  -H "Authorization: Bearer your-token"
```

### Stopping Services

```javascript
await controller.stop(sessionId)
```

## Security Considerations

1. **API Keys**: Use workspace secrets (`HYPHA_SECRET:`) to store sensitive API keys securely within your workspace
2. **Workspace Isolation**: Services are isolated by workspace by default
3. **Authentication**: All requests require valid Hypha authentication tokens
4. **Rate Limiting**: Configure appropriate rate limits in litellm settings
5. **Secret Management**: Never hardcode API keys in manifests - use workspace environment variables instead

## Troubleshooting

### Common Issues

**Service not found (404)**
- Check that the service_id matches your request URL
- Verify the service is running: `await controller.list_apps()`

**Multiple services error**
- Use unique service_id for each deployment
- Clean up old sessions before creating new ones

**Authentication errors (403)**
- Verify your Hypha token is valid and has workspace access
- Check that the service visibility is appropriate

**Model not available**
- Verify API keys are correct and have sufficient credits
- Check that the model name in requests matches your configuration

### Debug Mode

Enable detailed logging:

```json
{
  "litellm_settings": {
    "debug": true,
    "detailed_debug": true,
    "user_debug": true
  }
}
```

## Migration from Direct Provider APIs

If you're migrating from direct provider APIs:

1. **URL Changes**: Update base URLs to point to your LLM proxy
2. **Authentication**: Replace provider API keys with Hypha tokens
3. **Model Names**: Use the friendly names configured in your model_list
4. **Unified Interface**: Take advantage of consistent API across all providers

## Examples Repository

For more examples and templates, see the [Hypha examples repository](https://github.com/amun-ai/hypha-examples).
