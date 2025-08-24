# LLM Proxy with litellm

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

Here's a simple example of creating an LLM proxy service:

```javascript
// Create an LLM proxy app configuration
const llmApp = {
  "name": "My LLM Proxy",
  "type": "llm-proxy",
  "version": "1.0.0",
  "description": "Multi-provider LLM proxy service",
  "config": {
    "service_id": "my-llm-service",
    "model_list": [
      {
        "model_name": "gpt-3.5-turbo",
        "litellm_params": {
          "model": "gpt-3.5-turbo",
          "api_key": "your-openai-api-key"
        }
      },
      {
        "model_name": "claude-3-opus",
        "litellm_params": {
          "model": "claude-3-opus-20240229",
          "api_key": "your-anthropic-api-key"
        }
      }
    ],
    "litellm_settings": {
      "debug": false,
      "routing_strategy": "simple-shuffle"
    }
  }
}

// Install and start the LLM proxy
const controller = await api.get_service("public/server-apps")
const appInfo = await controller.install({source: JSON.stringify(llmApp)})
const session = await controller.start(appInfo.id, {wait_for_service: "my-llm-service"})
```

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

## Using the LLM Proxy

### HTTP REST API

Once deployed, your LLM proxy provides OpenAI-compatible endpoints:

```bash
# Chat completions
curl -X POST "https://your-server/workspace/apps/my-llm-service/v1/chat/completions" \
  -H "Authorization: Bearer your-hypha-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'

# List models
curl "https://your-server/workspace/apps/my-llm-service/v1/models" \
  -H "Authorization: Bearer your-hypha-token"

# Text completions
curl -X POST "https://your-server/workspace/apps/my-llm-service/v1/completions" \
  -H "Authorization: Bearer your-hypha-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

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

# Configure client to use your Hypha LLM proxy
client = openai.OpenAI(
    api_key="your-hypha-token",
    base_url="https://your-server/workspace/apps/my-llm-service/v1"
)

# Use any model configured in your proxy
response = client.chat.completions.create(
    model="claude-3-opus",  # Uses your configured Claude model
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

### Environment Variables

Store sensitive keys in environment variables:

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

1. **API Keys**: Store sensitive API keys in environment variables or secure configuration
2. **Workspace Isolation**: Services are isolated by workspace by default
3. **Authentication**: All requests require valid Hypha authentication tokens
4. **Rate Limiting**: Configure appropriate rate limits in litellm settings

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