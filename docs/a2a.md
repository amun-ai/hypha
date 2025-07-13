# Agent-to-Agent (A2A) Services

## Introduction

Hypha supports the Agent-to-Agent (A2A) Protocol, an open standard designed to facilitate communication and interoperability between independent AI agent systems. The A2A protocol enables agents to discover each other's capabilities, negotiate interaction modalities, manage collaborative tasks, and securely exchange information.

With Hypha's A2A service support, you can:
- Register AI agents that comply with the A2A protocol specification in both Python and JavaScript
- Automatically expose agents through standardized HTTP endpoints
- Enable agent discovery through Agent Cards
- Support various interaction patterns including streaming and push notifications
- Integrate with existing A2A-compatible clients and tools
- Use the same API across different programming languages with camelCase naming in JavaScript

## Prerequisites

- **Hypha Server**: Ensure you have Hypha installed and configured
- **A2A SDK**: Install the optional A2A SDK dependency: `pip install a2a-sdk`
- **Python 3.9+ or JavaScript**: The A2A implementation requires Python 3.9 or higher for Python services, or any modern JavaScript environment (Node.js or browser) for JavaScript services
- **Hypha-RPC**: Install the hypha-rpc client library for your language:
  - Python: `pip install hypha-rpc`
  - JavaScript: `npm install hypha-rpc` or use CDN: `https://cdn.jsdelivr.net/npm/hypha-rpc@latest/dist/hypha-rpc-websocket.min.js`

## Installation and Setup

### Installing A2A SDK

```bash
pip install a2a-sdk
```

### A2A Support in Hypha

A2A support is now built into Hypha using a dedicated middleware-based approach. No additional server configuration is required - A2A services are automatically detected and handled by the `A2ARoutingMiddleware`.

When you register a service with `type="a2a"`, the middleware:
1. Routes requests to the dedicated A2A URL namespace (`/{workspace}/a2a/{service_id}/`)
2. Detects and validates the A2A service type
3. Creates a `HyphaAgentExecutor` from your service
4. Sets up a `DefaultRequestHandler` with `InMemoryTaskStore`
5. Builds an `A2AStarletteApplication` with your agent card
6. Serves the A2A app as a standard ASGI application

This provides A2A services with their own dedicated URL space and specialized handling, while maintaining consistent behavior and performance with other Hypha services.

## Registering A2A Services

A2A services are registered using the standard Hypha service registration API with `type="a2a"`. The service only requires an Agent Card and a `run` function - the server handles all A2A protocol complexity.

### Basic A2A Service Registration

<!-- tabs:start -->
#### ** Python **

```python
import asyncio
from hypha_rpc import connect_to_server

async def text_generation_agent(message):
    """Simple agent run function."""
    # Extract text from message parts
    text_content = ""
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_content += part.get("text", "")
    
    # Simple text generation (replace with your AI model)
    response_text = f"Generated response for: {text_content}"
    
    # Return simple text response - server handles A2A Task wrapping
    return response_text

async def main():
    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Register the A2A service
    service_info = await server.register_service({
        "id": "text-generator",
        "name": "Text Generation Agent",
        "type": "a2a",
        "config": {
            "visibility": "public"
        },
        "agent_card": {
            "protocolVersion": "0.2.9",
            "name": "Text Generation Agent",
            "description": "A simple text generation agent that can create various types of content",
            "version": "1.0.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False
            },
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "text/markdown"],
            "skills": [
                {
                    "id": "text-generation",
                    "name": "Text Generation",
                    "description": "Generate text content based on user prompts",
                    "tags": ["text", "generation", "content", "writing"],
                    "examples": [
                        "Generate a short story about a robot",
                        "Write a technical explanation of machine learning"
                    ]
                }
            ]
        },
        "run": text_generation_agent
    })
    
    print(f"A2A agent registered: {service_info['id']}")
    await server.serve()

asyncio.run(main())
```

#### ** JavaScript **

First, install the `hypha-rpc` library:

```bash
npm install hypha-rpc
```

Or include it via CDN in your HTML file:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.66/dist/hypha-rpc-websocket.min.js"></script>
```

```javascript
function textGenerationAgent(message) {
    // Extract text from message parts
    let textContent = "";
    for (const part of message.parts || []) {
        if (part.kind === "text") {
            textContent += part.text || "";
        }
    }
    
    // Simple text generation (replace with your AI model)
    const responseText = `Generated response for: ${textContent}`;
    
    // Return simple text response - server handles A2A Task wrapping
    return responseText;
}

async function main() {
    // Connect to Hypha server
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://hypha.aicell.io"
    });
    
    // Register the A2A service
    const serviceInfo = await server.registerService({
        id: "text-generator",
        name: "Text Generation Agent",
        type: "a2a",
        config: {
            visibility: "public"
        },
        agent_card: {
            protocolVersion: "0.2.9",
            name: "Text Generation Agent",
            description: "A simple text generation agent that can create various types of content",
            version: "1.0.0",
            capabilities: {
                streaming: true,
                pushNotifications: false,
                stateTransitionHistory: false
            },
            defaultInputModes: ["text/plain", "application/json"],
            defaultOutputModes: ["text/plain", "text/markdown"],
            skills: [
                {
                    id: "text-generation",
                    name: "Text Generation",
                    description: "Generate text content based on user prompts",
                    tags: ["text", "generation", "content", "writing"],
                    examples: [
                        "Generate a short story about a robot",
                        "Write a technical explanation of machine learning"
                    ]
                }
            ]
        },
        run: textGenerationAgent
    });
    
    console.log(`A2A agent registered: ${serviceInfo.id}`);
    // Keep the server running (in a web context, this would be handled by the page lifecycle)
}

main();
```
<!-- tabs:end -->
```

### Streaming A2A Service

For streaming responses, simply return an async generator from your `run` function:

<!-- tabs:start -->
#### ** Python **

```python
import asyncio
from hypha_rpc import connect_to_server

async def streaming_chat_agent(message, context=None):
    """Streaming agent run function."""
    # Extract text from message parts
    text_content = ""
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_content += part.get("text", "")
    
    # Get user info if available
    user_id = context.get("user", {}).get("id") if context else "anonymous"
    
    # Return async generator for streaming - server handles SSE conversion
    async def stream_response():
        response_parts = [
            f"Hello {user_id}! ",
            "I received your message: ",
            f'"{text_content}". ',
            "Let me think about this... ",
            "Here's my detailed response: ",
            "This is a streaming response that comes in chunks. ",
            "Each chunk is sent as it becomes available. ",
            "This enables real-time interaction with the agent."
        ]
        
        for part in response_parts:
            yield part
            await asyncio.sleep(0.5)  # Simulate processing time
    
    return stream_response()

async def main():
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    service_info = await server.register_service({
        "id": "streaming-chat-agent",
        "name": "Streaming Chat Agent",
        "type": "a2a",
        "config": {
            "visibility": "public",
            "require_context": True  # Enable user context
        },
        "agent_card": {
            "protocolVersion": "0.2.9",
            "name": "Streaming Chat Agent",
            "description": "An AI chat agent with streaming response capabilities",
            "version": "1.0.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False
            },
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            },
            "security": [{"bearerAuth": []}],
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "text/markdown"],
            "skills": [
                {
                    "id": "chat",
                    "name": "Interactive Chat",
                    "description": "Engage in conversational interactions with streaming responses",
                    "tags": ["chat", "conversation", "streaming", "interactive"],
                    "examples": [
                        "Tell me about the weather",
                        "Help me write a Python function",
                        "Explain quantum computing in simple terms"
                    ]
                }
            ]
        },
        "run": streaming_chat_agent
    })
    
    print(f"Streaming A2A agent registered: {service_info['id']}")
    await server.serve()

asyncio.run(main())
```

#### ** JavaScript **

```javascript
async function streamingChatAgent(message, context) {
    // Extract text from message parts
    let textContent = "";
    for (const part of message.parts || []) {
        if (part.kind === "text") {
            textContent += part.text || "";
        }
    }
    
    // Get user info if available
    const userId = context?.user?.id || "anonymous";
    
    // Return async generator for streaming - server handles SSE conversion
    async function* streamResponse() {
        const responseParts = [
            `Hello ${userId}! `,
            "I received your message: ",
            `"${textContent}". `,
            "Let me think about this... ",
            "Here's my detailed response: ",
            "This is a streaming response that comes in chunks. ",
            "Each chunk is sent as it becomes available. ",
            "This enables real-time interaction with the agent."
        ];
        
        for (const part of responseParts) {
            yield part;
            await new Promise(resolve => setTimeout(resolve, 500)); // Simulate processing time
        }
    }
    
    return streamResponse();
}

async function main() {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://hypha.aicell.io"
    });
    
    const serviceInfo = await server.registerService({
        id: "streaming-chat-agent",
        name: "Streaming Chat Agent",
        type: "a2a",
        config: {
            visibility: "public",
            require_context: true  // Enable user context
        },
        agent_card: {
            protocolVersion: "0.2.9",
            name: "Streaming Chat Agent",
            description: "An AI chat agent with streaming response capabilities",
            version: "1.0.0",
            capabilities: {
                streaming: true,
                pushNotifications: false,
                stateTransitionHistory: false
            },
            securitySchemes: {
                bearerAuth: {
                    type: "http",
                    scheme: "bearer"
                }
            },
            security: [{"bearerAuth": []}],
            defaultInputModes: ["text/plain", "application/json"],
            defaultOutputModes: ["text/plain", "text/markdown"],
            skills: [
                {
                    id: "chat",
                    name: "Interactive Chat",
                    description: "Engage in conversational interactions with streaming responses",
                    tags: ["chat", "conversation", "streaming", "interactive"],
                    examples: [
                        "Tell me about the weather",
                        "Help me write a Python function",
                        "Explain quantum computing in simple terms"
                    ]
                }
            ]
        },
        run: streamingChatAgent
    });
    
    console.log(`Streaming A2A agent registered: ${serviceInfo.id}`);
    // Keep the server running (in a web context, this would be handled by the page lifecycle)
}

main();
```
<!-- tabs:end -->
```

### Using Structured Models for Agent Cards

You can use structured models to define your agent card with better organization and validation:

<!-- tabs:start -->
#### ** Python (with Pydantic) **

```python
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from hypha_rpc import connect_to_server

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    examples: List[str]
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None

class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False

class AgentProvider(BaseModel):
    organization: str
    url: str

class AgentCard(BaseModel):
    protocolVersion: str = "0.2.9"
    name: str
    description: str
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]
    provider: Optional[AgentProvider] = None
    securitySchemes: Optional[Dict[str, Any]] = None
    security: Optional[List[Dict[str, List[str]]]] = None

async def math_agent(message, context=None):
    """A simple math agent."""
    text_content = ""
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_content += part.get("text", "")
    
    # Simple math evaluation (in practice, use a proper math parser)
    try:
        if "+" in text_content:
            parts = text_content.split("+")
            result = sum(float(p.strip()) for p in parts)
            return f"The result is: {result}"
    except:
        pass
    
    return "I can help with simple math operations like addition."

def create_agent_card():
    """Function that returns the agent card configuration."""
    return AgentCard(
        name="Math Agent",
        description="A simple agent that can perform basic mathematical operations",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False
        ),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[
            AgentSkill(
                id="basic-math",
                name="Basic Math",
                description="Perform basic mathematical operations like addition",
                tags=["math", "calculator", "arithmetic"],
                examples=[
                    "What is 5 + 3?",
                    "Calculate 10 + 15 + 20"
                ]
            )
        ],
        provider=AgentProvider(
            organization="My Organization",
            url="https://my-org.com"
        )
    ).model_dump()  # Convert Pydantic model to dict

async def main():
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    service_info = await server.register_service({
        "id": "math-agent",
        "name": "Math Agent",
        "type": "a2a",
        "config": {"visibility": "public"},
        "agent_card": create_agent_card,  # Function that returns agent card
        "run": math_agent
    })
    
    print(f"Math agent registered: {service_info['id']}")
    await server.serve()

asyncio.run(main())
```

#### ** JavaScript (with Classes) **

```javascript
class AgentSkill {
    constructor(id, name, description, tags, examples, inputModes = null, outputModes = null) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.tags = tags;
        this.examples = examples;
        this.inputModes = inputModes;
        this.outputModes = outputModes;
    }
}

class AgentCapabilities {
    constructor(streaming = false, pushNotifications = false, stateTransitionHistory = false) {
        this.streaming = streaming;
        this.pushNotifications = pushNotifications;
        this.stateTransitionHistory = stateTransitionHistory;
    }
}

class AgentProvider {
    constructor(organization, url) {
        this.organization = organization;
        this.url = url;
    }
}

class AgentCard {
    constructor(name, description, version, capabilities, defaultInputModes, defaultOutputModes, skills, provider = null, securitySchemes = null, security = null) {
        this.protocolVersion = "0.2.9";
        this.name = name;
        this.description = description;
        this.version = version;
        this.capabilities = capabilities;
        this.defaultInputModes = defaultInputModes;
        this.defaultOutputModes = defaultOutputModes;
        this.skills = skills;
        this.provider = provider;
        this.securitySchemes = securitySchemes;
        this.security = security;
    }
}

function mathAgent(message, context) {
    // Extract text from message parts
    let textContent = "";
    for (const part of message.parts || []) {
        if (part.kind === "text") {
            textContent += part.text || "";
        }
    }
    
    // Simple math evaluation (in practice, use a proper math parser)
    try {
        if (textContent.includes("+")) {
            const parts = textContent.split("+");
            const result = parts.reduce((sum, part) => sum + parseFloat(part.trim()), 0);
            return `The result is: ${result}`;
        }
    } catch (error) {
        // Handle parsing errors
    }
    
    return "I can help with simple math operations like addition.";
}

function createAgentCard() {
    // Function that returns the agent card configuration
    return new AgentCard(
        "Math Agent",
        "A simple agent that can perform basic mathematical operations",
        "1.0.0",
        new AgentCapabilities(false, false),
        ["text/plain"],
        ["text/plain"],
        [
            new AgentSkill(
                "basic-math",
                "Basic Math",
                "Perform basic mathematical operations like addition",
                ["math", "calculator", "arithmetic"],
                [
                    "What is 5 + 3?",
                    "Calculate 10 + 15 + 20"
                ]
            )
        ],
        new AgentProvider(
            "My Organization",
            "https://my-org.com"
        )
    );
}

async function main() {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://hypha.aicell.io"
    });
    
    const serviceInfo = await server.registerService({
        id: "math-agent",
        name: "Math Agent",
        type: "a2a",
        config: { visibility: "public" },
        agent_card: createAgentCard,  // Function that returns agent card
        run: mathAgent
    });
    
    console.log(`Math agent registered: ${serviceInfo.id}`);
    // Keep the server running (in a web context, this would be handled by the page lifecycle)
}

main();
```
<!-- tabs:end -->
```

### Dynamic Agent Card Generation

Agent card functions can also be dynamic, allowing you to generate different configurations based on runtime conditions:

<!-- tabs:start -->
#### ** Python **

```python
def create_dynamic_agent_card(enable_streaming=False, user_permissions=None):
    """Dynamically create agent card based on conditions."""
    capabilities = AgentCapabilities(
        streaming=enable_streaming,
        pushNotifications=False
    )
    
    skills = [
        AgentSkill(
            id="basic-chat",
            name="Basic Chat",
            description="Basic conversational capabilities",
            tags=["chat", "conversation"],
            examples=["Hello", "How are you?"]
        )
    ]
    
    # Add advanced skills based on permissions
    if user_permissions and "advanced" in user_permissions:
        skills.append(
            AgentSkill(
                id="advanced-analysis",
                name="Advanced Analysis", 
                description="Advanced data analysis capabilities",
                tags=["analysis", "data", "advanced"],
                examples=["Analyze this dataset", "Generate insights"]
            )
        )
    
    return AgentCard(
        name="Dynamic Agent",
        description="An agent with dynamic capabilities",
        version="2.0.0",
        capabilities=capabilities,
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["text/plain"],
        skills=skills
    ).model_dump()

# Usage with dynamic configuration
service_info = await server.register_service({
    "id": "dynamic-agent",
    "type": "a2a",
    "config": {"visibility": "public"},
    "agent_card": lambda: create_dynamic_agent_card(
        enable_streaming=True,
        user_permissions=["advanced"]
    ),
    "run": my_agent_function
})
```

#### ** JavaScript **

```javascript
function createDynamicAgentCard(enableStreaming = false, userPermissions = null) {
    // Dynamically create agent card based on conditions
    const capabilities = new AgentCapabilities(
        enableStreaming,
        false
    );
    
    const skills = [
        new AgentSkill(
            "basic-chat",
            "Basic Chat",
            "Basic conversational capabilities",
            ["chat", "conversation"],
            ["Hello", "How are you?"]
        )
    ];
    
    // Add advanced skills based on permissions
    if (userPermissions && userPermissions.includes("advanced")) {
        skills.push(
            new AgentSkill(
                "advanced-analysis",
                "Advanced Analysis",
                "Advanced data analysis capabilities",
                ["analysis", "data", "advanced"],
                ["Analyze this dataset", "Generate insights"]
            )
        );
    }
    
    return new AgentCard(
        "Dynamic Agent",
        "An agent with dynamic capabilities",
        "2.0.0",
        capabilities,
        ["text/plain", "application/json"],
        ["text/plain"],
        skills
    );
}

// Usage with dynamic configuration
const serviceInfo = await server.registerService({
    id: "dynamic-agent",
    type: "a2a",
    config: { visibility: "public" },
    agent_card: () => createDynamicAgentCard(
        true,  // enableStreaming
        ["advanced"]  // userPermissions
    ),
    run: myAgentFunction
});
```
<!-- tabs:end -->
```

## Summary

A2A services in Hypha follow an extremely simple pattern across both Python and JavaScript:

1. **Register** with `type="a2a"` (Python) or `type: "a2a"` (JavaScript)
2. **Provide** an `agent_card` (A2A Agent Card specification)
3. **Implement** a `run` function (your agent logic)

The server handles all A2A protocol complexity - JSON-RPC conversion, task management, streaming, authentication, and response formatting. Your `run` function just needs to process the message and return a response (string, dict/object, list/array, or async generator for streaming).

**JavaScript developers**: Use camelCase naming conventions (e.g., `registerService`, `connectToServer`) and all the same functionality is available as shown in the examples above.

## A2A Service Configuration

### Simple Service Structure

A2A services in Hypha require only three main components:

```python
await server.register_service({
    "id": "my-agent",           # Service identifier
    "name": "My Agent",         # Display name
    "type": "a2a",             # Service type
    "config": {                # Standard Hypha config
        "visibility": "public", # or "protected"
        "require_context": True # Optional: enable user context
    },
    "agent_card": {            # A2A Agent Card specification (dict or function)
        # ... (see below)
    },
    "run": my_agent_function # Your agent logic
})
```

### Agent Card Specification

The `agent_card` field can be either a dictionary or a function that returns a dictionary. It must comply with the A2A Agent Card specification:

```python
agent_card = {
    "protocolVersion": "0.2.9",  # Required: A2A protocol version
    "name": "Your Agent Name",   # Required: Human-readable name
    "description": "Agent description",  # Required: Description of capabilities
    "version": "1.0.0",         # Required: Agent version
    
    # Required: Capabilities supported by your agent
    "capabilities": {
        "streaming": True,              # Support for SSE streaming
        "pushNotifications": False,     # Support for webhook notifications
        "stateTransitionHistory": False # Support for task history
    },
    
    # Required: Input/output media types
    "defaultInputModes": ["text/plain", "application/json"],
    "defaultOutputModes": ["text/plain", "text/markdown"],
    
    # Required: Agent skills/capabilities
    "skills": [
        {
            "id": "skill-id",
            "name": "Skill Name",
            "description": "What this skill does",
            "tags": ["tag1", "tag2"],
            "examples": ["Example usage 1", "Example usage 2"]
        }
    ],
    
    # Optional: Authentication requirements
    "securitySchemes": {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer"
        }
    },
    "security": [{"bearerAuth": []}],
    
    # Optional: Provider information
    "provider": {
        "organization": "Your Organization",
        "url": "https://your-website.com"
    }
}
```

**When to use functions:**
- Dynamic agent card generation based on runtime conditions
- Type safety and validation with Pydantic models
- Configuration that depends on environment variables or user permissions
- Agent cards that need to be computed or fetched from external sources

### Run Function

The `run` function is your agent's core logic:

```python
async def my_agent(message, context=None):
    """
    Args:
        message: A2A Message object with parts (text, file, data)
        context: Hypha user context (if require_context=True)
    
    Returns:
        - String: Simple text response
        - Dict: Structured response (converted to DataPart)
        - List: Multiple parts response
        - Async generator: Streaming response
    """
    # Your agent logic here
    return "Response text"
```

The server automatically handles:
- A2A protocol methods (message/send, message/stream, tasks/*)
- Task ID and context ID generation
- Response wrapping in A2A Task format
- Streaming via Server-Sent Events
- Authentication and authorization
```

## Service Endpoints

When you register an A2A service with ID `my-agent`, it becomes available at:

### Agent Card Endpoint
```
GET /{workspace}/a2a/my-agent/agent.json
```
Returns the Agent Card JSON document for discovery.

### A2A Protocol Endpoint
```
POST /{workspace}/a2a/my-agent
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": { ... },
  "id": 1
}
```

### Streaming Endpoint (if supported)
```
POST /{workspace}/a2a/my-agent
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "message/stream",
  "params": { ... },
  "id": 1
}
```
Returns Server-Sent Events stream.

## Example: File Processing Agent

```python
import asyncio
from hypha_rpc import connect_to_server

async def file_processing_agent(message, context=None):
    """Process files sent in the message."""
    parts = message.get("parts", [])
    results = []
    
    # Process each part of the message
    for part in parts:
        if part.get("kind") == "text":
            results.append({
                "type": "text_analysis",
                "content": f"Processed text: {part.get('text', '')}"
            })
        elif part.get("kind") == "file":
            file_info = part.get("file", {})
            mime_type = file_info.get("mimeType", "")
            
            if mime_type.startswith("image/"):
                # Process image
                result = await process_image(file_info)
                results.append(result)
            elif mime_type == "application/pdf":
                # Process PDF
                result = await process_pdf(file_info)
                results.append(result)
    
    # Return structured results - server wraps in A2A Task format
    return {
        "results": results,
        "summary": f"Processed {len(parts)} items"
    }

async def process_image(file_info):
    """Process image file."""
    # Simulate image analysis
    name = file_info.get("name", "unknown")
    mime_type = file_info.get("mimeType", "")
    
    return {
        "type": "image_analysis",
        "filename": name,
        "mime_type": mime_type,
        "description": "A sample image analysis result",
        "objects_detected": ["object1", "object2"],
        "text_extracted": "Sample extracted text",
        "confidence": 0.95
    }

async def process_pdf(file_info):
    """Process PDF file."""
    # Simulate PDF processing
    name = file_info.get("name", "unknown")
    
    return {
        "type": "pdf_analysis",
        "filename": name,
        "page_count": 5,
        "text_summary": "This is a sample PDF document analysis",
        "key_topics": ["topic1", "topic2", "topic3"],
        "entities": ["entity1", "entity2"]
    }

async def main():
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    service_info = await server.register_service({
        "id": "file-processor",
        "name": "File Processing Agent",
        "type": "a2a",
        "config": {
            "visibility": "public"
        },
        "agent_card": {
            "protocolVersion": "0.2.9",
            "name": "File Processing Agent",
            "description": "Process and analyze various file types including images and documents",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False
            },
            "defaultInputModes": ["text/plain", "application/json", "image/*", "application/pdf"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "skills": [
                {
                    "id": "image-analysis",
                    "name": "Image Analysis",
                    "description": "Analyze and describe images, extract text, detect objects",
                    "tags": ["image", "analysis", "ocr", "detection"],
                    "examples": [
                        "Analyze this image and describe what you see",
                        "Extract text from this document image"
                    ],
                    "inputModes": ["image/png", "image/jpeg", "image/gif"],
                    "outputModes": ["text/plain", "application/json"]
                },
                {
                    "id": "document-processing",
                    "name": "Document Processing", 
                    "description": "Process and extract information from documents",
                    "tags": ["document", "pdf", "text", "extraction"],
                    "examples": [
                        "Extract key information from this contract",
                        "Summarize this research paper"
                    ],
                    "inputModes": ["application/pdf", "text/plain"],
                    "outputModes": ["text/plain", "application/json"]
                }
            ]
        },
        "run": file_processing_agent
    })
    
    print(f"File processing agent registered: {service_info['id']}")
    await server.serve()

asyncio.run(main())
```

## Client Usage

Once your A2A service is registered, it can be accessed by any A2A-compatible client:

### Using Hypha-RPC Client

<!-- tabs:start -->
#### ** Python **

```python
import asyncio
from hypha_rpc import connect_to_server

async def test_agent():
    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    # Get the A2A service
    service = await server.get_service("my-workspace/text-generator")
    
    # Call the service directly
    response = await service.run({
        "role": "user",
        "parts": [
            {
                "kind": "text",
                "text": "Generate a short story about AI"
            }
        ],
        "messageId": "msg-123"
    })
    
    print("Response:", response)

asyncio.run(test_agent())
```

#### ** JavaScript **

```javascript
async function testAgent() {
    // Connect to Hypha server
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://hypha.aicell.io"
    });
    
    // Get the A2A service
    const service = await server.getService("my-workspace/text-generator");
    
    // Call the service directly
    const response = await service.run({
        role: "user",
        parts: [
            {
                kind: "text",
                text: "Generate a short story about AI"
            }
        ],
        messageId: "msg-123"
    });
    
    console.log("Response:", response);
}

testAgent();
```
<!-- tabs:end -->

### Using Python A2A Client

```python
import asyncio
from a2a import A2AClient

async def test_agent():
    # Connect to the A2A agent
    client = A2AClient()
    agent_url = "https://hypha.aicell.io/my-workspace/a2a/text-generator"
    
    # Send a message
    response = await client.send_message(
        agent_url,
        message={
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Generate a short story about AI"
                }
            ],
            "messageId": "msg-123"
        }
    )
    
    print("Response:", response)

asyncio.run(test_agent())
```

### Using HTTP Requests

<!-- tabs:start -->
#### ** Python **

```python
import httpx
import json

async def call_agent_http():
    agent_url = "https://hypha.aicell.io/my-workspace/a2a/text-generator"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Hello, AI agent!"
                    }
                ],
                "messageId": "msg-456"
            }
        },
        "id": 1
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            agent_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        print("Agent response:", result)

asyncio.run(call_agent_http())
```

#### ** JavaScript **

```javascript
async function callAgentHttp() {
    const agentUrl = "https://hypha.aicell.io/my-workspace/a2a/text-generator";
    
    const payload = {
        jsonrpc: "2.0",
        method: "message/send",
        params: {
            message: {
                role: "user",
                parts: [
                    {
                        kind: "text",
                        text: "Hello, AI agent!"
                    }
                ],
                messageId: "msg-456"
            }
        },
        id: 1
    };
    
    try {
        const response = await fetch(agentUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        console.log("Agent response:", result);
    } catch (error) {
        console.error("Error calling agent:", error);
    }
}

callAgentHttp();
```
<!-- tabs:end -->
```

## Authentication and Security

### Requiring Authentication

<!-- tabs:start -->
#### ** Python **

```python
async def secure_agent(message, context=None):
    """Agent with user context for authentication."""
    if context:
        user_id = context.get("user", {}).get("id")
        permissions = context.get("user", {}).get("permissions", [])
        
        # Check user permissions
        if "agent_access" not in permissions:
            raise Exception("Insufficient permissions")
        
        print(f"Processing request from user: {user_id}")
    
    # Process the message...
    return f"Secure response for user {user_id}"

# Register with authentication required
service_info = await server.register_service({
    "id": "secure-agent",
    "type": "a2a",
    "config": {
        "visibility": "protected",  # Requires authentication
        "require_context": True,    # Provides user context to run function
    },
    "agent_card": {
        # ... other fields ...
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer"
            }
        },
        "security": [{"bearerAuth": []}]
    },
    "run": secure_agent
})
```

#### ** JavaScript **

```javascript
function secureAgent(message, context) {
    // Agent with user context for authentication
    if (context) {
        const userId = context.user?.id;
        const permissions = context.user?.permissions || [];
        
        // Check user permissions
        if (!permissions.includes("agent_access")) {
            throw new Error("Insufficient permissions");
        }
        
        console.log(`Processing request from user: ${userId}`);
    }
    
    // Process the message...
    return `Secure response for user ${userId}`;
}

// Register with authentication required
const serviceInfo = await server.registerService({
    id: "secure-agent",
    type: "a2a",
    config: {
        visibility: "protected",  // Requires authentication
        require_context: true,    // Provides user context to run function
    },
    agent_card: {
        // ... other fields ...
        securitySchemes: {
            bearerAuth: {
                type: "http",
                scheme: "bearer"
            }
        },
        security: [{"bearerAuth": []}]
    },
    run: secureAgent
});
```
<!-- tabs:end -->
```

## Advanced Features

### Long-Running Tasks 

For long-running tasks, you can use asyncio background processing:

```python
import asyncio
from hypha_rpc import connect_to_server

# In-memory task storage (use proper database in production)
background_tasks = {}

async def long_running_agent(message, context=None):
    """Handle long-running tasks."""
    # Extract task details from message
    text_content = ""
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_content += part.get("text", "")
    
    # Start background processing
    task_id = f"bg-task-{asyncio.get_event_loop().time()}"
    task = asyncio.create_task(process_in_background(text_content))
    background_tasks[task_id] = task
    
    # Return immediate response indicating task is running
    return f"Task {task_id} started. Processing in background: {text_content}"

async def process_in_background(content):
    """Background processing function."""
    print(f"Starting background processing: {content}")
    
    # Simulate long processing
    await asyncio.sleep(30)
    
    print(f"Background processing completed: {content}")
    return f"Processed result for: {content}"

async def main():
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    service_info = await server.register_service({
        "id": "long-running-agent",
        "name": "Long Running Agent",
        "type": "a2a",
        "config": {
            "visibility": "public"
        },
        "agent_card": {
            "protocolVersion": "0.2.9",
            "name": "Long Running Agent",
            "description": "An agent that handles long-running background tasks",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,  # Server handles push notifications 
                "stateTransitionHistory": False
            },
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [
                {
                    "id": "background-processing",
                    "name": "Background Processing",
                    "description": "Process tasks in the background",
                    "tags": ["background", "long-running", "async"],
                    "examples": ["Process this large dataset"]
                }
            ]
        },
        "run": long_running_agent
    })
    
    print(f"Long-running agent registered: {service_info['id']}")
    await server.serve()

asyncio.run(main())
```

## Best Practices

### Error Handling

<!-- tabs:start -->
#### ** Python **

```python
import logging

logger = logging.getLogger(__name__)

async def robust_agent(message, context=None):
    """Agent with proper error handling."""
    try:
        # Validate message
        if not message:
            raise ValueError("No message provided")
        
        parts = message.get("parts", [])
        if not parts:
            raise ValueError("Message has no parts")
        
        # Process message safely
        result = await process_message_safely(message, context)
        return result
        
    except ValueError as e:
        # Client errors - return error message
        raise Exception(f"Invalid request: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error in agent: {e}")
        raise Exception("Internal server error")

async def process_message_safely(message, context):
    """Safely process the message with validation."""
    # Your agent logic here with proper validation
    text_content = ""
    for part in message.get("parts", []):
        if part.get("kind") == "text":
            text_content += part.get("text", "")
    
    if not text_content.strip():
        raise ValueError("No text content found in message")
    
    return f"Processed: {text_content}"
```

#### ** JavaScript **

```javascript
function robustAgent(message, context) {
    // Agent with proper error handling
    try {
        // Validate message
        if (!message) {
            throw new Error("No message provided");
        }
        
        const parts = message.parts || [];
        if (parts.length === 0) {
            throw new Error("Message has no parts");
        }
        
        // Process message safely
        const result = processMessageSafely(message, context);
        return result;
        
    } catch (error) {
        if (error.name === 'Error') {
            // Client errors - return error message
            throw new Error(`Invalid request: ${error.message}`);
        } else {
            console.error(`Unexpected error in agent: ${error}`);
            throw new Error("Internal server error");
        }
    }
}

function processMessageSafely(message, context) {
    // Safely process the message with validation
    let textContent = "";
    for (const part of message.parts || []) {
        if (part.kind === "text") {
            textContent += part.text || "";
        }
    }
    
    if (!textContent.trim()) {
        throw new Error("No text content found in message");
    }
    
    return `Processed: ${textContent}`;
}
```
<!-- tabs:end -->
```

### Resource Management

<!-- tabs:start -->
#### ** Python **

```python
import asyncio
from typing import Dict, Any

class ResourceManagedAgent:
    def __init__(self):
        self.max_concurrent_tasks = 10
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.active_requests = {}
    
    async def __call__(self, message: Dict[str, Any], context=None):
        """Agent run function with resource limits."""
        async with self.task_semaphore:
            return await self._process_message(message, context)
    
    async def _process_message(self, message, context):
        """Actual message processing with resource tracking."""
        request_id = f"req-{asyncio.get_event_loop().time()}"
        self.active_requests[request_id] = asyncio.get_event_loop().time()
        
        try:
            # Your agent logic here
            result = await self.generate_response(message)
            return result
        finally:
            # Clean up request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def generate_response(self, message):
        """Generate response with proper resource management."""
        # Simulate processing with timeout
        try:
            return await asyncio.wait_for(
                self.do_heavy_processing(message),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            raise Exception("Request timeout - processing took too long")
    
    async def do_heavy_processing(self, message):
        """Heavy processing that might take time."""
        # Your computationally intensive logic here
        await asyncio.sleep(1)  # Simulate work
        return "Processed response"

# Register the agent instance
agent_instance = ResourceManagedAgent()

service_info = await server.register_service({
    "id": "resource-managed-agent",
    "type": "a2a",
    "config": {"visibility": "public"},
    "agent_card": { ... },
    "run": agent_instance.__call__
})
```

#### ** JavaScript **

```javascript
class ResourceManagedAgent {
    constructor() {
        this.maxConcurrentTasks = 10;
        this.activeTasks = new Set();
        this.activeRequests = new Map();
    }
    
    async call(message, context) {
        // Agent run function with resource limits
        if (this.activeTasks.size >= this.maxConcurrentTasks) {
            throw new Error("Too many concurrent requests");
        }
        
        const taskId = `task-${Date.now()}-${Math.random()}`;
        this.activeTasks.add(taskId);
        
        try {
            return await this._processMessage(message, context);
        } finally {
            this.activeTasks.delete(taskId);
        }
    }
    
    async _processMessage(message, context) {
        // Actual message processing with resource tracking
        const requestId = `req-${Date.now()}`;
        this.activeRequests.set(requestId, Date.now());
        
        try {
            // Your agent logic here
            const result = await this.generateResponse(message);
            return result;
        } finally {
            // Clean up request tracking
            this.activeRequests.delete(requestId);
        }
    }
    
    async generateResponse(message) {
        // Generate response with proper resource management
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error("Request timeout - processing took too long")), 30000);
        });
        
        try {
            return await Promise.race([
                this.doHeavyProcessing(message),
                timeoutPromise
            ]);
        } catch (error) {
            throw error;
        }
    }
    
    async doHeavyProcessing(message) {
        // Heavy processing that might take time
        // Your computationally intensive logic here
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate work
        return "Processed response";
    }
}

// Register the agent instance
const agentInstance = new ResourceManagedAgent();

const serviceInfo = await server.registerService({
    id: "resource-managed-agent",
    type: "a2a",
    config: { visibility: "public" },
    agent_card: { /* ... */ },
    run: agentInstance.call.bind(agentInstance)
});
```
<!-- tabs:end -->
```

## Troubleshooting

### Common Issues

1. **Service not registering**: Ensure A2A support is enabled with `--enable-a2a`
2. **Agent Card validation errors**: Check that your Agent Card follows the A2A specification
3. **Handler not found**: Verify that all required handlers are implemented
4. **Authentication failures**: Check security schemes and token validation
5. **Streaming not working**: Ensure your handler returns an async generator

### Debugging

Enable debug logging for A2A services:

```python
import logging
logging.getLogger("hypha.a2a").setLevel(logging.DEBUG)
```

Check service status:

```bash
curl -X GET "https://hypha.aicell.io/my-workspace/services/my-agent"
```

Validate Agent Card:

```bash
curl -X GET "https://hypha.aicell.io/my-workspace/a2a/my-agent/agent.json"
```

### HTTP Endpoints

When you register an A2A service, Hypha automatically creates these endpoints:

- **Agent Card**: `/{workspace}/a2a/{service-id}/agent.json`
- **A2A Protocol**: `/{workspace}/a2a/{service-id}` (POST for JSON-RPC requests)

### Service Discovery

A2A services are discoverable through:

1. **Agent Card URL**: Standard A2A discovery mechanism
2. **Hypha Service API**: Listed alongside other Hypha services
3. **Service Search**: Indexed in Hypha's service search if enabled

### Integration with Hypha Features

A2A services inherit Hypha's capabilities:

- **Authentication**: Use Hypha's authentication schemes
- **Workspaces**: Scoped to workspace access controls
- **Service Management**: Start/stop/monitor via Hypha APIs
- **Logging**: Integrated with Hypha's logging system
- **Metrics**: Tracked with Hypha's metrics collection

## Conclusion

Hypha's A2A service support enables you to create AI agents that seamlessly integrate with the broader A2A ecosystem. By following the A2A protocol specification and using Hypha's simplified registration process, you can build agents that are discoverable, interoperable, and ready for production use.

The combination of Hypha's infrastructure capabilities with A2A's standardized protocol creates a powerful platform for building and deploying AI agents that can collaborate effectively with other systems and agents. 