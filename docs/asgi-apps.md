# Serving ASGI Applications with Hypha

## Introduction

When deploying ASGI web applications like those built with FastAPI, developers often rely on traditional servers and tools like **Uvicorn**. Uvicorn is a popular choice for serving ASGI applications due to its simplicity and ease of use. Typically, you can start a server with the following command:

```bash
uvicorn myapp:app --host 0.0.0.0 --port 8000
```

However, this approach has limitations, particularly when you need to expose your application to the public. Hosting a server with Uvicorn typically requires:
- A public IP address.
- DNS management.
- SSL certificates for secure connections.
- Firewall configurations to ensure access while maintaining security.

For many users, especially those working in environments with restrictive networks or behind firewalls, exposing a server publicly can be cumbersome and insecure. Moreover, maintaining server infrastructure and ensuring security compliance adds significant overhead.

**Hypha** provides a solution to these challenges by allowing you to serve ASGI applications without the need for traditional server setups. With Hypha, you can instantly deploy and make your web application publicly accessible, bypassing the need for public IP addresses, DNS, or server management. This is especially beneficial in scenarios where you want to quickly share your application without worrying about the complexities of infrastructure.

## The Hypha Serve Utility

To streamline the process of deploying ASGI applications, Hypha offers a utility similar to Uvicorn, but with added benefits. The `hypha_rpc.utils.serve` utility allows you to serve your FastAPI (or any ASGI-compatible) application directly through the Hypha platform. This tool provides a simple command-line interface that makes deployment quick and easy.

### Why Use Hypha Over Uvicorn?

While Uvicorn is a powerful tool for local development and serving applications within controlled environments, Hypha's serve utility offers:
- **Ease of Access**: No need for public IP, DNS management, or SSL certificates.
- **Security**: Hypha handles secure connections and access control.
- **Simplicity**: Deploy your application with a single command, without worrying about infrastructure management.
- **Flexibility**: For advanced use cases, Hypha allows direct registration of ASGI services, giving you fine-grained control.

### Serving Your ASGI App with Hypha

Let's explore how to use the `hypha_rpc.utils.serve` utility to deploy your FastAPI application.

## Prerequisites

- **Hypha**: Ensure you have Hypha installed and configured.
- **FastAPI**: For this example, we'll use FastAPI, but any ASGI-compatible framework will work.
- **Python 3.9+**: The utility requires Python 3.9 or higher.

## Installation

If you haven't installed FastAPI yet, you can do so using pip:

```bash
pip install fastapi
```

Ensure that `hypha_rpc` is installed and available in your environment.

## Usage

### Basic Example

Here's a simple example of how to serve a FastAPI application using the `hypha_rpc.utils.serve` utility.

### Step 1: Create a FastAPI App

Create a simple FastAPI app (e.g., `myapp.py`):

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Cat</title></head>
        <body><img src="https://cataas.com/cat?type=square" alt="cat"></body>
    </html>
    """

@app.get("/api/v1/test")
async def test():
    return {"message": "Hello, it works!"}
```

### Step 2: Serve the App Using the Hypha Serve Utility

You can serve your ASGI app with Hypha using the following command:

```bash
python -m hypha_rpc.utils.serve myapp:app --id=cat --name=Cat --server-url=https://hypha.aicell.io --workspace=my-workspace --token=your_token_here
```

### Parameters

- `myapp:app`: The `module:app` format where `myapp` is your Python file, and `app` is the FastAPI instance.
- `--id`: A unique identifier for your service.
- `--name`: A friendly name for your service.
- `--server-url`: The URL of the Hypha server.
- `--workspace`: The workspace you want to connect to within the Hypha server.
- `--token`: The authentication token for accessing the Hypha server.

### Using the `--login` Option

If you don't have a token or prefer to log in interactively, you can use the `--login` option:

```bash
python -m hypha_rpc.utils.serve myapp:app --id=cat --name=Cat --server-url=https://hypha.aicell.io --workspace=my-workspace --login
```

This will prompt you to log in, and it will automatically retrieve and use the token.

### Disabling SSL Verification

If you need to disable SSL verification (e.g., for testing purposes), you can use the `--disable-ssl` option:

```bash
python -m hypha_rpc.utils.serve myapp:app --id=cat --name=Cat --server-url=https://hypha.aicell.io --workspace=my-workspace --login --disable-ssl
```

### Accessing the Application

Once the app is running, you will see a URL printed in the console, which you can use to access your app. The URL will look something like this:

```
https://hypha.aicell.io/{workspace}/apps/{service_id}
```

Replace `{workspace}` and `{service_id}` with the actual workspace name and service ID you provided.

### Example Commands

**Serve with Token:**

```bash
python -m hypha_rpc.utils.serve myapp:app --id=cat --name=Cat --server-url=https://hypha.aicell.io --workspace=my-workspace --token=sflsflsdlfslfwei32r90jw
```

**Serve with Login and Disable SSL:**

```bash
python -m hypha_rpc.utils.serve myapp:app --id=cat --name=Cat --server-url=https://hypha.aicell.io --workspace=my-workspace --login --disable-ssl
```

## Advanced: Registering the ASGI Service Directly

For users who prefer more flexibility and control, Hypha also allows you to register the ASGI service directly through the platform, bypassing the utility function. This approach is useful if you need custom behavior or want to integrate additional logic into your application.

### Example of Direct Registration

Here's an example of how you can directly register a FastAPI service with Hypha:

```python
import asyncio
from hypha_rpc import connect_to_server
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Cat</title></head>
        <body><img src="https://cataas.com/cat?type=square" alt="cat"></body>
    </html>
    """

@app.get("/api/v1/test")
async def test():
    return {"message": "Hello, it works!"}

async def serve_fastapi(args, context=None):
    # context can be used for authorization, e.g., checking the user's permission
    # e.g., check user id against a list of allowed users
    scope = args["scope"]
    print(f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

async def main():
    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    svc_info = await server.register_service({
        "id": "cat",
        "name": "cat",
        "type": "asgi",
        "serve": serve_fastapi,
        "config": {"visibility": "public", , "require_context": True}
    })

    print(f"Access your app at:  {server.config.public_base_url}/{server.config.workspace}/apps/{svc_info['id'].split(':')[1]}")
    await server.serve()

asyncio.run(main())
```

## Streaming Responses

Hypha fully supports streaming responses from ASGI applications like FastAPI. When you serve an ASGI app through Hypha, the ASGIRoutingMiddleware preserves the ASGI protocol's streaming capabilities, allowing your application to send chunked responses to clients.

### Example: Creating a Streaming Endpoint

Here's an example of how to create a streaming endpoint with FastAPI served through Hypha:

```python
from fastapi import FastAPI
from starlette.responses import StreamingResponse
import asyncio

app = FastAPI()

@app.get("/stream")
async def stream_response():
    async def stream_generator():
        for i in range(10):
            yield f"Chunk {i}\n".encode()
            await asyncio.sleep(0.5)  # Simulate processing time
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain"
    )
```

### How Streaming Works with Hypha

When your ASGI application sends streaming responses:

1. **Protocol Preservation**: Hypha's ASGIRoutingMiddleware preserves the ASGI protocol by passing the original `scope`, `receive`, and `send` callables to your application.

2. **Chunked Transfer**: The middleware allows your app to call the `send` function multiple times with `"more_body": True` for each chunk, followed by a final chunk with `"more_body": False`.

3. **No Buffering**: Responses are streamed directly to the client without being fully buffered by Hypha, making it suitable for large responses or real-time data.

### Common Use Cases for Streaming

- **Large Dataset Processing**: Stream large results without loading everything into memory.
- **Real-time Updates**: Send real-time updates to clients as they become available.
- **Long-running Tasks**: Provide progress updates during long-running tasks.
- **Event Streams**: Implement server-sent events (SSE) for push notifications.

Streaming is particularly useful when working with large files, generating content incrementally, or providing real-time feedback to users while processing complex tasks.

## Running FastAPI in the Browser with Pyodide

Hypha also supports running FastAPI applications directly in the browser using Pyodide. This feature is particularly useful when you want to create a lightweight, client-side application that can be served without any server infrastructure.

### Step 1: Install FastAPI in Pyodide

To install FastAPI in a Pyodide environment, use `micropip`:

```python
import micropip
await micropip.install(["fastapi==0.70.0"])
```

### Step 2: Modify the FastAPI App for Pyodide

The FastAPI app remains largely the same, but you should be aware of a few key differences when running in Pyodide:

- Pyodide runs in a browser, so any blocking or long-running tasks should be handled with care.
- The installation of packages is done using `micropip` instead of `pip`.

### Example Code for Pyodide

Here's an example of how to create and serve a FastAPI app in the browser using Pyodide:

```python
import asyncio
import micropip
await micropip.install(["fastapi==0.70.0"])

from hypha_rpc import connect_to_server
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Cat</title></head>
        <body><img src="https://cataas.com/cat?type=square" alt="cat"></body>
    </html>
    """

@app.get("/api/v1/test")
async def test():
    return {"message": "Hello, it works!"}

async def serve_fastapi(args, context=None):
    # context can be used for authorization, e.g., checking the user's permission
    # e.g., check user id against a list of allowed users
    scope = args["scope"]
    print(f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

async def main():
    # Connect to Hypha server
    server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
    
    svc_info = await server.register_service({
        "id": "cat",
        "name": "cat",
        "type": "asgi",
        "serve": serve_fastapi,
        "config": {"visibility": "public", "require_context": True}
    })

    print(f"Access your app at: {server.config.workspace}/apps/{svc_info['id'].split(':')[1]}")
    await server.serve()

await main()
```

### Accessing the Application

Just like in the standard Python environment, the application will be accessible through the URL provided by the Hypha server.

## Launch OpenAI Chat Server with Hypha

Hypha provides a convenient way to emulate an OpenAI Chat server by allowing developers to register custom models or agents with the platform and serve them using the OpenAI-compatible API. This feature allows you to connect any language model or text generator to Hypha and provide a seamless interface for OpenAI API-compatible clients. This includes popular chat clients like [bettergpt.chat](https://bettergpt.chat/) or Python clients using the OpenAI API.

By leveraging the Hypha platform, developers can deploy their custom models and enable clients to interact with them via a familiar API, reducing friction when integrating custom AI models into existing workflows. This flexibility opens up various possibilities, such as creating custom chatbots, domain-specific language models, or even agents capable of handling specific tasks.

### How It Works

Using Hypha, you can register any text-generating function as a model, expose it using the OpenAI Chat API, and serve it through a Hypha endpoint. This allows clients to interact with your custom model just like they would with OpenAI's models.

With the Hypha OpenAI Chat Server, you can:
- Register any LLM or custom text generation function under a `model_id`.
- Serve the model with an OpenAI-compatible API.
- Allow clients (such as `bettergpt.chat` or Python OpenAI clients) to interact with your custom model using standard API calls.
- Secure access to your models with token-based authentication.

### Example: Setting Up a Custom OpenAI Chat Server

#### Step 1: Define a Text Generation Function

Start by creating a text generator function that will emulate a language model. In this example, we will create a simple random markdown generator:

```python
import random
from hypha_rpc.utils.serve import create_openai_chat_server

# Random text generator for Markdown with cat images
async def random_markdown_generator(request):
    words = [
        "hello",
        "world",
        "foo",
        "bar",
        "chatbot",
        "test",
        "api",
        "response",
        "markdown",
    ]
    length = request.get("max_tokens", 50)

    for _ in range(length // 5):  # Insert random text every 5 tokens
        markdown_content = ""
        # Add random text
        markdown_content += f"{random.choice(words)} "

        # Occasionally add a cat image in Markdown format
        if random.random() < 0.3:  # 30% chance to insert an image
            markdown_content += (
                f"\n![A random cat](https://cataas.com/cat?{random.randint(1, 1000)})\n"
            )
        if random.random() < 0.1:  # 10% chance to insert a cat video
            markdown_content += f'\n<iframe width="560" height="315" src="https://www.youtube.com/embed/7TavVZMewpY" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n'
        yield markdown_content

# Register the model
app = create_openai_chat_server({"cat-chat-model": random_markdown_generator})
```

#### Step 2: Serve the OpenAI Chat Server

You can now serve the model through Hypha using the following steps:

1. **Start the server** by registering the ASGI app with Hypha, specifying the model and server details.

```bash
python -m hypha_rpc.utils.serve myapp:app --id=openai-chat --name=OpenAIChat --server-url=https://hypha.aicell.io --workspace=my-workspace --token=your_token_here
```

2. This will expose your `random_markdown_generator` function as the `cat-chat-model` through the OpenAI API.

### Example API Usage with OpenAI-Compatible Clients

Once your OpenAI-compatible chat server is running, it can be accessed by any client that supports OpenAI's API, including Python OpenAI SDK or web-based clients like [bettergpt.chat](https://bettergpt.chat).

#### Step 1: Generating a Token

In your application, you can generate a token for authentication:

```python
server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
token = await server.generate_token()
```

#### Step 2: Interacting with the Model via OpenAI Python Client

You can use the generated token and set the custom OpenAI API endpoint URL to interact with your custom model:

```python
import openai

# Use the custom OpenAI server endpoint
openai.api_base = f"{WS_SERVER_URL}/{workspace}/apps/openai-chat/v1"
openai.api_key = token

# Example request
response = openai.ChatCompletion.create(
    model="cat-chat-model",
    messages=[{"role": "user", "content": "Show me a cat image!"}],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message['content'])
```

In this example, the custom model `cat-chat-model` will generate random markdown text along with cat images.

#### Step 3: Using Web-Based Clients (e.g., bettergpt.chat)

To use a web-based client like [bettergpt.chat](https://bettergpt.chat), simply set the API endpoint to your custom Hypha URL and input the generated token.

- **API Endpoint**: `https://hypha.aicell.io/{workspace}/apps/openai-chat/v1`
- **Token**: (Use the token generated above)

This allows you to interact with your custom language model through the web client interface, just like you would with OpenAI models.

### Advantages of Using Hypha for OpenAI Chat Server

1. **Flexibility**: You can register any text generator, including fine-tuned LLMs, agents, or simple functions, and serve them through the OpenAI-compatible API.
   
2. **Integration**: Once registered, your model can be used by any OpenAI API-compatible client, enabling seamless integration with existing applications, SDKs, or platforms.

3. **Security**: By utilizing Hypha's token-based authentication, you ensure that only authorized clients can access your custom models, enhancing security.

4. **No Infrastructure Overhead**: With Hypha managing the server infrastructure, you can focus on developing models and generating responses without worrying about hosting, scaling, or maintaining servers.

### Example: Using Multiple Models

Hypha also allows you to register and serve multiple models concurrently:

```python
# Define another generator function
async def basic_text_generator(request):
    messages = ["This is a basic response.", "Hello from the chat model!", "Enjoy your conversation!"]
    yield random.choice(messages)

# Register multiple models
model_registry = {
    "cat-chat-model": random_markdown_generator,
    "basic-chat-model": basic_text_generator
}

# Serve the models through OpenAI API
app = create_openai_chat_server(model_registry)
```

In this case, you can interact with either `cat-chat-model` or `basic-chat-model` by specifying the `model_id` in the API requests.

### Streaming Responses with OpenAI Chat Server

Hypha's OpenAI Chat Server fully supports streaming responses, which is particularly useful for LLM applications where you want to display partial results as they're generated. The implementation automatically translates the generator function's yielded responses into the proper streaming format expected by OpenAI clients.

#### Example: Creating a Streaming Chat Response

Here's an example of how to create a streaming LLM-like generator that works with OpenAI clients:

```python
import asyncio
import random
from hypha_rpc.utils.serve import create_openai_chat_server

# Streaming text generator that simulates an LLM typing response
async def streaming_text_generator(request):
    # Get the user's message from the request
    messages = request.get("messages", [])
    user_message = messages[-1]["content"] if messages else "Hello"
    
    # Simulate a thinking delay
    await asyncio.sleep(0.5)
    
    # Define a response text
    response_text = f"I received your message: '{user_message}'. Here's my detailed response:\n\n"
    response_text += "1. First, I want to acknowledge your question.\n"
    response_text += "2. Let me think about this carefully...\n"
    response_text += "3. Based on my analysis, here's what I think...\n"
    
    # Stream the response word by word
    words = response_text.split()
    for word in words:
        # Yield one word at a time
        yield word + " "
        # Add a small random delay to simulate typing
        await asyncio.sleep(random.uniform(0.05, 0.2))

# Register the streaming model
app = create_openai_chat_server({"streaming-model": streaming_text_generator})
```

#### Using the Streaming Model with OpenAI Python Client

You can use the streaming model with the OpenAI Python client like this:

```python
import openai
import sys

# Use the custom OpenAI server endpoint
openai.api_base = f"{SERVER_URL}/{workspace}/apps/openai-chat/v1"
openai.api_key = token

# Create a streaming completion
response = openai.ChatCompletion.create(
    model="streaming-model",
    messages=[{"role": "user", "content": "Tell me about streaming responses"}],
    stream=True  # Enable streaming
)

# Process the streaming response
for chunk in response:
    # Extract the content delta
    content = chunk.choices[0].delta.get("content", "")
    if content:
        # Print without newline and flush to show realtime updates
        sys.stdout.write(content)
        sys.stdout.flush()
```

#### Benefits of Streaming in Chat Applications

- **Improved User Experience**: Users see the response being generated in real-time, creating a more engaging experience.
- **Faster Perceived Response Time**: Users see the beginning of the response immediately instead of waiting for the complete message.
- **Progress Visibility**: For longer responses, users can start reading while the rest is being generated.
- **Connection Confirmation**: Users receive immediate feedback that the system is working on their request.

The streaming capability is particularly valuable for AI chat interfaces where responses might take several seconds to generate completely.

## Conclusion

Hypha provides a flexible, scalable, and powerful platform for serving custom ASGI applications and emulating OpenAI-compatible chat servers. Whether you're using the `hypha_rpc.utils.serve` utility to deploy ASGI apps, directly registering ASGI services, or running FastAPI in the browser via Pyodide, Hypha removes the complexities of traditional server setups, allowing you to focus on developing your applications without worrying about infrastructure management.

Additionally, Hypha's ability to emulate an OpenAI Chat Server opens new opportunities for serving custom language models and chatbots. By providing an OpenAI-compatible API, Hypha enables seamless integration with Python clients or web-based platforms like [bettergpt.chat](https://bettergpt.chat). This makes it easy to deploy, scale, and secure your models for a wide range of use cases, from research to production-ready solutions.

In summary, Hypha simplifies the deployment process, whether you're serving ASGI applications or custom AI models, and expands the possibilities for sharing, scaling, and accessing web applications and intelligent agents globally.
