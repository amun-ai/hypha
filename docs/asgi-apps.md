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

Here’s a simple example of how to serve a FastAPI application using the `hypha_rpc.utils.serve` utility.

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

If you don’t have a token or prefer to log in interactively, you can use the `--login` option:

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

Here’s an example of how you can directly register a FastAPI service with Hypha:

```python
import asyncio
from hypha_rpc import connect_to_server
from fastapi import FastAPI, HTMLResponse

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
        "type": "ASGI",
        "serve": serve_fastapi,
        "config": {"visibility": "public"}
    })

    print(f"Access your app at: {server.config.workspace}/apps/{svc_info['id'].split(':')[1]}")
    await server.serve()

asyncio.run(main())
```

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

Here’s an example of how to create and serve a FastAPI app in the browser using Pyodide:

```python
import asyncio
import micropip
await micropip.install(["fast

api==0.70.0"])

from hypha_rpc import connect_to_server
from fastapi import FastAPI, HTMLResponse

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
        "type": "ASGI",
        "serve": serve_fastapi,
        "config": {"visibility": "public"}
    })

    print(f"Access your app at: {server.config.workspace}/apps/{svc_info['id'].split(':')[1]}")
    await server.serve()

await main()
```

### Accessing the Application

Just like in the standard Python environment, the application will be accessible through the URL provided by the Hypha server.

## Conclusion

Hypha offers a versatile and powerful way to serve ASGI applications, whether through the convenient `hypha_rpc.utils.serve` utility, by directly registering your ASGI service, or even by running FastAPI directly in the browser with Pyodide. By removing the need for traditional server setups, Hypha enables you to focus on your application without the overhead of managing infrastructure. Whether you’re deploying in a local environment or directly in a browser, Hypha simplifies the process, making it easier than ever to share your ASGI apps with the world. The added flexibility of direct service registration and browser-based FastAPI apps expands the possibilities for deploying and sharing your web applications.
