// A simple demo of running a FastAPI ASGI app with Hypha in Deno
// run with:
// deno run --allow-net --allow-read --allow-env scripts/deno-demo-asgi-app.js

import pyodideModule from "npm:pyodide/pyodide.js";

const pyodide = await pyodideModule.loadPyodide();

await pyodide.loadPackage("ssl")

// Install micropip for installing Python packages
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");

// Install required packages
await micropip.install(['fastapi==0.70.0', 'hypha-rpc']);

// Add WebSocket to Python global scope
pyodide.globals.set("WebSocket", WebSocket);

const pythonCode = `
import asyncio
import sys
from hypha_rpc import connect_to_server
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Create FastAPI app
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
    scope = args["scope"]
    print(f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

async def main():
    try:
        print("Connecting to Hypha server...")
        server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
        print("Successfully connected!")

        print("Registering ASGI service...")
        svc_info = await server.register_service({
            "id": "cat",
            "name": "cat",
            "type": "asgi",
            "serve": serve_fastapi,
            "config": {
                "visibility": "public",
                "require_context": True
            }
        })
        
        service_id = svc_info['id'].split(':')[1]
        print(f"Service registered successfully!")
        print(f"Access your app at: {server.config.public_base_url}/{server.config.workspace}/apps/{service_id}")
        print("Starting server...")
        await server.serve()
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

# Create and get event loop
loop = asyncio.get_event_loop()
loop.create_task(main())
`;

try {
    // Run the Python code
    const result = await pyodide.runPythonAsync(pythonCode);
    console.log("Python code executed successfully:", result);
    
    // Keep the JavaScript process running
    await new Promise(() => {});
} catch (error) {
    console.error("Error running Python code:", error);
    // Exit the process with an error code
    Deno.exit(1);
} 