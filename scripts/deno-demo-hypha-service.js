// A simple demo of a Hypha service that can be run in Deno
// run with:
// deno run --allow-net --allow-read --allow-env --unstable scripts/deno-demo-hypha-service.js

import pyodideModule from "npm:pyodide/pyodide.js";

const pyodide = await pyodideModule.loadPyodide();

// Install micropip for installing Python packages
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");

// Install hypha-rpc
await micropip.install('hypha-rpc');

// Add WebSocket to Python global scope
pyodide.globals.set("WebSocket", WebSocket);

const pythonCode = `
import asyncio
import sys
from hypha_rpc import connect_to_server

async def start_server(server_url):
    try:
        print(f"Attempting to connect to server at {server_url}...")
        server = await connect_to_server({"server_url": server_url})
        print("Successfully connected to server!")

        def hello(name):
            print("Hello " + name)
            return "Hello " + name

        print("Registering service...")
        svc = await server.register_service({
            "name": "Hello World",
            "id": "hello-world",
            "config": {
                "visibility": "public"
            },
            "hello": hello
        })

        print(f"Hello world service registered at workspace: {server.config.workspace}, id: {svc.id}")
        print(f'You can use this service using the service id: {svc.id}')
        print(f"You can also test the service via the HTTP proxy: {server_url}/{server.config.workspace}/services/{svc.id.split('/')[1]}/hello?name=John")

        print("Now running server...")
        await server.serve()
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        raise

async def main():
    server_url = "https://hypha.aicell.io"
    await start_server(server_url)

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