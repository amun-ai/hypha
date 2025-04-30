// A static file server implementation with Hypha in Deno
// This script serves files from the mounted directory at /home/pyodide
// Run with:
// deno run --allow-net --allow-read --allow-write --allow-env scripts/deno-http-server.js
// To compile to a binary:
// deno compile --allow-net --allow-read --allow-write --allow-env scripts/deno-http-server.js

import pyodideModule from "npm:pyodide/pyodide.js";

const pyodide = await pyodideModule.loadPyodide();
await pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, { root: "." }, "/home/pyodide")
await pyodide.loadPackage("ssl")
// Install micropip for installing Python packages
await pyodide.loadPackage("micropip");
const micropip = pyodide.pyimport("micropip");

// Install required packages
await micropip.install(['fastapi==0.70.0', 'hypha-rpc']);

// Add WebSocket to Python global scope
pyodide.globals.set("WebSocket", WebSocket);

const pythonCode = `
import os
import asyncio
import sys
import traceback
from hypha_rpc import connect_to_server
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
import mimetypes
from datetime import datetime
from pathlib import Path

# Create FastAPI app
app = FastAPI(title="Static File Server")

def format_size(size):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def generate_directory_listing(path: str, files: list):
    """Generate an HTML page for directory listing"""
    full_path = os.path.join("/home/pyodide", path)
    items = []
    
    # Add parent directory link if not at root
    if path != "":
        items.append({
            'name': '..',
            'path': '#',  # We'll handle this with JavaScript
            'is_dir': True,
            'size': '',
            'modified': ''
        })
    
    # Add all files and directories
    for name in sorted(os.listdir(full_path)):
        file_path = os.path.join(full_path, name)
        stat = os.stat(file_path)
        items.append({
            'name': name,
            'path': os.path.join(path, name) if path else name,
            'is_dir': os.path.isdir(file_path),
            'size': format_size(stat.st_size) if not os.path.isdir(file_path) else '',
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Directory listing for /{path}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            a {{ text-decoration: none; color: #0366d6; cursor: pointer; }}
            .directory {{ color: #6f42c1; }}
        </style>
        <script>
            function goToParentDirectory() {{
                const currentPath = window.location.pathname;
                // Remove trailing slash if exists
                const normalizedPath = currentPath.endsWith('/') ? currentPath.slice(0, -1) : currentPath;
                // Get parent path
                const parentPath = normalizedPath.substring(0, normalizedPath.lastIndexOf('/'));
                // If we're already at root, stay there
                const targetPath = parentPath || '/';
                window.location.href = targetPath;
            }}
        </script>
    </head>
    <body>
        <h1>Directory listing for /{path}</h1>
        <table>
            <tr>
                <th>Name</th>
                <th>Size</th>
                <th>Last Modified</th>
            </tr>
    """
    
    for item in items:
        name_display = item['name'] + ('/' if item['is_dir'] else '')
        css_class = 'directory' if item['is_dir'] else ''
        if item['name'] == '..':
            html += f"""
                <tr>
                    <td><a onclick="goToParentDirectory()" class="{css_class}">{name_display}</a></td>
                    <td>{item['size']}</td>
                    <td>{item['modified']}</td>
                </tr>
            """
        else:
            html += f"""
                <tr>
                    <td><a href="{item['path']}" class="{css_class}">{name_display}</a></td>
                    <td>{item['size']}</td>
                    <td>{item['modified']}</td>
                </tr>
            """
    
    html += """
        </table>
    </body>
    </html>
    """
    return html

@app.get("/{path:path}")
async def serve_static(path: str):
    try:
        # Normalize the path to prevent directory traversal
        normalized_path = os.path.normpath(path)
        if normalized_path.startswith(".."):
            return Response(status_code=403, content="Access denied")
            
        full_path = os.path.join("/home/pyodide", normalized_path)
        
        if not os.path.exists(full_path):
            return Response(status_code=404, content="File not found")
            
        if os.path.isdir(full_path):
            return HTMLResponse(generate_directory_listing(path, os.listdir(full_path)))
        else:
            # Guess the mime type for the file
            mime_type, _ = mimetypes.guess_type(full_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
                
            with open(full_path, "rb") as f:
                return Response(content=f.read(), media_type=mime_type)
    except Exception as e:
        print(f"Error serving {path}: {str(e)}")
        return Response(status_code=500, content="Internal server error")

async def serve_fastapi(args, context=None):
    scope = args["scope"]
    print(f'{context["user"]["id"]} - {scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

async def main():
    try:
        print("Connecting to Hypha server...")
        server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
        print("Successfully connected!")

        print("Registering static file server service...")
        svc_info = await server.register_service({
            "id": "files",
            "name": "Static File Server",
            "type": "asgi",
            "serve": serve_fastapi,
            "config": {
                "visibility": "public",
                "require_context": True
            }
        })
        
        service_id = svc_info['id'].split(':')[1]
        print(f"Service registered successfully!")
        print(f"Access your files at: {server.config.public_base_url}/{server.config.workspace}/apps/{service_id}/")
        print("Serving files from /home/pyodide...")
        await server.serve()
    except Exception as e:
        print(f"Error occurred: {str(e)}", file=sys.stderr)
        traceback.print_exc()
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