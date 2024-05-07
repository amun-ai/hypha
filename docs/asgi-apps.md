## Serve ASGI Web Applications

ASGI is a standard to create async web applications. With hypha, you can register a service which serves an ASGI application, e.g. a FastAPI app which serve a web page, or a web API.

The following example shows how to use FastAPI to create a web application in the browser, e.g. https://imjoy-notebook.netlify.app/, and serve it through Hypha.

```python
import asyncio
from imjoy_rpc.hypha import connect_to_server

import micropip

# This assumes you are in a Pyodide environment
await micropip.install(["fastapi==0.70.0", "pyotritonclient==0.1.37", "numpy", "imageio", "Pillow", "matplotlib"])

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Connect to the Hypha server
server_url = "https://ai.imjoy.io"
server = await connect_to_server({"server_url": server_url})

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <html>
        <head>
            <title>Cat</title>
        </head>
        <body>
            <img src="https://cataas.com/cat?type=square" alt="cat">
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/api/v1/test")
async def test():
    return {"message": "Hello, it works!", "server_url": server_url}

async def serve_fastapi(args):
    scope = args["scope"]
    print(f'{scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

svc_info = await server.register_service({
    "id": "cat",
    "name": "cat",
    "type": "ASGI",
    "serve": serve_fastapi,
    "config":{
        "visibility": "public"
    }
})

print(f"Test it with the HTTP proxy: {server_url}/{server.config.workspace}/apps/{svc_info['id'].split(':')[1]}")
```

This will create a web page which you can view in the browser, and also an API endpoint at `/api/v1/test`.
