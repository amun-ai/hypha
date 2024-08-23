# Serve ASGI Web Applications

ASGI is a standard to create async web applications. With hypha, you can register a service which serves an ASGI application, e.g. a FastAPI app which serve a web page, or a web API.

The following example shows how to use FastAPI to create a web application in the browser, e.g. https://imjoy-notebook.netlify.app/, and serve it through Hypha.

You need to first install fastapi:

```python
# This assumes you are in a Pyodide environment, if not, you can install fastapi using `pip install fastapi==0.70.0`
import micropip
await micropip.install(["fastapi==0.70.0"])
```

Then you can create a FastAPI app and serve it through Hypha:

```python
import asyncio
from hypha_rpc import connect_to_server

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Connect to the Hypha server
server_url = "https://hypha.aicell.io"

async def main():
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
    await server.serve()

# Assuming if you are running in a Jupyter notebook
# which support top-level await, if not, you can use asyncio.run(main())
await main()
```

This will create a web page which you can view in the browser, and also an API endpoint at `/api/v1/test`.
