
## Using hypha in Pyodide
**This is an experimental feature**

Pyodide is a Python runtime for the web, which allows you to run Python code in the browser. Experimentally, you can use Hypha in Pyodide to run web applications in the browser, e.g. running hypha server in a service worker, and create a browser-based hypha-app-engine with many browser tabs (served from the same origin).

```python
import micropip

await micropip.install(['hypha', 'hypha-rpc'])

import asyncio
from hypha_rpc import login, connect_to_server

from hypha.server import get_argparser, create_application

try:
    # patch http libraries for pyodide
    import pyodide_http

    pyodide_http.patch_all()  # Patch all libraries
except ImportError:
    pass


arg_parser = get_argparser()
opt = arg_parser.parse_args()

# create an FastAPI application
app = create_application(opt)

# Let's use the public server to serve the fastapi application
SERVER_URL = "https://ai.imjoy.io"
async def show_login_url(context):
    print("Please visit the following the link or scan the QR code to login")
    print(context['login_url'])

token = await login({"server_url": SERVER_URL, "login_callback": show_login_url})
print("Successfully logged in!")

server = await connect_to_server(
    {"server_url": SERVER_URL, "token": token}
)
workspace = server.config["workspace"]

async def serve_fastapi(args):
    scope = args["scope"]
    print(f'{scope["client"]} - {scope["method"]} - {scope["path"]}')
    await app(args["scope"], args["receive"], args["send"])

await server.register_service({
    "id": "demo-hypha-server",
    "name": "Demo Hypha Server",
    "description": "Serve a demo hypha server in side the browser",
    "type": "asgi",
    "serve": serve_fastapi,
    "config":{
        "visibility": "public"
    }
})
print(f"Server app running at {server.config.public_base_url}/{workspace}/apps/demo-hypha-server")
```