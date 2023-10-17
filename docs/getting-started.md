# Getting Started

## Installation

To install the Hypha package, run the following command:

```bash
pip install -U hypha
```

If you need full support for server-side browser applications, use the following command instead:

```bash
pip install -U hypha[server-apps]
playwright install
```

## Starting the Server

To start the Hypha server, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9000
```

If you want to enable server apps (browsers running on the server side), run the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9000 --enable-server-apps
```

You can test if the server is running by visiting [http://localhost:9000](http://localhost:9000) and checking the Hypha server version.

Alternatively, you can use our public testing server at [https://ai.imjoy.io](https://ai.imjoy.io).

## Serving Static Files

If you want to serve static files (e.g., HTML, JS, CSS) for your web applications, you can mount additional directories using the `--static-mounts` argument. This allows you to specify the mount path and the local directory.

To serve static files from the `./webtools/` directory at the path `/tools` on your server, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9000 --static-mounts /tools:./webtools/
```

You can mount multiple directories by providing additional `--static-mounts` arguments. For example, to mount the `./images/` directory at `/images`, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9000 --static-mounts /tools:./webtools/ /images:./images/
```

After running the command, you can access files from these directories via the Hypha server at `http://localhost:9000/tools` and `http://localhost:9000/images`, respectively.

## Connecting from a Client

Hypha provides native support for Python and JavaScript clients. For other languages, you can use the built-in HTTP proxy of Hypha.

Ensure that the server is running, and you can connect to it using the `hypha` module under `imjoy-rpc` in a client script. You can either register a service or use an existing service.

### Registering a Service

To register a service in Python, install the `imjoy-rpc` library:

```bash
pip install imjoy-rpc
```

The following code registers a service called "Hello World" with the ID "hello-world" and a function called `hello()`. The function takes a single argument, `name`, and prints "Hello" followed by the name. The function also returns a string containing "Hello" followed by the name.

We provide two versions of the code: an asynchronous version for native CPython or Pyodide-based Python in the browser (without thread support), and a synchronous version for native Python with thread support (more details about the [synchronous wrapper](/imjoy-rpc?id=synchronous-wrapper)):

<!-- tabs:start -->
#### ** Asynchronous Worker **

```python
import asyncio
from imjoy_rpc.hypha import connect_to_server

async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})
    
    def hello(name):
        print("Hello " + name)
        return "Hello " + name

    await server.register_service({
        "name": "Hello World",
        "id": "hello-world",
        "config": {
            "visibility": "public"
        },
        "hello": hello
    })
    
    print(f"Hello world service registered at workspace: {server.config.workspace}")
    print(f"Test it with the HTTP proxy: {server_url}/{server.config.workspace}/services/hello-world/hello?name=John")

if __name__ == "__main__":
    server_url = "http://localhost:9000"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()
```

#### ** Synchronous Worker **

```python
from imjoy_rpc.hypha.sync import connect_to_server

def start_server(server_url):
    server = connect_to_server({"server_url": server_url})
    
    def hello(name):
        print("Hello " + name)
        return "Hello " + name

    server.register_service({
        "name": "Hello World",
        "id": "hello-world",
        "config": {
            "visibility": "public"
        },
        "hello": hello
    })
    
    print(f"Hello world service registered at workspace: {server.config.workspace}")
    print(f"Test it with the HTTP proxy: {server_url}/{server.config.workspace}/services/hello-world/hello?name=John")

if __name__ == "__main__":
    server_url = "http://localhost:9000"
    start_server(server_url)
```
<!-- tabs:end -->

Run the server via `python hello-world-worker.py`.

Note: You don't need to run the client script on the same server. If you want to connect to the server from another computer, make sure to change the `server_url` to an URL with the external IP or domain name of the server.

### Using the Service

If you keep the Python service running, you can connect to it from either a Python client or a JavaScript client on the same or a different host.

#### Python Client

Install the `imjoy-rpc` library:

```bash
pip install imjoy-rpc
```

Use the following code to connect to the server and access the service. The code first connects to the server and then gets the service by its ID. The service can then be used like a normal Python object.

Similarily, you can also use the `connect_to_server_sync` function to connect to the server synchronously (available since `imjoy-rpc>=0.5.25.post0`).

<!-- tabs:start -->
#### ** Asynchronous Client **

```python
import asyncio
from imjoy_rpc.hypha import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "http://localhost:9000"})

    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = await server.get_service("hello-world")
    ret = await svc.hello("John")
    print(ret)
if __name__ == "__main__":
    asyncio.run(main())
```

#### ** Synchronous Client **

```python
import asyncio
from imjoy_rpc.hypha.sync import connect_to_server

def main():
    server = connect_to_server({"server_url": "http://localhost:9000"})

    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = server.get_service("hello-world")
    ret = svc.hello("John")
    print(ret)

if __name__ == "__main__":
    main()
```
<!-- tabs:end -->

**NOTE: In Python, the recommended way to interact with the server to use asynchronous functions with `asyncio`. However, if you need to use synchronous functions, you can use `from imjoy_rpc.hypha.sync import login, connect_to_server` (available since `imjoy-rpc>=0.5.25.post0`) instead. The have the exact same arguments as the asynchronous versions. For more information, see [Synchronous Wrapper](/#/imjoy-rpc?id=synchronous-wrapper)**

#### JavaScript Client

Include the following script in your HTML file to load the `imjoy-rpc` client:

```html
<script src="https://cdn.jsdelivr.net/npm/imjoy-rpc@0.5.6/dist/hypha-rpc-websocket.min.js"></script>
```

Use the following code in JavaScript to connect to the server and access an existing service:

```javascript
async function main(){
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "http://localhost:9000"})
    const svc = await server.getService("hello-world")
    const ret = await svc.hello("John")
    console.log(ret)
}
```

### Peer-to-Peer Connection via WebRTC

By default all the clients connected to Hypha server communicate via the websocket connection or the HTTP proxy. This is suitable for most use cases which involves lightweight data exchange. However, if you need to transfer large data or perform real-time communication, you can use the WebRTC connection between clients. With imjoy-rpc, you can easily create a WebRTC connection between two clients easily. See the [WebRTC support in ImJoy RPC V2](/#/imjoy-rpc?id=peer-to-peer-connection-via-webrtc) for more details.

### User Login and Token-Based Authentication

To access the full features of the Hypha server, users need to log in and obtain a token for authentication. The new `login()` function provides a convenient way to display a login URL, once the user click it and login, it can then return the token for connecting to the server.

Here is an example of how the login process works using the `login()` function:

<!-- tabs:start -->
#### ** Asynchronous Client **

```python
from imjoy_rpc.hypha import login, connect_to_server

token = await login({"server_url": "https://ai.imjoy.io"})
# A login URL will be printed to the console
# The user needs to open the URL in a browser and log in
# Once the user logs in, the login function will return
# with the token for connecting to the server
server = await connect_to_server({"server_url": "https://ai.imjoy.io", "token": token})
# ...use the server api...
```

#### ** Synchronous Client **

```python
from imjoy_rpc.hypha.sync import login, connect_to_server

token = login({"server_url": "https://ai.imjoy.io"})
server = connect_to_server({"server_url": "https://ai.imjoy.io", "token": token})

# ...use the server api...
```
<!-- tabs:end -->

Login in javascript:
```javascript
async function main(){
    const token = await hyphaWebsocketClient.login({"server_url": "http://localhost:9000"})
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "http://localhost:9000", "token": token})
    // ... use the server...
}
```

The output will provide a URL for the user to open in their browser and

 perform the login process. Once the user clicks the link and successfully logs in, the `login()` function will return, providing the token.

The `login()` function also supports additional arguments:

```python
token = await login(
    {
        "server_url": SERVER_URL,
        "login_callback": login_callback,
        "login_timeout": 3,
    }
)
```

If no `login_callback` is passed, the login URL will be printed to the console. You can also pass a callback function as `login_callback` to perform custom actions during the login process.

For example, here is a callback function for displaying the login URL, QR code to print the QR code to the console), or launching a browser for the user to log in:

```python

# Require `pip install qrcode[pil]`
from hypha.utils.qrcode import display_qrcode

async def callback(context):
    """
    Callback function for login.
    This function is used for display the login URL,
    Or launch the browser, display a QR code etc. for the user to login
    Once done, it should return;

    The context is a dictionary contains the following keys:
     - login_url: the login URL
     - report_url: the report URL
     - key: the key for the login
    """
    print(f"By passing login: {context['login_url']}")
    # Show the QR code for login
    display_qrcode(context["login_url"])
```

### Service Authorization

In the previous example, we registered a public service (`config.visibility = "public"`) that can be accessed by any client. If you want to limit service access to a subset of clients, there are two ways to provide authorization.

1. Connecting to the Same Workspace: Set `config.visibility` to `"private"`. Authorization is achieved by generating a token from the client that registered the service (using `server.config.workspace` and `server.generate_token()`). Another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).
2. Using User Context: When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (or `"private"` to limit access for clients from the same workspace). Each service function needs to accept a keyword argument called `context`. The server will provide the context information containing `user` for each service function call. The service function can then check whether `context.user["id"]` is allowed to access the service. On the client side, you need to log in and generate a token by calling the `login({"server_url": xxxx})` function. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.

### Custom Initialization and Service Integration with Hypha Server

Hypha's flexibility allows services to be registered from scripts running on the same host as the server or on a different one. To further accommodate complex applications, Hypha supports the initiation of "built-in" services in conjunction with server startup. This can be achieved using the `--startup-functions` option.

The `--startup-functions` option allows you to provide a URI pointing to a Python function intended for custom

 server initialization tasks. The specified function can perform various tasks, such as registering services, configuring the server, or launching additional processes. The URI should follow the format `<python module or script file>:<entrypoint function name>`, providing a straightforward way to customize your server's startup behavior.

For example, to start the server with a custom startup function, use the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9000 --startup-functions=./example-startup-function.py:hypha_startup
```

Here's an example of `example-startup-function.py`:

```python
"""Example startup function file for Hypha."""

async def hypha_startup(server):
    """Hypha startup function."""

    # Register a test service
    await server.register_service(
        {
            "id": "test-service",
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "test": lambda x: print(f"Test: {x}"),
        }
    )
```

Note that the startup function file will be loaded as a Python module. You can also specify an installed Python module by using the format `my_pip_module:hypha_startup`. In both cases, make sure to specify the entrypoint function name (`hypha_startup` in this case). The function should accept a single positional argument, `server`, which represents the server object used in the client script.

Multiple startup functions can be specified by providing additional `--startup-functions` arguments. For example, to specify two startup functions, use the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9000 --startup-functions=./example-startup-function.py:hypha_startup ./example-startup-function2.py:hypha_startup
```

#### Launching External Services Using Commands

If you need to start services written in languages other than Python or requiring a different Python environment than your Hypha server, you can use the `launch_external_services` utility function available in the `hypha.utils` module.

Here's an example of using `launch_external_services` within a startup function initiated with the `--startup-functions` option:

```python
from hypha.utils import launch_external_services

async def hypha_startup(server):
    # ...

    await launch_external_services(
        server,
        "python ./tests/example_service_script.py --server-url={server_url} --service-id=external-test-service --workspace={workspace} --token={token}",
        name="example_service_script",
        check_services=["external-test-service"],
    )
```

In this example, `launch_external_services` starts an external service defined in `./tests/example_service_script.py`. The command string uses placeholders like `{server_url}`, `{workspace}`, and `{token}`, which are automatically replaced with their actual values during execution.

By using `launch_external_services`, you can seamlessly integrate external services into your Hypha server, regardless of the programming language or Python environment they utilize.
