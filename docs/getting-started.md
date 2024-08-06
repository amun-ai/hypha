# Getting Started

## Installation

To install the Hypha package, run the following command:

```bash
pip install -U "hypha>=0.20"
```

If you need full support for server-side browser applications, use the following command instead:

```bash
pip install -U "hypha[server-apps]>=0.20"
playwright install
```

## Starting the Server

To start the Hypha server, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527
```

If you want to enable server apps (browsers running on the server side), run the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 --enable-server-apps
```

You can test if the server is running by visiting [http://localhost:9527](http://localhost:9527) and checking the Hypha server version.

Alternatively, you can use our public testing server at [https://ai.imjoy.io](https://ai.imjoy.io).

## Serving Static Files

If you want to serve static files (e.g., HTML, JS, CSS) for your web applications, you can mount additional directories using the `--static-mounts` argument. This allows you to specify the mount path and the local directory.

To serve static files from the `./webtools/` directory at the path `/tools` on your server, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527 --static-mounts /tools:./webtools/
```

You can mount multiple directories by providing additional `--static-mounts` arguments. For example, to mount the `./images/` directory at `/images`, use the following command:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527 --static-mounts /tools:./webtools/ /images:./images/
```

After running the command, you can access files from these directories via the Hypha server at `http://localhost:9527/tools` and `http://localhost:9527/images`, respectively.

## Connecting from a Client

Hypha provides native support for Python and JavaScript clients. For other languages, you can use the built-in HTTP proxy of Hypha (see details in a later section about "Login and Using Services from the HTTP proxy").

Ensure that the server is running, and you can connect to it using the `hypha` module under `hypha-rpc` in a client script. You can either register a service or use an existing service.

**WARNING: If you use hypha older than 0.15.x or lower, you will need to use `imjoy-rpc` instead of `hypha-rpc`.**

### Registering a Service

To register a service in Python, install the `hypha-rpc` library:

```bash
pip install hypha-rpc
```

The following code registers a service called "Hello World" with the ID "hello-world" and a function called `hello()`. The function takes a single argument, `name`, and prints "Hello" followed by the name. The function also returns a string containing "Hello" followed by the name.

We provide two versions of the code: an asynchronous version for native CPython or Pyodide-based Python in the browser (without thread support), and a synchronous version for native Python with thread support (more details about the [synchronous wrapper](/hypha-rpc?id=synchronous-wrapper)):

<!-- tabs:start -->
#### ** Asynchronous Worker **

```python
import asyncio
from hypha_rpc import connect_to_server

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
    server_url = "http://localhost:9527"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()
```

#### ** Synchronous Worker **

```python
from hypha_rpc.sync import connect_to_server

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
    server_url = "http://localhost:9527"
    start_server(server_url)
```
<!-- tabs:end -->

Run the server via `python hello-world-worker.py`.

Note: You don't need to run the client script on the same server. If you want to connect to the server from another computer, make sure to change the `server_url` to an URL with the external IP or domain name of the server.

### Using the Service

If you keep the Python service running, you can connect to it from either a Python client or a JavaScript client on the same or a different host.

#### Python Client

Install the `hypha-rpc` library:

```bash
pip install hypha-rpc
```

Use the following code to connect to the server and access the service. The code first connects to the server and then gets the service by its ID. The service can then be used like a normal Python object.

Similarily, you can also use the `connect_to_server_sync` function to connect to the server synchronously.

<!-- tabs:start -->
#### ** Asynchronous Client **

```python
import asyncio
from hypha_rpc import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "http://localhost:9527"})

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
from hypha_rpc.sync import connect_to_server

def main():
    server = connect_to_server({"server_url": "http://localhost:9527"})

    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = server.get_service("hello-world")
    ret = svc.hello("John")
    print(ret)

if __name__ == "__main__":
    main()
```
<!-- tabs:end -->

**NOTE: In Python, the recommended way to interact with the server to use asynchronous functions with `asyncio`. However, if you need to use synchronous functions, you can use `from hypha_rpc.sync import login, connect_to_server` instead. The have the exact same arguments as the asynchronous versions. For more information, see [Synchronous Wrapper](/hypha-rpc?id=synchronous-wrapper)**

#### JavaScript Client

Include the following script in your HTML file to load the `hypha-rpc` client:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.13/dist/hypha-rpc-websocket.min.js"></script>
```

Use the following code in JavaScript to connect to the server and access an existing service:

```javascript
async function main(){
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "http://localhost:9527"})
    const svc = await server.get_service("hello-world")
    const ret = await svc.hello("John")
    console.log(ret)
}
```

### Peer-to-Peer Connection via WebRTC

By default all the clients connected to Hypha server communicate via the websocket connection or the HTTP proxy. This is suitable for most use cases which involves lightweight data exchange. However, if you need to transfer large data or perform real-time communication, you can use the WebRTC connection between clients. With hypha-rpc, you can easily create a WebRTC connection between two clients easily. See the [WebRTC support](/hypha-rpc?id=peer-to-peer-connection-via-webrtc) for more details.

### User Login and Token-Based Authentication

To access the full features of the Hypha server, users need to log in and obtain a token for authentication. The new `login()` function provides a convenient way to display a login URL, once the user click it and login, it can then return the token for connecting to the server.

Here is an example of how the login process works using the `login()` function:

<!-- tabs:start -->
#### ** Asynchronous Client **

```python
from hypha_rpc import login, connect_to_server

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
from hypha_rpc.sync import login, connect_to_server

token = login({"server_url": "https://ai.imjoy.io"})
server = connect_to_server({"server_url": "https://ai.imjoy.io", "token": token})

# ...use the server api...
```
<!-- tabs:end -->

Login in javascript:
```javascript
async function main(){
    const token = await hyphaWebsocketClient.login({"server_url": "http://localhost:9527"})
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "http://localhost:9527", "token": token})
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

For example, here is a callback function for displaying the login URL, or launching a browser for the user to login:

```python

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
    print(f"Go to login url: {context['login_url']}")
```

### User Authentication and Service Authorization

Hypha provide built-in support for user authentication, and based on the user context, you can also customize the service authorization within each service function.

In the previous example, we registered a public service (`config.visibility = "public"`) that can be accessed by any client. If you want to limit service access to a subset of clients, there are two ways to provide authorization.

1. Connecting to the Same Workspace: Set `config.visibility` to `"private"`. Authorization is achieved by generating a token from the client that registered the service (using `server.config.workspace` and `server.generate_token()`). Another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).

2. Using User Context: When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (or `"private"` to limit access for clients from the same workspace). Each service function needs to accept a keyword argument called `context`. The server will provide the context information containing `user` for each service function call. The service function can then check whether `context.user["id"]` is allowed to access the service. On the client side, you need to log in and generate a token by calling the `login({"server_url": xxxx})` function. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.

Here is an example for how to enable the user context in the service function:
```python

async def start_service(server):
    """Hypha startup function."""

    def test(x, context=None): # Note that the context argument is added
        current_user = context["user"]
        if current_user["email"] not in authorized_users:
            raise Exception(f"User {current_user['email']} is not authorized to access this service.")
        print(f"Test: {x}")

    # Register a test service
    await server.register_service(
        {
            "id": "test-service",
            "config": {
                "visibility": "public",
                "require_context": True, # enable user context
            },
            "test": test,
        }
    )
```

By default, hypha server uses a user authentication system based on [Auth0](https://auth0.com) controlled by us. You can also setup your own auth0 account to use it with your own hypha server. See [Setup Authentication](./setup-authentication) for more details.

### Login and Using Services from the HTTP proxy

#### Obtain login token via http requests
For clients other than Python or javascript (without hypha-rpc) support, you can use Hypha server's built-in HTTP proxy for services to obtain the token. Here is an example of how to obtain the token via HTTP requests:

First, let's initiate the login process by calling the `start` function of the `hypha-login` service:
```bash
curl -X POST "https://ai.imjoy.io/public/services/hypha-login/start" -H "Content-Type: application/json" -d '{}'
```
It will return something like:
```json
{"login_url":"https://ai.imjoy.io/public/apps/hypha-login/?key=mihDumpHGYxkPdSEKB7GgM","key":"mihDumpHGYxkPdSEKB7GgM","report_url":"https://ai.imjoy.io/public/services/hypha-login/report"}
```
Now you can print the `login_url` to the console or open it in a browser to login.

Immediately after displaying the url, you can call `check` to wait for the user to login and get the token, for example:
```bash
curl -X POST "https://ai.imjoy.io/public/services/hypha-login/check" -H "Content-Type: application/json" -d '{"key":"mihDumpHGYxkPdSEKB7GgM", "timeout": 10}'
```

For the above request:
 - replace `mihDumpHGYxkPdSEKB7GgM` with the actual key you obtained from the `start` function
 - and you can adjust the `timeout` to wait for the user to login, the above example will wait for 10 seconds.

This should wait and return the token if the user has successfully logged in, otherwise, it will return a timeout error.

For details, see the python implementation of the login function [here](https://github.com/imjoy-team/imjoy-rpc/blob/master/python/hypha_rpc/hypha/websocket_client.py#L175).

#### Using the Token in Service Requests

With the token, you can now request any protected service by passing the `token` as `Authorization: Bearer <token>` in the header, for example:
```bash
curl -X POST "https://ai.imjoy.io/public/services/hello-world/hello" -H "Content-Type: application/json" -H "Authorization : Bearer <token>" -d '{"name": "John"}'
```
For more details, see the service request api endpoint [here](https://ai.imjoy.io/api-docs#/default/service_function__workspace__services__service___keys__post).

### Custom Initialization and Service Integration with Hypha Server

Hypha's flexibility allows services to be registered from scripts running on the same host as the server or on a different one. To further accommodate complex applications, Hypha supports the initiation of "built-in" services in conjunction with server startup. This can be achieved using the `--startup-functions` option.

The `--startup-functions` option allows you to provide a URI pointing to a Python function intended for custom

 server initialization tasks. The specified function can perform various tasks, such as registering services, configuring the server, or launching additional processes. The URI should follow the format `<python module or script file>:<entrypoint function name>`, providing a straightforward way to customize your server's startup behavior.

For example, to start the server with a custom startup function, use the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 --startup-functions=./example-startup-function.py:hypha_startup
```

Here's an example of `example-startup-function.py`:

```python
"""Example startup function file for Hypha."""

async def hypha_startup(server):
    """Hypha startup function."""

    def test(x):
        print(f"Test: {x}")

    # Register a test service
    await server.register_service(
        {
            "id": "test-service",
            "config": {
                "visibility": "public",
                "require_context": False,
            },
            "test": test,
        }
    )
```

Note that the startup function file will be loaded as a Python module. You can also specify an installed Python module by using the format `my_pip_module:hypha_startup`. In both cases, make sure to specify the entrypoint function name (`hypha_startup` in this case). The function should accept a single positional argument, `server`, which represents the server object used in the client script.

Multiple startup functions can be specified by providing additional `--startup-functions` arguments. For example, to specify two startup functions, use the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 --startup-functions=./example-startup-function.py:hypha_startup ./example-startup-function2.py:hypha_startup
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
