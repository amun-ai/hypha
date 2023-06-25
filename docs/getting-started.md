
# Getting Started

### Installation

Run the following command:
```
pip install -U hypha
```

If you want full support with server-side browser applications, run the following command instead:
```
pip install -U hypha[server-apps]
playwright install
```

### Start the server

Start the hypha server with the following command:
```
python3 -m hypha.server --host=0.0.0.0 --port=9000
```


If you want to enable server apps (i.e. browsers running on the server side), run:

```
python -m hypha.server --host=0.0.0.0 --port=9000 --enable-server-apps
```


To test it, you should be able to visit http://localhost:9000 and see the version of the hypha server.

In addition to run your own server, you can also use our public testing server: https://ai.imjoy.io

### Serve static files

Sometimes, it is useful to serve static files (e.g. HTML, JS, CSS etc.) for your own web applications. You can mount additional directories for serving static files by using the `--static-mounts` argument. This requires you to specify the mount path and the local directory. For example, to serve static files from the directory `./webtools/` at the path `/tools` on your server, you can use the following command:

```
python3 -m hypha.server --host=0.0.0.0 --port=9000 --static-mounts /tools:./webtools/
```

Multiple directories can be mounted by providing additional `--static-mounts` arguments. For instance, to mount an additional directory `./images/` at `/images`, you would use:

```
python3 -m hypha.server --host=0.0.0.0 --port=9000 --static-mounts /tools:./webtools/ --static-mounts /images:./images/
```

After running the command, you should be able to access files from these directories via your hypha server at `http://localhost:9000/tools` and `http://localhost:9000/images` respectively.

### Connect from a client

We currently provide native support for both Python and Javascript client, for other languages, you can use access services using the built-in HTTP proxy of Hypha.

Keep the above server running, and now you can connect to it with the `hypha` module under `imjoy-rpc` in a client script. You can either register a service or use an existing service.

#### Register a service

In Python, you can install the `imjoy-rpc` library:

```
pip install imjoy-rpc
```

Here is a complete client example in Python, you can save the following content as `hello-world-worker.py` and start the server via `python hello-world-worker.py`:

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
    
    print(f"hello world service regisered at workspace: {server.config.workspace}")
    print(f"Test it with the http proxy: {server_url}/{server.config.workspace}/services/hello-world/hello?name=John")

if __name__ == "__main__":
    server_url = "http://localhost:9000"
    loop = asyncio.get_event_loop()
    loop.create_task(start_server(server_url))
    loop.run_forever()
```


You don't need to run the client script on the same server, just make sure you change the corresponding `server_url` (to an URL with the external ip or domain name of the server) if you try to connect to the server from another computer.

#### Using the service

If you keep the above python service running, you can also connect from either a Python client or Javascript client (on the same or a different host):

In Python:
```
pip install imjoy-rpc
```

```python
import asyncio
from imjoy_rpc.hypha import connect_to_server

async def main():
    server = await connect_to_server({"server_url":  "http://localhost:9000"})

    # get an existing service
    # since hello-world is registered as a public service, we can access it with only the name "hello-world"
    svc = await server.get_service("hello-world")
    ret = await svc.hello("John")
    print(ret)

asyncio.run(main())
```

In Javascript:

Make sure you load the imjoy-rpc client:
```html
<script src="https://cdn.jsdelivr.net/npm/imjoy-rpc@0.5.6/dist/hypha-rpc-websocket.min.js"></script>
```

Then in a javascript you can do:
```javascript
async function main(){
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "http://localhost:9000"})
    const svc = await server.getService("hello-world")
    const ret = await svc.hello("John")
    console.log(ret)
}
```

### Service authorization

In the above example, we registered a public service (`config.visibility = "public"`) which can be access by any clients. There are two ways for providing authorization if you want to limit the service access to a subset of the client.

 1. Connecting to the same workspace. In this case, we can set `config.visibility` to `"private"`, the authorization is achived by generating a token from the client which registered the service (via `server.config.workspace` and `server.generate_token()`), and another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).
 2. Using user context. When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (you can also set `config.visibility` to `"private"` if yo want to limit the access for clients from the same workspace). Each of the service functions will need to accept a keyword argument called `context`. For each service function call the server will be responsible to providing the context information containing `user`. Each service function can then check whether the `context.user.id` is allowed to access the service. On the client which uses the service, it need to login and generate a token from https://ai.imjoy.io/apps/built-in/account-manager.html. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.
 

### Custom initialization and service integration with hypha server

Hypha's flexibility allows for services to be registered from scripts running either on the same host as the server or a different one. To further accommodate complex applications, Hypha supports the initiation of "built-in" services in tandem with server startup. This is achieved through the `--startup-function-uri` option. 

This command-line argument enables users to provide a URI pointing to a Python function intended for custom server initialization tasks. The specified function can conduct a variety of tasks such as registering services, configuring the server, or even launching additional processes. The URI should adhere to the format `<python module or script file>:<entrypoint function name>`, providing a direct and straightforward way to customize your server's startup behavior.

```bash
python -m hypha.server --host=0.0.0.0 --port=9000 --startup-function-uri=./example-startup-function.py:hypha_startup
```

Here's an example of `example-startup-function.py`:

```python
"""Example startup function file for Hypha."""

async def hypha_startup(server):
    """Hypha startup function."""

    # The server object passed into this function is identical to the one in the client script.
    # You can register more functions or call other functions using this server object.
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

Note that the startup function file will be loaded as a Python module, but you can also specify an installed python module e.g. `my_pip_module:hypha_startup`. In both cases, don't forget to specify the entrypoint function name (`hypha_startup` in this case). This function should accept a single positional argument, `server`, which is the server object, the same as the one used in the client script.

#### Launching External Services Using Commands

Sometimes, the services you want to start with your server may not be written in Python or might require a different Python environment from your Hypha server. For instance, you might want to register a service written in Javascript.

In these situations, we offer a utility function, `launch_external_services`, which is available in the [Hypha utils module](../hypha/utils.py). This function enables you to launch external services from within your startup function.

Consider the following example (which can be used in a startup function initiated with the `--startup-function-uri` option):

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

In this snippet, `launch_external_services` initiates an external service defined in `./tests/example_service_script.py`. The command string uses placeholders like `{server_url}`, `{workspace}`, and `{token}`, which the utility function automatically replaces with their actual values during execution.

You can find the contents of the `./tests/example_service_script.py` script [here](../tests/example_service_script.py).

By using `launch_external_services`, you can seamlessly integrate external services into your Hypha server, regardless of the programming language or Python environment they utilize.