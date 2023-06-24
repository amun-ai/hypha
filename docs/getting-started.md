
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

### Service Authorization

In the above example, we registered a public service (`config.visibility = "public"`) which can be access by any clients. There are two ways for providing authorization if you want to limit the service access to a subset of the client.

 1. Connecting to the same workspace. In this case, we can set `config.visibility` to `"private"`, the authorization is achived by generating a token from the client which registered the service (via `server.config.workspace` and `server.generate_token()`), and another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).
 2. Using user context. When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (you can also set `config.visibility` to `"private"` if yo want to limit the access for clients from the same workspace). Each of the service functions will need to accept a keyword argument called `context`. For each service function call the server will be responsible to providing the context information containing `user`. Each service function can then check whether the `context.user.id` is allowed to access the service. On the client which uses the service, it need to login and generate a token from https://ai.imjoy.io/apps/built-in/account-manager.html. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.
 

### Start services with server

Services can be registered from scripts running from the same or different host of the server. In many applications, it is useful to provide "builit-in" services which can be started together with the server. We provide the `--services-config` option to specify a service config yaml file:
```
python -m hypha.server --host=0.0.0.0 --port=9000 --services-config=./services_config.yaml
```

Here is an exmaple of the `services_config.yaml`:
```yaml
format_version: 1
services:
    - module: ./tests/external_services.py
      entrypoint: register_services
      kwargs:
        service_id: "internal-test-service"
    - command: python ./tests/external_services.py --server-url={server_url} --service-id=external-test-service --workspace={workspace} --token={token}
      workspace: "public"
      check_services:
        - external-test-service
```

There are two services in the above example for two types of services provided via a python module and command.

#### Specify service via Python module
It contains the following fields:
 * `module`: The python module, a python file path to be loaded.
 * `entrypoint`: The entrypoint function name, the function should accept a single positional argument `server` which is the server object, additional keyword arguments can be provided via `kwargs`.
 * `kwargs`: Additional keyword arguments to be passed to the entrypoint function.

With the above fields, you can provide a python file which contains an entrypoint function, define a function where you can register_services to the server. For example, you can save the following content as `./external_services.py`:

```python
async def register_services(server, **kwargs):
    """Register the services."""

    # The server object is the same as the one in the client script
    # You can register more functions or call other functions in the server object
    await server.register_service(
        {
            "id": kwargs["service_id"],
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "test": lambda x: print(f"Test: {x}"),
        }
    )
```

#### Specify service via command
It contains the following fields:
 * `command`: The command to be executed, it can be a list of strings or a string. You can also provide a `command` with a command string, the command string can contain the following variables:
    - `{server_url}`: The server url
    - `{workspace}`: The workspace, if the workspace is not specified, it will be "public", otherwise, it will be the workspace id specified in the config below.
    - `{token}`: The token
 * `workspace`: The workspace to be used, if not specified, it will be "public".
 * `check_services`: A list of service ids to be checked after starting the command, if any of the service is not available, the server will be stopped.

For example, one can write a hypha client script which can be started together with the server. For instance, you can save the following content as `./external_services.py`:

```python
import argparse
import asyncio
import logging

from imjoy_rpc.hypha import connect_to_server

async def start_service(server_url, service_id, workspace=None, token=None):
    """Start the service."""
    client_id = service_id + "-client"
    print(f"Starting service...")
    server = await connect_to_server(
        {
            "client_id": client_id,
            "server_url": server_url,
            "workspace": workspace,
            "token": token,
        }
    )
    await server.register_service(
        {
            "id": service_id,
            "config": {
                "visibility": "public",
                "require_context": True,
            },
            "test": lambda x: print(f"Test: {x}"),
        }
    )
    print(
        f"Service (client_id={client_id}, service_id={service_id}) started successfully, available at {server_url}/{server.config.workspace}/services"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test services")
    parser.add_argument(
        "--server-url", type=str, default="https://ai.imjoy.io/", help="The server url"
    )
    parser.add_argument(
        "--service-id", type=str, default="test-service", help="The service id"
    )
    parser.add_argument(
        "--workspace", type=str, default=None, help="The workspace name"
    )
    parser.add_argument("--token", type=str, default=None, help="The token")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    loop = asyncio.get_event_loop()
    loop.create_task(
        start_service(
            args.server_url,
            args.service_id,
            workspace=args.workspace,
            token=args.token,
        )
    )
    loop.run_forever()
```

Then you can add the following service config to the `services_config.yaml`:

```yaml
format_version: 1
services:
    - command: python ./tests/external_services.py --server-url={server_url} --service-id=external-test-service --workspace={workspace} --token={token}
      workspace: "public"
      check_services:
        - external-test-service
```


