![PyPI](https://img.shields.io/pypi/v/imjoy.svg?style=popout)
# Hypha

Hypha is an application framework for large-scale data management and AI model serving, it allows creating computational platforms consists of computational and user interface components.

Hypha server act as a hub for connecting different components through [imjoy-rpc](https://github.com/imjoy-team/imjoy-rpc).

## Installation

Run the following command:
```
pip install -U hypha
```

If you want full support with server-side browser applications, run the following command instead:
```
pip install -U hypha[server-apps]
playwright install
```

## Usage
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
    
    print(f"hello world service regisered at workspace: {api.config.workspace}")
    print(f"Test it with the http proxy: {server_url}/{api.config.workspace}/services/hello-world/hello?name=John")

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

#### Service Authorization

In the above example, we registered a public service (`config.visibility = "public"`) which can be access by any clients. There are two ways for providing authorization if you want to limit the service access to a subset of the client.

 1. Connecting to the same workspace. In this case, we can set `config.visibility` to `"private"`, the authorization is achived by generating a token from the client which registered the service (via `server.config.workspace` and `server.generate_token()`), and another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).
 2. Using user context. When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (you can also set `config.visibility` to `"private"` if yo want to limit the access for clients from the same workspace). Each of the service functions will need to accept a keyword argument called `context`. For each service function call the server will be responsible to providing the context information containing `user`. Each service function can then check whether the `context.user.id` is allowed to access the service. On the client which uses the service, it need to login and generate a token from https://ai.imjoy.io/apps/built-in/account-manager.html. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.
 


## Development

- We use [`black`](https://github.com/ambv/black) for code formatting.

```
  git clone git@github.com:imjoy-team/hypha.git
  # Enter directory.
  cd hypha
  # Install all development requirements and package in development mode.
  pip3 install -r requirements_dev.txt
```

- Run `tox` to run all tests and lint, including checking that `black` doesn't change any files.
