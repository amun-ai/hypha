# Migration Guide

## Migrating from Hypha 0.15.x to 0.20.x

In the new Hypha version, several breaking changes have been introduced to improve the RPC connection, making it more stable and secure. Most importantly, it now supports type annotation for Large Language Models (LLMs), which is ideal for building chatbots, AI agents, and more. Additionally, it features automatic reconnection when the connection is lost, ensuring a more reliable connection. Here is a brief guide to help you migrate your code from the old version to the new version.

### Python

#### 1. Use `hypha-rpc` instead of `imjoy-rpc`

To connect to the server, instead of installing the `imjoy-rpc` module, you will need to install the `hypha-rpc` module. The `hypha-rpc` module is a standalone module that provides the RPC connection to the Hypha server. You can install it using pip:

```bash
# pip install imjoy-rpc # previous install
pip install -U hypha-rpc # new install
```

We also changed our versioning strategy, we use the same version number for the server and client, so it's easier to match the client and server versions. For example, `hypha-rpc` version `0.20.38` is compatible with Hypha server version `0.20.38`.

#### 2. Change the imports to use `hypha-rpc`

Now you need to do the following changes in your code:

```python
# from imjoy_rpc.hypha import connect_to_server # previous import
from hypha_rpc import connect_to_server # new import
```

#### 3. Use snake_case for all service function names

Previously, we supported both snake_case and camelCase for the function names in Python, but in the new version, we only support snake_case for all function names.

Here is a suggested list of search and replace operations to update your code:
 - `.registerService` -> `.register_service`
 - `.unregisterService` -> `.unregister_service`
 - `.getService` -> `.get_service`
 - `.listServices` -> `.list_services`
 - `.registerCodec` -> `.register_codec`

Example:

```python
from hypha_rpc import connect_to_server # new import
async def start_server(server_url):
    server = await connect_to_server({"server_url": server_url})
    # register a service using server.register_service 
    # `server.registerService` is not supported anymore
    info = await server.register_service({
        "name": "Hello World",
        "id": "hello-world",
        "config": {
            "visibility": "public"
        },
        "hello": hello
    })

    # get a service using server.get_service
    # `server.getService` is not supported anymore
    svc = await server.get_service("hello-world")
    print(await svc.hello("John"))
```

#### 4. Changes in getting service

In the new version, we make the argument of `get_service(service_id)` a string (dictionary is not supported anymore).

The full service id format: `{workspace}/{client_id}:{service_id}@{app_id}`. We support the following shorted formats with conversion:

 - short service id format: `{service_id}`=>`{current_workspace}/*/{service_id}`, e.g. `await server.get_service("hello-world")`
 - service id format with workspace: `{workspace}/{service_id}` => `{workspace}/*/{service_id}`, e.g. `await server.get_service("public/hello-world")`
 - service id format with app id: `{service_id}@{app_id}` => `{current_workspace}/*/{service_id}@{app_id}`, e.g. `await server.get_service("hello-world@my-app-id")`
 - service id format with client id: `{client_id}:{service_id}` => `{current_workspace}/{client_id}:{service_id}`, e.g. `await server.get_service("my-client-id:hello-world")`
 - service id format with client id and app id: `{client_id}:{service_id}@{app_id}` => `{current_workspace}/{client_id}:{service_id}@{app_id}`, e.g. `await server.get_service("my-client-id:hello-world@my-app-id")`

**Note: Instead of return null, the new `get_service()` function will raise error if not found**

#### 5. Fix config options for register_service

Previously we allow passing keywords arguements for `get_service` (e.g. `register_service({...}, overwrite=True)`), now you need to pass it as a dictionary, e.g. `register_service({...}, {"overwrite": True})`.

#### 6. Optionally, annotate functions with JSON schema using Pydantic

To make your functions more compatible with LLMs, you can optionally use Pydantic to annotate them with JSON schema. This helps in creating well-defined interfaces for your services.

We created a tutorial to introduce this new feature: [service type annotation](./service-type-annotation.md).

Here is a quick example using Pydantic:

```python
import asyncio
from pydantic import BaseModel, Field
from hypha_rpc.utils.schema import schema_function

class UserInfo(BaseModel):
    name: str = Field(..., description="Name of the user")
    email: str = Field(..., description="Email of the user")
    age: int = Field(..., description="Age of the user")
    address: str = Field(..., description="Address of the user")

@schema_function
async def register_user(user_info: UserInfo) -> str:
    """Register a new user."""
    return f"User {user_info.name} registered"


async def main():
    server = await connect_to_server({"server_url": "https://hypha.amun.ai"})

    svc = await server.register_service({
        "name": "User Service",
        "id": "user-service",
        "config": {
            "visibility": "public"
        },
        "description": "Service for registering users",
        "register_user": register_user
    })

loop = asyncio.get_event_loop()
loop.create_task(main())
loop.run_forever()
```

### JavaScript

#### 1. Use `hypha-rpc` instead of `imjoy-rpc`

To connect to the server, instead of using the `imjoy-rpc` module, you will need to use the `hypha-rpc` module. The `hypha-rpc` module is a standalone module that provides the RPC connection to the Hypha server. You can include it in your HTML using a script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.38/dist/hypha-rpc-websocket.min.js"></script>
```

We also changed our versioning strategy, we use the same version number for the server and client, so it's easier to match the client and server versions. For example, `hypha-rpc` version `0.20.38` is compatible with Hypha server version `0.20.38`.

#### 2. Change the connection method and use camelCase for service function names

In JavaScript, the connection method and service function names use camelCase. 

Here is a suggested list of search and replace operations to update your code:

- `connect_to_server` -> `connectToServer`
- `.register_service` -> `.registerService`
- `.unregister_service` -> `.unregisterService`
- `.get_service` -> `.getService`
- `.list_services` -> `.listServices`
- `.register_codec` -> `.registerCodec`

Here is an example of how the updated code might look:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.38/dist/hypha-rpc-websocket.min.js"></script>
<script>
async function main(){
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "https://hypha.amun.ai"});
    // register a service using server.registerService 
    const info = await server.registerService({
        name: "Hello World",
        id: "hello-world",
        config: {
            visibility: "public"
        },
        hello: async (name) => `Hello ${name}`
    });

    // get a service using server.getService
    const svc = await server.getService("hello-world");
    const ret = await svc.hello("John");
    console.log(ret);
}
main();
</script>
```

#### 3. Changes in getting service

**Input Argument Change for `getService`**
In the new version, we make the argument of `getService(serviceId)` a string (object is not supported anymore).

The full service id format: `{workspace}/{clientId}:{serviceId}@{appId}`. We support the following shorted formats with conversion:

 - short service id format: `{serviceId}`=>`{currentWorkspace}/*/{serviceId}`, e.g. `await server.getService("hello-world")`
 - service id format with workspace: `{workspace}/{serviceId}` => `{workspace}/*/{serviceId}`, e.g. `await server.getService("public/hello-world")`
 - service id format with app id: `{serviceId}@{appId}` => `{currentWorkspace}/*/{serviceId}@{appId}`, e.g. `await server.getService("hello-world@my-app-id")`
 - service id format with client id: `{clientId}:{serviceId}` => `{currentWorkspace}/{clientId}:{serviceId}`, e.g. `await server.getService("my-client-id:hello-world")`
 - service id format with client id and app id: `{clientId}:{serviceId}@{appId}` => `{currentWorkspace}/{clientId}:{serviceId}@{appId}`, e.g. `await server.getService("my-client-id:hello-world@my-app-id")`


**Note: Instead of return null, the new `getService()` function will raise error if not found**

#### 4. Optionally, manually annotate functions with JSON schema

To make your functions more compatible with LLMs, you can optionally annotate them with JSON schema. This helps in creating well-defined interfaces for your services.

We created a tutorial to introduce this new feature: [service type annotation](./service-type-annotation.md).

Here is a quick example in JavaScript:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.38/dist/hypha-rpc-websocket.min.js"></script>

<script>
async function main(){
    const { connectToServer, schemaFunction } = hyphaWebsocketClient;
    const server = await connectToServer({"server_url": "https://hypha.amun.ai"});
    
    function getCurrentWeather(location, unit = "fahrenheit") {
        if (location.toLowerCase().includes("tokyo")) {
            return JSON.stringify({ location: "Tokyo", temperature: "10", unit: unit });
        } else if (location.toLowerCase().includes("san francisco")) {
            return JSON.stringify({ location: "San Francisco", temperature: "72", unit: unit });
        } else if (location.toLowerCase().includes("paris")) {
            return JSON.stringify({ location: "Paris", temperature: "22", unit: unit });
        } else {
            return JSON.stringify({ location: location, temperature: "unknown" });
        }
    }

    const getCurrentWeatherAnnotated = schemaFunction(getCurrentWeather, {
        name: "getCurrentWeather",
        description: "Get the current weather in a given location",
        parameters: {
            type: "object",
            properties: {
                location: { type: "string", description: "The city and state, e.g. San Francisco, CA" },
                unit: { type: "string", enum: ["celsius", "fahrenheit"] }
            },
            required: ["location"]
        }
    });

    await server.registerService({
        name: "Weather Service",
        id: "weather-service",
        getCurrentWeather: getCurrentWeatherAnnotated
    });

    const svc = await server.getService("weather-service");
    const ret = await svc.getCurrentWeather("Tokyo");
    console.log(ret);
}
main();
</script>
```

By following this guide, you should be able to smoothly transition your code from Hypha 0.15.x to 0.20.x and take advantage of the new features, including the support for Large Language Models through type annotations and JSON schema generation.
