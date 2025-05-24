
# Tutorial: Using Typing with Hypha for Supporting Large Language Models

To help you understand the new feature, here is a well-structured tutorial for using typing in your Hypha services.

### Python

**Quick Start:**

Here is a simple function in Python with type annotation:

```python
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower()):
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
}
```

We can use the following JSON schema to describe the function signature:

```json
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
```

With the above JSON schema, we can generate the function call for the `get_current_weather` function in the LLMs. See [the documentation about function calling from OpenAI for more details](https://platform.openai.com/docs/guides/function-calling).

Manual JSON schema generation can be tedious, so we provide decorators `schema_function` and `schema_method` to automatically generate the JSON schema for the function, from the type hints and Pydantic models.

Here is an example of using the `schema_function` decorator with Pydantic models:

```python
from pydantic import BaseModel, Field
from hypha_rpc.utils.schema import schema_function

class UserInfo(BaseModel):
    """User information."""
    name: str = Field(..., description="Name of the user")
    email: str = Field(..., description="Email of the user")
    age: int = Field(..., description="Age of the user")
    address: str = Field(..., description="Address of the user")

@schema_function
def register_user(user_info: UserInfo) -> str:
    """Register a new user."""
    return f"User {user_info.name} registered"
```

Now, let's create a service in a Python client, then connect to it with another Python client and a JavaScript client.

### Step-by-Step Guide

**Python Client: Service Registration**

```python
from pydantic import BaseModel, Field
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

async def main():
    server = await connect_to_server({"server_url": "https://hypha.amun.ai"})

    class UserInfo(BaseModel):
        name: str = Field(..., description="Name of the user")
        email: str = Field(..., description="Email of the user")
        age: int = Field(..., description="Age of the user")
        address: str = Field(..., description="Address of the user")

    @schema_function
    def register_user(user_info: UserInfo) -> str:
        return f"User {user_info.name} registered"

    await server.register_service({
        "name": "User Service",
        "id": "user-service",
        "description": "Service for registering users",
        "register_user": register_user
    })

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
```

**Python Client: Service Usage**

```python
from hypha_rpc import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "https://hypha.amun.ai"})
    svc = await server.get_service("user-service")

    result = await svc.register_user({
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
        "address": "1234 Main St"
    })
    print(result)

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
```

**JavaScript Client: Service Usage**

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.54/dist/hypha-rpc-websocket.min.js"></script>
<script>
async function main() {
    const server = await hyphaWebsocketClient.connectToServer({"server_url": "https://hypha.amun.ai"});
    const svc = await server.getService("user-service");

    const result = await svc.register_user({
        name: "Alice",
        email: "alice@example.com",
        age: 30,
        address: "1234 Main St"
    });
    console.log(result);
}
main();
</script>
```

This complete tutorial demonstrates how to use typing with Hypha to support Large Language Models, showing service registration in Python and how to connect and use the service from both Python and JavaScript clients.
