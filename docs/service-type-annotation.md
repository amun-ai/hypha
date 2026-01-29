
# Tutorial: Using Typing with Hypha for Supporting Large Language Models

To help you understand the new feature, here is a well-structured tutorial for using typing in your Hypha services.

## Overview

Hypha supports multiple approaches for type annotations that automatically generate JSON schemas for Large Language Models (LLMs):

1. **Basic Type Hints**: Simple Python type annotations
2. **Pydantic Models**: Structured data models with validation
3. **Pydantic Field**: Enhanced type annotations with descriptions and constraints
4. **Combined Approach**: Using both Pydantic models and Field for maximum flexibility

## Approach 1: Basic Type Hints

**Quick Start:**

Here is a simple function in Python with basic type annotation:

```python
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
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

## Approach 2: Pydantic Models

Pydantic models provide structured data validation and automatic schema generation. This approach is ideal for complex data structures.

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

**Benefits of Pydantic Models:**
- ✅ Structured data validation
- ✅ Nested object support
- ✅ Automatic JSON schema generation
- ✅ Type safety and IDE support
- ✅ Reusable across multiple functions

## Approach 3: Pydantic Field with Direct Parameters

For simpler functions, you can use Pydantic Field directly on function parameters. This approach provides rich descriptions and constraints without creating separate model classes.

```python
from pydantic import Field
from hypha_rpc.utils.schema import schema_function

@schema_function
def calculate_weather(
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA"),
    unit: str = Field(default="fahrenheit", description="Temperature unit", enum=["celsius", "fahrenheit"]),
    include_humidity: bool = Field(default=False, description="Whether to include humidity data")
) -> dict:
    """Get the current weather in a given location with optional humidity data."""
    weather_data = {
        "location": location,
        "temperature": "22",
        "unit": unit
    }
    
    if include_humidity:
        weather_data["humidity"] = "65%"
    
    return weather_data
```

**Benefits of Direct Field Usage:**
- ✅ Rich parameter descriptions
- ✅ Enum constraints and validation
- ✅ Default values with descriptions
- ✅ No need for separate model classes
- ✅ Cleaner function signatures

## Approach 4: Combined Approach (Recommended)

For maximum flexibility and clarity, combine both approaches. Use Pydantic models for complex data structures and Field for simple parameters.

```python
from pydantic import BaseModel, Field
from hypha_rpc.utils.schema import schema_function
from typing import List, Optional

class Address(BaseModel):
    """Address information."""
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    state: str = Field(..., description="State or province")
    zip_code: str = Field(..., description="ZIP or postal code")

class UserProfile(BaseModel):
    """Complete user profile information."""
    name: str = Field(..., description="Full name of the user")
    email: str = Field(..., description="Email address")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    addresses: List[Address] = Field(default_factory=list, description="List of user addresses")
    preferences: dict = Field(default_factory=dict, description="User preferences")

@schema_function
def create_user_profile(
    profile: UserProfile,
    send_welcome_email: bool = Field(default=True, description="Send welcome email after registration"),
    email_template: str = Field(default="default", description="Email template to use", enum=["default", "premium", "custom"])
) -> dict:
    """Create a new user profile with optional welcome email."""
    result = {
        "user_id": f"user_{hash(profile.email)}",
        "profile": profile.dict(),
        "status": "created"
    }
    
    if send_welcome_email:
        result["email_sent"] = True
        result["email_template"] = email_template
    
    return result
```

**Benefits of Combined Approach:**
- ✅ Best of both worlds
- ✅ Complex data in Pydantic models
- ✅ Simple parameters with Field descriptions
- ✅ Maximum flexibility for different use cases
- ✅ Clear separation of concerns

## Advanced Field Features

Pydantic Field provides many advanced features for better schema generation:

```python
from pydantic import Field
from typing import List, Optional
from hypha_rpc.utils.schema import schema_function

@schema_function
def advanced_example(
    # String with pattern validation
    email: str = Field(..., description="Email address", pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    
    # Number with range constraints
    age: int = Field(..., ge=0, le=120, description="Age in years"),
    
    # Float with precision
    score: float = Field(..., ge=0.0, le=100.0, description="Test score"),
    
    # List with constraints
    tags: List[str] = Field(default_factory=list, max_items=10, description="List of tags"),
    
    # Optional field
    notes: Optional[str] = Field(None, max_length=500, description="Optional notes"),
    
    # Enum with description
    priority: str = Field(default="medium", enum=["low", "medium", "high"], description="Task priority level")
) -> dict:
    """Advanced example showing various Field features."""
    return {
        "email": email,
        "age": age,
        "score": score,
        "tags": tags,
        "notes": notes,
        "priority": priority
    }
```

## When to Use Each Approach

| Approach | Best For | Example Use Case |
|----------|----------|------------------|
| **Basic Type Hints** | Simple functions with basic types | Utility functions, simple calculations |
| **Pydantic Models** | Complex data structures, reusable schemas | User profiles, configuration objects, API responses |
| **Direct Field Usage** | Functions with rich parameter descriptions | API endpoints, tool functions with constraints |
| **Combined Approach** | Complex services with mixed data types | Full-featured services, enterprise applications |

## Complete Example: Weather Service

Here's a complete example showing all approaches in a weather service:

```python
from pydantic import BaseModel, Field
from hypha_rpc.utils.schema import schema_function
from typing import List, Optional
from enum import Enum

class WeatherUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"

class Location(BaseModel):
    """Geographic location."""
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude coordinate")

class WeatherData(BaseModel):
    """Weather information."""
    temperature: float = Field(..., description="Temperature value")
    unit: WeatherUnit = Field(..., description="Temperature unit")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    description: str = Field(..., description="Weather description")

@schema_function
def get_weather(
    location: Location,
    unit: WeatherUnit = Field(default=WeatherUnit.CELSIUS, description="Preferred temperature unit"),
    include_forecast: bool = Field(default=False, description="Include 5-day forecast"),
    language: str = Field(default="en", description="Response language", enum=["en", "es", "fr", "de"])
) -> WeatherData:
    """Get current weather for a specific location."""
    # Simulated weather data
    weather = WeatherData(
        temperature=22.5,
        unit=unit,
        humidity=65.0,
        description="Partly cloudy"
    )
    
    if include_forecast:
        # Add forecast logic here
        pass
    
    return weather

@schema_function
def get_weather_batch(
    locations: List[Location] = Field(..., max_items=10, description="List of locations to check"),
    unit: WeatherUnit = Field(default=WeatherUnit.CELSIUS, description="Temperature unit for all results")
) -> List[WeatherData]:
    """Get weather for multiple locations."""
    results = []
    for location in locations:
        weather = WeatherData(
            temperature=20.0 + hash(location.city) % 15,  # Simulated variation
            unit=unit,
            humidity=60.0 + hash(location.city) % 30,
            description="Sunny"
        )
        results.append(weather)
    
    return results
```

## Step-by-Step Guide

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
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.87/dist/hypha-rpc-websocket.min.js"></script>
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
