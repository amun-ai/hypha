# Service Load Balancing

## Overview

Load balancing distributes workload across multiple service instances for better performance and availability. In Hypha, services expose numeric metrics through simple functions, and clients use the `select:criteria:function` syntax to route requests to the optimal instance.

**Key Requirements:**

- Service functions must return `int` or `float` values directly
- Multiple instances of the same service must be registered
- Client specifies selection criteria when getting the service

## Syntax

```python
service = await client.get_service(
    "service-name",
    mode="select:criteria:function_name"
)
```

Where:

- **criteria**: `min`, `max`, or `first_success`
- **function_name**: Service function that returns `int` or `float`

## Selection Criteria

### `min` - Select Minimum Value

```python
mode="select:min:get_load"        # Lowest load
mode="select:min:get_response_time"  # Fastest response
```

### `max` - Select Maximum Value

```python
mode="select:max:get_priority"    # Highest priority
mode="select:max:get_capacity"    # Most capacity
```

### `first_success` - First Responding Service

```python
mode="select:first_success:get_health"  # First available
```

## Service Implementation

Services must expose functions that return numeric values:

```python
class MyService:
    def __init__(self, instance_id):
        self.instance_id = instance_id
        self.current_load = 0.0
    
    async def get_load(self):
        """Return load as float (0.0 to 1.0)."""
        return self.current_load
    
    async def get_queue_length(self):
        """Return queue length as int."""
        return 5
    
    async def get_priority(self):
        """Return priority score as int."""
        return 10
    
    async def process_task(self, data):
        """Main service functionality."""
        return f"Processed by {self.instance_id}"

# Register the service
await api.register_service({
    "id": "my-service",
    "get_load": service.get_load,
    "get_queue_length": service.get_queue_length, 
    "get_priority": service.get_priority,
    "process_task": service.process_task
})
```

## Complete Example

```python
import asyncio
from hypha_rpc import connect_to_server

class TaskProcessor:
    def __init__(self, name, base_load):
        self.name = name
        self.load = base_load
    
    async def get_load(self):
        return self.load  # Must return float/int
    
    async def process(self, task):
        return f"Task '{task}' processed by {self.name}"

async def main():
    # Start three service instances
    instances = [
        TaskProcessor("fast", 0.2),
        TaskProcessor("medium", 0.5),
        TaskProcessor("slow", 0.8)
    ]
    
    # Register services (simplified registration)
    for i, instance in enumerate(instances):
        async with connect_to_server({"name": f"worker-{i}"}) as api:
            await api.register_service({
                "id": "task-processor",
                "get_load": instance.get_load,
                "process": instance.process
            })
    
    # Use load balancing
    async with connect_to_server({"name": "client"}) as client:
        # Always select least loaded instance
        service = await client.get_service(
            "task-processor",
            mode="select:min:get_load"
        )
        
        result = await service.process("important-task")
        print(result)  # Will use "fast" instance (load=0.2)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

```python
config = {
    "mode": "select:min:get_load",
    "select_timeout": 2.0  # Function call timeout (default: 2.0s)
}
```

## Error Handling

- Services without the specified function are skipped
- If no services respond, random selection is used as fallback
- Individual failures don't affect overall selection

This approach provides simple, efficient load balancing without complex routing logic. 
