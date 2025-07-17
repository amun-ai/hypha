# Autoscaling for Server Apps

## Motivation

In modern web applications and AI services, demand can vary dramatically throughout the day. A machine learning model might receive just a few requests per hour during off-peak times, but suddenly face hundreds of concurrent requests when integrated into a popular application. Similarly, web applications experience traffic spikes during product launches, marketing campaigns, or viral content.

Traditional approaches to handling variable load include:
- **Over-provisioning**: Running multiple instances all the time (wasteful and expensive)
- **Manual scaling**: Manually starting/stopping instances based on monitoring (slow and error-prone)
- **Complex infrastructure**: Setting up sophisticated load balancers and orchestration systems (time-consuming and maintenance-heavy)

**Hypha's autoscaling feature** addresses these challenges by automatically managing application instances based on real-time load. This provides:

- **Cost Efficiency**: Only run the instances you need, when you need them
- **Performance**: Automatically scale up before users experience slowdowns
- **Simplicity**: No complex infrastructure setup required
- **Reliability**: Maintain minimum instances for availability while scaling up during peaks

Whether you're serving FastAPI applications, AI models, or any other server apps, autoscaling ensures optimal resource utilization and responsive performance without manual intervention.

## Overview

The Hypha server supports automatic scaling of server applications based on load. This feature allows applications to automatically start and stop instances based on demand, ensuring optimal resource utilization and responsive performance.

The autoscaling system monitors the load on application instances and automatically scales them up or down based on configurable thresholds. It works by:

1. **Monitoring Load**: Tracking requests per minute for each application instance
2. **Making Scaling Decisions**: Comparing current load against configured thresholds
3. **Scaling Actions**: Starting new instances when load is high, stopping instances when load is low
4. **Respecting Limits**: Ensuring the number of instances stays within configured min/max bounds
5. **Service Selection**: Automatically distributing requests across available instances

## Getting Started

### Prerequisites

Before using autoscaling, ensure you have:
- Hypha server running (local or remote)
- Basic understanding of [Server Apps](/apps)
- Familiarity with [ASGI Apps](/asgi-apps) or [Serverless Functions](/serverless-functions) if using those

### Quick Start: Your First Autoscaling App

Let's create a simple autoscaling application step by step:

#### Step 1: Create a Simple App

```python
# save as simple_app.py
import asyncio
from hypha_rpc import connect_to_server

# Simple app that can be scaled
app_source = """
let requestCount = 0;
api.export({
    async setup() {
        console.log("Scalable app ready");
    },
    async processRequest(data) {
        requestCount++;
        // Simulate some processing work
        await new Promise(resolve => setTimeout(resolve, 10));
        return {
            processed: data,
            requestCount: requestCount,
            instanceId: Math.random().toString(36).substring(7)
        };
    }
});
"""

async def main():
    # Connect to Hypha server
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    controller = await api.get_service("public/server-apps")
    
    # Configure autoscaling
    autoscaling_config = {
        "enabled": True,
        "min_instances": 1,
        "max_instances": 3,
        "target_requests_per_instance": 20,
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.3,
        "scale_up_cooldown": 30,
        "scale_down_cooldown": 60,
    }
    
    # Install app with autoscaling
    app_info = await controller.install(
        source=app_source,
        config={
            "name": "My First Autoscaling App",
            "type": "window",
            "autoscaling": autoscaling_config,
        }
    )
    
    print(f"App installed with ID: {app_info['id']}")
    print(f"You can now access it using: api.get_service('default@{app_info['id']}')")
    
    # The app will automatically start instances based on load!

if __name__ == "__main__":
    asyncio.run(main())
```

#### Step 2: Test the Autoscaling

```python
# save as test_scaling.py
import asyncio
from hypha_rpc import connect_to_server

async def test_autoscaling():
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    
    # Access the app - this will start it automatically (lazy loading)
    service = await api.get_service("default@my-first-autoscaling-app")
    
    # Generate some load
    print("Generating load...")
    tasks = []
    for i in range(50):  # 50 concurrent requests
        tasks.append(service.processRequest(f"request_{i}"))
    
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} requests")
    
    # Check how many instances are running
    controller = await api.get_service("public/server-apps")
    running_apps = await controller.list_running()
    instances = [app for app in running_apps if app["app_id"] == "my-first-autoscaling-app"]
    print(f"Currently running {len(instances)} instances")

if __name__ == "__main__":
    asyncio.run(test_autoscaling())
```

## Configuration

Autoscaling is configured through the `autoscaling` parameter when installing an application:

```python
autoscaling_config = {
    "enabled": True,
    "min_instances": 1,
    "max_instances": 5,
    "target_requests_per_instance": 100,
    "scale_up_threshold": 0.8,
    "scale_down_threshold": 0.3,
    "scale_up_cooldown": 300,
    "scale_down_cooldown": 600,
}

app_info = await controller.install(
    source=app_source,
    config={
        "name": "My Scalable App",
        "type": "window",
        "autoscaling": autoscaling_config,
    }
)
```

### Configuration Parameters

- **`enabled`** (bool): Whether autoscaling is enabled for this app
- **`min_instances`** (int): Minimum number of instances to keep running (default: 1)
- **`max_instances`** (int): Maximum number of instances allowed (default: 10)
- **`target_requests_per_instance`** (int): Target number of requests per minute per instance (default: 100)
- **`scale_up_threshold`** (float): Scale up when load exceeds this fraction of target (default: 0.9)
- **`scale_down_threshold`** (float): Scale down when load falls below this fraction of target (default: 0.3)
- **`scale_up_cooldown`** (int): Seconds to wait between scale-up actions (default: 300)
- **`scale_down_cooldown`** (int): Seconds to wait between scale-down actions (default: 600)

## Service Selection Mode with Autoscaling

When autoscaling creates multiple instances of your application, you need a way to distribute incoming requests across these instances. Hypha provides **service selection modes** that work seamlessly with autoscaling.

### Understanding Service Selection

When you have multiple instances of an app running, accessing a service like `default@my-app` needs to decide which instance to route the request to. The `service_selection_mode` configuration determines this behavior:

```python
app_info = await controller.install(
    source=app_source,
    config={
        "name": "My Scalable App",
        "type": "window",
        "autoscaling": autoscaling_config,
        "service_selection_mode": "random",  # Key addition for autoscaling
    }
)
```

### Available Selection Modes

- **`random`**: Randomly select an instance (good for load distribution)
- **`first`**: Always use the first available instance
- **`last`**: Always use the last available instance
- **`min_load`**: Route to the instance with the lowest current load
- **`exact`**: Require exact match (fails if multiple instances exist)

### Recommended Mode for Autoscaling

For autoscaling applications, **`random`** is typically the best choice because:
- It naturally distributes load across instances
- It's simple and doesn't require complex load calculations
- It works well with the autoscaling algorithm

```python
# Recommended configuration for autoscaling
config = {
    "name": "Scalable Service",
    "type": "window",
    "autoscaling": {
        "enabled": True,
        "min_instances": 1,
        "max_instances": 5,
        "target_requests_per_instance": 50,
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.3,
    },
    "service_selection_mode": "random",  # Essential for autoscaling
}
```

## How It Works

### Load Monitoring

The autoscaling system monitors the load on each application instance by tracking:
- **Request Rate**: Number of requests per minute
- **Average Load**: Load averaged across all instances of an app

Only instances with load balancing enabled (those with `__rlb` suffix in their client ID) are monitored for autoscaling.

### Scaling Decisions

Every 10 seconds, the autoscaling manager:

1. **Calculates Current Load**: Gets the average load across all instances
2. **Compares to Thresholds**: 
   - Scale up if: `average_load > scale_up_threshold * target_requests_per_instance`
   - Scale down if: `average_load < scale_down_threshold * target_requests_per_instance`
3. **Checks Limits**: Ensures scaling won't violate min/max instance limits
4. **Respects Cooldowns**: Prevents scaling if within cooldown period

### Scaling Actions

- **Scale Up**: Starts a new instance of the application
- **Scale Down**: Stops the least loaded instance (but never the last one if at min_instances)

## End-to-End Example: Autoscaling FastAPI Application

Let's create a complete example that demonstrates autoscaling with a FastAPI application, including lazy loading and service selection.

### Step 1: Create the FastAPI Application

```python
# save as scalable_fastapi_app.py
import asyncio
import time
from hypha_rpc import connect_to_server

# FastAPI app source code
fastapi_app_source = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Scalable FastAPI App</title>
</head>
<body>
<script id="worker" type="javascript/worker">
importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

async function loadPyodideAndPackages() {
    self.pyodide = await loadPyodide();
    await self.pyodide.loadPackage(["micropip"]);
    await self.pyodide.runPython(`
        import micropip
        await micropip.install(['fastapi==0.104.1', 'uvicorn'])
    `);
}

loadPyodideAndPackages().then(() => {
    self.pyodide.runPython(`
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
        import time
        import random
        import asyncio
        from hypha_rpc import connect_to_server
        
        app = FastAPI(title="Scalable Image Processing API")
        
        # Instance information
        instance_id = f"instance_{random.randint(1000, 9999)}"
        start_time = time.time()
        request_count = 0
        
        @app.get("/", response_class=HTMLResponse)
        async def home():
            return '''
            <h1>Scalable Image Processing API</h1>
            <p>Instance ID: ''' + instance_id + '''</p>
            <p>Uptime: ''' + str(int(time.time() - start_time)) + ''' seconds</p>
            <p>Requests processed: ''' + str(request_count) + '''</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/api/status">GET /api/status</a> - Instance status</li>
                <li><a href="/api/process">POST /api/process</a> - Process data</li>
                <li><a href="/api/heavy-task">POST /api/heavy-task</a> - CPU intensive task</li>
            </ul>
            '''
        
        @app.get("/api/status")
        async def get_status():
            global request_count
            request_count += 1
            return JSONResponse({
                "instance_id": instance_id,
                "uptime": int(time.time() - start_time),
                "requests_processed": request_count,
                "status": "healthy"
            })
        
        @app.post("/api/process")
        async def process_data(data: dict):
            global request_count
            request_count += 1
            
            # Simulate some processing time
            await asyncio.sleep(0.1)
            
            return JSONResponse({
                "processed_by": instance_id,
                "input": data,
                "result": f"Processed {data} at {time.time()}",
                "request_number": request_count
            })
        
        @app.post("/api/heavy-task")
        async def heavy_task(task: dict):
            global request_count
            request_count += 1
            
            # Simulate CPU-intensive work
            iterations = task.get("iterations", 1000)
            result = 0
            for i in range(iterations):
                result += i * i
                if i % 100 == 0:
                    await asyncio.sleep(0.001)  # Yield control
            
            return JSONResponse({
                "processed_by": instance_id,
                "task": task,
                "result": result,
                "request_number": request_count
            })
        
        # ASGI application wrapper
        async def serve_fastapi(args, context=None):
            scope = args["scope"]
            if context:
                print(f'{context["user"]["id"]} - {scope["method"]} {scope["path"]}')
            await app(args["scope"], args["receive"], args["send"])
        
        # Connect to Hypha and register the service
        async def main():
            server = await connect_to_server({"server_url": "ws://localhost:9527"})
            
            await server.register_service({
                "id": "fastapi-service",
                "name": "FastAPI Service",
                "type": "asgi",
                "serve": serve_fastapi,
                "config": {"visibility": "public", "require_context": True}
            })
            
            print(f"FastAPI service registered by instance {instance_id}")
            await server.serve()
        
        await main()
    `);
});
</script>
<script>
const blob = new Blob([
    document.querySelector('#worker').textContent
], { type: "text/javascript" });
const worker = new Worker(window.URL.createObjectURL(blob), {type: "module"});
worker.onerror = console.error;
worker.onmessage = console.log;
const config = Object.fromEntries(new URLSearchParams(window.location.search));
if (!config.server_url) config.server_url = window.location.origin;
worker.postMessage(config);
</script>
</body>
</html>
"""

async def install_scalable_fastapi_app():
    # Connect to Hypha server
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    controller = await api.get_service("public/server-apps")
    
    # Configure autoscaling for FastAPI app
    autoscaling_config = {
        "enabled": True,
        "min_instances": 1,
        "max_instances": 4,
        "target_requests_per_instance": 30,  # Scale up after 30 requests/min
        "scale_up_threshold": 0.8,  # Scale up when load > 24 req/min
        "scale_down_threshold": 0.3,  # Scale down when load < 9 req/min
        "scale_up_cooldown": 45,  # Wait 45s between scale-ups
        "scale_down_cooldown": 90,  # Wait 90s between scale-downs
    }
    
    # Install the FastAPI app with autoscaling
    app_info = await controller.install(
        source=fastapi_app_source,
        config={
            "name": "Scalable FastAPI Image Processor",
            "type": "web-python",
            "version": "1.0.0",
            "description": "A scalable FastAPI application for image processing",
            "autoscaling": autoscaling_config,
            "service_selection_mode": "random",  # Important for load distribution
        },
        overwrite=True
    )
    
    print(f"✅ FastAPI app installed successfully!")
    print(f"   App ID: {app_info['id']}")
    print(f"   Autoscaling enabled: {app_info['autoscaling']['enabled']}")
    print(f"   Min instances: {app_info['autoscaling']['min_instances']}")
    print(f"   Max instances: {app_info['autoscaling']['max_instances']}")
    print(f"\n🚀 You can access the app using:")
    print(f"   service = await api.get_service('fastapi-service@{app_info['id']}')")
    print(f"   Or visit: http://localhost:9527/{api.config['workspace']}/apps/{app_info['id']}")
    
    return app_info

if __name__ == "__main__":
    asyncio.run(install_scalable_fastapi_app())
```

### Step 2: Test the Autoscaling FastAPI App

```python
# save as test_fastapi_scaling.py
import asyncio
import aiohttp
from hypha_rpc import connect_to_server

async def test_fastapi_autoscaling():
    # Connect to Hypha
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    controller = await api.get_service("public/server-apps")
    
    # Access the FastAPI service - this will start it automatically (lazy loading)
    print("🔍 Accessing FastAPI service (lazy loading)...")
    service = await api.get_service("fastapi-service@scalable-fastapi-image-processor")
    
    print("✅ Service loaded successfully!")
    
    # Test 1: Check initial status
    print("\n📊 Initial status check:")
    initial_status = await service.get("/api/status")
    print(f"   Instance ID: {initial_status['instance_id']}")
    print(f"   Requests processed: {initial_status['requests_processed']}")
    
    # Test 2: Generate moderate load
    print("\n🔄 Generating moderate load...")
    tasks = []
    for i in range(20):
        tasks.append(service.post("/api/process", {"data": f"item_{i}"}))
    
    results = await asyncio.gather(*tasks)
    print(f"   Processed {len(results)} requests")
    
    # Check running instances
    running_apps = await controller.list_running()
    app_instances = [app for app in running_apps if app["app_id"] == "scalable-fastapi-image-processor"]
    print(f"   Currently running {len(app_instances)} instances")
    
    # Test 3: Generate heavy load to trigger scaling
    print("\n🚀 Generating heavy load to trigger scaling...")
    heavy_tasks = []
    for i in range(50):
        if i % 2 == 0:
            heavy_tasks.append(service.post("/api/heavy-task", {"iterations": 2000}))
        else:
            heavy_tasks.append(service.post("/api/process", {"data": f"heavy_item_{i}"}))
    
    # Execute tasks in batches to simulate sustained load
    batch_size = 10
    for i in range(0, len(heavy_tasks), batch_size):
        batch = heavy_tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        print(f"   Completed batch {i//batch_size + 1}/{len(heavy_tasks)//batch_size + 1}, processed {len(batch_results)} requests")
        
        # Check instances after each batch
        running_apps = await controller.list_running()
        app_instances = [app for app in running_apps if app["app_id"] == "scalable-fastapi-image-processor"]
        print(f"   Running instances: {len(app_instances)}")
        
        # Small delay to allow autoscaling to react
        await asyncio.sleep(2)
    
    # Test 4: Wait for potential scaling
    print("\n⏳ Waiting for autoscaling to respond...")
    await asyncio.sleep(30)
    
    # Final status check
    running_apps = await controller.list_running()
    final_instances = [app for app in running_apps if app["app_id"] == "scalable-fastapi-image-processor"]
    print(f"\n📈 Final state: {len(final_instances)} instances running")
    
    # Check status of each instance
    for i, instance in enumerate(final_instances):
        try:
            # Get instance-specific service
            instance_service = await api.get_service(f"fastapi-service@{instance['app_id']}")
            status = await instance_service.get("/api/status")
            print(f"   Instance {i+1}: {status['instance_id']} - {status['requests_processed']} requests")
        except Exception as e:
            print(f"   Instance {i+1}: Error getting status - {e}")
    
    print("\n✅ FastAPI autoscaling test completed!")

if __name__ == "__main__":
    asyncio.run(test_fastapi_autoscaling())
```

### Step 3: HTTP Client Test

You can also test the autoscaling FastAPI app using HTTP requests:

```python
# save as test_fastapi_http.py
import asyncio
import aiohttp
from hypha_rpc import connect_to_server

async def test_fastapi_http():
    # Connect to get workspace info
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    
    # Generate token for HTTP access
    token = await api.generate_token()
    workspace = api.config["workspace"]
    
    # HTTP endpoint URL
    base_url = f"http://localhost:9527/{workspace}/apps/scalable-fastapi-image-processor"
    
    async with aiohttp.ClientSession() as session:
        # Test the home page
        async with session.get(f"{base_url}/", 
                              headers={"Authorization": f"Bearer {token}"}) as response:
            html = await response.text()
            print("📄 Home page loaded successfully!")
            print(f"   Response length: {len(html)} characters")
        
        # Test API endpoints
        print("\n🔗 Testing API endpoints:")
        
        # Status endpoint
        async with session.get(f"{base_url}/api/status", 
                              headers={"Authorization": f"Bearer {token}"}) as response:
            status = await response.json()
            print(f"   Status: {status}")
        
        # Process endpoint
        async with session.post(f"{base_url}/api/process", 
                               json={"data": "test_data"},
                               headers={"Authorization": f"Bearer {token}"}) as response:
            result = await response.json()
            print(f"   Process result: {result}")
        
        # Generate load via HTTP
        print("\n🌊 Generating HTTP load...")
        tasks = []
        for i in range(30):
            tasks.append(
                session.post(f"{base_url}/api/heavy-task", 
                           json={"iterations": 1500},
                           headers={"Authorization": f"Bearer {token}"})
            )
        
        # Execute requests
        responses = await asyncio.gather(*tasks)
        print(f"   Completed {len(responses)} HTTP requests")
        
        # Check final instance count
        controller = await api.get_service("public/server-apps")
        running_apps = await controller.list_running()
        app_instances = [app for app in running_apps if app["app_id"] == "scalable-fastapi-image-processor"]
        print(f"   Running instances after HTTP load: {len(app_instances)}")

if __name__ == "__main__":
    asyncio.run(test_fastapi_http())
```

## Integration with Other Hypha Features

Autoscaling works seamlessly with other Hypha features:

### ASGI Applications

As shown in the FastAPI example above, autoscaling is particularly powerful with [ASGI Apps](/asgi-apps). You can:
- Scale FastAPI, Django, or other ASGI applications
- Use the `hypha_rpc.utils.serve` utility with autoscaling
- Handle streaming responses across multiple instances

### Serverless Functions

You can also use autoscaling with [Serverless Functions](/serverless-functions):

```python
# Autoscaling serverless function
function_source = """
api.export({
    "process-data": async function(event) {
        // Simulate processing
        await new Promise(resolve => setTimeout(resolve, 100));
        return {
            status: 200,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                message: "Data processed",
                timestamp: new Date().toISOString(),
                path: event.path
            })
        };
    }
});
"""

app_info = await controller.install(
    source=function_source,
    config={
        "name": "Scalable Function Service",
        "type": "functions",
        "autoscaling": {
            "enabled": True,
            "min_instances": 1,
            "max_instances": 3,
            "target_requests_per_instance": 25,
        },
        "service_selection_mode": "random",
    }
)
```

### Lazy Loading

All autoscaling apps support lazy loading. This means:
- Apps don't start until first accessed
- Automatic scaling begins once the app is running
- Perfect for on-demand services

```python
# App starts automatically when first accessed
service = await api.get_service("my-service@my-scalable-app")
result = await service.process_data("test")  # App starts here if not running
```

## Best Practices

### 1. Choose Appropriate Thresholds

- Set `target_requests_per_instance` based on your app's processing capacity
- Use `scale_up_threshold` of 0.7-0.9 to scale up before overload
- Use `scale_down_threshold` of 0.2-0.4 to avoid rapid scaling oscillations

### 2. Configure Reasonable Cooldowns

- `scale_up_cooldown`: 60-300 seconds (scale up quickly under load)
- `scale_down_cooldown`: 300-600 seconds (scale down slowly to avoid oscillations)

### 3. Set Appropriate Limits

- `min_instances`: At least 1 for availability
- `max_instances`: Based on resource constraints and expected peak load

### 4. Use Service Selection Mode

- Always set `service_selection_mode: "random"` for autoscaling apps
- This ensures proper load distribution across instances

### 5. Monitor and Adjust

- Monitor your app's performance and scaling behavior
- Adjust thresholds based on observed load patterns
- Use logs to understand scaling decisions

## Monitoring

You can monitor autoscaling behavior through:

### Application Logs
```python
logs = await controller.get_logs(session_id)
# Look for autoscaling messages in logs
```

### Running Instances
```python
running_apps = await controller.list_running()
instances = [app for app in running_apps if app["app_id"] == app_id]
print(f"Currently running {len(instances)} instances")
```

### Load Metrics
The system tracks load metrics using Prometheus metrics that can be monitored externally.

## Limitations

1. **Load Balancing Required**: Only works with load balancing enabled clients
2. **Monitoring Interval**: Scaling decisions are made every 10 seconds
3. **Startup Time**: New instances take time to start and become available
4. **State Management**: Stateful applications may need special consideration

## Troubleshooting

### Autoscaling Not Working

1. **Check Configuration**: Ensure `enabled: true` in autoscaling config
2. **Verify Load Balancing**: Make sure client IDs have `__rlb` suffix
3. **Check Thresholds**: Verify thresholds are appropriate for your load
4. **Review Cooldowns**: Ensure cooldown periods haven't blocked scaling
5. **Service Selection Mode**: Ensure it's set to "random" for proper load distribution

### Unexpected Scaling Behavior

1. **Monitor Load**: Check actual load vs. configured thresholds
2. **Review Logs**: Look for scaling decision messages in logs
3. **Check Limits**: Verify min/max instance limits are appropriate
4. **Adjust Thresholds**: Fine-tune based on observed behavior

## Advanced Features

### Custom Metrics
While the default implementation uses request rate, the system is designed to support custom metrics in the future.

### Integration with External Systems
The autoscaling system can be integrated with external monitoring and alerting systems through the Prometheus metrics it exposes.

## Complete Real-World Example: AI Model Serving

Here's a comprehensive example showing how to build a production-ready AI model serving system with autoscaling:

```python
# save as ai_model_server.py
import asyncio
import json
import time
from hypha_rpc import connect_to_server

# AI Model Server with autoscaling
ai_model_source = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>AI Model Server</title>
</head>
<body>
<script id="worker" type="javascript/worker">
// Simulate AI model loading and inference
class AIModel {
    constructor() {
        this.model_id = Math.random().toString(36).substring(7);
        this.loaded = false;
        this.inference_count = 0;
        this.load_time = Date.now();
    }
    
    async load() {
        console.log(`Loading AI model ${this.model_id}...`);
        // Simulate model loading time
        await new Promise(resolve => setTimeout(resolve, 2000));
        this.loaded = true;
        console.log(`Model ${this.model_id} loaded successfully`);
    }
    
    async predict(input) {
        if (!this.loaded) {
            await this.load();
        }
        
        // Simulate inference time
        const inference_time = 100 + Math.random() * 200; // 100-300ms
        await new Promise(resolve => setTimeout(resolve, inference_time));
        
        this.inference_count++;
        
        return {
            model_id: this.model_id,
            prediction: `AI prediction for: ${input}`,
            confidence: Math.random(),
            inference_time: inference_time,
            inference_count: this.inference_count
        };
    }
    
    getStats() {
        return {
            model_id: this.model_id,
            loaded: this.loaded,
            inference_count: this.inference_count,
            uptime: Date.now() - this.load_time
        };
    }
}

// Initialize the model
const aiModel = new AIModel();

// Export the API
self.onmessage = async function(e) {
    const hyphaWebsocketClient = await import("ws://localhost:9527/assets/hypha-rpc-websocket.mjs");
    const config = e.data;
    
    const api = await hyphaWebsocketClient.connectToServer(config);
    
    api.export({
        async setup() {
            console.log(`AI Model server ${aiModel.model_id} initializing...`);
            await aiModel.load();
            console.log(`AI Model server ${aiModel.model_id} ready`);
        },
        
        async predict(input) {
            console.log(`Processing prediction request: ${input}`);
            const result = await aiModel.predict(input);
            console.log(`Prediction completed by ${result.model_id}`);
            return result;
        },
        
        async batch_predict(inputs) {
            console.log(`Processing batch prediction: ${inputs.length} inputs`);
            const results = [];
            for (const input of inputs) {
                const result = await aiModel.predict(input);
                results.push(result);
            }
            return {
                batch_size: inputs.length,
                results: results,
                processed_by: aiModel.model_id
            };
        },
        
        async get_model_stats() {
            return aiModel.getStats();
        },
        
        async health_check() {
            return {
                status: "healthy",
                model_loaded: aiModel.loaded,
                model_id: aiModel.model_id,
                timestamp: Date.now()
            };
        }
    });
};
</script>
<script>
const blob = new Blob([
    document.querySelector('#worker').textContent
], { type: "text/javascript" });
const worker = new Worker(window.URL.createObjectURL(blob), {type: "module"});
worker.onerror = console.error;
worker.onmessage = console.log;
const config = Object.fromEntries(new URLSearchParams(window.location.search));
if (!config.server_url) config.server_url = window.location.origin;
worker.postMessage(config);
</script>
</body>
</html>
"""

async def deploy_ai_model_server():
    # Connect to Hypha
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    controller = await api.get_service("public/server-apps")
    
    # Production-ready autoscaling configuration
    autoscaling_config = {
        "enabled": True,
        "min_instances": 2,  # Always keep 2 instances for availability
        "max_instances": 8,  # Scale up to 8 instances under heavy load
        "target_requests_per_instance": 60,  # 1 request per second per instance
        "scale_up_threshold": 0.8,  # Scale up when load > 48 req/min
        "scale_down_threshold": 0.4,  # Scale down when load < 24 req/min
        "scale_up_cooldown": 60,   # Quick scale up (1 minute)
        "scale_down_cooldown": 300,  # Slow scale down (5 minutes)
    }
    
    # Deploy the AI model server
    app_info = await controller.install(
        source=ai_model_source,
        config={
            "name": "Production AI Model Server",
            "type": "window",
            "version": "1.0.0",
            "description": "Scalable AI model serving with automatic load balancing",
            "autoscaling": autoscaling_config,
            "service_selection_mode": "random",
            "startup_config": {
                "timeout": 30,  # Allow time for model loading
                "wait_for_service": "default",
                "stop_after_inactive": 3600,  # Stop after 1 hour of inactivity
            }
        },
        overwrite=True
    )
    
    print("🤖 AI Model Server deployed successfully!")
    print(f"   App ID: {app_info['id']}")
    print(f"   Min instances: {autoscaling_config['min_instances']}")
    print(f"   Max instances: {autoscaling_config['max_instances']}")
    print(f"   Target load: {autoscaling_config['target_requests_per_instance']} req/min per instance")
    
    return app_info

async def test_ai_model_server():
    # Connect to Hypha
    api = await connect_to_server({"server_url": "ws://localhost:9527"})
    controller = await api.get_service("public/server-apps")
    
    # Access the AI model service (lazy loading)
    print("🚀 Accessing AI Model Server...")
    model_service = await api.get_service("default@production-ai-model-server")
    
    # Test 1: Single prediction
    print("\n🔮 Testing single prediction...")
    prediction = await model_service.predict("What is the weather today?")
    print(f"   Prediction: {prediction['prediction']}")
    print(f"   Confidence: {prediction['confidence']:.2f}")
    print(f"   Processed by: {prediction['model_id']}")
    
    # Test 2: Batch prediction
    print("\n📦 Testing batch prediction...")
    batch_inputs = [
        "Translate: Hello world",
        "Summarize: The quick brown fox...",
        "Classify: This is a positive review",
        "Generate: A creative story about...",
        "Analyze: Stock market trends"
    ]
    
    batch_result = await model_service.batch_predict(batch_inputs)
    print(f"   Batch size: {batch_result['batch_size']}")
    print(f"   Processed by: {batch_result['processed_by']}")
    
    # Test 3: Load testing
    print("\n🌊 Load testing to trigger autoscaling...")
    load_tasks = []
    
    for i in range(100):
        if i % 3 == 0:
            # Batch predictions
            load_tasks.append(model_service.batch_predict([
                f"Batch item {i}-1",
                f"Batch item {i}-2",
                f"Batch item {i}-3"
            ]))
        else:
            # Single predictions
            load_tasks.append(model_service.predict(f"Load test query {i}"))
    
    # Execute load test in waves
    batch_size = 20
    for i in range(0, len(load_tasks), batch_size):
        batch = load_tasks[i:i+batch_size]
        print(f"   Executing batch {i//batch_size + 1}/{len(load_tasks)//batch_size + 1}")
        
        start_time = time.time()
        results = await asyncio.gather(*batch)
        end_time = time.time()
        
        print(f"   Completed {len(results)} requests in {end_time - start_time:.2f}s")
        
        # Check instance count
        running_apps = await controller.list_running()
        instances = [app for app in running_apps if app["app_id"] == "production-ai-model-server"]
        print(f"   Running instances: {len(instances)}")
        
        # Brief pause to allow autoscaling
        await asyncio.sleep(5)
    
    # Test 4: Model statistics
    print("\n📊 Final model statistics...")
    await asyncio.sleep(10)  # Wait for final scaling
    
    try:
        stats = await model_service.get_model_stats()
        print(f"   Model ID: {stats['model_id']}")
        print(f"   Inferences: {stats['inference_count']}")
        print(f"   Uptime: {stats['uptime']/1000:.1f}s")
        
        health = await model_service.health_check()
        print(f"   Health: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        
    except Exception as e:
        print(f"   Error getting stats: {e}")
    
    # Final instance count
    running_apps = await controller.list_running()
    final_instances = [app for app in running_apps if app["app_id"] == "production-ai-model-server"]
    print(f"\n🎯 Final result: {len(final_instances)} instances running")
    
    print("\n✅ AI Model Server testing completed!")

async def main():
    print("=== AI Model Server Deployment ===")
    await deploy_ai_model_server()
    
    print("\n=== Testing AI Model Server ===")
    await test_ai_model_server()

if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive example demonstrates:

1. **Production Configuration**: Realistic autoscaling parameters for AI workloads
2. **Model Loading**: Simulates actual AI model initialization
3. **Multiple Endpoints**: Different types of inference (single, batch)
4. **Health Monitoring**: Status and statistics endpoints
5. **Load Testing**: Realistic load patterns that trigger autoscaling
6. **Error Handling**: Robust error handling and monitoring

## Conclusion

Hypha's autoscaling feature provides a powerful, easy-to-use solution for automatically managing application instances based on demand. By combining autoscaling with service selection modes, lazy loading, and integration with ASGI apps and serverless functions, you can build robust, cost-effective applications that automatically adapt to changing load patterns.

The key benefits include:
- **Zero-configuration scaling**: Just set your parameters and let Hypha handle the rest
- **Cost efficiency**: Only run instances when needed
- **High availability**: Maintain minimum instances for reliability
- **Seamless integration**: Works with existing Hypha features and applications

Whether you're building a simple web service or a complex AI model serving system, autoscaling ensures your applications remain responsive and cost-effective under any load condition. 