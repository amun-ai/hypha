# Development Guide

This guide covers advanced Hypha features for developers who want to build production applications, configure complex deployments, and extend Hypha's capabilities.

> **New to Hypha?** Start with the [Getting Started Guide](getting-started.md) to learn the basics first.

## Advanced Server Configuration

### Built-in S3 (Minio) Server

For features requiring S3 object storage (like Server Apps or Artifact Management), Hypha provides a convenient built-in Minio server. To start the Hypha server along with this built-in S3 server, use the `--start-minio-server` flag:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527 --start-minio-server
```

This automatically:
- Starts a Minio server process.
- Enables S3 support (`--enable-s3`).
- Configures the necessary S3 connection details (`--endpoint-url`, `--access-key-id`, `--secret-access-key`).

**Note:** You cannot use `--start-minio-server` if you are also manually providing S3 connection details (e.g., `--endpoint-url`). Choose one method or the other.

#### Customizing the Built-in Minio Server

You can customize the built-in Minio server using these options:
- `--minio-workdir`: Specify a directory for Minio data (defaults to a temporary directory), it must be an absolute path.
- `--minio-port`: Set the port for the Minio server (defaults to 9000).
- `--minio-root-user`: Set the root user (defaults to `minioadmin`).
- `--minio-root-password`: Set the root password (defaults to `minioadmin`).
- `--minio-version`: Specify a specific version of the Minio server to use.
- `--mc-version`: Specify a specific version of the Minio client to use.
- `--minio-file-system-mode`: Enable file system mode with specific compatible versions.

Example with custom Minio settings:
```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527 \
    --start-minio-server \
    --minio-workdir=/tmp/minio_data \
    --minio-port=9001 \
    --minio-root-user=myuser \
    --minio-root-password=mypassword
```

#### Minio File System Mode

For better filesystem-like behavior, you can enable file system mode with the `--minio-file-system-mode` flag:

```bash
python3 -m hypha.server --host=0.0.0.0 --port=9527 \
    --start-minio-server \
    --minio-file-system-mode
```

When file system mode is enabled, Hypha uses specific versions of Minio that are compatible with file system operations:
- Minio server: `RELEASE.2022-10-24T18-35-07Z`
- Minio client: `RELEASE.2022-10-29T10-09-23Z`

This mode optimizes Minio for use as a direct file system, which means:
1. Files are stored in their raw format, allowing direct access from the file system
2. The .minio.sys directory is automatically cleaned up to avoid version conflicts
3. The Minio server process is properly terminated when the application shuts down

File system mode is particularly useful for development environments and when you need to access the stored files directly without using S3 API calls.

**Note:** In file system mode, some advanced S3 features like versioning may not be available, but basic operations like storing and retrieving files will work consistently.

#### Running with Built-in S3 (Minio) in Docker

When running Hypha with the built-in Minio server inside a Docker container, additional considerations are necessary:

1. **Volume Mounting**: For data persistence, mount a volume for the Minio data directory:
   ```bash
   docker run -v /host/path/to/minio_data:/app/minio_data your-hypha-image \
     python -m hypha.server --host=0.0.0.0 --port=9527 \
     --start-minio-server \
     --minio-workdir=/app/minio_data
   ```

2. **Executable Path**: You may need to specify the Minio executable path with `--executable-path`:
   ```bash
   docker run your-hypha-image \
     python -m hypha.server --host=0.0.0.0 --port=9527 \
     --start-minio-server \
     --executable-path=/path/to/minio
   ```

3. **Permissions**: Ensure the Minio working directory is writable by the container user:
   ```bash
   docker run -v /host/path/to/minio_data:/app/minio_data your-hypha-image \
     chown -R container_user:container_user /app/minio_data && \
     python -m hypha.server --host=0.0.0.0 --port=9527 \
     --start-minio-server \
     --minio-workdir=/app/minio_data
   ```

4. **Port Exposure**: Remember to expose both the Hypha port and Minio port:
   ```bash
   docker run -p 9527:9527 -p 9000:9000 -v /host/path/to/minio_data:/app/minio_data your-hypha-image \
     python -m hypha.server --host=0.0.0.0 --port=9527 \
     --start-minio-server
   ```

### Server Apps

If you want to enable server apps (browsers running on the server side), you need to enable S3 storage first. You can either configure an external S3 provider or use the built-in Minio server as described above. 

First, install the full server apps support:
```bash
pip install -U "hypha[server-apps]"
playwright install
```

To start with server apps enabled, use the `--enable-server-apps` flag along with your S3 configuration method:

Using built-in Minio:
```bash
python -m hypha.server --host=0.0.0.0 --port=9527 --start-minio-server --enable-server-apps
```

Using external S3 (example):
```bash
python -m hypha.server --host=0.0.0.0 --port=9527 \
    --enable-s3 \
    --endpoint-url=<your-s3-endpoint> \
    --access-key-id=<your-key-id> \
    --secret-access-key=<your-secret> \
    --enable-server-apps
```

### Serving Static Files

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

### Running with Uvicorn

You can also start the server as a uvicorn server by running:
```bash
# arguments are passed in the environment variables, e.g. HYPHA_ENABLE_SERVER_APPS=true
python -m uvicorn hypha.server:app --host=0.0.0.0 --port=9527
```

## Advanced Authentication and Authorization

### User Authentication and Service Authorization

Hypha provides built-in support for user authentication, and based on the user context, you can also customize the service authorization within each service function.

In the basic examples, we registered public services (`config.visibility = "public"`) that can be accessed by any client. If you want to limit service access to a subset of clients, there are two ways to provide authorization:

1. **Connecting to the Same Workspace**: Set `config.visibility` to `"private"`. Authorization is achieved by generating a token from the client that registered the service (using `server.config.workspace` and `server.generate_token()`). Another client can connect to the same workspace using the token (`connect_to_server({"workspace": xxxx, "token": xxxx, "server_url": xxxx})`).

2. **Using User Context**: When registering a service, set `config.require_context` to `True` and `config.visibility` to `"public"` (or `"private"` to limit access for clients from the same workspace). Each service function needs to accept a keyword argument called `context`. The server will provide the context information containing `user` for each service function call. The service function can then check whether `context.user["id"]` is allowed to access the service. On the client side, you need to log in and generate a token by calling the `login({"server_url": xxxx})` function. The token is then used in `connect_to_server({"token": xxxx, "server_url": xxxx})`.

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

### Advanced Login Options

You can specify additional arguments in the `login()` function:

```python
token = await login(
    {
        "server_url": SERVER_URL,
        "workspace": "my-workspace",
        "expires_in": 3600,
        "login_callback": login_callback,
        "login_timeout": 3,
    }
)
```

If a token is generated for a specific workspace, when calling `connect_to_server`, you need to specify the same workspace as well:

```python
async with connect_to_server({"server_url": "https://ai.imjoy.io", "token": token, "workspace": "my-workspace"}) as server:
    # ...use the server api...
    pass
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

By default, Hypha server uses a user authentication system based on [Auth0](https://auth0.com). You can configure your own Auth0 account or implement custom authentication providers. See [Authentication](./auth) for complete details on authentication options.

## HTTP Proxy and API Access

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

### Client Shortcuts

Besides connecting and then calling `get_service`, you can use handy shortcuts.

#### Python
```python
from hypha_rpc import get_remote_service

# URL format: http://<server>/<workspace>/services/<client_id>:<service_id>
# client_id is optional
svc = await get_remote_service(
    "http://localhost:9527/ws-user-xxx/services/CLIENTID:hello-world"
)
print(await svc.hello("World"))
```

#### JavaScript
```javascript
// Equivalent one-call access from JS
const svc = await hyphaWebsocketClient.getRemoteService(
  "http://localhost:9527/ws-user-xxx/services/CLIENTID:hello-world"
);
console.log(await svc.hello("World"));
```

> These shortcuts are great for quick calls and scripting without managing an explicit server connection lifecycle.

## Workspace Management

### Environment Variables

Hypha allows you to store and share configuration variables within a workspace. These environment variables are useful for sharing API keys, database URLs, and other configuration parameters between services without hardcoding them. Variables are persisted in S3 storage and shared between all authorized clients in the same workspace.

**Note:** Both `set_env` and `get_env` require `read_write` permission on the workspace.

<!-- tabs:start -->
#### ** Python (Async) **

```python
from hypha_rpc import connect_to_server

async def main():
    # Connect with a token that has read_write permission
    async with connect_to_server({
        "server_url": "https://ai.imjoy.io",
        "token": "your-token-here"
    }) as server:
        # Set environment variables
        await server.set_env("DATABASE_URL", "postgres://localhost:5432/mydb")
        await server.set_env("API_KEY", "your-secret-key")
        # Remove an environment variable
        await server.set_env("API_KEY", None)
        
        # Get a specific environment variable
        db_url = await server.get_env("DATABASE_URL")
        print(f"Database URL: {db_url}")
        
        # Get all environment variables
        all_vars = await server.get_env()
        print(f"All variables: {all_vars}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

#### ** JavaScript **

```javascript
async function main() {
    // Connect with a token that has read_write permission
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: "https://ai.imjoy.io",
        token: "your-token-here"
    });
    
    // Set environment variables
    await server.setEnv({key: "DATABASE_URL", value: "postgres://localhost:5432/mydb", _rkwargs: true});
    await server.setEnv({key: "API_KEY", value: "your-secret-key", _rkwargs: true});
    // Remove an environment variable
    await server.setEnv({key: "API_KEY", value: null, _rkwargs: true});
    // Get a specific environment variable
    const dbUrl = await server.getEnv({key: "DATABASE_URL", value: null, _rkwargs: true});
    console.log("Database URL:", dbUrl);
    
    // Get all environment variables
    const allVars = await server.getEnv();
    console.log("All variables:", allVars);
}

main();
```
<!-- tabs:end -->

## Service Monitoring and Health Checks

### Service Probes

Probes are useful for monitoring the status of services. They can be used to check whether a service is running correctly or to retrieve information about the service. Probes are executed by the server and can be used to monitor the health of services.

We create a convenient shortcut to register probes for monitoring services. Here is an example of how to register a probe for a service:

```python
from hypha_rpc import connect_to_server

async def start_server(server_url):
    async with connect_to_server({"server_url": server_url}) as server:
        # Assuming you have some services registered already
        # And you want to monitor the service status

        def check_readiness():
            # Check if the service is ready
            # Replace this with your own readiness check
            return {"status": "ok"}
        
        def check_liveness():
            # Check if the service is alive
            # Replace this with your own liveness check
            return {"status": "ok"}

        # Register probes for the service
        await server.register_probes({
            "readiness": check_readiness,
            "liveness": check_liveness,
        })

        # This will register probes service where you can accessed via hypha or the HTTP proxy
        print(f"Probes registered at workspace: {server.config.workspace}")
        print(f"Test it with the HTTP proxy: {server_url}/{server.config.workspace}/services/probes/readiness")

        # Keep the server running forever
        await server.serve()

if __name__ == "__main__":
    server_url = "https://my-hypha-server.org"
    asyncio.run(start_server(server_url))
```

For running the service as a worker in a kubernetes cluster, you can use the `liveness` probe to check if the service is alive and the `readiness` probe to check if the service is ready to serve requests.

The probes can be configured as follows:

```yaml
livenessProbe:
    exec:
        command:
        - /bin/sh
        - -c
        - curl -sf https://my-hypha-server.org/workspace/services/probes/liveness || exit 1
    initialDelaySeconds: 10
    periodSeconds: 5

readinessProbe:
    exec:
        command:
        - /bin/sh
        - -c
        - curl -sf https://my-hypha-server.org/workspace/services/probes/readiness || exit 1
    initialDelaySeconds: 10
    periodSeconds: 5
```

(Make sure to replace `https://my-hypha-server.org` with your actual hypha server URL)

## Advanced Service Features

### Application-Level Service Configuration with App ID

Hypha supports application-level configuration for services through the `app_id` parameter. This feature allows you to group distributed services under a single application manifest and share configuration settings across all services belonging to the same application.

#### Creating Application Manifests

An application manifest defines the configuration and metadata for a group of related services. You can create applications using the artifact manager and include app-level settings like `service_selection_mode`.

<!-- tabs:start -->
#### ** Python **

```python
from hypha_rpc import connect_to_server

async def create_application(server_url, token):
    async with connect_to_server({
        "server_url": server_url,
        "token": token
    }) as server:
        # Get the artifact manager
        artifact_manager = await server.get_service("public/artifact-manager")
        
        # Create an application manifest
        app_manifest = {
            "name": "my-distributed-app",
            "description": "A distributed application with multiple service instances",
            "type": "application",
            "service_selection_mode": "random",  # How to select between multiple instances
            "services": [
                {
                    "id": "worker-service",
                    "name": "Worker Service",
                    "type": "compute",
                    "description": "A compute worker service"
                },
                {
                    "id": "storage-service", 
                    "name": "Storage Service",
                    "type": "storage",
                    "description": "A storage service"
                }
            ]
        }
        
        # Create the application artifact
        app_artifact = await artifact_manager.create(
            type="application",
            manifest=app_manifest,
            alias="my-distributed-app"
        )
        
        # Commit the artifact to make it available
        await artifact_manager.commit(artifact_id=app_artifact.id)
        
        print(f"Application created with ID: {app_artifact.id}")
        return app_artifact

if __name__ == "__main__":
    import asyncio
    asyncio.run(create_application("https://ai.imjoy.io", "your-token"))
```

#### ** JavaScript **

```javascript
async function createApplication(serverUrl, token) {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: serverUrl,
        token: token
    });
    
    // Get the artifact manager
    const artifactManager = await server.getService("public/artifact-manager");
    
    // Create an application manifest
    const appManifest = {
        name: "my-distributed-app",
        description: "A distributed application with multiple service instances",
        type: "application",
        service_selection_mode: "random",  // How to select between multiple instances
        services: [
            {
                id: "worker-service",
                name: "Worker Service",
                type: "compute",
                description: "A compute worker service"
            },
            {
                id: "storage-service",
                name: "Storage Service", 
                type: "storage",
                description: "A storage service"
            }
        ]
    };
    
    // Create the application artifact
    const appArtifact = await artifactManager.create({
        type: "application",
        manifest: appManifest,
        alias: "my-distributed-app",
        _rkwargs: true
    });
    
    // Commit the artifact to make it available
    await artifactManager.commit({artifact_id: appArtifact.id, _rkwargs: true});
    
    console.log(`Application created with ID: ${appArtifact.id}`);
    return appArtifact;
}

// Usage
createApplication("https://ai.imjoy.io", "your-token");
```
<!-- tabs:end -->

#### Registering Services with App ID

Once you have created an application manifest, you can register services with the `app_id` parameter to associate them with the application. This allows the services to inherit app-level configuration.

<!-- tabs:start -->
#### ** Python **

```python
from hypha_rpc import connect_to_server

async def register_worker_service(server_url, token, app_id):
    async with connect_to_server({
        "server_url": server_url,
        "token": token
    }) as server:
        def process_task(task_data):
            # Process the task
            print(f"Processing task: {task_data}")
            return f"Task completed: {task_data}"
        
        def get_status():
            return {"status": "ready", "load": 0.5}
        
        # Register service with app_id
        service_info = await server.register_service({
            "id": "worker-service-instance-1",
            "name": "Worker Service Instance 1",
            "type": "compute",
            "app_id": app_id,  # Associate with the application
            "config": {
                "visibility": "public"
            },
            "process_task": process_task,
            "get_status": get_status
        })
        
        print(f"Service registered: {service_info.id}")
        print(f"Associated with app: {app_id}")
        
        # Keep the service running
        await server.serve()

if __name__ == "__main__":
    import asyncio
    # Use the app_id from the application you created
    asyncio.run(register_worker_service("https://ai.imjoy.io", "your-token", "my-distributed-app"))
```

#### ** JavaScript **

```javascript
async function registerWorkerService(serverUrl, token, appId) {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: serverUrl,
        token: token
    });
    
    function processTask(taskData) {
        // Process the task
        console.log(`Processing task: ${taskData}`);
        return `Task completed: ${taskData}`;
    }
    
    function getStatus() {
        return {status: "ready", load: 0.5};
    }
    
    // Register service with app_id
    const serviceInfo = await server.registerService({
        id: "worker-service-instance-1",
        name: "Worker Service Instance 1",
        type: "compute",
        app_id: appId,  // Associate with the application
        config: {
            visibility: "public"
        },
        process_task: processTask,
        get_status: getStatus
    });
    
    console.log(`Service registered: ${serviceInfo.id}`);
    console.log(`Associated with app: ${appId}`);
}

// Usage
registerWorkerService("https://ai.imjoy.io", "your-token", "my-distributed-app");
```
<!-- tabs:end -->

#### Service Selection Modes

The `service_selection_mode` in the application manifest controls how Hypha selects between multiple service instances when accessing services by name. This is particularly useful for distributed applications where you may have multiple instances of the same service running.

**Available selection modes:**
- `"random"`: Randomly select from available instances
- `"first"`: Always use the first available instance
- `"last"`: Always use the last available instance
- `"exact"`: Require exact service ID match (default behavior)

#### Using Services with App-Level Configuration

When you have multiple service instances registered with the same `app_id`, you can access them using the service name with the app ID, and Hypha will automatically apply the selection mode from the application manifest.

<!-- tabs:start -->
#### ** Python **

```python
from hypha_rpc import connect_to_server

async def use_distributed_service(server_url, token):
    async with connect_to_server({
        "server_url": server_url,
        "token": token
    }) as server:
        # Access service by name with app_id
        # Hypha will use the service_selection_mode from the application manifest
        worker_service = await server.get_service("worker-service@my-distributed-app")
        
        # Call the service - Hypha will automatically select an instance
        # based on the app's service_selection_mode (e.g., "random")
        result = await worker_service.process_task("important-computation")
        print(f"Result: {result}")
        
        # Check status of the selected instance
        status = await worker_service.get_status()
        print(f"Service status: {status}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(use_distributed_service("https://ai.imjoy.io", "your-token"))
```

#### ** JavaScript **

```javascript
async function useDistributedService(serverUrl, token) {
    const server = await hyphaWebsocketClient.connectToServer({
        server_url: serverUrl,
        token: token
    });
    
    // Access service by name with app_id
    // Hypha will use the service_selection_mode from the application manifest
    const workerService = await server.getService("worker-service@my-distributed-app");
    
    // Call the service - Hypha will automatically select an instance
    // based on the app's service_selection_mode (e.g., "random")
    const result = await workerService.process_task("important-computation");
    console.log(`Result: ${result}`);
    
    // Check status of the selected instance
    const status = await workerService.get_status();
    console.log(`Service status: ${status}`);
}

// Usage
useDistributedService("https://ai.imjoy.io", "your-token");
```
<!-- tabs:end -->

#### Benefits of App-Level Configuration

1. **Centralized Configuration**: All services belonging to an application share the same configuration, making it easier to manage distributed systems.

2. **Automatic Load Balancing**: With `service_selection_mode` set to `"random"`, requests are automatically distributed across available service instances.

3. **Service Discovery**: Services can be accessed by logical names rather than specific instance IDs, making the system more flexible.

4. **Validation**: Hypha validates that the `app_id` exists before registering services, preventing configuration errors.

5. **Scalability**: You can easily add or remove service instances without changing client code.

This feature is particularly useful for:
- **Microservices architectures** where you have multiple instances of the same service
- **Distributed computing** where you want to balance load across worker nodes
- **High availability setups** where you need redundant service instances
- **Development environments** where you want to easily switch between different service configurations

## Custom Initialization and Startup Functions

### Custom Initialization and Service Integration with Hypha Server

Hypha's flexibility allows services to be registered from scripts running on the same host as the server or on a different one. To further accommodate complex applications, Hypha supports the initiation of "built-in" services in conjunction with server startup. This can be achieved using the `--startup-functions` option.

The `--startup-functions` option allows you to provide a URI pointing to a Python function intended for custom server initialization tasks. The specified function can perform various tasks, such as registering services, configuring the server, implementing custom authentication, or launching additional processes. The URI should follow the format `<python module or script file>:<entrypoint function name>`, providing a straightforward way to customize your server's startup behavior.

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

    # You can also configure custom authentication here
    # await server["set_parse_token_function"](custom_parse_token)
    # await server["set_generate_token_function"](custom_generate_token)

    # Note: In startup functions, the server connection is managed by Hypha,
    # so you don't need to use async with or call disconnect()
```

Note that the startup function file will be loaded as a Python module. You can also specify an installed Python module by using the format `my_pip_module:hypha_startup`. In both cases, make sure to specify the entrypoint function name (`hypha_startup` in this case). The function should accept a single positional argument, `server`, which represents the server object used in the client script.

Multiple startup functions can be specified by providing additional `--startup-functions` arguments. For example, to specify two startup functions, use the following command:

```bash
python -m hypha.server --host=0.0.0.0 --port=9527 --startup-functions=./example-startup-function.py:hypha_startup ./example-startup-function2.py:hypha_startup
```

For implementing custom authentication providers via startup functions, see the [Authentication](./auth#custom-authentication-providers) documentation.

## Custom Workers

### Creating Custom Workers

You can also create custom workers that extend Hypha's capabilities to run different types of applications. Workers can be implemented in Python or JavaScript and registered as services.

#### Python Worker Example

Here's a simple Python worker that can be registered as a startup function:

```python
"""Custom worker example for Hypha."""

from hypha.workers.base import BaseWorker, WorkerConfig, SessionInfo, SessionStatus
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CustomWorker(BaseWorker):
    """Custom worker for executing tasks."""
    
    def __init__(self):
        super().__init__()
        self.running_tasks = {}
    
    @property
    def supported_types(self) -> List[str]:
        return ["custom-task", "background-job"]
    
    @property
    def name(self) -> str:
        return "Custom Worker"
    
    @property
    def description(self) -> str:
        return "A custom worker for executing tasks"
    
    async def _start_session(self, config: WorkerConfig) -> Dict[str, Any]:
        """Start a custom worker session."""
        logger.info(f"Starting custom session for app {config.app_id}")
        
        # Your custom session logic here
        task_info = {
            "status": "running",
            "logs": [f"Started task for {config.app_id}"],
            "result": None
        }
        
        return {
            "task_info": task_info,
            "local_url": f"http://localhost:8000/tasks/{config.app_id}",
            "public_url": f"{config.public_base_url}/tasks/{config.app_id}",
        }
    
    async def _stop_session(self, session_id: str) -> None:
        """Stop a custom worker session."""
        logger.info(f"Stopping custom session {session_id}")
        # Your cleanup logic here
    
    async def _get_session_logs(
        self, 
        session_id: str, 
        log_type: Optional[str] = None,
        offset: int = 0,
        limit: Optional[int] = None
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Get logs for a custom worker session."""
        session_data = self._session_data.get(session_id, {})
        logs = session_data.get("task_info", {}).get("logs", [])
        
        if log_type:
            return logs[offset:offset+limit] if limit else logs[offset:]
        else:
            return {"log": logs[offset:offset+limit] if limit else logs[offset:]}

async def hypha_startup(server):
    """Hypha startup function to register the custom worker."""
    
    # Create and initialize the custom worker
    custom_worker = CustomWorker()
    await server.register_service(custom_worker.get_service())
    
    logger.info("Custom worker registered successfully")
```

#### JavaScript Worker Example

For JavaScript workers, you can create a simple standalone script:

```html
<script src="https://cdn.jsdelivr.net/npm/hypha-rpc@0.20.85/dist/hypha-rpc-websocket.min.js"></script>
```
    
```javascript
// custom-worker.js
const { connectToServer } = hyphaWebsocketClient;

// Worker functions
async function start(config) {
    const sessionId = `${config.workspace}/${config.client_id}`;
    console.log(`Starting session ${sessionId} for app ${config.app_id}`);
    
    // Your custom session startup logic here
    const sessionData = {
        status: "running",
        created_at: new Date().toISOString(),
        logs: [`Session ${sessionId} started`]
    };
    
    // Store session data (use proper storage in production)
    global.sessionStorage = global.sessionStorage || {};
    global.sessionStorage[sessionId] = sessionData;
    
    return {
        session_id: sessionId,
        app_id: config.app_id,
        workspace: config.workspace,
        client_id: config.client_id,
        status: "running",
        app_type: config.app_type,
        created_at: sessionData.created_at,
        metadata: config.metadata
    };
}

async function stop(sessionId) {
    console.log(`Stopping session ${sessionId}`);
    if (global.sessionStorage[sessionId]) {
        global.sessionStorage[sessionId].status = "stopped";
    }
}

async function listSessions(workspace) {
    const sessions = [];
    for (const [sessionId, sessionData] of Object.entries(global.sessionStorage || {})) {
        if (sessionId.startsWith(workspace + "/")) {
            sessions.push({
                session_id: sessionId,
                status: sessionData.status,
                created_at: sessionData.created_at
            });
        }
    }
    return sessions;
}

async function getLogs(sessionId) {
    const sessionData = global.sessionStorage[sessionId];
    if (!sessionData) {
        throw new Error(`Session ${sessionId} not found`);
    }
    return { log: sessionData.logs || [], error: [] };
}

// Register the worker
async function registerWorker() {
    try {
        const server = await connectToServer({
            server_url: "http://localhost:9527"
        });
        
        const workerService = await server.registerService({
            name: "JavaScript Custom Worker",
            description: "A custom worker implemented in JavaScript",
            type: "server-app-worker",
            config: {
                visibility: "protected",
                run_in_executor: true,
            },
            supported_types: ["javascript-custom", "js-worker"],
            start: start,
            stop: stop,
            get_logs: getLogs,
        });
        
        console.log(`JavaScript worker registered: ${workerService.id}`);
        return server;
        
    } catch (error) {
        console.error("Failed to register worker:", error);
        throw error;
    }
}

// Start the worker
registerWorker().then(server => {
    console.log("JavaScript worker is running...");
}).catch(error => {
    console.error("Worker startup failed:", error);
    process.exit(1);
});
```

Run the JavaScript worker with:

```bash
npm install hypha-rpc
node custom-worker.js
```

This approach makes it simple to create custom workers without complex wrapper classes - you just implement the required functions and register them as a service.

## Advanced Communication Features

### Peer-to-Peer Connection via WebRTC

For large data or low-latency streaming between clients, Hypha supports P2P via WebRTC. You can establish a WebRTC channel after a normal connection. See the hypha-rpc docs for details on P2P setup: [WebRTC support](/hypha-rpc?id=peer-to-peer-connection-via-webrtc).

## Production Deployment

### Using a Public Server

Alternatively, you can use our public testing server at [https://ai.imjoy.io](https://ai.imjoy.io) for development and testing purposes.

### Kubernetes Deployment

For production deployments, see the [Kubernetes guide](k8s.md) for detailed instructions on deploying Hypha in a Kubernetes cluster.

### Docker Deployment

For containerized deployments, see the Dockerfile in the repository root and the docker-compose examples in the `docker-compose/` directory.

## Compatibility and Versions

- If you use Hypha older than 0.15.x, use `imjoy-rpc` instead of `hypha-rpc`.
- JavaScript CDN examples reference `hypha-rpc@0.20.85`. For the latest, prefer npm and pin versions per your policy.
- When mixing async and sync Python code, prefer async; use `hypha_rpc.sync` wrappers only when necessary.
- Server apps require Playwright setup: `pip install -U "hypha[server-apps]" && playwright install`.

## Next Steps

Now that you've learned about Hypha's advanced features, you might want to explore:

- **[Server Apps](apps.md)** - Build web applications that run on the server
- **[Artifact Management](artifact-manager.md)** - Store and version your data and models  
- **[Vector Search](vector-search.md)** - Add semantic search capabilities
- **[Kubernetes Deployment](k8s.md)** - Deploy Hypha in production environments
