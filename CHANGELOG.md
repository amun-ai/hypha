# Hypha Change Log

### 0.20.39

 - Revise artifact manager to use artifact id as the primary key, remove `prefix` based keys.
 - Support versioning and custom config (e.g. artifact specific s3 credentials) for the artifact manager.
 - Use SQLModel and support database migration using `alembic`.

### 0.20.38

 - Support event logging in the workspace, use `log_event` to log events in the workspace and use `get_events` to get the events in the workspace. The events will be persists in the SQL database.
 - Allow passing workspace and expires_in to the `login` function to generate workspace specific token.
 - When using http endpoint to access the service, you can now pass workspace specific token to the http header `Authorization` to access the service. (Previously, all the services are assumed to be accessed from the same service provider workspace)
 - Breaking Change: Remove `info`, `warning`, `error`, `critical`, `debug` from the `hypha` module, use `log` or `log_event` instead.
 - Support basic observability for the workspace, including workspace status, event bus and websocket connection status.
 - Support download statistics for the artifacts in the artifact manager.
 - Change http endpoint from `/{workspace}/artifact/{artifact_id}` to `/{workspace}/artifacts/{artifact_id}` to make it consistent with the other endpoints.

### 0.20.37
 - Add s3-proxy to allow accessing s3 presigned url in case the s3 server is not directly accessible. Use `--enable-s3-proxy` to enable the s3 proxy when starting Hypha.
 - Add `artifact-manager` service to provide comprehensive artifact management, used for creating gallery-like service portal. The artifact manager service is backed by s3 storage and supports presigned url for direct access to the artifacts. This is a replacement of the previous `card` service.

### 0.20.36

 - Upgrade hypha-rpc to support updating reconnection token (otherwise it generate token expired error after some time)

### 0.20.35

 - Upgrade hypha-rpc to fix reset timer

### 0.20.34
 
 - Fix persistent workspace unloaded issue when s3 is not available.
 - Improve ASGI support for streaming response.

### 0.20.33

 - Add `delete_workspace` to the workspace api.
 - Add workspaces panel to the web ui.

### 0.20.31

 - Upgrade hypha-rpc to fix ssl issue with the hypha-rpc client.

### 0.20.30

 - Fix server crashing bug when websocket.send is called after the connection is closed.

### 0.20.20

 - Fix static files not included in the package

### 0.20.19

 - Support invoke token
 - Add basic web ui for the workspace
 - BREAKING Change: Change the signature, now you need to pass a dictionary as options for `get_service`, `get_service_info`, `register_service` etc.

### 0.20.15

 - Add `revoke_token` to the workspace api.
 - Simplify http endpoints to a fixed pattern such as "{workspace}/services/*" and "{workspace}/apps/*".
 - To avoid naming convension, workspace names now must contain at least one hyphens, and only lowercase letters, numbers and hyphens are allowed.

### 0.20.14

 - Make `get_service` more restricted to support only service id string, see [migration guide](./docs/migration-guide.md) for more details.
 - Clean up http endpoints for the services.
 - Remove local cache of the server apps, we now always use s3 as the primary storage.

### 0.20.12

 - New Feature: In order to support large language models' function calling feature, hypha support built-in type annotation. With `hypha-rpc>=0.20.12`, we also support type annotation for the service functions in JSON Schema format. In Python, you can use `Pydantic` or simple python type hint, or directly write the json schema for Javascript service functions. This allows you to specify the inputs spec for functions.
 - Add type support for the `hypha` module. It allows you to register a type in the workspace using `register_service_type`, `get_service_type`, `list_service_types`. When registering a new service, you can specify the type and enable type check for the service. The type check will be performed when calling the service function. The type check is only available in Python.
 - Fix reconnecton issue in the client.
 - Support case conversion, which allows converting the service functions to snake_case or camelCase in `get_service` (Python) or `getService` (JavaScript).
 - **Breaking Changes**: In Python, all the function names uses snake case, and in JavaScript, all the function names uses camel case. For example, you should call `server.getService` instead of `server.get_service` in JavaScript, and `server.get_service` instead of `server.getService` in Python.
 - **Breaking Changes**: The new version of Hypha (0.20.0+) improves the RPC connection to make it more stable and secure, most importantly it supports automatic reconnection when the connection is lost. This also means breaking changes to the previous version. In the new version you will need a new library called `hypha-rpc` (instead of the hypha submodule in the `imjoy-rpc` module) to connect to the server.

