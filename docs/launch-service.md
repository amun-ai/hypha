# Launch Services from the Command Line

## Introduction

In certain scenarios, you may need to launch external services written in other programming languages or requiring a different Python environment than your Hypha server. For example, you might want to run a Python script that creates a Hypha service or start a non-Python-based service.

Hypha offers a utility function, `launch_external_services`, that allows you to run external commands or scripts while keeping the associated Hypha service alive. This utility makes it easy to integrate and manage external services alongside your Hypha server.

## The `launch_external_services` Utility

The `launch_external_services` utility is available in the `hypha_rpc.utils.launch` module. It allows you to start external services (e.g., Python scripts or other command-line tools) and ensure that they stay running as long as the Hypha server is active.

### Use Cases

- **Launching services written in different languages**: You can use this utility to start services written in JavaScript, Go, Rust, etc.
- **Managing services in separate Python environments**: If you need a different environment than the one used by your Hypha server (e.g., a different Python version or environment), you can use this utility to launch the service in the appropriate environment.
- **Seamless integration**: External services are managed and monitored like regular Hypha services.

## Example Usage

Below is an example of how to use the `launch_external_services` utility in a startup function. This example demonstrates how to run a Python script that creates a Hypha service:

```python
from hypha_rpc.utils.launch import launch_external_services

async def hypha_startup(server):
    # Example of launching an external service
    await launch_external_services(
        server,
        "python ./tests/example_service_script.py --server-url={server_url} --service-id=external-test-service --workspace={workspace} --token={token}",
        name="example_service_script",
        check_services=["external-test-service"],
    )
```

### Explanation

- **`server`**: The Hypha server instance.
- **Command string**: In this example, a Python script (`example_service_script.py`) is executed, which registers an external service in Hypha. The command uses placeholders like `{server_url}`, `{workspace}`, and `{token}` that are dynamically replaced with their actual values.
- **`name`**: This is the friendly name given to the external service (`example_service_script` in this case).
- **`check_services`**: This parameter specifies which services to check and keep alive (e.g., `external-test-service`).

### Command Placeholders

The command string supports several placeholders that are automatically replaced with the appropriate values during execution:

- **`{server_url}`**: The URL of the Hypha server.
- **`{workspace}`**: The workspace name associated with the Hypha server.
- **`{token}`**: The authentication token for accessing the Hypha server.

These placeholders allow you to dynamically insert values, making it easier to write reusable commands.

## Keeping Services Alive

One of the key benefits of `launch_external_services` is that it ensures the external service remains alive as long as the Hypha server is running. If the service stops unexpectedly, Hypha will attempt to restart it to maintain availability.

## Example Command with Placeholders

Here's an example of a typical command using placeholders:

```bash
python ./my_script.py --server-url={server_url} --service-id=my-service --workspace={workspace} --token={token}
```

This command launches a Python script (`my_script.py`) that registers a service with the specified `server_url`, `workspace`, and `token`. The placeholders are automatically substituted with actual values when the command is executed.

## Conclusion

The `launch_external_services` utility provides a powerful way to manage external services alongside Hypha, whether those services are written in a different programming language or require a different Python environment. This utility simplifies the process of launching and maintaining external services, ensuring they stay alive and functional throughout the lifetime of your Hypha server.

By integrating external services seamlessly into Hypha, you can extend the functionality of your system and manage diverse services with minimal overhead. Whether you’re dealing with Python scripts or other command-line tools, `launch_external_services` makes it easy to keep everything running smoothly.
