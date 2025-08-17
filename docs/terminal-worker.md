# Hypha Terminal Worker

The Hypha Terminal Worker is a powerful service that enables executing shell commands and running interactive terminal sessions within the Hypha ecosystem. This worker provides full terminal emulation capabilities, allowing users to run bash scripts, Python interpreters, and other command-line tools remotely.

## ⚠️ CRITICAL SECURITY WARNING

**The Terminal Worker provides unrestricted command execution capabilities. This means it can:**
- Execute ANY shell command with the permissions of the user running the worker
- Access ALL files readable by the worker process
- Connect to databases, internal services, and network resources
- Read environment variables and secrets
- Modify system files if permissions allow
- Install software and execute arbitrary code

**ONLY enable this worker if you:**
- Fully trust all users who will have access
- Have proper network isolation and security controls
- Understand the security implications
- Have audit logging and monitoring in place
- Run the worker with minimal necessary permissions

**For production environments, consider:**
- Running in isolated containers or VMs
- Using restricted user accounts with minimal permissions
- Implementing network segmentation
- Enabling comprehensive audit logging
- Using authorization controls to limit access to specific workspaces

## Overview

The Terminal Worker provides:
- **Full terminal emulation**: Complete pseudo-terminal (PTY) support for interactive sessions
- **Multi-shell support**: Run bash, sh, zsh, Python interpreters, or any shell
- **Session persistence**: Long-running terminal sessions that maintain state
- **Clean output rendering**: Automatic removal of ANSI escape codes for readable output
- **Screen state tracking**: Get the current terminal screen content at any time
- **File staging**: Automatic download of files from artifact storage
- **Working directory isolation**: Each session gets its own isolated working directory
- **Real-time command execution**: Execute commands and get immediate feedback

### Key Features

- **Interactive Sessions**: Maintain persistent terminal sessions similar to SSH
- **Command Execution**: Run individual commands or scripts with timeout control
- **Output Cleaning**: Automatic removal of terminal control sequences for clean output
- **Raw Mode**: Optional raw output with all ANSI codes preserved
- **Screen Rendering**: Get the terminal screen as it would appear when copying from a terminal
- **Session Management**: Start, stop, and manage multiple isolated sessions

## Prerequisites

- Python 3.9 or higher
- `hypha` package installed (`pip install hypha`)
- `ptyprocess` package for terminal emulation
- Network access to the Hypha server
- Valid authentication token for the target workspace

## Quick Start

### Starting the Terminal Worker

The Terminal Worker can be started as a standalone service or as a built-in worker:

#### Standalone Worker (CLI)

```bash
# Basic usage with command line arguments
python -m hypha.workers.terminal \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token your-token-here

# Using environment variables
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace  
export HYPHA_TOKEN=your-token-here
python -m hypha.workers.terminal --verbose

# With custom working directory and authorization
python -m hypha.workers.terminal \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token your-token-here \
  --working-dir /opt/terminal-sessions \
  --authorized-workspaces workspace1,workspace2 \
  --visibility protected
```

#### Built-in Worker (Startup Function)

```bash
# Start Hypha server with terminal worker enabled
export TERMINAL_WORKING_DIR=/tmp/terminal-sessions
export TERMINAL_AUTHORIZED_WORKSPACES=trusted-workspace
python -m hypha.server --startup-functions hypha.workers.terminal:hypha_startup
```

## Configuration Options

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--server-url` | Hypha server URL | Yes | - |
| `--workspace` | Workspace name | Yes | - |
| `--token` | Authentication token | Yes | - |
| `--service-id` | Service ID for the worker | No | Auto-generated |
| `--client-id` | Client ID for the worker | No | Auto-generated |
| `--visibility` | Service visibility: `public` or `protected` | No | `protected` |
| `--working-dir` | Base directory for session working directories | No | `/tmp/hypha_sessions_<uuid>` |
| `--authorized-workspaces` | Comma-separated list of authorized workspaces | No | `root` |
| `--use-local-url` | Use local URLs for server communication | No | `false` |
| `--disable-ssl` | Disable SSL verification | No | `false` |
| `--verbose` / `-v` | Enable verbose logging | No | `false` |

### Environment Variables

All command line arguments can be set using environment variables:

| Environment Variable | Equivalent Argument |
|---------------------|-------------------|
| `HYPHA_SERVER_URL` | `--server-url` |
| `HYPHA_WORKSPACE` | `--workspace` |
| `HYPHA_TOKEN` | `--token` |
| `HYPHA_SERVICE_ID` | `--service-id` |
| `HYPHA_CLIENT_ID` | `--client-id` |
| `HYPHA_VISIBILITY` | `--visibility` |
| `TERMINAL_WORKING_DIR` | `--working-dir` |
| `TERMINAL_AUTHORIZED_WORKSPACES` | `--authorized-workspaces` |

## Usage Examples

### Basic Terminal Session

```python
import asyncio
from hypha_rpc import connect_to_server

async def run_terminal_session():
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    # Get the server apps controller
    controller = await server.get_service("public/server-apps")
    
    # Install a terminal app
    app_info = await controller.install(
        source="#!/bin/bash\necho 'Terminal ready!'",
        manifest={
            "name": "My Terminal Session",
            "type": "terminal",
            "startup_command": "/bin/bash"
        },
        overwrite=True
    )
    
    # Start the terminal session
    session_info = await controller.start(app_info["id"])
    session_id = session_info["id"]
    
    # Get the terminal worker directly
    workers = await server.list_services({"type": "server-app-worker"})
    terminal_workers = [w for w in workers if "terminal" in w.get("supported_types", [])]
    
    if terminal_workers:
        worker = await server.get_service(terminal_workers[0]["id"])
        
        # Execute commands
        result = await worker.execute(session_id, "ls -la")
        print("Output:", result["outputs"][0]["text"])
        
        # Execute with raw output (includes ANSI codes)
        result = await worker.execute(
            session_id, 
            "echo -e '\\033[31mRed text\\033[0m'",
            config={"raw_output": True}
        )
        print("Raw output:", result["outputs"][0]["text"])
        
        # Get the current screen content (cleaned)
        screen_log = await worker.get_logs(session_id, type="screen")
        print("Current screen log:", screen_log)
    
    # Stop the session
    await controller.stop(session_id)

asyncio.run(run_terminal_session())
```

### Interactive Python Session

```python
async def python_interactive_session():
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    controller = await server.get_service("public/server-apps")
    
    # Create an interactive Python terminal
    app_info = await controller.install(
        source="# Python interactive terminal",
        manifest={
            "name": "Python REPL",
            "type": "terminal",
            "startup_command": "python3 -i"
        },
        overwrite=True
    )
    
    session_info = await controller.start(app_info["id"])
    session_id = session_info["id"]
    
    # Get the terminal worker
    workers = await server.list_services({"type": "server-app-worker"})
    terminal_workers = [w for w in workers if "terminal" in w.get("supported_types", [])]
    
    if terminal_workers:
        worker = await server.get_service(terminal_workers[0]["id"])
        
        # Execute Python code interactively
        result = await worker.execute(session_id, "import numpy as np")
        result = await worker.execute(session_id, "arr = np.array([1, 2, 3, 4, 5])")
        result = await worker.execute(session_id, "print(f'Mean: {arr.mean()}')")
        
        # Clean output without command echo
        print("Output:", result["outputs"][0]["text"])  # Just "Mean: 3.0"
        
        # Get the full screen to see all interactions
        screen = await worker.get_logs(session_id, type="screen")
        print("Full session:", screen)
    
    await controller.stop(session_id)

asyncio.run(python_interactive_session())
```

### File Staging Example

```python
async def terminal_with_files():
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    controller = await server.get_service("public/server-apps")
    
    # Terminal app with file staging
    app_info = await controller.install(
        source="#!/bin/bash\nls -la\ncat data.txt",
        manifest={
            "name": "Terminal with Files",
            "type": "terminal",
            "startup_command": "/bin/bash",
            "files_to_stage": [
                "data.txt",              # Download single file
                "configs/",              # Download entire folder
                "input.csv:output.csv"   # Download and rename
            ]
        },
        overwrite=True
    )
    
    # Files will be automatically downloaded before the session starts
    session_info = await controller.start(app_info["id"])
    
    # The files are now available in the session's working directory
    workers = await server.list_services({"type": "server-app-worker"})
    terminal_workers = [w for w in workers if "terminal" in w.get("supported_types", [])]
    
    if terminal_workers:
        worker = await server.get_service(terminal_workers[0]["id"])
        
        # Files are available immediately
        result = await worker.execute(session_info["id"], "cat data.txt")
        print("File content:", result["outputs"][0]["text"])
    
    await controller.stop(session_info["id"])

asyncio.run(terminal_with_files())
```

## Terminal Worker as a Foundation Service

The Terminal Worker can serve as a foundational service for bootstrapping other workers and services:

### Bootstrap Other Workers

```python
async def bootstrap_conda_worker():
    """Use terminal worker to install and start a conda worker."""
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    controller = await server.get_service("public/server-apps")
    
    # Start a terminal session
    app_info = await controller.install(
        source="#!/bin/bash\necho 'Ready to install workers'",
        manifest={
            "name": "Worker Bootstrap Terminal",
            "type": "terminal",
            "startup_command": "/bin/bash"
        },
        overwrite=True
    )
    
    session_info = await controller.start(app_info["id"])
    session_id = session_info["id"]
    
    # Get terminal worker
    workers = await server.list_services({"type": "server-app-worker"})
    terminal_workers = [w for w in workers if "terminal" in w.get("supported_types", [])]
    
    if terminal_workers:
        worker = await server.get_service(terminal_workers[0]["id"])
        
        # Install conda if not present
        result = await worker.execute(session_id, """
            if ! command -v conda &> /dev/null; then
                wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
                export PATH="$HOME/miniconda3/bin:$PATH"
            fi
        """)
        
        # Install hypha and dependencies
        await worker.execute(session_id, "pip install hypha hypha-rpc")
        
        # Start a conda worker in the background
        await worker.execute(session_id, f"""
            export HYPHA_SERVER_URL={server.config.server_url}
            export HYPHA_WORKSPACE={server.config.workspace}
            export HYPHA_TOKEN={server.config.token}
            export HYPHA_SERVICE_ID=my-conda-worker
            nohup python -m hypha.workers.conda > conda_worker.log 2>&1 &
        """)
        
        print("Conda worker started successfully!")
    
    # The terminal session can be stopped while the conda worker continues running
    await controller.stop(session_id)

asyncio.run(bootstrap_conda_worker())
```

## API Reference

### Worker Service Methods

#### `start(config: WorkerConfig) -> str`
Start a new terminal session.

**Parameters:**
- `config`: Worker configuration including manifest with `startup_command`

**Returns:** Session ID string

#### `execute(session_id: str, script: str, config: Optional[Dict] = None) -> Dict`
Execute a command in the terminal session.

**Parameters:**
- `session_id`: The session ID from `start()`
- `script`: Command or script to execute
- `config`: Optional configuration
  - `timeout`: Execution timeout in seconds (default: 5.0)
  - `raw_output`: If True, return raw output with ANSI codes (default: False)

**Returns:** Dictionary with execution results:
```python
{
    "status": "ok",  # or "error"
    "outputs": [
        {
            "type": "stream",
            "name": "stdout",
            "text": "cleaned output"  # or raw output if raw_output=True
        },
        {
            "type": "screen",
            "name": "screen_diff",
            "text": "what changed on screen"
        }
    ]
}
```

#### `stop(session_id: str) -> None`
Stop a terminal session and clean up resources.

#### `get_logs(session_id: str, type: str = None) -> dict`
Get logs for a terminal session.

**Parameters:**
- `session_id`: Session ID
- `type`: Log type filter. Special value `"screen"` returns rendered terminal screen

**Returns:**
- If `type="screen"`: Cleaned terminal screen content as string
- Otherwise: Log entries as list or dict

#### `resize_terminal(session_id: str, rows: int, cols: int) -> Dict`
Resize the terminal window dimensions.

#### `read_terminal(session_id: str) -> Dict`
Read any pending output from the terminal.

## Output Cleaning and Rendering

The Terminal Worker provides intelligent output cleaning:

### Default Behavior (Cleaned Output)
- Removes ANSI escape sequences and color codes
- Removes terminal control sequences (OSC codes)
- Removes command echo from output
- Removes trailing prompts (>>>, $, #)
- Returns only the actual command output

### Raw Output Mode
Set `config={"raw_output": True}` to get:
- Original output with all ANSI codes
- Command echo included
- Terminal control sequences preserved
- Useful for terminal applications that need formatting

### Screen Rendering
The `get_logs(session_id, type="screen")` method returns:
- Rendered terminal screen as plain text
- Similar to selecting and copying from a terminal
- All ANSI codes removed
- Proper handling of carriage returns and overwrites

## Security Best Practices

### 1. Workspace Isolation
```bash
# Limit access to specific workspaces
python -m hypha.workers.terminal \
  --authorized-workspaces trusted-workspace1,trusted-workspace2 \
  --visibility protected
```

### 2. Run with Minimal Permissions
```bash
# Create a restricted user
sudo useradd -m -s /bin/bash terminal-worker
sudo -u terminal-worker python -m hypha.workers.terminal ...
```

### 3. Container Isolation
```dockerfile
FROM python:3.11-slim
RUN pip install hypha
RUN useradd -m worker
USER worker
CMD ["python", "-m", "hypha.workers.terminal"]
```

### 4. Network Segmentation
- Run the worker in an isolated network segment
- Use firewall rules to restrict outbound connections
- Monitor and log all network activity

### 5. Audit Logging
- Enable comprehensive logging with `--verbose`
- Monitor all executed commands
- Set up alerts for suspicious activity

## Deployment Examples

### Docker Deployment

```yaml
version: '3.8'

services:
  terminal-worker:
    image: python:3.11-slim
    command: |
      bash -c "
        pip install hypha &&
        python -m hypha.workers.terminal --verbose
      "
    environment:
      - HYPHA_SERVER_URL=https://hypha.aicell.io
      - HYPHA_WORKSPACE=my-workspace
      - HYPHA_TOKEN=${HYPHA_TOKEN}
      - HYPHA_SERVICE_ID=terminal-worker
      - HYPHA_VISIBILITY=protected
      - TERMINAL_AUTHORIZED_WORKSPACES=trusted-workspace
    volumes:
      - /tmp/terminal-sessions:/tmp/terminal-sessions
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: terminal-worker
spec:
  replicas: 1
  selector:
    matchLabels:
      app: terminal-worker
  template:
    metadata:
      labels:
        app: terminal-worker
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: terminal-worker
        image: python:3.11-slim
        command: ["python", "-m", "hypha.workers.terminal", "--verbose"]
        env:
        - name: HYPHA_SERVER_URL
          value: "https://hypha.aicell.io"
        - name: HYPHA_WORKSPACE
          value: "my-workspace"
        - name: HYPHA_TOKEN
          valueFrom:
            secretKeyRef:
              name: hypha-secrets
              key: token
        - name: TERMINAL_AUTHORIZED_WORKSPACES
          value: "trusted-workspace"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "100m"
      volumes:
      - name: tmp
        emptyDir: {}
```

## Troubleshooting

### Common Issues

#### 1. Process Not Alive
```
Error: Terminal process for session X is not alive
```
**Solution**: The terminal process has terminated. Start a new session.

#### 2. Command Timeout
```
Execution timed out after X seconds
```
**Solution**: Increase timeout in config: `{"timeout": 30.0}`

#### 3. Permission Denied
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Check file permissions and user privileges

#### 4. Working Directory Issues
```
Failed to create session working directory
```
**Solution**: Ensure the working directory is writable:
```bash
mkdir -p /tmp/terminal-sessions
chmod 755 /tmp/terminal-sessions
```

## Limitations and Considerations

1. **Security**: Full command execution - use with extreme caution
2. **Resource Usage**: Each session maintains a persistent process
3. **Network Latency**: Commands execute remotely, network delays apply
4. **Session Persistence**: Sessions are lost if the worker restarts
5. **Output Buffering**: Large outputs may be truncated in screen buffer
6. **Interactive Programs**: Some interactive programs may not work perfectly

## Support

For issues and questions:
- GitHub Issues: [https://github.com/amun-ai/hypha/issues](https://github.com/amun-ai/hypha/issues)
- Documentation: [https://docs.hypha.app/](https://docs.hypha.app/)

## Final Security Reminder

⚠️ **The Terminal Worker grants full command execution privileges. It can access databases, read secrets, modify files, and execute arbitrary code. Only enable this worker in trusted environments with proper security controls.**

Consider using more restricted alternatives when possible:
- **Conda Worker**: For Python code execution with package management
- **Container-based Workers**: For isolated execution environments
- **Function-specific Workers**: For limited, predefined operations

Always follow the principle of least privilege and enable only the minimum necessary capabilities for your use case.