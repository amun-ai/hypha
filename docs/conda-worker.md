# Hypha Conda Environment Worker

The Hypha Conda Environment Worker is a standalone service that enables executing Python code in isolated conda environments, similar to a Jupyter notebook. This worker automatically uses **mamba** when available (falling back to conda), and can be deployed separately from the main Hypha server, allowing for flexible distributed computing setups and better resource management.

## Overview

The Conda Environment Worker provides:
- **Jupyter-like execution**: Execute Python code cells and get stdout/stderr output
- **Isolated environments**: Each session runs in a separate conda environment  
- **Dependency management**: Automatic installation of conda and pip packages
- **Environment caching**: Reuses existing environments to improve performance
- **Remote execution**: Can run on different machines from the Hypha server
- **Resource isolation**: Prevents dependency conflicts between different applications
- **Mamba support**: Automatically uses [mamba](https://mamba.readthedocs.io/) when available for faster package installation

### Why Mamba?

The worker automatically detects and prefers [mamba](https://mamba.readthedocs.io/) over conda when available. Mamba is a drop-in replacement for conda that offers:

- **Faster dependency solving**: Uses libsolv for much faster dependency resolution
- **Parallel downloads**: Downloads repository data and packages in parallel using multi-threading
- **Better performance**: Core parts implemented in C++ for maximum efficiency
- **Full compatibility**: Uses the same command line interface and package formats as conda

According to the [mamba documentation](https://raw.githubusercontent.com/mamba-org/mamba/refs/heads/main/README.md), mamba can significantly reduce environment creation time while maintaining full compatibility with the conda ecosystem.

## Prerequisites

- Python 3.9 or higher
- Conda or Mamba installed and available in PATH (Mamba is preferred for better performance)
- `hypha` package installed (`pip install hypha` or `pip install hypha[server]`)
- Network access to the Hypha server
- Valid authentication token for the target workspace

## Quick Start with CLI

The fastest way to start a Conda Environment Worker is using the built-in CLI:

```bash
# Basic usage with command line arguments
python -m hypha.workers.conda \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token your-token-here

# Or using environment variables
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace  
export HYPHA_TOKEN=your-token-here
python -m hypha.workers.conda --verbose
```

This will start a conda worker that connects to the specified Hypha server and registers itself as a service. The CLI supports various configuration options detailed in the [Configuration Options](#configuration-options) section below.

## Installation Methods

### Method 1: Manual Conda Setup

This method involves setting up your own conda environment and installing the required dependencies.

#### Step 1: Create a Dedicated Environment

**Option A: Using Mamba (Recommended for faster installation)**

```bash
# Install mamba if not already available
conda install mamba -n base -c conda-forge

# Create a new environment using mamba
mamba create -n hypha-conda-worker python=3.11 -y
conda activate hypha-conda-worker

# Install required dependencies with mamba
mamba install -c conda-forge conda-pack numpy -y
pip install hypha hypha-rpc
```

**Option B: Using Conda**

```bash
# Create a new conda environment for the worker
conda create -n hypha-conda-worker python=3.11 -y
conda activate hypha-conda-worker

# Install required dependencies
conda install -c conda-forge conda-pack numpy -y
pip install hypha hypha-rpc
```

#### Step 2: Verify Installation

```bash
# Check which package manager is available (mamba is preferred)
which mamba && echo "Mamba available" || echo "Mamba not found"
which conda && echo "Conda available" || echo "Conda not found"

# Test that conda-pack is available
python -c "import conda_pack; print('conda-pack available')"

# Check hypha installation
python -c "import hypha; print(f'Hypha version: {hypha.__version__}')"
```

#### Step 3: Start the Worker

```bash
# Using command line arguments
python -m hypha.workers.conda \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token your-token-here \
  --service-id my-conda-worker \
  --visibility public

# Or using environment variables
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace
export HYPHA_TOKEN=your-token-here
export HYPHA_SERVICE_ID=my-conda-worker
export HYPHA_VISIBILITY=public
python -m hypha.workers.conda
```

### Method 2: Using Hypha Docker Container

This method uses the official Hypha Docker container which includes all necessary dependencies.

#### Step 1: Pull the Docker Image

```bash
# Pull the latest Hypha image
docker pull ghcr.io/amun-ai/hypha:0.20.71

# Or pull a specific version
docker pull ghcr.io/amun-ai/hypha:latest
```

#### Step 2: Run the Conda Worker in Docker

```bash
# Run with command line arguments
docker run -it --rm \
  -v $(pwd)/conda_cache:/tmp/conda_cache \
  -e HYPHA_SERVER_URL=https://hypha.aicell.io \
  -e HYPHA_WORKSPACE=my-workspace \
  -e HYPHA_TOKEN=your-token-here \
  -e HYPHA_SERVICE_ID=my-conda-worker \
  -e HYPHA_VISIBILITY=public \
  -e HYPHA_CACHE_DIR=/tmp/conda_cache \
  ghcr.io/amun-ai/hypha:0.20.71 \
  python -m hypha.workers.conda --verbose

# Or create a docker-compose.yml file
```

#### Step 3: Docker Compose Setup (Optional)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  conda-worker:
    image: ghcr.io/amun-ai/hypha:0.20.71
    command: python -m hypha.workers.conda --verbose
    environment:
      - HYPHA_SERVER_URL=https://hypha.aicell.io
      - HYPHA_WORKSPACE=my-workspace
      - HYPHA_TOKEN=your-token-here
      - HYPHA_SERVICE_ID=my-conda-worker
      - HYPHA_VISIBILITY=public
      - HYPHA_CACHE_DIR=/tmp/conda_cache
    volumes:
      - ./conda_cache:/tmp/conda_cache
    restart: unless-stopped
```

Then start with:

```bash
docker-compose up -d
```

## Configuration Options

The Conda Environment Worker supports various configuration options through command line arguments or environment variables.

### Command Line Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--server-url` | Hypha server URL (e.g., https://hypha.aicell.io) | Yes | - |
| `--workspace` | Workspace name | Yes | - |
| `--token` | Authentication token | Yes | - |
| `--service-id` | Service ID for the worker | No | Auto-generated |
| `--visibility` | Service visibility: `public` or `protected` | No | `protected` |
| `--cache-dir` | Directory for caching conda environments | No | `~/.hypha_conda_cache` |
| `--working-dir` | Base directory for session working directories | No | `/tmp/hypha_sessions_<uuid>` |
| `--authorized-workspaces` | Comma-separated list of authorized workspaces | No | All workspaces |
| `--verbose` / `-v` | Enable verbose logging | No | False |

### Environment Variables

All command line arguments can be set using environment variables with the `HYPHA_` prefix:

| Environment Variable | Equivalent Argument |
|---------------------|-------------------|
| `HYPHA_SERVER_URL` | `--server-url` |
| `HYPHA_WORKSPACE` | `--workspace` |
| `HYPHA_TOKEN` | `--token` |
| `HYPHA_SERVICE_ID` | `--service-id` |
| `HYPHA_VISIBILITY` | `--visibility` |
| `HYPHA_CACHE_DIR` | `--cache-dir` |
| `HYPHA_WORKING_DIR` | `--working-dir` |
| `HYPHA_AUTHORIZED_WORKSPACES` | `--authorized-workspaces` |

### Example Configurations

#### Basic Setup
```bash
python -m hypha.workers.conda \
  --server-url https://hypha.aicell.io \
  --workspace my-workspace \
  --token abc123...
```

#### Production Setup with Custom Cache
```bash
python -m hypha.workers.conda \
  --server-url https://my-hypha-server.com \
  --workspace production \
  --token prod-token-123 \
  --service-id prod-conda-worker-01 \
  --visibility protected \
  --cache-dir /opt/conda_cache \
  --verbose
```

#### Environment Variables Setup
```bash
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_WORKSPACE=my-workspace
export HYPHA_TOKEN=your-token-here
export HYPHA_SERVICE_ID=my-conda-worker
export HYPHA_CACHE_DIR=/custom/cache/path

python -m hypha.workers.conda --verbose
```

## Usage

Once the worker is running, it can execute Python code in isolated conda environments, similar to a Jupyter notebook. Here's how to use it:

### 1. Create a Python Application

Create a Python script that will run during environment initialization:

```python
# main.py
import numpy as np
import sys
import platform

print("=== Environment Info ===")
print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
print(f"Platform: {platform.system()}")
print(f"NumPy version: {np.__version__}")

# Your initialization code here
print("Application initialized successfully!")
```

### 2. Create an Application Manifest

```yaml
# manifest.yaml
name: My Conda App
type: conda-jupyter-kernel
version: 1.0.0
description: A Python app with conda dependencies
entry_point: main.py
dependencies:
  - python=3.11
  - numpy
  - pandas
channels:
  - conda-forge
files_to_stage:  # Optional: files/folders to download from artifact manager
  - data.csv                    # Download single file
  - models/                      # Download entire folder recursively
  - source.txt:renamed.txt      # Download and rename file
  - raw_data/:processed_data/   # Download and rename folder
```

#### Files to Stage Feature

The `files_to_stage` option in the manifest allows you to automatically download files and folders from the Hypha artifact manager into the session's working directory before your application starts. This is useful for staging data files, models, configuration files, or any other resources your application needs.

**Supported formats:**
- **Single file**: `"data.csv"` - Downloads the file to the working directory
- **Folder (recursive)**: `"models/"` - Downloads all files and subfolders within the specified directory
- **Renamed file**: `"source.txt:target.txt"` - Downloads `source.txt` and saves it as `target.txt`
- **Renamed folder**: `"raw_data/:processed_data/"` - Downloads the `raw_data` folder and saves it as `processed_data`

Files are downloaded to the session's working directory, which is isolated per session and automatically cleaned up when the session ends. The download happens after environment creation but before the initialization script runs, ensuring your code has access to all staged files.

### 3. Deploy and Execute Code

```python
# Client code to use the conda worker
import asyncio
from hypha_rpc import connect_to_server

async def main():
    # Connect to Hypha server
    server = await connect_to_server({
        "server_url": "https://hypha.aicell.io",
        "workspace": "my-workspace",
        "token": "your-token-here"
    })
    
    # Get the conda worker service
    conda_worker = await server.get_service("my-conda-worker")
    
    # Start a session
    session_id = await conda_worker.start({
        "id": "my-session",
        "app_id": "my-conda-app",
        "workspace": "my-workspace",
        "manifest": {
            "type": "conda-jupyter-kernel",
            "dependencies": ["python=3.11", "numpy", "pandas"],
            "channels": ["conda-forge"],
            "entry_point": "main.py"
        },
        # ... other config fields
    })
    
    # Method 1: Execute code directly via conda worker
    code = """
import numpy as np
import pandas as pd

# Create some data
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({'values': data})

# Perform calculations
mean_val = df['values'].mean()
std_val = df['values'].std()
sum_val = df['values'].sum()

print(f"Mean: {mean_val}")
print(f"Standard deviation: {std_val}")
print(f"Sum: {sum_val}")
"""
    
    result = await conda_worker.execute(
        session_id=session_id,
        script=code,
        config={"timeout": 30.0}
    )
    
    if result["status"] == "ok":
        # Print all outputs from the execution
        for output in result.get("outputs", []):
            if output["type"] == "stream" and output["name"] == "stdout":
                print("Output:", output["text"])
    else:
        print("Error:", result.get("error", {}))
    
   
    # Stop the session
    await conda_worker.stop(session_id)

asyncio.run(main())
```

## Interactive Code Execution

The Conda Environment Worker supports **interactive code execution** via the `execute()` method, allowing you to run Python code in an already-running conda environment session, similar to executing cells in a Jupyter notebook.

### Execute Method Overview

The execute method provides Jupyter-like functionality with the following features:

- **Real-time execution**: Execute Python code in running sessions
- **Jupyter-compatible outputs**: Returns structured output including stdout, stderr, and display data
- **Progress callbacks**: Optional progress reporting during execution
- **Timeout handling**: Configurable execution timeouts
- **Error handling**: Comprehensive error reporting with tracebacks

### Execute Method API

```python

session_info = await app_contrller.start("my-conda-worker")
conda_worker = await server.get_service(session_info["worker_id"])
# Direct worker execution
result = await conda_worker.execute(
    session_id="my-session",
    script="print('Hello World!')",
    config={
        "timeout": 30.0,    # Execution timeout in seconds
    },
    progress_callback=lambda info: print(f"📍 {info['message']}")
)
```

### Execution Results Format

The execute method returns results in Jupyter notebook format:

```python
{
    "status": "ok",  # or "error"
    "outputs": [
        {
            "type": "stream",
            "name": "stdout", 
            "text": "Hello World!\n"
        },
        {
            "type": "execute_result",
            "data": {"text/plain": "42"},
            "metadata": {},
            "execution_count": 1
        }
    ],
    "error": None  # or error details if status == "error"
}
```

### Error Handling in Execute

When code execution fails, the result includes detailed error information:

```python
# Example error result
{
    "status": "error",
    "outputs": [],
    "error": {
        "ename": "NameError",
        "evalue": "name 'undefined_var' is not defined",
        "traceback": [
            "Traceback (most recent call last):",
            "  File \"<stdin>\", line 1, in <module>",
            "NameError: name 'undefined_var' is not defined"
        ]
    }
}
```

### Progress Callbacks for Execute

Monitor execution progress with callbacks:

```python
def execution_progress(info):
    """Handle execution progress updates."""
    emoji_map = {
        "info": "ℹ️",
        "success": "✅", 
        "error": "❌",
        "warning": "⚠️"
    }
    emoji = emoji_map.get(info.get("type", ""), "🔸")
    print(f"{emoji} {info.get('message', '')}")

result = await conda_worker.execute(
    session_id=session_id,
    script="import time; time.sleep(2); print('Done!')",
    progress_callback=execution_progress
)
```

## Common Use Cases

### Interactive Data Analysis

```python
# Execute code for data analysis
code = """
import numpy as np
import pandas as pd

# Create sample data
data = np.random.randn(100, 3)
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

# Perform analysis
print(f"Dataset shape: {df.shape}")
print(f"Column A mean: {df['A'].mean():.3f}")
print(f"Column B std: {df['B'].std():.3f}")
print(df.head())
"""

result = await conda_worker.execute(session_id, code)
if result["status"] == "ok":
    for output in result.get("outputs", []):
        if output["type"] == "stream" and output["name"] == "stdout":
            print(output["text"])
```

### Environment Information

```python
# Check environment details
code = """
import sys
import platform
import numpy as np

print(f"Python version: {sys.version}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"NumPy version: {np.__version__}")
print(f"Python executable: {sys.executable}")
"""

result = await conda_worker.execute(session_id, code)
if result["status"] == "ok":
    for output in result.get("outputs", []):
        if output["type"] == "stream" and output["name"] == "stdout":
            print(output["text"])
```

### Package Testing

```python
# Test if packages are available and working
code = """
try:
    import requests
    response = requests.get('https://httpbingo.org/json')
    print(f"Requests working: {response.status_code == 200}")
    print(f"Requests version: {requests.__version__}")
except ImportError as e:
    print(f"Requests not available: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"Matplotlib available: {plt.__version__}")
except ImportError as e:
    print(f"Matplotlib not available: {e}")
"""

result = await conda_worker.execute(session_id, code)
if result["status"] == "ok":
    for output in result.get("outputs", []):
        if output["type"] == "stream" and output["name"] == "stdout":
            print(output["text"])
```

### Handling Execution Results

The `execute()` method returns a dictionary with Jupyter-compatible output format:

```python
result = await conda_worker.execute(session_id, code)

if result["status"] == "ok":
    print("Code executed successfully!")
    # Extract outputs from the result
    for output in result.get("outputs", []):
        if output["type"] == "stream" and output["name"] == "stdout":
            print("Output:", output["text"])
else:
    print("Execution failed!")
    error = result.get("error", {})
    print("Error:", error.get("ename", "Unknown"))
    print("Error message:", error.get("evalue", ""))
    # Print traceback if available
    if "traceback" in error:
        print("Traceback:")
        for line in error["traceback"]:
            print(line)
```

## Environment Caching

The worker automatically caches conda environments to improve performance. When you start a session, the initialization script runs once, but the environment stays available for multiple `execute()` calls:

- **Hash-based caching**: Environments are cached based on dependencies and channels
- **LRU eviction**: Least recently used environments are removed when cache is full
- **Age-based cleanup**: Environments older than 30 days are automatically removed
- **Validation**: Cached environments are validated before reuse
- **Session isolation**: Each session gets its own Python namespace while sharing the environment

### Cache Configuration

- **Default location**: `~/.hypha_conda_cache`
- **Max cache size**: 10 environments (configurable)
- **Max age**: 30 days
- **Custom location**: Use `--cache-dir` or `HYPHA_CACHE_DIR`

## Monitoring and Logging

### Log Levels

- **INFO**: General operational information
- **DEBUG**: Detailed debugging information
- **WARNING**: Warning messages
- **ERROR**: Error messages

### Verbose Output

Enable verbose logging with `--verbose` or `-v`:

```bash
python -m hypha.workers.conda --verbose
```

### Sample Output

```
Starting Hypha Conda Environment Worker...
  Package Manager: mamba
  Server URL: https://hypha.aicell.io
  Workspace: my-workspace
  Service ID: my-conda-worker
  Visibility: public
  Cache Dir: /home/user/.hypha_conda_cache

✅ Conda Environment Worker (using mamba) registered successfully!
   Service ID: my-conda-worker
   Supported types: ['conda-jupyter-kernel']
   Visibility: public

Worker is ready to process conda environment requests...
Press Ctrl+C to stop the worker.
```

## Troubleshooting

### Common Issues

#### 1. Package Manager Not Found
```
Error: Neither mamba nor conda package manager found
```
**Solution**: Install mamba (recommended) or conda and ensure it's available in PATH
```bash
# Check what's available
which mamba && mamba --version
which conda && conda --version

# Install mamba (recommended)
conda install mamba -n base -c conda-forge

# Or install miniconda/miniforge if neither is available
```

#### 2. Code Execution Errors
```
result["status"] == "error"
```
**Solution**: Check the error details and outputs in the result:
```python
result = await conda_worker.execute(session_id, code)
if result["status"] == "error":
    error = result.get("error", {})
    print(f"Error: {error.get('ename', 'Unknown')}")
    print(f"Error message: {error.get('evalue', '')}")
    
    # Check stderr from outputs
    for output in result.get("outputs", []):
        if output["type"] == "stream" and output["name"] == "stderr":
            print(f"Stderr: {output['text']}")
```

#### 3. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Check write permissions for cache directory
```bash
mkdir -p ~/.hypha_conda_cache
chmod 755 ~/.hypha_conda_cache
```

#### 4. Import Errors in Code
```
ModuleNotFoundError: No module named 'package_name'
```
**Solution**: Add the missing package to your manifest dependencies:
```yaml
dependencies:
  - python=3.11
  - numpy
  - package_name  # Add missing package
```

#### 5. Network Connection Issues
```
Failed to connect to server
```
**Solution**: 
- Verify server URL is correct
- Check network connectivity
- Ensure token is valid and not expired

#### 6. Environment Creation Failures
```
Failed to create environment using mamba/conda
```
**Solution**:
- Check if requested packages exist in specified channels
- Verify conda/mamba channels are accessible
- Try with different package versions
- If using mamba fails, the worker will automatically fall back to conda
- Check mamba/conda configuration: `mamba config --show` or `conda config --show`

### Performance Tips

1. **Use environment caching**: Don't disable caching in production
2. **Optimize dependencies**: Use specific versions to improve cache hits
3. **Resource allocation**: Ensure sufficient disk space for environments
4. **Network optimization**: Use local conda channels when possible

### Security Considerations

1. **Token security**: Store tokens securely, use environment variables
2. **Network security**: Use HTTPS for server connections
3. **Workspace isolation**: Use appropriate visibility settings
4. **Resource limits**: Monitor resource usage in production environments

## API Reference

### Worker Service Methods

#### `start(config: WorkerConfig) -> str`  
Start a new conda environment session with the specified dependencies. The initialization script will run during startup.

**Parameters:**
- `config`: Worker configuration including manifest with dependencies, channels, and entry point

**Returns:** Session ID string

#### `execute(session_id: str, script: str, config: Optional[Dict[str, Any]] = None, progress_callback: Optional[Callable] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
Execute Python code in the conda environment session, similar to a Jupyter notebook cell.

**Parameters:**
- `session_id`: The session ID returned from `start()`
- `script`: Python code string to execute
- `config`: Optional execution configuration (e.g., `{"timeout": 30.0}`)
- `progress_callback`: Optional callback for execution progress updates
- `context`: Optional context information

**Returns:** Dictionary with Jupyter-compatible output format:
```python
{
    "status": "ok",  # or "error"
    "outputs": [     # List of output objects
        {
            "type": "stream",
            "name": "stdout",
            "text": "output text"
        }
    ],
    "error": None    # Error details if status == "error"
}
```

#### `stop(session_id: str) -> None`
Stop a conda environment session and clean up resources.

#### `get_logs(session_id: str, type: str = None, offset: int = 0, limit: int = None) -> Dict[str, Any]`
Get logs for a conda environment session.

**Parameters:**
- `session_id`: Session ID
- `type`: Optional log type filter ("stdout", "stderr", "info", "error", "progress")
- `offset`: Starting position for log entries  
- `limit`: Maximum number of log entries to return

## Working Directory Configuration

The Conda Worker uses a working directory to store session-specific files and data. Each session gets its own isolated subdirectory within the base working directory.

### Working Directory Features

- **Session isolation**: Each session gets its own subdirectory (e.g., `/tmp/conda-sessions/session-123/`)
- **Automatic cleanup**: Session directories are removed when sessions are stopped
- **File staging**: Files downloaded via `files_to_stage` are placed in the session's working directory
- **Temporary storage**: Working directories use temporary storage since sessions are ephemeral

### Configuration Options

#### Command Line
```bash
python -m hypha.workers.conda \
  --working-dir /opt/conda-sessions \
  --cache-dir /opt/conda-cache
```

#### Environment Variables
```bash
export CONDA_WORKING_DIR=/opt/conda-sessions
export CONDA_CACHE_DIR=/opt/conda-cache
export CONDA_AUTHORIZED_WORKSPACES=public,root
```

#### Built-in Worker (Startup Function)
When using the conda worker as a built-in startup function:
```bash
export CONDA_WORKING_DIR=/data/conda-sessions
export CONDA_CACHE_DIR=/data/conda-cache
export CONDA_AUTHORIZED_WORKSPACES=workspace1,workspace2
python -m hypha.server --startup-functions hypha.workers.conda:hypha_startup
```

### Deployment Configurations

#### Helm Chart Configuration

When deploying with the Hypha Helm chart, configure the conda worker in `values.yaml`:

```yaml
# Configure built-in conda worker
condaWorker:
  enabled: true
  workingDir: "/tmp/conda-sessions"  # Temporary storage for session data
  cacheDir: "/data/conda-cache"       # Persistent storage for environment cache (optional)
  authorizedWorkspaces: ""             # Empty means all workspaces

# Add conda worker to startup functions
env:
  - name: HYPHA_STARTUP_FUNCTIONS
    value: "hypha.workers.conda:hypha_startup"

# Optional: Enable persistent storage only if you want to cache conda environments
# This can significantly speed up environment creation
persistence:
  enabled: true
  size: 10Gi
  accessModes:
    - ReadWriteMany
```

The Helm chart automatically:
- Creates the necessary directories with proper permissions
- Uses temporary storage for session working directories (no persistence needed)
- Optionally mounts persistent storage for conda environment cache
- Sets environment variables for the conda worker

#### Docker Compose Configuration

```yaml
version: '3.8'

services:
  hypha-server:
    image: ghcr.io/amun-ai/hypha:latest
    environment:
      - HYPHA_STARTUP_FUNCTIONS=hypha.workers.conda:hypha_startup
      - CONDA_WORKING_DIR=/tmp/conda-sessions    # Temporary storage
      - CONDA_CACHE_DIR=/data/conda-cache         # Optional: persistent cache
    volumes:
      - hypha-cache:/data    # Optional: for caching environments
    command: python -m hypha.server

volumes:
  hypha-cache:    # Optional: only needed if you want to cache environments
    driver: local
```

#### Kubernetes Deployment

For production Kubernetes deployments without Helm:

```yaml
# Optional: PVC for conda environment cache
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: conda-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypha-server
spec:
  replicas: 1
  template:
    spec:
      initContainers:
        - name: init-cache
          image: busybox
          command: ['sh', '-c']
          args:
            - |
              # Only create cache directory in persistent storage
              mkdir -p /data/conda-cache
              chmod 755 /data/conda-cache
              chown -R 1000:1000 /data/conda-cache
          volumeMounts:
            - name: cache
              mountPath: /data
          securityContext:
            runAsUser: 0
      containers:
        - name: hypha
          image: ghcr.io/amun-ai/hypha:latest
          env:
            - name: HYPHA_STARTUP_FUNCTIONS
              value: "hypha.workers.conda:hypha_startup"
            - name: CONDA_WORKING_DIR
              value: "/tmp/conda-sessions"    # Temporary storage
            - name: CONDA_CACHE_DIR
              value: "/tmp/conda-cache"       # Persistent cache
          volumeMounts:
            - name: cache
              mountPath: /data
      volumes:
        - name: cache
          persistentVolumeClaim:
            claimName: conda-cache
```

### Storage Recommendations

#### Working Directory (`CONDA_WORKING_DIR`)
- **Storage Type**: Temporary (e.g., `/tmp`)
- **Persistence**: Not required
- **Purpose**: Stores session-specific files that are automatically cleaned up
- **Size**: Depends on your workload, typically a few GB is sufficient

#### Cache Directory (`CONDA_CACHE_DIR`)
- **Storage Type**: Persistent (optional but recommended)
- **Persistence**: Recommended for production to speed up environment creation
- **Purpose**: Caches conda environments to avoid re-downloading packages
- **Size**: 10-20 GB recommended, depending on the variety of environments

### Best Practices

1. **Use temporary storage for working directories**: Session data is ephemeral and doesn't need persistence
2. **Consider persistent cache**: While optional, caching environments significantly improves performance
3. **Set appropriate permissions**: Ensure directories are writable by the application user (typically UID 1000)
4. **Monitor disk usage**: Conda environments can consume significant disk space in the cache
5. **Automatic cleanup**: Working directories are cleaned automatically when sessions end
6. **Network storage for cache**: When using network storage (NFS, EFS) for cache, ensure sufficient IOPS

## Advanced Configuration

### Custom Docker Setup

For advanced Docker deployments, you can customize the container:

```dockerfile
FROM ghcr.io/amun-ai/hypha:0.20.71

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install additional conda packages
RUN conda install -c conda-forge \
    scikit-learn \
    matplotlib \
    jupyter

# Set custom environment variables
ENV HYPHA_CACHE_DIR=/opt/conda_cache
ENV HYPHA_VISIBILITY=protected

# Create cache directory
RUN mkdir -p /opt/conda_cache

# Set the default command
CMD ["python", "-m", "hypha.workers.conda", "--verbose"]
```

### Kubernetes Deployment

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypha-conda-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypha-conda-worker
  template:
    metadata:
      labels:
        app: hypha-conda-worker
    spec:
      containers:
      - name: conda-worker
        image: ghcr.io/amun-ai/hypha:0.20.71
        command: ["python", "-m", "hypha.workers.conda", "--verbose"]
        env:
        - name: HYPHA_SERVER_URL
          value: "https://hypha.aicell.io"
        - name: HYPHA_WORKSPACE
          value: "production"
        - name: HYPHA_TOKEN
          valueFrom:
            secretKeyRef:
              name: hypha-secrets
              key: token
        - name: HYPHA_VISIBILITY
          value: "protected"
        - name: HYPHA_CACHE_DIR
          value: "/tmp/conda_cache"
        volumeMounts:
        - name: conda-cache
          mountPath: /tmp/conda_cache
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: conda-cache
        emptyDir:
          sizeLimit: 10Gi
```

## Support

For issues and questions:
- GitHub Issues: [https://github.com/amun-ai/hypha/issues](https://github.com/amun-ai/hypha/issues)
- Documentation: [https://amun-ai.github.io/hypha/](https://amun-ai.github.io/hypha/)
- Community: Join our Discord or forum discussions