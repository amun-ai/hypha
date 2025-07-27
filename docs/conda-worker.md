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
type: python-conda
version: 1.0.0
description: A Python app with conda dependencies
entry_point: main.py
dependencies:
  - python=3.11
  - numpy
  - pandas
channels:
  - conda-forge
```

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
            "type": "python-conda",
            "dependencies": ["python=3.11", "numpy", "pandas"],
            "channels": ["conda-forge"],
            "entry_point": "main.py"
        },
        # ... other config fields
    })
    
    # Execute code like in a Jupyter notebook
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
    
    result = await conda_worker.execute_code(session_id, code)
    
    if result.success:
        print("Output:", result.stdout)
    else:
        print("Error:", result.error)
        print("Stderr:", result.stderr)
    
    # Stop the session
    await conda_worker.stop(session_id)

asyncio.run(main())
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

result = await conda_worker.execute_code(session_id, code)
print(result.stdout)
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

result = await conda_worker.execute_code(session_id, code)
print(result.stdout)
```

### Package Testing

```python
# Test if packages are available and working
code = """
try:
    import requests
    response = requests.get('https://httpbin.org/json')
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

result = await conda_worker.execute_code(session_id, code)
print(result.stdout)
```

### Handling Execution Results

The `execute_code()` method returns an `ExecutionResult` object with the following fields:

```python
result = await conda_worker.execute_code(session_id, code)

if result.success:
    print("Code executed successfully!")
    print("Output:", result.stdout)
else:
    print("Execution failed!")
    print("Error:", result.error)
    print("Stderr:", result.stderr)

# Timing information (if available)
if result.timing:
    print(f"Execution time: {result.timing.execution_time:.2f}s")
```

## Environment Caching

The worker automatically caches conda environments to improve performance. When you start a session, the initialization script runs once, but the environment stays available for multiple `execute_code()` calls:

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
   Supported types: ['python-conda']
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
ExecutionResult.success = False
```
**Solution**: Check the `stderr` and `error` fields in the result:
```python
result = await conda_worker.execute_code(session_id, code)
if not result.success:
    print(f"Error: {result.error}")
    print(f"Stderr: {result.stderr}")
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

#### `execute_code(session_id: str, code: str) -> ExecutionResult`
Execute Python code in the conda environment session, similar to a Jupyter notebook cell.

**Parameters:**
- `session_id`: The session ID returned from `start()`
- `code`: Python code string to execute

**Returns:** ExecutionResult with `success`, `stdout`, `stderr`, and `error` fields

#### `stop(session_id: str) -> None`
Stop a conda environment session and clean up resources.

#### `get_logs(session_id: str, type: str = None, offset: int = 0, limit: int = None) -> Union[Dict[str, List[str]], List[str]]`
Get logs for a conda environment session.

**Parameters:**
- `session_id`: Session ID
- `type`: Optional log type filter ("stdout", "stderr", "info", "error", "progress")
- `offset`: Starting position for log entries  
- `limit`: Maximum number of log entries to return

#### `get_session_info(session_id: str) -> SessionInfo`
Get information about a conda environment session including status and metadata.

#### `list_sessions(workspace: str) -> List[SessionInfo]`
List all conda environment sessions for a workspace.

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