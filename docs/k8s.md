# Kubernetes Worker

The Kubernetes Worker enables Hypha to launch and manage applications as Kubernetes pods within your cluster. This worker provides a bridge between Hypha's application management system and Kubernetes' container orchestration capabilities.

## Overview

The Kubernetes Worker (`k8s.py`) allows you to:

- **Launch containerized applications** as Kubernetes pods with custom configurations
- **Execute commands** inside running pods (equivalent to `kubectl exec`)
- **Inject Hypha environment variables** automatically for seamless integration
- **Run other Hypha workers** (like conda worker) inside Kubernetes pods
- **Scale applications** using Kubernetes' native capabilities
- **Leverage Kubernetes features** like resource limits, health checks, and persistent volumes

## Prerequisites

### Kubernetes Access

The Hypha server needs access to a Kubernetes cluster with appropriate permissions:

1. **In-cluster deployment**: When Hypha runs inside Kubernetes, it can use service account credentials
2. **External access**: When Hypha runs outside Kubernetes, it needs a valid kubeconfig file

### RBAC Permissions

The service account used by Hypha needs the following permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hypha-k8s-worker
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["create", "delete", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/log", "pods/status"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["get", "list"]
```

### Python Dependencies

```bash
pip install kubernetes>=27.2.0
```

## Installation

### Option 1: Using Helm Chart (Recommended)

The easiest way to enable the Kubernetes worker is through the Hypha Helm chart:

```yaml
# values.yaml
k8sWorker:
  enabled: true
  namespace: default  # Optional: defaults to release namespace
  defaultTimeout: 3600  # Optional: default pod timeout in seconds
  imagePullPolicy: IfNotPresent  # Optional: image pull policy
```

Deploy with:

```bash
helm install hypha ./helm-charts/hypha-server -f values.yaml
```

### Option 2: Manual Server Configuration

Enable the worker when starting the Hypha server:

```bash
python -m hypha.server --enable-k8s-worker
```

Or set environment variables:

```bash
export HYPHA_ENABLE_K8S_WORKER=true
export HYPHA_K8S_NAMESPACE=default
export HYPHA_K8S_DEFAULT_TIMEOUT=3600
export HYPHA_K8S_IMAGE_PULL_POLICY=IfNotPresent

python -m hypha.server
```

## Basic Usage

### 1. Connect to Hypha Server

```python
from hypha import connect_to_server

# Connect to server with k8s worker enabled
server = await connect_to_server({
    "server_url": "https://your-hypha-server.com",
    "token": "your-token"
})
```

### 2. Get the Kubernetes Worker

```python
# Get the worker service
k8s_worker = await server.get_service("k8s-worker")
```

### 3. Launch a Pod

```python
# Define pod configuration
config = {
    "id": "my-app-session",
    "type": "k8s-pod",
    "manifest": {
        "image": "ubuntu:20.04",
        "command": ["sleep", "3600"],
        "env": {
            "MY_VAR": "my_value",
            "DEBUG": "true"
        },
        "image_pull_policy": "IfNotPresent"
    }
}

# Start the pod
session_id = await k8s_worker.start(config)
print(f"Pod started with session ID: {session_id}")
```

### 4. Execute Commands in Pod

```python
# Execute simple command
result = await k8s_worker.execute(session_id, "whoami")
print(f"Output: {result['outputs'][0]['text']}")

# Execute complex command with pipes
result = await k8s_worker.execute(
    session_id, 
    "ps aux | grep python | wc -l"
)

# Execute with custom configuration
result = await k8s_worker.execute(
    session_id,
    "long-running-command",
    config={
        "timeout": 300,
        "shell": "/bin/bash"
    }
)
```

### 5. Get Logs

```python
# Get all logs
logs = await k8s_worker.get_logs(session_id)

# Get specific log type with limit
stdout_logs = await k8s_worker.get_logs(
    session_id, 
    type="stdout", 
    limit=100
)
```

### 6. Stop the Pod

```python
await k8s_worker.stop(session_id)
```

## Advanced Configuration

### Pod Manifest Options

The pod manifest supports various Kubernetes pod specifications:

```python
manifest = {
    # Required
    "image": "python:3.9",
    
    # Optional - Container configuration
    "command": ["python", "-m", "http.server"],
    "args": ["8000"],
    "working_dir": "/app",
    
    # Optional - Environment variables
    "env": {
        "PYTHONPATH": "/app",
        "DEBUG": "true"
    },
    
    # Optional - Kubernetes options
    "image_pull_policy": "Always",  # Always, IfNotPresent, Never
    "restart_policy": "Never",      # Always, OnFailure, Never
    "timeout": 7200,               # Pod timeout in seconds
    
    # Optional - Resource limits (if supported)
    "resources": {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "500m", "memory": "512Mi"}
    },
    
    # Optional - Volume mounts (if configured)
    "volumes": [
        {
            "name": "data-volume",
            "mount_path": "/data",
            "pvc_name": "my-pvc"
        }
    ]
}
```

### Environment Variable Injection

The Kubernetes worker automatically injects Hypha-specific environment variables into every pod:

- `HYPHA_SERVER_URL`: URL of the Hypha server
- `HYPHA_WORKSPACE`: Current workspace name
- `HYPHA_TOKEN`: Authentication token
- `HYPHA_CLIENT_ID`: Client identifier
- `HYPHA_APP_ID`: Application identifier

These variables enable other Hypha workers to connect back to the server when running inside pods.

## Integration Examples

### Running Conda Worker in Kubernetes

Launch a conda worker inside a Kubernetes pod:

```python
config = {
    "id": "conda-k8s-session",
    "type": "k8s-pod", 
    "manifest": {
        "image": "continuumio/miniconda3:latest",
        "command": ["python", "-m", "hypha.workers.conda"],
        "env": {
            "HYPHA_RUN_FROM_ENV": "true",
            "HYPHA_VERBOSE": "true"
        }
    }
}

session_id = await k8s_worker.start(config)

# The conda worker will automatically connect to Hypha
# and register itself using the injected environment variables
```

### Custom Python Application

```python
config = {
    "id": "python-app-session",
    "type": "k8s-pod",
    "manifest": {
        "image": "python:3.9",
        "command": ["python", "-c"],
        "args": ["""
import os
from hypha import connect_to_server

async def main():
    # Connect using injected environment variables
    server = await connect_to_server({
        'server_url': os.environ['HYPHA_SERVER_URL'],
        'token': os.environ['HYPHA_TOKEN'],
        'workspace': os.environ['HYPHA_WORKSPACE']
    })
    
    # Register your service
    await server.register_service({
        'name': 'my-k8s-service',
        'type': 'my-service',
        'config': {'version': '1.0'}
    })

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
        """],
        "env": {
            "PYTHONUNBUFFERED": "1"
        }
    }
}
```

### Jupyter Notebook Server

```python
config = {
    "id": "jupyter-session",
    "type": "k8s-pod",
    "manifest": {
        "image": "jupyter/scipy-notebook:latest",
        "command": ["start-notebook.sh"],
        "args": [
            "--NotebookApp.token=''",
            "--NotebookApp.password=''",
            "--NotebookApp.allow_origin='*'",
            "--NotebookApp.port=8888"
        ],
        "env": {
            "JUPYTER_ENABLE_LAB": "yes"
        }
    }
}
```

## Execute Method Reference

The `execute` method provides `kubectl exec` functionality:

### Basic Execution

```python
# Simple command
result = await k8s_worker.execute(session_id, "echo hello")

# Command with arguments  
result = await k8s_worker.execute(session_id, "ls -la /tmp")
```

### Complex Scripts

```python
# Multi-line script
script = """
echo "Starting backup..."
tar -czf /tmp/backup.tar.gz /app/data
echo "Backup completed: $(ls -lh /tmp/backup.tar.gz)"
"""

result = await k8s_worker.execute(session_id, script)
```

### Configuration Options

```python
result = await k8s_worker.execute(
    session_id,
    "long-command",
    config={
        "timeout": 600,           # Execution timeout in seconds
        "container": "main",      # Container name (for multi-container pods)
        "shell": "/bin/bash"      # Shell to use for complex commands
    }
)
```

### Result Format

The execute method returns results in Jupyter-like format:

```python
{
    "status": "ok",  # or "error"
    "outputs": [
        {
            "type": "stream",
            "name": "stdout",
            "text": "command output"
        },
        {
            "type": "stream", 
            "name": "stderr",
            "text": "error output"
        }
    ],
    # Only present on error
    "error": {
        "ename": "CommandError",
        "evalue": "Command failed with return code 1",
        "traceback": ["error details"]
    }
}
```

## Helm Chart Integration

### Enabling in Values

```yaml
# values.yaml
k8sWorker:
  enabled: true
  namespace: hypha-workers  # Namespace for launched pods
  defaultTimeout: 7200      # Default pod timeout
  imagePullPolicy: Always   # Image pull policy

# Optional: Service account configuration
serviceAccount:
  create: true
  annotations: {}
  name: ""

# Optional: RBAC configuration  
rbac:
  create: true
```

### Custom RBAC

If you need custom RBAC permissions:

```yaml
# custom-rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hypha-k8s-custom
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log", "pods/status", "events"]
  verbs: ["*"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims"]
  verbs: ["get", "list"]
```

### Environment Variables

The Helm chart automatically sets these environment variables:

```yaml
env:
  - name: HYPHA_ENABLE_K8S_WORKER
    value: "true"
  - name: HYPHA_K8S_NAMESPACE
    value: {{ .Values.k8sWorker.namespace | default .Release.Namespace }}
  - name: HYPHA_K8S_DEFAULT_TIMEOUT
    value: {{ .Values.k8sWorker.defaultTimeout | default 3600 | quote }}
  - name: HYPHA_K8S_IMAGE_PULL_POLICY
    value: {{ .Values.k8sWorker.imagePullPolicy | default "IfNotPresent" }}
```

## Security Considerations

### Service Account Permissions

- Use **least privilege principle** - only grant necessary permissions
- **Scope RBAC to namespace** where pods will be created
- **Avoid cluster-wide permissions** unless absolutely necessary

### Pod Security

The worker automatically applies security contexts:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  capabilities:
    drop: ["ALL"]
  seccompProfile:
    type: RuntimeDefault
```

### Image Security

- Use **specific image tags** instead of `latest`
- **Scan images** for vulnerabilities before deployment
- Use **private registries** for sensitive applications
- Configure **image pull secrets** if needed

### Network Security

- Use **NetworkPolicies** to restrict pod communications
- **Avoid exposing services** unnecessarily
- Use **encrypted communication** (TLS) where possible

## Monitoring and Observability

### Pod Status Monitoring

```python
# Get session information
info = await k8s_worker.get_session_info(session_id)
print(f"Status: {info.status}")
print(f"Created: {info.created_at}")

# List all sessions
sessions = await k8s_worker.list_sessions("my-workspace")
for session in sessions:
    print(f"Session {session.session_id}: {session.status}")
```

### Log Aggregation

```python
# Get recent logs
logs = await k8s_worker.get_logs(session_id, limit=1000)

# Stream logs (if supported)
async for log_entry in k8s_worker.stream_logs(session_id):
    print(f"[{log_entry.timestamp}] {log_entry.message}")
```

### Resource Usage

Monitor resource usage through Kubernetes:

```bash
# Check pod resource usage
kubectl top pods -n hypha-workers

# Get pod details
kubectl describe pod <pod-name> -n hypha-workers

# Check events
kubectl get events -n hypha-workers --sort-by='.lastTimestamp'
```

## Troubleshooting

### Common Issues

#### 1. Worker Not Available

**Error**: `WorkerNotAvailableError: Kubernetes configuration not found`

**Solutions**:

- Ensure kubeconfig is available and valid
- Check in-cluster service account permissions
- Verify Kubernetes cluster connectivity

```bash
# Test kubectl access
kubectl auth can-i create pods --namespace=hypha-workers

# Check service account permissions
kubectl auth can-i create pods --as=system:serviceaccount:hypha:hypha-server
```

#### 2. Pod Creation Fails

**Error**: `WorkerError: Failed to create pod`

**Solutions**:

- Check RBAC permissions
- Verify namespace exists
- Check image availability and pull secrets
- Review resource quotas

```bash
# Check namespace
kubectl get namespace hypha-workers

# Check resource quotas
kubectl describe resourcequota -n hypha-workers

# Check image pull secrets  
kubectl get secrets -n hypha-workers
```

#### 3. Pod Fails to Start

**Error**: `WorkerError: Pod failed to start`

**Solutions**:

- Check pod logs: `kubectl logs <pod-name> -n hypha-workers`
- Verify image exists and is pullable
- Check resource requests vs. cluster capacity
- Review pod security contexts

#### 4. Execute Commands Fail

**Error**: `KubernetesApiError during command execution`

**Solutions**:

- Ensure pod is in "Running" state
- Check container name (for multi-container pods)
- Verify shell exists in container (`/bin/sh`, `/bin/bash`)
- Check command syntax and permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export HYPHA_VERBOSE=true
```

### Health Checks

```python
# Check worker health
async def check_k8s_worker_health():
    try:
        worker = await server.get_service("k8s-worker")
        # Try a simple operation
        sessions = await worker.list_sessions("test")
        print("✓ K8s worker is healthy")
        return True
    except Exception as e:
        print(f"✗ K8s worker health check failed: {e}")
        return False
```

## Performance Optimization

### Resource Management

```python
# Set appropriate resource limits
manifest = {
    "image": "python:3.9",
    "resources": {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1000m", "memory": "1Gi"}
    }
}
```

### Pod Startup Optimization

- Use **smaller base images** (alpine, distroless)
- **Pre-pull frequently used images** on nodes
- Use **init containers** for setup tasks
- Configure **readiness probes** appropriately

### Execution Optimization

- **Batch commands** when possible to reduce exec calls
- Use **appropriate timeouts** to avoid hanging
- **Stream large outputs** instead of buffering
- **Clean up pods** promptly after use

## Migration Guide

### From Docker to Kubernetes

If migrating from Docker-based workers:

1. **Convert Docker commands** to Kubernetes manifests
2. **Update environment variables** to use Kubernetes secrets/configmaps
3. **Adapt volume mounts** to use PVCs or other Kubernetes volumes
4. **Update networking** to use Kubernetes services

### Version Compatibility

- **Kubernetes 1.20+**: Full feature support
- **Kubernetes 1.19**: Basic functionality (limited exec features)
- **Python 3.8+**: Required for async/await support
- **kubernetes-client 27.2.0+**: Required for stream support

## Best Practices

### Configuration Management

- Use **Kubernetes ConfigMaps** for configuration
- Store **secrets** in Kubernetes Secrets, not environment variables
- Use **Helm values** for environment-specific configuration
- **Version your manifests** with your application code

### Pod Resource Management

- Set **appropriate resource requests and limits**
- Use **horizontal pod autoscaling** for scalable workloads
- **Monitor resource usage** and adjust as needed
- Use **pod disruption budgets** for critical applications

### Development Workflow

- **Test locally** with minikube or kind before deploying
- Use **staging environments** for validation
- **Automate deployment** through CI/CD pipelines
- **Monitor applications** in production

### Maintenance

- **Regularly update** Kubernetes cluster and worker images
- **Review and rotate** service account tokens
- **Monitor for deprecated** API versions
- **Backup critical data** before major changes

## API Reference

### KubernetesWorker Methods

#### `start(config: WorkerConfig) -> str`

Start a new pod session with the given configuration.

#### `stop(session_id: str) -> None`

Stop and delete the pod for the given session.

#### `execute(session_id: str, script: str, config: dict = None) -> dict`

Execute a command inside the running pod.

#### `get_logs(session_id: str, type: str = None, offset: int = 0, limit: int = None) -> dict`

Retrieve logs from the pod.

#### `list_sessions(workspace: str = None) -> List[SessionInfo]`

List all active sessions, optionally filtered by workspace.

#### `get_session_info(session_id: str) -> SessionInfo`

Get detailed information about a specific session.

### Configuration Schema

```python
WorkerConfig = {
    "id": str,                    # Session identifier
    "type": "k8s-pod",           # Worker type
    "app_id": str,               # Application identifier
    "workspace": str,            # Workspace name
    "client_id": str,            # Client identifier
    "server_url": str,           # Hypha server URL
    "token": str,                # Authentication token
    "manifest": {                # Pod configuration
        "image": str,            # Container image (required)
        "command": List[str],    # Container command (optional)
        "args": List[str],       # Container arguments (optional)
        "env": Dict[str, str],   # Environment variables (optional)
        "working_dir": str,      # Working directory (optional)
        "image_pull_policy": str, # Image pull policy (optional)
        "restart_policy": str,   # Restart policy (optional)
        "timeout": int,          # Pod timeout in seconds (optional)
        "resources": dict,       # Resource requests/limits (optional)
        "volumes": List[dict]    # Volume mounts (optional)
    }
}
```

This documentation provides comprehensive guidance for using the Kubernetes worker in production environments. For additional examples and updates, see the `examples/` directory and the project repository.
