# Hypha Server Helm Chart

This Helm chart deploys a Hypha server instance on Kubernetes.

## Installation

```bash
# Add the Hypha repository
helm repo add amun-ai https://docs.amun.ai
helm repo update

# Install the chart
helm install hypha-server amun-ai/hypha-server
```

## Configuration

The following table lists the main configurable parameters of the Hypha Server chart and their default values.

### Basic Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `ghcr.io/amun-ai/hypha` |
| `image.tag` | Image tag | `0.21.52` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Kubernetes service type | `ClusterIP` |
| `service.port` | Service port | `9520` |

### Ingress Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class | `nginx` |
| `ingress.hosts[0].host` | Hostname | `hypha.amun.ai` |
| `ingress.tls` | TLS configuration | Configured for `hypha.amun.ai` |

### Persistence Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `persistence.accessModes` | Access modes | `["ReadWriteMany"]` |
| `persistence.mountPath` | Mount path | `/data` |

**Note**: Persistence is optional and mainly benefits:
- SQLite database storage (if not using external database)
- Conda environment caching (if conda worker is enabled)
- Artifact storage (if S3 is not configured)

### Built-in Conda Worker

The chart can deploy a built-in conda worker for executing Python code in isolated environments. Configure it using environment variables:

#### Enabling the Conda Worker

```yaml
# values.yaml
env:
  - name: HYPHA_STARTUP_FUNCTIONS
    value: "hypha.workers.conda:hypha_startup"
  # Optional: Configure directories (these are the defaults)
  - name: CONDA_WORKING_DIR
    value: "/tmp/conda-sessions"
  - name: CONDA_CACHE_DIR
    value: "/tmp/conda-cache"
  # Optional: Restrict to specific workspaces
  # - name: CONDA_AUTHORIZED_WORKSPACES
  #   value: "workspace1,workspace2"
```

#### Conda Worker with Persistent Cache

For better performance, use persistent storage for the conda cache:

```yaml
persistence:
  enabled: true
  size: 20Gi

env:
  - name: HYPHA_STARTUP_FUNCTIONS
    value: "hypha.workers.conda:hypha_startup"
  - name: CONDA_WORKING_DIR
    value: "/tmp/conda-sessions"  # Temporary (no persistence needed)
  - name: CONDA_CACHE_DIR
    value: "/data/conda-cache"    # Persistent cache for better performance
```

### Kubernetes Worker

The chart can enable a Kubernetes worker for launching pods dynamically.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `k8sWorker.enabled` | Enable Kubernetes worker | `false` |
| `k8sWorker.namespace` | Target namespace for pods | Same as release |
| `k8sWorker.defaultTimeout` | Default pod timeout (seconds) | `3600` |
| `k8sWorker.imagePullPolicy` | Pull policy for worker pods | `IfNotPresent` |

### Environment Variables

Common environment variables can be configured in `env`:

```yaml
env:
  - name: HYPHA_JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: HYPHA_JWT_SECRET
  - name: HYPHA_DATABASE_URI
    value: "sqlite+aiosqlite:////data/hypha-app-database.db"
  - name: HYPHA_PUBLIC_BASE_URL
    value: "https://hypha.amun.ai"
```

See the [values.yaml](values.yaml) file for all available environment variables and their descriptions.

## Examples

### Minimal Installation

```bash
helm install hypha-server ./hypha-server
```

### Production with PostgreSQL and Redis

```yaml
# custom-values.yaml
env:
  - name: HYPHA_DATABASE_URI
    value: "postgresql+asyncpg://user:pass@postgres:5432/hypha"
  - name: HYPHA_REDIS_URI
    value: "redis://redis:6379/0"

replicaCount: 3

persistence:
  enabled: false  # Not needed with external database
```

```bash
helm install hypha-server ./hypha-server -f custom-values.yaml
```

### With Conda Worker and S3 Storage

```yaml
# custom-values.yaml
env:
  - name: HYPHA_STARTUP_FUNCTIONS
    value: "hypha.workers.conda:hypha_startup"
  - name: CONDA_CACHE_DIR
    value: "/data/conda-cache"  # Use persistent cache
  - name: HYPHA_ENABLE_S3
    value: "true"
  - name: HYPHA_ENDPOINT_URL
    value: "https://s3.amazonaws.com"
  - name: HYPHA_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: s3-credentials
        key: access-key
  - name: HYPHA_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: s3-credentials
        key: secret-key

persistence:
  enabled: true
  size: 20Gi
```

## Upgrading

```bash
helm upgrade hypha-server ./hypha-server
```

## Uninstalling

```bash
helm uninstall hypha-server
```

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -l app.kubernetes.io/name=hypha-server
kubectl logs -l app.kubernetes.io/name=hypha-server
```

### Check Conda Worker
If conda worker is enabled:
```bash
kubectl exec -it <pod-name> -- ls -la /tmp/conda-sessions
kubectl exec -it <pod-name> -- ls -la /tmp/conda-cache
```

### Persistence Issues
If using persistence, check PVC status:
```bash
kubectl get pvc
kubectl describe pvc <pvc-name>
```

## License

This Helm chart is part of the Hypha project and is licensed under the MIT License.