# MinIO Helm Chart

This Helm chart deploys MinIO using the Bitnami MinIO chart as a dependency.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- Bitnami Helm repository

## Security Setup (Recommended)

For production deployments, create a Kubernetes secret for MinIO credentials:

```bash
kubectl create secret generic minio-credentials \
  --from-literal=ACCESS_KEY_ID=minioadmin \
  --from-literal=SECRET_ACCESS_KEY=minioadmin \
  --namespace=hypha
```

The chart is configured to use this secret by default. If you don't create the secret, you can uncomment the hardcoded credentials in `values.yaml`.

## Installation

### Add the Bitnami repository

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### Install the chart

```bash
# Install with default values
helm install my-minio ./helm-charts/minio

# Install with custom values
helm install my-minio ./helm-charts/minio -f values.yaml
```

### Update dependencies

If you need to update the chart dependencies:

```bash
cd helm-charts/minio
helm dependency update
```

## Configuration

The following table lists the configurable parameters of the MinIO chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `minio.enabled` | Enable MinIO deployment | `true` |
| `minio.architecture` | MinIO architecture (standalone/distributed) | `standalone` |
| `minio.auth.existingSecret` | Secret name for credentials | `minio-credentials` |
| `minio.auth.existingSecretKeys.rootUser` | Secret key for username | `ACCESS_KEY_ID` |
| `minio.auth.existingSecretKeys.rootPassword` | Secret key for password | `SECRET_ACCESS_KEY` |
| `minio.service.type` | Kubernetes service type | `ClusterIP` |
| `minio.persistence.enabled` | Enable persistence | `true` |
| `minio.persistence.size` | Persistent volume size | `8Gi` |
| `minio.resources.requests.memory` | Memory requests | `256Mi` |
| `minio.resources.requests.cpu` | CPU requests | `250m` |
| `minio.resources.limits.memory` | Memory limits | `512Mi` |
| `minio.resources.limits.cpu` | CPU limits | `500m` |

## Usage Examples

### Basic installation

```bash
helm install my-minio ./helm-charts/minio
```

### Custom values

Create a custom values file (`my-values.yaml`):

```yaml
minio:
  auth:
    # Use custom secret
    existingSecret: "my-minio-secret"
    existingSecretKeys:
      rootUser: "USERNAME"
      rootPassword: "PASSWORD"
  persistence:
    size: 20Gi
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
```

Or use hardcoded credentials (not recommended for production):

```yaml
minio:
  auth:
    # Comment out existingSecret to use hardcoded values
    # existingSecret: "minio-credentials"
    rootUser: "myuser"
    rootPassword: "mypassword"
```

Install with custom values:

```bash
helm install my-minio ./helm-charts/minio -f my-values.yaml
```

### Enable ingress

```yaml
minio:
  ingress:
    enabled: true
    ingressClassName: nginx
    hosts:
      - host: minio.example.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: minio-tls
        hosts:
          - minio.example.com
```

### Enable metrics

```yaml
minio:
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true
```

## Accessing MinIO

### Port forwarding

```bash
kubectl port-forward svc/my-minio-minio 9000:9000
kubectl port-forward svc/my-minio-minio 9001:9001
```

### Access URLs

- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

### Default credentials

The default credentials depend on your configuration:

- **With secret (recommended)**: Use the credentials from the `minio-credentials` secret
- **Without secret**: Username: `minioadmin`, Password: `minioadmin`

To check your current credentials:
```bash
kubectl get secret minio-credentials -o jsonpath='{.data.ACCESS_KEY_ID}' | base64 -d
kubectl get secret minio-credentials -o jsonpath='{.data.SECRET_ACCESS_KEY}' | base64 -d
```

## Uninstalling

```bash
helm uninstall my-minio
```

## Troubleshooting

### Check pod status

```bash
kubectl get pods -l app.kubernetes.io/name=minio
```

### View logs

```bash
kubectl logs -l app.kubernetes.io/name=minio
```

### Check persistent volumes

```bash
kubectl get pvc -l app.kubernetes.io/name=minio
```

## More Information

For more detailed configuration options, refer to the [Bitnami MinIO chart documentation](https://github.com/bitnami/charts/tree/main/bitnami/minio). 