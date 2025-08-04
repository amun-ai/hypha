# Hypha Helm Charts

## Quick Start - Minimal Hypha Server

Get started with Hypha in one command:

```bash
kubectl create namespace hypha
export HYPHA_JWT_SECRET=$(openssl rand -base64 32)
kubectl create secret generic hypha-secrets --from-literal=HYPHA_JWT_SECRET=$HYPHA_JWT_SECRET --namespace=hypha
helm repo add amun-ai https://docs.amun.ai
helm repo update
helm install hypha-server amun-ai/hypha-server --namespace=hypha
```

This deploys a single-instance Hypha server with SQLite storage - perfect for getting started!

## For Production: Hypha Server Kit (Full Stack)

For production use, you'll likely need:
- **PostgreSQL** for robust data storage and concurrent access
- **Redis** for horizontal scaling across multiple server instances  
- **MinIO (S3)** for storing artifacts and files

Use the complete server kit:

```bash
helm repo add amun-ai https://docs.amun.ai
helm repo update
helm install hypha-server-kit amun-ai/hypha-server-kit --namespace=hypha
```

Or build from source: see the [Hypha Server Kit README](./hypha-server-kit/README.md) for detailed configuration.

---

## Advanced Configuration

### Manual Installation from Source

If you prefer to build from source or need custom configuration:

```bash
git clone https://github.com/amun-ai/hypha.git
cd hypha/helm-charts
kubectl create namespace hypha
export HYPHA_JWT_SECRET=$(openssl rand -base64 32)
kubectl create secret generic hypha-secrets --from-literal=HYPHA_JWT_SECRET=$HYPHA_JWT_SECRET --namespace=hypha
helm install hypha-server ./hypha-server --namespace=hypha
```

### Configurations and Ingress

Hypha uses Environment Variables to configure most of its behavior.

For comprehensive documentation on:
- **Environment Variables**: All available Hypha configuration options
- **Ingress Settings**: Detailed setup for various cloud providers (AWS, GCP, Azure) and bare metal

See the [Hypha Server Kit README](./hypha-server-kit/README.md).

### Custom Domain Configuration

To use your own domain, edit the `values.yaml` file and replace `hypha.amun.ai` with your domain name, then configure ingress appropriately.

### Database Configuration

The minimal version uses **SQLite** by default (single file database). For production with multiple users or horizontal scaling, configure **PostgreSQL**:

```yaml
env:
  - name: HYPHA_DATABASE_URI
    value: "postgresql+asyncpg://user:password@your-postgres-host:5432/hypha-db"
persistence:
  enabled: false  # No local storage needed with external DB
```

### Scaling Configuration

For horizontal scaling, configure **Redis** and set multiple replicas:

```yaml
env:
  - name: HYPHA_REDIS_URI
    value: "redis://your-redis-host:6379/0"
  - name: HYPHA_SERVER_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.uid
replicaCount: 3
```

### Storage Configuration

For artifact storage, configure **S3/MinIO**:

```yaml
env:
  - name: HYPHA_ENABLE_S3
    value: "true"
  - name: HYPHA_ACCESS_KEY_ID
    value: "your-access-key"
  - name: HYPHA_SECRET_ACCESS_KEY
    value: "your-secret-key"
  - name: HYPHA_ENDPOINT_URL
    value: "https://your-s3-endpoint"
```

For a complete production setup with all services included, use the **Hypha Server Kit** instead of configuring these manually.

## Upgrading and Management

```bash
# Upgrade
helm upgrade hypha-server amun-ai/hypha-server --namespace=hypha

# Uninstall  
helm uninstall hypha-server --namespace=hypha
```

## Setting up on Azure Kubernetes Service (AKS)

If you are deploying Hypha on Azure Kubernetes Service (AKS), you will need to configure the ingress to use the Azure Application Gateway. You can follow the instructions in the [aks-hypha.md](aks-hypha.md) file.