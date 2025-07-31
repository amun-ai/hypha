# Hypha Helm Charts

This repository contains two Helm charts for deploying Hypha on Kubernetes:

1. **[Hypha Server (Minimal)](#minimal-hypha-server)** - A lightweight deployment of Hypha Server only
2. **[Hypha Server Kit (Full)](#hypha-server-kit-full)** - Complete deployment with PostgreSQL, Redis, and MinIO included

## Comparison

| Feature | Minimal Version | Kit Version |
|---------|-----------------|-------------|
| **Hypha Server** | ✅ Included | ✅ Included |
| **PostgreSQL** | ❌ Not included | ✅ Included |
| **Redis** | ❌ Not included | ✅ Included |
| **MinIO (S3)** | ❌ Not included | ✅ Included |
| **Default Storage** | SQLite (local file) | PostgreSQL |
| **Production Ready** | Requires external services | ✅ All-in-one |
| **Configuration** | Manual setup | Automated via `.env` |
| **Best For** | Development, custom setups | Quick deployment, production |

## Minimal Hypha Server

The minimal chart (`./hypha-server`) deploys only the Hypha Server application. This is suitable when you:
- Already have PostgreSQL, Redis, or S3 storage available
- Want a lightweight deployment for development/testing
- Need fine-grained control over your infrastructure

**Note**: By default, the minimal version uses SQLite for storage. For production use, we recommend configuring an external PostgreSQL database.

## Hypha Server Kit (Full)

The kit chart (`./hypha-server-kit`) provides a complete, production-ready deployment including:
- **Hypha Server** - Core application
- **PostgreSQL** - Database backend
- **Redis** - Cache and message broker for scaling
- **MinIO** - S3-compatible object storage

This is ideal for:
- Quick production deployments
- Proof of concepts
- Development environments
- When you need all components configured and working together

### Quick Start - Kit Version

```bash
cd helm-charts/hypha-server-kit
cp .env.example .env
# Edit .env with your configuration
./apply-env.sh .env hypha
helm install hypha-server-kit . --namespace hypha
```

For detailed instructions, see the [Hypha Server Kit README](./hypha-server-kit/README.md).

---

## Minimal Version Deployment Guide

Below are the instructions for deploying the minimal Hypha Server chart.

### Prerequisites

Make sure you have the following installed:
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [helm](https://helm.sh/docs/intro/install/)

### Usage

First, clone this repository:
```bash
git clone https://github.com/amun-ai/hypha.git
cd hypha/helm-charts
```

Now you need to create a namespace for Hypha:
```bash
kubectl create namespace hypha
```

Then, run the bash command to create a `HYPHA_JWT_SECRET` apply these secrets to the cluster:
```bash
export HYPHA_JWT_SECRET=abcde123 # change this to your own secret
kubectl create secret generic hypha-secrets \
  --from-literal=HYPHA_JWT_SECRET=$HYPHA_JWT_SECRET \
  --dry-run=client -o yaml | kubectl apply --namespace=hypha -f -
```

Verify that the secret `hypha-secrets` has been created:
```bash
kubectl get secrets --namespace=hypha
```

Next, let's check the values in `values.yaml` and make sure they are correct. Importantly, search and replace all the `hypha.amun.ai` to your own public domain name and make sure the ingress is correctly configured.

**Note**: By default, Hypha uses persistent storage. If you want to disable persistence (e.g., for testing or development), set `persistence.enabled: false` in the values file.


Finally, we can install the helm chart:
```bash
helm install hypha-server ./hypha-server --namespace=hypha
```

If you make changes to the chart, you can upgrade the chart:
```bash
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

To uninstall the chart:
```bash
helm uninstall hypha-server --namespace=hypha
```

## Database Configuration

Hypha supports two database backends: **SQLite** (default) and **PostgreSQL**. The choice of database affects your persistence requirements:

### SQLite (Default) - Requires Persistent Storage

When using SQLite (the default), Hypha stores the database file on the local filesystem. This requires persistent storage to be enabled:

```yaml
persistence:
  enabled: true
  volumeName: hypha-app-storage
  mountPath: /data
  size: 20Gi
  accessModes:
    - ReadWriteOnce
  storageClass: "fast-ssd"
```

**Important**: If you disable persistence with SQLite, your data will be lost when the pod restarts.

### PostgreSQL - No Persistent Storage Required

When using PostgreSQL, Hypha connects to an external database, so you don't need persistent storage for the Hypha application itself. The PostgreSQL database handles its own persistence.

To use PostgreSQL, you need to:

1. **Install PostgreSQL** (see section below)
2. **Configure the DATABASE_URI** in your Hypha values:

```yaml
env:
  # ... existing environment variables ...
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: POSTGRES_PASSWORD
  - name: HYPHA_DATABASE_URI
    value: "postgresql+asyncpg://hypha-admin:$(POSTGRES_PASSWORD)@hypha-sql-database-postgresql.hypha.svc.cluster.local:5432/hypha-db"
```

**Note**: The `POSTGRES_PASSWORD` environment variable must be defined before `HYPHA_DATABASE_URI` for the password interpolation to work correctly.

3. **Disable persistence** for the Hypha application:

```yaml
persistence:
  enabled: false
```

### Persistence Configuration for SQLite

If you're using SQLite (default), you can configure persistence options in the `values.yaml` file:

#### Disable Persistence (Not Recommended for SQLite)

For testing or development environments, you can disable persistence entirely:

```yaml
persistence:
  enabled: false
```

**Warning**: This will cause data loss when pods restart if you are not using postgresql, nor the persistence setting.

#### Customize Persistence

You can customize the persistence configuration:

```yaml
persistence:
  enabled: true
  volumeName: hypha-app-storage
  mountPath: /data
  size: 20Gi  # Increase storage size
  accessModes:
    - ReadWriteMany  # For shared storage
  storageClass: "fast-ssd"  # Use specific storage class
```

## Install PostgreSQL for Production Use

For production deployments, it's recommended to use PostgreSQL instead of SQLite. PostgreSQL provides better performance, concurrent access, and data integrity.

### Prerequisites

Before installing PostgreSQL, you need to add the PostgreSQL password to your `hypha-secrets`:

```bash
# Add PostgreSQL password to existing secrets
kubectl patch secret hypha-secrets --namespace=hypha --type='json' -p='[{"op": "add", "path": "/data/POSTGRES_PASSWORD", "value": "'$(echo -n "your-secure-password" | base64)'"}]'
```

### Installing PostgreSQL

1. **Add the Bitnami Helm repository**:
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

2. **Navigate to the PostgreSQL chart directory**:
```bash
cd postgresql
```

3. **Deploy PostgreSQL using Helm**:
```bash
helm upgrade --install hypha-sql-database bitnami/postgresql \
  -f values.yaml \
  --namespace hypha \
  --version 16.1.0
```

### PostgreSQL Configuration

The provided `values.yaml` file configures PostgreSQL with:
- **Security**: Non-root user, proper security contexts
- **Storage**: 10Gi persistent volume using `retainer` storage class
- **Authentication**: Uses the existing `hypha-secrets` for passwords
- **Database**: Creates `hypha-db` database with `hypha-admin` user

### Connecting to PostgreSQL

#### From Inside the Cluster

For Hypha to connect to PostgreSQL, use this hostname:
```
hypha-sql-database-postgresql.hypha.svc.cluster.local:5432
```

#### From Your Local Machine

To connect directly to PostgreSQL for debugging:

```bash
# Forward the PostgreSQL port
kubectl port-forward --namespace hypha svc/hypha-sql-database-postgresql 5432:5432

# Get the password from secrets
export POSTGRES_PASSWORD=$(kubectl get secret --namespace hypha hypha-secrets -o jsonpath="{.data.POSTGRES_PASSWORD}" | base64 -d)

# Connect using psql
PGPASSWORD="$POSTGRES_PASSWORD" psql --host 127.0.0.1 -U hypha-admin -d hypha-db -p 5432
```

### Testing PostgreSQL Connection

You can test the PostgreSQL connection using the provided test script:

```bash
# Set the password environment variable
export POSTGRES_PASSWORD=$(kubectl get secret --namespace hypha hypha-secrets -o jsonpath="{.data.POSTGRES_PASSWORD}" | base64 -d)

# Run the test script
cd postgresql
python test_db.py
```

### Upgrading Hypha to Use PostgreSQL

After installing PostgreSQL, update your Hypha configuration:

1. **Update the Hypha values** to use PostgreSQL. Edit your `hypha-server/values.yaml` file:

```yaml
env:
  # ... existing environment variables ...
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: POSTGRES_PASSWORD
  - name: HYPHA_DATABASE_URI
    value: "postgresql+asyncpg://hypha-admin:$(POSTGRES_PASSWORD)@hypha-sql-database-postgresql.hypha.svc.cluster.local:5432/hypha-db"

persistence:
  enabled: false  # Disable persistence since PostgreSQL handles storage
```

**Note**: The `POSTGRES_PASSWORD` environment variable must be defined before `HYPHA_DATABASE_URI` for the password interpolation to work correctly.

2. **Upgrade the Hypha deployment**:
```bash
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

### PostgreSQL Customization

You can customize PostgreSQL by modifying the `postgresql/values.yaml` file:

```yaml
primary:
  persistence:
    size: 20Gi  # Increase storage size
    storageClass: "fast-ssd"  # Use different storage class
  
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
```

### Install Released Helm Charts

You can also install the released helm charts from the [hypha helm repository](https://amun-ai.github.io/hypha):

```bash
helm repo add hypha https://amun-ai.github.io/hypha
helm repo update
helm install hypha-server hypha/hypha-server --namespace=hypha
```

To override the values, you can prepare a `values.yaml` file with the values you want to override and install the helm chart with the following command:
```bash
helm install hypha-server hypha/hypha-server --namespace=hypha -f values.yaml
```

## Install Redis for scaling

Hypha can use an external Redis to store global state and pub/sub messages shared between the server instances.

If you want to scale Hypha horizontally, you can install Redis as a standalone service, and this will allow you to run multiple Hypha server instances to serve more users.

First, install the Redis helm chart:
```bash
helm install redis ./redis --namespace=hypha
```

Now change the `values.yaml` file to add the `redis-uri` to the `startupCommand`:
```yaml
startupCommand:
  command: ["python", "-m", "hypha.server"]
  args:
    - "--host=0.0.0.0"
    - "--port=9520"
    - "--public-base-url=$(PUBLIC_BASE_URL)"
    - "--redis-uri=redis://redis.hypha.svc.cluster.local:6379/0"
```

To actually support multiple server instances, you need to set the `replicaCount` to more than 1 in the `values.yaml` file:

```yaml
replicaCount: 3
```

You also need to set the `HYPHA_SERVER_ID` environment variable to the pod's UID in the `values.yaml` file:
```yaml
env:
  - name: HYPHA_SERVER_ID
    valueFrom:
      # Use the pod's UID as the server ID
      fieldRef:
        fieldPath: metadata.uid
```

Make sure to update the `values.yaml` file with the correct `redis-uri` and `replicaCount`, and add the `HYPHA_SERVER_ID` environment variable, then upgrade the helm chart:
```bash
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

## Install MinIO for S3-compatible storage

Hypha can use MinIO as an S3-compatible storage backend for artifacts and file storage. This provides scalable object storage capabilities.

### Installing MinIO

First, create the MinIO credentials secret:
```bash
kubectl create secret generic minio-credentials \
  --from-literal=ACCESS_KEY_ID=minioadmin \
  --from-literal=SECRET_ACCESS_KEY=minioadmin \
  --namespace=hypha
```

Then add the Bitnami repository and install the MinIO helm chart:
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
cd minio
helm dependency update
helm install minio . --namespace=hypha
```

### Configuring Hypha to use MinIO

After installing MinIO, you need to configure Hypha to use it. Update the `hypha-server/values.yaml` file to enable S3 storage:

```yaml
env:
  # ... existing environment variables ...
  
  # Enable S3 storage
  - name: HYPHA_ENABLE_S3
    value: "true"
  - name: HYPHA_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: minio-credentials
        key: ACCESS_KEY_ID
  - name: HYPHA_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: minio-credentials
        key: SECRET_ACCESS_KEY
  - name: HYPHA_ENDPOINT_URL
    value: "http://minio.hypha.svc.cluster.local:9000"
  - name: HYPHA_S3_ADMIN_TYPE
    value: "minio"
  - name: HYPHA_ENABLE_S3_PROXY
    value: "true"
```

### Using the same secret for both MinIO and Hypha

Since both MinIO and Hypha use the same `minio-credentials` secret, no additional configuration is needed. The secret created during MinIO installation will be automatically used by Hypha when you enable S3 storage.

### Upgrading Hypha with MinIO configuration

After updating the values, upgrade the Hypha server:
```bash
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

### Accessing MinIO

You can access the MinIO console to manage buckets and files:
```bash
kubectl port-forward svc/minio 9001:9001 --namespace=hypha
```

Then open http://localhost:9001 in your browser and login with the credentials from the `minio-credentials` secret:

```bash
# Get the credentials
kubectl get secret minio-credentials -o jsonpath='{.data.ACCESS_KEY_ID}' | base64 -d
kubectl get secret minio-credentials -o jsonpath='{.data.SECRET_ACCESS_KEY}' | base64 -d
```

### MinIO Configuration Options

You can customize MinIO by creating a custom values file (`minio-values.yaml`):

```yaml
minio:
  auth:
    # Use custom secret with different key names
    existingSecret: "minio-credentials"
    existingSecretKeys:
      rootUser: "ACCESS_KEY_ID"
      rootPassword: "SECRET_ACCESS_KEY"
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
helm install minio . --namespace=hypha -f minio-values.yaml
```

For more MinIO configuration options, see the [MinIO chart documentation](helm-charts/minio/README.md).

## Complete Configuration Example

Here's an example of a complete `hypha-server/values.yaml` configuration using PostgreSQL, MinIO, and Redis:

```yaml
env:
  # JWT Secret
  - name: HYPHA_JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: HYPHA_JWT_SECRET
  
  # Basic configuration
  - name: HYPHA_HOST
    value: "0.0.0.0"
  - name: HYPHA_PORT
    value: "9520"
  - name: HYPHA_PUBLIC_BASE_URL
    value: "https://hypha.amun.ai"
  
  # PostgreSQL configuration
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: POSTGRES_PASSWORD
  - name: HYPHA_DATABASE_URI
    value: "postgresql+asyncpg://hypha-admin:$(POSTGRES_PASSWORD)@hypha-sql-database-postgresql.hypha.svc.cluster.local:5432/hypha-db"
  
  # Redis configuration (for scaling)
  - name: HYPHA_REDIS_URI
    value: "redis://redis.hypha.svc.cluster.local:6379/0"
  
  # MinIO S3 configuration
  - name: HYPHA_ENABLE_S3
    value: "true"
  - name: HYPHA_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: minio-credentials
        key: ACCESS_KEY_ID
  - name: HYPHA_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: minio-credentials
        key: SECRET_ACCESS_KEY
  - name: HYPHA_ENDPOINT_URL
    value: "http://minio.hypha.svc.cluster.local:9000"
  - name: HYPHA_S3_ADMIN_TYPE
    value: "minio"
  - name: HYPHA_ENABLE_S3_PROXY
    value: "true"
  
  # Server ID for multiple instances
  - name: HYPHA_SERVER_ID
    valueFrom:
      fieldRef:
        fieldPath: metadata.uid

# Disable persistence when using PostgreSQL
persistence:
  enabled: false

# Enable multiple replicas when using Redis
replicaCount: 3
```

## Setting up on Azure Kubernetes Service (AKS)

If you are deploying Hypha on Azure Kubernetes Service (AKS), you will need to configure the ingress to use the Azure Application Gateway. You can follow the instructions in the [aks-hypha.md](aks-hypha.md) file.
