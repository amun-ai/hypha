# Minimal Hypha Helm Chart

This is a **minimal helm chart** to deploy Hypha on a Kubernetes cluster. This does not include standalone redis or s3, if you are interested in more advanced deployment, please refer to the [hypha-helm-charts](https://github.com/amun-ai/hypha-helm-charts) repository.

## Prerequisites

Make sure you have the following installed:
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [helm](https://helm.sh/docs/intro/install/)

## Usage

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

## Persistence Configuration

Hypha uses persistent storage by default to store the application database and other data. You can configure persistence options in the `values.yaml` file:

### Disable Persistence

For testing or development environments, you can disable persistence entirely:

```yaml
persistence:
  enabled: false
```

### Customize Persistence

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
helm dependency build
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
helm install minio . --namespace=hypha -f minio-values.yaml
```

For more MinIO configuration options, see the [MinIO chart documentation](helm-charts/minio/README.md).

## Setting up on Azure Kubernetes Service (AKS)

If you are deploying Hypha on Azure Kubernetes Service (AKS), you will need to configure the ingress to use the Azure Application Gateway. You can follow the instructions in the [aks-hypha.md](aks-hypha.md) file.
