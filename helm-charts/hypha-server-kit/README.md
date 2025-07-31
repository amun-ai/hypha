# Hypha Server Kit Helm Chart

A comprehensive Helm chart that bundles Hypha Server with all its dependencies (MinIO, PostgreSQL, Redis) in a single deployment package. This guide provides detailed instructions for beginners to advanced users on deploying and configuring Hypha in various environments.

## Overview

This chart simplifies the deployment of Hypha by including:
- **Hypha Server**: The core application for building computational workflows
- **MinIO**: S3-compatible object storage for file management (optional)
- **PostgreSQL**: Database backend for persistent data storage (optional)
- **Redis**: Cache and message broker for real-time features (optional)

Each dependency can be enabled/disabled independently, with automatic configuration when enabled.

## Prerequisites

- Kubernetes 1.19+ cluster
- Helm 3.2.0+ installed
- kubectl configured to access your cluster
- PV provisioner support in the underlying infrastructure (for persistence)
- (Optional) cert-manager for automatic TLS certificates
- (Optional) NGINX Ingress Controller for web access

### Installing Prerequisites

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# Install cert-manager (for automatic TLS certificates)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

## Quick Start

For a quick deployment with default settings:

```bash
# Clone the repository
git clone https://github.com/amun-ai/hypha.git
cd hypha

# Deploy with default configuration
helm install hypha-server-kit ./helm-charts/hypha-server-kit \
  --namespace hypha --create-namespace

# Check deployment status
kubectl get pods -n hypha
```

## Installation

### Basic Installation

```bash
# Install from local directory
helm install hypha-server-kit ./helm-charts/hypha-server-kit \
  --namespace hypha \
  --create-namespace
```

### Installation with Custom Values

```bash
# Create custom values file
cat > my-values.yaml <<EOF
hypha-server:
  ingress:
    hosts:
      - host: hypha.example.com
        paths:
          - path: /
            pathType: Prefix
  persistence:
    size: 50Gi

postgresql:
  auth:
    password: my-secure-password
    postgresPassword: my-secure-admin-password

redis:
  auth:
    password: my-redis-password

minio:
  auth:
    rootUser: myadmin
    rootPassword: my-minio-password
EOF

# Install with custom values
helm install hypha-server-kit ./helm-charts/hypha-server-kit \
  --namespace hypha \
  --create-namespace \
  -f my-values.yaml
```

## Configuration Management

### Using Environment Variables with ConfigMap

We provide a script to generate ConfigMaps from .env files. First, copy the example environment file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your specific configuration
vim .env  # or use your preferred editor
```

Then use the provided script to create a ConfigMap:

```bash
# Create ConfigMap from your .env file
./create-configmap-from-env.sh .env hypha-config hypha
```

### Integrating ConfigMap with Helm Deployment

```bash
# Create ConfigMap from .env file
./create-configmap-from-env.sh .env.production hypha-config hypha

# Update your values.yaml to use the ConfigMap
cat > custom-values.yaml <<EOF
hypha-server:
  env:
    - name: HYPHA_HOST
      valueFrom:
        configMapKeyRef:
          name: hypha-config
          key: HYPHA_HOST
    - name: HYPHA_PORT
      valueFrom:
        configMapKeyRef:
          name: hypha-config
          key: HYPHA_PORT
    # Add more environment variables as needed
EOF
```

## Ingress Configuration

### For Cloud Providers (AWS, GCP, Azure)

#### AWS with ALB Ingress Controller

```yaml
hypha-server:
  ingress:
    enabled: true
    className: alb
    annotations:
      alb.ingress.kubernetes.io/target-type: ip
      alb.ingress.kubernetes.io/scheme: internet-facing
      alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:region:account-id:certificate/cert-id
      alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
      alb.ingress.kubernetes.io/ssl-redirect: '443'
      alb.ingress.kubernetes.io/healthcheck-path: /health
    hosts:
      - host: hypha.example.com
        paths:
          - path: /
            pathType: Prefix
```

#### GCP with GCE Ingress

```yaml
hypha-server:
  ingress:
    enabled: true
    className: gce
    annotations:
      kubernetes.io/ingress.global-static-ip-name: hypha-ip
      networking.gke.io/managed-certificates: hypha-cert
      kubernetes.io/ingress.class: gce
    hosts:
      - host: hypha.example.com
        paths:
          - path: /*
            pathType: ImplementationSpecific
```

#### Azure with Application Gateway

```yaml
hypha-server:
  ingress:
    enabled: true
    className: azure/application-gateway
    annotations:
      appgw.ingress.kubernetes.io/ssl-redirect: "true"
      appgw.ingress.kubernetes.io/backend-protocol: http
      appgw.ingress.kubernetes.io/backend-path-prefix: "/"
    hosts:
      - host: hypha.example.com
        paths:
          - path: /
            pathType: Prefix
```

### For Bare Metal / On-Premises

#### Using MetalLB for Load Balancing

```bash
# Install MetalLB
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.7/config/manifests/metallb-native.yaml

# Configure IP address pool
cat <<EOF | kubectl apply -f -
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: first-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.240-192.168.1.250  # Adjust to your network
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: example
  namespace: metallb-system
EOF
```

#### Option 1: NodePort Service (Direct Access - No Ingress)

For bare metal deployments where you want direct access without ingress:

```yaml
hypha-server:
  service:
    type: NodePort
    nodePort: 30080  # Access via any-node-ip:30080
  
  ingress:
    enabled: false  # Disable ingress completely
```

Access Hypha directly at: `http://<any-node-ip>:30080`

#### Option 2: NGINX Ingress with NodePort

If you want to use ingress but don't have a load balancer:

```yaml
hypha-server:
  service:
    type: ClusterIP  # Keep default service type
  
  ingress:
    enabled: true
    className: nginx
    annotations:
      nginx.ingress.kubernetes.io/proxy-body-size: "0"
      nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
      nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    hosts:
      - host: hypha.local  # Add to /etc/hosts
        paths:
          - path: /
            pathType: Prefix

# Configure NGINX Ingress Controller to use NodePort
# helm install ingress-nginx ingress-nginx/ingress-nginx \
#   --set controller.service.type=NodePort \
#   --set controller.service.nodePorts.http=30080 \
#   --set controller.service.nodePorts.https=30443
```

#### Using Traefik Ingress

```yaml
hypha-server:
  ingress:
    enabled: true
    className: traefik
    annotations:
      traefik.ingress.kubernetes.io/router.entrypoints: web,websecure
      traefik.ingress.kubernetes.io/router.tls: "true"
      traefik.ingress.kubernetes.io/router.tls.certresolver: letsencrypt
    hosts:
      - host: hypha.example.com
        paths:
          - path: /
            pathType: Prefix
```

## Security and Secrets

### Managing Secrets Securely

#### Option 1: Using Kubernetes Secrets (Manual)

```bash
# Create secrets manually
kubectl create secret generic hypha-secrets \
  --from-literal=JWT_SECRET=$(openssl rand -base64 32) \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 16) \
  --from-literal=REDIS_PASSWORD=$(openssl rand -base64 16) \
  --from-literal=MINIO_ROOT_PASSWORD=$(openssl rand -base64 16) \
  --namespace hypha
```

#### Option 2: Using Sealed Secrets

```bash
# Install Sealed Secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml

# Install kubeseal CLI
brew install kubeseal  # On macOS
# Or download from GitHub releases

# Create a secret and seal it
echo -n 'mypassword' | kubectl create secret generic hypha-secrets \
  --dry-run=client \
  --from-file=password=/dev/stdin \
  -o yaml | kubeseal -o yaml > sealed-secrets.yaml

# Apply sealed secret
kubectl apply -f sealed-secrets.yaml
```

#### Option 3: Using External Secrets Operator

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace

# Example: Using AWS Secrets Manager
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: hypha-secret-store
  namespace: hypha
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        secretRef:
          accessKeyIDSecretRef:
            name: awssm-secret
            key: access-key
          secretAccessKeySecretRef:
            name: awssm-secret
            key: secret-access-key
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: hypha-secrets
  namespace: hypha
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: hypha-secret-store
    kind: SecretStore
  target:
    name: hypha-secrets
  data:
  - secretKey: JWT_SECRET
    remoteRef:
      key: hypha/jwt-secret
  - secretKey: POSTGRES_PASSWORD
    remoteRef:
      key: hypha/postgres-password
EOF
```

### Security Best Practices

1. **Generate Strong Secrets**:
```bash
# Generate secure random passwords
JWT_SECRET=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 24)
REDIS_PASSWORD=$(openssl rand -base64 24)
MINIO_PASSWORD=$(openssl rand -base64 24)
```

2. **Rotate Secrets Regularly**:
```bash
# Create a script for secret rotation
cat > rotate-secrets.sh <<'EOF'
#!/bin/bash
NAMESPACE=hypha

# Backup current secrets
kubectl get secret hypha-secrets -n $NAMESPACE -o yaml > secrets-backup-$(date +%Y%m%d).yaml

# Generate new secrets
JWT_SECRET=$(openssl rand -base64 32)

# Update secrets
kubectl patch secret hypha-secrets -n $NAMESPACE --type='json' \
  -p='[{"op": "replace", "path": "/data/JWT_SECRET", "value": "'$(echo -n $JWT_SECRET | base64)'"}]'

# Restart pods to pick up new secrets
kubectl rollout restart deployment -n $NAMESPACE
EOF
```

3. **RBAC Configuration**:
```yaml
# Create service account with minimal permissions
apiVersion: v1
kind: ServiceAccount
metadata:
  name: hypha-sa
  namespace: hypha
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: hypha-role
  namespace: hypha
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: hypha-rolebinding
  namespace: hypha
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: hypha-role
subjects:
- kind: ServiceAccount
  name: hypha-sa
  namespace: hypha
```

## Authentication Service

Hypha uses Auth0 internally to manage authentication. This allows the use of various authentication providers, including Google, GitHub, and traditional username/password authentication.

### Default Authentication Configuration

By default, Hypha uses a shared Auth0 configuration managed by the Hypha team. This provides immediate access to authentication without any additional setup. Supported providers include:
- Username/Password authentication
- Google OAuth
- GitHub OAuth

### Custom Auth0 Configuration

You can also set up your own Auth0 account for complete control over authentication. To use your own Auth0:

1. **Create an Auth0 Account**:
   - Sign up at https://auth0.com
   - Create a new application (type: Single Page Application)
   - Configure allowed callback URLs, logout URLs, and web origins

2. **Configure Hypha to Use Your Auth0**:
   ```yaml
   hypha-server:
     env:
       - name: HYPHA_AUTH0_DOMAIN
         value: "your-tenant.auth0.com"
       - name: HYPHA_AUTH0_CLIENT_ID
         value: "your-auth0-client-id"
       - name: HYPHA_AUTH0_CLIENT_SECRET
         valueFrom:
           secretKeyRef:
             name: hypha-auth-secrets
             key: auth0-client-secret
       - name: HYPHA_AUTH0_AUDIENCE
         value: "https://your-api-identifier"
   ```

3. **Enable Desired Authentication Providers**:
   - In Auth0 Dashboard, go to Authentication > Social
   - Enable and configure providers (Google, GitHub, etc.)
   - Set up proper scopes and permissions

For detailed Auth0 setup instructions, see: https://docs.amun.ai/#/setup-authentication

## Environment Variables

### Core Hypha Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HYPHA_HOST` | Host to bind the server | `0.0.0.0` | No |
| `HYPHA_PORT` | Port to bind the server | `9520` | No |
| `HYPHA_LOGLEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` | No |
| `HYPHA_JWT_SECRET` | Secret for JWT token generation | - | Yes |
| `HYPHA_PUBLIC_BASE_URL` | Public URL for the server | - | Yes |
| `HYPHA_DATABASE_URI` | Database connection string | `sqlite:////data/hypha.db` | No |
| `HYPHA_REDIS_URI` | Redis connection string | - | No |
| `HYPHA_RESET_REDIS` | Reset Redis on startup | `false` | No |
| **S3/Storage Configuration** |
| `HYPHA_ENABLE_S3` | Enable S3 storage | `false` | No |
| `HYPHA_S3_ADMIN_TYPE` | S3 admin type (generic, minio) | `generic` | No |
| `HYPHA_ENABLE_S3_PROXY` | Enable S3 proxy | `false` | No |
| `HYPHA_ENDPOINT_URL` | S3 endpoint URL (internal) | - | If S3 enabled |
| `HYPHA_ENDPOINT_URL_PUBLIC` | S3 public endpoint URL | - | If S3 enabled |
| `HYPHA_ACCESS_KEY_ID` | S3 access key | - | If S3 enabled |
| `HYPHA_SECRET_ACCESS_KEY` | S3 secret key | - | If S3 enabled |
| **Feature Flags** |
| `HYPHA_ENABLE_SERVER_APPS` | Enable server apps | `false` | No |
| `HYPHA_ENABLE_MCP` | Enable MCP (Model Context Protocol) | `false` | No |
| `HYPHA_ENABLE_A2A` | Enable A2A (App-to-App) communication | `false` | No |
| `HYPHA_ENABLE_SERVICE_SEARCH` | Enable service search | `false` | No |
| **Authentication** |
| `HYPHA_AUTH0_DOMAIN` | Custom Auth0 domain | - | No |
| `HYPHA_AUTH0_CLIENT_ID` | Custom Auth0 client ID | - | No |
| `HYPHA_AUTH0_CLIENT_SECRET` | Custom Auth0 client secret | - | No |
| `HYPHA_AUTH0_AUDIENCE` | Custom Auth0 audience | - | No |

### Advanced Configuration

Here's a comprehensive example of environment variables configuration for hypha-server:

```yaml
hypha-server:
  env:
    # Core Configuration
    - name: HYPHA_HOST
      value: "0.0.0.0"
    - name: HYPHA_PORT
      value: "9520"
    - name: HYPHA_LOGLEVEL
      value: "INFO"
    - name: HYPHA_PUBLIC_BASE_URL
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: PUBLIC_BASE_URL
    
    # Security
    - name: HYPHA_JWT_SECRET
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: JWT_SECRET
    
    # Database Configuration
    - name: HYPHA_DATABASE_URI
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: HYPHA_DATABASE_URI
    
    # Redis Configuration (optional)
    # - name: HYPHA_REDIS_URI
    #   value: "redis://redis.hypha.svc.cluster.local:6379/0"
    # - name: HYPHA_RESET_REDIS
    #   value: "true"
    
    # S3/MinIO Configuration
    - name: HYPHA_ENABLE_S3
      value: "true"
    - name: HYPHA_S3_ADMIN_TYPE
      value: "generic"  # or "minio" for MinIO-specific features
    - name: HYPHA_ENABLE_S3_PROXY
      value: "true"
    - name: HYPHA_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: ACCESS_KEY_ID
    - name: HYPHA_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: SECRET_ACCESS_KEY
    - name: HYPHA_ENDPOINT_URL
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: ENDPOINT_URL
    - name: HYPHA_ENDPOINT_URL_PUBLIC
      valueFrom:
        secretKeyRef:
          name: hypha-secrets
          key: ENDPOINT_URL_PUBLIC
    
    # Feature Flags
    - name: HYPHA_ENABLE_SERVER_APPS
      value: "true"
    - name: HYPHA_ENABLE_MCP
      value: "true"
    - name: HYPHA_ENABLE_A2A
      value: "true"
    # - name: HYPHA_ENABLE_SERVICE_SEARCH
    #   value: "true"

    
    # Performance tuning (optional)
    # - name: HYPHA_WORKER_THREADS
    #   value: "4"
    # - name: HYPHA_MAX_CONNECTIONS
    #   value: "1000"
    
    # Monitoring (optional)
    # - name: HYPHA_ENABLE_METRICS
    #   value: "true"
    # - name: HYPHA_ENABLE_TRACING
    #   value: "true"
```

### Creating the Required Secrets

Before deploying with the above configuration, create the necessary secrets:

```bash
# Create the main secrets
kubectl create secret generic hypha-secrets \
  --from-literal=JWT_SECRET=$(openssl rand -base64 32) \
  --from-literal=ACCESS_KEY_ID=your-s3-access-key \
  --from-literal=SECRET_ACCESS_KEY=your-s3-secret-key \
  --from-literal=ENDPOINT_URL=http://minio:9000 \
  --from-literal=ENDPOINT_URL_PUBLIC=https://s3.example.com \
  --from-literal=PUBLIC_BASE_URL=https://hypha.example.com \
  --from-literal=HYPHA_DATABASE_URI="postgresql://hypha:password@postgresql:5432/hypha_db" \
  --namespace hypha
```

## Production Deployment

### Production Checklist

- [ ] **Security**
  - [ ] Change all default passwords
  - [ ] Enable TLS/SSL for all services
  - [ ] Configure network policies
  - [ ] Enable pod security policies
  - [ ] Set up RBAC properly
  - [ ] Scan images for vulnerabilities

- [ ] **High Availability**
  - [ ] Configure multiple replicas for Hypha
  - [ ] Enable PostgreSQL replication
  - [ ] Configure Redis Sentinel or Cluster
  - [ ] Set up MinIO in distributed mode

- [ ] **Persistence**
  - [ ] Configure storage classes
  - [ ] Set up backup procedures
  - [ ] Test restore procedures
  - [ ] Configure snapshot policies

- [ ] **Monitoring**
  - [ ] Deploy Prometheus and Grafana
  - [ ] Configure alerts
  - [ ] Set up log aggregation
  - [ ] Enable distributed tracing

- [ ] **Resource Management**
  - [ ] Set resource requests and limits
  - [ ] Configure HPA (Horizontal Pod Autoscaler)
  - [ ] Set up node affinity rules
  - [ ] Configure PodDisruptionBudgets

### Production Values Example

```yaml
# production-values.yaml
global:
  storageClass: "fast-ssd"

hypha-server:
  replicaCount: 3
  
  image:
    pullPolicy: Always
  
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
      nginx.ingress.kubernetes.io/rate-limit: "100"
    hosts:
      - host: hypha.production.com
    tls:
      - secretName: hypha-tls
        hosts:
          - hypha.production.com
  
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  
  persistence:
    size: 100Gi
    storageClass: "fast-ssd"
  
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000

postgresql:
  architecture: replication
  
  auth:
    database: hypha_prod
    username: hypha_prod_user
    existingSecret: postgresql-secrets
  
  primary:
    persistence:
      size: 500Gi
      storageClass: "fast-ssd"
    
    resources:
      requests:
        cpu: "4"
        memory: "8Gi"
      limits:
        cpu: "8"
        memory: "16Gi"
  
  readReplicas:
    replicaCount: 2
    persistence:
      size: 500Gi
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"

redis:
  architecture: replication
  
  auth:
    existingSecret: redis-secrets
  
  master:
    persistence:
      size: 50Gi
      storageClass: "fast-ssd"
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
  
  replica:
    replicaCount: 2
    persistence:
      size: 50Gi
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"

minio:
  mode: distributed
  
  auth:
    existingSecret: minio-secrets
  
  persistence:
    size: 1Ti
    storageClass: "fast-ssd"
  
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"

monitoring:
  enabled: true
  prometheus:
    enabled: true
    retention: 30d
  grafana:
    enabled: true
    adminExistingSecret: grafana-admin-secret

backup:
  enabled: true
  schedule: "0 2 * * *"
  retention: 30
  s3:
    bucket: hypha-backups
    endpoint: s3.amazonaws.com
    existingSecret: backup-s3-secret
```

## Monitoring and Maintenance

### Setting Up Monitoring Stack

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Configure ServiceMonitors for Hypha
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hypha-metrics
  namespace: hypha
spec:
  selector:
    matchLabels:
      app: hypha-server
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF
```

### Log Aggregation with EFK Stack

```bash
# Install Elasticsearch
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch \
  --namespace logging --create-namespace

# Install Kibana
helm install kibana elastic/kibana \
  --namespace logging

# Install Filebeat
helm install filebeat elastic/filebeat \
  --namespace logging \
  --set filebeatConfig.filebeat\.yml.output.elasticsearch.hosts=["elasticsearch-master:9200"]
```

### Backup Procedures

```bash
# Create backup script
cat > backup-hypha.sh <<'EOF'
#!/bin/bash
NAMESPACE=hypha
BACKUP_DIR=/backups/$(date +%Y%m%d-%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
kubectl exec -n $NAMESPACE hypha-postgresql-0 -- \
  pg_dump -U postgres hypha_db | gzip > $BACKUP_DIR/postgres.sql.gz

# Backup MinIO data
kubectl exec -n $NAMESPACE hypha-minio-0 -- \
  mc mirror minio/hypha-storage $BACKUP_DIR/minio/

# Backup Hypha data volume
kubectl exec -n $NAMESPACE hypha-server-0 -- \
  tar czf - /data | gzip > $BACKUP_DIR/hypha-data.tar.gz

# Upload to S3
aws s3 sync $BACKUP_DIR s3://hypha-backups/$(date +%Y%m%d-%H%M%S)/

echo "Backup completed: $BACKUP_DIR"
EOF

# Schedule with CronJob
kubectl create cronjob hypha-backup \
  --image=amazon/aws-cli:latest \
  --schedule="0 2 * * *" \
  --namespace=hypha \
  -- /scripts/backup-hypha.sh
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Pods Stuck in Pending State

```bash
# Check PVC status
kubectl get pvc -n hypha

# Check events
kubectl describe pod <pod-name> -n hypha

# Solution: Check storage class
kubectl get storageclass

# If no default storage class, set one:
kubectl patch storageclass <storage-class-name> \
  -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

#### 2. Connection Refused Errors

```bash
# Check service endpoints
kubectl get endpoints -n hypha

# Check pod logs
kubectl logs -l app=hypha-server -n hypha

# Test internal connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -n hypha -- \
  wget -O- http://hypha-server-kit:9520/health
```

#### 3. Authentication Issues

```bash
# Verify secrets are created
kubectl get secrets -n hypha

# Check secret contents (base64 encoded)
kubectl get secret hypha-secrets -n hypha -o yaml

# Regenerate JWT secret
kubectl patch secret hypha-secrets -n hypha \
  --type='json' -p='[{"op": "replace", "path": "/data/JWT_SECRET", 
  "value": "'$(openssl rand -base64 32 | base64)'"}]'

# Restart pods to pick up new secret
kubectl rollout restart deployment hypha-server -n hypha
```

#### 4. Performance Issues

```bash
# Check resource usage
kubectl top pods -n hypha
kubectl top nodes

# Check for throttling
kubectl describe pod <pod-name> -n hypha | grep -A 5 "Limits:"

# Scale up if needed
kubectl scale deployment hypha-server --replicas=5 -n hypha
```

#### 5. Storage Issues

```bash
# Check disk usage
kubectl exec -it <pod-name> -n hypha -- df -h

# Expand PVC (if storage class supports it)
kubectl patch pvc <pvc-name> -n hypha \
  -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Clean up old data
kubectl exec -it <pod-name> -n hypha -- \
  find /data -name "*.log" -mtime +30 -delete
```

### Debug Commands

```bash
# Get all resources in namespace
kubectl get all -n hypha

# Describe all pods
kubectl describe pods -n hypha

# Get logs from all containers
kubectl logs -l app=hypha-server -n hypha --all-containers=true

# Access pod shell
kubectl exec -it <pod-name> -n hypha -- /bin/bash

# Port forward for local debugging
kubectl port-forward svc/hypha-server-kit 9520:9520 -n hypha

# Check cluster DNS
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup hypha-server-kit.hypha.svc.cluster.local
```

### Health Checks

```bash
# Create health check script
cat > health-check.sh <<'EOF'
#!/bin/bash
NAMESPACE=hypha

echo "Checking Hypha deployment health..."

# Check pods
echo -e "\n=== Pod Status ==="
kubectl get pods -n $NAMESPACE

# Check services
echo -e "\n=== Service Status ==="
kubectl get svc -n $NAMESPACE

# Check endpoints
echo -e "\n=== Endpoints ==="
kubectl get endpoints -n $NAMESPACE

# Test Hypha API
echo -e "\n=== Hypha API Test ==="
kubectl run -it --rm api-test --image=curlimages/curl --restart=Never -n $NAMESPACE -- \
  curl -s http://hypha-server-kit:9520/health

# Test MinIO
echo -e "\n=== MinIO Test ==="
kubectl run -it --rm minio-test --image=minio/mc --restart=Never -n $NAMESPACE -- \
  mc alias set minio http://hypha-server-kit-minio:9000 admin changeme123

# Test PostgreSQL
echo -e "\n=== PostgreSQL Test ==="
kubectl exec -it hypha-server-kit-postgresql-0 -n $NAMESPACE -- \
  pg_isready -U hypha -d hypha_db

# Test Redis
echo -e "\n=== Redis Test ==="
kubectl exec -it hypha-server-kit-redis-master-0 -n $NAMESPACE -- \
  redis-cli ping
EOF

chmod +x health-check.sh
```

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/amun-ai/hypha/issues
- **Documentation**: https://docs.amun.ai