# Hypha Server Kit Helm Chart

A comprehensive Helm chart that bundles Hypha Server with all its dependencies (MinIO, PostgreSQL, Redis) in a single deployment package.

## What's Included

This kit includes:

- **Hypha Server**: Core application
- **PostgreSQL**: Database backend (persistent storage)
- **Redis**: Cache and message broker (for scaling)
- **MinIO**: S3-compatible object storage

Each component is pre-configured to work together seamlessly.


## Prerequisites

- Kubernetes 1.19+ cluster
- Helm 3.2.0+ installed
- kubectl configured to access your cluster
- (Optional) NGINX Ingress Controller for web access
- (Optional) cert-manager for automatic TLS certificates

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install NGINX Ingress Controller if needed
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# Install cert-manager (for automatic TLS certificates) if needed
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

## Quick Start

### 1. Clone and Navigate

```bash
git clone https://github.com/amun-ai/hypha.git
cd hypha/helm-charts/hypha-server-kit
```

### 2. Create Secrets

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file and update all passwords and secrets
# IMPORTANT: This file now contains ONLY secrets - all other config is in values.yaml
nano .env

# Create Kubernetes secrets from your .env file
./create-secrets.sh .env hypha
```

### 3. Configure Values

```bash
# Edit values.yaml to customize your deployment
nano values.yaml

# Key settings to update:
# - hypha-server.ingress.hosts[0].host: Your domain name
# - hypha-server.env: Update HYPHA_PUBLIC_BASE_URL and other configuration
# - hypha-server.env: Update HYPHA_ENDPOINT_URL_PUBLIC for S3 access (if using)
# - All non-secret configuration is now in values.yaml
```

### 4. Install Hypha Server Kit

```bash
# Install the chart
helm install hypha-server-kit . --namespace hypha
```

### 5. Verify Installation

```bash
# Check all pods are running
kubectl get pods -n hypha

# Check services
kubectl get svc -n hypha
```

## Configuration

### Secrets Management

All sensitive data (passwords, tokens, keys) is stored in Kubernetes secrets. The `.env.example` file contains ONLY secret values:

- **JWT Secret**: Authentication token secret
- **Database Passwords**: PostgreSQL credentials  
- **Redis Password**: Cache authentication
- **MinIO Credentials**: S3-compatible storage access
- **OAuth Secrets**: Optional authentication provider secrets
- **SMTP Password**: Optional email sending password

All other configuration (domains, ports, feature flags, etc.) is now in `values.yaml`.

### Configuration Files

1. **`.env`** - Contains ONLY secret values like passwords and tokens (create from `.env.example`)
2. **`values.yaml`** - All non-secret configuration including:
   - Domain names and URLs
   - Feature flags and settings
   - Resource limits
   - Performance tuning
   - Authentication provider configuration (non-secret parts)
   - SMTP configuration (non-secret parts)
3. **`create-secrets.sh`** - Script that creates Kubernetes secrets from `.env`



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


## Common Operations

### Access Hypha

If using ingress:
```bash
# Open in browser
https://hypha.example.com
```

For local testing without ingress:
```bash
# Port forward
kubectl port-forward svc/hypha-server-kit 9520:9520 -n hypha
# Access at http://localhost:9520
```

### Update Configuration

```bash
# Edit your .env file with new secret values
nano .env

# Re-create the secrets
./create-secrets.sh .env hypha

# For non-secret configuration changes, edit values.yaml
nano values.yaml

# Apply the changes
helm upgrade hypha-server-kit . --namespace hypha

# Or restart pods to pick up secret changes
kubectl rollout restart deployment -n hypha
```

### Upgrade the Chart

```bash
# Update values if needed
nano values.yaml

# Upgrade the release
helm upgrade hypha-server-kit . --namespace hypha
```

### Uninstall

```bash
# Remove the chart
helm uninstall hypha-server-kit --namespace hypha

# Optionally remove the namespace
kubectl delete namespace hypha
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


## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -n hypha
kubectl describe pod <pod-name> -n hypha
kubectl logs <pod-name> -n hypha
```

### Verify Secrets
```bash
kubectl get secret hypha-secrets -n hypha
kubectl describe secret hypha-secrets -n hypha
```

### Test Connectivity
```bash
# Test Hypha API
kubectl run -it --rm test --image=curlimages/curl --restart=Never -n hypha -- \
  curl http://hypha-server-kit:9520/health

# Test PostgreSQL
kubectl exec -it hypha-server-kit-postgresql-0 -n hypha -- \
  pg_isready -U hypha -d hypha_db

# Test Redis
kubectl exec -it hypha-server-kit-redis-master-0 -n hypha -- \
  redis-cli ping
```


## Production Considerations

1. **Change All Default Passwords**: Update all passwords in `.env` before deploying
2. **Configure Ingress**: Set up proper domain names and TLS certificates
3. **Resource Limits**: Adjust CPU/memory in `values.yaml` based on your needs
4. **Persistence**: Ensure your cluster has appropriate storage classes
5. **Backups**: Implement backup strategies for PostgreSQL and MinIO data
6. **Monitoring**: Consider adding Prometheus/Grafana for monitoring

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/amun-ai/hypha/issues
- **Documentation**: https://docs.amun.ai