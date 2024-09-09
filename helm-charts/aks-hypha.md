# Install Hypha on Azure Kubernetes (AKS)

This guide walks you through the process of installing Hypha on an Azure Kubernetes Service (AKS) cluster, setting up an ingress controller, and configuring TLS certificates using `cert-manager` with Let's Encrypt.

## Prerequisites

Ensure the following tools are installed:

- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [Helm](https://helm.sh/docs/intro/install/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

## Steps to Install Hypha on Azure Kubernetes

### 1. Create an AKS Cluster

First, create an AKS cluster using the Azure CLI. Replace `<your-resource-group>` and `<your-cluster-name>` with your own values:

```bash
az group create --name <your-resource-group> --location <your-location>
az aks create --resource-group <your-resource-group> --name <your-cluster-name> --node-count 3 --enable-addons monitoring --generate-ssh-keys
```

After the AKS cluster is created, connect to it:

```bash
az aks get-credentials --resource-group <your-resource-group> --name <your-cluster-name>
```

### 2. Setup Ingress Controller in Azure K8s

Azure Kubernetes Service (AKS) comes with a built-in ingress controller called `Application Gateway Ingress Controller (AGIC)`. To enable the ingress controller on AKS:

```bash
az aks enable-addons --resource-group <your-resource-group> --name <your-cluster-name> --addons ingress-appgw --appgw-id <your-appgw-id>
```

This command will enable the ingress controller, and the `nginx` deployment will be added under the `app-routing-system` namespace. The ingress controller's class will be `webapprouting.kubernetes.azure.com`.

### 3. Setup `cert-manager` for TLS Certificates

`cert-manager` automates the management and issuance of TLS certificates in Kubernetes, and we’ll use it with Let's Encrypt for SSL.

#### Step 1: Install `cert-manager` using Helm

Add the `cert-manager` Helm repository:

```bash
helm repo add jetstack https://charts.jetstack.io
helm repo update
```

Install `cert-manager` and its required CRDs:

```bash
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --version v1.12.2 \
  --set installCRDs=true
```

#### Step 2: Create a Let's Encrypt `ClusterIssuer`

After installing `cert-manager`, create a `ClusterIssuer` for Let's Encrypt:

Create a file named `clusterissuer.yaml`:

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: webapprouting.kubernetes.azure.com
```

(Remember to replace `your-email@example.com` to your actual email.

Apply the `ClusterIssuer` to the cluster:

```bash
kubectl apply -f clusterissuer.yaml
```

Verify the `ClusterIssuer`:

```bash
kubectl get clusterissuer letsencrypt-prod
kubectl describe clusterissuer letsencrypt-prod
```

### 4. Install Hypha

#### Step 1: Clone the Hypha Helm Chart Repository

Download the Hypha Helm chart from the GitHub repository:

```bash
git clone https://github.com/amun-ai/hypha.git
cd hypha/helm-charts
```

#### Step 2: Create a Namespace for Hypha

Create a new namespace for Hypha:

```bash
kubectl create namespace hypha
```

#### Step 3: Create Secrets for Hypha

Generate a `JWT_SECRET` and store it as a Kubernetes secret. Change the secret to your own:

```bash
export JWT_SECRET=abcde123 # change this to your own secret
kubectl create secret generic hypha-secrets \
  --from-literal=JWT_SECRET=$JWT_SECRET \
  --dry-run=client -o yaml | kubectl apply --namespace=hypha -f -
```

Verify that the secret `hypha-secrets` has been created:

```bash
kubectl get secrets --namespace=hypha
```

#### Step 4: Configure the `values.yaml` File

Update the `values.yaml` file to match your deployment. You’ll need to modify the hostnames and ensure the ingress is correctly configured for your domain.

Here’s a sample `values.yaml` for your deployment:

```yaml
replicaCount: 1

image:
  repository: ghcr.io/amun-ai/hypha
  pullPolicy: IfNotPresent
  tag: "0.20.34"

serviceAccount:
  create: true
  automount: true

service:
  type: ClusterIP
  port: 9520

ingress:
  enabled: true
  className: webapprouting.kubernetes.azure.com
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: hypha.my-company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: hypha.my-company.com-tls
      hosts:
        - hypha.my-company.com

resources: {}

livenessProbe:
  httpGet:
    path: /health/liveness
    port: 9520
readinessProbe:
  httpGet:
    path: /health/readiness
    port: 9520

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 100
  targetCPUUtilizationPercentage: 80

env:
  - name: JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: hypha-secrets
        key: JWT_SECRET
  - name: PUBLIC_BASE_URL
    value: "https://hypha.my-company.com"

startupCommand:
  command: ["python", "-m", "hypha.server"]
  args:
    - "--host=0.0.0.0"
    - "--port=9520"
    - "--public-base-url=$(PUBLIC_BASE_URL)"
    # - "--redis-uri=redis://redis.hypha.svc.cluster.local:6379/0"
```

Replace all instances of `hypha.my-company.com` with your own domain.

#### Step 4: Install Redis for Scaling

If you want to scale Hypha horizontally, you can install Redis as a standalone service. This will allow you to run multiple Hypha server instances to serve more users.

First, install the Redis Helm chart:

```bash
helm install redis ./redis --namespace hypha
```

This will install Redis in the `hypha` namespace, make sure you update the `values.yaml` file to add the `redis-uri` to the `startupCommand`:
  
```yaml
startupCommand:
  command: ["python", "-m", "hypha.server"]
  args:
    - "--host=0.0.0.0"
    - "--port=9520"
    - "--public-base-url=$(PUBLIC_BASE_URL)"
    - "--redis-uri=redis://redis.hypha.svc.cluster.local:6379/0"
```

#### Step 6: Install Hypha Using Helm

Now that the configuration is set, install Hypha using Helm (make sure you are in the `helm-charts` directory):

```bash
helm install hypha-server ./hypha-server --namespace hypha
```

To upgrade the Helm chart with any changes:

```bash
helm upgrade hypha-server ./hypha-server --namespace hypha
```

To uninstall the Helm chart:

```bash
helm uninstall hypha-server --namespace hypha
```

### 5. Verify TLS and Ingress Setup

#### Step 1: Check Ingress

Ensure that the ingress resource is created and points to the correct `ClusterIssuer`. Run the following command to check the ingress status:

```bash
kubectl get ingress -n hypha
```

#### Step 2: Check TLS Certificate

Check if the certificate has been issued by Let's Encrypt and the secret was created:

```bash
kubectl get secret hypha.my-company.com-tls -n hypha
```
(replace `hypha.my-company.com` to your domain name)

If the secret exists, it means the certificate has been issued successfully.

#### Step 3: Verify in Browser

Open your browser and navigate to `https://hypha.my-company.com`. The site should be secured with an SSL certificate from Let's Encrypt.

---

### Troubleshooting

1. **Certificate Request Stuck:**

   Check the certificate request status:

   ```bash
   kubectl get certificaterequest -n hypha
   ```

   If the request is stuck, describe it to see what went wrong:

   ```bash
   kubectl describe certificaterequest <certificate-request-name> -n hypha
   ```

2. **Check `cert-manager` Logs:**

   If you're having trouble with certificate issuance, check the logs of `cert-manager`:

   ```bash
   kubectl logs -n cert-manager deploy/cert-manager
   ```

### Conclusion

By following these steps, you will have a fully functioning Hypha application running on an AKS cluster with a secure ingress using `cert-manager` and Let's Encrypt. This setup ensures automatic management of TLS certificates, making your application more secure and easier to maintain.
