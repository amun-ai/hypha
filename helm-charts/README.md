# Minimal Hypha Helm Chart

This is a **minimal helm chart** to deploy Hypha on a Kubernetes cluster. This does not include standalone redis or s3, if you are interested in more advanced deployment, please refer to the [hypha-helm-charts](https://github.com/amun-ai/hypha-helm-charts) repository.

## Prerequisites

Make sure you have the following installed:
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [helm](https://helm.sh/docs/intro/install/)

## Usage

First, clone this repository:
```bash
git clone https://amun-ai.github.com/hypha.git
cd hypha/helm-charts
```

Now you need to create a namespace for Hypha:
```bash
kubectl create namespace hypha
```

Then, run the bash command to create a `JWT_SECRET` apply these secrets to the cluster:
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

Next, let's check the values in `values.yaml` and make sure they are correct. Importantly, search and replace all the `hypha.amun.ai` to your own public domain name and make sure the ingress is correctly configured.


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

Now, upgrade the helm chart:
```bash
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

## Setting up on Azure Kubernetes Service (AKS)

If you are deploying Hypha on Azure Kubernetes Service (AKS), you will need to configure the ingress to use the Azure Application Gateway. You can follow the instructions in the [aks-hypha.md](aks-hypha.md) file.
