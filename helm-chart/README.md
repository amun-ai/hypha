# Minimal Hypha Helm Chart

This is a **minimal helm chart** to deploy Hypha on a Kubernetes cluster. This does not include standalone redis or s3, if you are interested in more advanced deployment, please refer to the [hypha-helm-charts](https://github.com/amun-ai/hypha-helm-charts) repository.

## Prerequisites

Make sure you have the following installed:
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [helm](https://helm.sh/docs/intro/install/)

## Usage

First, we need to create secrets for hypha by copy the file `.env-template` to `.env`, then change the values inside .env file.

After that, run the bash command to apply these secrets to the cluster:
```bash
sh apply-secrets.sh
```

Verify that the secret `hypha-secrets` has been created:
```bash
kubectl get secrets --namespace=hypha
```

Next, let's check the values in `values.yaml` and make sure they are correct. Importantly, search and replace all the `hypha.amun.ai` to your own public domain name and make sure the ingress is correctly configured.


Finally, we can install the helm chart:
```
helm install hypha-server ./hypha-server --namespace=hypha
```

If you make changes to the chart, you can upgrade the chart:
```
helm upgrade hypha-server ./hypha-server --namespace=hypha
```

To uninstall the chart:
```
helm uninstall hypha-server --namespace=hypha
```
