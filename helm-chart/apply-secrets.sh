#!/bin/bash

# Load the .env file
set -a
source .env
set +a

# Create the Kubernetes secret
kubectl create secret generic hypha-secrets \
  --from-literal=JWT_SECRET=$JWT_SECRET \
  --dry-run=client -o yaml | kubectl apply --namespace=hypha -f -
