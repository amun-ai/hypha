#!/bin/bash

# Load the .env file
set -a
source .env
set +a

# Create the Kubernetes secret
kubectl create secret generic hypha-secrets \
  --from-literal=HYPHA_JWT_SECRET=$HYPHA_JWT_SECRET \
  --dry-run=client -o yaml | kubectl apply --namespace=hypha -f -
