#!/bin/bash

# Script to create Kubernetes ConfigMap from .env file
# Usage: ./create-configmap-from-env.sh <env-file> <configmap-name> <namespace>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <env-file> <configmap-name> <namespace>"
    echo "Example: $0 .env.production hypha-config hypha"
    exit 1
fi

ENV_FILE=$1
CONFIGMAP_NAME=$2
NAMESPACE=$3

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file '$ENV_FILE' not found!"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" > /dev/null 2>&1; then
    echo "Creating namespace '$NAMESPACE'..."
    kubectl create namespace "$NAMESPACE"
fi

# Create ConfigMap from env file
kubectl create configmap "$CONFIGMAP_NAME" \
  --from-env-file="$ENV_FILE" \
  --namespace="$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "ConfigMap '$CONFIGMAP_NAME' created/updated in namespace '$NAMESPACE'"

# Display the ConfigMap
echo -e "\nConfigMap contents:"
kubectl get configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" -o yaml