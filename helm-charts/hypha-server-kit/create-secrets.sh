#!/bin/bash

# Script to create Kubernetes Secrets from .env file
# Usage: ./create-secrets.sh <env-file> <namespace>
# Example: ./create-secrets.sh .env hypha

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <env-file> <namespace>"
    echo "Example: $0 .env hypha"
    exit 1
fi

ENV_FILE=$1
NAMESPACE=$2
SECRET_NAME="hypha-secrets"

# Check if env file exists
if [ ! -f "$ENV_FILE" ]; then
    print_error "Environment file '$ENV_FILE' not found!"
    echo "Please copy .env.example to .env and configure it:"
    echo "  cp .env.example .env"
    exit 1
fi

# Check if namespace exists, create if not
if ! kubectl get namespace "$NAMESPACE" > /dev/null 2>&1; then
    print_info "Creating namespace '$NAMESPACE'..."
    kubectl create namespace "$NAMESPACE"
fi

# Extract only secret values from env file
print_info "Extracting secrets from environment file..."

# Create temporary file for secrets
SECRETS_FILE=$(mktemp)

# Variables to store MinIO credentials
MINIO_USER=""
MINIO_PASS=""

# Extract specific secrets from the env file
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    if [[ -z "$key" ]] || [[ "$key" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Remove leading/trailing whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    
    # Store MinIO credentials for later
    if [ "$key" = "MINIO_ROOT_USER" ]; then
        MINIO_USER="$value"
    elif [ "$key" = "MINIO_ROOT_PASSWORD" ]; then
        MINIO_PASS="$value"
    fi
    
    # Only include specific secrets
    case "$key" in
        HYPHA_JWT_SECRET|\
        HYPHA_DATABASE_URI|\
        HYPHA_REDIS_URI|\
        POSTGRES_PASSWORD|\
        HYPHA_ACCESS_KEY_ID|\
        HYPHA_SECRET_ACCESS_KEY|\
        REDIS_PASSWORD|\
        MINIO_ROOT_USER|\
        MINIO_ROOT_PASSWORD)
            echo "${key}=${value}" >> "$SECRETS_FILE"
            ;;
    esac
done < "$ENV_FILE"

# Add MinIO-specific keys for Bitnami chart compatibility
if [ -n "$MINIO_USER" ] && [ -n "$MINIO_PASS" ]; then
    print_info "Adding MinIO-specific keys for Bitnami chart compatibility..."
    echo "root-user=${MINIO_USER}" >> "$SECRETS_FILE"
    echo "root-password=${MINIO_PASS}" >> "$SECRETS_FILE"
fi

# Create/update Secrets
if [ -s "$SECRETS_FILE" ]; then
    print_info "Creating/updating Secret '$SECRET_NAME'..."
    kubectl create secret generic "$SECRET_NAME" \
        --from-env-file="$SECRETS_FILE" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    print_info "Secret '$SECRET_NAME' created/updated successfully"
else
    print_error "No secret variables found in the environment file!"
    exit 1
fi

# Clean up temporary file
rm -f "$SECRETS_FILE"

# Display summary
echo ""
print_info "Summary:"
echo "  Namespace: $NAMESPACE"
echo "  Secret: $SECRET_NAME"
echo ""
print_info "To verify the secret:"
echo "  kubectl get secret $SECRET_NAME -n $NAMESPACE"
echo ""
print_info "Next steps:"
echo "  1. Review and update values.yaml with your configuration"
echo "  2. Run: helm install hypha-server-kit . -n $NAMESPACE"