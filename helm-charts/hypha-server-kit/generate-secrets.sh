#!/bin/bash

# Hypha Server Kit - Secret Generation Script
# This script generates all required secrets and saves them to a .env file
# Usage: ./generate-secrets.sh [output-file]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default output file
OUTPUT_FILE="${1:-.env}"

print_info "Generating secrets for Hypha Server Kit..."
print_info "Output file: $OUTPUT_FILE"

# Check if output file exists
if [ -f "$OUTPUT_FILE" ]; then
    print_warning "File '$OUTPUT_FILE' already exists!"
    read -p "Do you want to overwrite it? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Operation cancelled."
        exit 0
    fi
fi

# Generate secrets
print_info "Generating random secrets..."

# Core Hypha secrets
HYPHA_JWT_SECRET=$(openssl rand -base64 32)
HYPHA_ACCESS_KEY_ID=$(openssl rand -base64 32 | tr -d '=' | head -c 20)
HYPHA_SECRET_ACCESS_KEY=$(openssl rand -base64 32)

# Database and Redis passwords (URL-safe)
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '=' | tr -d '+' | tr -d '/')
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '=' | tr -d '+' | tr -d '/')

# Construct connection URIs
HYPHA_DATABASE_URI="postgresql://hypha:${POSTGRES_PASSWORD}@hypha-server-kit-postgresql:5432/hypha_db"
HYPHA_REDIS_URI="redis://:${REDIS_PASSWORD}@hypha-server-kit-redis-master:6379/0"

# Create .env file
cat > "$OUTPUT_FILE" << EOF
# Hypha Server Kit Secrets
# Generated on: $(date)
# DO NOT commit this file to version control!

# Core Hypha Configuration
HYPHA_JWT_SECRET=$HYPHA_JWT_SECRET

# S3/MinIO Credentials (shared between Hypha and MinIO)
HYPHA_ACCESS_KEY_ID=$HYPHA_ACCESS_KEY_ID
HYPHA_SECRET_ACCESS_KEY=$HYPHA_SECRET_ACCESS_KEY

# Database Configuration
HYPHA_DATABASE_URI=$HYPHA_DATABASE_URI
POSTGRES_PASSWORD=$POSTGRES_PASSWORD

# Redis Configuration  
HYPHA_REDIS_URI=$HYPHA_REDIS_URI
REDIS_PASSWORD=$REDIS_PASSWORD

EOF

print_success "Secrets generated and saved to '$OUTPUT_FILE'"

echo ""
print_info "Summary of generated secrets:"
echo "  - HYPHA_JWT_SECRET: JWT authentication secret"
echo "  - HYPHA_ACCESS_KEY_ID: S3/MinIO access key (20 chars)"
echo "  - HYPHA_SECRET_ACCESS_KEY: S3/MinIO secret key"
echo "  - HYPHA_DATABASE_URI: Complete PostgreSQL connection string"
echo "  - POSTGRES_PASSWORD: PostgreSQL password (also embedded in URI)"
echo "  - HYPHA_REDIS_URI: Complete Redis connection string"
echo "  - REDIS_PASSWORD: Redis password (also embedded in URI)"

echo ""
print_info "Next steps:"
echo "  1. Review and customize '$OUTPUT_FILE' if needed"
echo "  2. Update values.yaml with your domain and configuration"
echo "  3. Create Kubernetes secrets: ./create-secrets.sh '$OUTPUT_FILE' hypha"
echo "  4. Install the chart: helm install hypha-server-kit . --namespace hypha"

echo ""
print_warning "Security reminders:"
echo "  - Keep '$OUTPUT_FILE' secure and do not commit to version control"
echo "  - Consider adding '$OUTPUT_FILE' to your .gitignore"
echo "  - For production, review all generated passwords and consider using your own"
echo "  - Regularly rotate secrets in production environments"