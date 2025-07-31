# Install PostgreSQL

```bash
# 1. Add Bitnami Helm repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# 2. Create values.yaml file with storageClass: retainer and secret configuration

# 3. Deploy PostgreSQL using Helm with values file
helm upgrade --install hypha-sql-database bitnami/postgresql \
  -f values.yaml \
  --namespace hypha \
  --version 16.1.0
```

For deployment, a `values.yaml` file is provided with the following configuration:
- Using the existing `hypha-secrets` for passwords
- Setting storage class to `retainer` for improved data retention (Retain reclaim policy)
- Configuring proper security context for container security

## Connection Information

After deploying the PostgreSQL database, you can connect to it using:

```bash
# Forward the PostgreSQL port to your local machine
kubectl port-forward --namespace hypha svc/hypha-sql-database-postgresql 5432:5432

# Get the password from hypha-secrets
export POSTGRES_PASSWORD=$(kubectl get secret --namespace hypha hypha-secrets -o jsonpath="{.data.POSTGRES_PASSWORD}" | base64 -d)

# Connect using psql
PGPASSWORD="$POSTGRES_PASSWORD" psql --host 127.0.0.1 -U hypha-admin -d hypha-db -p 5432
```


For connecting from inside the Kubernetes cluster (e.g., from another pod), use the following hostname:
```
DB_HOST = "hypha-sql-database-postgresql.hypha.svc.cluster.local"
```
