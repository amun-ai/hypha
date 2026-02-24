# Hypha Server with Docker Compose

This directory contains a comprehensive Docker Compose setup for running Hypha Server with all its dependencies: PostgreSQL (database), Redis (cache/messaging), and MinIO (S3 storage).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1. Clone the repository and navigate to the docker-compose directory:

   ```bash
   git clone https://github.com/amun-ai/hypha.git
   cd hypha/docker-compose
   ```

2. Configure the environment variables:

   ```bash
   cp .env-docker .env
   ```

   **IMPORTANT**: Open the `.env` file and update the default passwords and secrets:
   - `HYPHA_JWT_SECRET`: Change to a secure random string (32+ characters)
   - `POSTGRES_PASSWORD`: PostgreSQL database password
   - `REDIS_PASSWORD`: Redis cache password
   - `HYPHA_ACCESS_KEY_ID` & `HYPHA_SECRET_ACCESS_KEY`: MinIO credentials

3. Create data directories:

   ```bash
   mkdir -p ./data/{hypha,postgres,redis,minio}
   ```

4. Start the services:

   ```bash
   docker-compose up -d
   ```

5. Check that all services are running:

   ```bash
   docker-compose ps
   ```

6. Access the services:
   - **Hypha Server**: <http://localhost:9527>
   - **MinIO Console**: <http://localhost:9001> (login with HYPHA_ACCESS_KEY_ID/HYPHA_SECRET_ACCESS_KEY)
   - **PostgreSQL**: localhost:5432 (use POSTGRES_USER/POSTGRES_PASSWORD)
   - **Redis**: localhost:6379 (use REDIS_PASSWORD)

## Services

### Hypha Server

The main Hypha application server with full feature set enabled.

- **Image**: `ghcr.io/amun-ai/hypha:0.21.53`
- **Port**: 9527
- **Features Enabled**:
  - Server Apps
  - MCP (Model Context Protocol)
  - A2A (App-to-App communication)
  - S3 Storage with proxy
  - Database migrations
- **Data**: Stored in `./data/hypha`
- **Dependencies**: PostgreSQL, Redis, MinIO

### PostgreSQL Database

Primary database for Hypha's persistent data.

- **Image**: `postgres:15-alpine`
- **Port**: 5432
- **Database**: `hypha_db`
- **Data**: Stored in `./data/postgres`
- **Features**: 
  - Automatic health checks
  - Persistent storage
  - Configurable credentials

### Redis Cache

In-memory cache and message broker for scaling and performance.

- **Image**: `redis:7-alpine`
- **Port**: 6379
- **Data**: Stored in `./data/redis` (with persistence)
- **Features**:
  - Password protection
  - Append-only file (AOF) persistence
  - Health checks

### MinIO S3 Storage

S3-compatible object storage for artifacts and files.

- **Image**: `minio/minio:RELEASE.2025-03-12T18-04-18Z-cpuv1`
- **Ports**:
  - 9000: S3 API
  - 9001: Web Console
- **Data**: Stored in `./data/minio`
- **Features**:
  - Default bucket creation
  - Management console
  - Health monitoring

## Configuration

### Environment Variables

All configuration is managed through environment variables in the `.env` file:

#### Core Settings
- `HYPHA_JWT_SECRET`: JWT token secret (required, 32+ chars)
- `HYPHA_PUBLIC_BASE_URL`: Public URL for the server
- `HYPHA_LOGLEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

#### Database Settings
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: PostgreSQL credentials
- Database connection is automatically configured

#### Cache Settings
- `REDIS_PASSWORD`: Password for Redis authentication
- `HYPHA_RESET_REDIS`: Whether to reset Redis on startup

#### Storage Settings
- `HYPHA_ACCESS_KEY_ID`, `HYPHA_SECRET_ACCESS_KEY`: MinIO credentials
- `HYPHA_WORKSPACE_BUCKET`: Default bucket name
- `HYPHA_ENDPOINT_URL_PUBLIC`: Public URL for S3 access

### Data Persistence

All services use persistent storage with local directories:

```bash
./data/
├── hypha/      # Hypha server data and cache
├── postgres/   # PostgreSQL database files
├── redis/      # Redis persistent data
└── minio/      # MinIO object storage
```

You can customize these paths in the `.env` file:

```bash
HYPHA_DATA_DIR=/path/to/hypha/data
POSTGRES_DATA_DIR=/path/to/postgres/data
REDIS_DATA_DIR=/path/to/redis/data
MINIO_DATA_DIR=/path/to/minio/data
```

### Health Checks

All services include health checks with appropriate startup periods:
- **Hypha**: HTTP health endpoints
- **PostgreSQL**: `pg_isready` checks
- **Redis**: PING command
- **MinIO**: HTTP health endpoint

### Service Dependencies

Services start in the correct order:
1. PostgreSQL, Redis, MinIO (in parallel)
2. Hypha Server (after all dependencies are healthy)

## Advanced Configuration

### Feature Flags

Enable/disable features via environment variables:

```bash
HYPHA_ENABLE_SERVICE_SEARCH=true    # Service discovery
HYPHA_STARTUP_FUNCTIONS=...         # Custom startup functions
HYPHA_STATIC_MOUNTS=...            # Static file mounts
HYPHA_TRITON_SERVERS=...           # Triton inference servers
```

### Production Deployment

For production use:

1. **Security**: Change all default passwords
2. **Networking**: Use Docker secrets for sensitive data
3. **Storage**: Consider using named volumes or external storage
4. **Monitoring**: Add health monitoring and log aggregation
5. **Backup**: Implement regular database and storage backups

## Operations

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f hypha-server
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f minio
```

### Database Operations

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d hypha_db

# Database backup
docker-compose exec postgres pg_dump -U postgres hypha_db > backup.sql

# Database restore
docker-compose exec -T postgres psql -U postgres -d hypha_db < backup.sql
```

### Redis Operations

```bash
# Connect to Redis
docker-compose exec redis redis-cli -a your_redis_password

# Monitor Redis
docker-compose exec redis redis-cli -a your_redis_password monitor
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

### Upgrading

```bash
# Update images
docker-compose pull

# Restart with new images
docker-compose up -d

# Check updated services
docker-compose ps
```

## Troubleshooting

### Service Won't Start

1. Check logs: `docker-compose logs [service-name]`
2. Verify environment variables in `.env`
3. Ensure data directories exist and have correct permissions
4. Check port conflicts with `docker-compose ps`

### Connection Issues

1. Verify all services are healthy: `docker-compose ps`
2. Check network connectivity between services
3. Validate credentials in `.env` file
4. Review firewall and port forwarding settings

### Performance Issues

1. Monitor resource usage: `docker stats`
2. Adjust resource limits in docker-compose.yml
3. Optimize PostgreSQL and Redis configurations
4. Consider scaling with multiple replicas

## Development

This setup is ideal for:
- Local development and testing
- Feature development with full stack
- Integration testing
- Production-like environment simulation

For minimal development, you can disable services by commenting them out in docker-compose.yml.