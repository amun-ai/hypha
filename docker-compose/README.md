# Hypha Server with Docker Compose

This directory contains a Docker Compose setup for running Hypha Server with MinIO as the S3 storage backend.

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
   
   Open the `.env` file and modify the values as needed:
   - `HYPHA_JWT_SECRET`: Change to a secure random string
   - `HYPHA_ACCESS_KEY_ID`: MinIO access key (default: minioadmin)
   - `HYPHA_SECRET_ACCESS_KEY`: MinIO secret key (default: minioadmin)

3. Create data directories:
   ```bash
   mkdir -p ./data/hypha ./data/minio
   ```

4. Start the services:
   ```bash
   docker-compose up -d
   ```

5. Access the services:
   - Hypha Server: http://localhost:9520
   - MinIO Console: http://localhost:9001 (login with HYPHA_ACCESS_KEY_ID/HYPHA_SECRET_ACCESS_KEY)

## Services

### Hypha Server

The Hypha Server provides the backend API for the Hypha application.

- Image: `ghcr.io/amun-ai/hypha:0.20.54`
- Port: 9520
- Data: Stored in the local directory `./data/hypha`

### MinIO

MinIO provides S3-compatible object storage for Hypha.

- Image: `minio/minio:RELEASE.2025-03-12T18-04-18Z-cpuv1`
- Ports:
  - 9000: S3 API
  - 9001: Web Console
- Data: Stored in the local directory `./data/minio`

## Configuration

### Environment Variables

The Hypha Server is configured to use MinIO as the S3 backend. The following environment variables are set:

- `HYPHA_ACCESS_KEY_ID`: The access key for MinIO
- `HYPHA_SECRET_ACCESS_KEY`: The secret key for MinIO
- `ENDPOINT_URL`: The URL of the MinIO server (http://minio:9000)
- `HYPHA_ENDPOINT_URL_PUBLIC`: The publicly accessible URL of the MinIO server (http://localhost:9000)

### Data Storage

By default, this setup uses local directories for data storage:

- `HYPHA_DATA_DIR`: Directory where Hypha Server stores its data (default: `./data/hypha`)
- `MINIO_DATA_DIR`: Directory where MinIO stores its data (default: `./data/minio`)

You can customize these directories in the `.env` file:

```
HYPHA_DATA_DIR=/path/to/your/hypha/data
MINIO_DATA_DIR=/path/to/your/minio/data
```

Using local directories makes it easier to:
- Access your data directly from the host system
- Back up your data using standard file system tools
- Share data between different container setups
- Mount network storage or external drives

## Stopping the Services

To stop the services, run:
```bash
docker-compose down
```

## Upgrading

To upgrade to a newer version of Hypha Server, update the image tag in the `docker-compose.yml` file and run:
```bash
docker-compose pull
docker-compose up -d
``` 