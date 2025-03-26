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
   - `JWT_SECRET`: Change to a secure random string
   - `S3_ACCESS_KEY`: MinIO access key (default: minioadmin)
   - `S3_SECRET_KEY`: MinIO secret key (default: minioadmin)

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the services:
   - Hypha Server: http://localhost:9520
   - MinIO Console: http://localhost:9001 (login with S3_ACCESS_KEY/S3_SECRET_KEY)

## Services

### Hypha Server

The Hypha Server provides the backend API for the Hypha application.

- Image: `ghcr.io/amun-ai/hypha:0.20.47.post5`
- Port: 9520
- Data: Stored in the `hypha-data` volume

### MinIO

MinIO provides S3-compatible object storage for Hypha.

- Image: `minio/minio:RELEASE.2025-03-12T18-04-18Z-cpuv1`
- Ports:
  - 9000: S3 API
  - 9001: Web Console
- Data: Stored in the `minio-data` volume

## Configuration

The Hypha Server is configured to use MinIO as the S3 backend. The following environment variables are set:

- `S3_ACCESS_KEY`: The access key for MinIO
- `S3_SECRET_KEY`: The secret key for MinIO
- `S3_ENDPOINT_URL`: The URL of the MinIO server (http://minio:9000)
- `S3_ENDPOINT_URL_PUBLIC`: The publicly accessible URL of the MinIO server (http://localhost:9000)

## Stopping the Services

To stop the services, run:
```bash
docker-compose down
```

To stop the services and remove the volumes, run:
```bash
docker-compose down -v
```

## Upgrading

To upgrade to a newer version of Hypha Server, update the image tag in the `docker-compose.yml` file and run:
```bash
docker-compose pull
docker-compose up -d
``` 