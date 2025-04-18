"""Test the hypha module."""

import uuid
import time
import asyncio

SIO_PORT = 38283
SIO_PORT2 = 38223
SIO_PORT_REDIS_1 = SIO_PORT + 8
SIO_PORT_REDIS_2 = SIO_PORT + 9
SIO_PORT_SQLITE = SIO_PORT + 20
TRITON_PORT = 38284
WS_SERVER_URL = f"ws://127.0.0.1:{SIO_PORT}/ws"
SERVER_URL = f"http://127.0.0.1:{SIO_PORT}"
SERVER_URL_REDIS_1 = f"http://127.0.0.1:{SIO_PORT_REDIS_1}"
SERVER_URL_REDIS_2 = f"http://127.0.0.1:{SIO_PORT_REDIS_2}"
SERVER_URL_SQLITE = f"http://127.0.0.1:{SIO_PORT_SQLITE}"

MINIO_PORT = 38483
MINIO_SERVER_URL = f"http://127.0.0.1:{MINIO_PORT}"
MINIO_SERVER_URL_PUBLIC = f"http://localhost:{MINIO_PORT}"
MINIO_ROOT_USER = "minio"
MINIO_ROOT_PASSWORD = str(uuid.uuid4())
REDIS_PORT = 6338

POSTGRES_PORT = 5432
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "mysecretpassword"
POSTGRES_DB = "postgres"

POSTGRES_URI = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"


def find_item(items, key, value):
    """Find an item with key or attributes in an object list."""
    filtered = [
        item
        for item in items
        if (item[key] if isinstance(item, dict) else getattr(item, key)) == value
    ]
    if len(filtered) == 0:
        return None

    return filtered[0]


async def wait_for_workspace_ready(api, timeout=30):
    """Wait for workspace to be ready."""
    start_time = time.time()
    while True:
        status = await api.check_status()
        if status["status"] == "ready":
            break
        if time.time() - start_time > timeout:
            raise TimeoutError("Workspace failed to become ready")
        await asyncio.sleep(0.1)
    return status
