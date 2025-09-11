"""Test the hypha module."""

import uuid
import time
import asyncio
import os

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
# Use a fixed password or get from environment to avoid multiple UUID generations
MINIO_ROOT_PASSWORD = os.environ.get(
    "TEST_MINIO_ROOT_PASSWORD", "test-minio-password-123"
)
REDIS_PORT = 6338

POSTGRES_PORT = 15432
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "mysecretpassword"
POSTGRES_DB = "postgres"

POSTGRES_URI = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@127.0.0.1:{POSTGRES_PORT}/{POSTGRES_DB}"


def find_item(items, key_or_predicate, value=None):
    """Find an item with key or attributes in an object list, or using a predicate function."""
    if callable(key_or_predicate):
        # If it's a function, use it as a predicate
        filtered = [item for item in items if key_or_predicate(item)]
    else:
        # Original behavior with key and value
        key = key_or_predicate
        filtered = [
            item
            for item in items
            if (item[key] if isinstance(item, dict) else getattr(item, key)) == value
        ]
    if len(filtered) == 0:
        return None

    return filtered[0]


async def wait_for_workspace_ready(api, timeout=30):
    """Wait for workspace to be ready without errors."""
    start_time = time.time()
    while True:
        status = await api.check_status()
        if status["status"] == "ready":
            # Check if there are any errors
            if status.get("errors"):
                # Workspace has errors, keep waiting or fail
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Workspace ready but with errors: {status['errors']}")
                # Continue waiting to see if errors get resolved
                await asyncio.sleep(0.5)
                continue
            # Ready without errors
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Workspace failed to become ready, current status: {status}")
        await asyncio.sleep(0.1)
    return status
