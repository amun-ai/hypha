"""Test the hypha module."""
import uuid

SIO_PORT = 38283
SIO_PORT2 = 38223
BACKUP_SIO_PORT = SIO_PORT + 1
WS_SERVER_URL = f"ws://127.0.0.1:{SIO_PORT}/ws"
SERVER_URL = f"http://127.0.0.1:{SIO_PORT}"
BACKUP_WS_SERVER_URL = f"ws://127.0.0.1:{BACKUP_SIO_PORT}/ws"
BACKUP_SERVER_URL = f"http://127.0.0.1:{BACKUP_SIO_PORT}"

MINIO_PORT = 38483
MINIO_SERVER_URL = f"http://127.0.0.1:{MINIO_PORT}"
MINIO_SERVER_URL_PUBLIC = f"http://localhost:{MINIO_PORT}"
MINIO_ROOT_USER = "minio"
MINIO_ROOT_PASSWORD = str(uuid.uuid4())
REDIS_PORT = 6383

TRITON_SERVERS = "https://ai.imjoy.io/triton"


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
