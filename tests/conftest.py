"""Provide common pytest fixtures."""
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import requests
from requests import RequestException
import pytest_asyncio

from hypha.core import TokenConfig, UserInfo, auth
from hypha.core.auth import generate_presigned_token
from hypha.minio import setup_minio_executables

from . import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
    MINIO_SERVER_URL_PUBLIC,
    SIO_PORT,
    SIO_PORT2,
    REDIS_PORT,
    BACKUP_SIO_PORT,
    TRITON_SERVERS,
)

JWT_SECRET = str(uuid.uuid4())
os.environ["JWT_SECRET"] = JWT_SECRET
test_env = os.environ.copy()


@pytest_asyncio.fixture
def event_loop():
    """Create an event loop for each test."""
    yield asyncio.get_event_loop()


def pytest_sessionfinish(session, exitstatus):
    """Clean up after the tests."""
    asyncio.get_event_loop().close()


@pytest_asyncio.fixture(name="test_user_token", scope="session")
def generate_authenticated_user():
    """Generate a test user token."""
    # Patch the JWT_SECRET
    auth.JWT_SECRET = JWT_SECRET

    user_info = UserInfo(
        id="root",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scopes=[],
        expires_at=None,
    )
    config = {}
    config["scopes"] = []
    token_config = TokenConfig.parse_obj(config)
    token = generate_presigned_token(user_info, token_config)
    yield token


@pytest_asyncio.fixture(name="fastapi_server", scope="session")
def fastapi_server_fixture(minio_server):
    """Start server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT}",
            "--enable-server-apps",
            "--enable-s3",
            "--redis-uri=/tmp/redis.db",
            "--reset-redis",
            f"--redis-port={REDIS_PORT}",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            f"--triton-servers={TRITON_SERVERS}",
        ],
        env=test_env,
    ) as proc:

        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{SIO_PORT}/health/liveness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="fastapi_server_backup", scope="session")
def fastapi_server_backup_fixture(minio_server, fastapi_server):
    """Start a backup server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={BACKUP_SIO_PORT}",
            "--enable-server-apps",
            "--enable-s3",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            f"--triton-servers={TRITON_SERVERS}",
        ],
        env=test_env,
    ) as proc:

        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{BACKUP_SIO_PORT}/health/liveness"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="fastapi_subpath_server")
def fastapi_subpath_server_fixture(minio_server):
    """Start server (under /my/engine) as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT2}",
            "--base-path=/my/engine",
        ],
        env=test_env,
    ) as proc:

        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT2}/my/engine/health/liveness"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="minio_server", scope="session")
def minio_server_fixture():
    """Start minio server as test fixture and tear down after test."""
    setup_minio_executables("./bin")
    dirpath = tempfile.mkdtemp()
    my_env = os.environ.copy()
    my_env["MINIO_ROOT_USER"] = MINIO_ROOT_USER
    my_env["MINIO_ROOT_PASSWORD"] = MINIO_ROOT_PASSWORD
    with subprocess.Popen(
        [
            "./bin/minio",
            "server",
            f"--address=:{MINIO_PORT}",
            f"--console-address=:{MINIO_PORT+1}",
            f"{dirpath}",
        ],
        env=my_env,
    ) as proc:

        timeout = 10
        print(
            "Trying to connect to the minio server...",
            f"{MINIO_SERVER_URL}/minio/health/live",
        )
        while timeout > 0:
            try:
                print(".", end="")
                response = requests.get(f"{MINIO_SERVER_URL}/minio/health/live")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        print("\nMinio server started.")
        yield

        proc.terminate()
        shutil.rmtree(dirpath, ignore_errors=True)
