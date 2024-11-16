"""Provide common pytest fixtures."""
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from threading import Thread

import pytest_asyncio
import requests
from requests import RequestException
import pytest_asyncio
from sqlalchemy import create_engine

from hypha.core import UserInfo, auth, UserPermission
from hypha.core.auth import generate_presigned_token, create_scope
from hypha.minio import setup_minio_executables
from redis import Redis

from . import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
    MINIO_SERVER_URL_PUBLIC,
    REDIS_PORT,
    SIO_PORT,
    SIO_PORT2,
    SIO_PORT_REDIS_1,
    SIO_PORT_REDIS_2,
    SIO_PORT_SQLITE,
    TRITON_PORT,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
    POSTGRES_URI,
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


def _generate_token(id, roles):
    # Patch the JWT_SECRET
    auth.JWT_SECRET = JWT_SECRET

    root_user_info = UserInfo(
        id=id,
        is_anonymous=False,
        email=id + "@test.com",
        parent=None,
        roles=roles,
        scope=create_scope(workspaces={f"ws-user-{id}": UserPermission.admin}),
        expires_at=None,
    )
    token = generate_presigned_token(root_user_info, 18000)
    yield token


@pytest_asyncio.fixture(name="root_user_token", scope="session")
def generate_root_user_token():
    """Generate a root user token."""
    auth.JWT_SECRET = JWT_SECRET
    root_user_info = UserInfo(
        id="root",
        is_anonymous=False,
        email="root@test.com",
        parent=None,
        roles=[],
        scope=create_scope(workspaces={"*": UserPermission.admin}),
        expires_at=None,
    )
    token = generate_presigned_token(root_user_info, 1800)
    yield token


@pytest_asyncio.fixture(name="test_user_token", scope="session")
def generate_authenticated_user():
    """Generate a test user token."""
    yield from _generate_token("user-1", [])


@pytest_asyncio.fixture(name="test_user_token_2", scope="session")
def generate_authenticated_user_2():
    """Generate a test user token."""
    yield from _generate_token("user-2", [])


@pytest_asyncio.fixture(name="test_user_token_temporary", scope="session")
def generate_authenticated_user_temporary():
    """Generate a temporary test user token."""
    yield from _generate_token("test-user", ["temporary-test-user"])


@pytest_asyncio.fixture(name="triton_server", scope="session")
def triton_server():
    """Start a triton server as test fixture and tear down after test."""
    from .start_pytriton_server import start_triton_server

    thread = Thread(
        target=start_triton_server, args=(TRITON_PORT, TRITON_PORT + 1), daemon=True
    )
    thread.start()
    timeout = 10
    while timeout > 0:
        try:
            response = requests.get(f"http://127.0.0.1:{TRITON_PORT}/v2/health/live")
            if response.ok:
                break
        except RequestException:
            pass
        timeout -= 0.1
        time.sleep(0.1)
    yield


@pytest_asyncio.fixture(name="postgres_server", scope="session")
def postgres_server():
    """Start a fresh PostgreSQL server as a test fixture and tear down after the test."""
    # Check if the container is already running
    existing_container = subprocess.run(
        ["docker", "ps", "-q", "-f", "name=hypha-postgres"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Stop and remove the existing container if it is running
    if existing_container:
        if os.environ.get("GITHUB_ACTIONS") != "true":
            print(
                "Stopping and removing existing PostgreSQL container:",
                existing_container,
            )
            subprocess.run(["docker", "stop", "hypha-postgres"], check=True)
            # Attempt to remove the container, but ignore errors if it does not exist
            subprocess.run(["docker", "rm", "hypha-postgres"], check=False)

            # Start a new PostgreSQL container
            print("Starting a new PostgreSQL container")
            subprocess.Popen(
                [
                    "docker",
                    "run",
                    "--name",
                    "hypha-postgres",
                    "--rm",
                    "-e",
                    f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
                    "-p",
                    f"{POSTGRES_PORT}:5432",
                    "-d",
                    "postgres",
                ]
            )
            time.sleep(2)  # Give the container time to initialize
        else:
            print("Using existing PostgreSQL container:", existing_container)
    else:
        # Start a new PostgreSQL container
        print("Starting a new PostgreSQL container")
        subprocess.Popen(
            [
                "docker",
                "run",
                "--name",
                "hypha-postgres",
                "--rm",
                "-e",
                f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
                "-p",
                f"{POSTGRES_PORT}:5432",
                "-d",
                "postgres",
            ]
        )
        time.sleep(2)

    # Wait for the PostgreSQL server to become available
    timeout = 10
    while timeout > 0:
        try:
            engine = create_engine(
                f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
            )
            with engine.connect() as connection:
                connection.execute("SELECT 1")
            break
        except Exception:
            timeout -= 1
            time.sleep(1)

    yield  # Test code executes here

    if os.environ.get("GITHUB_ACTIONS") != "true":
        # Stop the PostgreSQL container
        print("Stopping PostgreSQL container")
        subprocess.run(["docker", "stop", "hypha-postgres"], check=True)


@pytest_asyncio.fixture(name="redis_server", scope="session")
def redis_server():
    """Start a redis server as test fixture and tear down after test."""
    try:
        r = Redis(host="localhost", port=REDIS_PORT)
        r.ping()
        yield
    except Exception:
        # user docker to start redis
        subprocess.Popen(
            ["docker", "run", "-d", "-p", f"{REDIS_PORT}:6379", "redis:6.2.5"]
        )
        timeout = 10
        while timeout > 0:
            try:
                r = Redis(host="localhost", port=REDIS_PORT)
                r.ping()
                break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield
        subprocess.Popen(["docker", "stop", "redis"])
        subprocess.Popen(["docker", "rm", "redis"])
        time.sleep(1)


@pytest_asyncio.fixture(name="fastapi_server", scope="session")
def fastapi_server_fixture(minio_server, postgres_server):
    """Start server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT}",
            "--enable-server-apps",
            "--enable-s3",
            f"--database-uri={POSTGRES_URI}",
            "--migrate-database=head",
            "--reset-redis",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            "--enable-s3-proxy",
            f"--workspace-bucket=my-workspaces",
            "--s3-admin-type=minio",
            f"--triton-servers=http://127.0.0.1:{TRITON_PORT}",
            "--static-mounts=/tests:./tests",
            "--startup-functions",
            "hypha.utils:_example_hypha_startup",
            "./tests/example-startup-function.py:hypha_startup",
        ],
        env=test_env,
    ) as proc:
        timeout = 20
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{SIO_PORT}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Server (fastapi_server) did not start in time")
        response = requests.get(f"http://127.0.0.1:{SIO_PORT}/health/liveness")
        assert response.ok
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="fastapi_server_sqlite", scope="session")
def fastapi_server_sqlite_fixture(minio_server):
    """Start server as test fixture and tear down after test."""
    # create a temporary directory for the artifacts database
    dirpath = tempfile.mkdtemp()
    print(f"Artifacts database directory: {dirpath}")
    db_path = f"sqlite+aiosqlite:///{dirpath}/artifacts.db"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT_SQLITE}",
            "--enable-server-apps",
            "--enable-s3",
            f"--database-uri={db_path}",
            "--migrate-database=head",
            "--reset-redis",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            f"--workspace-bucket=my-workspaces",
            "--s3-admin-type=generic",
        ],
        env=test_env,
    ) as proc:
        timeout = 20
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT_SQLITE}/health/readiness"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Server (fastapi_server) did not start in time")
        response = requests.get(f"http://127.0.0.1:{SIO_PORT_SQLITE}/health/liveness")
        assert response.ok
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="fastapi_server_redis_1", scope="session")
def fastapi_server_redis_1(redis_server, minio_server):
    """Start server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT_REDIS_1}",
            # "--enable-server-apps",
            # "--enable-s3",
            # need to define it so the two server can communicate
            f"--public-base-url=http://my-public-url.com",
            "--server-id=server-0",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            "--reset-redis",
            # f"--endpoint-url={MINIO_SERVER_URL}",
            # f"--access-key-id={MINIO_ROOT_USER}",
            # f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            # f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
        ],
        env=test_env,
    ) as proc:
        timeout = 20
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT_REDIS_1}/health/readiness"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Server (fastapi_server_redis_1) did not start in time")
        yield
        proc.kill()
        proc.terminate()


@pytest_asyncio.fixture(name="fastapi_server_redis_2", scope="session")
def fastapi_server_redis_2(redis_server, minio_server, fastapi_server):
    """Start a backup server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT_REDIS_2}",
            # "--enable-server-apps",
            # "--enable-s3",
            # need to define it so the two server can communicate
            f"--public-base-url=http://my-public-url.com",
            "--server-id=server-1",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            # f"--endpoint-url={MINIO_SERVER_URL}",
            # f"--access-key-id={MINIO_ROOT_USER}",
            # f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            # f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
        ],
        env=test_env,
    ) as proc:
        timeout = 20
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT_REDIS_2}/health/readiness"
                )
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Server (fastapi_server_redis_2) did not start in time")
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
                    f"http://127.0.0.1:{SIO_PORT2}/my/engine/health/readiness"
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
    print(f"Minio data directory: {dirpath}")
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
        # shutil.rmtree(dirpath, ignore_errors=True)
