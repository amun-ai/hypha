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
import secrets

# Set JWT_SECRET environment variables BEFORE importing any hypha modules
# This ensures all modules use the same JWT_SECRET
JWT_SECRET = str(uuid.uuid4())
os.environ["JWT_SECRET"] = JWT_SECRET
os.environ["HYPHA_JWT_SECRET"] = JWT_SECRET  # Also set HYPHA_JWT_SECRET to ensure consistency

import requests
from requests import RequestException
import pytest
import pytest_asyncio
from sqlalchemy import create_engine, text

from hypha.core import UserInfo, auth, UserPermission
from hypha.core.auth import generate_auth_token, create_scope
from hypha.minio import setup_minio_executables, start_minio_server
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
os.environ["ACTIVITY_CHECK_INTERVAL"] = "0.3"
test_env = os.environ.copy()


def _wait_for_server_health(server_url, server_name, max_timeout=40):
    """Helper function to wait for server health checks with robust retry logic."""
    timeout = max_timeout
    check_interval = 0.1
    backoff_factor = 1.1
    max_interval = 1.0
    
    while timeout > 0:
        try:
            # Check readiness
            response = requests.get(
                f"{server_url}/health/readiness",
                timeout=5
            )
            if response.ok:
                # Additional verification: check liveness
                liveness_response = requests.get(
                    f"{server_url}/health/liveness",
                    timeout=5
                )
                if liveness_response.ok:
                    return True
        except (RequestException, Exception) as e:
            # Log the exception for debugging in the second half of timeout period
            if timeout < max_timeout * 0.5:
                print(f"Server ({server_name}) not ready: {e}")
        
        time.sleep(check_interval)
        timeout -= check_interval
        # Exponential backoff for check interval
        check_interval = min(check_interval * backoff_factor, max_interval)
    
    return False


def _cleanup_server_process(proc, server_name):
    """Helper function to cleanup server processes robustly."""
    try:
        proc.terminate()
        # Give the process a chance to terminate gracefully
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as e:
        print(f"Error during {server_name} cleanup: {e}")
        try:
            proc.kill()
        except Exception:
            pass


@pytest_asyncio.fixture
def event_loop():
    """Create an event loop for each test."""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Event loop is closed")
    except RuntimeError:
        # Create a new event loop if none exists or is closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    yield loop
    
    # Clean up the loop after the test
    try:
        if not loop.is_closed():
            # Cancel all pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus):
    """Clean up after the tests."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        # No event loop in the current thread, which is fine
        pass


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
    # Since generate_auth_token is now async, we need to run it in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        token = loop.run_until_complete(generate_auth_token(root_user_info, 18000))
        yield token
    finally:
        loop.close()


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
    # Since generate_auth_token is now async, we need to run it in an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        token = loop.run_until_complete(generate_auth_token(root_user_info, 1800))
        yield token
    finally:
        loop.close()


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


@pytest_asyncio.fixture(name="test_user_token_5", scope="session")
def generate_authenticated_user_5():
    """Generate a test user token."""
    yield from _generate_token("user-5", [])


@pytest_asyncio.fixture(name="test_user_token_6", scope="session")
def generate_authenticated_user_6():
    """Generate a test user token."""
    yield from _generate_token("user-6", [])


@pytest_asyncio.fixture(name="test_user_token_7", scope="session")
def generate_authenticated_user_7():
    """Generate a test user token."""
    yield from _generate_token("user-7", [])


@pytest_asyncio.fixture(name="test_user_token_8", scope="session")
def generate_authenticated_user_8():
    """Generate a test user token."""
    yield from _generate_token("user-8", [])


@pytest_asyncio.fixture(name="test_user_token_9", scope="session")
def generate_authenticated_user_9():
    """Generate a test user token."""
    yield from _generate_token("user-9", [])


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

            # Start a new PostgreSQL container with pgvector
            print("Starting a new PostgreSQL container with pgvector")
            result = subprocess.run(
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
                    "oeway/postgresql-pgvector:17.6.0-pgvector-0.8.1",
                ],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Started container: {result.stdout.strip()}")
            time.sleep(8)  # Give more time for pgvector container to initialize
        else:
            print("Using existing PostgreSQL container:", existing_container)
    else:
        # Check if the PostgreSQL pgvector image exists locally
        image_exists = subprocess.run(
            ["docker", "images", "-q", "oeway/postgresql-pgvector:17.6.0-pgvector-0.8.1"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        if not image_exists:
            # Pull the PostgreSQL pgvector image if it does not exist locally
            print("Pulling PostgreSQL pgvector Docker image...")
            subprocess.run(["docker", "pull", "oeway/postgresql-pgvector:17.6.0-pgvector-0.8.1"], check=True)
        else:
            print("PostgreSQL pgvector Docker image already exists locally.")

        # Start a new PostgreSQL container with pgvector
        print("Starting a new PostgreSQL container with pgvector")
        result = subprocess.run(
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
                "oeway/postgresql-pgvector:17.6.0-pgvector-0.8.1",
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Started container: {result.stdout.strip()}")
        time.sleep(8)  # Give more time for pgvector container to initialize

    # Wait for the PostgreSQL server to become available
    timeout = 30  # Increase timeout for pgvector container
    while timeout > 0:
        try:
            # First check if container is still running
            container_check = subprocess.run(
                ["docker", "ps", "-q", "-f", "name=hypha-postgres"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            
            if not container_check:
                print("Container stopped unexpectedly. Checking logs...")
                logs = subprocess.run(
                    ["docker", "logs", "hypha-postgres"],
                    capture_output=True,
                    text=True,
                )
                print(f"Container logs: {logs.stderr}")
                raise Exception("PostgreSQL container failed to start")
            
            engine = create_engine(
                f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@127.0.0.1:{POSTGRES_PORT}/{POSTGRES_DB}"
            )
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                # Also create pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("PostgreSQL with pgvector is ready")
            break
        except Exception as e:
            if timeout > 1:
                timeout -= 1
                time.sleep(1)
            else:
                print(f"Failed to connect to PostgreSQL: {e}")
                # Try to get container logs for debugging
                logs = subprocess.run(
                    ["docker", "logs", "hypha-postgres"],
                    capture_output=True,
                    text=True,
                )
                print(f"Container logs: {logs.stderr}")
                raise Exception(f"PostgreSQL server did not become available: {e}")

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
        yield f"redis://127.0.0.1:{REDIS_PORT}/0"
    except Exception:
        # Pull the Redis image
        subprocess.run(["docker", "pull", "redis/redis-stack:7.2.0-v13"], check=True)
        # user docker to start redis
        subprocess.Popen(
            [
                "docker",
                "run",
                "-d",
                "-p",
                f"{REDIS_PORT}:6379",
                "redis/redis-stack:7.2.0-v13",
            ]
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
        yield f"redis://127.0.0.1:{REDIS_PORT}/0"
        subprocess.Popen(["docker", "stop", "redis"])
        subprocess.Popen(["docker", "rm", "redis"])
        time.sleep(1)


@pytest_asyncio.fixture(name="fastapi_server", scope="session")
def fastapi_server_fixture(minio_server, postgres_server):
    """Start server as test fixture and tear down after test."""
    # Generate a strong root token for testing
    ROOT_TOKEN = secrets.token_urlsafe(32)
    os.environ["HYPHA_ROOT_TOKEN"] = ROOT_TOKEN  # Store for tests to use
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT}",
            "--enable-server-apps",
            "--enable-s3",
            "--enable-a2a",
            "--enable-mcp",
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
            "--cache-dir=./bin/cache",
            "--executable-path=./bin",
            f"--triton-servers=http://127.0.0.1:{TRITON_PORT}",
            "--static-mounts=/tests:./tests",
            "--s3-cleanup-period=2",
            f"--root-token={ROOT_TOKEN}",
            "--startup-functions",
            "hypha.utils:_example_hypha_startup",
            "./tests/example-startup-function.py:hypha_startup",
            "hypha.workers.python_eval:hypha_startup",
            "hypha.workers.conda:hypha_startup",
            "hypha.workers.terminal:hypha_startup",
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
            "--enable-s3-proxy",
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
            "--enable-server-apps",
            "--enable-s3",
            # need to define it so the two server can communicate
            f"--public-base-url=http://my-public-url.com",
            "--server-id=server-0",
            f"--redis-uri=redis://127.0.0.1:{REDIS_PORT}/0",
            "--reset-redis",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
            f"--endpoint-url-public={MINIO_SERVER_URL_PUBLIC}",
            "--workspace-bucket=my-workspaces",
            "--s3-admin-type=minio",
            "--enable-service-search",
        ],
        env=test_env,
        # Ensure the process has its own process group
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    ) as proc:
        # Wait for server to be ready
        server_url = f"http://127.0.0.1:{SIO_PORT_REDIS_1}"
        if not _wait_for_server_health(server_url, "fastapi_server_redis_1"):
            # Try to get more information about why the server failed to start
            try:
                if proc.poll() is not None:
                    print(f"Server process exited with code: {proc.returncode}")
                else:
                    print("Server process is still running but not responding to health checks")
            except Exception:
                pass
            raise TimeoutError("Server (fastapi_server_redis_1) did not start in time")
        
        yield
        _cleanup_server_process(proc, "fastapi_server_redis_1")


@pytest_asyncio.fixture(name="fastapi_server_redis_2", scope="session")
def fastapi_server_redis_2(redis_server, minio_server, fastapi_server):
    """Start a backup server as test fixture and tear down after test."""
    # Add a small delay to ensure the first server is fully initialized
    time.sleep(1)
    
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
        # Ensure the process has its own process group
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    ) as proc:
        # Wait for server to be ready
        server_url = f"http://127.0.0.1:{SIO_PORT_REDIS_2}"
        if not _wait_for_server_health(server_url, "fastapi_server_redis_2"):
            # Try to get more information about why the server failed to start
            try:
                if proc.poll() is not None:
                    print(f"Server process exited with code: {proc.returncode}")
                else:
                    print("Server process is still running but not responding to health checks")
            except Exception:
                pass
            raise TimeoutError("Server (fastapi_server_redis_2) did not start in time")
        
        yield
        _cleanup_server_process(proc, "fastapi_server_redis_2")


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
    # Use start_minio_server to manage the Minio instance
    proc, server_url, workdir = start_minio_server(
        executable_path="./bin",  # Specify where executables are/should be downloaded
        workdir=tempfile.mkdtemp(),  # Create a temporary directory for this session
        port=MINIO_PORT,
        console_port=MINIO_PORT + 1,  # Keep console port logic if needed
        root_user=MINIO_ROOT_USER,
        root_password=MINIO_ROOT_PASSWORD,
        timeout=10,  # Keep timeout
    )

    if not proc:
        raise RuntimeError(f"Failed to start Minio server at {MINIO_SERVER_URL}")

    print(f"Minio server started successfully at {server_url}")
    print(f"Minio data directory: {workdir}")

    yield server_url  # Yield the URL provided by the function

    print("Stopping Minio server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)  # Wait for graceful termination
        print("Minio server stopped.")
    except subprocess.TimeoutExpired:
        print("Minio server did not terminate gracefully, killing...")
        proc.kill()  # Force kill if necessary
    finally:
        # Clean up any lingering MinIO Docker containers
        try:
            print("Cleaning up MinIO Docker containers...")
            
            # Stop any running MinIO containers
            result = subprocess.run([
                "docker", "ps", "-q", "--filter", "ancestor=minio/minio"
            ], capture_output=True, text=True, timeout=10)
            
            if result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                print(f"Found {len(container_ids)} running MinIO containers, stopping them...")
                subprocess.run([
                    "docker", "stop"
                ] + container_ids, capture_output=True, text=True, timeout=30)
                
                # Remove the stopped containers
                subprocess.run([
                    "docker", "rm"
                ] + container_ids, capture_output=True, text=True, timeout=10)
            
            # Also clean up any containers with "minio" in the name
            result = subprocess.run([
                "docker", "ps", "-aq", "--filter", "name=minio"
            ], capture_output=True, text=True, timeout=10)
            
            if result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                print(f"Found {len(container_ids)} MinIO containers by name, cleaning them...")
                subprocess.run([
                    "docker", "stop"
                ] + container_ids, capture_output=True, text=True, timeout=30)
                subprocess.run([
                    "docker", "rm"
                ] + container_ids, capture_output=True, text=True, timeout=10)
            
            print("MinIO Docker container cleanup completed.")
        except subprocess.TimeoutExpired:
            print("Warning: MinIO container cleanup timed out")
        except FileNotFoundError:
            print("Docker not found - skipping MinIO container cleanup")
        except Exception as e:
            print(f"Warning: Error during MinIO container cleanup: {e}")
        
        # Clean up the temporary directory
        if workdir and os.path.exists(workdir):
            print(f"Cleaning up Minio data directory: {workdir}")
            try:
                shutil.rmtree(workdir, ignore_errors=True)
            except Exception as e:
                print(f"Error removing Minio workdir {workdir}: {e}")


@pytest_asyncio.fixture(name="conda_available", scope="session")
def conda_available_fixture():
    """Check if conda or mamba is available and skip tests if not."""
    import shutil

    # Check for mamba first (preferred)
    mamba_path = shutil.which("mamba")
    if mamba_path is not None:
        try:
            result = subprocess.run(
                ["mamba", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"Using mamba for conda tests: {result.stdout.strip()}")
                yield mamba_path
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Fall back to conda
    conda_path = shutil.which("conda")
    if conda_path is None:
        pytest.skip(
            "Neither mamba nor conda available - skipping conda integration tests"
        )

    # Check if we can run conda commands
    try:
        result = subprocess.run(
            ["conda", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            pytest.skip("Conda command failed - skipping conda integration tests")
        print(f"Using conda for conda tests: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Conda command not accessible - skipping conda integration tests")

    yield conda_path


@pytest_asyncio.fixture(name="conda_test_workspace")
def conda_test_workspace_fixture(conda_available):
    """Provide a clean workspace for conda integration tests."""
    # Create a temporary directory for conda test environments
    test_workspace = tempfile.mkdtemp(prefix="hypha_conda_test_")

    # Create subdirectories
    cache_dir = os.path.join(test_workspace, "cache")
    envs_dir = os.path.join(test_workspace, "envs")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(envs_dir, exist_ok=True)

    workspace_info = {
        "workspace_dir": test_workspace,
        "cache_dir": cache_dir,
        "envs_dir": envs_dir,
    }

    yield workspace_info

    # Cleanup - remove all test environments and cache
    print(f"Cleaning up conda test workspace: {test_workspace}")
    try:
        shutil.rmtree(test_workspace, ignore_errors=True)
    except Exception as e:
        print(f"Error cleaning up conda test workspace: {e}")


@pytest_asyncio.fixture(name="conda_integration_server")
def conda_integration_server_fixture():
    """Provide a mock server for conda integration tests."""

    class MockIntegrationServer:
        def __init__(self):
            self.registered_services = []

        async def register_service(self, service_config):
            self.registered_services.append(service_config)
            return {"id": service_config.get("id", "test-service")}

    yield MockIntegrationServer()


@pytest_asyncio.fixture(name="custom_auth_server")
def custom_auth_server_fixture():
    """Start a server with custom authentication for testing."""
    # Use a different port for the custom auth server
    CUSTOM_AUTH_PORT = 38999
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={CUSTOM_AUTH_PORT}",
            "--startup-functions",
            "./tests/custom_auth_startup_test.py:hypha_startup",
        ],
        env=test_env,
    ) as proc:
        timeout = 20
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{CUSTOM_AUTH_PORT}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Custom auth server did not start in time")
        
        response = requests.get(f"http://127.0.0.1:{CUSTOM_AUTH_PORT}/health/liveness")
        assert response.ok
        
        yield f"http://127.0.0.1:{CUSTOM_AUTH_PORT}"
        
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest_asyncio.fixture(name="local_auth_server")
def local_auth_server_fixture():
    """Start a server with local authentication for testing."""
    # Use a different port for the local auth server
    LOCAL_AUTH_PORT = 39000
    
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={LOCAL_AUTH_PORT}",
            "--enable-local-auth",
            "--enable-s3",
            "--reset-redis",
            "--start-minio-server",
            "--minio-root-user=minioadmin",
            "--minio-root-password=minioadmin",
        ],
        env=test_env,
    ) as proc:
        timeout = 30
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{LOCAL_AUTH_PORT}/health/readiness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        if timeout <= 0:
            raise TimeoutError("Local auth server did not start in time")
        
        response = requests.get(f"http://127.0.0.1:{LOCAL_AUTH_PORT}/health/liveness")
        assert response.ok
        
        yield f"http://127.0.0.1:{LOCAL_AUTH_PORT}"
        
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
