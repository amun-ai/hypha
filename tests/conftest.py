"""Provide common pytest fixtures."""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest
import requests
from requests import RequestException

from hypha.minio import setup_minio_executables

from . import (
    MINIO_PORT,
    MINIO_ROOT_PASSWORD,
    MINIO_ROOT_USER,
    MINIO_SERVER_URL,
    SIO_PORT,
    SIO_PORT2,
)


@pytest.fixture(name="socketio_server", scope="session")
def socketio_server_fixture(minio_server):
    """Start server as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT}",
            "--enable-server-apps",
            "--enable-s3",
            f"--endpoint-url={MINIO_SERVER_URL}",
            f"--access-key-id={MINIO_ROOT_USER}",
            f"--secret-access-key={MINIO_ROOT_PASSWORD}",
        ]
    ) as proc:

        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(f"http://127.0.0.1:{SIO_PORT}/liveness")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield
        proc.kill()
        proc.terminate()


@pytest.fixture(name="socketio_subpath_server")
def socketio_subpath_server_fixture(minio_server):
    """Start server (under /my/engine) as test fixture and tear down after test."""
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hypha.server",
            f"--port={SIO_PORT2}",
            "--base-path=/my/engine",
        ]
    ) as proc:

        timeout = 10
        while timeout > 0:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{SIO_PORT2}/my/engine/liveness"
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


@pytest.fixture(name="minio_server", scope="session")
def minio_server_fixture():
    """Start minio server as test fixture and tear down after test."""
    setup_minio_executables()
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
        while timeout > 0:
            try:
                response = requests.get(f"{MINIO_SERVER_URL}/minio/health/live")
                if response.ok:
                    break
            except RequestException:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield

        proc.terminate()
        shutil.rmtree(dirpath)
