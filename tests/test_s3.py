"""Test S3 services."""

import os
import subprocess
import sys
import tempfile
import time

import asyncio
import aioboto3
import pytest
import io
import zipfile
import requests
import shutil
from hypha_rpc import connect_to_server
from hypha.core.auth import (
    generate_presigned_token,
    create_scope,
    UserInfo,
    UserPermission,
)

from . import WS_SERVER_URL, SERVER_URL, SIO_PORT_SQLITE, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_s3_proxy(minio_server, fastapi_server_sqlite, test_user_token):
    """Test s3 proxy."""
    api = await connect_to_server(
        {
            "name": "anonymous client",
            "server_url": f"ws://127.0.0.1:{SIO_PORT_SQLITE}/ws",
            "token": test_user_token,
        }
    )
    # Generate credentials and get the s3controller
    s3controller = await api.get_service("public/s3-storage")

    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("file1.txt", "This is the content of file 1.")
        zf.writestr("file2.txt", "This is the content of file 2.")
        zf.writestr(
            "subdir/file3.txt", "This is the content of file 3 inside a subdirectory."
        )
    zip_buffer.seek(0)

    presigned_url = await s3controller.generate_presigned_url(
        f"apps/test.zip", client_method="put_object"
    )
    # Ensure we're testing the S3 proxy, not direct MinIO access
    assert "/s3/" in presigned_url, "Presigned URL should be a proxy URL"
    # Use requests to upload the ZIP file to the presigned URL
    response = requests.put(
        presigned_url,
        data=zip_buffer.read(),
        headers={"Content-Type": "application/zip"},
    )
    assert response.status_code == 200, "Failed to upload the ZIP file to S3."

    # Test retrieving the first file inside the ZIP
    presigned_url = await s3controller.generate_presigned_url(
        f"apps/test.zip", client_method="get_object"
    )
    # Ensure we're testing the S3 proxy, not direct MinIO access
    assert "/s3/" in presigned_url, "Presigned URL should be a proxy URL"
    response = requests.get(presigned_url)
    assert response.status_code == 200
    assert response.content == zip_buffer.getvalue()

    # Test get with range
    response = requests.get(
        presigned_url, headers={"Range": "bytes=0-10", "Origin": "http://localhost"}
    )
    assert response.status_code == 206
    assert response.content == zip_buffer.getvalue()[:11]
    assert response.headers["access-control-allow-origin"] == "http://localhost"

    # Test retrieving the first file inside the ZIP
    presigned_url = await s3controller.generate_presigned_url(
        f"apps/test.zip", client_method="get_object"
    )
    # Ensure we're testing the S3 proxy, not direct MinIO access
    assert "/s3/" in presigned_url, "Presigned URL should be a proxy URL"
    response = requests.get(presigned_url)
    assert response.status_code == 200
    assert response.content == zip_buffer.getvalue()

    # Test get with range
    response = requests.get(
        presigned_url, headers={"Range": "bytes=0-10", "Origin": "http://localhost"}
    )
    assert response.status_code == 206
    assert response.content == zip_buffer.getvalue()[:11]
    assert response.headers["access-control-allow-origin"] == "http://localhost"


# pylint: disable=too-many-statements
async def test_s3(minio_server, fastapi_server, test_user_token_8):
    """Test s3 service."""
    api = await connect_to_server(
        {"name": "anonymous client", "server_url": WS_SERVER_URL}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()
    s3controller = await api.get_service("public/s3-storage")
    with pytest.raises(
        Exception, match=r".*Permission denied: workspace is read-only.*"
    ):
        info = await s3controller.generate_credential()
    with pytest.raises(
        Exception, match=r".*Permission denied: workspace is read-only.*"
    ):
        info = await s3controller.generate_presigned_url("", client_method="put_object")
    content = os.urandom(1024)
    response = requests.put(
        f"{SERVER_URL}/{workspace}/files/my-data-small.txt",
        headers={"Authorization": f"Bearer {token}"},
        data=content,
    )
    assert not response.ok
    assert response.status_code == 403

    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token_8}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    s3controller = await api.get_service("public/s3-storage")
    info = await s3controller.generate_credential()
    print("Testing s3 client access...")

    def test_file_requests():
        print("Uploading small file...")
        # Upload small file (<5MB)
        content = os.urandom(2 * 1024 * 1024)
        response = requests.put(
            f"{SERVER_URL}/{workspace}/files/my-data-small.txt",
            headers={"Authorization": f"Bearer {token}"},
            data=content,
        )
        assert (
            response.status_code == 200
        ), f"failed to upload {response.reason}: {response.text}"

        print("Uploading large file...")
        # Upload large file with 100MB (>5M will trigger multi-part upload)
        content = os.urandom(10 * 1024 * 1024)
        response = requests.put(
            f"{SERVER_URL}/{workspace}/files/my-data-large.txt",
            headers={"Authorization": f"Bearer {token}"},
            data=content,
        )
        assert (
            response.status_code == 200
        ), f"failed to upload {response.reason}: {response.text}"

        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert find_item(response["children"], "name", "my-data-small.txt")
        assert find_item(response["children"], "name", "my-data-large.txt")

        # Test request with range
        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/my-data-large.txt",
            headers={"Authorization": f"Bearer {token}", "Range": "bytes=10-1033"},
            data=content,
        )
        assert len(response.content) == 1024
        assert response.content == content[10:1034]
        assert response.ok

        # Delete the large file
        response = requests.delete(
            f"{SERVER_URL}/{workspace}/files/my-data-large.txt",
            headers={"Authorization": f"Bearer {token}"},
            data=content,
        )
        assert (
            response.status_code == 200
        ), f"failed to delete {response.reason}: {response.text}"

        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
        assert find_item(response["children"], "name", "my-data-small.txt")
        assert not find_item(response["children"], "name", "my-data-large.txt")

        # Should fail if we don't pass the token
        response = requests.get(f"{SERVER_URL}/{workspace}/files/hello.txt")
        assert not response.ok
        assert response.status_code == 403

        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200

        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/hello.txt",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.ok
        assert response.content == b"hello"

        response = requests.get(
            f"{SERVER_URL}/{workspace}/files/he",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 404

    async with aioboto3.Session().resource(
        "s3",
        endpoint_url=info["endpoint_url"],
        aws_access_key_id=info["access_key_id"],
        aws_secret_access_key=info["secret_access_key"],
        region_name=info["region_name"],
    ) as s3_client:
        bucket = await s3_client.Bucket(info["bucket"])

        # Listing the root folder should fail
        with pytest.raises(Exception, match=r".*An error occurred (AccessDenied)*"):
            async for s3_object in bucket.objects.all():
                print(s3_object)

        obj = await s3_client.Object(info["bucket"], info["prefix"] + "hello.txt")
        with open("/tmp/hello.txt", "w", encoding="utf-8") as fil:
            fil.write("hello")
        await obj.upload_file("/tmp/hello.txt")

        obj = await s3_client.Object(info["bucket"], info["prefix"] + "hello2.txt")
        with open("/tmp/hello2.txt", "w", encoding="utf-8") as fil:
            fil.write("hello")
        await obj.upload_file("/tmp/hello2.txt")
        await s3controller.delete_file("hello2.txt")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, test_file_requests)
        await asyncio.sleep(0.1)
        items = [item async for item in bucket.objects.filter(Prefix=info["prefix"])]
        assert find_item(
            items,
            "key",
            f"{workspace}/hello.txt",
        )

        url = await s3controller.generate_presigned_url("hello.txt")
        assert url.startswith("http") and "X-Amz-Algorithm" in url
        # Ensure we're testing the S3 proxy, not direct MinIO access
        assert "/s3/" in url, "Presigned URL should be a proxy URL"

        response = requests.get(url)
        assert response.ok, response.text
        assert response.content == b"hello"

        items = await s3controller.list_files()
        # assert len(items) == 3 # 2 files and 1 folder (applications)
        assert find_item(items, "name", "my-data-small.txt")
        assert find_item(items, "name", "hello.txt")
        # Upload without the prefix should fail
        obj = await s3_client.Object(info["bucket"], "hello.txt")
        with pytest.raises(Exception, match=r".*An error occurred (AccessDenied)*"):
            await obj.upload_file("/tmp/hello.txt")

        # Delete the entire folder
        response = requests.delete(
            f"{SERVER_URL}/{workspace}/files/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert (
            response.status_code == 200
        ), f"failed to delete {response.reason}: {response.text}"

        obj = await s3_client.Object(info["bucket"], info["prefix"] + "hello.txt")
        with pytest.raises(Exception, match=r".*HeadObject operation: Not Found*"):
            assert await obj.content_length

    await api.disconnect()


# def test_built_in_minio_server():
#     """Test built-in Minio server functionality."""
#     # Create a temporary directory for Minio data
#     temp_dir = tempfile.mkdtemp()
#     port = 9999  # Use a different port to avoid conflicts
#     try:
#         # Start the Hypha server with built-in Minio
#         process = subprocess.Popen(
#             [
#                 sys.executable,
#                 "-m",
#                 "hypha.server",
#                 "--start-minio-server",
#                 f"--minio-workdir={temp_dir}",
#                 f"--minio-port={port}",
#                 "--minio-root-user=testuser",
#                 "--minio-root-password=testpassword",
#                 "--enable-s3",
#                 "--port=9555",  # Different port for Hypha
#             ]
#         )

#         # Wait for the server to start
#         server_url = "http://127.0.0.1:9555"
#         timeout = 20  # Longer timeout as both Hypha and Minio need to start

#         while timeout > 0:
#             try:
#                 response = requests.get(f"{server_url}/health/readiness")
#                 if response.ok:
#                     break
#             except requests.RequestException:
#                 pass
#             timeout -= 0.5
#             time.sleep(0.5)

#         if timeout <= 0:
#             raise TimeoutError("Server did not start in time")

#         # Verify the Minio port is being used
#         minio_response = requests.get(f"http://127.0.0.1:{port}/minio/health/live")
#         assert minio_response.ok, "Minio server is not running on the expected port"

#     finally:
#         # Clean up
#         process.terminate()
#         try:
#             process.wait(timeout=5)
#         except subprocess.TimeoutExpired:
#             process.kill()

#         shutil.rmtree(temp_dir, ignore_errors=True)


# # Test that trying to use both flags raises an error
# def test_minio_server_with_s3_settings():
#     """Test that using both built-in Minio and S3 settings raises an error."""
#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Try to start with conflicting settings
#         process = subprocess.Popen(
#             [
#                 sys.executable,
#                 "-m",
#                 "hypha.server",
#                 "--start-minio-server",
#                 f"--minio-workdir={temp_dir}",
#                 "--endpoint-url=http://example.com",  # This should cause a conflict
#                 "--port=9666",
#             ],
#             stderr=subprocess.PIPE,
#         )

#         # The process should exit with an error
#         _, stderr = process.communicate(timeout=5)
#         stderr_text = stderr.decode()

#         # Check that the process exited with an error
#         assert process.returncode != 0

#         # Check that the error message is as expected
#         assert "Cannot use --start-minio-server with S3 settings" in stderr_text
