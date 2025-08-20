"""Test S3 services."""

import os
import subprocess
import sys
import tempfile
import time
import httpx
import time

import asyncio
import aioboto3
import pytest
import io
import zipfile
import requests
import shutil
from hypha_rpc import connect_to_server

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


async def test_s3_multipart_upload(minio_server, fastapi_server, test_user_token):
    """Test S3 multipart upload functionality."""
    api = await connect_to_server(
        {
            "name": "test client for multipart",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    s3controller = await api.get_service("public/s3-storage")
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test put_file function
    file_path = "test_simple_upload.txt"
    put_url = await s3controller.put_file(file_path)
    assert put_url.startswith("http")
    assert "X-Amz-Algorithm" in put_url

    # Test get_file function
    get_url = await s3controller.get_file(file_path)
    assert get_url.startswith("http")
    assert "X-Amz-Algorithm" in get_url

    # Test multipart upload service functions
    multipart_file_path = "test_multipart_upload.txt"
    part_count = 3

    # Test put_file_start_multipart
    multipart_info = await s3controller.put_file_start_multipart(
        file_path=multipart_file_path, part_count=part_count, expires_in=3600
    )

    # Verify the response structure
    assert "upload_id" in multipart_info
    assert "parts" in multipart_info
    assert len(multipart_info["parts"]) == part_count

    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]

    # Verify each part has the expected structure
    for i, part in enumerate(part_urls):
        assert part["part_number"] == i + 1
        assert "url" in part
        assert part["url"].startswith("http")

    # Create real test data and upload each part
    part_size = 5 * 1024 * 1024  # 5MB minimum per part for S3
    test_parts_data = [
        b"A" * part_size,  # Part 1: 5MB of 'A's
        b"B" * part_size,  # Part 2: 5MB of 'B's  
        b"C" * (part_size // 2),  # Part 3: 2.5MB of 'C's (last part can be smaller)
    ]
    expected_content = b"".join(test_parts_data)

    # Upload each part and collect real ETags
    uploaded_parts = []
    async with httpx.AsyncClient(timeout=120) as upload_client:
        for i, (part_info, part_data) in enumerate(zip(part_urls, test_parts_data)):
            response = await upload_client.put(part_info["url"], content=part_data)
            assert response.status_code == 200, f"Failed to upload part {i+1}: {response.text}"
            
            # Extract real ETag from response headers
            etag = response.headers["ETag"].strip('"').strip("'")
            uploaded_parts.append({
                "part_number": part_info["part_number"], 
                "etag": etag
            })

    # Test put_file_complete_multipart with real uploaded parts
    result = await s3controller.put_file_complete_multipart(
        upload_id=upload_id, parts=uploaded_parts
    )
    # Should succeed with real uploads
    assert result["success"] is True
    assert "message" in result
    
    # Verify the file was uploaded correctly by downloading and checking content
    get_url = await s3controller.get_file(multipart_file_path)
    async with httpx.AsyncClient(timeout=60) as download_client:
        response = await download_client.get(get_url)
        assert response.status_code == 200
        assert response.content == expected_content, "Multipart upload content verification failed"

    # Test create-multipart-upload endpoint
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/create-multipart-upload",
            params={
                "path": "test_http_multipart.txt",
                "part_count": 2,
                "expires_in": 3600,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        if response.status_code == 200:
            upload_data = response.json()
            assert "upload_id" in upload_data
            assert "parts" in upload_data
            assert len(upload_data["parts"]) == 2

            # Test complete-multipart-upload endpoint with real uploads
            http_part_size = 5 * 1024 * 1024  # 5MB minimum per part
            http_parts_data = [
                b"X" * http_part_size,  # Part 1: 5MB of 'X's
                b"Y" * (http_part_size // 2),  # Part 2: 2.5MB of 'Y's (last part can be smaller)
            ]
            http_expected_content = b"".join(http_parts_data)

            # Upload each part and collect real ETags
            http_uploaded_parts = []
            for i, (part_info, part_data) in enumerate(zip(upload_data["parts"], http_parts_data)):
                upload_response = await client.put(part_info["url"], content=part_data)
                assert upload_response.status_code == 200, f"Failed to upload HTTP part {i+1}: {upload_response.text}"
                
                # Extract real ETag from response headers
                etag = upload_response.headers["ETag"].strip('"').strip("'")
                http_uploaded_parts.append({
                    "part_number": part_info["part_number"], 
                    "etag": etag
                })

            complete_response = await client.post(
                f"{SERVER_URL}/{workspace}/complete-multipart-upload",
                json={"upload_id": upload_data["upload_id"], "parts": http_uploaded_parts},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )

            # Should succeed with real uploads
            assert complete_response.status_code == 200, f"Complete multipart failed: {complete_response.text}"
            success_data = complete_response.json()
            assert "success" in success_data
            assert success_data["success"] is True
            
            # Add a small delay to handle S3 eventual consistency
            await asyncio.sleep(0.1)
            
            # Verify the HTTP multipart uploaded file by downloading it
            http_get_response = await client.get(
                f"{SERVER_URL}/{workspace}/files/test_http_multipart.txt",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert http_get_response.status_code == 200, f"File not found after multipart upload completion. Response: {http_get_response.text}"
            assert http_get_response.content == http_expected_content, "HTTP multipart upload content verification failed"
        else:
            print(
                f"Create multipart upload failed: {response.status_code} - {response.text}"
            )

    await api.disconnect()


async def test_s3_service_functions(minio_server, fastapi_server, test_user_token):
    """Test all S3 service functions including the new ones."""
    api = await connect_to_server(
        {
            "name": "test client for service functions",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    s3controller = await api.get_service("public/s3-storage")
    workspace = api.config["workspace"]
    token = await api.generate_token()

    # Test put_file and get_file functions
    test_file_path = "service_test/test_file.txt"
    test_content = b"This is test content for service functions"

    # Test put_file function
    put_url = await s3controller.put_file(test_file_path)
    print(f"PUT URL: {put_url}")
    assert put_url.startswith("http")
    assert "X-Amz-Algorithm" in put_url

    # Upload using the presigned URL
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(put_url, content=test_content)
        assert response.status_code == 200, f"Upload failed: {response.text}"

    # Test get_file function
    get_url = await s3controller.get_file(test_file_path)
    print(f"GET URL: {get_url}")
    assert get_url.startswith("http")
    assert "X-Amz-Algorithm" in get_url

    # Download using the presigned URL
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(get_url)
        assert response.status_code == 200
        assert response.content == test_content

    # Test remove_file function - remove a file
    result = await s3controller.remove_file(test_file_path)
    print(f"Remove file result: {result}")
    assert (
        result["success"] is True
    ), f"Remove failed with message: {result.get('message', 'No message')}"
    assert "deleted successfully" in result["message"]

    # Verify file is deleted by trying to get it (should fail)
    try:
        get_url_after_delete = await s3controller.get_file(test_file_path)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(get_url_after_delete)
            # Should return 404 or similar error
            assert response.status_code in [404, 403], "File should be deleted"
    except Exception:
        # Expected - file should not exist
        pass

    # Test remove_file function - create directory structure and remove directory
    dir_test_files = [
        "test_directory/file1.txt",
        "test_directory/file2.txt",
        "test_directory/subfolder/file3.txt",
    ]

    # Upload files to create directory structure
    for file_path in dir_test_files:
        put_url = await s3controller.put_file(file_path)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(
                put_url, content=f"Content for {file_path}".encode()
            )
            assert response.status_code == 200

    # List files to verify they exist
    files_before = await s3controller.list_files("test_directory/")
    assert len(files_before) >= 2  # Should have files and subfolder

    # Test remove_file function - remove entire directory
    result = await s3controller.remove_file("test_directory/")
    assert result["success"] is True
    assert "Directory deleted successfully" in result["message"]

    # Verify directory is empty
    try:
        files_after = await s3controller.list_files("test_directory/")
        assert len(files_after) == 0, "Directory should be empty after deletion"
    except Exception:
        # Expected if directory no longer exists
        pass

    # Test remove_file with non-existent path
    result = await s3controller.remove_file("non_existent_file.txt")
    assert result["success"] is False
    assert "Path does not exist" in result["message"]

    # Test multipart upload with actual file content
    large_file_path = "multipart_test/large_file.bin"

    # Create test data (simulate large file in parts)
    part_size = 5 * 1024 * 1024  # 5MB per part (minimum for S3)
    parts_data = [
        b"A" * part_size,  # Part 1: 5MB of 'A's
        b"B" * part_size,  # Part 2: 5MB of 'B's
        b"C" * (part_size // 2),  # Part 3: 2.5MB of 'C's (last part can be smaller)
    ]
    expected_content = b"".join(parts_data)

    # Start multipart upload
    multipart_info = await s3controller.put_file_start_multipart(
        file_path=large_file_path, part_count=len(parts_data), expires_in=3600
    )

    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]

    # Upload each part
    uploaded_parts = []
    for i, (part_info, part_data) in enumerate(zip(part_urls, parts_data)):
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(part_info["url"], content=part_data)
            assert response.status_code == 200, f"Failed to upload part {i+1}"

            etag = response.headers["ETag"].strip('"').strip("'")
            uploaded_parts.append(
                {"part_number": part_info["part_number"], "etag": etag}
            )

    # Complete multipart upload
    result = await s3controller.put_file_complete_multipart(
        upload_id=upload_id, parts=uploaded_parts
    )
    assert result["success"] is True
    assert "File uploaded successfully" in result["message"]

    # Verify the uploaded file by downloading it
    get_url = await s3controller.get_file(large_file_path)
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(get_url)
        assert response.status_code == 200
        assert response.content == expected_content, "Multipart upload content mismatch"

    # Clean up - remove the multipart uploaded file
    result = await s3controller.remove_file(large_file_path)
    assert result["success"] is True

    # Test list_files function more thoroughly
    test_files = [
        "list_test/doc1.txt",
        "list_test/doc2.txt",
        "list_test/folder1/doc3.txt",
        "list_test/folder2/doc4.txt",
    ]

    # Upload test files
    for file_path in test_files:
        put_url = await s3controller.put_file(file_path)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(
                put_url, content=f"Content for {file_path}".encode()
            )
            assert response.status_code == 200

    # Test listing root of test directory
    files = await s3controller.list_files("list_test/")
    file_names = [f["name"] for f in files]
    assert "doc1.txt" in file_names
    assert "doc2.txt" in file_names
    assert "folder1" in file_names or any("folder1/" in f["name"] for f in files)
    assert "folder2" in file_names or any("folder2/" in f["name"] for f in files)

    # Clean up test files
    result = await s3controller.remove_file("list_test/")
    assert result["success"] is True

    await api.disconnect()


async def test_s3_ttl_functionality(minio_server, fastapi_server, test_user_token):
    """Test TTL functionality for uploaded files."""
    api = await connect_to_server(
        {
            "name": "test client for TTL functionality",
            "server_url": SERVER_URL,
            "token": test_user_token,
        }
    )
    s3controller = await api.get_service("public/s3-storage")
    workspace = api.config["workspace"]

    # Test TTL with put_file
    ttl_file_path = "ttl_test/short_lived_file.txt"
    ttl_seconds = 3  # Very short TTL for testing

    # Upload file with TTL
    put_url = await s3controller.put_file(ttl_file_path, ttl=ttl_seconds)
    assert put_url.startswith("http")
    assert "X-Amz-Algorithm" in put_url

    # Upload the file
    test_content = b"This file should expire soon"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.put(put_url, content=test_content)
        assert response.status_code == 200, f"Upload failed: {response.text}"

    # Verify file exists immediately after upload
    get_url = await s3controller.get_file(ttl_file_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(get_url)
        assert response.status_code == 200
        assert response.content == test_content

    # Test TTL with multipart upload
    multipart_ttl_path = "ttl_test/multipart_short_lived.bin"

    # Create small multipart upload for testing
    part_size = 5 * 1024 * 1024  # 5MB minimum
    parts_data = [
        b"A" * part_size,  # Part 1
        b"B" * (part_size // 2),  # Part 2 (smaller final part)
    ]

    # Start multipart upload with TTL
    multipart_info = await s3controller.put_file_start_multipart(
        file_path=multipart_ttl_path,
        part_count=len(parts_data),
        expires_in=3600,
        ttl=ttl_seconds,
    )

    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]

    # Upload parts
    uploaded_parts = []
    for i, (part_info, part_data) in enumerate(zip(part_urls, parts_data)):
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(part_info["url"], content=part_data)
            assert response.status_code == 200, f"Failed to upload part {i+1}"

            etag = response.headers["ETag"].strip('"').strip("'")
            uploaded_parts.append(
                {"part_number": part_info["part_number"], "etag": etag}
            )

    # Complete multipart upload
    result = await s3controller.put_file_complete_multipart(
        upload_id=upload_id, parts=uploaded_parts
    )
    assert result["success"] is True

    # Verify multipart file exists immediately after upload
    get_url_multipart = await s3controller.get_file(multipart_ttl_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(get_url_multipart)
        assert response.status_code == 200
        expected_content = b"".join(parts_data)
        assert response.content == expected_content

    print(f"Files uploaded successfully with TTL={ttl_seconds}s:")
    print(f"  - Single file: {ttl_file_path}")
    print(f"  - Multipart file: {multipart_ttl_path}")

    # Wait for TTL to expire plus cleanup period
    # TTL is 3 seconds, cleanup runs every 2 seconds, so wait 6 seconds to ensure cleanup runs
    wait_time = ttl_seconds + 3  # Add buffer for cleanup to run
    print(f"Waiting {wait_time} seconds for TTL to expire and cleanup to run...")
    await asyncio.sleep(wait_time)

    # Verify that files have been deleted by TTL cleanup
    print("Verifying files have been deleted by TTL cleanup...")

    # Try to get the single file - should fail (404 or similar)

    get_url_after_ttl = await s3controller.get_file(ttl_file_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(get_url_after_ttl)
        # File should be deleted, so we expect an error
        assert response.status_code in [
            404
        ], f"File should be deleted but got status {response.status_code}"
        print("✅ Single file successfully deleted by TTL cleanup")

    # Try to get the multipart file - should fail (404 or similar)

    get_url_multipart_after_ttl = await s3controller.get_file(multipart_ttl_path)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(get_url_multipart_after_ttl)
        # File should be deleted, so we expect an error
        assert response.status_code in [
            404
        ], f"Multipart file should be deleted but got status {response.status_code}"
        print("✅ Multipart file successfully deleted by TTL cleanup")

    # Test that files are not in the listing
    files_list = await s3controller.list_files("ttl_test/")
    file_names = [f["name"] for f in files_list]
    assert (
        ttl_file_path.split("/")[-1] not in file_names
    ), f"TTL file still exists in listing: {file_names}"
    assert (
        multipart_ttl_path.split("/")[-1] not in file_names
    ), f"TTL multipart file still exists in listing: {file_names}"
    print("✅ Files successfully removed from directory listing")

    print("🎉 TTL functionality test completed successfully!")

    await api.disconnect()
