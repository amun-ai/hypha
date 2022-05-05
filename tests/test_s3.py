"""Test S3 services."""
import os

import asyncio
import aioboto3
import pytest
import requests
from imjoy_rpc.hypha.websocket_client import connect_to_server

from . import WS_SERVER_URL, SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


# pylint: disable=too-many-statements
async def test_s3(minio_server, fastapi_server, test_user_token):
    """Test s3 service."""
    api = await connect_to_server(
        {"name": "test client", "server_url": WS_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    s3controller = await api.get_service("s3-storage")
    assert s3controller
    s3controller = await api.get_service("public/*:s3-storage")
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
        region_name="EU",
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

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, test_file_requests)
        await asyncio.sleep(0.1)
        items = [item async for item in bucket.objects.filter(Prefix=info["prefix"])]
        assert find_item(
            items,
            "key",
            f"{workspace}/hello.txt",
        )

        url = await s3controller.generate_presigned_url(
            info["bucket"], info["prefix"] + "hello.txt"
        )
        assert url.startswith("http") and "X-Amz-Algorithm" in url

        response = requests.get(url)
        assert response.ok
        assert response.content == b"hello"

        items = await s3controller.list_files()
        assert len(items) == 2
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
