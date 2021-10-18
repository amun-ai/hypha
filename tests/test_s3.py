"""Test S3 services."""
import os

import boto3
import pytest
import requests
from imjoy_rpc import connect_to_server

from . import SIO_SERVER_URL, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_s3(minio_server, socketio_server, test_user_token):
    """Test s3 service."""
    api = await connect_to_server(
        {"name": "test client", "server_url": SIO_SERVER_URL, "token": test_user_token}
    )
    workspace = api.config["workspace"]
    token = await api.generate_token()

    s3controller = await api.get_service("s3-storage")
    assert s3controller
    s3controller = await api.get_service({"workspace": "root", "name": "s3-storage"})
    info = await s3controller.generate_credential()
    s3_client = boto3.Session().resource(
        "s3",
        endpoint_url=info["endpoint_url"],
        aws_access_key_id=info["access_key_id"],
        aws_secret_access_key=info["secret_access_key"],
        region_name="EU",
    )
    bucket = s3_client.Bucket(info["bucket"])

    # Listing the root folder should fail
    with pytest.raises(Exception, match=r".*An error occurred (AccessDenied)*"):
        print(list(bucket.objects.all()))

    obj = s3_client.Object(info["bucket"], info["prefix"] + "hello.txt")
    with open("/tmp/hello.txt", "w", encoding="utf-8") as fil:
        fil.write("hello")
    obj.upload_file("/tmp/hello.txt")

    # Upload small file (<5MB)
    content = os.urandom(2 * 1024 * 1024)
    response = requests.put(
        f"{SIO_SERVER_URL}/{workspace}/files/my-data-small.txt",
        headers={"Authorization": f"Bearer {token}"},
        data=content,
    )
    assert (
        response.status_code == 200
    ), f"failed to upload {response.reason}: {response.text}"

    # Upload large file with 100MB
    content = os.urandom(100 * 1024 * 1024)
    response = requests.put(
        f"{SIO_SERVER_URL}/{workspace}/files/my-data-large.txt",
        headers={"Authorization": f"Bearer {token}"},
        data=content,
    )
    assert (
        response.status_code == 200
    ), f"failed to upload {response.reason}: {response.text}"

    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/",
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    assert find_item(response["children"], "name", "my-data-small.txt")
    assert find_item(response["children"], "name", "my-data-large.txt")

    # Test request with range
    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/my-data-large.txt",
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=10-1033"},
        data=content,
    )
    assert len(response.content) == 1024
    assert response.content == content[10:1034]
    assert response.ok

    # Delete the large file
    response = requests.delete(
        f"{SIO_SERVER_URL}/{workspace}/files/my-data-large.txt",
        headers={"Authorization": f"Bearer {token}"},
        data=content,
    )
    assert (
        response.status_code == 200
    ), f"failed to delete {response.reason}: {response.text}"

    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/",
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    assert find_item(response["children"], "name", "my-data-small.txt")
    assert not find_item(response["children"], "name", "my-data-large.txt")

    # Should fail if we don't pass the token
    response = requests.get(f"{SIO_SERVER_URL}/{workspace}/files/hello.txt")
    assert not response.ok

    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200

    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/hello.txt",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.ok
    assert response.content == b"hello"

    response = requests.get(
        f"{SIO_SERVER_URL}/{workspace}/files/he",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404

    assert find_item(
        list(bucket.objects.filter(Prefix=info["prefix"])),
        "key",
        f"{workspace}/hello.txt",
    )

    url = await s3controller.generate_presigned_url(
        info["bucket"], info["prefix"] + "hello.txt"
    )
    assert url.startswith("http") and "X-Amz-Algorithm" in url

    # Upload without the prefix should fail
    obj = s3_client.Object(info["bucket"], "hello.txt")
    with pytest.raises(Exception, match=r".*An error occurred (AccessDenied)*"):
        obj.upload_file("/tmp/hello.txt")

    await api.disconnect()
