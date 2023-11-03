"""Test minio client."""
import pytest
from hypha.minio import MinioClient
from . import MINIO_SERVER_URL, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, find_item

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


# pylint: disable=too-many-statements
async def test_minio(minio_server):
    """Test minio client."""
    minio_client = MinioClient(
        MINIO_SERVER_URL, MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, executable_path="bin"
    )
    username = "tmp-user"
    username2 = "tmp-user-2"

    await minio_client.admin_user_add(username, "239udslfj3")
    # overwrite the password
    await minio_client.admin_user_add(username, "23923432423j3")
    await minio_client.admin_user_add(username2, "234slfj3")
    user_list = await minio_client.admin_user_list()

    assert find_item(user_list, "accessKey", username)
    assert find_item(user_list, "accessKey", username2)
    await minio_client.admin_user_disable(username)
    user_list = await minio_client.admin_user_list()
    user1 = find_item(user_list, "accessKey", username)
    assert user1["userStatus"] == "disabled"
    await minio_client.admin_user_enable(username)
    user_list = await minio_client.admin_user_list()
    user1 = find_item(user_list, "accessKey", username)
    assert user1["userStatus"] == "enabled"
    user = await minio_client.admin_user_info(username)
    assert user["userStatus"] == "enabled"
    await minio_client.admin_user_remove(username2)
    user_list = await minio_client.admin_user_list()
    assert find_item(user_list, "accessKey", username2) is None

    await minio_client.admin_group_add("my-group", username)
    ginfo = await minio_client.admin_group_info("my-group")
    assert ginfo["groupName"] == "my-group"
    assert username in ginfo["members"]
    assert ginfo["groupStatus"] == "enabled"

    await minio_client.admin_group_add("my-group", username)

    await minio_client.admin_group_disable("my-group")
    ginfo = await minio_client.admin_group_info("my-group")
    assert ginfo["groupStatus"] == "disabled"

    await minio_client.admin_group_remove("my-group", ginfo["members"])
    ginfo = await minio_client.admin_group_info("my-group")
    assert ginfo.get("members") is None

    # remove empty group
    await minio_client.admin_group_remove("my-group")
    with pytest.raises(Exception, match=r".*Failed to run mc command*"):
        await minio_client.admin_group_info("my-group")

    await minio_client.admin_group_add("my-group", username)

    await minio_client.admin_user_add(username2, "234slfj3")
    await minio_client.admin_group_add("my-group", username2)
    userinfo = await minio_client.admin_user_info(username2)
    assert "my-group" in str(userinfo["memberOf"])

    ginfo = await minio_client.admin_group_info("my-group")
    assert username in ginfo["members"] and username2 in ginfo["members"]

    await minio_client.admin_group_enable("my-group")
    ginfo = await minio_client.admin_group_info("my-group")
    assert ginfo["groupStatus"] == "enabled"

    await minio_client.admin_policy_create(
        "admins",
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:ListAllMyBuckets"],
                    "Resource": ["arn:aws:s3:::*"],
                }
            ],
        },
    )
    response = await minio_client.admin_policy_info("admins")
    assert response["policy"] == "admins"
    policy_list = await minio_client.admin_policy_list()
    print(policy_list)
    assert find_item(policy_list, "policy", "admins")

    await minio_client.admin_policy_attach("admins", user=username)
    userinfo = await minio_client.admin_user_info(username)
    assert userinfo["policyName"] == "admins"

    await minio_client.admin_policy_attach("admins", group="my-group")
    ginfo = await minio_client.admin_group_info("my-group")
    assert ginfo["groupPolicy"] == "admins"
