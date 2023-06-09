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

    minio_client.admin_user_add(username, "239udslfj3")
    # overwrite the password
    minio_client.admin_user_add(username, "23923432423j3")
    minio_client.admin_user_add(username2, "234slfj3")
    user_list = minio_client.admin_user_list()

    assert find_item(user_list, "accessKey", username)
    assert find_item(user_list, "accessKey", username2)
    minio_client.admin_user_disable(username)
    user_list = minio_client.admin_user_list()
    user1 = find_item(user_list, "accessKey", username)
    assert user1["userStatus"] == "disabled"
    minio_client.admin_user_enable(username)
    user_list = minio_client.admin_user_list()
    user1 = find_item(user_list, "accessKey", username)
    assert user1["userStatus"] == "enabled"
    user = minio_client.admin_user_info(username)
    assert user["userStatus"] == "enabled"
    minio_client.admin_user_remove(username2)
    user_list = minio_client.admin_user_list()
    assert find_item(user_list, "accessKey", username2) is None

    minio_client.admin_group_add("my-group", username)
    ginfo = minio_client.admin_group_info("my-group")
    assert ginfo["groupName"] == "my-group"
    assert username in ginfo["members"]
    assert ginfo["groupStatus"] == "enabled"

    minio_client.admin_group_add("my-group", username)

    minio_client.admin_group_disable("my-group")
    ginfo = minio_client.admin_group_info("my-group")
    assert ginfo["groupStatus"] == "disabled"

    minio_client.admin_group_remove("my-group", ginfo["members"])
    ginfo = minio_client.admin_group_info("my-group")
    assert ginfo.get("members") is None

    # remove empty group
    minio_client.admin_group_remove("my-group")
    with pytest.raises(Exception, match=r".*Failed to run mc command*"):
        minio_client.admin_group_info("my-group")

    minio_client.admin_group_add("my-group", username)

    minio_client.admin_user_add(username2, "234slfj3")
    minio_client.admin_group_add("my-group", username2)
    userinfo = minio_client.admin_user_info(username2)
    assert "my-group" in str(userinfo["memberOf"])

    ginfo = minio_client.admin_group_info("my-group")
    assert username in ginfo["members"] and username2 in ginfo["members"]

    minio_client.admin_group_enable("my-group")
    ginfo = minio_client.admin_group_info("my-group")
    assert ginfo["groupStatus"] == "enabled"

    minio_client.admin_policy_create(
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
    response = minio_client.admin_policy_info("admins")
    assert response["policy"] == "admins"
    policy_list = minio_client.admin_policy_list()
    print(policy_list)
    assert find_item(policy_list, "policy", "admins")

    minio_client.admin_policy_attach("admins", user=username)
    userinfo = minio_client.admin_user_info(username)
    assert userinfo["policyName"] == "admins"

    minio_client.admin_policy_attach("admins", group="my-group")
    ginfo = minio_client.admin_group_info("my-group")
    assert ginfo["groupPolicy"] == "admins"
