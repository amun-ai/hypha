"""Provide an s3 interface."""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from email.utils import formatdate
from typing import Any, Dict

import botocore
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from asgiproxy.simple_proxy import make_simple_proxy_app
from asgiproxy.context import ProxyContext
from asgiproxy.config import BaseURLProxyConfigMixin, ProxyConfig

from hypha.core import UserInfo, WorkspaceInfo, UserPermission
from hypha.minio import MinioClient
from hypha.utils import (
    generate_password,
    list_objects_async,
    remove_objects_async,
    safe_join,
)

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("s3")
logger.setLevel(LOGLEVEL)


class FSFileResponse(FileResponse):
    """Represent an FS File Response."""

    chunk_size = 4096

    def __init__(self, s3client, bucket: str, key: str, **kwargs) -> None:
        """Set up the instance."""
        self.s3client = s3client
        self.bucket = bucket
        self.key = key
        super().__init__(key, **kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Make the call."""
        request_headers = Headers(scope=scope)

        range_header = request_headers.get("range", None)
        async with self.s3client as s3_client:
            try:
                kwargs = {"Bucket": self.bucket, "Key": self.key}
                if range_header is not None:
                    kwargs["Range"] = range_header
                obj_info = await s3_client.get_object(**kwargs)
                last_modified = formatdate(
                    datetime.timestamp(obj_info["LastModified"]), usegmt=True
                )
                self.headers.setdefault(
                    "content-length", str(obj_info["ContentLength"])
                )
                self.headers.setdefault(
                    "content-range", str(obj_info.get("ContentRange"))
                )
                self.headers.setdefault("last-modified", last_modified)
                self.headers.setdefault("etag", obj_info["ETag"])
            except ClientError as err:
                self.status_code = 404
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": "File not found, details: "
                        f"{json.dumps(err.response)}".encode("utf-8"),
                        "more_body": False,
                    }
                )
            except Exception as err:
                raise RuntimeError(
                    f"File at path {self.path} does not exist, details: {err}"
                ) from err
            else:
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                )
                if scope["method"].upper() == "HEAD":
                    await send(
                        {"type": "http.response.body", "body": b"", "more_body": False}
                    )
                else:
                    total_size = obj_info["ContentLength"]
                    sent_size = 0
                    chunks = obj_info["Body"].iter_chunks(chunk_size=self.chunk_size)
                    async for chunk in chunks:
                        sent_size += len(chunk)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": sent_size < total_size,
                            }
                        )

            if self.background is not None:
                await self.background()


DEFAULT_CORS_POLICY = {
    "CORSRules": [
        {
            "AllowedHeaders": ["*"],
            "ExposeHeaders": ["Accept-Ranges", "Content-Length", "Content-Range"],
            "AllowedMethods": ["GET", "HEAD"],
            "AllowedOrigins": ["*"],
            "MaxAgeSeconds": 3000,
        }
    ]
}


class S3Controller:
    """Represent an S3 controller."""

    # pylint: disable=too-many-statements

    def __init__(
        self,
        store,
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        region_name=None,
        endpoint_url_public=None,
        s3_admin_type="generic",
        enable_s3_proxy=False,
        workspace_bucket="hypha-workspaces",
        workspace_etc_dir="etc",
        executable_path="",
    ):
        """Set up controller."""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.s3_admin_type = s3_admin_type
        self.enable_s3_proxy = enable_s3_proxy
        if self.s3_admin_type == "minio":
            self.minio_client = MinioClient(
                endpoint_url,
                access_key_id,
                secret_access_key,
                executable_path=executable_path,
            )
        else:
            self.minio_client = None
        self.endpoint_url_public = endpoint_url_public or endpoint_url
        self.store = store
        self.workspace_bucket = workspace_bucket
        # self.local_log_dir = Path(local_log_dir)
        self.workspace_etc_dir = workspace_etc_dir.rstrip("/")
        s3client = self.create_client_sync()
        try:
            s3client.create_bucket(Bucket=self.workspace_bucket)
            logger.info("Bucket created: %s", self.workspace_bucket)
            try:
                if self.s3_admin_type != "minio":
                    s3client.put_bucket_cors(
                        Bucket=self.workspace_bucket,
                        CORSConfiguration=DEFAULT_CORS_POLICY,
                    )
            except Exception as ex:
                logger.error("Failed to apply CORS policy: %s", ex)
        except s3client.exceptions.BucketAlreadyExists:
            pass
        except s3client.exceptions.BucketAlreadyOwnedByYou:
            pass

        if self.minio_client:
            self.minio_client.admin_user_add_sync("root", generate_password())

        store.register_public_service(self.get_s3_service())
        store.set_s3_controller(self)
        cache = store.get_redis_cache()

        router = APIRouter()

        @router.put("/{workspace}/files/{path:path}")
        async def upload_file(
            workspace: str,
            path: str,
            request: Request,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Upload file."""
            ws = await store.get_workspace_info(workspace, load=True)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {workspace}",
                    },
                )
            if ws.read_only:
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: workspace ({workspace}) is read-only",
                    },
                )
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {workspace}",
                    },
                )
            if not user_info.check_permission(ws.id, UserPermission.read_write):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )
            path = safe_join(workspace, path)

            return await self._upload_file(path, request)

        if self.enable_s3_proxy:

            class S3ProxyConfig(BaseURLProxyConfigMixin, ProxyConfig):
                # Set your S3 root endpoint
                upstream_base_url = self.endpoint_url
                rewrite_host_header = (
                    self.endpoint_url.replace("https://", "")
                    .replace("http://", "")
                    .split("/")[0]
                )

            config = S3ProxyConfig()
            context = ProxyContext(config=config)
            s3_app = make_simple_proxy_app(context, proxy_websocket_handler=None)
            self.store.mount_app("/s3", s3_app, "s3-proxy")

        @router.get("/{workspace}/files/{path:path}")
        @router.delete("/{workspace}/files/{path:path}")
        async def get_or_delete_file(
            workspace: str,
            path: str,
            request: Request,
            max_length: int = 1000,
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Get or delete file."""
            ws = await store.get_workspace_info(workspace, load=True)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {workspace}",
                    },
                )
            if not user_info.check_permission(ws.id, UserPermission.read_write):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )

            if request.method == "GET":
                path = safe_join(workspace, path)
                return await self._get_files(path, max_length)

            if request.method == "DELETE":
                if ws.read_only:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "success": False,
                            "detail": f"Permission denied: workspace ({workspace}) is read-only",
                        },
                    )
                return await self._delete_dir_or_files(
                    path, {"ws": workspace, "user": user_info.model_dump()}
                )

        store.register_router(router)

    async def load_workspace_info(self, workspace_name, user_info):
        """Load workspace from s3 record."""
        if not user_info.check_permission(workspace_name, UserPermission.read_write):
            raise PermissionError(
                f"Permission denied: {workspace_name}, {user_info.id}"
            )
        config_path = f"{self.workspace_etc_dir}/{workspace_name}/manifest.json"
        try:
            async with self.create_client_async() as s3_client:
                response = await s3_client.get_object(
                    Bucket=self.workspace_bucket,
                    Key=config_path,
                )
                if (
                    "ResponseMetadata" in response
                    and "HTTPStatusCode" in response["ResponseMetadata"]
                ):
                    response_code = response["ResponseMetadata"]["HTTPStatusCode"]
                    if response_code != 200:
                        logger.info(
                            "Failed to download file: %s, status code: %d",
                            config_path,
                            response_code,
                        )
                        return None

                if "ETag" not in response:
                    return None
                data = await response["Body"].read()
                config = json.loads(data.decode("utf-8"))
                workspace = WorkspaceInfo.model_validate(config)
                logger.info("Loaded workspace from s3: %s", workspace_name)
                return workspace
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                return None

    async def _upload_file(self, path: str, request: Request):
        """Upload file."""
        async with self.create_client_async() as s3_client:
            mpu = await s3_client.create_multipart_upload(
                Bucket=self.workspace_bucket, Key=path
            )
            parts_info = {}
            futures = []
            count = 0
            # Stream support:
            # https://github.com/tiangolo/fastapi/issues/58#issuecomment-469355469
            current_chunk = b""
            async for chunk in request.stream():
                current_chunk += chunk
                if len(current_chunk) > 5 * 1024 * 1024:
                    count += 1
                    part_fut = s3_client.upload_part(
                        Bucket=self.workspace_bucket,
                        ContentLength=len(current_chunk),
                        Key=path,
                        PartNumber=count,
                        UploadId=mpu["UploadId"],
                        Body=current_chunk,
                    )
                    futures.append(part_fut)
                    current_chunk = b""
            # if multipart upload is activated
            if len(futures) > 0:
                if len(current_chunk) > 0:
                    # upload the last chunk
                    count += 1
                    part_fut = s3_client.upload_part(
                        Bucket=self.workspace_bucket,
                        ContentLength=len(current_chunk),
                        Key=path,
                        PartNumber=count,
                        UploadId=mpu["UploadId"],
                        Body=current_chunk,
                    )
                    futures.append(part_fut)

                parts = await asyncio.gather(*futures)
                parts_info["Parts"] = [
                    {"PartNumber": i + 1, "ETag": part["ETag"]}
                    for i, part in enumerate(parts)
                ]

                response = await s3_client.complete_multipart_upload(
                    Bucket=self.workspace_bucket,
                    Key=path,
                    UploadId=mpu["UploadId"],
                    MultipartUpload=parts_info,
                )
            else:
                response = await s3_client.put_object(
                    Body=current_chunk,
                    Bucket=self.workspace_bucket,
                    Key=path,
                    ContentLength=len(current_chunk),
                )

            assert "ETag" in response
            return JSONResponse(
                status_code=200,
                content=response,
            )

    async def _get_files(self, path: str, max_length: int):
        """Get files."""
        async with self.create_client_async() as s3_client:
            # List files in the folder
            if path.endswith("/"):
                items = await list_objects_async(
                    s3_client,
                    self.workspace_bucket,
                    path,
                    max_length=max_length,
                )
                if len(items) == 0:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": f"Directory does not exists: {path}",
                        },
                    )

                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "type": "directory",
                        "name": path.split("/")[-1],
                        "children": items,
                    },
                )
            # Download the file
            try:
                return FSFileResponse(
                    self.create_client_async(), self.workspace_bucket, path
                )
            except ClientError:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"File does not exists: {path}",
                    },
                )

    async def _delete_dir_or_files(self, path: str, context: dict):
        """Delete files."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        assert user_info.check_permission(
            workspace, UserPermission.read
        ), f"Permission denied: {workspace}"
        full_path = safe_join(workspace, path)

        async with self.create_client_async() as s3_client:
            if full_path.endswith("/"):
                await remove_objects_async(s3_client, self.workspace_bucket, full_path)
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                    },
                )
            try:
                response = await s3_client.delete_object(
                    Bucket=self.workspace_bucket, Key=full_path
                )
                response["success"] = True
                return JSONResponse(
                    status_code=200,
                    content=response,
                )
            except ClientError:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"File does not exists: {path}",
                    },
                )

    def create_client_sync(self):
        """Create client sync."""
        # Documentation for botocore client:
        # https://botocore.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
        return botocore.session.get_session().create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

    def create_client_async(self, public=False):
        """Create client async."""
        return get_session().create_client(
            "s3",
            endpoint_url=self.endpoint_url_public if public else self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

    async def cleanup_workspace(self, workspace: WorkspaceInfo, force=False):
        """Clean up workspace."""
        assert isinstance(workspace, WorkspaceInfo)
        if workspace.read_only and not force:
            return

        if not workspace.persistent or force:
            async with self.create_client_async() as s3client:
                # remove workspace etc files
                await remove_objects_async(
                    s3client,
                    self.workspace_bucket,
                    f"{self.workspace_etc_dir}/{workspace.id}/",
                )
                # remove files
                await remove_objects_async(
                    s3client, self.workspace_bucket, workspace.id + "/"
                )
                logger.info("Workspace's S3 storage cleaned up: %s", workspace.id)

        if self.minio_client:
            try:
                group_info = await self.minio_client.admin_group_info(workspace.id)
                # remove all the members
                await self.minio_client.admin_group_remove(
                    workspace.id, group_info["members"]
                )
                # now remove the empty group
                await self.minio_client.admin_group_remove(workspace.id)
            except Exception as ex:
                logger.error("Failed to remove minio group: %s, %s", workspace.id, ex)

    async def setup_workspace(self, workspace: WorkspaceInfo):
        """Set up workspace."""
        assert isinstance(workspace, WorkspaceInfo)
        # Save the workspace info
        # workspace_dir = self.local_log_dir / workspace.id
        # os.makedirs(workspace_dir, exist_ok=True)
        await self.save_workspace_config(workspace)
        if self.minio_client:
            # make sure we have the root user in every workspace
            await self.minio_client.admin_group_add(workspace.id, "root")
            policy_name = "policy-ws-" + workspace.id
            # policy example:
            # https://aws.amazon.com/premiumsupport/knowledge-center/iam-s3-user-specific-folder/
            await self.minio_client.admin_policy_create(
                policy_name,
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "AllowUserToSeeTheBucketInTheConsole",
                            "Action": ["s3:GetBucketLocation"],
                            "Effect": "Allow",
                            "Resource": [f"arn:aws:s3:::{self.workspace_bucket}"],
                        },
                        {
                            "Sid": "AllowRootAndHomeListingOfWorkspaceBucket",
                            "Action": ["s3:ListBucket"],
                            "Effect": "Allow",
                            "Resource": [f"arn:aws:s3:::{self.workspace_bucket}"],
                            "Condition": {
                                "StringEquals": {
                                    "s3:prefix": [f"{workspace.id}/"],
                                    "s3:delimiter": ["/"],
                                }
                            },
                        },
                        {
                            "Sid": "AllowListingOfWorkspaceFolder",
                            "Action": ["s3:ListBucket"],
                            "Effect": "Allow",
                            "Resource": [f"arn:aws:s3:::{self.workspace_bucket}"],
                            "Condition": {
                                "StringLike": {"s3:prefix": [f"{workspace.id}/*"]}
                            },
                        },
                        {
                            "Sid": "AllowAllS3ActionsInWorkspaceFolder",
                            "Action": ["s3:*"],
                            "Effect": "Allow",
                            "Resource": [
                                f"arn:aws:s3:::{self.workspace_bucket}/{workspace.id}",
                                f"arn:aws:s3:::{self.workspace_bucket}/{workspace.id}/*",
                            ],
                        },
                    ],
                },
            )
            try:
                await self.minio_client.admin_policy_attach(
                    policy_name, group=workspace.id
                )
            except Exception as ex:
                logger.error(
                    "Failed to attach policy to the group: %s, %s", workspace.id, ex
                )

    async def save_workspace_config(self, workspace: WorkspaceInfo):
        """Save workspace."""
        assert isinstance(workspace, WorkspaceInfo)
        if workspace.read_only:
            return
        async with self.create_client_async() as s3_client:
            response = await s3_client.put_object(
                Body=workspace.model_dump_json().encode("utf-8"),
                Bucket=self.workspace_bucket,
                Key=f"{self.workspace_etc_dir}/{workspace.id}/manifest.json",
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to save workspace ({workspace.id}) manifest: {response}"

    async def generate_credential(self, context: dict = None):
        """Generate credential."""
        assert self.minio_client, "Minio client is not available"
        workspace = context["ws"]
        ws = await self.store.get_workspace_info(workspace, load=True)
        assert ws, f"Workspace {workspace} not found."
        if ws.read_only:
            raise Exception("Permission denied: workspace is read-only")
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws.id, UserPermission.read_write):
            raise PermissionError(
                f"User {user_info.id} does not have write"
                f" permission to the workspace {workspace}"
            )
        password = generate_password()
        await self.minio_client.admin_user_add(user_info.id, password)
        # Make sure the user is in the workspace
        await self.minio_client.admin_group_add(workspace, user_info.id)
        return {
            "endpoint_url": self.endpoint_url_public,  # Return the public endpoint
            "access_key_id": user_info.id,
            "secret_access_key": password,
            "region_name": self.region_name,
            "bucket": self.workspace_bucket,
            "prefix": workspace + "/",  # important to have the trailing slash
        }

    async def list_files(
        self,
        path: str = "",
        max_length: int = 1000,
        context: dict = None,
    ) -> Dict[str, Any]:
        """List files in the folder."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if path.startswith("/"):
            assert user_info.check_permission(
                "*", UserPermission.admin
            ), "Permission denied: only admin can access the root folder."
            # remove the leading slash
            full_path = path[1:]
        else:
            assert user_info.check_permission(
                workspace, UserPermission.read
            ), f"Permission denied: {workspace}"
            full_path = safe_join(workspace, path)
        async with self.create_client_async() as s3_client:
            # List files in the folder
            if not full_path.endswith("/"):
                full_path += "/"
            items = await list_objects_async(
                s3_client, self.workspace_bucket, full_path, max_length=max_length
            )
            return items

    async def generate_presigned_url(
        self,
        path: str,
        client_method="get_object",
        expiration=3600,
        context: dict = None,
    ):
        """Generate presigned url."""
        try:
            workspace = context["ws"]
            ws = await self.store.get_workspace_info(workspace, load=True)
            assert ws, f"Workspace {workspace} not found."
            if ws.read_only and client_method != "get_object":
                raise Exception("Permission denied: workspace is read-only")
            if path.startswith("/"):
                user_info = UserInfo.model_validate(context["user"])
                assert user_info.check_permission(
                    "*", UserPermission.admin
                ), "Permission denied: only admin can access the root folder."
                # remove the leading slash
                path = path[1:]
            else:
                path = safe_join(workspace, path)
            public = False if self.enable_s3_proxy else True
            async with self.create_client_async(public=public) as s3_client:
                url = await s3_client.generate_presigned_url(
                    client_method,
                    Params={"Bucket": self.workspace_bucket, "Key": path},
                    ExpiresIn=expiration,
                )
                if self.enable_s3_proxy:
                    url = f"{self.store.public_base_url}/s3/{self.workspace_bucket}/{path}?{url.split('?')[1]}"
                return url

        except ClientError as err:
            logging.error(
                err
            )  # FIXME: If we raise the error why do we need to log it first?
            raise

    async def delete_directory(self, path: str, context: dict = None):
        """Delete directory."""
        # make sure the path is a directory
        assert path.endswith("/"), "Path should end with a slash for directory"
        return await self._delete_dir_or_files(path, context)

    async def delete_file(self, path: str, context: dict = None):
        """Delete file."""
        assert not path.endswith("/"), "Path should not end with a slash for file"
        return await self._delete_dir_or_files(path, context)

    def get_s3_service(self):
        """Get s3 controller."""
        svc = {
            "id": "s3-storage",
            "name": "S3 Storage",
            "config": {"visibility": "public", "require_context": True},
            "list_files": self.list_files,
            "delete_directory": self.delete_directory,
            "delete_file": self.delete_file,
            "generate_presigned_url": self.generate_presigned_url,
        }
        if self.minio_client:
            svc["generate_credential"] = self.generate_credential
        return svc
