"""Provide an s3 interface."""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from email.utils import formatdate
from typing import Any, Dict

import botocore
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request, Query, Body
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from hypha.utils.asgi_proxy import (
    make_simple_proxy_app,
    ProxyContext,
    BaseURLProxyConfigMixin,
    ProxyConfig,
)

from hypha.core import UserInfo, WorkspaceInfo, UserPermission
from pydantic import BaseModel
from typing import List
import httpx
import json
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


class PartETag(BaseModel):
    part_number: int
    etag: str


class CompleteMultipartUploadRequest(BaseModel):
    upload_id: str
    parts: List[PartETag]


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
            "AllowedMethods": ["GET", "HEAD", "PUT", "POST", "DELETE"],
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
        cleanup_period=300,
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
        self.cleanup_period = cleanup_period

        # Use Redis cache for multipart upload info (same as artifact manager)
        self._cache = store.get_redis_cache()
        self._redis = store.get_redis()
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

        # Initialize cleanup task as None - will be started when server starts
        self._cleanup_task = None

        # Register startup callback to start the cleanup task
        store.get_event_bus().on_local("startup", self._on_startup)

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
            context = {"ws": workspace, "user": user_info.model_dump()}
            return await self._upload_file(path, request, context)

        @router.post("/{workspace}/create-multipart-upload")
        async def create_multipart_upload(
            workspace: str,
            path: str,
            part_count: int = Query(
                ..., gt=0, description="The total number of parts for the upload."
            ),
            expires_in: int = Query(
                3600,
                description="The number of seconds for the presigned URLs to expire.",
            ),
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Initiate a multipart upload and get presigned URLs for all parts."""
            ws = await store.get_workspace_info(workspace, load=True)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exist: {workspace}",
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
            if not user_info.check_permission(ws.id, UserPermission.read_write):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )

            context = {"ws": workspace, "user": user_info.model_dump()}

            try:
                result = await self.put_file_start_multipart(
                    file_path=path,
                    part_count=part_count,
                    expires_in=expires_in,
                    context=context,
                )
                return JSONResponse(status_code=200, content=result)
            except ValueError as e:
                return JSONResponse(
                    status_code=400, content={"success": False, "detail": str(e)}
                )
            except Exception as e:
                logger.error(f"Failed to create multipart upload: {e}")
                return JSONResponse(
                    status_code=500, content={"success": False, "detail": str(e)}
                )

        @router.post("/{workspace}/complete-multipart-upload")
        async def complete_multipart_upload(
            workspace: str,
            data: CompleteMultipartUploadRequest = Body(...),
            user_info: store.login_optional = Depends(store.login_optional),
        ):
            """Complete the multipart upload and finalize the file in S3."""
            ws = await store.get_workspace_info(workspace, load=True)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exist: {workspace}",
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
            if not user_info.check_permission(ws.id, UserPermission.read_write):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )

            upload_id = data.upload_id
            parts = data.parts
            context = {"ws": workspace, "user": user_info.model_dump()}

            try:
                # Convert parts to the format expected by the service function
                parts_data = [
                    {"part_number": p.part_number, "etag": p.etag} for p in parts
                ]

                result = await self.put_file_complete_multipart(
                    upload_id=upload_id, parts=parts_data, context=context
                )
                return JSONResponse(status_code=200, content=result)
            except ValueError as e:
                return JSONResponse(
                    status_code=400, content={"success": False, "detail": str(e)}
                )
            except ClientError as e:
                logger.error(
                    f"Failed to complete multipart upload: {e.response['Error']['Message']}"
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "detail": f"Failed to complete multipart upload: {e.response['Error']['Message']}",
                    },
                )
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                return JSONResponse(
                    status_code=500, content={"success": False, "detail": str(e)}
                )

        if self.enable_s3_proxy:

            class S3ProxyConfig(BaseURLProxyConfigMixin, ProxyConfig):
                # Set your S3 root endpoint
                upstream_base_url = self.endpoint_url
                rewrite_host_header = (
                    self.endpoint_url.replace("https://", "")
                    .replace("http://", "")
                    .split("/")[0]
                )

                def get_upstream_url(self, scope):
                    # Strip the /s3/ prefix from the path before forwarding to MinIO
                    path = scope["path"]
                    if path.startswith("/s3/"):
                        path = path[4:]  # Remove "/s3/"
                    from urllib.parse import urljoin

                    return urljoin(self.upstream_base_url, path)

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

    async def _upload_file(self, path: str, request: Request, context: dict):
        """Upload file using simple PUT (no multipart)."""
        # Generate presigned URL for upload
        presigned_url = await self.generate_presigned_url(
            path, client_method="put_object", expiration=3600, context=context
        )

        # Stream the request body to the presigned URL
        async with httpx.AsyncClient(timeout=300) as client:
            # Use an async generator to stream chunks
            async def body_generator():
                async for chunk in request.stream():
                    yield chunk

            # Stream the body to the presigned URL
            response = await client.put(
                presigned_url,
                content=body_generator(),
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": request.headers.get("content-length"),
                },
            )

        if response.status_code != 200:
            return JSONResponse(
                status_code=response.status_code,
                content={
                    "success": False,
                    "detail": f"S3 upload failed with status {response.status_code}: {response.text}",
                },
            )

        content = json.dumps({"success": True, "message": "File uploaded successfully"})
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(content)),
            "ETag": response.headers.get("ETag"),
        }
        return Response(content=content, headers=headers)

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

    async def _delete_full_path(self, full_path: str):
        """Delete files using already-joined full path."""
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
                        "detail": f"File does not exists: {full_path}",
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
        return await self._delete_full_path(full_path)

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
                workspace, UserPermission.read_write
            ), f"Permission denied: {workspace}"
            full_path = safe_join(workspace, path)
        if not full_path.endswith("/"):
            full_path += "/"
        return await self._delete_full_path(full_path)

    async def delete_file(self, path: str, context: dict = None):
        """Delete file."""
        assert not path.endswith("/"), "Path should not end with a slash for file"
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
                workspace, UserPermission.read_write
            ), f"Permission denied: {workspace}"
            full_path = safe_join(workspace, path)
        return await self._delete_full_path(full_path)

    async def put_file(
        self,
        file_path: str,
        use_proxy: bool = None,
        use_local_url: bool = False,
        expires_in: float = 3600,
        ttl: int = None,
        context: dict = None,
    ) -> str:
        """Generate a presigned URL for uploading a file to S3."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        ws = await self.store.get_workspace_info(workspace, load=True)
        if not ws:
            raise ValueError(f"Workspace {workspace} not found.")
        if ws.read_only:
            raise PermissionError("Permission denied: workspace is read-only")
        if not user_info.check_permission(ws.id, UserPermission.read_write):
            raise PermissionError(f"Permission denied: {workspace}")

        # Schedule file for deletion if TTL is specified
        if ttl is not None and ttl > 0:
            await self._schedule_file_deletion(file_path, workspace, ttl)

        return await self.generate_presigned_url(
            file_path,
            client_method="put_object",
            expiration=expires_in,
            context=context,
        )

    async def get_file(
        self,
        file_path: str,
        use_proxy: bool = None,
        use_local_url: bool = False,
        expires_in: float = 3600,
        context: dict = None,
    ) -> str:
        """Generate a presigned URL for downloading a file from S3."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        ws = await self.store.get_workspace_info(workspace, load=True)
        if not ws:
            raise ValueError(f"Workspace {workspace} not found.")
        if not user_info.check_permission(ws.id, UserPermission.read):
            raise PermissionError(f"Permission denied: {workspace}")

        return await self.generate_presigned_url(
            file_path,
            client_method="get_object",
            expiration=expires_in,
            context=context,
        )

    async def put_file_start_multipart(
        self,
        file_path: str,
        part_count: int,
        expires_in: int = 3600,
        ttl: int = None,
        context: dict = None,
    ) -> dict:
        """Initiate a multipart upload and return presigned URLs for all parts."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        ws = await self.store.get_workspace_info(workspace, load=True)
        if not ws:
            raise ValueError(f"Workspace {workspace} not found.")
        if ws.read_only:
            raise PermissionError("Permission denied: workspace is read-only")
        if not user_info.check_permission(ws.id, UserPermission.read_write):
            raise PermissionError(f"Permission denied: {workspace}")

        if part_count <= 0 or part_count > 10000:
            raise ValueError("part_count must be between 1 and 10000")

        full_path = safe_join(workspace, file_path)

        async with self.create_client_async() as s3_client:
            # Create multipart upload
            response = await s3_client.create_multipart_upload(
                Bucket=self.workspace_bucket, Key=full_path
            )
            upload_id = response["UploadId"]

            # Generate presigned URLs for all parts
            parts = []
            for part_number in range(1, part_count + 1):
                part_url = await s3_client.generate_presigned_url(
                    "upload_part",
                    Params={
                        "Bucket": self.workspace_bucket,
                        "Key": full_path,
                        "PartNumber": part_number,
                        "UploadId": upload_id,
                    },
                    ExpiresIn=expires_in,
                )

                # Apply S3 proxy replacement if enabled
                if self.enable_s3_proxy:
                    part_url = f"{self.store.public_base_url}/s3/{self.workspace_bucket}/{full_path}?{part_url.split('?')[1]}"

                parts.append({"part_number": part_number, "url": part_url})

            # Cache the upload info for later completion (same as artifact manager)
            cache_key = f"multipart_upload:{upload_id}"
            await self._cache.set(
                cache_key,
                {
                    "file_path": file_path,
                    "full_path": full_path,
                    "workspace": workspace,
                    "ttl": ttl,  # Store TTL for scheduling on completion
                },
                ttl=expires_in + 10,
            )

            return {"upload_id": upload_id, "parts": parts}

    async def put_file_complete_multipart(
        self,
        upload_id: str,
        parts: list,
        context: dict = None,
    ) -> dict:
        """Complete a multipart upload."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        ws = await self.store.get_workspace_info(workspace, load=True)
        if not ws:
            raise ValueError(f"Workspace {workspace} not found.")
        if ws.read_only:
            raise PermissionError("Permission denied: workspace is read-only")
        if not user_info.check_permission(ws.id, UserPermission.read_write):
            raise PermissionError(f"Permission denied: {workspace}")

        if not parts:
            raise ValueError("Parts list cannot be empty")

        # We need to find the file path from the upload_id
        # This is a limitation - we don't have the file path stored with the upload_id
        # For now, we'll let the S3 complete_multipart_upload handle the validation

        async with self.create_client_async() as s3_client:
            # Format parts for S3 API
            parts_for_s3 = [
                {"PartNumber": part["part_number"], "ETag": part["etag"]}
                for part in parts
            ]

            # Get upload info from cache (same as artifact manager)
            cache_key = f"multipart_upload:{upload_id}"
            upload_info = await self._cache.get(cache_key)
            if not upload_info:
                raise ValueError(f"Upload ID {upload_id} not found or expired")

            file_key = upload_info["full_path"]

            try:
                # Complete the multipart upload
                response = await s3_client.complete_multipart_upload(
                    Bucket=self.workspace_bucket,
                    Key=file_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts_for_s3},
                )

                # Schedule file for deletion if TTL was specified
                file_ttl = upload_info.get("ttl")
                if file_ttl is not None and file_ttl > 0:
                    await self._schedule_file_deletion(
                        upload_info["file_path"], upload_info["workspace"], file_ttl
                    )

                # Clean up the upload info from cache
                await self._cache.delete(cache_key)

                return {
                    "success": True,
                    "message": "File uploaded successfully",
                    "etag": response.get("ETag"),
                    "location": response.get("Location"),
                }

            except ClientError as e:
                # Let the error propagate with S3's error message
                raise e

    async def remove_file(
        self,
        path: str,
        context: dict = None,
    ) -> dict:
        """Remove a file or directory from S3. Combines functionality of delete_file and delete_directory."""
        workspace = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        ws = await self.store.get_workspace_info(workspace, load=True)
        if not ws:
            raise ValueError(f"Workspace {workspace} not found.")
        if ws.read_only:
            raise PermissionError("Permission denied: workspace is read-only")
        if not user_info.check_permission(ws.id, UserPermission.read_write):
            raise PermissionError(f"Permission denied: {workspace}")

        if path.startswith("/"):
            # Admin access to root folder
            if not user_info.check_permission("*", UserPermission.admin):
                raise PermissionError(
                    "Permission denied: only admin can access the root folder."
                )
            # Remove the leading slash
            full_path = path[1:]
        else:
            full_path = safe_join(workspace, path)

        async with self.create_client_async() as s3_client:
            try:
                # Check if it's a directory by looking for a trailing slash or checking if objects exist with this prefix
                if path.endswith("/"):
                    # It's explicitly a directory request
                    # Ensure full_path ends with "/" for remove_objects_async
                    if not full_path.endswith("/"):
                        full_path += "/"
                    await remove_objects_async(
                        s3_client, self.workspace_bucket, full_path
                    )
                    return {
                        "success": True,
                        "message": f"Directory deleted successfully: {path}",
                    }

                # Try deleting as a file first - but check if it exists first
                try:
                    # Check if the file exists
                    await s3_client.head_object(
                        Bucket=self.workspace_bucket, Key=full_path
                    )
                    # If head_object succeeds, the file exists, now delete it
                    response = await s3_client.delete_object(
                        Bucket=self.workspace_bucket, Key=full_path
                    )
                    return {
                        "success": True,
                        "message": f"File deleted successfully: {path}",
                    }
                except ClientError as head_error:
                    if head_error.response["Error"]["Code"] == "404":
                        # File doesn't exist, check if it's a directory
                        pass
                    else:
                        # Some other error
                        logger.error(f"Error checking file {full_path}: {head_error}")
                        return {
                            "success": False,
                            "message": f"Error accessing file: {head_error.response['Error']['Message']}",
                        }

                # Check if it's a directory by listing objects with this prefix
                directory_path = full_path + "/"
                paginator = s3_client.get_paginator("list_objects_v2")

                async for page in paginator.paginate(
                    Bucket=self.workspace_bucket, Prefix=directory_path, MaxKeys=1
                ):
                    if "Contents" in page and len(page["Contents"]) > 0:
                        # It's a directory with contents
                        await remove_objects_async(
                            s3_client, self.workspace_bucket, directory_path
                        )
                        return {
                            "success": True,
                            "message": f"Directory deleted successfully: {path}",
                        }

                # If we get here, neither file nor directory was found
                return {"success": False, "message": f"Path does not exist: {path}"}

            except ClientError as e:
                logger.error(f"Failed to remove path {path}: {e}")
                return {
                    "success": False,
                    "message": f"Failed to remove path: {e.response['Error']['Message']}",
                }

    async def _schedule_file_deletion(
        self, file_path: str, workspace: str, ttl_seconds: int
    ):
        """Schedule a file for deletion after TTL expires."""
        now = time.time()
        expires_at = now + ttl_seconds
        # Store deletion info in Redis with expiration
        full_path = safe_join(workspace, file_path)

        # Add to sorted set for efficient range queries
        # Ensure the key is a string, not bytes
        await self._redis.zadd("s3_expiry_index", {str(full_path): expires_at})

    async def cleanup_expired_files(self):
        """Clean up expired files. Called by periodic task."""
        try:
            now = time.time()

            # Get expired files from sorted set
            expired_keys = await self._redis.zrangebyscore("s3_expiry_index", 0, now)

            if not expired_keys:
                return  # No expired files

            deleted_count = 0
            async with self.create_client_async() as s3_client:
                for s3_key in expired_keys:
                    try:
                        # Ensure s3_key is a string, not bytes
                        s3_key_str = (
                            s3_key.decode("utf-8")
                            if isinstance(s3_key, bytes)
                            else str(s3_key)
                        )

                        # Delete the file from S3
                        await s3_client.delete_object(
                            Bucket=self.workspace_bucket, Key=s3_key_str
                        )
                        deleted_count += 1
                        logger.info(f"Deleted expired file: {s3_key_str}")
                    except Exception as e:
                        logger.error(f"Failed to delete expired file {s3_key}: {e}")

                    # Remove from expiry index
                    await self._redis.zrem("s3_expiry_index", s3_key)

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired files")

        except Exception as e:
            logger.error(f"Error during expired files cleanup: {e}")

    async def _periodic_cleanup(self):
        """Periodic task to clean up expired files."""
        cleanup_interval = self.cleanup_period

        while True:
            try:
                await asyncio.sleep(cleanup_interval)

                # Only run cleanup when the system is not too busy
                # Basic check: if there are too many concurrent connections, skip this round
                try:
                    await self.cleanup_expired_files()
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}")
                    # Continue the loop even if cleanup fails

            except asyncio.CancelledError:
                logger.info("Periodic cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in periodic cleanup task: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)

    def start_cleanup_task(self):
        """Start the periodic cleanup task. Called when event loop is available."""
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                logger.info("Started S3 TTL cleanup task")
            except RuntimeError:
                # No event loop available yet, will be started later
                pass

    async def _on_startup(self, data=None):
        """Handle startup event to start cleanup task."""
        self.start_cleanup_task()

    def get_s3_service(self):
        """Get s3 controller."""
        svc = {
            "id": "s3-storage",
            "name": "S3 Storage",
            "config": {"visibility": "public", "require_context": True},
            "list_files": self.list_files,
            "delete_directory": self.delete_directory,  # deprecated, use remove_file instead
            "delete_file": self.delete_file,  # deprecated, use remove_file instead
            "generate_presigned_url": self.generate_presigned_url,  # deprecated, use put_file or get_file instead
            "put_file": self.put_file,
            "get_file": self.get_file,
            "put_file_start_multipart": self.put_file_start_multipart,
            "put_file_complete_multipart": self.put_file_complete_multipart,
            "remove_file": self.remove_file,
        }
        if self.minio_client:
            svc["generate_credential"] = self.generate_credential
        return svc
