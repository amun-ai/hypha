"""Provide an s3 interface."""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from email.utils import formatdate
from pathlib import Path
from typing import Any, Dict

import botocore
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send

from hypha.core import ClientInfo, UserInfo, WorkspaceInfo
from hypha.core.auth import login_optional
from hypha.core.store import RedisStore
from hypha.minio import MinioClient
from hypha.utils import (
    generate_password,
    list_objects_async,
    list_objects_sync,
    remove_objects_sync,
    safe_join,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("s3")
logger.setLevel(logging.INFO)


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
                if self.send_header_only:
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


class JSONResponse(Response):
    """Represent a JSON response.

    This implementation is needed because some of the S3 response
    contains datetime which is not json serializable.
    It works by setting `default=str` which converts the datetime
    into a string.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """Render the content."""
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            default=str,  # This will convert everything unknown to a string
        ).encode("utf-8")


class FSRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A rotating file handler for working with fsspec."""

    def __init__(self, s3_client, s3_bucket, s3_prefix, start_index, *args, **kwargs):
        """Set up the file handler."""
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.file_index = start_index
        super().__init__(*args, **kwargs)

    def doRollover(self):
        """Rollover the file."""
        # TODO: we need to write the logs if we logout
        if self.stream:
            self.stream.close()
            self.stream = None
            name = self.baseFilename + "." + str(self.file_index)
            with open(self.baseFilename, "rb") as fil:
                body = fil.read()
            response = self.s3_client.put_object(
                Body=body,
                Bucket=self.s3_bucket,
                Key=self.s3_prefix + name,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to save log ({self.s3_prefix + name}) to S3: {response}"
            self.file_index += 1

        super().doRollover()


def setup_logger(
    s3_client, bucket, prefix, start_index, name, log_file, level=logging.INFO
):
    """Set up a logger."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    handler = FSRotatingFileHandler(
        s3_client, bucket, prefix, start_index, log_file, maxBytes=2000000
    )
    handler.setFormatter(formatter)

    named_logger = logging.getLogger(name)
    named_logger.setLevel(level)
    named_logger.addHandler(handler)

    return named_logger


class S3Controller:
    """Represent an S3 controller."""

    # pylint: disable=too-many-statements

    def __init__(
        self,
        store: RedisStore,
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        endpoint_url_public=None,
        workspace_bucket="hypha-workspaces",
        local_log_dir="./logs",
        workspace_etc_dir="etc",
        executable_path="",
    ):
        """Set up controller."""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.minio_client = MinioClient(
            endpoint_url,
            access_key_id,
            secret_access_key,
            executable_path=executable_path,
        )
        self.endpoint_url_public = endpoint_url_public or endpoint_url
        self.store = store
        self.workspace_bucket = workspace_bucket
        self.local_log_dir = Path(local_log_dir)
        self.workspace_etc_dir = workspace_etc_dir.rstrip("/")
        event_bus = store.get_event_bus()

        s3client = self.create_client_sync()
        try:
            s3client.create_bucket(Bucket=self.workspace_bucket)
            logger.info("Bucket created: %s", self.workspace_bucket)
        except s3client.exceptions.BucketAlreadyExists:
            pass
        except s3client.exceptions.BucketAlreadyOwnedByYou:
            pass

        self.s3client = s3client

        self.minio_client.admin_user_add_sync("root", generate_password())
        store.register_public_service(self.get_s3_service())
        store.set_workspace_loader(self._workspace_loader)

        event_bus.on_local(
            "workspace_registered",
            lambda w: asyncio.ensure_future(self._setup_workspace(w)),
        )
        event_bus.on_local(
            "workspace_changed",
            lambda w: asyncio.ensure_future(self._save_workspace_config(w)),
        )
        event_bus.on_local(
            "workspace_removed",
            lambda w: asyncio.ensure_future(self._cleanup_workspace(w)),
        )
        event_bus.on_local(
            "client_registered", lambda w: asyncio.ensure_future(self._setup_client(w))
        )

        router = APIRouter()

        @router.put("/{workspace}/files/{path:path}")
        async def upload_file(
            workspace: str,
            path: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Upload file."""
            ws = await store.get_workspace(workspace)
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
            if not await store.check_permission(ws, user_info):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )
            path = safe_join(workspace, path)

            return await self._upload_file(path, request)

        @router.get("/{workspace}/files/{path:path}")
        @router.delete("/{workspace}/files/{path:path}")
        async def get_or_delete_file(
            workspace: str,
            path: str,
            request: Request,
            max_length: int = 1000,
            user_info: login_optional = Depends(login_optional),
        ):
            """Get or delete file."""
            ws = await store.get_workspace(workspace)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {workspace}",
                    },
                )
            if not await store.check_permission(ws, user_info):
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "detail": f"Permission denied: {workspace}",
                    },
                )
            path = safe_join(workspace, path)
            if request.method == "GET":
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
                return await self._delete_files(path)

        store.register_router(router)

    async def _workspace_loader(self, workspace_name, user_info):
        """Load workspace from s3 record."""
        if not await self.store.check_permission(workspace_name, user_info):
            return None
        config_path = f"{self.workspace_etc_dir}/{workspace_name}/config.json"
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

    async def _delete_files(self, path: str):
        """Delete files."""
        if path.endswith("/"):
            remove_objects_sync(self.s3client, self.workspace_bucket, path)
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                },
            )
        async with self.create_client_async() as s3_client:
            try:
                response = await s3_client.delete_object(
                    Bucket=self.workspace_bucket, Key=path
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
            region_name="EU",
        )

    def create_client_async(self, public=False):
        """Create client async."""
        return get_session().create_client(
            "s3",
            endpoint_url=self.endpoint_url_public if public else self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="EU",
        )

    async def _setup_client(self, client: dict):
        """Set up client."""
        client = ClientInfo.model_validate(client)
        user_info = client.user_info
        if user_info.id == "root" or user_info.is_anonymous:
            return
        # Make sure we created an account for the user
        try:
            await self.minio_client.admin_user_info(user_info.id)
        except Exception:  # pylint: disable=broad-except
            # Note: we don't store the credentials, it can only be regenerated
            await self.minio_client.admin_user_add(user_info.id, generate_password())

        await self.minio_client.admin_group_add(client.workspace, client.user_info.id)

    async def list_users(
        self,
    ):
        """List users."""
        path = self.workspace_etc_dir + "/"
        async with self.create_client_async() as s3_client:
            items = await list_objects_async(s3_client, self.workspace_bucket, path)
        return items

    async def _cleanup_workspace(self, workspace: dict):
        """Clean up workspace."""
        workspace = WorkspaceInfo.model_validate(workspace)
        if workspace.read_only:
            return
        # TODO: if the program shutdown unexpectedly, we need to clean it up
        # We should empty the group before removing it
        group_info = await self.minio_client.admin_group_info(workspace.name)
        # remove all the members
        await self.minio_client.admin_group_remove(
            workspace.name, group_info["members"]
        )
        # now remove the empty group
        await self.minio_client.admin_group_remove(workspace.name)

        # TODO: we will remove the files if it's not persistent
        if not workspace.persistent:
            # remove workspace etc files
            remove_objects_sync(
                self.s3client,
                self.workspace_bucket,
                f"{self.workspace_etc_dir}/{workspace.name}/",
            )
            # remove files
            remove_objects_sync(
                self.s3client, self.workspace_bucket, workspace.name + "/"
            )

    async def _setup_workspace(self, workspace: dict):
        """Set up workspace."""
        workspace = WorkspaceInfo.model_validate(workspace)
        if workspace.read_only:
            return
        # make sure we have the root user in every workspace
        await self.minio_client.admin_group_add(workspace.name, "root")
        policy_name = "policy-ws-" + workspace.name
        # policy example:
        # https://aws.amazon.com/premiumsupport/knowledge-center/iam-s3-user-specific-folder/
        await self.minio_client.admin_policy_create(
            policy_name,
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AllowUserToSeeTheBucketInTheConsole",
                        "Action": ["s3:ListAllMyBuckets", "s3:GetBucketLocation"],
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
                                "s3:prefix": ["", f"{workspace.name}/"],
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
                            "StringLike": {"s3:prefix": [f"{workspace.name}/*"]}
                        },
                    },
                    {
                        "Sid": "AllowAllS3ActionsInWorkspaceFolder",
                        "Action": ["s3:*"],
                        "Effect": "Allow",
                        "Resource": [
                            f"arn:aws:s3:::{self.workspace_bucket}/{workspace.name}/*"
                        ],
                    },
                ],
            },
        )

        await self.minio_client.admin_policy_attach(policy_name, group=workspace.name)

        # Save the workspace info
        workspace_dir = self.local_log_dir / workspace.name
        os.makedirs(workspace_dir, exist_ok=True)
        self._save_workspace_config(workspace.model_dump())

        # find out the latest log file number
        log_base_name = str(workspace_dir / "log.txt")

        items = list_objects_sync(self.s3client, self.workspace_bucket, log_base_name)
        # sort the log files based on the last number
        items = sorted(items, key=lambda file: -int(file["name"].split(".")[-1]))
        # if len(items) > 0:
        #     start_index = int(items[0]["name"].split(".")[-1]) + 1
        # else:
        #     start_index = 0

        # ready_logger = setup_logger(
        #     self.s3client,
        #     self.workspace_bucket,
        #     workspace.name,
        #     start_index,
        #     workspace.name,
        #     log_base_name,
        # )

    def _save_workspace_config(self, workspace: dict):
        """Save workspace."""
        workspace = WorkspaceInfo.model_validate(workspace)
        if workspace.read_only:
            return
        workspace = WorkspaceInfo.model_validate(workspace)
        response = self.s3client.put_object(
            Body=workspace.model_dump_json().encode("utf-8"),
            Bucket=self.workspace_bucket,
            Key=f"{self.workspace_etc_dir}/{workspace.name}/config.json",
        )
        assert (
            "ResponseMetadata" in response
            and response["ResponseMetadata"]["HTTPStatusCode"] == 200
        ), f"Failed to save workspace ({workspace.name}) config: {response}"

    async def generate_credential(self, context: dict = None):
        """Generate credential."""
        workspace = context["from"].split("/")[0]
        ws = await self.store.get_workspace(workspace)
        if ws.read_only:
            raise Exception("Permission denied: workspace is read-only")
        user_info = UserInfo.model_validate(context["user"])
        if workspace == "public" or not await self.store.check_permission(
            workspace, user_info
        ):
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
        workspace = context["from"].split("/")[0]
        path = safe_join(workspace, path)
        async with self.create_client_async() as s3_client:
            # List files in the folder
            if not path.endswith("/"):
                path += "/"
            items = await list_objects_async(
                s3_client, self.workspace_bucket, path, max_length=max_length
            )
            if len(items) == 0:
                raise Exception(f"Directory does not exists: {path}")

            return items

    async def generate_presigned_url(
        self,
        bucket_name,
        object_name,
        client_method="get_object",
        expiration=3600,
        context: dict = None,
    ):
        """Generate presigned url."""
        try:
            workspace = context["from"].split("/")[0]
            ws = await self.store.get_workspace(workspace)
            if ws.read_only and client_method != "get_object":
                raise Exception("Permission denied: workspace is read-only")
            if bucket_name != self.workspace_bucket or not object_name.startswith(
                workspace + "/"
            ):
                raise Exception(
                    f"Permission denied: bucket name must be {self.workspace_bucket} "
                    "and the object name should be prefixed with workspace name + '/'."
                )
            async with self.create_client_async(public=True) as s3_client:
                url = await s3_client.generate_presigned_url(
                    client_method,
                    Params={"Bucket": bucket_name, "Key": object_name},
                    ExpiresIn=expiration,
                )
                return url

        except ClientError as err:
            logging.error(
                err
            )  # FIXME: If we raise the error why do we need to log it first?
            raise

    def get_s3_service(self):
        """Get s3 controller."""
        return {
            "id": "s3-storage",
            "name": "S3 Storage",
            "type": "s3-storage",
            "config": {"visibility": "public", "require_context": True},
            "list_files": self.list_files,
            "generate_credential": self.generate_credential,
            "generate_presigned_url": self.generate_presigned_url,
        }
