"""Provide an s3 interface."""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from email.utils import formatdate
from pathlib import Path
from typing import Any

import botocore
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response
from starlette.datastructures import Headers
from starlette.types import Receive, Scope, Send
from hypha.core import WorkspaceInfo

from hypha.core.auth import login_optional
from hypha.minio import MinioClient
from hypha.utils import (
    generate_password,
    safe_join,
    list_objects_sync,
    list_objects_async,
    remove_objects_sync,
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
                    # Tentatively ignoring type checking failure to work around the
                    # wrong type definitions for aiofiles that come with typeshed. See
                    # https://github.com/python/typeshed/pull/4650

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
            self.s3_client.put_object(
                Body=body,
                Bucket=self.s3_bucket,
                Key=self.s3_prefix + name,
            )
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
        core_interface,
        endpoint_url=None,
        access_key_id=None,
        secret_access_key=None,
        workspace_bucket="hypha-workspaces",
        local_log_dir="./logs",
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
        self.core_interface = core_interface
        self.workspace_bucket = workspace_bucket
        self.local_log_dir = Path(local_log_dir)
        event_bus = core_interface.event_bus

        s3client = self.create_client_sync()
        try:
            s3client.create_bucket(Bucket=self.workspace_bucket)
            logger.info("Bucket created: %s", self.workspace_bucket)
        except s3client.exceptions.BucketAlreadyExists:
            pass
        except s3client.exceptions.BucketAlreadyOwnedByYou:
            pass

        self.s3client = s3client

        self.minio_client.admin_user_add(
            core_interface.root_user.id, generate_password()
        )
        core_interface.register_service_as_root(self.get_s3_service())

        event_bus.on("workspace_registered", self.setup_workspace)
        event_bus.on("workspace_removed", self.cleanup_workspace)
        event_bus.on("plugin_registered", self.setup_plugin)

        router = APIRouter()

        @router.put("/{workspace}/files/{path:path}")
        async def upload_file(
            workspace: str,
            path: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Upload file."""
            ws = core_interface.get_workspace(workspace)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {ws}",
                    },
                )
            if not core_interface.check_permission(ws, user_info):
                return JSONResponse(
                    status_code=403,
                    content={"success": False, "detail": f"Permission denied: {ws}"},
                )
            path = safe_join(workspace, path)

            async with self.create_client_async() as s3_client:
                mpu = await s3_client.create_multipart_upload(
                    Bucket=self.workspace_bucket, Key=path
                )
                parts_info = {}
                futs = (
                    []
                )  # FIXME: What does this contain? We should give a better name.
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
                        futs.append(part_fut)
                        current_chunk = b""
                # if multipart upload is activated
                if len(futs) > 0:
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
                        futs.append(part_fut)

                    parts = await asyncio.gather(*futs)
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

        @router.get("/{workspace}/files/{path:path}")
        @router.delete("/{workspace}/files/{path:path}")
        async def get_or_delete_file(
            workspace: str,
            path: str,
            request: Request,
            user_info: login_optional = Depends(login_optional),
        ):
            """Get or delete file."""
            ws = core_interface.get_workspace(workspace)
            if not ws:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "detail": f"Workspace does not exists: {ws}",
                    },
                )
            if not core_interface.check_permission(ws, user_info):
                return JSONResponse(
                    status_code=403,
                    content={"success": False, "detail": f"Permission denied: {ws}"},
                )
            path = safe_join(workspace, path)
            if request.method == "GET":
                async with self.create_client_async() as s3_client:
                    # List files in the folder
                    if path.endswith("/"):
                        items = await list_objects_async(
                            s3_client, self.workspace_bucket, path
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
                                "success": False,
                                "type": "directory",
                                "name": path.split("/")[-1],
                                "children": items,
                            },
                        )
                    # Download the file
                    try:
                        # FIXME: Commented code
                        # response = await s3_client.head_object(
                        #     Bucket=self.workspace_bucket, Key=path
                        # )
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

            if request.method == "DELETE":
                if path.endswith("/"):
                    return JSONResponse(
                        status_code=404,
                        content={
                            "success": False,
                            "detail": "Removing directory is not supported.",
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

        core_interface.register_router(router)

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

    def create_client_async(self):
        """Create client async."""
        return get_session().create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="EU",
        )

    def setup_plugin(self, plugin):
        """Set up plugin."""
        user_info = plugin.user_info
        # Make sure we created an account for the user
        try:
            self.minio_client.admin_user_info(user_info.id)
        except Exception:  # pylint: disable=broad-except
            # Note: we don't store the credentials, it can only be regenerated
            self.minio_client.admin_user_add(user_info.id, generate_password())

        self.minio_client.admin_group_add(plugin.workspace.name, plugin.user_info.id)

    def cleanup_workspace(self, workspace):
        """Clean up workspace."""
        if workspace.read_only:
            return
        # TODO: if the program shutdown unexpectedly, we need to clean it up
        # We should empty the group before removing it
        group_info = self.minio_client.admin_group_info(workspace.name)
        # remove all the members
        self.minio_client.admin_group_remove(workspace.name, group_info["members"])
        # now remove the empty group
        self.minio_client.admin_group_remove(workspace.name)

        # TODO: we will remove the files if it's not persistent
        if not workspace.persistent:
            remove_objects_sync(
                self.s3client, self.workspace_bucket, workspace.name + "/"
            )

    def setup_workspace(self, workspace: WorkspaceInfo):
        """Set up workspace."""
        if workspace.read_only:
            return
        # make sure we have the root user in every workspace
        self.minio_client.admin_group_add(
            workspace.name, self.core_interface.root_user.id
        )
        policy_name = "policy-ws-" + workspace.name
        # policy example:
        # https://aws.amazon.com/premiumsupport/knowledge-center/iam-s3-user-specific-folder/
        self.minio_client.admin_policy_add(
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

        self.minio_client.admin_policy_set(policy_name, group=workspace.name)

        # Save the workspace info
        workspace_dir = self.local_log_dir / workspace.name
        os.makedirs(workspace_dir, exist_ok=True)
        self.s3client.put_object(
            Body=workspace.json().encode("utf-8"),
            Bucket=self.workspace_bucket,
            Key=str(workspace_dir / "_workspace_config.json"),
        )

        # find out the latest log file number
        log_base_name = str(workspace_dir / "log.txt")

        items = list_objects_sync(self.s3client, self.workspace_bucket, log_base_name)
        # sort the log files based on the last number
        items = sorted(items, key=lambda file: -int(file["name"].split(".")[-1]))
        if len(items) > 0:
            start_index = int(items[0]["name"].split(".")[-1]) + 1
        else:
            start_index = 0

        ready_logger = setup_logger(
            self.s3client,
            self.workspace_bucket,
            workspace.name,
            start_index,
            workspace.name,
            log_base_name,
        )
        workspace.set_logger(ready_logger)

    def generate_credential(self):
        """Generate credential."""
        user_info = self.core_interface.current_user.get()
        workspace = self.core_interface.current_workspace.get()
        if workspace.read_only:
            raise PermissionError(f"Workspace ({workspace.name}) is read-only.")
        password = generate_password()
        self.minio_client.admin_user_add(user_info.id, password)
        # Make sure the user is in the workspace
        self.minio_client.admin_group_add(workspace.name, user_info.id)
        return {
            "endpoint_url": self.endpoint_url,
            "access_key_id": user_info.id,
            "secret_access_key": password,
            "bucket": self.workspace_bucket,
            "prefix": workspace.name + "/",  # important to have the trailing slash
        }

    async def generate_presigned_url(
        self, bucket_name, object_name, client_method="get_object", expiration=3600
    ):
        """Generate presigned url."""
        try:
            workspace = self.core_interface.current_workspace.get()
            if workspace.read_only:
                raise PermissionError(f"Workspace ({workspace.name}) is read-only.")
            if bucket_name != self.workspace_bucket or not object_name.startswith(
                workspace.name + "/"
            ):
                raise Exception(
                    f"Permission denied: bucket name must be {self.workspace_bucket} "
                    "and the object name should be prefixed with workspace.name + '/'."
                )
            async with self.create_client_async() as s3_client:
                return await s3_client.generate_presigned_url(
                    client_method,
                    Params={"Bucket": bucket_name, "Key": object_name},
                    ExpiresIn=expiration,
                )
        except ClientError as err:
            logging.error(
                err
            )  # FIXME: If we raise the error why do we need to log it first?
            raise

    def get_s3_service(self):
        """Get s3 controller."""
        return {
            "_rintf": True,
            "name": "s3-storage",
            "type": "s3-storage",
            "config": {"visibility": "public"},
            "generate_credential": self.generate_credential,
            "generate_presigned_url": self.generate_presigned_url,
        }
