import logging
import traceback
import time
import os
import sys
import uuid_utils as uuid
import random
import re
import json
import math
import asyncio
from io import BytesIO
import zipfile
from sqlalchemy import (
    event,
    Column,
    JSON,
    UniqueConstraint,
    select,
    text,
    and_,
    or_,
    update,
)
from sqlalchemy.sql import func
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
)

from datetime import datetime
from stat import S_IFREG
from stream_zip import ZIP_32, async_stream_zip
import httpx

from hrid import HRID
from hypha.utils import remove_objects_async, list_objects_async, safe_join
from hypha.utils.zenodo import ZenodoClient
from hypha.core import WorkspaceInfo
from botocore.exceptions import ClientError
from hypha.s3 import FSFileResponse
from aiobotocore.session import get_session
from botocore.config import Config
from zipfile import ZipFile
import zlib
from hypha.utils import zip_utils

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import (
    RedirectResponse,
    StreamingResponse,
    JSONResponse,
    Response,
)
from fastapi import Body
from hypha.core import (
    UserInfo,
    UserPermission,
    Artifact,
    CollectionArtifact,
)
from hypha.vectors import VectorSearchEngine
from hypha_rpc.utils import ObjectProxy
from jsonschema import validate
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from typing import Optional, Union, List, Any, Dict
import asyncio
import struct
import mimetypes
import aiohttp
from jinja2 import Template
from pydantic import BaseModel

# Logger setup
LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(LOGLEVEL)


class PartETag(BaseModel):
    part_number: int
    etag: str


class CompleteMultipartUploadRequest(BaseModel):
    upload_id: str
    parts: List[PartETag]


def make_json_safe(data):
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(v) for v in data]
    elif data == float("inf"):
        return "Infinity"
    elif data == float("-inf"):
        return "-Infinity"
    elif isinstance(data, float) and math.isnan(data):
        return "NaN"
    else:
        return data


# SQLModel model for storing artifacts
class ArtifactModel(SQLModel, table=True):  # `table=True` makes it a table model
    __tablename__ = "artifacts"

    id: str = Field(default_factory=lambda: str(uuid.uuid7()), primary_key=True)
    type: Optional[str] = Field(default=None)
    workspace: str = Field(index=True)
    parent_id: Optional[str] = Field(default=None, foreign_key="artifacts.id")
    alias: Optional[str] = Field(default=None)
    manifest: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    staging: Optional[Union[dict, list]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    download_count: float = Field(default=0.0)
    view_count: float = Field(default=0.0)
    file_count: int = Field(default=0)
    created_at: int = Field()
    created_by: Optional[str] = Field(default=None)
    last_modified: int = Field()
    versions: Optional[list] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    config: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    secrets: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Relationships
    parent: Optional["ArtifactModel"] = Relationship(
        back_populates="children",
        sa_relationship_kwargs={"remote_side": "ArtifactModel.id"},
    )
    children: List["ArtifactModel"] = Relationship(back_populates="parent")

    # Table constraints
    __table_args__ = (
        UniqueConstraint("workspace", "alias", name="_workspace_alias_uc"),
    )


def update_summary(summary, field, _metadata):
    if field.startswith("."):
        field = field[1:]
    else:
        field = f"manifest.{field}"
    if "=" in field:
        key, value_field = field.split("=", 1)
        value = get_nested_value(value_field, _metadata)
        set_nested_value(summary, key, value)
    else:
        value = get_nested_value(field, _metadata)
        set_nested_value(summary, field, value)


def model_to_dict(model):
    return {
        column.name: getattr(model, column.name) for column in model.__table__.columns
    }


def get_nested_value(field, _metadata):
    keys = field.split(".")
    value = _metadata

    for key in keys:
        value = value.get(key)
        if value is None:
            break
    return value


def set_nested_value(dictionary, field, value):
    keys = field.split(".")
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value


def convert_legacy_staging(staging):
    """Convert legacy list-based staging to new dict-based staging for backward compatibility."""
    if staging is None:
        return None
    
    if isinstance(staging, dict):
        return staging  # Already in new format
    
    if isinstance(staging, list):
        # Convert list to dict format
        new_staging = {"files": []}
        
        for item in staging:
            if isinstance(item, dict) and "_intent" in item:
                # Move intent markers to top level
                new_staging["_intent"] = item["_intent"]
            else:
                # Regular file entries go to files list
                new_staging["files"].append(item)
        
        return new_staging
    
    return staging  # Return as-is if unknown format


class ArtifactController:
    """Artifact Controller using SQLAlchemy for database backend and S3 for file storage."""

    def __init__(
        self,
        store,
        s3_controller,
        workspace_bucket="hypha-workspaces",
        artifacts_dir="artifacts",
    ):
        """Set up controller with SQLAlchemy database and S3 for file storage."""
        self.engine = store.get_sql_engine()
        self.SessionLocal = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self.s3_controller = s3_controller
        self.workspace_bucket = workspace_bucket
        self.store = store
        self._cache = store.get_redis_cache()
        self._openai_client = self.store.get_openai_client()
        self._vector_engine = VectorSearchEngine(
            store.get_redis(), store.get_cache_dir()
        )
        router = APIRouter()
        self._artifacts_dir = artifacts_dir

        # HTTP endpoint for getting an artifact
        @router.get("/{workspace}/artifacts/")
        @router.get("/{workspace}/artifacts/{artifact_alias}")
        async def get_artifact(
            workspace: str,
            artifact_alias: str = None,
            keywords: str = None,
            filters: str = None,
            silent: bool = False,
            version: str = None,
            offset: int = 0,
            mode: str = "AND",
            limit: int = 100,
            order_by: str = None,
            pagination: bool = False,
            stage: str = "false",
            no_cache: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Get artifact metadata, manifest, and config (excluding secrets) or list all top-level artifacts."""
            if artifact_alias is None:
                # List all top-level artifacts in the workspace
                try:
                    cache_key = f"workspace_artifacts:{workspace}:{offset}:{limit}:{order_by}:{keywords}:{filters}:{mode}:{stage}"
                    if not no_cache:
                        logger.info(
                            f"Responding to workspace list request ({workspace}) from cache"
                        )
                        cached_results = await self._cache.get(cache_key)
                        if cached_results:
                            return cached_results
                    if keywords:
                        keywords = keywords.split(",")
                    if filters:
                        filters = json.loads(filters)

                    # Convert stage parameter:
                    # 'true' -> True (only staged artifacts)
                    # 'false' -> False (only committed artifacts)
                    # 'all' -> 'all' (both staged and committed artifacts)
                    if stage == "true":
                        stage_param = True
                    elif stage == "false":
                        stage_param = False
                    elif stage == "all":
                        stage_param = "all"
                    else:
                        # Default to committed artifacts for any invalid value
                        stage_param = False

                    logger.info(
                        f"HTTP workspace artifacts endpoint: stage={stage}, converted to stage_param={stage_param}"
                    )

                    results = await self.list_children(
                        parent_id=None,
                        keywords=keywords,
                        filters=filters,
                        mode=mode,
                        offset=offset,
                        limit=limit,
                        order_by=order_by,
                        pagination=pagination,
                        silent=silent,
                        stage=stage_param,
                        context={"user": user_info.model_dump(), "ws": workspace},
                    )
                    await self._cache.set(cache_key, results, ttl=60)
                    return results
                except PermissionError:
                    raise HTTPException(status_code=403, detail="Permission denied")
                except SQLAlchemyError as e:
                    logger.error(f"Database error: {str(e)}")
                    raise HTTPException(
                        status_code=500, detail="Internal server error."
                    )
                except Exception as e:
                    logger.error(f"Unhandled error in http list_artifacts: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"An unexpected error occurred: {str(e)}",
                    )
            try:
                artifact = await self.read(
                    artifact_id=f"{workspace}/{artifact_alias}",
                    version=version,
                    silent=silent,
                    context={"user": user_info.model_dump(), "ws": workspace},
                )
                return artifact
            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error.")
            except ClientError as e:
                logger.error(f"S3 client error: {str(e)}")
                raise HTTPException(status_code=500, detail="File storage error.")
            except Exception as e:
                logger.error(f"Unhandled error in http get_artifact: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"An unexpected error occurred: {str(e)}"
                )

        # HTTP endpoint for listing child artifacts
        @router.get("/{workspace}/artifacts/{artifact_alias}/children")
        async def list_children(
            workspace: str,
            artifact_alias: str,
            keywords: str = None,
            filters: str = None,
            offset: int = 0,
            mode: str = "AND",
            limit: int = 100,
            order_by: str = None,
            pagination: bool = False,
            silent: bool = False,
            stage: str = "false",
            no_cache: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """List child artifacts of a specified artifact."""
            try:
                parent_id = f"{workspace}/{artifact_alias}"
                cache_key = f"artifact_children:{parent_id}:{offset}:{limit}:{order_by}:{keywords}:{filters}:{mode}:{stage}"
                if not no_cache:
                    logger.info(f"Responding to list request ({parent_id}) from cache")
                    cached_results = await self._cache.get(cache_key)
                    if cached_results:
                        return cached_results
                if keywords:
                    keywords = keywords.split(",")
                if filters:
                    filters = json.loads(filters)

                # Convert stage parameter:
                # 'true' -> True (only staged artifacts)
                # 'false' -> False (only committed artifacts)
                # 'all' -> 'all' (both staged and committed artifacts)
                if stage == "true":
                    stage_param = True
                elif stage == "false":
                    stage_param = False
                elif stage == "all":
                    stage_param = "all"
                else:
                    # Default to committed artifacts for any invalid value
                    stage_param = False

                logger.info(
                    f"HTTP list_children endpoint: stage={stage}, converted to stage_param={stage_param}"
                )

                results = await self.list_children(
                    parent_id=parent_id,
                    offset=offset,
                    limit=limit,
                    order_by=order_by,
                    keywords=keywords,
                    filters=filters,
                    mode=mode,
                    pagination=pagination,
                    silent=silent,
                    stage=stage_param,
                    context={"user": user_info.model_dump(), "ws": workspace},
                )
                await self._cache.set(cache_key, results, ttl=60)

                return results
            except KeyError:
                raise HTTPException(status_code=404, detail="Parent artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error.")
            except ClientError as e:
                logger.error(f"S3 client error: {str(e)}")
                raise HTTPException(status_code=500, detail="File storage error.")
            except Exception as e:
                logger.error(f"Unhandled exception in http list_children: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"An unexpected error occurred: {str(e)}"
                )

        @router.get("/{workspace}/artifacts/{artifact_alias}/create-zip-file")
        async def create_zip_file(
            workspace: str,
            artifact_alias: str,
            files: List[str] = Query(None, alias="file"),
            token: str = None,
            version: str = None,
            silent: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            try:
                # Validate artifact and permissions
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
                session = await self._get_session(read_only=True)
                if token:
                    user_info = await self.store.parse_user_token(token)

                try:
                    async with session.begin():
                        # Fetch artifact and check permissions
                        (
                            artifact,
                            parent_artifact,
                        ) = await self._get_artifact_with_permission(
                            user_info, artifact_id, "get_file", session
                        )
                except Exception as e:
                    raise e
                finally:
                    await session.close()

                version_index = self._get_version_index(artifact, version)
                s3_config = self._get_s3_config(artifact, parent_artifact)

                async with self._create_client_async(s3_config) as s3_client:
                    if files is None:
                        # List all files in the artifact
                        root_dir_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{version_index}",
                        )

                        async def list_all_files(dir_path=""):
                            try:
                                dir_key = f"{root_dir_key}/{dir_path}".strip("/")
                                items = await list_objects_async(
                                    s3_client,
                                    s3_config["bucket"],
                                    dir_key + "/",
                                )
                                for item in items:
                                    item_path = f"{dir_path}/{item['name']}".strip("/")
                                    if item["type"] == "file":
                                        yield item_path
                                    elif item["type"] == "directory":
                                        async for sub_item in list_all_files(item_path):
                                            yield sub_item
                            except Exception as e:
                                logger.error(f"Error listing files: {str(e)}")
                                raise HTTPException(
                                    status_code=500,
                                    detail=f"Error listing files: {str(e)}",
                                )

                        # Convert async generator to list to ensure all files are listed before streaming
                        files = [path async for path in list_all_files()]
                    else:

                        async def validate_files(files):
                            for file in files:
                                yield file

                        files = [file for file in files]

                    logger.info(f"Creating ZIP file for artifact: {artifact_alias}")

                    async def file_stream_generator(presigned_url: str):
                        """Fetch file content from presigned URL in chunks."""
                        try:
                            async with httpx.AsyncClient() as client:
                                async with client.stream(
                                    "GET", presigned_url
                                ) as response:
                                    if response.status_code != 200:
                                        logger.error(
                                            f"Failed to fetch file from URL: {presigned_url}, Status: {response.status_code}"
                                        )
                                        raise HTTPException(
                                            status_code=404,
                                            detail=f"Failed to fetch file: {presigned_url}",
                                        )
                                    async for chunk in response.aiter_bytes(
                                        1024 * 64
                                    ):  # 64KB chunks
                                        logger.debug(
                                            f"Yielding chunk of size: {len(chunk)}"
                                        )
                                        yield chunk
                        except Exception as e:
                            logger.error(f"Error fetching file stream: {str(e)}")
                            raise HTTPException(
                                status_code=500,
                                detail="Error fetching file content",
                            )

                    async def member_files():
                        """Yield file metadata and content for stream_zip."""
                        modified_at = datetime.now()
                        mode = S_IFREG | 0o600
                        total_weight = 0
                        if artifact.config and "download_weights" in artifact.config:
                            download_weights = artifact.config.get(
                                "download_weights", {}
                            )
                        else:
                            download_weights = {}

                        # Check for special "create-zip-file" download weight
                        special_zip_weight = download_weights.get("create-zip-file")
                        if special_zip_weight is not None and not silent:
                            total_weight = special_zip_weight
                            logger.info(
                                f"Using special zip download weight: {total_weight}"
                            )

                        try:
                            if isinstance(files, list):
                                file_list = files
                            else:
                                file_list = [path async for path in files]

                            for path in file_list:
                                file_key = safe_join(
                                    s3_config["prefix"],
                                    f"{artifact.id}/v{version_index}",
                                    path,
                                )
                                logger.info(f"Adding file to ZIP: {file_key}")
                                try:
                                    presigned_url = (
                                        await s3_client.generate_presigned_url(
                                            "get_object",
                                            Params={
                                                "Bucket": s3_config["bucket"],
                                                "Key": file_key,
                                            },
                                        )
                                    )
                                    # Add to total weight unless silent or special zip weight is set
                                    if not silent and special_zip_weight is None:
                                        download_weight = (
                                            download_weights.get(path) or 0
                                        )  # Default to 0 if no weight specified
                                        total_weight += download_weight

                                    yield (
                                        path,
                                        modified_at,
                                        mode,
                                        ZIP_32,
                                        file_stream_generator(presigned_url),
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error processing file {path}: {str(e)}"
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail=f"Error processing file: {path}",
                                    )

                            if total_weight > 0 and not silent:
                                logger.info(
                                    f"Bumping download count for artifact: {artifact_alias} by {total_weight}"
                                )
                                try:
                                    async with session.begin():
                                        await self._increment_stat(
                                            session,
                                            artifact.id,
                                            "download_count",
                                            increment=total_weight,
                                        )
                                        await session.commit()
                                except Exception as e:
                                    logger.error(
                                        f"Error bumping download count for artifact ({artifact_alias}): {str(e)}"
                                    )
                                    raise e
                                finally:
                                    await session.close()
                        except Exception as e:
                            raise HTTPException(
                                status_code=500, detail=f"Error listing files: {str(e)}"
                            )

                    # Return the ZIP file as a streaming response
                    return StreamingResponse(
                        async_stream_zip(member_files()),
                        media_type="application/zip",
                        headers={
                            "Content-Disposition": f"attachment; filename={artifact_alias}.zip"
                        },
                    )

            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except HTTPException as e:
                logger.error(f"HTTPException: {str(e)}")
                raise e  # Re-raise HTTPExceptions to be handled by FastAPI
            except Exception as e:
                logger.error(f"Unhandled exception in create_zip: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )

        @router.get(
            "/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path:path}"
        )
        async def get_zip_file_content(
            workspace: str,
            artifact_alias: str,
            zip_file_path: str,
            path: str = "",
            version: str = None,
            stage: bool = False,
            token: str = None,
            user_info: store.login_optional = Depends(store.login_optional),
        ) -> Response:
            """
            Serve content from a zip file stored in S3 without fully unzipping it.
            `zip_file_path` is the path to the zip file, `path` is the relative path inside the zip file.
            You can also use '/~/' to separate the zip file path and the relative path inside the zip file.
            """

            try:
                # Validate artifact and permissions
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
                session = await self._get_session(read_only=True)
                if token:
                    user_info = await self.store.parse_user_token(token)

                if stage:
                    assert (
                        version is None or version == "stage"
                    ), "You cannot specify a version when using stage mode."
                    version = "stage"

                if "/~/" in zip_file_path:
                    assert (
                        not path
                    ), "You cannot specify a path when using a zip file with a tilde."
                    zip_file_path, path = zip_file_path.split("/~/", 1)

                async with session.begin():
                    # Fetch artifact and check permissions
                    (
                        artifact,
                        parent_artifact,
                    ) = await self._get_artifact_with_permission(
                        user_info, artifact_id, "get_file", session
                    )
                    version_index = self._get_version_index(artifact, version)
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    async with self._create_client_async(s3_config) as s3_client:
                        # Full key of the ZIP file in the S3 bucket
                        s3_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{version_index}/{zip_file_path}",
                        )

                        # Fetch the ZIP file metadata from S3 (to avoid downloading the whole file)
                        try:
                            # Fetch the ZIP file metadata from S3 (to avoid downloading the whole file)
                            zip_file_metadata = await s3_client.head_object(
                                Bucket=self.workspace_bucket, Key=s3_key
                            )
                            content_length = zip_file_metadata["ContentLength"]
                        except ClientError as e:
                            # Check if the error is due to the file not being found (404)
                            if e.response["Error"]["Code"] == "404":
                                return JSONResponse(
                                    status_code=404,
                                    content={
                                        "success": False,
                                        "detail": f"ZIP file not found: {s3_key}",
                                    },
                                )
                            else:
                                # For other types of errors, raise a 500 error
                                return JSONResponse(
                                    status_code=500,
                                    content={
                                        "success": False,
                                        "detail": f"Failed to fetch file metadata: {str(e)}",
                                    },
                                )

                        # Fetch the ZIP's central directory from cache or download if not cached
                        cache_key = f"zip_tail:{self.workspace_bucket}:{s3_key}:{content_length}"
                        zip_tail = await self._cache.get(cache_key)
                        if zip_tail is None:
                            zip_tail = await zip_utils.fetch_zip_tail(
                                s3_client, self.workspace_bucket, s3_key, content_length
                            )
                            await self._cache.set(cache_key, zip_tail, ttl=60)

                        # Process zip file using the utility function
                        return await zip_utils.get_zip_file_content(
                            s3_client,
                            self.workspace_bucket,
                            s3_key,
                            path=path,
                            zip_tail=zip_tail,
                            cache_instance=self._cache,
                        )

            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except HTTPException as e:
                logger.error(f"HTTPException: {str(e)}")
                raise e  # Re-raise HTTPExceptions to be handled by FastAPI
            except Exception as e:
                logger.error(f"Unhandled exception in create_zip: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )
            finally:
                await session.close()

        # HTTP endpoint for retrieving files within an artifact
        @router.get("/{workspace}/artifacts/{artifact_alias}/files/{path:path}")
        async def get_file(
            workspace: str,
            artifact_alias: str,
            path: str = "",
            silent: bool = False,
            version: str = None,
            stage: bool = False,
            token: str = None,
            limit: int = 1000,
            use_proxy: bool = None,
            expires_in: int = 3600,
            use_local_url: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Retrieve a file within an artifact or list files in a directory."""
            try:
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
                if stage:
                    assert (
                        version is None or version == "stage"
                    ), "You cannot specify a version when using stage mode."
                    version = "stage"
                session = await self._get_session(read_only=True)
                if token:
                    user_info = await self.store.parse_user_token(token)
                async with session.begin():
                    # Fetch artifact and check permissions
                    (
                        artifact,
                        parent_artifact,
                    ) = await self._get_artifact_with_permission(
                        user_info, artifact_id, "get_file", session
                    )
                    version_index = self._get_version_index(artifact, version)
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{version_index}",
                        path,
                    )
                    if path.endswith("/") or path == "":
                        async with self._create_client_async(
                            s3_config,
                        ) as s3_client:
                            items = await list_objects_async(
                                s3_client,
                                s3_config["bucket"],
                                file_key.rstrip("/") + "/",
                                max_length=limit,
                            )
                            if not items:
                                raise HTTPException(
                                    status_code=404,
                                    detail="File or directory not found",
                                )
                            return items

                    # Increment download count unless silent
                    if not silent:
                        if artifact.config and "download_weights" in artifact.config:
                            download_weights = artifact.config.get(
                                "download_weights", {}
                            )
                        else:
                            download_weights = {}
                        download_weight = download_weights.get(path) or 0
                        if download_weight > 0:
                            await self._increment_stat(
                                session,
                                artifact.id,
                                "download_count",
                                increment=download_weight,
                            )
                        await session.commit()
                    # Use proxy based on use_proxy parameter (None means use server config)
                    if use_proxy is None:
                        use_proxy = self.s3_controller.enable_s3_proxy
                    
                    if use_proxy:
                        s3_client = self._create_client_async(s3_config)
                        return FSFileResponse(s3_client, s3_config["bucket"], file_key)
                    else:
                        async with self._create_client_async(
                            s3_config,
                        ) as s3_client:
                            # generate presigned url, then redirect
                            presigned_url = await s3_client.generate_presigned_url(
                                "get_object",
                                Params={
                                    "Bucket": s3_config["bucket"],
                                    "Key": file_key,
                                },
                                ExpiresIn=expires_in,
                            )
                            if use_local_url:
                                # Use local URL (endpoint_url is already local for S3)
                                # No replacement needed for direct S3 access
                                if use_local_url is True:
                                    pass
                                else:
                                    presigned_url = presigned_url.replace(
                                        s3_config["endpoint_url"], use_local_url
                                    )
                            elif s3_config["public_endpoint_url"]:
                                # replace the endpoint with the proxy base URL
                                presigned_url = presigned_url.replace(
                                    s3_config["endpoint_url"],
                                    s3_config["public_endpoint_url"],
                                )
                            return RedirectResponse(url=presigned_url, status_code=307)

            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except HTTPException as e:
                raise e  # Re-raise HTTPExceptions to be handled by FastAPI
            except Exception as e:
                logger.error(f"Unhandled exception in http get_file: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )
            finally:
                await session.close()

        @router.post("/{workspace}/artifacts/{artifact_alias}/create-multipart-upload")
        async def create_multipart_upload(
            workspace: str,
            artifact_alias: str,
            path: str,
            download_weight: float = 0,
            part_count: int = Query(
                ..., gt=0, description="The total number of parts for the upload."
            ),
            expires_in: int = Query(
                3600,
                description="The number of seconds for the presigned URLs to expire.",
            ),
            use_proxy: bool = None,
            use_local_url: Union[bool, str] = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """
            Initiate a multipart upload and get a list of presigned URLs for all parts.
            """
            context = {"ws": workspace, "user": user_info.model_dump()}
            artifact_id = self._validate_artifact_id(artifact_alias, context=context)

            try:
                return await self.put_file_start_multipart(
                    artifact_id=artifact_id,
                    file_path=path,
                    part_count=part_count,
                    download_weight=download_weight,
                    expires_in=expires_in,
                    use_proxy=use_proxy,
                    use_local_url=use_local_url,
                    context=context,
                )
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except Exception as e:
                logger.error(f"Failed to create multipart upload: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @router.post(
            "/{workspace}/artifacts/{artifact_alias}/complete-multipart-upload"
        )
        async def complete_multipart_upload(
            workspace: str,
            artifact_alias: str,
            data: CompleteMultipartUploadRequest = Body(...),
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """
            Complete the multipart upload and finalize the file in S3.
            The 'parts' list should contain dicts with 'PartNumber' and 'ETag'.
            """
            context = {"ws": workspace, "user": user_info.model_dump()}
            upload_id = data.upload_id
            parts = data.parts
            artifact_id = self._validate_artifact_id(artifact_alias, context=context)

            try:
                # Convert parts to the format expected by the service function
                parts_data = [
                    {"part_number": p.part_number, "etag": p.etag} for p in parts
                ]

                return await self.put_file_complete_multipart(
                    artifact_id=artifact_id,
                    upload_id=upload_id,
                    parts=parts_data,
                    context=context,
                )
            except HTTPException:
                raise
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except ClientError as e:
                logger.error(
                    f"Failed to complete multipart upload: {e.response['Error']['Message']}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to complete multipart upload: {e.response['Error']['Message']}",
                )
            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # HTTP endpoint for uploading files to an artifact
        @router.put("/{workspace}/artifacts/{artifact_alias}/files/{file_path:path}")
        async def upload_file(
            request: Request,
            workspace: str,
            artifact_alias: str,
            file_path: str,
            download_weight: float = 0,
            use_proxy: bool = None,
            use_local_url: Union[bool, str] = False,
            expires_in: int = 3600,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """
            Upload a file by streaming it directly to S3.

            This endpoint handles file uploads by streaming the request body
            directly to the S3 storage backend, avoiding local temporary storage.
            This is efficient for handling large files. After a successful upload,
            it updates the artifact's manifest.
            """
            context = {"ws": workspace, "user": user_info.model_dump()}
            try:
                artifact_id = self._validate_artifact_id(
                    artifact_alias, context=context
                )
                
                # Check if artifact is in staging mode first
                session = await self._get_session(read_only=True)
                try:
                    async with session.begin():
                        result = await session.execute(
                            select(ArtifactModel).where(self._get_artifact_id_cond(artifact_id))
                        )
                        artifact = result.scalar_one_or_none()
                        if not artifact:
                            raise KeyError(f"Artifact not found: {artifact_id}")
                        if artifact.staging is None:
                            raise ValueError("Artifact must be in staging mode to upload files")
                finally:
                    await session.close()
                
                presigned_url = await self.put_file(
                    artifact_id=artifact_id,
                    file_path=file_path,
                    download_weight=download_weight,
                    use_proxy=use_proxy,
                    use_local_url=use_local_url,
                    expires_in=expires_in,
                    context=context,
                )
                # stream the request.stream() to the url
                async with httpx.AsyncClient() as client:
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

                if not (200 == response.status_code):
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"S3 upload failed with status {response.status_code}: {response.text}",
                        headers=response.headers,
                    )
                content = json.dumps(
                    {"success": True, "message": f"File uploaded successfully"}
                )
                headers = {
                    "Content-Type": "application/json",
                    "Content-Length": str(len(content)),
                    "ETag": response.headers.get("ETag"),
                }
                return Response(content=content, headers=headers)
            except HTTPException:
                raise
            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except AssertionError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except ClientError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to upload file: {e.response['Error']['Message']}",
                )
            except Exception as e:
                logger.error(f"Unhandled exception in upload_file: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload file: {traceback.format_exc()}",
                )

        # HTTP endpoint for serving static sites
        @router.get("/{workspace}/site/{artifact_alias}/{file_path:path}")
        async def serve_site_file(
            request: Request,
            workspace: str,
            artifact_alias: str,
            file_path: Optional[str] = None,
            stage: Optional[bool] = False,
            token: Optional[str] = None,
            version: Optional[str] = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Serve files from a static site artifact with optional Jinja2 template rendering."""
            try:
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )

                if stage:
                    assert (
                        version is None or version == "stage"
                    ), "You cannot specify a version when using stage mode."
                    version = "stage"

                session = await self._get_session(read_only=True)
                if token:
                    user_info = await self.store.parse_user_token(token)

                async with session.begin():
                    # Fetch artifact and check permissions
                    (
                        artifact,
                        parent_artifact,
                    ) = await self._get_artifact_with_permission(
                        user_info, artifact_id, "get_file", session
                    )

                    # Verify this is a site artifact
                    if artifact.type != "site":
                        raise HTTPException(
                            status_code=400,
                            detail="This artifact is not a site artifact",
                        )

                    # Default to index.html if file_path is empty or just "/"
                    if not file_path or file_path == "/":
                        file_path = "index.html"

                    # Increment view count for the artifact
                    await self._increment_stat(session, artifact.id, "view_count")
                    await session.commit()

                    # Get artifact configuration
                    config = artifact.config or {}
                    templates = config.get("templates", [])
                    template_engine = config.get("template_engine")
                    custom_headers = config.get("headers", {})

                    version_index = self._get_version_index(artifact, version)
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{version_index}",
                        file_path,
                    )

                    # Check if file needs template rendering
                    if (
                        template_engine
                        and template_engine == "jinja2"
                        and file_path in templates
                    ):

                        # Get file content for template rendering
                        async with self._create_client_async(s3_config) as s3_client:
                            try:
                                obj_info = await s3_client.get_object(
                                    Bucket=s3_config["bucket"], Key=file_key
                                )
                                content = await obj_info["Body"].read()
                                content = content.decode("utf-8")
                            except ClientError:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"File not found: {file_path}",
                                )

                        # Prepare Jinja2 template context
                        artifact_data = self._generate_artifact_data(
                            artifact, parent_artifact
                        )

                        # Exclude secrets from config for template context
                        config_for_template = {
                            k: v for k, v in config.items() if k != "secrets"
                        }

                        template_context = {
                            "PUBLIC_BASE_URL": self.store.public_base_url,
                            "LOCAL_BASE_URL": self.store.local_base_url,
                            "BASE_URL": f"/{workspace}/site/{artifact_alias}/",
                            "VIEW_COUNT": artifact_data.get("view_count", 0),
                            "DOWNLOAD_COUNT": artifact_data.get("download_count", 0),
                            "MANIFEST": artifact_data.get("manifest", {}),
                            "CONFIG": config_for_template,
                            "WORKSPACE": workspace,
                            "USER": (
                                user_info.model_dump()
                                if user_info and not user_info.is_anonymous
                                else None
                            ),
                        }

                        # Render template
                        try:
                            rendered = Template(content).render(**template_context)
                        except Exception as e:
                            logger.error(f"Template rendering error: {e}")
                            raise HTTPException(
                                status_code=500,
                                detail=f"Template rendering failed: {str(e)}",
                            )

                        # Get MIME type
                        mime_type, _ = mimetypes.guess_type(file_path)
                        mime_type = mime_type or "text/html"

                        # Prepare headers
                        response_headers = {}

                        # Add custom headers from config
                        if custom_headers:
                            for key, value in custom_headers.items():
                                # Special handling for CORS origin
                                if (
                                    key == "Access-Control-Allow-Origin"
                                    and value == "*"
                                ):
                                    response_headers[key] = value
                                else:
                                    response_headers[key] = value

                        return Response(
                            content=rendered,
                            media_type=mime_type,
                            headers=response_headers,
                        )

                    else:
                        # Regular file (non-templated) - stream through proxy
                        try:
                            # Use S3 proxy streaming for proper content delivery
                            s3_client = self._create_client_async(s3_config)

                            # Get MIME type for proper content-type header
                            mime_type, _ = mimetypes.guess_type(file_path)
                            mime_type = mime_type or "application/octet-stream"

                            # Prepare custom headers
                            response_headers = {}
                            if custom_headers:
                                for key, value in custom_headers.items():
                                    # Special handling for CORS origin
                                    if (
                                        key == "Access-Control-Allow-Origin"
                                        and value == "*"
                                    ):
                                        response_headers[key] = value
                                    else:
                                        response_headers[key] = value

                            # Set content-type header
                            response_headers["Content-Type"] = mime_type

                            # Return streaming response using FSFileResponse for proper range support
                            return FSFileResponse(
                                s3_client,
                                s3_config["bucket"],
                                file_key,
                                media_type=mime_type,
                                headers=response_headers,
                            )

                        except ClientError:
                            raise HTTPException(
                                status_code=404, detail=f"File not found: {file_path}"
                            )

            except HTTPException:
                raise
            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except Exception as e:
                logger.error(f"Unhandled exception in serve_site_file: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Internal server error: {str(e)}"
                )
            finally:
                await session.close()

        # Register public services
        artifact_service = self.get_artifact_service()
        self.store.register_public_service(artifact_service)
        self.store.set_artifact_manager(self)
        self.store.register_router(router)

    async def _execute_with_retry(
        self,
        session: AsyncSession,
        query,
        description: str,
        max_retries: int = 3,
        base_delay: float = 0.05,
        max_delay: float = 1.0,
    ):
        """Execute an SQLAlchemy query with retry logic for database lock errors."""
        for attempt in range(max_retries):
            try:
                result = await session.execute(query)
                return result  # Success
            except OperationalError as e:
                if "database is locked" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (2**attempt) + random.uniform(0, base_delay),
                            max_delay,
                        )
                        logger.warning(
                            f"Database locked during {description} (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Database locked after {max_retries} attempts during {description}. Giving up."
                        )
                        raise e  # Re-raise the final lock error
                else:
                    raise e  # Re-raise other OperationalErrors
            except Exception as e:
                raise e  # Re-raise any other exceptions

        # Should not be reached if logic is correct, but as a safeguard:
        raise RuntimeError(
            f"Failed to execute query for '{description}' after {max_retries} retries."
        )

    async def init_db(self):
        """Initialize the database and create tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(ArtifactModel.metadata.create_all)
            logger.info("Database tables created successfully.")

    async def _get_session(self, read_only=False):
        """Return an SQLAlchemy async session. If read_only=True, ensure no modifications are allowed."""
        session = self.SessionLocal()
        if read_only:
            sync_session = session.sync_session

            @event.listens_for(sync_session, "before_flush")
            def prevent_flush(session, flush_context, instances):
                raise RuntimeError("This session is read-only.")

            @event.listens_for(sync_session, "after_transaction_end")
            def end_transaction(session, transaction):
                if not transaction._parent:
                    session.rollback()

        return session

    def _get_artifact_id_cond(self, artifact_id):
        """Get the SQL condition for an artifact ID."""
        if "/" in artifact_id:
            assert (
                len(artifact_id.split("/")) == 2
            ), "Invalid artifact ID format, it should be `workspace/alias`."
            ws, alias = artifact_id.split("/")
            return and_(
                ArtifactModel.workspace == ws,
                ArtifactModel.alias == alias,
            )
        return ArtifactModel.id == artifact_id

    async def _get_artifact(self, session, artifact_id):
        """
        Retrieve an artifact by its unique ID or by parent ID and alias.
        """
        query = select(ArtifactModel).where(
            self._get_artifact_id_cond(artifact_id),
        )
        result = await self._execute_with_retry(
            session, query, description=f"_get_artifact for '{artifact_id}'"
        )
        artifact = result.scalar_one_or_none()
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        return artifact

    async def _get_artifact_with_parent(self, session, artifact_id):
        """
        Retrieve an artifact and its parent artifact in a single SQL query.
        """
        query = select(ArtifactModel).where(self._get_artifact_id_cond(artifact_id))

        try:
            result = await self._execute_with_retry(
                session,
                query,
                description=f"_get_artifact_with_parent for '{artifact_id}'",
            )
            artifact = result.scalar_one_or_none()
            if not artifact:
                # Add debug logging
                logger.debug(f"Failed to find artifact with ID '{artifact_id}'")
                # Try to get the workspace and alias separately for better error message
                if "/" in artifact_id:
                    ws, alias = artifact_id.split("/")
                    query = select(ArtifactModel).where(
                        and_(
                            ArtifactModel.workspace == ws,
                            ArtifactModel.alias == alias,
                        )
                    )
                    result = await self._execute_with_retry(
                        session,
                        query,
                        description=f"_get_artifact_with_parent existence check for '{artifact_id}'",
                    )
                    if result.scalar_one_or_none():
                        raise KeyError(
                            f"Artifact exists but failed to retrieve with ID '{artifact_id}', this may be a race condition."
                        )
                raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")

            parent_artifact = None
            if artifact and artifact.parent_id:
                parent_query = select(ArtifactModel).where(
                    self._get_artifact_id_cond(artifact.parent_id)
                )
                parent_result = await self._execute_with_retry(
                    session,
                    parent_query,
                    description=f"_get_artifact_with_parent parent query for '{artifact.parent_id}'",
                )
                parent_artifact = parent_result.scalar_one_or_none()
                if not parent_artifact:
                    logger.warning(
                        f"Parent artifact {artifact.parent_id} not found for {artifact_id}"
                    )

            return artifact, parent_artifact

        except Exception as e:
            raise

    def _generate_artifact_data(
        self, artifact: ArtifactModel, parent_artifact: ArtifactModel = None
    ):
        artifact_data = model_to_dict(artifact)
        if parent_artifact:
            artifact_data["parent_id"] = (
                f"{parent_artifact.workspace}/{parent_artifact.alias}"
            )
        artifact_data["id"] = f"{artifact.workspace}/{artifact.alias}"
        artifact_data["_id"] = artifact.id
        # Exclude 'secrets' from artifact_data to prevent exposure
        artifact_data.pop("secrets", None)
        
        # Convert dict-based staging back to list format for API backward compatibility
        if artifact_data.get("staging") and isinstance(artifact_data["staging"], dict):
            staging_dict = artifact_data["staging"]
            staging_list = staging_dict.get("files", [])
            
            # Add intent markers as list items for backward compatibility
            if "_intent" in staging_dict:
                staging_list.append({"_intent": staging_dict["_intent"]})
            
            artifact_data["staging"] = staging_list
        
        return artifact_data

    def _expand_permission(self, permission):
        """Get the list of operations allowed by the permission code."""
        if isinstance(permission, str):
            permission_map = {
                "n": [],
                "l": [
                    "list",
                ],
                "l+": ["list", "create", "edit"],
                "lv": ["list", "list_vectors"],
                "lv+": [
                    "list",
                    "list_vectors",
                    "create",
                    "commit",
                    "add_vectors",
                ],
                "lf": ["list", "list_files"],
                "lf+": ["list", "list_files", "create", "commit", "put_file"],
                "r": [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "search_vectors",
                    "get_vector",
                ],
                "r+": [
                    "read",
                    "get_file",
                    "put_file",
                    "list_files",
                    "list",
                    "search_vectors",
                    "get_vector",
                    "create",
                    "add_vectors",
                ],
                "rw": [
                    "read",
                    "get_file",
                    "get_vector",
                    "search_vectors",
                    "list_files",
                    "list_vectors",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "add_vectors",
                    "remove_file",
                    "remove_vectors",
                ],
                "rd+": [
                    "read",
                    "get_file",
                    "get_vector",
                    "search_vectors",
                    "list_files",
                    "list_vectors",
                    "list",
                    "edit",
                    # "commit", can create but not commit
                    "put_file",
                    "add_vectors",
                    "remove_file",
                    "remove_vectors",
                    "create",
                ],
                "rw+": [
                    "read",
                    "get_file",
                    "get_vector",
                    "search_vectors",
                    "list_files",
                    "list_vectors",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "add_vectors",
                    "remove_file",
                    "remove_vectors",
                    "create",
                ],
                "*": [
                    "read",
                    "get_file",
                    "get_vector",
                    "search_vectors",
                    "list_files",
                    "list_vectors",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "add_vectors",
                    "remove_file",
                    "remove_vectors",
                    "create",
                    "reset_stats",
                    "publish",
                    "delete",
                ],
            }
            return permission_map.get(permission, [])
        else:
            return permission  # Assume it's already a list

    async def _check_permissions(self, artifact, user_info, operation):
        """Check whether a user has permission to perform an operation on an artifact."""
        # Check specific artifact permissions
        permissions = {}
        if artifact and artifact.config:
            permissions = artifact.config.get("permissions", {})

        # Specific user permission
        if user_info.id in permissions:
            if operation in self._expand_permission(permissions[user_info.id]):
                return True

        # Check based on user's authentication status
        if not user_info.is_anonymous and "@" in permissions:
            if operation in self._expand_permission(permissions["@"]):
                return True

        # Global permission for all users
        if "*" in permissions:
            if operation in self._expand_permission(permissions["*"]):
                return True
        return False

    async def _get_artifact_with_permission(
        self,
        user_info: UserInfo,
        artifact_id: str,
        operation: Union[str, List[str]],
        session: AsyncSession,
    ):
        """
        Check whether a user has the required permission to perform an operation on an artifact.
        Fetches artifact and parent artifacts, checking permissions hierarchically.
        If multiple operations are provided, all must be satisfied.
        """
        # Map operation to permission level
        operation_map = {
            "list": UserPermission.read,
            "read": UserPermission.read,
            "get_vector": UserPermission.read,
            "get_file": UserPermission.read,
            "list_files": UserPermission.read,
            "list_vectors": UserPermission.read,
            "search_vectors": UserPermission.read,
            "create": UserPermission.read_write,
            "edit": UserPermission.read_write,
            "commit": UserPermission.read_write,
            "add_vectors": UserPermission.read_write,
            "put_file": UserPermission.read_write,
            "remove_vectors": UserPermission.read_write,
            "remove_file": UserPermission.read_write,
            "set_secret": UserPermission.read_write,
            "set_download_weight": UserPermission.read_write,
            "delete": UserPermission.admin,
            "reset_stats": UserPermission.admin,
            "publish": UserPermission.admin,
            "get_secret": UserPermission.admin,
        }

        # Handle both single operation and list of operations
        operations = [operation] if isinstance(operation, str) else operation

        # Check all operations are supported
        for op in operations:
            if op not in operation_map:
                raise ValueError(f"Operation '{op}' is not supported.")

        # Get highest required permission level
        required_perms = [operation_map[op] for op in operations]
        highest_required_perm = max(required_perms)

        # Fetch the primary artifact
        artifact, parent_artifact = await self._get_artifact_with_parent(
            session, artifact_id
        )
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")

        # Check permissions on the artifact itself - all operations must be allowed
        artifact_permissions_satisfied = True
        for op in operations:
            if not await self._check_permissions(artifact, user_info, op):
                artifact_permissions_satisfied = False
                break
        if artifact_permissions_satisfied:
            return artifact, parent_artifact

        # Check permissions on parent artifact (collection) if it exists - all operations must be allowed
        if parent_artifact:
            parent_permissions_satisfied = True
            for op in operations:
                if not await self._check_permissions(parent_artifact, user_info, op):
                    parent_permissions_satisfied = False
                    break
            if parent_permissions_satisfied:
                return artifact, parent_artifact

        # Finally, check workspace-level permission using the highest required permission
        if user_info.check_permission(artifact.workspace, highest_required_perm):
            return artifact, parent_artifact

        operation_str = (
            operation if isinstance(operation, str) else f"operations {operations}"
        )
        raise PermissionError(
            f"User does not have permission to perform the operation '{operation_str}' on the artifact."
        )

    async def _increment_stat(self, session, artifact_id, field, increment=1.0):
        """
        Increment the specified statistic field for an artifact.
        :param session: Database session.
        :param artifact_id: ID of the artifact.
        :param field: 'view_count' or 'download_count'.
        :param increment: Amount to increment.
        """
        if field not in ["view_count", "download_count"]:
            raise ValueError("Field must be 'view_count' or 'download_count'.")
        stmt = (
            update(ArtifactModel)
            .where(self._get_artifact_id_cond(artifact_id))
            .values({field: getattr(ArtifactModel, field) + increment})
            .execution_options(synchronize_session="fetch")
        )
        await session.execute(stmt)

    def _get_zenodo_client(self, artifact, parent_artifact=None, publish_to=None):
        """
        Get a Zenodo client using credentials from the artifact's or parent artifact's secrets.
        """
        # merge secrets from parent and artifact
        if parent_artifact and parent_artifact.secrets:
            secrets = parent_artifact.secrets.copy()
        else:
            secrets = {}
        secrets.update(artifact.secrets or {})

        if publish_to == "zenodo":
            zenodo_token = secrets.get("ZENODO_ACCESS_TOKEN")
            if not zenodo_token:
                raise ValueError("Zenodo access token is not configured in secrets.")
            zenodo_client = ZenodoClient(zenodo_token, "https://zenodo.org")
        elif publish_to == "sandbox_zenodo":
            zenodo_token = secrets.get("SANDBOX_ZENODO_ACCESS_TOKEN")
            if not zenodo_token:
                raise ValueError(
                    "Sandbox Zenodo access token is not configured in secrets."
                )
            zenodo_client = ZenodoClient(zenodo_token, "https://sandbox.zenodo.org")
        else:
            raise ValueError(f"Publishing to '{publish_to}' is not supported.")

        return zenodo_client

    def _get_s3_config(self, artifact=None, parent_artifact=None, workspace=None):
        """
        Get S3 configuration using credentials from the artifact's or parent artifact's secrets.
        If artifact and parent_artifact are None, returns the default S3 configuration.
        When both artifact and parent_artifact are None, workspace must be provided.

        Args:
            artifact: The artifact to get S3 config for
            parent_artifact: The parent artifact to inherit S3 config from
            workspace: The workspace to use for default config when artifact is None

        Returns:
            Dictionary with S3 configuration
        """
        # If artifact is None, use default config
        if artifact is None:
            assert (
                workspace is not None
            ), "Workspace must be provided when artifact is None"
            default_config = {
                "endpoint_url": self.s3_controller.endpoint_url,
                "access_key_id": self.s3_controller.access_key_id,
                "secret_access_key": self.s3_controller.secret_access_key,
                "region_name": self.s3_controller.region_name,
                "bucket": self.workspace_bucket,
                "prefix": safe_join(workspace, self._artifacts_dir),
                "public_endpoint_url": None,
            }
            if self.s3_controller.enable_s3_proxy:
                default_config["public_endpoint_url"] = (
                    f"{self.store.public_base_url}/s3"
                )
            return default_config

        # merge secrets from parent and artifact
        if parent_artifact and parent_artifact.secrets:
            secrets = parent_artifact.secrets.copy()
        else:
            secrets = {}
        if artifact.secrets:
            secrets.update(artifact.secrets)

        endpoint_url = secrets.get("S3_ENDPOINT_URL")
        access_key_id = secrets.get("S3_ACCESS_KEY_ID")
        secret_access_key = secrets.get("S3_SECRET_ACCESS_KEY")
        region_name = secrets.get("S3_REGION_NAME")
        bucket = secrets.get("S3_BUCKET")
        prefix = secrets.get("S3_PREFIX")
        public_endpoint_url = secrets.get("S3_PUBLIC_ENDPOINT_URL")

        if not access_key_id or not secret_access_key:
            s3_config = {
                "endpoint_url": self.s3_controller.endpoint_url,
                "access_key_id": self.s3_controller.access_key_id,
                "secret_access_key": self.s3_controller.secret_access_key,
                "region_name": self.s3_controller.region_name,
                "bucket": self.workspace_bucket,
                "prefix": safe_join(artifact.workspace, self._artifacts_dir),
                "public_endpoint_url": None,
            }
            if self.s3_controller.enable_s3_proxy:
                s3_config["public_endpoint_url"] = f"{self.store.public_base_url}/s3"
            return s3_config
        else:
            return {
                "endpoint_url": endpoint_url,
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
                "region_name": region_name,
                "bucket": bucket or self.workspace_bucket,
                "prefix": prefix or "",
                "public_endpoint_url": public_endpoint_url,
            }

    def _create_client_async(self, s3_config):
        """
        Create an S3 client using credentials from secrets.
        """
        endpoint_url = s3_config.get("endpoint_url")
        access_key_id = s3_config.get("access_key_id")
        secret_access_key = s3_config.get("secret_access_key")
        region_name = s3_config.get("region_name")

        assert (
            endpoint_url and access_key_id and secret_access_key and region_name
        ), "S3 credentials are not configured."

        return get_session().create_client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
            config=Config(connect_timeout=60, read_timeout=300),
        )

    async def _count_files_in_prefix(self, s3_client, bucket_name, prefix):
        paginator = s3_client.get_paginator("list_objects_v2")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        operation_parameters = {
            "Bucket": bucket_name,
            "Prefix": prefix,
        }
        file_count = 0

        async for page in paginator.paginate(**operation_parameters):
            if "Contents" in page:
                file_count += len(page["Contents"])

        return file_count

    async def _generate_candidate_aliases(
        self,
        alias_pattern: str = None,
        id_parts: dict = None,
        additional_parts: dict = None,
        max_candidates: int = 10,
    ):
        """
        Generate a list of candidate aliases based on the alias pattern and id_parts.
        """
        if alias_pattern is None:
            hrid = HRID(delimeter="-", hridfmt=("adjective", "noun", "verb", "adverb"))
            return [hrid.generate() for _ in range(max_candidates)]
        placeholder_pattern = re.compile(r"\{(\w+)\}")
        placeholders = placeholder_pattern.findall(alias_pattern)
        if not placeholders:
            return [alias_pattern]

        if additional_parts:
            id_parts = {**id_parts, **additional_parts}

        candidates = set()
        attempts = 0
        while len(candidates) < max_candidates and attempts < max_candidates * 10:
            attempts += 1
            alias = alias_pattern
            for placeholder in placeholders:
                if placeholder in id_parts and isinstance(id_parts[placeholder], list):
                    value = random.choice(id_parts[placeholder])
                elif placeholder in id_parts:
                    value = id_parts[placeholder]
                else:
                    value = str(uuid.uuid4())
                alias = alias.replace(f"{{{placeholder}}}", value)
            candidates.add(alias)
        return list(candidates)

    async def _batch_alias_exists(self, session, workspace, aliases):
        """
        Check which aliases already exist under the given parent artifact.
        """
        query = select(ArtifactModel.alias).where(
            and_(
                ArtifactModel.workspace == workspace,
                ArtifactModel.alias.in_(aliases)
            )
        )
        result = await self._execute_with_retry(
            session,
            query,
            description=f"_batch_alias_exists for workspace '{workspace}'",
        )
        existing_aliases = set(row[0] for row in result.fetchall())
        return existing_aliases

    async def _generate_unique_alias(
        self,
        session,
        workspace,
        alias_pattern=None,
        id_parts=None,
        additional_parts=None,
        max_attempts=10,
    ):
        """
        Generate a unique alias under the given parent artifact.
        """
        for _ in range(max_attempts):
            candidates = await self._generate_candidate_aliases(
                alias_pattern, id_parts, additional_parts
            )
            existing_aliases = await self._batch_alias_exists(
                session, workspace, candidates
            )
            unique_aliases = set(candidates) - existing_aliases
            if unique_aliases:
                return unique_aliases.pop()
        raise RuntimeError(
            "Could not generate a unique alias within the maximum attempts."
        )

    async def _save_version_to_s3(self, version_index: int, artifact, s3_config):
        """
        Save the staged version to S3 and update the artifact's manifest.
        """
        assert version_index >= 0, "Version index must be non-negative."
        async with self._create_client_async(s3_config) as s3_client:
            version_key = safe_join(
                s3_config["prefix"],
                artifact.id,
                f"v{version_index}.json",
            )
            await s3_client.put_object(
                Bucket=s3_config["bucket"],
                Key=version_key,
                Body=json.dumps(model_to_dict(artifact)),
            )

        # Check if we're using a custom S3 config (not the default one)
        # and create a backup in the default S3 bucket
        is_default_s3 = (
            s3_config["bucket"] == self.workspace_bucket
            and s3_config["endpoint_url"] == self.s3_controller.endpoint_url
            and s3_config["access_key_id"] == self.s3_controller.access_key_id
        )

        if not is_default_s3:
            # Get default S3 config using the workspace from the artifact
            default_s3_config = self._get_s3_config(workspace=artifact.workspace)

            # Save backup to default S3
            async with self._create_client_async(default_s3_config) as backup_s3_client:
                backup_version_key = safe_join(
                    default_s3_config["prefix"],
                    artifact.id,
                    f"v{version_index}.json",
                )

                # Just save the full artifact data as-is
                await backup_s3_client.put_object(
                    Bucket=default_s3_config["bucket"],
                    Key=backup_version_key,
                    Body=json.dumps(model_to_dict(artifact)),
                )

    async def _load_version_from_s3(
        self,
        artifact,
        parent_artifact,
        version_index: int,
        s3_config,
        include_secrets=False,
    ):
        """
        Load the version from S3.
        """
        assert version_index >= 0, "Version index must be non-negative."
        s3_config = self._get_s3_config(artifact, parent_artifact)
        version_key = safe_join(
            s3_config["prefix"],
            artifact.id,
            f"v{version_index}.json",
        )
        async with self._create_client_async(s3_config) as s3_client:
            response = await s3_client.get_object(
                Bucket=s3_config["bucket"], Key=version_key
            )
            body = await response["Body"].read()
            data = json.loads(body)
            if not include_secrets:
                data.pop("secrets", None)
            return data

    def _get_version_index(self, artifact, version):
        """Get the version index based on the version string."""
        if version is None:
            version_index = max(0, len(artifact.versions) - 1)
        elif version == "latest":
            version_index = len(artifact.versions) - 1
            return version_index
        elif version == "stage":
            version_index = len(artifact.versions)
        elif isinstance(version, str):
            # find the version index
            version_index = -1
            for i, v in enumerate(artifact.versions):
                if v["version"] == version:
                    version_index = i
                    break
            if version_index < 0:
                if version.isdigit():
                    version_index = int(version)
                raise ValueError(f"Artifact version '{version}' does not exist.")
        elif isinstance(version, int):
            assert version >= 0, "Version must be non-negative."
            version_index = version
        else:
            raise ValueError("Version must be a string or an integer.")

        return version_index

    async def _delete_version_files_from_s3(
        self, version_index: int, artifact, s3_config, delete_files=False
    ):
        """
        Delete the staged version from S3.
        """
        assert version_index >= 0, "Version index must be non-negative."
        async with self._create_client_async(s3_config) as s3_client:
            version_key = safe_join(
                s3_config["prefix"],
                artifact.id,
                f"v{version_index}",
            )
            await s3_client.delete_object(
                Bucket=s3_config["bucket"], Key=version_key + ".json"
            )
            if delete_files:
                await remove_objects_async(
                    s3_client,
                    s3_config["bucket"],
                    version_key + "/",
                )

    async def create(
        self,
        alias: str = None,
        workspace: str = None,
        parent_id: str = None,
        manifest: dict = None,
        type="generic",
        config: dict = None,
        secrets: dict = None,
        version: str = None,
        comment: str = None,
        overwrite: bool = False,
        stage: bool = False,  # Add stage parameter
        context: dict = None,
    ):
        """Create a new artifact and store its manifest in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        user_info = UserInfo.model_validate(context["user"])

        # Validate the manifest
        if type == "collection":
            CollectionArtifact.model_validate(manifest)
        elif type == "generic":
            Artifact.model_validate(manifest)

        if isinstance(manifest, ObjectProxy):
            manifest = ObjectProxy.toDict(manifest)

        manifest = manifest and make_json_safe(manifest)
        config = config and make_json_safe(config)

        # for backward compatibility, if version is "stage", set stage to True and version to None
        if version == "stage":
            stage = True
            version = None

        if alias:
            alias = alias.strip()
            if "/" in alias:
                ws, alias = alias.split("/")
                if workspace and ws != workspace:
                    raise ValueError(
                        "Workspace must match the alias workspace, if provided."
                    )
                workspace = ws
        created_at = int(time.time())
        session = await self._get_session()
        try:
            async with session.begin():
                parent_artifact = None
                if parent_id:
                    parent_id = self._validate_artifact_id(parent_id, context)
                    if stage:
                        parent_artifact, _ = await self._get_artifact_with_permission(
                            user_info, parent_id, "create", session
                        )
                    else:
                        # if not staging, we need to check if the user has both create and commit permissions
                        parent_artifact, _ = await self._get_artifact_with_permission(
                            user_info, parent_id, ["create", "commit"], session
                        )
                    parent_id = parent_artifact.id
                    if workspace:
                        assert (
                            workspace == parent_artifact.workspace
                        ), "Workspace must match parent artifact's workspace."
                    workspace = parent_artifact.workspace
                    if not parent_artifact.manifest:
                        raise ValueError(
                            f"Parent artifact with ID '{parent_id}' must be committed before creating a child artifact."
                        )
                else:
                    workspace = workspace or context["ws"]
                    if not user_info.check_permission(
                        workspace, UserPermission.read_write
                    ):
                        raise PermissionError(
                            f"User does not have permission to create an orphan artifact in the workspace '{workspace}'."
                        )

                winfo = await self.store.get_workspace_info(workspace, load=True)
                if not winfo:
                    raise ValueError(f"Workspace '{workspace}' does not exist.")
                if not winfo.persistent:
                    raise PermissionError(
                        f"Cannot create artifact in a non-persistent workspace `{workspace}`, you may need to login to a persistent workspace."
                    )

                config = config or {}
                id = str(uuid.uuid7())
                if alias and "{" in alias and "}" in alias:
                    id_parts = {}
                    additional_parts = {
                        "uuid": str(uuid.uuid7()),
                        "timestamp": str(int(time.time())),
                        "user_id": user_info.id,
                    }
                    publish_to = config.get("publish_to")
                    if publish_to:
                        zenodo_client = self._get_zenodo_client(
                            parent_artifact, publish_to=publish_to
                        )
                        deposition_info = await zenodo_client.create_deposition()
                        additional_parts["zenodo_id"] = str(deposition_info["id"])
                        additional_parts["zenodo_conceptrecid"] = str(
                            deposition_info["conceptrecid"]
                        )
                        config["zenodo"] = deposition_info

                    if publish_to not in ["zenodo", "sandbox_zenodo"]:
                        assert (
                            "{zenodo_id}" not in alias
                        ), "Alias cannot contain the '{zenodo_id}' placeholder, set publish_to to 'zenodo' or 'sandbox_zenodo'."
                        assert (
                            "{zenodo_conceptrecid}" not in alias
                        ), "Alias cannot contain the '{zenodo_conceptrecid}' placeholder, set publish_to to 'zenodo' or 'sandbox_zenodo'."

                    if parent_artifact and parent_artifact.config:
                        id_parts.update(parent_artifact.config.get("id_parts", {}))

                    alias = await self._generate_unique_alias(
                        session,
                        workspace,
                        alias,
                        id_parts,
                        additional_parts=additional_parts,
                    )
                elif alias is None:
                    alias = await self._generate_unique_alias(
                        session,
                        workspace,
                    )
                else:
                    if re.match(
                        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                        alias,
                    ):
                        raise ValueError(
                            "Alias should be a human readable string, it cannot be a UUID."
                        )
                    try:
                        # Check if the alias already exists
                        existing_artifact = await self._get_artifact(
                            session, f"{workspace}/{alias}"
                        )
                        if parent_id != existing_artifact.parent_id and not overwrite:
                            raise FileExistsError(
                                f"Artifact with alias '{alias}' already exists under a different parent artifact, please choose a different alias or remove the existing artifact (ID: {existing_artifact.workspace}/{existing_artifact.alias})"
                            )
                        elif not overwrite:
                            raise FileExistsError(
                                f"Artifact with alias '{alias}' already exists, please choose a different alias or remove the existing artifact (ID: {existing_artifact.workspace}/{existing_artifact.alias})."
                            )

                        # Check if overwriting a collection with children
                        if overwrite and existing_artifact.type == "collection":
                            children_count_query = select(func.count()).where(
                                ArtifactModel.parent_id == existing_artifact.id
                            )
                            children_result = await self._execute_with_retry(
                                session,
                                children_count_query,
                                description=f"check children count for overwrite of '{workspace}/{alias}'",
                            )
                            children_count = children_result.scalar()
                            if children_count > 0:
                                raise ValueError(
                                    f"Cannot overwrite collection '{existing_artifact.workspace}/{existing_artifact.alias}' as it has {children_count} child artifacts. Remove the children first or use a different alias."
                                )

                        id = existing_artifact.id
                    except KeyError:
                        overwrite = False

                parent_permissions = (
                    parent_artifact.config["permissions"] if parent_artifact else {}
                )
                permissions = config.get("permissions", {}) if config else {}
                permissions[user_info.id] = "*"
                permissions.update(parent_permissions)
                config["permissions"] = permissions

                # Determine if we should be in staging mode

                versions = []
                if not stage:  # Only create initial version if not in staging mode
                    if version in [
                        None,
                        "new",
                        "stage",
                    ]:  # Treat 'stage' as None if stage=False
                        version = "v0"  # Always start with v0 for new artifacts
                    if parent_artifact and not await self._check_permissions(
                        parent_artifact, user_info, "commit"
                    ):
                        raise PermissionError(
                            f"User does not have permission to commit an artifact in the collection '{parent_artifact.alias}'."
                        )
                    comment = comment or "Initial version"
                    versions.append(
                        {
                            "version": version,
                            "comment": comment,
                            "created_at": int(time.time()),
                        }
                    )

                assert manifest, "Manifest must be provided."
                
                if stage:
                    # When staging on create, save manifest/config in both database and staging
                    # The database version is for listing/searching, staging is for the draft
                    staging_dict = {
                        "manifest": manifest,
                        "config": config,
                        "secrets": secrets,
                        "type": type,
                        "files": [],
                        "_intent": "new_artifact"  # New artifact in staging mode
                    }
                    
                    new_artifact = ArtifactModel(
                        id=id,
                        workspace=workspace,
                        parent_id=parent_id,
                        alias=alias,
                        staging=staging_dict,
                        manifest=manifest,  # Save manifest for listing/searching
                        created_by=user_info.id,
                        created_at=created_at,
                        last_modified=int(time.time()),
                        config=config,  # Save config for permissions
                        secrets=secrets,
                        versions=[],  # No versions when staging
                        type=type,
                    )
                else:
                    # When not staging, save manifest/config directly to database
                    new_artifact = ArtifactModel(
                        id=id,
                        workspace=workspace,
                        parent_id=parent_id,
                        alias=alias,
                        staging=None,
                        manifest=manifest,
                        created_by=user_info.id,
                        created_at=created_at,
                        last_modified=int(time.time()),
                        config=config,
                        secrets=secrets,
                        versions=versions,
                        type=type,
                    )

                version_index = (
                    len(versions) if stage else 0
                )  # Use appropriate version index
                if overwrite:
                    await session.merge(new_artifact)
                else:
                    session.add(new_artifact)

                s3_config = self._get_s3_config(new_artifact, parent_artifact)
                if new_artifact.type == "vector-collection":
                    async with self._create_client_async(s3_config) as s3_client:
                        prefix = safe_join(
                            s3_config["prefix"],
                            f"{new_artifact.id}/v0",
                        )
                        await self._vector_engine.create_collection(
                            f"{new_artifact.workspace}/{new_artifact.alias}",
                            config.get("vector_fields", []),
                            overwrite=overwrite,
                            s3_client=s3_client,
                            bucket=s3_config["bucket"],
                            prefix=prefix,
                        )

                await session.commit()
                await self._save_version_to_s3(
                    version_index,
                    new_artifact,
                    s3_config,
                )

                if stage:
                    logger.info(
                        f"Created artifact with ID: {id} (staged), alias: {alias}, parent: {parent_id}"
                    )
                else:
                    logger.info(
                        f"Created artifact with ID: {id} (committed), alias: {alias}, parent: {parent_id}"
                    )

                return self._generate_artifact_data(new_artifact, parent_artifact)
        except Exception as e:
            raise e
        finally:
            await session.close()

    def _validate_artifact_id(self, artifact_id, context: dict):
        """Check if the artifact ID is valid and add workspace if missing."""
        assert isinstance(artifact_id, str), "Artifact ID must be a string."
        # Check if the artifact ID is a valid UUID, e.g.: 0192f4be-11f7-7610-9da2-ffc39feeb009
        if re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            artifact_id,
        ):
            return artifact_id
        elif "/" in artifact_id:
            assert (
                len(artifact_id.split("/")) == 2
            ), "Invalid artifact ID format, it should be `workspace/alias`."
            return artifact_id
        else:
            workspace = context["ws"]
            return f"{workspace}/{artifact_id}"

    async def reset_stats(self, artifact_id, context: dict):
        """Reset the artifact's download count and view count."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        user_info = UserInfo.model_validate(context["user"])
        artifact_id = self._validate_artifact_id(artifact_id, context)

        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "reset_stats", session
                )
                artifact.download_count = 0
                artifact.view_count = 0
                artifact.last_modified = int(time.time())
                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def edit(
        self,
        artifact_id,
        manifest=None,
        type=None,
        config: dict = None,
        secrets: dict = None,
        version: str = None,
        comment: str = None,
        stage: bool = False,
        context: dict = None,
    ):
        """Edit the artifact's manifest and save it in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        user_info = UserInfo.model_validate(context["user"])
        artifact_id = self._validate_artifact_id(artifact_id, context)
        session = await self._get_session()
        manifest = manifest and make_json_safe(manifest)
        config = config and make_json_safe(config)

        logger.info(
            f"Editing artifact {artifact_id} with version={version}, stage={stage}"
        )

        try:
            async with session.begin():
                if stage:
                    artifact, parent_artifact = (
                        await self._get_artifact_with_permission(
                            user_info, artifact_id, "edit", session
                        )
                    )
                else:
                    artifact, parent_artifact = (
                        await self._get_artifact_with_permission(
                            user_info, artifact_id, ["edit", "commit"], session
                        )
                    )

                # Validate version parameter - ONLY allow None (current version), "new" (create new), or "stage" (staging mode)
                # Do NOT allow editing historical versions by specific version names
                if version is not None and version not in ["new", "stage"]:
                    raise ValueError(
                        f"Invalid version '{version}'. Edit operation only supports:\n"
                        f"- None (default): Edit current/latest version\n"
                        f"- 'new': Create a new version\n"
                        f"- 'stage': Put into staging mode\n"
                        f"Historical versions cannot be edited directly."
                    )

                # Convert legacy staging format if needed
                if artifact.staging is not None:
                    artifact.staging = convert_legacy_staging(artifact.staging)
                    if isinstance(artifact.staging, dict) and artifact.staging != convert_legacy_staging(artifact.staging):
                        flag_modified(artifact, "staging")

                # Determine if we're in staging mode
                is_staging = artifact.staging is not None or stage
                
                if stage or is_staging:
                    # Handle staging mode
                    logger.info(f"Artifact {artifact_id} entering/in staging mode")
                    
                    # Validate version parameter for staging
                    if version not in [None, "new", "stage"]:
                        raise ValueError(
                            f"When in staging mode, version must be None, 'new', or 'stage', got '{version}'"
                        )
                    
                    # Initialize or update staging dict
                    if artifact.staging is None:
                        # First time entering staging mode - preserve current state as published
                        artifact.staging = {
                            "manifest": manifest or artifact.manifest,
                            "config": config or artifact.config,
                            "secrets": secrets or artifact.secrets,
                            "type": type or artifact.type,
                            "files": [],
                            "_intent": "new_version" if version == "new" else "edit_version"
                        }
                    else:
                        # Already in staging, update staged values
                        if manifest is not None:
                            artifact.staging["manifest"] = manifest
                        if config is not None:
                            artifact.staging["config"] = config  
                        if secrets is not None:
                            artifact.staging["secrets"] = secrets
                        if type is not None:
                            artifact.staging["type"] = type
                        # Always update intent based on version parameter
                        artifact.staging["_intent"] = "new_version" if version == "new" else "edit_version"
                    
                    flag_modified(artifact, "staging")
                    
                    # Calculate version_index for file storage
                    if artifact.staging.get("_intent") == "new_version":
                        version_index = len(artifact.versions or [])
                    else:
                        version_index = max(0, len(artifact.versions or []) - 1)
                    
                    logger.info(f"Staging mode: version_index={version_index}, intent={artifact.staging.get('_intent')}")
                else:
                    # Not in staging mode - edit directly and commit
                    logger.info(f"Artifact {artifact_id} direct edit mode")
                    
                    if version == "new":
                        # Create new version
                        versions = artifact.versions or []
                        new_version = f"v{len(versions)}"

                        if parent_artifact and not await self._check_permissions(
                            parent_artifact, user_info, "commit"
                        ):
                            raise PermissionError(
                                f"User does not have permission to commit an artifact to collection '{parent_artifact.alias}'."
                            )
                        versions.append(
                            {
                                "version": new_version,
                                "comment": comment,
                                "created_at": int(time.time()),
                            }
                        )
                        artifact.versions = versions
                        flag_modified(artifact, "versions")
                        version_index = len(versions) - 1
                        logger.info(f"Created new version {new_version} with index {version_index}")
                    else:
                        # Update existing latest version
                        versions = artifact.versions or []
                        if len(versions) == 0:
                            # Handle legacy case - create initial version
                            logger.info("Legacy case: no versions exist, creating v0")
                            if parent_artifact and not await self._check_permissions(
                                parent_artifact, user_info, "commit"
                            ):
                                raise PermissionError(
                                    f"User does not have permission to commit an artifact to collection '{parent_artifact.alias}'."
                                )
                            versions.append(
                                {
                                    "version": "v0",
                                    "comment": comment or "Initial version",
                                    "created_at": int(time.time()),
                                }
                            )
                            artifact.versions = versions
                            flag_modified(artifact, "versions")
                            version_index = 0
                        else:
                            version_index = len(versions) - 1
                            logger.info(f"Updating existing latest version at index: {version_index}")
                    
                    # Apply changes directly to artifact (not staging)
                    if manifest is not None:
                        artifact.manifest = manifest
                        flag_modified(artifact, "manifest")
                    if type is not None:
                        artifact.type = type
                    if config is not None:
                        if parent_artifact:
                            parent_permissions = parent_artifact.config.get("permissions", {})
                            permissions = config.get("permissions", {})
                            permissions[user_info.id] = "*"
                            permissions.update(parent_permissions)
                            config["permissions"] = permissions
                        artifact.config = config
                        flag_modified(artifact, "config")
                    if secrets is not None:
                        artifact.secrets = secrets
                        flag_modified(artifact, "secrets")

                artifact.last_modified = int(time.time())

                # Save to database
                session.add(artifact)
                await session.commit()
                
                if stage or is_staging:
                    logger.info(f"Edited artifact with ID: {artifact_id} (staged), alias: {artifact.alias}")
                else:
                    logger.info(f"Edited artifact with ID: {artifact_id} (committed), alias: {artifact.alias}")

                # Always save the current state (staged or committed) to S3
                await self._save_version_to_s3(
                    version_index,  # Use the calculated version_index
                    artifact,
                    self._get_s3_config(artifact, parent_artifact),
                )

                return self._generate_artifact_data(artifact, parent_artifact)
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def read(
        self,
        artifact_id,
        silent=False,
        version: str = None,
        stage: bool = False,
        context: dict = None,
    ):
        """Read the artifact's data including manifest and config."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        # Handle stage parameter
        if stage:
            assert (
                version is None or version == "stage"
            ), "You cannot specify a version when using stage mode."
            version = "stage"

        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "read", session
                )
                if not silent:
                    await self._increment_stat(session, artifact_id, "view_count")

                # Convert legacy staging format if needed
                if artifact.staging is not None:
                    artifact.staging = convert_legacy_staging(artifact.staging)

                # Handle different version requests
                if version == "stage":
                    # Return staged version if in staging mode
                    if artifact.staging is not None:
                        # Create artifact with staged data
                        staged_artifact = ArtifactModel(**model_to_dict(artifact))
                        staged_artifact.manifest = artifact.staging.get("manifest", artifact.manifest)
                        staged_artifact.config = artifact.staging.get("config", artifact.config)
                        staged_artifact.secrets = artifact.staging.get("secrets", artifact.secrets)
                        staged_artifact.type = artifact.staging.get("type", artifact.type)
                        artifact_data = self._generate_artifact_data(staged_artifact, parent_artifact)
                    else:
                        # No staging data - return published version but clear staging field
                        artifact_data = self._generate_artifact_data(artifact, parent_artifact)
                        artifact_data["staging"] = None
                else:
                    # Return published version (from database, not staging)
                    if artifact.staging is not None:
                        # Artifact is in staging mode, but we want published version
                        # Use database values, not staging values
                        published_artifact = ArtifactModel(**model_to_dict(artifact))
                        # Clear staging data to return published state
                        published_artifact.staging = None
                        artifact_data = self._generate_artifact_data(published_artifact, parent_artifact)
                    else:
                        # Normal case - not in staging mode
                        version_index = self._get_version_index(artifact, version)
                        if version_index == self._get_version_index(artifact, None):
                            # Latest version matches what's in database
                            artifact_data = self._generate_artifact_data(artifact, parent_artifact)
                        else:
                            # Historical version - load from S3
                            s3_config = self._get_s3_config(artifact, parent_artifact)
                            version_data = await self._load_version_from_s3(
                                artifact, parent_artifact, version_index, s3_config
                            )
                            version_artifact = ArtifactModel(**model_to_dict(artifact))
                            version_artifact.manifest = version_data.get("manifest")
                            version_artifact.config = version_data.get("config")
                            version_artifact.secrets = version_data.get("secrets")
                            artifact_data = self._generate_artifact_data(version_artifact, parent_artifact)

                if artifact.type == "collection":
                    # Use with_only_columns to optimize the count query
                    count_q = select(func.count()).where(
                        ArtifactModel.parent_id == artifact.id
                    )
                    result = await self._execute_with_retry(
                        session,
                        count_q,
                        description=f"read count query for '{artifact_id}'",
                    )
                    child_count = result.scalar()
                    artifact_data["config"] = artifact_data.get("config", {})
                    artifact_data["config"]["child_count"] = child_count
                elif artifact.type == "vector-collection":
                    artifact_data["config"] = artifact_data.get("config", {})
                    artifact_data["config"]["vector_count"] = (
                        await self._vector_engine.count(
                            f"{artifact.workspace}/{artifact.alias}"
                        )
                    )
                if not silent:
                    await session.commit()

                return artifact_data
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def commit(
        self,
        artifact_id,
        version: str = None,
        comment: str = None,
        context: dict = None,
    ):
        """Commit the artifact by finalizing the staged manifest and files."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        assert version != "stage", "Version cannot be 'stage' when committing."
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "commit", session
                )

                assert (
                    artifact.staging is not None
                ), "Artifact must be in staging mode to commit."

                # Convert legacy staging format if needed
                artifact.staging = convert_legacy_staging(artifact.staging)

                # Extract staged data from staging dict
                staging_dict = artifact.staging
                has_new_version_intent = staging_dict.get("_intent") == "new_version"
                
                # Get staged manifest, config, secrets, type
                staged_manifest = staging_dict.get("manifest")
                staged_config = staging_dict.get("config")
                staged_secrets = staging_dict.get("secrets")
                staged_type = staging_dict.get("type")
                staged_files = staging_dict.get("files", [])
                
                logger.info(
                    f"Committing staged data: has_new_version_intent={has_new_version_intent}"
                )

                # Apply staged data to artifact
                if staged_manifest is not None:
                    artifact.manifest = staged_manifest
                    flag_modified(artifact, "manifest")
                if staged_config is not None:
                    artifact.config = staged_config
                    flag_modified(artifact, "config")
                if staged_secrets is not None:
                    artifact.secrets = staged_secrets
                    flag_modified(artifact, "secrets")
                if staged_type is not None:
                    artifact.type = staged_type

                versions = artifact.versions or []

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    download_weights = {}
                    
                    # First, handle file removals
                    for file_info in staged_files:
                        if file_info.get("_remove"):
                            # This file needs to be removed
                            file_path = file_info["path"]
                            
                            # Determine which version to remove from
                            if has_new_version_intent:
                                # Removing from the previous version (we're creating new)
                                remove_from_version = max(0, len(versions) - 1)
                            else:
                                # Removing from current version
                                remove_from_version = max(0, len(versions) - 1)
                            
                            file_key = safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v{remove_from_version}/{file_path}",
                            )
                            
                            try:
                                await s3_client.delete_object(
                                    Bucket=s3_config["bucket"], Key=file_key
                                )
                                logger.info(f"Removed file '{file_path}' from version {remove_from_version}")
                            except ClientError as e:
                                if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                                    logger.warning(f"File '{file_path}' not found for removal")
                                else:
                                    logger.error(f"Error removing file '{file_path}': {e}")
                    
                    # Handle file additions (non-removal files)
                    for file_info in staged_files:
                        # Skip intent markers and removal markers when processing files
                        if "_intent" in file_info or "_remove" in file_info:
                            continue

                        # Files were placed in staging version index during put_file
                        if has_new_version_intent:
                            # Files were placed at new version index
                            staging_version = len(versions)
                        else:
                            # Files were placed at current version index
                            staging_version = max(0, len(versions) - 1)

                        # Determine final target version based on intent
                        if has_new_version_intent:
                            # Files will be moved to new version index
                            final_target_version = len(versions)
                        else:
                            # Files will be moved to existing latest version index
                            final_target_version = max(0, len(versions) - 1)

                        # Check if file exists at staging location
                        staging_file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{staging_version}/{file_info['path']}",
                        )

                        # Verify file exists
                        try:
                            await s3_client.head_object(
                                Bucket=s3_config["bucket"], Key=staging_file_key
                            )
                        except ClientError:
                            raise FileNotFoundError(
                                f"File '{file_info['path']}' does not exist in the artifact."
                            )

                        # Store download weight if specified
                        if (
                            file_info.get("download_weight") is not None
                            and file_info["download_weight"] > 0
                        ):
                            download_weights[file_info["path"]] = file_info[
                                "download_weight"
                            ]

                    if download_weights:
                        # Merge with existing download weights to preserve special weights like "create-zip-file"
                        existing_weights = artifact.config.get("download_weights", {})
                        existing_weights.update(download_weights)
                        artifact.config["download_weights"] = existing_weights
                        flag_modified(artifact, "config")

                    # Count files in the target version based on intent
                    if has_new_version_intent:
                        # Files are in new version index
                        file_count_version = len(versions)
                    else:
                        # Files are in existing latest version index
                        file_count_version = max(0, len(versions) - 1)

                    artifact.file_count = await self._count_files_in_prefix(
                        s3_client,
                        s3_config["bucket"],
                        safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{file_count_version}",
                        ),
                    )

                parent_artifact_config = (
                    parent_artifact.config if parent_artifact else {}
                )
                if (
                    parent_artifact_config
                    and "collection_schema" in parent_artifact_config
                ):
                    collection_schema = parent_artifact_config["collection_schema"]
                    try:
                        validate(instance=artifact.manifest, schema=collection_schema)
                    except Exception as e:
                        raise ValueError(f"ValidationError: {str(e)}")
                assert artifact.manifest, "Artifact must be in staging mode to commit."

                # Determine the version to create
                if has_new_version_intent or len(versions) == 0:
                    # Creating a new version
                    if version is not None:
                        # Validate that the specified version doesn't already exist
                        existing_versions = [v["version"] for v in versions]
                        if version in existing_versions:
                            raise ValueError(
                                f"Version '{version}' already exists. Existing versions: {existing_versions}"
                            )
                        assigned_version = version
                    else:
                        # Use sequential versioning
                        assigned_version = f"v{len(versions)}"

                    versions.append(
                        {
                            "version": assigned_version,
                            "comment": comment,
                            "created_at": int(time.time()),
                        }
                    )

                    logger.info(
                        f"Creating new version {assigned_version} for artifact {artifact_id}"
                    )
                else:
                    # Updating existing latest version
                    if version is not None:
                        logger.warning(
                            f"Version parameter '{version}' ignored when updating existing version"
                        )
                    assigned_version = versions[-1]["version"] if versions else "v0"
                    if versions and comment:
                        versions[-1]["comment"] = comment

                    logger.info(
                        f"Updating existing version {assigned_version} for artifact {artifact_id}"
                    )

                artifact.versions = versions
                flag_modified(artifact, "versions")

                # Save the version manifest to S3
                if has_new_version_intent or len(versions) == 1:
                    # Creating new version - use the index of the newly created version
                    version_index = len(versions) - 1
                else:
                    # Updating existing version - use the index of the existing latest version
                    version_index = len(versions) - 1

                await self._save_version_to_s3(
                    version_index,
                    artifact,
                    s3_config,
                )

                # Clear staging after successful commit
                artifact.staging = None
                artifact.last_modified = int(time.time())
                flag_modified(artifact, "manifest")
                flag_modified(artifact, "staging")
                await session.merge(artifact)
                await session.commit()

                logger.info(
                    f"Committed artifact with ID: {artifact_id}, alias: {artifact.alias}, version: {assigned_version}"
                )
                return self._generate_artifact_data(artifact, parent_artifact)
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def delete(
        self,
        artifact_id,
        delete_files=False,
        recursive=False,
        version=None,
        context: dict = None,
    ):
        """Delete an artifact from the database and S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        try:
            session = await self._get_session()
            # Get artifact first
            artifact, parent_artifact = await self._get_artifact_with_permission(
                user_info, artifact_id, "delete", session
            )

            # Handle version="stage" - check if artifact is in staging and warn user to use discard instead
            if version == "stage":
                if artifact.staging is not None:
                    raise ValueError(
                        "Cannot delete staged version. The artifact is in staging mode. "
                        "Please use the 'discard' function instead to remove staged changes."
                    )
                else:
                    raise ValueError(
                        "Cannot delete staged version. The artifact is not in staging mode."
                    )

            # If a specific version is requested, validate it exists
            if version is not None and version not in ["latest"]:
                versions = artifact.versions or []
                if isinstance(version, str):
                    # Check if version exists in versions history
                    version_exists = any(v["version"] == version for v in versions)
                    if not version_exists and not version.isdigit():
                        existing_versions = [v["version"] for v in versions]
                        raise ValueError(
                            f"Version '{version}' does not exist. Available versions: {existing_versions}"
                        )
                elif isinstance(version, int):
                    if version < 0 or version >= len(versions):
                        raise ValueError(
                            f"Version index {version} is out of range. Available indices: 0-{len(versions)-1}"
                        )

            if artifact.type == "vector-collection":
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    prefix = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v0",
                    )
                    await self._vector_engine.delete_collection(
                        f"{artifact.workspace}/{artifact.alias}",
                        s3_client=s3_client,
                        bucket=s3_config["bucket"],
                        prefix=prefix,
                    )

            s3_config = self._get_s3_config(artifact, parent_artifact)
            if version is None:
                # Delete all versions and the entire artifact
                # Handle recursive deletion first
                if recursive:
                    children = await self.list_children(artifact_id, context=context)
                    for child in children:
                        await self.delete(
                            child["id"], delete_files=delete_files, context=context
                        )

                # Delete files if requested
                if delete_files:
                    artifact_path = safe_join(
                        s3_config["prefix"],
                        artifact.id,
                    )
                    async with self._create_client_async(
                        s3_config,
                    ) as s3_client:
                        await remove_objects_async(
                            s3_client, s3_config["bucket"], artifact_path + "/"
                        )

                # Delete the artifact
                await session.refresh(artifact)
                artifact.parent_id = None
                await session.flush()
                await session.delete(artifact)
            else:
                # Delete specific version
                version_index = self._get_version_index(artifact, version)
                artifact.versions.pop(version_index)
                flag_modified(artifact, "versions")
                session.add(artifact)
                # Delete the version from S3
                await self._delete_version_files_from_s3(
                    version_index, artifact, s3_config, delete_files=delete_files
                )
            await session.commit()
            logger.info(
                f"Deleted artifact with ID: {artifact_id}, alias: {artifact.alias}, version: {version}"
            )
        except Exception as e:
            if session:
                await session.rollback()
            raise e
        finally:
            if session:
                await session.close()

    async def add_vectors(
        self,
        artifact_id: str,
        vectors: list,
        update: bool = False,
        embedding_models: Optional[Dict[str, str]] = None,
        context: dict = None,
    ):
        """
        Add vectors to a vector collection.
        """
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        max_retries = 3
        retry_delay = 0.5  # seconds

        try:
            for attempt in range(max_retries):
                try:
                    async with session.begin():
                        artifact, parent_artifact = (
                            await self._get_artifact_with_permission(
                                user_info, artifact_id, "add_vectors", session
                            )
                        )
                        assert (
                            artifact.type == "vector-collection"
                        ), "Artifact must be a vector collection."

                        assert (
                            artifact.manifest
                        ), "Artifact must be committed before upserting."
                        s3_config = self._get_s3_config(artifact, parent_artifact)
                        async with self._create_client_async(s3_config) as s3_client:
                            prefix = safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v0",
                            )
                            await self._vector_engine.add_vectors(
                                f"{artifact.workspace}/{artifact.alias}",
                                vectors,
                                update=update,
                                embedding_models=embedding_models
                                or artifact.config.get("embedding_models"),
                                s3_client=s3_client,
                                bucket=s3_config["bucket"],
                                prefix=prefix,
                            )
                        logger.info(f"Added vectors to artifact with ID: {artifact_id}")
                        break
                except KeyError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retrying add_vectors due to error: {str(e)}")
                        await asyncio.sleep(retry_delay)
                    else:
                        raise
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def search_vectors(
        self,
        artifact_id: str,
        query: Optional[Dict[str, Any]] = None,
        embedding_models: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = 5,
        offset: Optional[int] = 0,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
        context: dict = None,
    ):
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "search_vectors", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."

                embedding_models = embedding_models or artifact.config.get(
                    "embedding_models"
                )
                return await self._vector_engine.search_vectors(
                    f"{artifact.workspace}/{artifact.alias}",
                    query=query,
                    embedding_models=embedding_models,
                    filters=filters,
                    limit=limit,
                    offset=offset,
                    return_fields=return_fields,
                    order_by=order_by,
                    pagination=pagination,
                )
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def remove_vectors(
        self,
        artifact_id: str,
        ids: list,
        context: dict = None,
    ):
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "remove_vectors", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    prefix = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v0",
                    )
                    await self._vector_engine.remove_vectors(
                        f"{artifact.workspace}/{artifact.alias}",
                        ids,
                        s3_client=s3_client,
                        bucket=s3_config["bucket"],
                        prefix=prefix,
                    )
                logger.info(f"Removed vectors from artifact with ID: {artifact_id}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_vector(
        self,
        artifact_id: str,
        id: int,
        context: dict = None,
    ):
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "get_vector", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."
                return await self._vector_engine.get_vector(
                    f"{artifact.workspace}/{artifact.alias}", id
                )
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def list_vectors(
        self,
        artifact_id: str,
        offset: int = 0,
        limit: int = 10,
        return_fields: List[str] = None,
        order_by: str = None,
        pagination: bool = False,
        context: dict = None,
    ):
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "list_vectors", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."
                return await self._vector_engine.list_vectors(
                    f"{artifact.workspace}/{artifact.alias}",
                    offset=offset,
                    limit=limit,
                    return_fields=return_fields,
                    order_by=order_by,
                    pagination=pagination,
                )

        except Exception as e:
            raise e
        finally:
            await session.close()

    async def put_file(
        self,
        artifact_id,
        file_path,
        download_weight: float = 0,
        use_proxy=None,
        use_local_url=False,
        expires_in: int = 3600,
        context: dict = None,
    ):
        """Generate a pre-signed URL to upload a file to an artifact in S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        if download_weight < 0:
            raise ValueError(
                "Download weight must be a non-negative number: {download_weight}"
            )
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )
                
                # Require artifact to be in staging mode
                assert artifact.staging is not None, "Artifact must be in staging mode."
                
                # Convert legacy staging format if needed
                artifact.staging = convert_legacy_staging(artifact.staging)

                # Staging mode - files go to staging version
                staging_dict = artifact.staging
                has_new_version_intent = staging_dict.get("_intent") == "new_version"
                
                if has_new_version_intent:
                    # Creating new version - use next index
                    target_version_index = len(artifact.versions or [])
                else:
                    # Non-staging mode - files go to current version
                    target_version_index = max(0, len(artifact.versions or []) - 1)

                version_index = target_version_index

                logger.info(
                    f"Uploading file '{file_path}' to staging version {target_version_index} (has_new_version_intent: {has_new_version_intent})"
                )
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{target_version_index}/{file_path}",
                    )
                    presigned_url = await s3_client.generate_presigned_url(
                        "put_object",
                        Params={"Bucket": s3_config["bucket"], "Key": file_key},
                        ExpiresIn=expires_in,
                    )
                    # Use proxy based on use_proxy parameter (None means use server config)
                    if use_proxy is None:
                        use_proxy = self.s3_controller.enable_s3_proxy

                    if use_proxy:
                        if use_local_url:
                            # Use local URL for proxy within cluster
                            local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"], local_proxy_url
                            )
                        elif s3_config["public_endpoint_url"]:
                            # Use public proxy URL
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"],
                                s3_config["public_endpoint_url"],
                            )
                    elif use_local_url and not use_proxy:
                        # For S3 direct access with local URL, use the endpoint_url as-is
                        # (no replacement needed since endpoint_url is already the local/internal endpoint)
                        if use_local_url is True:
                            pass
                        else:
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"], use_local_url
                            )
                    # Add file to staging dict
                    staging_files = staging_dict.get("files", [])
                    
                    # Check if file already exists in staging
                    if not any(f["path"] == file_path for f in staging_files):
                        file_info = {"path": file_path}
                        # Only store download_weight if it's greater than 0 to save storage
                        if download_weight > 0:
                            file_info["download_weight"] = download_weight
                        staging_files.append(file_info)
                        staging_dict["files"] = staging_files
                        flag_modified(artifact, "staging")
                        
                    # Save artifact state to S3 (with staging info)
                    await self._save_version_to_s3(target_version_index, artifact, s3_config)
                    session.add(artifact)
                    await session.commit()
                    logger.info(
                        f"Put file '{file_path}' to artifact with ID: {artifact_id} (target version: {target_version_index})"
                    )
            return presigned_url
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def put_file_start_multipart(
        self,
        artifact_id,
        file_path,
        part_count: int,
        download_weight: float = 0,
        expires_in: int = 3600,
        use_proxy: bool = None,
        use_local_url: Union[bool, str] = False,
        context: dict = None,
    ):
        """
        Initiate a multipart upload and get a list of presigned URLs for all parts.
        This is a service function that can be called from hypha-rpc.
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        if download_weight < 0:
            raise ValueError(
                f"Download weight must be a non-negative number: {download_weight}"
            )
        if part_count <= 0:
            raise ValueError("Part count must be greater than 0")
        if part_count > 10000:
            raise ValueError("The maximum number of parts is 10,000.")

        session = await self._get_session()
        try:
            # First call put_file to ensure we can put the file and perform permission checks
            await self.put_file(
                artifact_id=artifact_id,
                file_path=file_path,
                download_weight=download_weight,
                use_proxy=use_proxy,
                use_local_url=use_local_url,
                expires_in=expires_in,
                context=context,
            )

            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )
                
                # Check if there's intent to create a new version
                # Convert legacy staging format if needed
                if artifact.staging is not None:
                    artifact.staging = convert_legacy_staging(artifact.staging)
                
                staging_dict = artifact.staging or {}
                has_new_version_intent = staging_dict.get("_intent") == "new_version"
                
                # Determine the correct version index based on intent
                if has_new_version_intent:
                    # Creating new version - use next index
                    version_index = len(artifact.versions or [])
                else:
                    # Editing current version - use current index
                    version_index = max(0, len(artifact.versions or []) - 1)
                s3_config = self._get_s3_config(artifact, parent_artifact)
                s3_key = safe_join(
                    s3_config["prefix"], f"{artifact.id}/v{version_index}/{file_path}"
                )

                async with self._create_client_async(s3_config) as s3_client:
                    # Create the multipart upload to get an UploadId
                    mpu = await s3_client.create_multipart_upload(
                        Bucket=s3_config["bucket"], Key=s3_key
                    )
                    upload_id = mpu["UploadId"]

                    # Use proxy based on use_proxy parameter (None means use server config)
                    if use_proxy is None:
                        use_proxy = self.s3_controller.enable_s3_proxy
                        
                    # Generate a presigned URL for each part
                    part_urls = []
                    for i in range(1, part_count + 1):
                        url = await s3_client.generate_presigned_url(
                            "upload_part",
                            Params={
                                "Bucket": s3_config["bucket"],
                                "Key": s3_key,
                                "UploadId": upload_id,
                                "PartNumber": i,
                            },
                            ExpiresIn=expires_in,
                        )
                        
                        # Apply URL replacement logic based on use_proxy and use_local_url
                        if use_proxy:
                            if use_local_url:
                                # Use local URL for proxy within cluster
                                local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                                url = url.replace(s3_config["endpoint_url"], local_proxy_url)
                            elif s3_config["public_endpoint_url"]:
                                # Use public proxy URL
                                url = url.replace(
                                    s3_config["endpoint_url"],
                                    s3_config["public_endpoint_url"],
                                )
                        elif use_local_url and not use_proxy:
                            # For S3 direct access with local URL, use the endpoint_url as-is
                            # (no replacement needed since endpoint_url is already the local/internal endpoint)
                            if use_local_url is True:
                                pass
                            else:
                                url = url.replace(
                                    s3_config["endpoint_url"], use_local_url
                                )
                        
                        part_urls.append({"part_number": i, "url": url})

                    # Cache the upload info for later completion
                    await self._cache.set(
                        f"multipart_upload:{upload_id}",
                        {
                            "s3_key": s3_key,
                            "parts": part_urls,
                        },
                        ttl=expires_in + 10,
                    )

                    return {
                        "upload_id": upload_id,
                        "parts": part_urls,
                    }
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def put_file_complete_multipart(
        self,
        artifact_id,
        upload_id: str,
        parts: list,
        context: dict = None,
    ):
        """
        Complete the multipart upload and finalize the file in S3.
        This is a service function that can be called from hypha-rpc.

        Args:
            artifact_id: The artifact ID
            upload_id: The upload ID returned from put_file_start_multipart
            parts: List of dicts with 'part_number' and 'etag' keys
            context: Context containing workspace and user info
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        # Get upload info from cache
        cache_key = f"multipart_upload:{upload_id}"
        upload_info = await self._cache.get(cache_key)
        if not upload_info:
            raise ValueError("Upload session expired or not found")

        session = await self._get_session()
        try:
            s3_key = upload_info["s3_key"]

            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )
                s3_config = self._get_s3_config(artifact, parent_artifact)

                # Sort parts by part number and format for S3
                sorted_parts = sorted(parts, key=lambda p: p["part_number"])
                formatted_parts = [
                    {"ETag": p["etag"], "PartNumber": p["part_number"]}
                    for p in sorted_parts
                ]

                async with self._create_client_async(s3_config) as s3_client:
                    await s3_client.complete_multipart_upload(
                        Bucket=s3_config["bucket"],
                        Key=s3_key,
                        MultipartUpload={"Parts": formatted_parts},
                        UploadId=upload_id,
                    )

                # Clean up cache
                await self._cache.delete(cache_key)

                return {"success": True, "message": "File uploaded successfully"}
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def remove_file(self, artifact_id, file_path, context: dict = None):
        """Remove a file from the artifact and update the staged manifest."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "remove_file", session
                )

                assert artifact.staging is not None, "Artifact must be in staging mode."
                
                # Convert legacy staging format if needed
                if artifact.staging is not None:
                    artifact.staging = convert_legacy_staging(artifact.staging)
                
                # Check if there's intent to create a new version
                staging_dict = artifact.staging or {}
                has_new_version_intent = staging_dict.get("_intent") == "new_version"

                # Determine the target version index where files are stored
                if has_new_version_intent:
                    # Creating new version - files to be removed are in the new version index
                    target_version_index = len(artifact.versions or [])
                else:
                    # Editing current version - files are in current version index
                    target_version_index = max(0, len(artifact.versions or []) - 1)

                # Check if the file is in the staging files list (was added during this staging session)
                staging_files = staging_dict.get("files", [])
                file_in_staging = any(f.get("path") == file_path for f in staging_files)

                if file_in_staging:
                    # File was added during this staging session, remove it from staging files
                    staging_dict["files"] = [
                        f for f in staging_files if f.get("path") != file_path
                    ]
                    flag_modified(artifact, "staging")
                else:
                    # File is not in staging files, so it must be from an existing version
                    # Add a removal marker to the staging files
                    staging_dict["files"].append({"path": file_path, "_remove": True})
                    flag_modified(artifact, "staging")

                session.add(artifact)
                await session.commit()

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    # Determine where to delete the file from
                    if file_in_staging:
                        # Delete from staging location (files added during this staging session)
                        # Use the same version index where files were placed
                        if has_new_version_intent:
                            staging_version_index = len(artifact.versions or [])
                        else:
                            staging_version_index = max(0, len(artifact.versions or []) - 1)
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{staging_version_index}/{file_path}",
                        )
                    else:
                        # File is from existing version, we don't delete it yet (commit will handle this)
                        # But we can optionally verify it exists in the target version
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{target_version_index}/{file_path}",
                        )
                        try:
                            await s3_client.head_object(
                                Bucket=s3_config["bucket"], Key=file_key
                            )
                            logger.info(
                                f"Marked file '{file_path}' for removal from version {target_version_index}"
                            )
                            # Don't delete the file here - let commit handle it
                            return
                        except ClientError as e:
                            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                                logger.warning(
                                    f"File '{file_path}' not found in existing version for artifact {artifact_id}, removing from staging anyway"
                                )
                                # File doesn't exist but we still want to remove it from staging
                                return
                            else:
                                logger.warning(
                                    f"Error checking file '{file_path}' for removal: {e}, proceeding anyway"
                                )
                                return

                    # Only delete from S3 if file was in staging
                    try:
                        await s3_client.delete_object(
                            Bucket=s3_config["bucket"], Key=file_key
                        )
                        logger.info(
                            f"Deleted staged file '{file_path}' from S3 at {file_key}"
                        )
                    except ClientError as e:
                        # Handle the case where the file doesn't exist in S3
                        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                            logger.warning(
                                f"File '{file_path}' not found in S3 for artifact {artifact_id}, but removed from staging manifest"
                            )
                        else:
                            # Re-raise other S3 errors
                            raise
                    except Exception as e:
                        # Log other unexpected errors but don't fail the operation
                        logger.warning(
                            f"Error deleting file '{file_path}' from S3 for artifact {artifact_id}: {e}"
                        )

                logger.info(
                    f"Removed file '{file_path}' from artifact with ID: {artifact_id}"
                )
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_file(
        self,
        artifact_id,
        file_path,
        silent=False,
        version=None,
        stage: bool = False,
        use_proxy=None,
        use_local_url: Union[bool, str] = False,
        expires_in: int = 3600,
        context: dict = None,
    ):
        """Generate a pre-signed URL to download a file from an artifact in S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        # Handle stage parameter
        if stage:
            assert (
                version is None or version == "stage"
            ), "You cannot specify a version when using stage mode."
            version = "stage"

        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "get_file", session
                )
                version_index = self._get_version_index(artifact, version)
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{version_index}/{file_path}",
                    )
                    try:
                        await s3_client.head_object(
                            Bucket=s3_config["bucket"], Key=file_key
                        )
                    except ClientError:
                        raise FileNotFoundError(
                            f"File '{file_path}' does not exist in the artifact."
                        )
                    presigned_url = await s3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": s3_config["bucket"], "Key": file_key},
                        ExpiresIn=expires_in,
                    )
                    # Use proxy based on use_proxy parameter (None means use server config)
                    if use_proxy is None:
                        use_proxy = self.s3_controller.enable_s3_proxy

                    if use_proxy:
                        if use_local_url:
                            # Use local URL for proxy within cluster
                            local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"], local_proxy_url
                            )
                        elif s3_config["public_endpoint_url"]:
                            # Use public proxy URL
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"],
                                s3_config["public_endpoint_url"],
                            )
                    elif use_local_url and not use_proxy:
                        # For S3 direct access with local URL, use the endpoint_url as-is
                        # (no replacement needed since endpoint_url is already the local/internal endpoint)
                        if use_local_url is True:
                            pass
                        else:
                            presigned_url = presigned_url.replace(
                                s3_config["endpoint_url"], use_local_url
                            )
                # Increment download count unless silent
                if not silent:
                    if artifact.config and "download_weights" in artifact.config:
                        download_weights = artifact.config.get("download_weights", {})
                    else:
                        download_weights = {}
                    download_weight = download_weights.get(file_path) or 0
                    if download_weight > 0:
                        await self._increment_stat(
                            session,
                            artifact.id,
                            "download_count",
                            increment=download_weight,
                        )
                    await session.commit()
            return presigned_url
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def list_files(
        self,
        artifact_id: str,
        dir_path: str = None,
        limit: int = 1000,
        version: str = None,
        stage: bool = False,
        include_pending: bool = False,
        context: dict = None,
    ):
        """List files in the specified artifact's S3 path."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])

        # Handle stage parameter
        if stage:
            assert (
                version is None or version == "stage"
            ), "You cannot specify a version when using stage mode."
            version = "stage"

        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "list_files", session
                )

                # Handle staging version - check intent to determine correct index
                if version == "stage":
                    # Convert legacy staging format if needed
                    if artifact.staging is not None:
                        artifact.staging = convert_legacy_staging(artifact.staging)
                    
                    # Check if there's intent to create a new version
                    staging_dict = artifact.staging or {}
                    has_new_version_intent = staging_dict.get("_intent") == "new_version"
                    
                    if has_new_version_intent:
                        # Creating new version - use next index
                        version_index = len(artifact.versions or [])
                    else:
                        # Editing current version - use current index
                        version_index = max(0, len(artifact.versions or []) - 1)
                else:
                    version_index = self._get_version_index(artifact, version)

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    if dir_path:
                        full_path = (
                            safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v{version_index}/{dir_path}",
                            )
                            + "/"
                        )
                    else:
                        full_path = (
                            safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v{version_index}",
                            )
                            + "/"
                        )
                    items = await list_objects_async(
                        s3_client,
                        s3_config["bucket"],
                        full_path,
                        max_length=limit,
                    )

                    # If we're in staging mode, filter out files marked for removal
                    if version == "stage" and artifact.staging is not None:
                        # Convert legacy staging format if needed
                        artifact.staging = convert_legacy_staging(artifact.staging)
                        staging_dict = artifact.staging or {}
                        staging_files = staging_dict.get("files", [])
                        
                        # Get list of files marked for removal
                        removed_files = [
                            f["path"] for f in staging_files 
                            if f.get("_remove", False) and "path" in f
                        ]
                        
                        # Filter out removed files from items
                        if removed_files:
                            if dir_path:
                                # Adjust removed files paths for directory context
                                removed_names = [
                                    path[len(dir_path) + 1:] if path.startswith(dir_path + "/") else path
                                    for path in removed_files
                                ]
                            else:
                                removed_names = removed_files
                            
                            items = [
                                item for item in items 
                                if item["name"] not in removed_names
                            ]

                    # If we're in staging mode, merge download weight info from staging manifest with S3 files
                    if version == "stage" and artifact.staging is not None:
                        staging_files_list = staging_dict.get("files", [])
                        # Create a mapping of file paths to download weights
                        download_weight_map = {}
                        for file_info in staging_files_list:
                            if "path" in file_info and "download_weight" in file_info:
                                # Adjust path for directory context if needed
                                file_path = file_info["path"]
                                if dir_path and file_path.startswith(dir_path + "/"):
                                    display_path = file_path[len(dir_path) + 1:]
                                else:
                                    display_path = file_path
                                download_weight_map[display_path] = file_info["download_weight"]
                        
                        # Merge download weights with S3 file items
                        for item in items:
                            if item["name"] in download_weight_map:
                                item["download_weight"] = download_weight_map[item["name"]]

                    # If include_pending is True and we're in staging mode, add pending files from staging manifest
                    if (
                        include_pending
                        and version == "stage"
                        and artifact.staging is not None
                    ):
                        # Get staging files from the dict structure
                        staging_files_list = staging_dict.get("files", [])
                        staging_files = [f for f in staging_files_list if "path" in f and not f.get("_remove", False)]
                        pending_files = []

                        for file_info in staging_files:
                            file_path = file_info["path"]
                            # Filter by dir_path if specified
                            if dir_path:
                                if not file_path.startswith(dir_path + "/"):
                                    continue
                                # Remove dir_path prefix for display
                                display_path = file_path[len(dir_path) + 1 :]
                            else:
                                display_path = file_path

                            # Check if this file is already in the S3 items list
                            if not any(item["name"] == display_path for item in items):
                                pending_files.append(
                                    {
                                        "name": display_path,
                                        "type": "file",
                                        "size": 0,  # We don't know the size yet
                                        "last_modified": None,
                                        "etag": None,
                                        "pending": True,  # Mark as pending
                                        **({
                                            "download_weight": file_info["download_weight"]
                                        } if "download_weight" in file_info else {}),
                                    }
                                )

                        # Add pending files to the items list
                        items.extend(pending_files)

                        # Sort items by name for consistent ordering
                        items.sort(key=lambda x: x["name"])

                        # Limit the results if needed
                        if len(items) > limit:
                            items = items[:limit]

            return items
        except Exception as e:
            raise e
        finally:
            await session.close()

    def _build_manifest_condition(self, manifest_key, operator, value, backend):
        """Helper function to build SQL conditions for manifest fields."""
        if operator == "$like":
            # Fuzzy matching
            if backend == "postgresql":
                return text(
                    f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'"
                )
            else:
                return text(
                    f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'"
                )
        elif operator == "$in":
            # Array containment - any of the values
            if backend == "postgresql":
                # Quote each value in the array
                quoted_values = [f"'{v}'" for v in value]
                array_str = f"ARRAY[{', '.join(quoted_values)}]"
                return text(f"(manifest->'{manifest_key}')::jsonb ?| {array_str}")
            else:
                conditions = []
                for v in value:
                    conditions.append(
                        text(f"json_extract(manifest, '$.{manifest_key}') LIKE '%{v}%'")
                    )
                return or_(*conditions)
        elif operator == "$all":
            # Array containment - all values
            if backend == "postgresql":
                return text(
                    f"(manifest->'{manifest_key}')::jsonb @> '{json.dumps(value)}'::jsonb"
                )
            else:
                array_str = json.dumps(value)[1:-1]  # Remove [] brackets
                return text(
                    f"json_extract(manifest, '$.{manifest_key}') LIKE '%{array_str}%'"
                )
        else:  # exact match
            if isinstance(value, str) and "*" in value:
                # Handle wildcard in simple string match
                if backend == "postgresql":
                    return text(
                        f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'"
                    )
                else:
                    return text(
                        f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'"
                    )
            else:
                if backend == "postgresql":
                    return text(f"manifest->>'{manifest_key}' = '{value}'")
                else:
                    return text(
                        f"json_extract(manifest, '$.{manifest_key}') = '{value}'"
                    )

    def _process_manifest_filter(self, manifest_filter, backend):
        """Process manifest filter with logical operators."""
        if not isinstance(manifest_filter, dict):
            return None

        conditions = []
        for key, value in manifest_filter.items():
            if key == "$and":
                # Process AND conditions
                and_conditions = [
                    self._process_manifest_filter(cond, backend) for cond in value
                ]
                conditions.append(and_(*[c for c in and_conditions if c is not None]))
            elif key == "$or":
                # Process OR conditions
                or_conditions = [
                    self._process_manifest_filter(cond, backend) for cond in value
                ]
                conditions.append(or_(*[c for c in or_conditions if c is not None]))
            elif isinstance(value, dict):
                # Process operators
                for op, op_value in value.items():
                    if op in ["$like", "$in", "$all"]:
                        conditions.append(
                            self._build_manifest_condition(key, op, op_value, backend)
                        )
            else:
                # Handle simple formats
                if isinstance(value, list):
                    # Array values - use $all operator by default
                    conditions.append(
                        self._build_manifest_condition(key, "$all", value, backend)
                    )
                elif isinstance(value, str) and "*" in value:
                    # String with wildcard - use $like operator
                    conditions.append(
                        self._build_manifest_condition(key, "$like", value, backend)
                    )
                else:
                    # Simple equality
                    conditions.append(
                        self._build_manifest_condition(key, "=", value, backend)
                    )

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return and_(*conditions)
        return None

    async def list_children(
        self,
        parent_id=None,
        keywords=None,
        filters=None,
        mode="AND",
        offset: int = 0,
        limit: int = 100,
        order_by: str = None,
        silent: bool = False,
        pagination: bool = False,
        stage: bool = False,
        context: dict = None,
    ):
        """
        List artifacts within a collection under a specific artifact_id.

        Parameters:
            stage: Controls which artifacts to return based on their staging status
                - True: Return only staged artifacts
                - False: Return only committed artifacts
                - 'all': Return both staged and committed artifacts
        """
        logger.info(f"list_children implementation: stage parameter value = {stage}")

        # Convert 'all' to None for internal implementation
        if stage == "all":
            stage = None

        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if parent_id:
            parent_id = self._validate_artifact_id(parent_id, context)
        user_info = UserInfo.model_validate(context.get("user"))

        session = await self._get_session(read_only=True)

        try:
            async with session.begin():
                if parent_id:
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, parent_id, "list", session
                    )
                else:
                    parent_artifact = None
                backend = (
                    self.engine.dialect.name
                )  # Database type (e.g., 'postgresql', 'sqlite')

                # Prepare base query for children artifacts
                if parent_artifact:
                    config = parent_artifact.config or {}
                    list_fields = config.get("list_fields")
                    if list_fields:
                        assert isinstance(
                            list_fields, list
                        ), "Invalid list_fields, it should be a list."
                        assert (
                            "secrets" not in list_fields
                        ), "Secrets cannot be included in list_fields."
                        # If list_fields is specified and is a list, select only those fields
                        query = select(
                            *[getattr(ArtifactModel, field) for field in list_fields]
                        ).where(ArtifactModel.parent_id == parent_artifact.id)
                        count_query = select(func.count()).where(
                            ArtifactModel.parent_id == parent_artifact.id
                        )
                    else:
                        # If list_fields is empty or not specified, select all columns
                        query = select(ArtifactModel).where(
                            ArtifactModel.parent_id == parent_artifact.id
                        )
                        count_query = select(func.count()).where(
                            ArtifactModel.parent_id == parent_artifact.id
                        )
                else:
                    query = select(ArtifactModel).where(
                        ArtifactModel.parent_id == None,
                        ArtifactModel.workspace == context["ws"],
                    )
                    count_query = select(func.count()).where(
                        ArtifactModel.parent_id == None,
                        ArtifactModel.workspace == context["ws"],
                    )
                conditions = []

                # Handle keyword-based search across manifest fields
                if keywords:
                    for keyword in keywords:
                        if backend == "postgresql":
                            condition = text(f"manifest::text ILIKE '%{keyword}%'")
                        else:
                            condition = text(
                                f"json_extract(manifest, '$') LIKE '%{keyword}%'"
                            )
                        conditions.append(condition)

                # Handle filter-based search with specific key-value matching
                if filters:
                    for key, value in filters.items():
                        if key == "version":
                            assert value in [
                                "stage",
                                "latest",
                                "committed",
                            ], "Invalid version value, it should be 'stage', 'latest' or 'committed'."
                            if value == "stage":
                                assert (
                                    stage != None
                                ), "Stage parameter cannot be used with version filter."
                                stage = True
                            elif value == "latest":
                                assert (
                                    stage != True
                                ), "Stage parameter cannot be used with version filter."
                                stage = False
                            elif value == "committed":
                                assert (
                                    stage != True
                                ), "Stage parameter cannot be used with version filter."
                                stage = False
                            continue

                        if key == "manifest" and isinstance(value, dict):
                            condition = self._process_manifest_filter(value, backend)
                            if condition is not None:
                                conditions.append(condition)
                            continue

                        elif key == "config" and isinstance(value, dict):
                            # Nested search within config fields, specifically permissions
                            for config_key, config_value in value.items():
                                if config_key == "permissions" and isinstance(
                                    config_value, dict
                                ):
                                    for user_id, permission in config_value.items():
                                        if backend == "postgresql":
                                            condition = text(
                                                f"permissions->>'{user_id}' = '{permission}'"
                                            )
                                        else:
                                            condition = text(
                                                f"json_extract(permissions, '$.{user_id}') = '{permission}'"
                                            )
                                        conditions.append(condition)
                            continue

                        else:
                            fixed_fields = {
                                "type": ArtifactModel.type,
                                "alias": ArtifactModel.alias,
                                "workspace": ArtifactModel.workspace,
                                "parent_id": ArtifactModel.parent_id,
                                "created_by": ArtifactModel.created_by,
                            }
                            range_fields = {
                                "created_at": ArtifactModel.created_at,
                                "last_modified": ArtifactModel.last_modified,
                                "download_count": ArtifactModel.download_count,
                                "view_count": ArtifactModel.view_count,
                            }
                            if key in fixed_fields:
                                model_field = fixed_fields[key]
                                condition = model_field == value
                                conditions.append(condition)
                                continue
                            elif key in range_fields:
                                model_field = range_fields[key]
                                if isinstance(value, list):
                                    condition = and_(
                                        (
                                            model_field >= value[0]
                                            if value[0] is not None
                                            else True
                                        ),
                                        (
                                            model_field <= value[1]
                                            if value[1] is not None
                                            else True
                                        ),
                                    )
                                else:
                                    condition = model_field >= value
                                conditions.append(condition)
                                continue
                            else:
                                raise ValueError(f"Invalid filter key: {key}")

                # Stage filtering logic
                if stage is not None:  # If stage is explicitly True or False
                    logger.info(f"Adding stage filter condition: stage={stage}")
                    if backend == "sqlite":
                        if stage:
                            # Return only staged artifacts
                            stage_condition = and_(
                                ArtifactModel.staging.is_not(None),
                                text("staging != 'null'"),
                            )
                            logger.info(
                                "Adding condition to return ONLY STAGED artifacts"
                            )
                        else:
                            # Return only committed artifacts
                            stage_condition = text("json_array_length(versions) > 0")
                            logger.info(
                                "Adding condition to return ONLY COMMITTED artifacts (with at least 1 version)"
                            )
                    else:
                        if stage:
                            # Return only staged artifacts
                            stage_condition = and_(
                                ArtifactModel.staging.is_not(None),
                                text("staging::text != 'null'"),
                            )
                            logger.info(
                                "Adding condition to return ONLY STAGED artifacts"
                            )
                        else:
                            # Return only committed artifacts
                            stage_condition = (
                                func.json_array_length(ArtifactModel.versions) > 0
                            )
                            logger.info(
                                "Adding condition to return ONLY COMMITTED artifacts (with at least 1 version)"
                            )

                    query = query.where(stage_condition)
                    count_query = count_query.where(stage_condition)
                else:
                    logger.info(
                        "No stage filter applied - returning BOTH staged and committed artifacts"
                    )

                # Combine conditions based on mode (AND/OR)
                if conditions:
                    query = (
                        query.where(or_(*conditions))
                        if mode == "OR"
                        else query.where(and_(*conditions))
                    )
                    count_query = (
                        count_query.where(or_(*conditions))
                        if mode == "OR"
                        else count_query.where(and_(*conditions))
                    )

                if pagination:
                    # Execute the count query
                    result = await self._execute_with_retry(
                        session,
                        count_query,
                        description=f"list_children count query for '{parent_id or context['ws']}'",
                    )
                    total_count = result.scalar()
                else:
                    total_count = None

                # Pagination and ordering
                order_by = order_by or "id"
                # Parse field name and direction - handle both < and > suffixes
                if order_by:
                    if "<" in order_by:
                        field_name = order_by.split("<")[0]
                        ascending = True
                    elif ">" in order_by:
                        field_name = order_by.split(">")[0]
                        ascending = False
                    else:
                        field_name = order_by
                        ascending = True
                else:
                    field_name = "id"
                    ascending = True

                # Handle JSON fields (manifest.* or config.*)
                if field_name.startswith("manifest."):
                    json_key = field_name[9:]  # Remove "manifest." prefix
                    if backend == "postgresql":
                        # For PostgreSQL, smart ordering: numeric values first, then text values
                        order_clause = f"""
                        CASE 
                            WHEN manifest->>'{json_key}' ~ '^-?[0-9]+\\.?[0-9]*$' 
                            THEN (manifest->>'{json_key}')::numeric 
                        END {'ASC' if ascending else 'DESC'} NULLS LAST,
                        CASE 
                            WHEN manifest->>'{json_key}' !~ '^-?[0-9]+\\.?[0-9]*$' 
                            THEN manifest->>'{json_key}' 
                        END {'ASC' if ascending else 'DESC'} NULLS LAST
                        """
                        query = query.order_by(text(order_clause))
                    else:
                        # For SQLite, smart ordering with type checking
                        order_clause = f"""
                        CASE 
                            WHEN typeof(json_extract(manifest, '$.{json_key}')) IN ('integer', 'real')
                            THEN json_extract(manifest, '$.{json_key}')
                        END {'ASC' if ascending else 'DESC'},
                        CASE 
                            WHEN typeof(json_extract(manifest, '$.{json_key}')) NOT IN ('integer', 'real')
                            THEN json_extract(manifest, '$.{json_key}')
                        END {'ASC' if ascending else 'DESC'}
                        """
                        query = query.order_by(text(order_clause))
                elif field_name.startswith("config."):
                    json_key = field_name[7:]  # Remove "config." prefix
                    if backend == "postgresql":
                        # For PostgreSQL, smart ordering: numeric values first, then text values
                        order_clause = f"""
                        CASE 
                            WHEN config->>'{json_key}' ~ '^-?[0-9]+\\.?[0-9]*$' 
                            THEN (config->>'{json_key}')::numeric 
                        END {'ASC' if ascending else 'DESC'} NULLS LAST,
                        CASE 
                            WHEN config->>'{json_key}' !~ '^-?[0-9]+\\.?[0-9]*$' 
                            THEN config->>'{json_key}' 
                        END {'ASC' if ascending else 'DESC'} NULLS LAST
                        """
                        query = query.order_by(text(order_clause))
                    else:
                        # For SQLite, smart ordering with type checking
                        order_clause = f"""
                        CASE 
                            WHEN typeof(json_extract(config, '$.{json_key}')) IN ('integer', 'real')
                            THEN json_extract(config, '$.{json_key}')
                        END {'ASC' if ascending else 'DESC'},
                        CASE 
                            WHEN typeof(json_extract(config, '$.{json_key}')) NOT IN ('integer', 'real')
                            THEN json_extract(config, '$.{json_key}')
                        END {'ASC' if ascending else 'DESC'}
                        """
                        query = query.order_by(text(order_clause))
                else:
                    # Handle regular fields
                    order_field_map = {
                        "id": ArtifactModel.id,
                        "view_count": ArtifactModel.view_count,
                        "download_count": ArtifactModel.download_count,
                        "last_modified": ArtifactModel.last_modified,
                        "created_at": ArtifactModel.created_at,
                    }
                    order_field = order_field_map.get(field_name, ArtifactModel.id)
                    query = query.order_by(
                        order_field.asc() if ascending else order_field.desc()
                    )

                query = query.limit(limit).offset(offset)

                # Execute the query
                result = await self._execute_with_retry(
                    session,
                    query,
                    description=f"list_children main query for '{parent_id or context['ws']}'",
                )
                artifacts = result.scalars().all()

                # Compile summary results for each artifact
                results = []
                for artifact in artifacts:
                    _artifact_data = self._generate_artifact_data(
                        artifact, parent_artifact
                    )
                    results.append(_artifact_data)

                # Increment view count for parent artifact if not in stage mode
                if not silent and parent_artifact:
                    await self._increment_stat(
                        session, parent_artifact.id, "view_count"
                    )
                    await session.commit()
                if pagination:
                    return {
                        "items": results,
                        "total": total_count,
                        "offset": offset,
                        "limit": limit,
                    }
                return results

        except Exception as e:
            raise e
        finally:
            await session.close()

    async def publish(
        self, artifact_id: str, to: str = None, metadata=None, context: dict = None
    ):
        """Publish the artifact to a public archive like Zenodo."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "publish", session
                )
                manifest = artifact.manifest
                assert manifest, "Manifest is empty or not committed."
                assert "name" in manifest, "Manifest must have a name."
                assert "description" in manifest, "Manifest must have a description."

                config = artifact.config or {}
                to = to or config.get("publish_to")
                zenodo_client = self._get_zenodo_client(
                    artifact, parent_artifact, publish_to=to
                )

                # Handle deposition state - create new version if already published
                existing_zenodo_record = config.get("zenodo")
                deposition_info = None

                if existing_zenodo_record:
                    # Check if this is a published record (has record_id or published DOI)
                    record_id = existing_zenodo_record.get(
                        "record_id"
                    ) or existing_zenodo_record.get("id")
                    is_published = (
                        "doi" in existing_zenodo_record
                        and existing_zenodo_record.get("state") == "done"
                    )

                    if is_published and record_id:
                        try:
                            logger.info(
                                f"Creating new version for published record {record_id}"
                            )
                            deposition_info = await zenodo_client.create_new_version(
                                record_id
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to create new version for record {record_id}: {e}"
                            )
                            # If creating new version fails, try to load the existing deposition
                            try:
                                deposition_info = await zenodo_client.load_deposition(
                                    record_id
                                )
                            except Exception as load_e:
                                logger.warning(
                                    f"Failed to load existing deposition {record_id}: {load_e}"
                                )
                                # If both fail, create a new deposition
                                deposition_info = (
                                    await zenodo_client.create_deposition()
                                )
                    else:
                        # Try to load the existing deposition if it's not published
                        try:
                            deposition_id = existing_zenodo_record.get("id")
                            if deposition_id:
                                deposition_info = await zenodo_client.load_deposition(
                                    str(deposition_id)
                                )
                        except Exception as e:
                            logger.warning(f"Failed to load existing deposition: {e}")
                            # If loading fails, create a new deposition
                            deposition_info = await zenodo_client.create_deposition()

                if not deposition_info:
                    deposition_info = await zenodo_client.create_deposition()

                _metadata = {
                    "title": manifest.get("name", "Untitled"),
                    "upload_type": "dataset" if artifact.type == "dataset" else "other",
                    "description": manifest.get(
                        "description", "No description provided."
                    ),
                    "creators": [
                        {"name": author.get("name")}
                        for author in manifest.get("authors", [{"name": user_info.id}])
                    ],
                    "access_right": "open",
                    "license": manifest.get("license", "cc-by"),
                    "keywords": manifest.get("tags", []),
                    "notes": f"Published automatically from Hypha ({self.store.public_base_url}).",
                }
                if metadata:
                    _metadata.update(metadata)
                await zenodo_client.update_metadata(deposition_info, _metadata)

                # Check if files already exist in the deposition (e.g., from a previous version)
                existing_files = await zenodo_client.list_files(deposition_info)
                existing_file_names = {f["filename"] for f in existing_files}

                async def upload_files(dir_path=""):
                    files = await self.list_files(
                        artifact.id,
                        dir_path=dir_path,
                        version="latest",
                        context=context,
                    )
                    for file_info in files:
                        name = file_info["name"]
                        if file_info["type"] == "directory":
                            await upload_files(safe_join(dir_path, name))
                        else:
                            file_path = safe_join(dir_path, name) if dir_path else name
                            if file_path in existing_file_names:
                                logger.info(
                                    f"File {file_path} already exists in deposition, skipping upload"
                                )
                                continue
                            url = await self.get_file(
                                artifact.id,
                                file_path,
                                version="latest",
                                use_proxy=False,
                                use_local_url=True,
                                context=context,
                            )
                            try:
                                await zenodo_client.import_file(
                                    deposition_info, file_path, url
                                )
                                logger.info(f"Successfully uploaded file: {file_path}")
                            except Exception as e:
                                logger.error(f"Failed to upload file {file_path}: {e}")
                                raise

                await upload_files()
                record = await zenodo_client.publish(deposition_info)

                artifact.config["zenodo"] = record
                flag_modified(artifact, "config")
                session.add(artifact)
                await session.commit()
                return record
        except Exception as e:
            # Re-raise the exception as-is since HTTP errors are now sanitized in ZenodoClient
            raise e
        finally:
            await session.close()

    async def prepare_workspace(self, workspace_info: WorkspaceInfo):
        """Prepare all artifacts in the workspace."""
        # Iterate through all artifacts in the workspace
        # If it's a vector collection, load the vectors from S3
        try:
            session = await self._get_session(read_only=True)
            async with session.begin():
                query = select(ArtifactModel).where(
                    ArtifactModel.workspace == workspace_info.id
                )
                result = await self._execute_with_retry(
                    session,
                    query,
                    description=f"prepare_workspace query for '{workspace_info.id}'",
                )
                artifacts = result.scalars().all()
                for artifact in artifacts:
                    if artifact.type == "vector-collection":
                        parent_artifact = (
                            artifact.parent_id
                            and await self._get_artifact(
                                session=session, artifact_id=artifact.parent_id
                            )
                        )
                        s3_config = self._get_s3_config(artifact, parent_artifact)
                        async with self._create_client_async(s3_config) as s3_client:
                            prefix = safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v0",
                            )
                            await self._vector_engine.load_collection(
                                f"{artifact.workspace}/{artifact.alias}",
                                s3_client=s3_client,
                                bucket=s3_config["bucket"],
                                prefix=prefix,
                            )
            logger.info(
                f"Artifacts (#{len(artifacts)}) in workspace {workspace_info.id} prepared."
            )
        except Exception as e:
            logger.error(f"Error preparing workspace: {traceback.format_exc()}")
            raise e
        finally:
            await session.close()

    async def close_workspace(self, workspace_info: WorkspaceInfo):
        """Archive all artifacts in the workspace."""
        try:
            session = await self._get_session(read_only=True)
            async with session.begin():
                query = select(ArtifactModel).where(
                    ArtifactModel.workspace == workspace_info.id
                )
                result = await self._execute_with_retry(
                    session,
                    query,
                    description=f"close_workspace query for '{workspace_info.id}'",
                )
                artifacts = result.scalars().all()
                for artifact in artifacts:
                    if artifact.type == "vector-collection":
                        # parent_artifact = (
                        #     artifact.parent_id
                        #     and await self._get_artifact(
                        #         session=session, artifact_id=artifact.parent_id
                        #     )
                        # )
                        # s3_config = self._get_s3_config(artifact, parent_artifact)
                        # async with self._create_client_async(s3_config) as s3_client:
                        #     prefix = safe_join(
                        #         s3_config["prefix"],
                        #         f"{artifact.id}/v0",
                        #     )
                        #     await self._vector_engine.dump_collection(
                        #         f"{artifact.workspace}/{artifact.alias}",
                        #         s3_client=s3_client,
                        #         bucket=s3_config["bucket"],
                        #         prefix=prefix,
                        #     )
                        # Delete the collection
                        await self._vector_engine.delete_collection(
                            f"{artifact.workspace}/{artifact.alias}"
                        )
            logger.info(
                f"Artifacts (#{len(artifacts)}) in workspace {workspace_info.id} prepared for closure."
            )
        except Exception as e:
            logger.error(
                f"Error closing workspace {workspace_info.id}: {traceback.format_exc()}"
            )
            raise e
        finally:
            await session.close()

    async def discard(
        self,
        artifact_id,
        context: dict = None,
    ):
        """Discard staged changes for an artifact, reverting to the last committed state."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()

        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "edit", session
                )

                if artifact.staging is None:
                    raise ValueError(
                        "No staged changes to discard. Artifact is not in staging mode."
                    )

                logger.info(f"Discarding staged changes for artifact {artifact_id}")

                # Get S3 config for cleaning up staged files and loading committed version
                s3_config = self._get_s3_config(artifact, parent_artifact)

                # Convert legacy staging format if needed
                if artifact.staging is not None:
                    artifact.staging = convert_legacy_staging(artifact.staging)
                
                # Determine where staged files are located based on intent
                staging_dict = artifact.staging or {}
                has_new_version_intent = staging_dict.get("_intent") == "new_version"

                if has_new_version_intent:
                    # Files are in new version index - delete them
                    staged_files_version_index = len(artifact.versions or [])
                    try:
                        await self._delete_version_files_from_s3(
                            staged_files_version_index,
                            artifact,
                            s3_config,
                            delete_files=True,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up staged files from new version: {e}"
                        )
                else:
                    # Files are in existing version - need to restore from committed state
                    if artifact.versions and len(artifact.versions) > 0:
                        # We need to restore the committed files by removing staged files
                        staged_files_version_index = max(0, len(artifact.versions) - 1)

                        # Delete only the files that were added during staging
                        try:
                            async with self._create_client_async(
                                s3_config
                            ) as s3_client:
                                staging_files = staging_dict.get("files", [])
                                for file_info in staging_files:
                                    if "_intent" in file_info or "_remove" in file_info:
                                        continue
                                    file_key = safe_join(
                                        s3_config["prefix"],
                                        f"{artifact.id}/v{staged_files_version_index}/{file_info['path']}",
                                    )
                                    try:
                                        await s3_client.delete_object(
                                            Bucket=s3_config["bucket"], Key=file_key
                                        )
                                        logger.info(
                                            f"Deleted staged file: {file_info['path']}"
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to delete staged file {file_info['path']}: {e}"
                                        )
                        except Exception as e:
                            logger.warning(
                                f"Failed to clean up staged files from existing version: {e}"
                            )

                # With the new staging dict structure, the database always has the published version
                # For newly created artifacts that were never committed, there's no previous version to restore
                # The manifest/config in the database are already the correct "published" values
                original_manifest_restored = True
                
                # If we couldn't restore from staging marker and artifact has committed versions, try S3
                if not original_manifest_restored and artifact.versions and len(artifact.versions) > 0:
                    # Load the latest committed version from S3 to restore manifest
                    latest_version_index = (
                        len(artifact.versions) - 1
                    )  # Latest version index
                    try:
                        committed_data = await self._load_version_from_s3(
                            artifact,
                            parent_artifact,
                            latest_version_index,
                            s3_config,
                            include_secrets=True,
                        )
                        # Restore manifest and other relevant fields from committed version
                        artifact.manifest = committed_data.get(
                            "manifest", artifact.manifest
                        )
                        artifact.type = committed_data.get("type", artifact.type)
                        if "config" in committed_data:
                            artifact.config = committed_data["config"]
                            flag_modified(artifact, "config")
                        if "secrets" in committed_data:
                            artifact.secrets = committed_data["secrets"]
                            flag_modified(artifact, "secrets")
                        flag_modified(artifact, "manifest")
                        flag_modified(artifact, "type")
                        logger.info(f"Restored manifest from latest committed version: {committed_data.get('manifest')}")
                    except Exception as e:
                        logger.error(
                            f"Failed to restore manifest from S3 for artifact {artifact_id}: {e}",
                            exc_info=True
                        )
                elif not original_manifest_restored:
                    logger.info(
                        f"No committed versions or staging markers to restore from, keeping current manifest"
                    )

                # Clear staging data
                artifact.staging = None
                artifact.last_modified = int(time.time())
                flag_modified(artifact, "staging")

                await session.merge(artifact)
                await session.commit()

                logger.info(
                    f"Successfully discarded staged changes for artifact {artifact_id}"
                )
                return self._generate_artifact_data(artifact, parent_artifact)

        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_secret(
        self,
        artifact_id: str,
        secret_key: str = None,
        context: dict = None,
    ):
        """Get a secret value from an artifact. Requires admin permission."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session(read_only=True)

        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "get_secret", session
                )

                secrets = artifact.secrets or {}
                if secret_key is None:
                    return secrets
                return secrets.get(secret_key)

        except Exception as e:
            raise e
        finally:
            await session.close()

    async def set_secret(
        self,
        artifact_id: str,
        secret_key: str,
        secret_value: str,
        context: dict = None,
    ):
        """Set a secret value for an artifact. Requires read_write permission."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()

        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "set_secret", session
                )

                # Initialize secrets dict if it doesn't exist
                if artifact.secrets is None:
                    artifact.secrets = {}
                # Update or remove the secret
                if secret_value is None:
                    if secret_key in artifact.secrets:
                        del artifact.secrets[secret_key]
                        logger.info(
                            f"Removed secret '{secret_key}' from artifact {artifact_id}"
                        )
                else:
                    artifact.secrets[secret_key] = secret_value
                    logger.info(
                        f"Updated secret '{secret_key}' for artifact {artifact_id}"
                    )

                artifact.last_modified = int(time.time())

                # Mark the field as modified for SQLAlchemy
                flag_modified(artifact, "secrets")

                session.add(artifact)
                await session.commit()

                logger.info(f"Updated secret '{secret_key}' for artifact {artifact_id}")

        except Exception as e:
            raise e
        finally:
            await session.close()

    async def set_download_weight(
        self,
        artifact_id: str,
        file_path: str,
        download_weight: float,
        context: dict = None,
    ):
        """Set download weight for a specific file in an artifact. Requires read_write permission."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        
        if download_weight < 0:
            raise ValueError(f"Download weight must be non-negative, got: {download_weight}")
        
        session = await self._get_session()

        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "set_download_weight", session
                )

                # Initialize config dict if it doesn't exist
                if artifact.config is None:
                    artifact.config = {}
                
                # Initialize download_weights dict if it doesn't exist
                if "download_weights" not in artifact.config:
                    artifact.config["download_weights"] = {}
                
                # Set or remove the download weight
                if download_weight <= 0:
                    # Remove the entry if weight is 0 to keep the config clean
                    if file_path in artifact.config["download_weights"]:
                        del artifact.config["download_weights"][file_path]
                        logger.info(
                            f"Removed download weight for file '{file_path}' in artifact {artifact_id}"
                        )
                else:
                    artifact.config["download_weights"][file_path] = download_weight
                    logger.info(
                        f"Set download weight {download_weight} for file '{file_path}' in artifact {artifact_id}"
                    )

                artifact.last_modified = int(time.time())

                # Mark the field as modified for SQLAlchemy
                flag_modified(artifact, "config")

                session.add(artifact)
                await session.commit()

                logger.info(f"Updated download weight for file '{file_path}' in artifact {artifact_id}")

        except Exception as e:
            raise e
        finally:
            await session.close()

    def get_artifact_service(self):
        """Return the artifact service definition."""
        return {
            "id": "artifact-manager",
            "config": {"visibility": "public", "require_context": True},
            "name": "Artifact Manager",
            "description": "Manage artifacts in a workspace.",
            "create": self.create,
            "reset_stats": self.reset_stats,
            "edit": self.edit,
            "read": self.read,
            "commit": self.commit,
            "discard": self.discard,
            "delete": self.delete,
            "put_file": self.put_file,
            "put_file_start_multipart": self.put_file_start_multipart,
            "put_file_complete_multipart": self.put_file_complete_multipart,
            "remove_file": self.remove_file,
            "get_file": self.get_file,
            "list": self.list_children,
            "list_files": self.list_files,
            "add_vectors": self.add_vectors,
            "search_vectors": self.search_vectors,
            "remove_vectors": self.remove_vectors,
            "get_vector": self.get_vector,
            "list_vectors": self.list_vectors,
            "publish": self.publish,
            "get_secret": self.get_secret,
            "set_secret": self.set_secret,
            "set_download_weight": self.set_download_weight,
        }
