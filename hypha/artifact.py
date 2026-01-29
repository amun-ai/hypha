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
import pickle
from functools import partial
from pathlib import Path
from urllib.parse import quote
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
import base64
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
from hypha.core import ScopeInfo
from hypha_rpc.utils import ObjectProxy
from hypha_rpc.utils.schema import schema_method
from jsonschema import validate
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from pydantic import Field as PydanticField
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
        
        # Initialize default vector engine
        # For SQLite, we'll create the engine on-demand when needed
        # For PostgreSQL, we can initialize pgvector as default
        sql_engine = store.get_sql_engine()
        if sql_engine.dialect.name == 'sqlite':
            # Don't initialize a default engine for SQLite
            # Collections must specify their engine type explicitly
            self._default_vector_engine = None
        else:
            from hypha.pgvector import PgVectorSearchEngine
            # PostgreSQL - use pgvector as default
            self._default_vector_engine = PgVectorSearchEngine(
                sql_engine, prefix="vec", cache_dir=store.get_cache_dir()
            )
        
        # Store vector engines by collection name
        self._vector_engines = {}
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

                    results = await self.list(
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

                results = await self.list(
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

        # HTTP endpoint for searching artifacts in workspace
        @router.get("/{workspace}/artifacts/-/search")
        async def search_artifacts(
            workspace: str,
            keywords: str = None,
            filters: str = None,
            offset: int = 0,
            mode: str = "AND",
            limit: int = 100,
            order_by: str = None,
            pagination: bool = False,
            stage: str = "false",
            no_cache: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Search artifacts across the entire workspace."""
            try:
                cache_key = f"artifact_search:{workspace}:{offset}:{limit}:{order_by}:{keywords}:{filters}:{mode}:{stage}"
                if not no_cache:
                    logger.info(f"Responding to search request ({workspace}) from cache")
                    cached_results = await self._cache.get(cache_key)
                    if cached_results:
                        return cached_results
                if keywords:
                    keywords = keywords.split(",")
                if filters:
                    filters = json.loads(filters)

                # Convert stage parameter
                if stage == "true":
                    stage_param = True
                elif stage == "false":
                    stage_param = False
                elif stage == "all":
                    stage_param = "all"
                else:
                    stage_param = False

                logger.info(
                    f"HTTP search_artifacts endpoint: stage={stage}, converted to stage_param={stage_param}"
                )

                results = await self.search(
                    keywords=keywords,
                    filters=filters,
                    mode=mode,
                    offset=offset,
                    limit=limit,
                    order_by=order_by,
                    pagination=pagination,
                    stage=stage_param,
                    context={"user": user_info.model_dump(), "ws": workspace},
                )
                await self._cache.set(cache_key, results, ttl=60)

                return results
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
                logger.error(f"Unhandled exception in http search_artifacts: {str(e)}")
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

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Re-open session for git zip creation
                    session = await self._get_session(read_only=True)
                    return await self._create_git_zip_file(
                        artifact,
                        parent_artifact,
                        files,
                        version,
                        silent,
                        artifact_alias,
                        session,
                    )

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
                        zip_tail_data = await self._cache.get(cache_key)
                        if zip_tail_data is None:
                            zip_tail_data = await zip_utils.fetch_zip_tail(
                                s3_client, self.workspace_bucket, s3_key, content_length
                            )
                            await self._cache.set(cache_key, zip_tail_data, ttl=60)

                        # Process zip file using the utility function
                        # Pass the full tuple to get_zip_file_content
                        return await zip_utils.get_zip_file_content(
                            s3_client,
                            self.workspace_bucket,
                            s3_key,
                            path=path,
                            zip_tail=zip_tail_data,  # Pass the tuple directly
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
            offset: int = 0,
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
                    # Validate scope for specialized tokens
                    # For file download, the token must have a scope matching the specific
                    # artifact and file path: get_file:{artifact_id}:{file_path}
                    # The workspace is validated via ScopeInfo.workspaces
                    if user_info.is_specialized_token():
                        # Get the artifact to validate scope
                        temp_session = await self._get_session(read_only=True)
                        async with temp_session.begin():
                            artifact_for_scope = await self._get_artifact(
                                temp_session, artifact_id
                            )
                            # Path must be URL-encoded to match the scope format (scopes are space-delimited in JWT)
                            expected_scope = f"get_file:{artifact_for_scope.id}:{quote(path, safe='')}"
                            if not user_info.has_scope(expected_scope):
                                allowed_scopes = ", ".join(user_info.scope.extra_scopes)
                                raise PermissionError(
                                    f"Token scope does not match requested file. "
                                    f"Token has scopes [{allowed_scopes}], but requested "
                                    f"file requires scope '{expected_scope}'."
                                )
                        await temp_session.close()
                async with session.begin():
                    # Fetch artifact and check permissions
                    (
                        artifact,
                        parent_artifact,
                    ) = await self._get_artifact_with_permission(
                        user_info, artifact_id, "get_file", session
                    )

                    # Check if this is a git-storage artifact
                    config = artifact.config or {}
                    if config.get("storage") == "git":
                        # Handle git-storage artifacts
                        return await self._get_git_file(
                            artifact,
                            parent_artifact,
                            path,
                            version,
                            silent,
                            limit,
                            offset,
                            session,
                        )

                    # Standard S3-based artifact storage
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
                                offset=offset,
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
                except Exception:
                    raise
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
        @router.get("/{workspace}/view/{artifact_alias}")
        @router.get("/{workspace}/view/{artifact_alias}/{file_path:path}")
        async def serve_view_page(
            request: Request,
            workspace: str,
            artifact_alias: str,
            file_path: Optional[str] = None,
            stage: Optional[bool] = False,
            token: Optional[str] = None,
            version: Optional[str] = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Serve files from a static site artifact with optional Jinja2 template rendering.
            
            Supports Redis caching for improved performance. Cache configuration via view_config:
            {
                "cache": {
                    "enabled": true,
                    "ttl": 300,  # Cache time-to-live in seconds
                    "patterns": ["*.html", "*.css", "*.js"]  # File patterns to cache
                }
            }
            
            Note: Only public (non-token), non-stage content is cached.
            """
            try:
                # Check cache first for non-stage, public artifacts
                cache_key = None
                if not stage and not token and self._cache:
                    # Only cache public, non-stage content
                    cache_key = f"artifact:view:{workspace}:{artifact_alias}:{version or 'latest'}:{file_path or 'index.html'}"
                    try:
                        cached_content = await self._cache.get(cache_key)
                        if cached_content:
                            # Parse cached response
                            cached_response = pickle.loads(cached_content)
                            return Response(
                                content=cached_response['content'],
                                media_type=cached_response['media_type'],
                                headers=cached_response.get('headers', {})
                            )
                    except Exception as e:
                        logger.debug(f"Cache retrieval failed: {e}")
                        # Continue without cache
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

                    # Check if view is disabled
                    view_config = artifact.config.get("view_config")
                    if view_config == "disabled":
                        raise HTTPException(
                            status_code=404,
                            detail="View access is disabled for this artifact",
                        )
                    
                    # If no view_config provided, load default template based on artifact type
                    if view_config is None:
                        artifact_type = artifact.type or "generic"
                        template_dir = Path(__file__).parent / "templates" / "artifacts" / artifact_type
                        
                        # Check if type-specific template exists
                        if not template_dir.exists():
                            # Fall back to default template if type-specific doesn't exist
                            template_dir = Path(__file__).parent / "templates" / "artifacts" / "default"
                        
                        # If still no template found, artifact cannot be served as view
                        if not template_dir.exists():
                            raise HTTPException(
                                status_code=400,
                                detail="This artifact type does not support view mode",
                            )
                        
                        # Load default configuration for this artifact type
                        view_config = {
                            "template_engine": "jinja2",
                            "templates": ["index.html"],
                            "root_directory": "/",
                        }

                    # Default to index.html if file_path is empty or just "/"
                    if not file_path or file_path == "/":
                        file_path = "index.html"
                    
                    # Strip leading slash to avoid absolute path issues in safe_join
                    if file_path and file_path.startswith("/"):
                        file_path = file_path[1:]

                    # Increment view count for the artifact
                    await self._increment_stat(session, artifact.id, "view_count")
                    await session.commit()

                    # Get artifact configuration
                    config = artifact.config or {}

                    # Check if this is a git-storage artifact
                    if config.get("storage") == "git":
                        # Handle git-storage static site serving
                        custom_headers = view_config.get("headers", {}) if isinstance(view_config, dict) else {}
                        return await self._serve_git_static_site_file(
                            artifact=artifact,
                            parent_artifact=parent_artifact,
                            file_path=file_path,
                            view_config=view_config if isinstance(view_config, dict) else {},
                            version=version,
                            custom_headers=custom_headers,
                        )

                    root_directory = view_config.get("root_directory", config.get("root_directory", "/"))
                    templates = view_config.get("templates", [])
                    template_engine = view_config.get("template_engine", "jinja2")
                    custom_headers = view_config.get("headers", {})
                    metadata = view_config.get("metadata", {})
                    use_builtin_template = view_config.get("use_builtin_template", False)

                    
                    # Handle root_directory properly - if it's "/", use file_path directly
                    if root_directory == "/":
                        abs_file_path = file_path
                    else:
                        abs_file_path = safe_join(root_directory, file_path)
                    
                    # Get S3 configuration and file key (needed for both templated and non-templated files)
                    version_index = self._get_version_index(artifact, version)
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{version_index}",
                        abs_file_path,
                    )

                    # Check if file needs template rendering
                    if (
                        template_engine
                        and template_engine == "jinja2"
                        and file_path in templates
                    ):
                        content = None
                        
                        # Check if we should use builtin template
                        if use_builtin_template or artifact.config.get("view_config") is None:
                            artifact_type = artifact.type or "default"
                            template_path = Path(__file__).parent / "templates" / "artifacts" / artifact_type / file_path
                            
                            # Try type-specific template first
                            if not template_path.exists():
                                # Fall back to default template
                                template_path = Path(__file__).parent / "templates" / "artifacts" / "default" / file_path
                            
                            if template_path.exists():
                                with open(template_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                            else:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Template file not found: {file_path}",
                                )
                        else:
                            # Try to get file content from artifact in S3
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
                                        detail=f"File not found: {file_path} (absolute path: {abs_file_path})",
                                    )

                        # Prepare Jinja2 template context
                        artifact_data = self._generate_artifact_data(
                            artifact, parent_artifact
                        )

                        template_context = {
                            "public_base_url": self.store.public_base_url,
                            "local_base_url": self.store.local_base_url,
                            "artifact": artifact_data,
                            "view_base_url": f"{self.store.public_base_url}/{workspace}/view/{artifact.alias}",
                            "artifact_base_url": f"{self.store.public_base_url}/{workspace}/artifacts/{artifact.alias}",
                            "workspace": workspace,
                            "user": user_info.model_dump(mode="json"),
                            "metadata": metadata,
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

                        # Cache the response if caching is enabled
                        if cache_key and self._cache and not stage and not token:
                            try:
                                # Check if this artifact/page is cacheable based on config
                                cache_config = view_config.get("cache", {})
                                cache_enabled = cache_config.get("enabled", False)
                                cache_ttl = cache_config.get("ttl", 300)  # Default 5 minutes
                                cache_patterns = cache_config.get("patterns", ["*.html", "*.css", "*.js"])
                                
                                # Check if file matches cache patterns
                                import fnmatch
                                should_cache = cache_enabled and any(
                                    fnmatch.fnmatch(file_path, pattern) 
                                    for pattern in cache_patterns
                                )
                                
                                # Also limit cache size - don't cache files larger than 1MB
                                if should_cache and len(rendered) <= 1024 * 1024:
                                    cache_data = {
                                        'content': rendered,
                                        'media_type': mime_type,
                                        'headers': response_headers
                                    }
                                    await self._cache.set(
                                        cache_key, 
                                        pickle.dumps(cache_data),
                                        expire=cache_ttl
                                    )
                            except Exception as e:
                                logger.debug(f"Failed to cache response: {e}")
                                # Continue without caching

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
                raise HTTPException(status_code=404, detail=f"Artifact `{artifact_alias}` not found")
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
        
        # Initialize default vector engine if needed
        if hasattr(self._default_vector_engine, 'initialize'):
            await self._default_vector_engine.initialize()

    def _get_or_create_vector_engine(self, collection_name: str, config: dict = None, artifact=None):
        """Get or create a vector engine for a specific collection.
        
        Args:
            collection_name: Name of the collection (workspace/alias format)
            config: Configuration dict that may contain vector_engine and s3_config
            artifact: The artifact object to get S3 config from
        
        Returns:
            Vector engine instance for this collection
        """
        # Check if we already have an engine for this collection
        if collection_name in self._vector_engines:
            return self._vector_engines[collection_name]
        
        # If no config provided, use default engine
        if not config:
            if self._default_vector_engine is None:
                raise ValueError(
                    "No vector engine configuration provided and no default engine available. "
                    "When using SQLite, you must specify 'vector_engine' in the collection config "
                    "(e.g., 'vector_engine': 's3vector')."
                )
            return self._default_vector_engine
        
        # Get engine type from config
        vector_engine_type = config.get("vector_engine")
        
        # If no engine type specified, use default behavior
        if not vector_engine_type:
            # For SQLite, we require explicit engine specification
            if self._default_vector_engine is None:
                raise ValueError(
                    "When using SQLite, you must specify 'vector_engine' in the collection config "
                    "(e.g., 'vector_engine': 's3vector')."
                )
            # For PostgreSQL, use pgvector as default
            return self._default_vector_engine
        
        # If pgvector explicitly requested
        if vector_engine_type == "pgvector":
            if self._default_vector_engine is None:
                # SQLite doesn't support pgvector
                raise ValueError(
                    "PgVector engine requires PostgreSQL and is not supported with SQLite. "
                    "Please use 's3vector' or another SQLite-compatible vector engine."
                )
            engine = self._default_vector_engine
        elif vector_engine_type == "s3vector":
            from hypha.s3vector import S3VectorSearchEngine
            # Create S3Vector engine with collection-specific config
            # First, get base S3 config using the artifact
            base_s3_config = self._get_s3_config(
                artifact=artifact,
                workspace=collection_name.split('/')[0] if '/' in collection_name else None
            )
            
            # Merge with any custom s3_config provided in the collection config
            custom_s3_config = config.get("s3_config", {})
            s3_config = {**base_s3_config, **custom_s3_config}
            
            # Extract S3 configuration values
            endpoint_url = s3_config.get("endpoint_url")
            access_key_id = s3_config.get("access_key_id")
            secret_access_key = s3_config.get("secret_access_key")
            region_name = s3_config.get("region_name", "us-east-1")
            bucket_name = s3_config.get("bucket")
            
            # Get embedding_model from main config, not s3_config
            embedding_model = config.get("embedding_model")
            
            # Create S3Vector engine
            engine = S3VectorSearchEngine(
                endpoint_url=endpoint_url,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                region_name=region_name,
                bucket_name=bucket_name,
                embedding_model=embedding_model,
                redis_client=self.store.get_redis(),
                cache_dir=self.store.get_cache_dir(),
                **{k: v for k, v in custom_s3_config.items() 
                   if k not in ['endpoint_url', 'access_key_id', 'secret_access_key', 
                               'region_name', 'bucket', 'prefix', 'embedding_model']}
            )
            
            # Store the engine for this collection
            self._vector_engines[collection_name] = engine
        else:
            raise ValueError(f"Unsupported vector engine type: {vector_engine_type}")
        
        return engine

    # Unified vector engine adapter methods
    async def _create_vector_collection(
        self,
        collection_name: str,
        dimension: int = 384,
        distance_metric: str = "cosine",
        overwrite: bool = False,
        config: dict = None,
        **kwargs
    ):
        """Create a vector collection using the configured engine."""
        # Get the appropriate engine for this collection
        # Note: artifact is None here since this is a new collection
        engine = self._get_or_create_vector_engine(collection_name, config, artifact=None)
        
        # Extract embedding_model from config if present
        embedding_model = config.get("embedding_model") if config else None
        
        return await engine.create_collection(
            collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
            overwrite=overwrite,
            embedding_model=embedding_model,
            **kwargs
        )

    async def _get_vector_engine_for_collection(self, collection_name: str):
        """Get the vector engine for an existing collection.
        
        This method retrieves the engine from cache or looks up the collection's config
        to determine which engine to use.
        """
        # Check cache first
        if collection_name in self._vector_engines:
            return self._vector_engines[collection_name]
        
        # Look up collection config from database
        session = await self._get_session(read_only=True)
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, collection_name)
                if artifact and artifact.config:
                    engine = self._get_or_create_vector_engine(collection_name, artifact.config, artifact=artifact)
                    return engine
        except Exception:
            raise
        finally:
            await session.close()
        
        # Fall back to default engine
        return self._default_vector_engine
    
    async def _delete_vector_collection(self, collection_name: str):
        """Delete a vector collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        return await engine.delete_collection(collection_name)

    async def _count_vectors(self, collection_name: str) -> int:
        """Count vectors in a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        return await engine.count(collection_name)

    async def _add_vectors_to_collection(
        self,
        collection_name: str,
        vectors: list,
        embedding_model: str = None,
        **kwargs
    ) -> list:
        """Add vectors to a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        
        # The embedding model should be taken from the collection config,
        # not passed as a parameter (vectors should use the same model)
        # So we don't pass embedding_model to the engine
        return await engine.add_vectors(
            collection_name,
            vectors,
            **kwargs
        )

    async def _search_vectors_in_collection(
        self,
        collection_name: str,
        query_vector=None,
        embedding_model: str = None,
        filters: dict = None,
        limit: int = 5,
        offset: int = 0,
        include_vectors: bool = False,
        order_by: str = None,
        pagination: bool = False,
        **kwargs
    ):
        """Search vectors in a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        
        # The embedding model should be taken from the collection config,
        # not passed as a parameter (search should use the same model as indexing)
        # So we don't pass embedding_model to the engine
        return await engine.search_vectors(
            collection_name,
            query_vector=query_vector,
            filters=filters,
            limit=limit,
            offset=offset,
            include_vectors=include_vectors,
            order_by=order_by,
            pagination=pagination,
            **kwargs
        )

    async def _remove_vectors_from_collection(
        self,
        collection_name: str,
        ids: list,
        **kwargs
    ):
        """Remove vectors from a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        return await engine.remove_vectors(collection_name, ids, **kwargs)

    async def _get_vector_from_collection(
        self,
        collection_name: str,
        vector_id: str
    ):
        """Get a single vector from a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        return await engine.get_vector(collection_name, vector_id)

    async def _list_vectors_in_collection(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 10,
        include_vectors: bool = False,
        order_by: str = None,
        pagination: bool = False,
        **kwargs
    ):
        """List vectors in a collection using the configured engine."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        return await engine.list_vectors(
            collection_name,
            offset=offset,
            limit=limit,
            include_vectors=include_vectors,
            order_by=order_by,
            pagination=pagination,
            **kwargs
        )

    async def _load_vector_collection(self, collection_name: str, s3_client, bucket: str, prefix: str, **kwargs):
        """Load collection from S3 storage - unified interface for both engines."""
        engine = await self._get_vector_engine_for_collection(collection_name)
        if hasattr(engine, 'load_collection'):
            # PgVector and VectorSearchEngine have load_collection method
            return await engine.load_collection(
                collection_name, s3_client, bucket, prefix, **kwargs
            )
        else:
            # S3Vector engine doesn't need load_collection as it reads directly from S3
            # For S3Vector, collection loading is implicit when accessing the collection
            logger.info(f"S3Vector engine doesn't require explicit collection loading for {collection_name}")
            return True

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
                    # Check if there's a staged artifact with this ID
                    staged_query = select(ArtifactModel).where(
                        and_(
                            ArtifactModel.workspace == ws,
                            ArtifactModel.alias == alias,
                            ArtifactModel.staging.isnot(None),
                        )
                    )
                    staged_result = await self._execute_with_retry(
                        session,
                        staged_query,
                        description=f"_get_artifact_with_parent staged check for '{artifact_id}'",
                    )
                    staged_artifact = staged_result.scalar_one_or_none()
                    if staged_artifact:
                        raise KeyError(
                            f"Artifact with ID '{artifact_id}' exists but is in staging mode (not committed). "
                            f"To use this artifact as a parent for child artifacts, you must first commit it using artifact_manager.commit('{artifact_id}'). "
                            f"Alternatively, create both parent and child in staging mode, then commit the child which will commit the entire hierarchy."
                        )

                    # Check if artifact exists at all (for race condition detection)
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
                    # Provide helpful error message for cross-workspace access
                    # Don't leak information about artifacts in other workspaces
                    raise KeyError(
                        f"Artifact with ID '{artifact_id}' does not exist. "
                        f"Note: When working with artifacts from other workspaces, "
                        f"use the full ID (workspace/alias) returned by create(), not just the alias."
                    )
                # Provide helpful error message for common mistake
                if "/" not in artifact_id:
                    raise KeyError(
                        f"Artifact with ID '{artifact_id}' does not exist. "
                        f"Note: When working with artifacts from other workspaces, "
                        f"use the full ID (workspace/alias) returned by create(), not just the alias."
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

        # Staging field handling:
        # - Internally we use dict format: {"files": [], "_intent": "...", "manifest": {...}, ...}
        # - For backward compatibility in API responses, convert dict to list format
        # - Committed artifacts have staging = None (returned as-is)
        if artifact_data.get("staging") is not None and isinstance(artifact_data["staging"], dict):
            staging_dict = artifact_data["staging"]
            staging_list = staging_dict.get("files", [])
            if staging_dict.get("_intent"):
                staging_list = staging_list + [{"_intent": staging_dict["_intent"]}]
            artifact_data["staging"] = staging_list if staging_list else None

        # Expand permissions for each key in the permissions config
        # This allows the frontend to check permissions without needing to know the expansion logic
        config = artifact.config or {}
        permissions = config.get("permissions", {})
        expanded_permissions = {}
        for key, perm in permissions.items():
            expanded_permissions[key] = self._expand_permission(perm)
        artifact_data["_permissions"] = expanded_permissions

        # Add git_url if storage is "git"
        if config.get("storage") == "git":
            from urllib.parse import quote
            base_url = self.store.public_base_url
            # URL-encode workspace and alias for safe use in git clone commands
            encoded_workspace = quote(artifact.workspace, safe='')
            encoded_alias = quote(artifact.alias, safe='')
            git_url = f"{base_url}/{encoded_workspace}/git/{encoded_alias}"
            artifact_data["git_url"] = git_url

        return artifact_data

    def _expand_permission(self, permission):
        """Get the list of operations allowed by the permission code."""
        if isinstance(permission, str):
            permission_map = {
                "n": [],
                "l": [
                    "list",
                ],
                "l+": ["list", "draft"],  # Can list and create drafts
                "lv": ["list", "list_vectors"],
                "lv+": [
                    "list",
                    "list_vectors",
                    "draft",  # Can create drafts
                    "attach",  # Can attach drafts to become children
                    "add_vectors",
                ],
                "lf": ["list", "list_files"],
                "lf+": ["list", "list_files", "draft", "attach", "put_file"],
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
                    "draft",  # Can create drafts
                    "attach",  # Can attach drafts to become children
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
                    "detach",  # Can detach from parent (for rw and higher)
                    "mutate",  # Can modify existing committed versions
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
                    # "commit", can create draft but not commit
                    "put_file",
                    "add_vectors",
                    "remove_file",
                    "remove_vectors",
                    "draft",  # Can create drafts
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
                    "draft",  # Can create drafts
                    "attach",  # Can attach drafts to become children
                    "detach",  # Can detach from parent
                    "mutate",  # Can modify existing committed versions
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
                    "create",  # Full create permission (legacy, includes draft+attach)
                    "draft",  # Can create drafts
                    "attach",  # Can attach drafts to become children
                    "detach",  # Can detach from parent
                    "mutate",  # Can modify existing committed versions
                    "reset_stats",
                    "publish",
                    "delete",
                ],
            }
            return permission_map.get(permission, [])
        else:
            return permission  # Assume it's already a list

    async def _check_permissions(self, artifact, user_info, operation):
        """Check whether a user has permission to perform an operation on an artifact.
        
        Permission checks are performed in the following order:
        1. Workspace-based permissions (workspace/* or workspace/user)
           - IMPORTANT: Only grants access if user's current_workspace matches
        2. Specific user permission (by user ID or email)
        3. Authenticated users permission (@)
        4. Global permission for all users (*)
        """
        # Check specific artifact permissions
        permissions = {}
        if artifact and artifact.config:
            permissions = artifact.config.get("permissions", {})

        # 1. Check workspace-based permissions
        # Parse all permission keys that contain "/" to check for workspace-based permissions
        current_workspace = getattr(user_info, 'current_workspace', None)
        
        for perm_key, perm_value in permissions.items():
            if "/" in perm_key:
                # This is a workspace-based permission
                parts = perm_key.split("/", 1)
                if len(parts) == 2:
                    workspace, target = parts
                    
                    # Check if user's current workspace matches
                    if current_workspace and current_workspace == workspace:
                        # Check different target patterns
                        if target == "*":
                            # All users from this workspace
                            if operation in self._expand_permission(perm_value):
                                return True
                        elif target == "@":
                            # Authenticated users from this workspace
                            if not user_info.is_anonymous:
                                if operation in self._expand_permission(perm_value):
                                    return True
                        elif target == user_info.id:
                            # Specific user ID from this workspace
                            if operation in self._expand_permission(perm_value):
                                return True
                        elif hasattr(user_info, 'email') and user_info.email and target == user_info.email:
                            # Specific email from this workspace
                            if operation in self._expand_permission(perm_value):
                                return True

        # 2. Check specific user permission (by ID or email)
        if user_info.id in permissions:
            if operation in self._expand_permission(permissions[user_info.id]):
                return True
        
        # Also check by email if available
        if hasattr(user_info, 'email') and user_info.email:
            if user_info.email in permissions:
                if operation in self._expand_permission(permissions[user_info.email]):
                    return True

        # 3. Check based on user's authentication status (authenticated users)
        if not user_info.is_anonymous and "@" in permissions:
            if operation in self._expand_permission(permissions["@"]):
                return True

        # 4. Global permission for all users
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
        artifact_permissions: List[str] = None,
        parent_permissions: List[str] = None,
    ):
        """
        Check whether a user has the required permission to perform an operation on an artifact.

        This function checks permissions on both the artifact and its parent collection independently.

        Args:
            user_info: User information
            artifact_id: ID of the artifact to check
            operation: Operation(s) to check permission for (for workspace fallback)
            session: Database session
            artifact_permissions: Specific permissions to check on the artifact itself
            parent_permissions: Specific permissions to check on the parent collection

        If artifact_permissions or parent_permissions are not specified, the function will
        determine them based on the operation type for backward compatibility.
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
            "draft": UserPermission.read_write,  # Creating drafts
            "attach": UserPermission.read_write,  # Attaching drafts as children
            "detach": UserPermission.read_write,  # Detaching from parent
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

        # Get highest required permission level for workspace fallback
        required_perms = [operation_map[op] for op in operations]
        highest_required_perm = max(required_perms)

        # Fetch the primary artifact
        artifact, parent_artifact = await self._get_artifact_with_parent(
            session, artifact_id
        )
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")

        # If specific permissions are not provided, determine them based on operations
        # This maintains backward compatibility
        if artifact_permissions is None and parent_permissions is None:
            # Default permission mapping for operations
            operation_to_permissions = {
                # Read operations - check both artifact and parent
                "read": (["read"], ["read"]),
                "list": (["list"], ["list"]),
                "get_file": (["get_file"], ["get_file"]),
                "list_files": (["list_files"], ["list_files"]),
                "get_vector": (["get_vector"], ["get_vector"]),
                "list_vectors": (["list_vectors"], ["list_vectors"]),
                "search_vectors": (["search_vectors"], ["search_vectors"]),
                
                # Write operations - check artifact only
                "edit": (["edit"], []),
                "commit": (["commit"], []),
                "put_file": (["put_file"], []),
                "add_vectors": (["add_vectors"], []),
                "remove_file": (["remove_file"], []),
                "remove_vectors": (["remove_vectors"], []),
                "set_secret": (["set_secret"], []),
                "set_download_weight": (["set_download_weight"], []),
                "delete": (["delete"], []),
                "reset_stats": (["reset_stats"], []),
                "publish": (["publish"], []),
                "get_secret": (["get_secret"], []),
                
                # Parent-only operations
                "draft": ([], ["draft"]),
                "attach": ([], ["attach"]),
                "detach": ([], ["detach"]),
                "mutate": ([], ["mutate"]),
                
                # Legacy operation
                "create": ([], ["draft", "attach"]),
            }
            
            # Collect all required permissions
            artifact_perms_set = set()
            parent_perms_set = set()
            for op in operations:
                if op in operation_to_permissions:
                    art_perms, par_perms = operation_to_permissions[op]
                    artifact_perms_set.update(art_perms)
                    parent_perms_set.update(par_perms)
                else:
                    # Unknown operation - check on artifact by default
                    artifact_perms_set.add(op)
            
            artifact_permissions = list(artifact_perms_set) if artifact_perms_set else None
            parent_permissions = list(parent_perms_set) if parent_perms_set else None

        # Check artifact permissions
        if artifact_permissions:
            for perm in artifact_permissions:
                if not await self._check_permissions(artifact, user_info, perm):
                    # Try parent as fallback for read operations
                    if perm in ["read", "list", "get_file", "list_files", "get_vector", "list_vectors", "search_vectors"]:
                        if parent_artifact and await self._check_permissions(parent_artifact, user_info, perm):
                            continue  # Permission granted through parent
                    
                    # Check workspace-level permission as final fallback
                    # Workspace owners always have highest permission
                    if not user_info.check_permission(artifact.workspace, operation_map.get(perm, UserPermission.read_write)):
                        raise PermissionError(
                            f"User does not have permission '{perm}' on the artifact."
                        )

        # Check parent permissions
        if parent_permissions and parent_artifact:
            for perm in parent_permissions:
                if not await self._check_permissions(parent_artifact, user_info, perm):
                    # Check workspace-level permission as fallback
                    # Workspace owners always have highest permission
                    if not user_info.check_permission(artifact.workspace, operation_map.get(perm, UserPermission.read_write)):
                        raise PermissionError(
                            f"User does not have permission '{perm}' on the parent collection."
                        )

        return artifact, parent_artifact

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

        # Check if proxy should be disabled for S3 operations
        # Default is to disable proxies to avoid issues with HTTP_PROXY env vars
        # that can cause put_object to hang with some S3/MinIO configurations
        disable_proxy = os.environ.get("HYPHA_S3_DISABLE_PROXY", "true").lower() != "false"
        proxies = {"http": None, "https": None} if disable_proxy else None

        return get_session().create_client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
            config=Config(
                connect_timeout=60,
                read_timeout=300,
                proxies=proxies,
            ),
        )

    async def _get_git_file(
        self,
        artifact,
        parent_artifact,
        path: str,
        version: str,
        silent: bool,
        limit: int,
        offset: int,
        session,
    ):
        """Get file or directory listing from a git-storage artifact.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            path: Path within the git repository
            version: Git ref/commit (defaults to HEAD), or 'stage' for staging area
            silent: If True, don't increment download count
            limit: Max items to return for directory listing
            offset: Offset for pagination
            session: Database session

        Returns:
            Response object (file content or directory listing)
        """
        from fastapi.responses import Response, StreamingResponse
        from hypha.git.repo import S3GitRepo
        import mimetypes

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Handle staging version - files are in S3 staging area, not in git
        if version == "stage":
            return await self._get_git_staging_file_response(
                artifact, s3_config, path, limit, offset
            )

        # Create S3 client factory for the git repo
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Check if repository is empty
        if await repo.is_empty():
            raise HTTPException(
                status_code=404,
                detail="Repository is empty (no commits)",
            )

        # Determine commit SHA from version
        commit_sha = None
        if version and version != "stage":
            # Try to resolve version as a ref or commit SHA
            refs = await repo.get_refs_async()
            # Check if it's a branch ref
            branch_ref = f"refs/heads/{version}".encode()
            if branch_ref in refs:
                commit_sha = refs[branch_ref]
            # Check if it's a tag ref
            elif f"refs/tags/{version}".encode() in refs:
                commit_sha = refs[f"refs/tags/{version}".encode()]
            # Try as a commit SHA
            elif len(version) >= 7:
                # Try to find matching commit
                sha_bytes = bytes.fromhex(version) if len(version) == 40 else None
                if sha_bytes and await repo.has_object_async(sha_bytes):
                    commit_sha = sha_bytes

        # Normalize path
        path = path.strip("/") if path else ""

        # Check if it's a directory listing request
        if path.endswith("/") or path == "":
            # List directory contents
            items = await repo.list_tree_async(path, commit_sha)
            if not items:
                # Check if the path exists but is empty or doesn't exist
                if path:
                    file_info = await repo.get_file_info_async(path, commit_sha)
                    if file_info and file_info["type"] == "blob":
                        # It's a file, not a directory
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{path}' is a file, not a directory. Remove trailing slash.",
                        )
                raise HTTPException(
                    status_code=404,
                    detail="Directory not found or empty",
                )

            # Apply pagination
            total = len(items)
            items = items[offset : offset + limit]

            # Format response similar to S3 listing
            result = []
            for item in items:
                entry = {
                    "name": item["name"],
                    "type": "directory" if item["type"] == "tree" else "file",
                }
                if item["size"] is not None:
                    entry["size"] = item["size"]
                result.append(entry)

            return {
                "items": result,
                "total": total,
                "offset": offset,
                "limit": limit,
            }

        # It's a file request
        content = await repo.get_file_content_async(path, commit_sha)
        if content is None:
            # Check if it's a directory
            items = await repo.list_tree_async(path, commit_sha)
            if items:
                # It's a directory, return listing
                total = len(items)
                items = items[offset : offset + limit]
                result = []
                for item in items:
                    entry = {
                        "name": item["name"],
                        "type": "directory" if item["type"] == "tree" else "file",
                    }
                    if item["size"] is not None:
                        entry["size"] = item["size"]
                    result.append(entry)
                return {
                    "items": result,
                    "total": total,
                    "offset": offset,
                    "limit": limit,
                }
            raise HTTPException(
                status_code=404,
                detail="File not found",
            )

        # Check if this is an LFS pointer (large file stored in S3)
        from hypha.git.lfs import LFSPointer
        lfs_pointer = LFSPointer.parse(content)
        if lfs_pointer is not None:
            # This is an LFS-tracked file - stream from S3 instead
            base_path = f"{s3_config['prefix']}/{artifact.id}"
            lfs_s3_path = lfs_pointer.get_s3_path(base_path)

            # Check if the LFS object exists in S3 using a fresh client
            async with self._create_client_async(s3_config) as s3_client:
                try:
                    head_response = await s3_client.head_object(
                        Bucket=s3_config["bucket"],
                        Key=lfs_s3_path,
                    )
                    lfs_size = head_response["ContentLength"]
                except Exception as e:
                    logger.warning(f"LFS object not found in S3: {lfs_s3_path}, error: {e}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"LFS object not found: {lfs_pointer.oid[:8]}...",
                    )

                # Increment download count for LFS files
                if not silent:
                    config = artifact.config or {}
                    download_weights = config.get("download_weights", {})
                    download_weight = download_weights.get(path) or 0
                    if download_weight > 0:
                        await self._increment_stat(
                            session,
                            artifact.id,
                            "download_count",
                            increment=download_weight,
                        )
                        await session.commit()

                # Determine content type
                content_type, _ = mimetypes.guess_type(path)
                if content_type is None:
                    content_type = "application/octet-stream"
                filename = path.split("/")[-1] if "/" in path else path

                # Stream large files from S3 using StreamingResponse
                async def stream_lfs_content():
                    """Stream LFS content from S3 in chunks."""
                    async with self._create_client_async(s3_config) as client:
                        response = await client.get_object(
                            Bucket=s3_config["bucket"],
                            Key=lfs_s3_path,
                        )
                        async for chunk in response["Body"].iter_chunks():
                            yield chunk

                from fastapi.responses import StreamingResponse
                return StreamingResponse(
                    stream_lfs_content(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                        "Content-Length": str(lfs_size),
                    },
                )

        # Regular (non-LFS) file - increment download count unless silent
        if not silent:
            config = artifact.config or {}
            download_weights = config.get("download_weights", {})
            download_weight = download_weights.get(path) or 0
            if download_weight > 0:
                await self._increment_stat(
                    session,
                    artifact.id,
                    "download_count",
                    increment=download_weight,
                )
                await session.commit()

        # Determine content type
        content_type, _ = mimetypes.guess_type(path)
        if content_type is None:
            content_type = "application/octet-stream"

        # Return file content
        filename = path.split("/")[-1] if "/" in path else path
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            },
        )

    async def _get_git_versions(
        self,
        artifact,
        parent_artifact,
    ) -> List[Dict[str, Any]]:
        """Get available versions (branches and tags) from a git-storage artifact.

        This method queries the git repository to get all branches and tags,
        returning them in a format compatible with the versions field.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance

        Returns:
            List of version dictionaries with structure:
            [
                {"type": "branch", "name": "main", "sha": "abc123...", "default": True},
                {"type": "branch", "name": "feature-x", "sha": "def456..."},
                {"type": "tag", "name": "v1.0", "sha": "789abc..."},
            ]
        """
        from hypha.git.repo import S3GitRepo

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Create S3 client factory for the git repo
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Handle empty repository
        if await repo.is_empty():
            return []

        # Get all refs
        refs = await repo.get_refs_async()

        # Determine the default branch (what HEAD points to)
        default_branch = None
        head_value = refs.get(b"HEAD")
        if head_value and head_value.startswith(b"ref: refs/heads/"):
            default_branch = head_value[16:].decode("utf-8")  # Remove "ref: refs/heads/" prefix

        versions = []

        # Process branches (refs/heads/*)
        for ref_name, sha in sorted(refs.items()):
            if ref_name.startswith(b"refs/heads/"):
                branch_name = ref_name[11:].decode("utf-8")  # Remove "refs/heads/" prefix
                # Skip symbolic refs
                if sha.startswith(b"ref: "):
                    continue
                sha_hex = sha.decode("utf-8") if isinstance(sha, bytes) else sha
                version_entry = {
                    "type": "branch",
                    "name": branch_name,
                    "sha": sha_hex,
                }
                if branch_name == default_branch:
                    version_entry["default"] = True
                versions.append(version_entry)

        # Process tags (refs/tags/*)
        for ref_name, sha in sorted(refs.items()):
            if ref_name.startswith(b"refs/tags/"):
                tag_name = ref_name[10:].decode("utf-8")  # Remove "refs/tags/" prefix
                # Skip peeled tags (they end with ^{})
                if tag_name.endswith("^{}"):
                    continue
                # Skip symbolic refs
                if sha.startswith(b"ref: "):
                    continue
                sha_hex = sha.decode("utf-8") if isinstance(sha, bytes) else sha
                versions.append({
                    "type": "tag",
                    "name": tag_name,
                    "sha": sha_hex,
                })

        return versions

    async def _resolve_git_version(
        self,
        repo,
        version: Optional[str],
    ) -> Optional[bytes]:
        """Resolve a version string to a Git commit SHA.

        Args:
            repo: S3GitRepo instance (must be initialized)
            version: Version string - can be:
                - None or "latest" -> HEAD
                - "stage" -> None (use staging, not yet implemented for git)
                - Branch name (e.g., "main") -> refs/heads/main
                - Tag name (e.g., "v1.0") -> refs/tags/v1.0
                - Commit SHA (full 40 chars or prefix)

        Returns:
            Commit SHA as bytes, or None if version=="stage" or repo is empty

        Raises:
            ValueError: If version cannot be resolved
        """
        if version == "stage":
            # For git storage, staging is not yet implemented
            # Return None to indicate use staging/working tree
            return None

        if version is None or version == "latest":
            # Default to HEAD
            return await repo.head_async()

        # Try to resolve version as a ref or commit SHA
        refs = await repo.get_refs_async()

        # Check if it's a branch ref
        branch_ref = f"refs/heads/{version}".encode()
        if branch_ref in refs:
            return refs[branch_ref]

        # Check if it's a tag ref
        tag_ref = f"refs/tags/{version}".encode()
        if tag_ref in refs:
            return refs[tag_ref]

        # Try as a commit SHA (full or partial)
        if len(version) >= 7:
            if len(version) == 40:
                try:
                    # Validate it's a valid hex SHA
                    bytes.fromhex(version)
                    # Return as hex bytes for consistency with refs
                    # Pack.object_offset handles both hex and binary formats
                    sha_hex_bytes = version.encode() if isinstance(version, str) else version
                    if await repo.has_object_async(sha_hex_bytes):
                        return sha_hex_bytes
                except ValueError:
                    pass
            # TODO: Could support partial SHA matching

        raise ValueError(f"Unknown version: {version}")

    async def _serve_git_static_site_file(
        self,
        artifact,
        parent_artifact,
        file_path: str,
        view_config: dict,
        version: Optional[str],
        custom_headers: Optional[dict] = None,
    ) -> Response:
        """Serve a file from a git-storage artifact as part of a static site.
        
        This implements a hybrid serving model:
        1. First checks if file is already published to S3 sites/ folder
        2. If not, reads from git repository, publishes to S3 (lazy publish), and serves
        
        The lazy publish approach ensures:
        - Fast subsequent requests (served directly from S3)
        - No upfront storage cost (only accessed files are copied)
        - Automatic cache invalidation on commit via cache key invalidation
        
        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            file_path: Path to the file within the git repository
            view_config: View configuration containing branch, root, etc.
            version: Optional version override (branch/tag/commit)
            custom_headers: Optional custom headers to include in response
            
        Returns:
            Response object with file content
            
        Raises:
            HTTPException: If file not found or repository is empty
        """
        from hypha.git.repo import S3GitRepo
        from hypha.git.lfs import LFSPointer
        
        s3_config = self._get_s3_config(artifact, parent_artifact)
        
        # Get branch from view_config or use provided version
        branch = version or view_config.get("branch", "main")
        root_directory = view_config.get("root_directory", view_config.get("root", "/"))
        
        # Build full file path within repository
        if root_directory and root_directory != "/":
            full_file_path = safe_join(root_directory.strip("/"), file_path)
        else:
            full_file_path = file_path
        
        # Initialize git repository
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()
        
        # Check if repository is empty
        if await repo.is_empty():
            raise HTTPException(
                status_code=404,
                detail="Repository is empty (no commits)",
            )
        
        # Resolve branch/version to commit SHA
        try:
            commit_sha = await self._resolve_git_version(repo, branch)
        except ValueError as e:
            raise HTTPException(
                status_code=404,
                detail=f"Branch or version not found: {branch}",
            )
        
        if commit_sha is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not resolve version: {branch}",
            )
        
        # Get short SHA for S3 site path (7 chars like git)
        if isinstance(commit_sha, bytes):
            sha_hex = commit_sha.hex() if len(commit_sha) == 20 else commit_sha.decode()
        else:
            sha_hex = commit_sha
        short_sha = sha_hex[:7]
        
        # Check if file is already published to S3 sites folder
        published_key = safe_join(
            s3_config["prefix"],
            f"{artifact.id}/sites/{short_sha}",
            full_file_path,
        )
        
        content = None
        content_length = None
        is_lfs = False
        lfs_s3_path = None
        
        async with self._create_client_async(s3_config) as s3_client:
            # Try to get from published location first (fast path)
            try:
                response = await s3_client.get_object(
                    Bucket=s3_config["bucket"],
                    Key=published_key,
                )
                content = await response["Body"].read()
                content_length = response.get("ContentLength", len(content))
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
                    raise
                # File not in published location, need to read from git
        
        if content is None:
            # Read from git repository (slow path)
            git_content = await repo.get_file_content_async(full_file_path.strip("/"), commit_sha)
            
            if git_content is None:
                # Try index file for directory requests
                index_file = view_config.get("index", "index.html")
                if not file_path or file_path.endswith("/"):
                    index_path = safe_join(full_file_path.rstrip("/"), index_file)
                    git_content = await repo.get_file_content_async(index_path.strip("/"), commit_sha)
                    if git_content is not None:
                        full_file_path = index_path
                        file_path = safe_join(file_path.rstrip("/"), index_file) if file_path else index_file
                        # Update published key for the index file
                        published_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/sites/{short_sha}",
                            full_file_path,
                        )
                
                if git_content is None:
                    # Check for custom error page
                    error_page = view_config.get("error_page")
                    if error_page:
                        error_path = safe_join(root_directory.strip("/"), error_page) if root_directory and root_directory != "/" else error_page
                        error_content = await repo.get_file_content_async(error_path.strip("/"), commit_sha)
                        if error_content is not None:
                            mime_type, _ = mimetypes.guess_type(error_page)
                            return Response(
                                content=error_content,
                                status_code=404,
                                media_type=mime_type or "text/html",
                                headers=custom_headers or {},
                            )
                    raise HTTPException(
                        status_code=404,
                        detail=f"File not found: {file_path}",
                    )
            
            # Check if this is an LFS pointer
            lfs_pointer = LFSPointer.parse(git_content)
            if lfs_pointer is not None:
                is_lfs = True
                base_path = f"{s3_config['prefix']}/{artifact.id}"
                lfs_s3_path = lfs_pointer.get_s3_path(base_path)
                content_length = lfs_pointer.size
            else:
                content = git_content
                content_length = len(content)
            
            # Lazy publish to S3 sites folder (don't wait, fire and forget for non-LFS)
            # For LFS files, we don't copy - just serve from LFS storage
            if not is_lfs and content is not None:
                async with self._create_client_async(s3_config) as s3_client:
                    try:
                        await s3_client.put_object(
                            Bucket=s3_config["bucket"],
                            Key=published_key,
                            Body=content,
                        )
                    except ClientError:
                        # Failed to publish, but we can still serve the content
                        logger.warning(f"Failed to lazy-publish file to S3: {published_key}")
        
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        
        # Prepare response headers
        response_headers = dict(custom_headers) if custom_headers else {}
        response_headers["Content-Type"] = mime_type
        
        # For LFS files, use streaming response from LFS storage
        if is_lfs and lfs_s3_path:
            s3_client = self._create_client_async(s3_config)
            return FSFileResponse(
                s3_client,
                s3_config["bucket"],
                lfs_s3_path,
                media_type=mime_type,
                headers=response_headers,
            )
        
        # Return regular response for non-LFS files
        return Response(
            content=content,
            media_type=mime_type,
            headers=response_headers,
        )

    async def _invalidate_static_site_cache(
        self,
        artifact,
        branch: str,
    ) -> None:
        """Invalidate Redis cache for a static site when its source branch is updated.
        
        This is called after a git commit to ensure fresh content is served.
        
        Args:
            artifact: The artifact model instance
            branch: The branch that was updated
        """
        if not self._cache:
            return
            
        # Check if this artifact has a static site view_config
        config = artifact.config or {}
        view_config = config.get("view_config")
        if not view_config or not isinstance(view_config, dict):
            return
            
        # Check if the updated branch is the site's source branch
        site_branch = view_config.get("branch", "main")
        if branch != site_branch and branch != "main":
            return
        
        # Invalidate cache by pattern
        # Cache key pattern: artifact:view:{workspace}:{alias}:*
        cache_pattern = f"artifact:view:{artifact.workspace}:{artifact.alias}:*"
        
        try:
            # Use Redis SCAN to find and delete matching keys
            # Note: This requires Redis (not fakeredis) for pattern-based deletion
            if hasattr(self._cache, 'delete_pattern'):
                await self._cache.delete_pattern(cache_pattern)
            elif hasattr(self._cache, 'keys'):
                # Fallback: get all matching keys and delete them
                keys = await self._cache.keys(cache_pattern)
                if keys:
                    await self._cache.delete(*keys)
            else:
                # Basic fallback: can't do pattern-based deletion
                logger.debug(f"Cache doesn't support pattern deletion, skipping cache invalidation")
        except Exception as e:
            logger.warning(f"Failed to invalidate static site cache: {e}")

    async def _list_git_files(
        self,
        artifact,
        parent_artifact,
        dir_path: Optional[str],
        limit: int,
        offset: int,
        version: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List files from a git-storage artifact.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            dir_path: Directory path to list (None for root)
            limit: Maximum number of files to return
            offset: Number of files to skip
            version: Git ref/commit to list from ("stage" for staging area)

        Returns:
            List of file metadata dictionaries
        """
        from hypha.git.repo import S3GitRepo

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Handle staging version - list from S3 staging area
        if version == "stage":
            return await self._list_git_staging_files(
                artifact, s3_config, dir_path, limit, offset
            )

        # Create S3 client factory for the git repo
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Handle empty repository
        if await repo.is_empty():
            return []

        # Resolve version to commit SHA
        try:
            commit_sha = await self._resolve_git_version(repo, version)
        except ValueError:
            # Version not found, return empty
            return []

        # Normalize path
        path = dir_path.strip("/") if dir_path else ""

        # List directory contents
        items = await repo.list_tree_async(path, commit_sha)

        # Apply pagination
        total = len(items)
        items = items[offset : offset + limit]

        # Format response similar to S3 listing
        result = []
        for item in items:
            entry = {
                "name": item["name"],
                "type": "directory" if item["type"] == "tree" else "file",
            }
            if item["size"] is not None:
                entry["size"] = item["size"]
            if item.get("sha"):
                entry["sha"] = item["sha"]
            result.append(entry)

        return result

    async def _list_git_staging_files(
        self,
        artifact,
        s3_config: dict,
        dir_path: Optional[str],
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        """List files from the git staging area in S3.

        For git storage, staged files are uploaded to:
        {prefix}/{artifact.id}/staging/{file_path}

        Args:
            artifact: The artifact model instance
            s3_config: S3 configuration dict
            dir_path: Directory path to list (None for root)
            limit: Maximum number of files to return
            offset: Number of files to skip

        Returns:
            List of file metadata dictionaries
        """
        # Staging area path: {prefix}/{artifact.id}/staging/
        if dir_path:
            staging_prefix = safe_join(
                s3_config["prefix"],
                f"{artifact.id}/staging/{dir_path.strip('/')}",
            ) + "/"
        else:
            staging_prefix = safe_join(
                s3_config["prefix"],
                f"{artifact.id}/staging",
            ) + "/"

        async with self._create_client_async(s3_config) as s3_client:
            items = await list_objects_async(
                s3_client,
                s3_config["bucket"],
                staging_prefix,
                max_length=limit,
                offset=offset,
            )

        # Mark items as staged
        for item in items:
            item["staged"] = True

        return items

    async def _get_git_staging_file_url(
        self,
        artifact,
        s3_config: dict,
        file_path: str,
        expires_in: int,
        use_proxy: Optional[bool],
        use_local_url: bool,
    ) -> str:
        """Get a presigned URL for a staged file in git-storage artifact.

        For git storage, staged files are uploaded to:
        {prefix}/{artifact.id}/staging/{file_path}

        Args:
            artifact: The artifact model instance
            s3_config: S3 configuration dict
            file_path: Path to the file within staging area
            expires_in: URL expiration time in seconds
            use_proxy: Whether to use proxy URL
            use_local_url: Whether to return localhost URL

        Returns:
            Presigned URL for downloading the staged file

        Raises:
            FileNotFoundError: If the staged file does not exist
        """
        # Staging area path: {prefix}/{artifact.id}/staging/{file_path}
        staging_key = safe_join(
            s3_config["prefix"],
            f"{artifact.id}/staging/{file_path}",
        )

        async with self._create_client_async(s3_config) as s3_client:
            # Check if the file exists
            try:
                await s3_client.head_object(
                    Bucket=s3_config["bucket"],
                    Key=staging_key,
                )
            except ClientError:
                raise FileNotFoundError(
                    f"Staged file '{file_path}' not found"
                )

            # Generate presigned URL
            presigned_url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_config["bucket"], "Key": staging_key},
                ExpiresIn=expires_in,
            )

            # Handle proxy/local URL replacement
            if use_proxy is None:
                use_proxy_resolved = self.s3_controller.enable_s3_proxy
            else:
                use_proxy_resolved = use_proxy

            if use_proxy_resolved:
                if use_local_url:
                    local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"], local_proxy_url
                    )
                elif s3_config["public_endpoint_url"]:
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"],
                        s3_config["public_endpoint_url"],
                    )
            elif use_local_url and not use_proxy_resolved:
                if use_local_url is not True:
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"], use_local_url
                    )

            return presigned_url

    async def _get_git_staging_file_response(
        self,
        artifact,
        s3_config: dict,
        path: str,
        limit: int,
        offset: int,
    ):
        """Get file content or directory listing from git staging area.

        For git storage, staged files are uploaded to:
        {prefix}/{artifact.id}/staging/{file_path}

        Args:
            artifact: The artifact model instance
            s3_config: S3 configuration dict
            path: Path within staging area
            limit: Max items to return for directory listing
            offset: Offset for pagination

        Returns:
            Response object (file content or directory listing)

        Raises:
            HTTPException: If file/directory not found
        """
        from fastapi.responses import Response, StreamingResponse
        import mimetypes

        # Normalize path
        path = path.strip("/") if path else ""

        # Check if it's a directory listing request
        if path.endswith("/") or path == "":
            # List directory contents from staging
            items = await self._list_git_staging_files(
                artifact, s3_config, path if path else None, limit, offset
            )
            if not items and path:
                # Check if it might be a file
                staging_key = safe_join(
                    s3_config["prefix"],
                    f"{artifact.id}/staging/{path}",
                )
                async with self._create_client_async(s3_config) as s3_client:
                    try:
                        await s3_client.head_object(
                            Bucket=s3_config["bucket"],
                            Key=staging_key,
                        )
                        # It's a file, not a directory
                        raise HTTPException(
                            status_code=400,
                            detail=f"'{path}' is a file, not a directory. Remove trailing slash.",
                        )
                    except ClientError:
                        pass
                raise HTTPException(
                    status_code=404,
                    detail="Directory not found or empty",
                )

            # Format response similar to S3 listing
            result = []
            for item in items:
                entry = {
                    "name": item["name"],
                    "type": item.get("type", "file"),
                }
                if item.get("size") is not None:
                    entry["size"] = item["size"]
                result.append(entry)

            return {
                "items": result,
                "total": len(items),
                "offset": offset,
                "limit": limit,
            }

        # It's a file request - get file from staging area
        staging_key = safe_join(
            s3_config["prefix"],
            f"{artifact.id}/staging/{path}",
        )

        async with self._create_client_async(s3_config) as s3_client:
            try:
                response = await s3_client.get_object(
                    Bucket=s3_config["bucket"],
                    Key=staging_key,
                )
                content = await response["Body"].read()
                content_length = response.get("ContentLength", len(content))
            except ClientError as e:
                if "NoSuchKey" in str(e) or "404" in str(e):
                    # Check if it's a directory (has children)
                    items = await self._list_git_staging_files(
                        artifact, s3_config, path, limit, offset
                    )
                    if items:
                        # It's a directory, return listing
                        result = []
                        for item in items:
                            entry = {
                                "name": item["name"],
                                "type": item.get("type", "file"),
                            }
                            if item.get("size") is not None:
                                entry["size"] = item["size"]
                            result.append(entry)
                        return {
                            "items": result,
                            "total": len(items),
                            "offset": offset,
                            "limit": limit,
                        }
                    raise HTTPException(
                        status_code=404,
                        detail="File not found in staging area",
                    )
                raise

        # Determine content type
        content_type, _ = mimetypes.guess_type(path)
        if content_type is None:
            content_type = "application/octet-stream"

        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Length": str(content_length)},
        )

    async def _get_git_file_url(
        self,
        artifact,
        parent_artifact,
        file_path: str,
        version: Optional[str],
        expires_in: int,
        use_proxy: Optional[bool],
        use_local_url: bool,
        user_info: UserInfo = None,
    ) -> str:
        """Get a URL for a file from git-storage artifact.

        For LFS files, returns a presigned URL directly to S3.
        For regular git files, returns an HTTP endpoint URL with embedded token
        that serves the file content.
        For staged files (version='stage'), returns presigned URL to S3 staging area.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            file_path: Path to the file
            version: Git ref/commit to read from, or 'stage' for staging area
            expires_in: URL expiration time in seconds
            use_proxy: Whether to use proxy URL
            use_local_url: Whether to return localhost URL
            user_info: User info for generating auth token

        Returns:
            URL string (either presigned S3 URL for LFS or HTTP endpoint URL)

        Raises:
            FileNotFoundError: If file not found
        """
        from hypha.git.repo import S3GitRepo
        from hypha.git.lfs import LFSPointer
        from hypha.core.auth import generate_auth_token
        from urllib.parse import quote, urlencode

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Handle staging version - files are in S3 staging area, not in git
        if version == "stage":
            return await self._get_git_staging_file_url(
                artifact,
                s3_config,
                file_path,
                expires_in,
                use_proxy,
                use_local_url,
            )

        # Create S3 client factory for the git repo
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Handle empty repository
        if await repo.is_empty():
            raise FileNotFoundError(f"Repository is empty, file '{file_path}' not found")

        # Resolve version to commit SHA
        try:
            commit_sha = await self._resolve_git_version(repo, version)
        except ValueError as e:
            raise FileNotFoundError(f"Version '{version}' not found: {e}")

        # Get file content to check if it exists and if it's LFS
        content = await repo.get_file_content_async(file_path, commit_sha)
        if content is None:
            raise FileNotFoundError(f"File '{file_path}' not found in repository")

        # Check if this is an LFS pointer
        lfs_pointer = LFSPointer.parse(content)
        if lfs_pointer is not None:
            # This is an LFS-tracked file - generate presigned URL to S3
            base_path = f"{s3_config['prefix']}/{artifact.id}"
            lfs_s3_path = lfs_pointer.get_s3_path(base_path)

            async with self._create_client_async(s3_config) as s3_client:
                # Verify the LFS object exists
                try:
                    await s3_client.head_object(
                        Bucket=s3_config["bucket"],
                        Key=lfs_s3_path,
                    )
                except ClientError:
                    raise FileNotFoundError(
                        f"LFS object not found: {lfs_pointer.oid[:8]}..."
                    )

                # Generate presigned URL
                presigned_url = await s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": s3_config["bucket"], "Key": lfs_s3_path},
                    ExpiresIn=expires_in,
                )

                # Handle proxy/local URL replacement
                if use_proxy is None:
                    use_proxy_resolved = self.s3_controller.enable_s3_proxy
                else:
                    use_proxy_resolved = use_proxy

                if use_proxy_resolved:
                    if use_local_url:
                        local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                        presigned_url = presigned_url.replace(
                            s3_config["endpoint_url"], local_proxy_url
                        )
                    elif s3_config["public_endpoint_url"]:
                        presigned_url = presigned_url.replace(
                            s3_config["endpoint_url"],
                            s3_config["public_endpoint_url"],
                        )
                elif use_local_url and not use_proxy_resolved:
                    if use_local_url is not True:
                        presigned_url = presigned_url.replace(
                            s3_config["endpoint_url"], use_local_url
                        )

                return presigned_url

        # Regular git file - generate HTTP endpoint URL with token
        # The HTTP endpoint /{workspace}/artifacts/{alias}/files/{path} handles git files
        if user_info is None:
            raise ValueError("user_info is required to generate download URL for git files")

        # Create a specialized token with get_file scope that is restricted to
        # this specific artifact and file path. This prevents token hijacking
        # where a leaked URL token could be used to download other files.
        # The workspace is already encoded in ScopeInfo.workspaces, so we only need
        # to include artifact_id and file_path in the extra scope.
        # Scope format: get_file:{artifact_id}:{url_encoded_file_path}
        # Note: file_path must be URL-encoded because scopes are space-delimited in JWT
        specific_scope = f"get_file:{artifact.id}:{quote(file_path, safe='')}"
        specialized_scope = ScopeInfo(
            workspaces=user_info.scope.workspaces.copy() if user_info.scope else {},
            current_workspace=user_info.scope.current_workspace if user_info.scope else None,
            client_id=user_info.scope.client_id if user_info.scope else None,
            extra_scopes=[specific_scope],  # Specialized scope for this specific file only
        )
        specialized_user_info = user_info.model_copy(update={
            "scope": specialized_scope,
            "current_workspace": specialized_scope.current_workspace
        })

        # Generate a token for the URL (with same expiry as requested)
        token = await generate_auth_token(specialized_user_info, expires_in)

        # Determine base URL
        if use_local_url:
            base_url = self.store.local_base_url if use_local_url is True else use_local_url
        else:
            base_url = self.store.public_base_url

        # Build the URL to the files endpoint
        # URL format: {base_url}/{workspace}/artifacts/{alias}/files/{file_path}?token={token}
        encoded_file_path = quote(file_path, safe='/')
        url = f"{base_url}/{artifact.workspace}/artifacts/{artifact.alias}/files/{encoded_file_path}"

        # Add query parameters
        params = {"token": token}
        if version:
            params["version"] = version

        url = f"{url}?{urlencode(params)}"

        return url

    async def _put_git_file(
        self,
        artifact,
        parent_artifact,
        file_path: str,
        download_weight: float,
        expires_in: int,
        use_proxy: Optional[bool],
        use_local_url: bool,
    ) -> str:
        """Stage a file for upload to a git-storage artifact.

        For git storage, files are uploaded to a staging area in S3,
        then on commit they are added to the git repository.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            file_path: Path where to store the file
            download_weight: Download weight for statistics
            expires_in: URL expiration time
            use_proxy: Whether to use proxy URL
            use_local_url: Whether to return localhost URL

        Returns:
            Presigned URL for uploading the file
        """
        s3_config = self._get_s3_config(artifact, parent_artifact)

        # For git storage, upload to a staging area
        # The staging area is: {prefix}/{artifact.id}/staging/{file_path}
        staging_key = safe_join(
            s3_config["prefix"],
            f"{artifact.id}/staging/{file_path}",
        )

        async with self._create_client_async(s3_config) as s3_client:
            presigned_url = await s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": s3_config["bucket"], "Key": staging_key},
                ExpiresIn=expires_in,
            )

            # Handle proxy/local URL replacement
            if use_proxy is None:
                use_proxy_resolved = self.s3_controller.enable_s3_proxy
            else:
                use_proxy_resolved = use_proxy

            if use_proxy_resolved:
                if use_local_url:
                    local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"], local_proxy_url
                    )
                elif s3_config["public_endpoint_url"]:
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"],
                        s3_config["public_endpoint_url"],
                    )
            elif use_local_url and not use_proxy_resolved:
                if use_local_url is not True:
                    presigned_url = presigned_url.replace(
                        s3_config["endpoint_url"], use_local_url
                    )

            return presigned_url

    async def _commit_git(
        self,
        artifact,
        parent_artifact,
        version: Optional[str],
        comment: Optional[str],
        user_info,
        session,
    ) -> Dict[str, Any]:
        """Commit staged changes to a git-storage artifact.

        This method:
        1. Gets staged files from the S3 staging area
        2. Creates git blob objects for each file
        3. Builds git tree structure
        4. Creates a git commit object
        5. Updates refs (creates tag if version specified)
        6. Cleans up staging area

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            version: Version tag for this commit (creates git tag)
            comment: Commit message
            user_info: User info for commit authorship
            session: Database session

        Returns:
            Commit result dict with status, version, commit_sha
        """
        from hypha.git.repo import S3GitRepo
        from dulwich.objects import Blob, Tree, Commit

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Create repo instance
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Get staged files from artifact.staging (for removal tracking and download weights)
        staging_dict = artifact.staging or {}
        staged_files_manifest = staging_dict.get("files", [])

        # Build lookup for file metadata from manifest
        manifest_lookup = {}
        files_to_remove = []
        for file_info in staged_files_manifest:
            if file_info.get("_remove"):
                files_to_remove.append(file_info["path"])
            elif "path" in file_info:
                manifest_lookup[file_info["path"]] = file_info

        # List all files from S3 staging area directly
        # This ensures all uploaded files are committed, even if the manifest
        # is incomplete due to race conditions in concurrent put_file calls
        staging_prefix = f"{s3_config['prefix']}/{artifact.id}/staging/"
        files_to_commit = []

        async with self._create_client_async(s3_config) as s3_client:
            # List all files in the staging area
            staged_s3_files = await list_objects_async(
                s3_client,
                s3_config["bucket"],
                staging_prefix,
                delimiter="",  # List all files recursively
            )

            for s3_file in staged_s3_files:
                if s3_file.get("type") != "file":
                    continue

                # When delimiter="" (recursive listing), s3_file["name"] is the full key
                # We need to extract the relative path by removing the staging_prefix
                full_key = s3_file["name"]
                if not full_key.startswith(staging_prefix):
                    logger.warning(f"Unexpected S3 key format: {full_key}")
                    continue
                file_path = full_key[len(staging_prefix):]

                try:
                    response = await s3_client.get_object(
                        Bucket=s3_config["bucket"],
                        Key=full_key,
                    )
                    content = await response["Body"].read()

                    # Get download_weight from manifest if available
                    file_info = manifest_lookup.get(file_path, {})
                    files_to_commit.append({
                        "path": file_path,
                        "content": content,
                        "download_weight": file_info.get("download_weight", 0),
                    })
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                        logger.warning(f"Staged file not found: {full_key}")
                    else:
                        raise

        if not files_to_commit and not files_to_remove:
            # Even with no staged files, we need to ensure the artifact has a version
            # to be visible in list() when stage=False (which filters by versions > 0).
            # This handles the case where files were pushed via git directly.
            versions = artifact.versions or []
            if not versions:
                versions.append({
                    "version": "v0",
                    "comment": comment or "Initial version",
                    "created_at": int(time.time()),
                })
                artifact.versions = versions
                flag_modified(artifact, "versions")

            # Clear staging
            artifact.staging = None
            flag_modified(artifact, "staging")

            # Count files in repository from HEAD if available
            current_head = await repo.head_async()
            if current_head:
                all_items = await repo.list_tree_async("", current_head)
                artifact.file_count = len([i for i in all_items if i["type"] == "blob"])

            return {
                "status": "no_changes",
                "artifact_id": f"{artifact.workspace}/{artifact.alias}",
            }

        # Get current HEAD (if exists)
        current_head = await repo.head_async()

        # Build the tree structure
        # We need to merge new files with existing tree if there's a parent commit
        tree_items = {}

        if current_head:
            # Get existing tree from parent commit
            parent_commit = await repo.get_object_async(current_head)
            parent_tree = await repo.get_object_async(parent_commit.tree)

            # Copy existing tree items (recursively for nested trees)
            async def copy_tree_items(tree_sha, path_prefix=""):
                tree = await repo.get_object_async(tree_sha)
                for entry in tree.items():
                    name, mode, sha = entry
                    item_path = f"{path_prefix}{name.decode()}" if path_prefix else name.decode()
                    if mode == 0o40000:  # Directory
                        await copy_tree_items(sha, f"{item_path}/")
                    else:
                        tree_items[item_path] = {"mode": mode, "sha": sha}

            await copy_tree_items(parent_tree.id)

        # Remove files marked for removal
        for path in files_to_remove:
            if path in tree_items:
                del tree_items[path]

        # Add/update new files
        blobs_to_add = []
        for file_data in files_to_commit:
            blob = Blob.from_string(file_data["content"])
            blobs_to_add.append((blob, None))
            tree_items[file_data["path"]] = {
                "mode": 0o100644,
                "sha": blob.id,
            }

        if not tree_items:
            # Empty tree after removals
            return {
                "status": "no_changes",
                "artifact_id": f"{artifact.workspace}/{artifact.alias}",
            }

        # Build tree hierarchy from flat paths
        def build_tree_hierarchy(items):
            """Build nested tree structure from flat path dict."""
            root = {}
            for path, info in items.items():
                parts = path.split("/")
                current = root
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {"_children": {}}
                    current = current[part]["_children"]
                current[parts[-1]] = info
            return root

        def create_tree_object(node):
            """Recursively create tree objects from hierarchy."""
            tree = Tree()
            for name, info in sorted(node.items()):
                if name == "_children":
                    continue
                if "_children" in info:
                    # Subdirectory
                    subtree = create_tree_object(info["_children"])
                    blobs_to_add.append((subtree, None))
                    tree.add(name.encode(), 0o40000, subtree.id)
                else:
                    # File
                    tree.add(name.encode(), info["mode"], info["sha"])
            return tree

        tree_hierarchy = build_tree_hierarchy(tree_items)
        root_tree = create_tree_object(tree_hierarchy)
        blobs_to_add.append((root_tree, None))

        # Create commit
        commit = Commit()
        commit.tree = root_tree.id

        # Set author/committer
        author_name = getattr(user_info, "name", None) or user_info.id or "Anonymous"
        author_email = getattr(user_info, "email", None) or "anonymous@hypha"
        author_string = f"{author_name} <{author_email}>".encode()
        commit.author = commit.committer = author_string

        # Set commit time
        import time as time_module
        commit_time = int(time_module.time())
        commit.author_time = commit.commit_time = commit_time
        commit.author_timezone = commit.commit_timezone = 0

        # Set commit message
        commit.message = (comment or "Update files").encode()

        # Set parent
        if current_head:
            commit.parents = [current_head]
        else:
            commit.parents = []

        blobs_to_add.append((commit, None))

        # Add all objects to the repository
        await repo._object_store.add_objects_async(blobs_to_add)

        # Update refs
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Create tag if version specified
        if version:
            tag_ref = f"refs/tags/{version}".encode()
            await repo.set_ref_async(tag_ref, commit.id)

        # Clean up staging area
        async with self._create_client_async(s3_config) as s3_client:
            for file_data in files_to_commit:
                staging_key = f"{staging_prefix}{file_data['path']}"
                try:
                    await s3_client.delete_object(
                        Bucket=s3_config["bucket"],
                        Key=staging_key,
                    )
                except Exception as e:
                    logger.warning(f"Failed to clean up staging file {staging_key}: {e}")

        # Update download weights in artifact config if any
        download_weights = {}
        for file_data in files_to_commit:
            if file_data.get("download_weight", 0) > 0:
                download_weights[file_data["path"]] = file_data["download_weight"]

        if download_weights:
            if artifact.config is None:
                artifact.config = {}
            existing_weights = artifact.config.get("download_weights", {})
            existing_weights.update(download_weights)
            artifact.config["download_weights"] = existing_weights
            flag_modified(artifact, "config")

        # Clear artifact staging
        artifact.staging = None
        flag_modified(artifact, "staging")

        # Count files in repository
        all_items = await repo.list_tree_async("", commit.id)
        artifact.file_count = len([i for i in all_items if i["type"] == "blob"])

        # Ensure git artifact has a version entry to be visible in list()
        # when stage=False (which filters by json_array_length(versions) > 0)
        versions = artifact.versions or []
        if not versions:
            versions.append({
                "version": "v0",
                "comment": comment or "Initial version",
                "created_at": int(time_module.time()),
            })
            artifact.versions = versions
            flag_modified(artifact, "versions")

        # Invalidate static site cache if this artifact is configured as a static site
        target_branch = version or "main"
        await self._invalidate_static_site_cache(artifact, target_branch)

        return {
            "status": "committed",
            "version": version or "main",
            "commit_sha": commit.id.hex() if isinstance(commit.id, bytes) else commit.id,
            "artifact_id": f"{artifact.workspace}/{artifact.alias}",
            "files_added": len(files_to_commit),
            "files_removed": len(files_to_remove),
        }

    async def _remove_git_file(
        self,
        artifact,
        parent_artifact,
        file_path: str,
    ) -> Dict[str, str]:
        """Mark a file for removal from a git-storage artifact.

        For git storage, file removal is staged and applied on commit.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            file_path: Path to the file to remove

        Returns:
            Status dict
        """
        # Just mark the file for removal in staging
        # The actual git commit will handle the tree update
        staging_dict = artifact.staging or {}
        if artifact.staging is None:
            artifact.staging = {}
            staging_dict = artifact.staging

        staging_dict = convert_legacy_staging(staging_dict)
        staging_files = staging_dict.get("files", [])

        # Check if file was added in this staging session
        file_in_staging = any(
            f.get("path") == file_path and not f.get("_remove")
            for f in staging_files
        )

        if file_in_staging:
            # Remove from staging list (file was added but not committed yet)
            staging_dict["files"] = [
                f for f in staging_files if f.get("path") != file_path
            ]

            # Also remove from S3 staging area
            s3_config = self._get_s3_config(artifact, parent_artifact)
            staging_key = safe_join(
                s3_config["prefix"],
                f"{artifact.id}/staging/{file_path}",
            )
            async with self._create_client_async(s3_config) as s3_client:
                try:
                    await s3_client.delete_object(
                        Bucket=s3_config["bucket"],
                        Key=staging_key,
                    )
                except Exception:
                    pass  # File might not exist in staging
        else:
            # Mark for removal (file exists in git repo)
            staging_files.append({"path": file_path, "_remove": True})
            staging_dict["files"] = staging_files

        artifact.staging = staging_dict
        flag_modified(artifact, "staging")

        return {"status": "staged_for_removal", "path": file_path}

    async def _create_git_zip_file(
        self,
        artifact,
        parent_artifact,
        files: List[str],
        version: str,
        silent: bool,
        artifact_alias: str,
        session,
    ):
        """Create a zip file from a git-storage artifact.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            files: List of file paths to include (None for all files)
            version: Git ref/commit (defaults to HEAD)
            silent: If True, don't increment download count
            artifact_alias: Alias for the zip filename
            session: Database session

        Returns:
            StreamingResponse with zip content
        """
        from hypha.git.repo import S3GitRepo

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Create S3 client factory for the git repo
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Check if repository is empty
        if await repo.is_empty():
            raise HTTPException(
                status_code=404,
                detail="Repository is empty (no commits)",
            )

        # Determine commit SHA from version
        commit_sha = None
        if version and version != "stage":
            refs = await repo.get_refs_async()
            branch_ref = f"refs/heads/{version}".encode()
            if branch_ref in refs:
                commit_sha = refs[branch_ref]
            elif f"refs/tags/{version}".encode() in refs:
                commit_sha = refs[f"refs/tags/{version}".encode()]
            elif len(version) >= 7:
                sha_bytes = bytes.fromhex(version) if len(version) == 40 else None
                if sha_bytes and await repo.has_object_async(sha_bytes):
                    commit_sha = sha_bytes

        # If no files specified, list all files recursively
        if files is None:
            async def list_all_files_recursive(dir_path=""):
                """Recursively list all files in the git tree."""
                items = await repo.list_tree_async(dir_path, commit_sha)
                for item in items:
                    item_path = f"{dir_path}/{item['name']}".strip("/")
                    if item["type"] == "tree":
                        async for sub_path in list_all_files_recursive(item_path):
                            yield sub_path
                    else:
                        yield item_path

            files = [path async for path in list_all_files_recursive()]

        logger.info(f"Creating ZIP file for git artifact: {artifact_alias}")

        # Import LFS pointer parser
        from hypha.git.lfs import LFSPointer

        # Base path for LFS objects
        base_path = f"{s3_config['prefix']}/{artifact.id}"

        async def git_file_stream_generator(file_path: str):
            """Stream file content from git repository in chunks.

            For regular git files, yields content from the git object store.
            For LFS-tracked files, streams directly from S3.
            """
            content = await repo.get_file_content_async(file_path, commit_sha)
            if content is None:
                logger.error(f"File not found in git repo: {file_path}")
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {file_path}",
                )

            # Check if this is an LFS pointer
            lfs_pointer = LFSPointer.parse(content)
            if lfs_pointer is not None:
                # Stream from S3 LFS storage instead of git object store
                lfs_s3_path = lfs_pointer.get_s3_path(base_path)
                logger.info(f"Streaming LFS file for ZIP: {file_path} (oid={lfs_pointer.oid[:8]}...)")

                async with self._create_client_async(s3_config) as s3_client:
                    try:
                        response = await s3_client.get_object(
                            Bucket=s3_config["bucket"],
                            Key=lfs_s3_path,
                        )
                        async for chunk in response["Body"].iter_chunks():
                            yield chunk
                    except Exception as e:
                        logger.error(f"Failed to stream LFS object {lfs_pointer.oid[:8]}...: {e}")
                        raise HTTPException(
                            status_code=404,
                            detail=f"LFS object not found: {lfs_pointer.oid[:8]}...",
                        )
            else:
                # Regular git file - yield in chunks (64KB)
                chunk_size = 1024 * 64
                for i in range(0, len(content), chunk_size):
                    yield content[i : i + chunk_size]

        async def member_files():
            """Yield file metadata and content for stream_zip."""
            modified_at = datetime.now()
            mode = S_IFREG | 0o600
            total_weight = 0

            config = artifact.config or {}
            download_weights = config.get("download_weights", {})

            # Check for special "create-zip-file" download weight
            special_zip_weight = download_weights.get("create-zip-file")
            if special_zip_weight is not None and not silent:
                total_weight = special_zip_weight
                logger.info(f"Using special zip download weight: {total_weight}")

            for path in files:
                logger.info(f"Adding git file to ZIP: {path}")
                # Add to total weight unless silent or special zip weight is set
                if not silent and special_zip_weight is None:
                    download_weight = download_weights.get(path) or 0
                    total_weight += download_weight

                yield (
                    path,
                    modified_at,
                    mode,
                    ZIP_32,
                    git_file_stream_generator(path),
                )

            if total_weight > 0 and not silent:
                logger.info(
                    f"Bumping download count for git artifact: {artifact_alias} by {total_weight}"
                )
                async with session.begin():
                    await self._increment_stat(
                        session,
                        artifact.id,
                        "download_count",
                        increment=total_weight,
                    )
                    await session.commit()

        return StreamingResponse(
            async_stream_zip(member_files()),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={artifact_alias}.zip"
            },
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
            hrid = HRID(delimiter="-", elements=["adjective", "noun", "verb", "adverb"])
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

    @schema_method
    async def create(
        self,
        alias: Optional[str] = PydanticField(
            None,
            description="Unique alias for the artifact within the workspace (e.g., 'my-dataset', 'model-v2'). If not provided, a random alias will be generated. Must contain only alphanumeric characters, hyphens, and underscores."
        ),
        artifact_id: Optional[str] = PydanticField(
            None,
            description="Full artifact identifier in format 'workspace/alias'. Use this OR alias+workspace, not both. Example: 'public/my-dataset'."
        ),
        workspace: Optional[str] = PydanticField(
            None,
            description="Target workspace for the artifact. Ignored if parent_id is provided (inherits parent's workspace). Defaults to current workspace from context."
        ),
        parent_id: Optional[str] = PydanticField(
            None,
            description="Parent collection ID to create this artifact under. Format: 'workspace/collection-alias'. When provided, artifact inherits parent's workspace and permissions."
        ),
        manifest: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Artifact metadata including name, description, version, authors, etc. Must conform to artifact type schema."
        ),
        type: str = PydanticField(
            "generic",
            description="Artifact type. Options: 'generic' (default), 'collection' (folder-like container), 'vector-collection' (vector database), 'application' (Hypha app), 'model' (ML model), 'dataset' (data collection)."
        ),
        config: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Configuration options including permissions, visibility, and storage settings. For vector-collection type: use 'dimension' (int) and 'distance_metric' ('cosine', 'l2', 'inner_product'). Example: {'permissions': {'*': 'r'}, 'dimension': 768, 'distance_metric': 'cosine'}"
        ),
        secrets: Optional[Dict[str, str]] = PydanticField(
            None,
            description="Secret key-value pairs stored encrypted. Use for API keys, credentials, etc. Only accessible to users with write permission."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version tag for the artifact (e.g., 'v1.0', 'latest'). Creates a versioned snapshot. Use 'stage' for staging mode (deprecated - use stage parameter instead)."
        ),
        comment: Optional[str] = PydanticField(
            None,
            description="Version comment or changelog entry. Stored with version metadata for tracking changes."
        ),
        overwrite: bool = PydanticField(
            False,
            description="Whether to overwrite existing artifact with same alias. If False, raises error if artifact exists."
        ),
        stage: bool = PydanticField(
            False,
            description="Create artifact in staging mode. Staged changes can be committed or discarded later. Useful for multi-step artifact creation."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Create a new artifact in the artifact storage system.
        
        This method creates a new artifact with metadata, configuration, and optional parent relationships.
        Artifacts can be files, collections, applications, models, or datasets. When created under a parent
        collection, the artifact inherits the parent's workspace and base permissions.
        
        Returns:
            Dictionary containing artifact details including:
            - id: Full artifact identifier (workspace/alias)
            - alias: The artifact alias
            - workspace: The workspace containing the artifact
            - manifest: The artifact metadata
            - download_count, view_count: Usage statistics
            - created_at, last_modified: Timestamps
            
        Examples:
            # Create a simple artifact
            artifact = await create(
                alias="my-dataset",
                manifest={"name": "My Dataset", "description": "Training data"},
                type="dataset"
            )
            
            # Create artifact in a collection
            artifact = await create(
                alias="image-001",
                parent_id="public/image-collection",
                manifest={"name": "Sample Image"}
            )
            
            # Create with public read permissions
            artifact = await create(
                alias="public-model",
                config={"permissions": {"*": "r"}},
                manifest={"name": "Public Model"}
            )
            
            # Create a vector collection for embeddings
            vector_collection = await create(
                alias="embeddings",
                type="vector-collection",
                config={"dimension": 768, "distance_metric": "cosine"},
                manifest={"name": "Document Embeddings"}
            )
            
        Raises:
            ValueError: If both artifact_id and alias/workspace are specified
            PermissionError: If user lacks permission to create in the workspace
            HTTPException: If artifact with alias already exists and overwrite=False
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        user_info = UserInfo.from_context(context)

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

        # Handle artifact_id parameter (full ID)
        if artifact_id:
            if alias or workspace:
                raise ValueError(
                    "Cannot specify both artifact_id and alias/workspace. "
                    "Use either artifact_id='workspace/alias' or alias='alias' + workspace='workspace'."
                )
            if "/" not in artifact_id:
                raise ValueError(
                    "artifact_id must be in format 'workspace/alias'. "
                    f"Got '{artifact_id}'"
                )
            workspace, alias = artifact_id.split("/", 1)
        elif alias:
            alias = alias.strip()
            if "/" in alias:
                ws, alias = alias.split("/")
                if workspace and ws != workspace:
                    raise ValueError(
                        "Workspace must match the alias workspace, if provided."
                    )
                workspace = ws

        # Validate alias - reserved names not allowed
        if alias and alias in ("-", "~", "_"):
            raise ValueError(
                f"Alias '{alias}' is reserved and cannot be used. Please choose a different alias."
            )

        created_at = int(time.time())
        session = await self._get_session()
        try:
            async with session.begin():
                parent_artifact = None
                parent_workspace = None
                
                # Resolve parent workspace if parent_id is provided
                if parent_id:
                    # Validate parent_id and extract its workspace if it has one
                    if "/" in parent_id:
                        parent_workspace, _ = parent_id.split("/", 1)
                    parent_id = self._validate_artifact_id(parent_id, context)
                    
                    # Get parent artifact with list permission (minimum needed)
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, parent_id, "list", session
                    )
                    parent_id = parent_artifact.id
                    parent_workspace = parent_artifact.workspace
                    
                    if not parent_artifact.manifest:
                        raise ValueError(
                            f"Parent artifact with ID '{parent_id}' must be committed before creating a child artifact."
                        )
                
                # Workspace resolution logic:
                # 1. If workspace is explicitly provided, validate it matches parent's workspace (if parent exists)
                # 2. If no workspace provided but parent exists, use parent's workspace
                # 3. If no workspace and no parent, use current user's workspace
                if workspace:
                    # Workspace explicitly provided
                    if parent_workspace and workspace != parent_workspace:
                        raise ValueError(
                            f"Specified workspace '{workspace}' does not match parent's workspace '{parent_workspace}'. "
                            f"Child artifacts must be created in the same workspace as their parent."
                        )
                elif parent_workspace:
                    # No workspace specified, but parent exists - use parent's workspace
                    workspace = parent_workspace
                else:
                    # No workspace specified and no parent - use current workspace
                    workspace = context["ws"]
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

                        # Check if overwriting an artifact that has children (is_collection)
                        existing_config = existing_artifact.config or {}
                        if overwrite and existing_config.get("is_collection"):
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
                                    f"Cannot overwrite artifact '{existing_artifact.workspace}/{existing_artifact.alias}' as it has {children_count} child artifacts. Remove the children first or use a different alias."
                                )

                        id = existing_artifact.id
                    except KeyError:
                        overwrite = False

                parent_permissions = (
                    parent_artifact.config["permissions"] if parent_artifact else {}
                )
                permissions = config.get("permissions", {}) if config else {}
                # Set creator as admin (this should override parent permission for the creator)
                permissions[user_info.id] = "*"
                config["permissions"] = permissions

                # Set is_collection flag based on type or parent_id
                # If type is "collection", always set is_collection to True
                if type == "collection":
                    config["is_collection"] = True

                # If creating a child artifact (parent_id specified), mark parent as collection
                if parent_artifact:
                    parent_config = parent_artifact.config or {}
                    if not parent_config.get("is_collection"):
                        parent_config["is_collection"] = True
                        parent_artifact.config = parent_config
                        flag_modified(parent_artifact, "config")
                        session.add(parent_artifact)

                # Check permissions based on stage mode
                if parent_artifact:
                    if stage:
                        # For draft creation, check draft permission on parent
                        has_draft = await self._check_permissions(parent_artifact, user_info, "draft")
                        if not has_draft:
                            # Check workspace-level permission as fallback
                            # Workspace owners always have highest permission
                            if not user_info.check_permission(workspace, UserPermission.read_write):
                                raise PermissionError(
                                    f"User does not have permission to create a draft in the collection '{parent_artifact.alias}'."
                                )
                    else:
                        # For direct commit, check attach permission on parent
                        has_attach = await self._check_permissions(parent_artifact, user_info, "attach")
                        if not has_attach:
                            # Check workspace-level permission as fallback
                            # Workspace owners always have highest permission
                            if not user_info.check_permission(workspace, UserPermission.read_write):
                                raise PermissionError(
                                    f"User does not have permission to attach an artifact to the collection '{parent_artifact.alias}'."
                                )

                versions = []
                if not stage:  # Only create initial version if not in staging mode
                    if version in [
                        None,
                        "new",
                        "stage",
                    ]:  # Treat 'stage' as None if stage=False
                        version = "v0"  # Always start with v0 for new artifacts
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
                    # Use "new_version" intent so commit creates version 0
                    staging_dict = {
                        "manifest": manifest,
                        "config": config,
                        "secrets": secrets,
                        "type": type,
                        "files": [],
                        "_intent": "new_version"
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
                    assert "vector_fields" not in config, ("legacy `vector_fields` is not supported, please update the new api which uses `dimension` and `distance_metric` instead.")
                    
                    # Check if vector_config is in manifest (new approach)
                    if manifest and "vector_config" in manifest:
                        vector_config = manifest["vector_config"]
                        # Move vector_config from manifest to config
                        config["dimension"] = vector_config.get("dimension", 384)
                        config["distance_metric"] = vector_config.get("distance_metric", "cosine").lower()
                        config["embedding_model"] = vector_config.get("embedding_model")
                        config["vector_engine"] = vector_config.get("engine", config.get("vector_engine"))
                        # Remove vector_config from manifest since we moved it to config
                        del manifest["vector_config"]
                        new_artifact.manifest = manifest
                    
                    # Unified configuration for both engines
                    dimension = config.get("dimension", 384)
                    distance_metric = config.get("distance_metric", "cosine").lower()
                    
                    # Store the config for future reference
                    config["dimension"] = dimension
                    config["distance_metric"] = distance_metric
                    
                    # Handle S3 config inheritance for s3vector engine
                    if config.get("vector_engine") == "s3vector":
                        # If no s3_config provided, inherit from parent or use artifact's S3 config
                        if "s3_config" not in config:
                            # Use the S3 config from parent artifact or current artifact
                            # The prefix should be the same as artifact files
                            config["s3_config"] = {
                                "endpoint_url": s3_config.get("endpoint_url"),
                                "access_key_id": s3_config.get("access_key"),
                                "secret_access_key": s3_config.get("secret_key"),
                                "region_name": s3_config.get("region_name", "us-east-1"),
                                "bucket": s3_config.get("bucket"),
                                "prefix": s3_config.get("prefix"),  # Use the same prefix as artifact files
                            }
                            # Remove None values
                            config["s3_config"] = {k: v for k, v in config["s3_config"].items() if v is not None}
                    
                    # Store the updated config for future reference
                    new_artifact.config = config
                    
                    await self._create_vector_collection(
                        f"{new_artifact.workspace}/{new_artifact.alias}",
                        dimension=dimension,
                        distance_metric=distance_metric,
                        overwrite=overwrite,
                        config=config,
                    )

                # Initialize Git storage if storage type is "git"
                storage_type = config.get("storage", "raw")
                if storage_type == "git":
                    from functools import partial
                    from hypha.git.repo import S3GitRepo

                    default_branch = config.get("git_default_branch", "main")
                    git_prefix = f"{s3_config['prefix']}/{new_artifact.id}/.git"

                    # Create S3 client factory for Git repo
                    s3_client_factory = partial(self._create_client_async, s3_config)

                    await S3GitRepo.init_bare(
                        s3_client_factory,
                        s3_config["bucket"],
                        git_prefix,
                        default_branch=default_branch.encode(),
                        s3_config=s3_config,
                    )

                    # Store Git configuration
                    config["storage"] = "git"
                    config["git_default_branch"] = default_branch
                    new_artifact.config = config

                    logger.info(f"Initialized Git storage for artifact {new_artifact.workspace}/{new_artifact.alias}")

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

    @schema_method
    async def reset_stats(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact to reset statistics for. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        context: dict = PydanticField(
            ...,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, str]:
        """Reset usage statistics for an artifact.
        
        This method resets the download count and view count to zero. Useful for
        restarting statistics tracking after major updates or for testing purposes.
        Requires write permission on the artifact.
        
        Returns:
            Dictionary with status message confirming statistics were reset
            
        Examples:
            # Reset stats for an artifact
            await reset_stats("my-dataset")
            
            # Reset after major version update
            await edit("my-model", manifest={"version": "2.0"})
            await reset_stats("my-model")
            
        Raises:
            ValueError: If artifact_id is invalid or context missing
            PermissionError: If user lacks write permission
            HTTPException: If artifact not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        user_info = UserInfo.from_context(context)
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

    @schema_method
    async def edit(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to edit. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        manifest: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Updated manifest metadata. Merged with existing manifest. Set individual fields to update them."
        ),
        type: Optional[str] = PydanticField(
            None,
            description="Change artifact type. Options: 'generic', 'collection', 'application', 'model', 'dataset'. Use with caution as it may affect artifact behavior."
        ),
        config: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Updated configuration including permissions. Merged with existing config. Example: {'permissions': {'user123': 'rw'}} to grant read-write access."
        ),
        secrets: Optional[Dict[str, str]] = PydanticField(
            None,
            description="Updated secret key-value pairs. Stored encrypted. Replaces all existing secrets if provided."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version tag for this edit (e.g., 'v2.0'). Creates a new version snapshot. Use 'stage' for staging mode (deprecated - use stage parameter)."
        ),
        comment: Optional[str] = PydanticField(
            None,
            description="Version comment or changelog for this edit. Stored with version metadata."
        ),
        stage: bool = PydanticField(
            False,
            description="Edit in staging mode. Changes are not immediately visible and must be committed. Useful for multi-step edits."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Edit an existing artifact's metadata, configuration, or type.
        
        This method updates an artifact's properties. Changes can be staged for later commit
        or applied immediately. When editing in stage mode, changes accumulate until committed
        or discarded. Manifest and config updates are merged with existing values.
        
        Returns:
            Dictionary containing updated artifact information
            
        Examples:
            # Update manifest metadata
            await edit(
                "my-dataset",
                manifest={"description": "Updated description", "version": "2.0"}
            )
            
            # Stage multiple edits
            await edit("my-model", manifest={"status": "training"}, stage=True)
            await edit("my-model", config={"visibility": "private"}, stage=True)
            await commit("my-model")  # Apply all staged changes
            
            # Update permissions
            await edit(
                "shared-doc",
                config={"permissions": {"team-alpha": "r", "admin": "rw"}}
            )
            
            # Create a new version with changes
            await edit(
                "production-model",
                manifest={"accuracy": 0.95},
                version="v2.0",
                comment="Improved model accuracy"
            )
            
        Raises:
            ValueError: If artifact_id is invalid or context missing
            PermissionError: If user lacks write permission
            HTTPException: If artifact not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        user_info = UserInfo.from_context(context)
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
                    
                    # If editing a committed version (not staging), check mutate permission from parent
                    # This allows collections to prevent modification of existing versions
                    if parent_artifact and artifact.versions:
                        has_mutate = await self._check_permissions(parent_artifact, user_info, "mutate")
                        if not has_mutate:
                            # Check workspace-level permission as fallback
                            if not user_info.check_permission(artifact.workspace, UserPermission.admin):
                                raise PermissionError(
                                    f"Collection '{parent_artifact.alias}' does not allow modification of committed versions (no mutate permission)."
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

                # Determine staging mode behavior:
                # - If stage=True: Enter/continue staging mode
                # - If stage=False and artifact.staging is None: Direct edit and commit
                # - If stage=False and artifact.staging is not None: Apply edits to staging and commit
                is_already_staged = artifact.staging is not None

                if stage:
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
                    # Not in staging mode OR committing previously staged changes
                    if is_already_staged:
                        logger.info(f"Artifact {artifact_id} was in staging mode, now committing staged changes")

                        # Apply staged changes first (if not overridden by new edits)
                        staging_dict = artifact.staging

                        # Use staged values if new values not provided
                        if manifest is None and staging_dict.get("manifest") is not None:
                            manifest = staging_dict["manifest"]
                        if config is None and staging_dict.get("config") is not None:
                            config = staging_dict["config"]
                        if secrets is None and staging_dict.get("secrets") is not None:
                            secrets = staging_dict["secrets"]
                        if type is None and staging_dict.get("type") is not None:
                            type = staging_dict["type"]

                        # If version was not specified, check the staging intent
                        if version is None:
                            staging_intent = staging_dict.get("_intent")
                            if staging_intent == "new_version":
                                version = "new"

                        # Clear staging since we're committing
                        artifact.staging = None
                        flag_modified(artifact, "staging")
                    else:
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

                if stage:
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

    @schema_method
    async def read(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier. Can be: 1) Full ID 'workspace/alias', 2) Just alias (uses current workspace), 3) Special alias like 'applications' for collections."
        ),
        silent: bool = PydanticField(
            False,
            description="If True, does not increment view count statistics. Use for automated/background reads."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version to read (e.g., 'v1.0', 'latest'). Use 'stage' to read staged changes. If not specified, reads current committed version."
        ),
        stage: bool = PydanticField(
            False,
            description="Read staged (uncommitted) changes. Equivalent to version='stage'. Cannot be used with other version values."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Read an artifact's metadata, manifest, and configuration.
        
        This method retrieves complete artifact information including metadata, manifest,
        configuration, and statistics. Supports reading specific versions or staged changes.
        Automatically increments view count unless silent=True.
        
        Returns:
            Dictionary containing:
            - id: Full artifact identifier
            - alias: Artifact alias
            - workspace: Containing workspace
            - manifest: Artifact metadata and content
            - config: Configuration including permissions
            - download_count, view_count: Usage statistics
            - created_at, last_modified: Timestamps
            - versions: List of available versions (if any)
            - parent_id: Parent collection ID (if applicable)
            
        Examples:
            # Read an artifact from current workspace
            artifact = await read("my-dataset")
            
            # Read from specific workspace
            artifact = await read("public/shared-model")
            
            # Read specific version
            artifact = await read("my-model", version="v1.0")
            
            # Read staged changes
            artifact = await read("my-dataset", stage=True)
            
            # Silent read (no stats increment)
            artifact = await read("config-file", silent=True)
            
        Raises:
            ValueError: If artifact_id is invalid or context missing
            PermissionError: If user lacks read permission
            HTTPException: If artifact not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

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

                # For git-storage artifacts, populate versions from git refs (branches and tags)
                config = artifact.config or {}
                if config.get("storage") == "git":
                    try:
                        artifact_data["versions"] = await self._get_git_versions(
                            artifact, parent_artifact
                        )
                    except Exception as e:
                        # If we can't fetch git versions (e.g., empty repo), use empty list
                        logger.warning(f"Failed to fetch git versions for {artifact_id}: {e}")
                        artifact_data["versions"] = []

                # Check if artifact is a collection (has is_collection flag or type is collection for backward compatibility)
                artifact_config = artifact.config or {}
                if artifact_config.get("is_collection") or artifact.type == "collection":
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
                    # Ensure is_collection is set in the response
                    artifact_data["config"]["is_collection"] = True
                if artifact.type == "vector-collection":
                    artifact_data["config"] = artifact_data.get("config", {})
                    artifact_data["config"]["vector_count"] = (
                        await self._count_vectors(
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

    @schema_method
    async def commit(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to commit. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version tag for this commit (e.g., 'v1.0', 'release-2024'). If not specified, updates the current version."
        ),
        comment: Optional[str] = PydanticField(
            None,
            description="Commit message or changelog entry describing the changes. Stored with version metadata for tracking."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Commit staged changes to an artifact.
        
        This method finalizes all staged changes (manifest, files, config) and makes them
        permanent. After commit, staged changes become visible to other users with read
        permission. Optionally creates a new version snapshot. This operation moves files
        from staging to permanent storage.
        
        Returns:
            Dictionary containing:
            - status: Commit status message
            - version: The version that was committed
            - artifact_id: Full artifact identifier
            
        Examples:
            # Commit staged changes
            await edit("my-dataset", manifest={"status": "validated"}, stage=True)
            await put_file("my-dataset", "data.csv", stage=True)
            await commit("my-dataset", comment="Added validated data")
            
            # Commit with version tag
            await edit("model", manifest={"accuracy": 0.95}, stage=True)
            await commit("model", version="v2.0", comment="Improved model")
            
            # Simple commit without version
            await commit("config-file")
            
        Raises:
            ValueError: If artifact_id is invalid or version='stage'
            PermissionError: If user lacks commit permission
            HTTPException: If artifact not found or no staged changes exist
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
        session = await self._get_session()
        assert version != "stage", "Version cannot be 'stage' when committing."
        try:
            async with session.begin():
                # Get the artifact and check commit permission on it
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "commit", session
                )
                
                # If this is a first commit (draft to version), check attach permission on parent
                # Even artifact creators need attach permission to commit their drafts
                if parent_artifact and len(artifact.versions or []) == 0:
                    # This is a draft being attached to parent as first version
                    # Check attach permission on parent collection
                    has_attach = await self._check_permissions(parent_artifact, user_info, "attach")
                    if not has_attach:
                        # Check workspace-level permission as fallback
                        # Workspace owners always have highest permission
                        if not user_info.check_permission(artifact.workspace, UserPermission.read_write):
                            raise PermissionError(
                                f"User does not have permission to attach artifact to the collection '{parent_artifact.alias}'."
                            )

                assert (
                    artifact.staging is not None
                ), "Artifact must be in staging mode to commit."

                # Convert legacy staging format if needed
                artifact.staging = convert_legacy_staging(artifact.staging)

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Handle git-storage artifacts
                    result = await self._commit_git(
                        artifact,
                        parent_artifact,
                        version,
                        comment,
                        user_info,
                        session,
                    )

                    # Update manifest if staged
                    staging_dict = artifact.staging or {}
                    staged_manifest = staging_dict.get("manifest")
                    if staged_manifest is not None:
                        artifact.manifest = staged_manifest
                        flag_modified(artifact, "manifest")

                    # Clear staging after successful commit
                    artifact.staging = None
                    flag_modified(artifact, "staging")

                    # Update artifact timestamp
                    artifact.last_modified = int(time.time())

                    session.add(artifact)
                    await session.commit()

                    logger.info(
                        f"Committed git artifact with ID: {artifact_id}, version: {result.get('version')}"
                    )
                    return self._generate_artifact_data(artifact, parent_artifact)

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

    @schema_method
    async def delete(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to delete. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        delete_files: bool = PydanticField(
            False,
            description="If True, also delete all files stored in S3 for this artifact. If False, only removes database entry."
        ),
        recursive: bool = PydanticField(
            False,
            description="For collections: if True, recursively delete all child artifacts. If False, only delete if collection is empty."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Specific version to delete. If not specified, deletes entire artifact with all versions."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, str]:
        """Delete an artifact and optionally its associated files.
        
        This method removes an artifact from the database and optionally from S3 storage.
        For collections, can recursively delete all children. Deletion is permanent and
        cannot be undone. Requires delete permission on the artifact.
        
        Returns:
            Dictionary with status message confirming deletion
            
        Examples:
            # Delete artifact (keep files in S3)
            await delete("old-dataset")
            
            # Delete artifact and all its files
            await delete("temp-data", delete_files=True)
            
            # Delete a specific version
            await delete("my-model", version="v1.0")
            
            # Recursively delete collection and contents
            await delete("old-collection", recursive=True, delete_files=True)
            
        Raises:
            ValueError: If artifact_id is invalid or context missing
            PermissionError: If user lacks delete permission
            HTTPException: If artifact not found or collection not empty (when recursive=False)
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

        try:
            session = await self._get_session()
            # Get artifact first - check delete permission on the artifact itself
            artifact, parent_artifact = await self._get_artifact_with_permission(
                user_info, artifact_id, "delete", session
            )
            
            # If artifact has a parent and is being detached (not fully deleted),
            # check if user has "detach" permission on the parent collection
            # Even artifact creators need detach permission to remove from collection
            if parent_artifact and version is None and not recursive:
                # This is a detach operation (removing child from parent collection)
                # Check detach permission on the parent collection
                has_detach = await self._check_permissions(parent_artifact, user_info, "detach")
                if not has_detach:
                    # Check workspace-level permission as fallback
                    if not user_info.check_permission(artifact.workspace, UserPermission.admin):
                        raise PermissionError(
                            f"User does not have permission to detach artifact from the collection '{parent_artifact.alias}'."
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
                    await self._delete_vector_collection(
                        f"{artifact.workspace}/{artifact.alias}"
                    )

            s3_config = self._get_s3_config(artifact, parent_artifact)
            if version is None:
                # Delete all versions and the entire artifact
                # Handle recursive deletion first
                if recursive:
                    children = await self.list(artifact_id, context=context)
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
        context: dict = None,
    ):
        """
        Add vectors to a vector collection.
        
        Each vector should have the structure:
        {
            "id": "unique_identifier",
            "vector": [0.1, 0.2, ...],  # The embedding vector
            "manifest": {...}  # Additional metadata/payload
        }
        
        If collection_schema is defined in the artifact config, the manifest
        will be validated against it.
        """
        user_info = UserInfo.from_context(context)
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
                        
                        # Get collection_schema if defined
                        collection_schema = artifact.config.get("collection_schema")
                        
                        # Validate vectors
                        validated_vectors = []
                        for vec in vectors:
                            # Validate vector structure
                            if "id" not in vec:
                                raise ValueError("Each vector must have an 'id' field")
                            if "vector" not in vec:
                                raise ValueError("Each vector must have a 'vector' field")
                            
                            # Validate manifest against schema if provided
                            manifest = vec.get("manifest", {})
                            if collection_schema and manifest:
                                try:
                                    import jsonschema
                                    jsonschema.validate(manifest, collection_schema)
                                except jsonschema.ValidationError as e:
                                    raise ValueError(f"Manifest validation failed for vector {vec['id']}: {str(e)}")
                            
                            # Use vector as-is (no transformation needed)
                            validated_vectors.append(vec)
                        
                        async with self._create_client_async(s3_config) as s3_client:
                            prefix = safe_join(
                                s3_config["prefix"],
                                f"{artifact.id}/v0",
                            )
                            
                            # Get embedding model configuration from the artifact config
                            embedding_model = artifact.config.get("embedding_model")
                            
                            result = await self._add_vectors_to_collection(
                                f"{artifact.workspace}/{artifact.alias}",
                                validated_vectors,
                                embedding_model=embedding_model,
                            )
                        logger.info(f"Added vectors to artifact with ID: {artifact_id}")
                        return result  # Return the list of IDs
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

    @schema_method
    async def duplicate(
        self,
        source_artifact_id: str = PydanticField(
            ...,
            description="Source artifact to duplicate. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        target_alias: Optional[str] = PydanticField(
            None,
            description="Alias for the duplicated artifact. If not provided, generates one like 'original-alias-copy-1'."
        ),
        target_workspace: Optional[str] = PydanticField(
            None,
            description="Target workspace for the duplicate. If not provided, uses source artifact's workspace."
        ),
        target_parent_id: Optional[str] = PydanticField(
            None,
            description="Parent collection for the duplicate. Format: 'workspace/collection-alias'. Overrides target_workspace if provided."
        ),
        overwrite: bool = PydanticField(
            False,
            description="Whether to overwrite if artifact with target_alias already exists."
        ),
        remove_secrets: bool = PydanticField(
            True,
            description="Whether to remove secrets from the duplicate. Recommended True when copying to different workspace."
        ),
        stage: bool = PydanticField(
            False,
            description="Create duplicate in staging mode. Useful for reviewing before committing."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Duplicate an artifact with all its files and metadata.
        
        This method creates a complete copy of an artifact, including all files, manifest,
        and configuration. Can copy within the same workspace or to a different workspace.
        Secrets are removed by default for security when copying across workspaces.
        
        Returns:
            Dictionary containing the new artifact's information
            
        Examples:
            # Simple duplicate in same workspace
            copy = await duplicate("my-dataset")
            
            # Copy to different workspace
            copy = await duplicate(
                "public/template",
                target_workspace="my-workspace",
                target_alias="my-copy"
            )
            
            # Copy into a collection
            copy = await duplicate(
                "source-file",
                target_parent_id="my-workspace/my-collection"
            )
            
            # Overwrite existing artifact
            copy = await duplicate(
                "v2-model",
                target_alias="latest-model",
                overwrite=True
            )
            
        Raises:
            PermissionError: If user lacks read permission on source or write permission on target
            HTTPException: If source not found or target exists (when overwrite=False)
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        
        source_artifact_id = self._validate_artifact_id(source_artifact_id, context)
        user_info = UserInfo.from_context(context)
        
        session = await self._get_session()
        try:
            async with session.begin():
                # Get source artifact with read permission
                source_artifact, source_parent = await self._get_artifact_with_permission(
                    user_info, source_artifact_id, "read", session
                )
                
                if not source_artifact.manifest:
                    raise ValueError(
                        f"Source artifact '{source_artifact_id}' must be committed before it can be duplicated."
                    )
                
                # Use source workspace if target workspace not specified
                if target_workspace is None:
                    target_workspace = source_artifact.workspace
                
                # Check permissions for target workspace
                if target_parent_id:
                    target_parent_id = self._validate_artifact_id(target_parent_id, context)
                    target_parent, _ = await self._get_artifact_with_permission(
                        user_info, target_parent_id, "create", session
                    )
                    # Ensure target workspace matches parent's workspace
                    if target_workspace and target_workspace != target_parent.workspace:
                        raise ValueError(
                            "Target workspace must match the parent artifact's workspace."
                        )
                    target_workspace = target_parent.workspace
                else:
                    # Creating orphan artifact in target workspace
                    if not user_info.check_permission(
                        target_workspace, UserPermission.read_write
                    ):
                        raise PermissionError(
                            f"User does not have permission to create an artifact in workspace '{target_workspace}'."
                        )
                    target_parent = None
                
                # Check if target workspace exists and is persistent
                winfo = await self.store.get_workspace_info(target_workspace, load=True)
                if not winfo:
                    raise ValueError(f"Workspace '{target_workspace}' does not exist.")
                if not winfo.persistent:
                    raise PermissionError(
                        f"Cannot create artifact in a non-persistent workspace `{target_workspace}`."
                    )
        except Exception:
            raise
        finally:
            await session.close()

        # Create the new artifact with copied manifest and config
        new_config = dict(source_artifact.config) if source_artifact.config else {}
        # Remove source-specific config like zenodo info
        new_config.pop("zenodo", None)
        new_config.pop("id_parts", None)
        
        # Copy permissions but ensure current user has full access
        if "permissions" not in new_config:
            new_config["permissions"] = {}
        new_config["permissions"][user_info.id] = "*"
        
        # Copy manifest and optionally remove secrets
        new_manifest = dict(source_artifact.manifest)
        if remove_secrets and "secrets" in new_manifest:
            del new_manifest["secrets"]
        
        # Create the new artifact in stage mode so we can add files before committing
        new_artifact_info = await self.create(
            alias=target_alias,
            workspace=target_workspace,
            parent_id=target_parent_id,
            manifest=new_manifest,
            type=source_artifact.type,
            config=new_config,
            secrets=None,  # Don't copy secrets for security
            overwrite=overwrite,
            version="stage",  # Create in stage mode to add files
            context=context,
        )
        
        new_artifact_id = new_artifact_info["id"]
        
        # Copy all files from source to target
        source_s3_config = self._get_s3_config(source_artifact, source_parent)
        
        # Get target artifact for S3 config
        session = await self._get_session()
        try:
            async with session.begin():
                target_artifact, target_parent = await self._get_artifact_with_permission(
                    user_info, new_artifact_id, "put_file", session
                )
                target_s3_config = self._get_s3_config(target_artifact, target_parent)
                
                # Determine target version - since we created in stage mode, check staging
                # The target artifact was just created in stage mode, so files go to v0
                # which is where staging files are stored
                target_version = 0
                
                # List all files in the source artifact
                source_versions = source_artifact.versions or []
                latest_version = len(source_versions) - 1 if source_versions else 0
                source_prefix = safe_join(
                    source_s3_config["prefix"],
                    f"{source_artifact.id}/v{latest_version}/"
                )
                
                async with self._create_client_async(source_s3_config) as source_s3:
                    async with self._create_client_async(target_s3_config) as target_s3:
                        # List and copy all files
                        paginator = source_s3.get_paginator("list_objects_v2")
                        page_iterator = paginator.paginate(
                            Bucket=source_s3_config["bucket"],
                            Prefix=source_prefix
                        )
                        
                        file_count = 0
                        async for page in page_iterator:
                            if "Contents" in page:
                                for obj in page["Contents"]:
                                    # Get relative file path
                                    file_path = obj["Key"][len(source_prefix):]
                                    if not file_path:  # Skip the prefix itself
                                        continue
                                    
                                    # Copy file from source to target
                                    source_key = obj["Key"]
                                    # Since we're creating in stage mode, files should go to staging location
                                    # which will become v0 after commit
                                    target_key = safe_join(
                                        target_s3_config["prefix"],
                                        f"{target_artifact.id}/v{target_version}/{file_path}"
                                    )
                                    
                                    # Download from source and upload to target
                                    response = await source_s3.get_object(
                                        Bucket=source_s3_config["bucket"],
                                        Key=source_key
                                    )
                                    body = await response["Body"].read()
                                    
                                    await target_s3.put_object(
                                        Bucket=target_s3_config["bucket"],
                                        Key=target_key,
                                        Body=body,
                                        ContentType=response.get("ContentType", "application/octet-stream"),
                                    )
                                    file_count += 1
                        
                        # Update file count in the new artifact
                        target_artifact.file_count = file_count
                        await session.commit()

                        logger.info(
                            f"Successfully duplicated artifact '{source_artifact_id}' to '{new_artifact_id}' with {file_count} files"
                        )
        except Exception:
            raise
        finally:
            await session.close()

        # Conditionally commit the new artifact after copying files
        if not stage:
            await self.commit(new_artifact_id, context=context)
            # Return the committed artifact info
            return await self.read(new_artifact_id, context=context)
        else:
            # Return the staged artifact info
            return await self.read(new_artifact_id, version="stage", context=context)

    async def search_vectors(
        self,
        artifact_id: str,
        query: Optional[Dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: Optional[int] = 5,
        offset: Optional[int] = 0,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
        context: dict = None,
    ):
        user_info = UserInfo.from_context(context)
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "search_vectors", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."

                # Transform query for PgVector format
                query_vector = None
                if query:
                    if "vector" in query:
                        query_vector = query["vector"]
                    elif isinstance(query, dict):
                        # Find the first vector-like field
                        for key, value in query.items():
                            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float)):
                                query_vector = value
                                break
                    elif isinstance(query, (list, tuple)):
                        query_vector = query
                
                # Get embedding model configuration from the artifact config
                embedding_model = artifact.config.get("embedding_model")
                
                return await self._search_vectors_in_collection(
                    f"{artifact.workspace}/{artifact.alias}",
                    query_vector=query_vector,
                    embedding_model=embedding_model,
                    filters=filters,
                    limit=limit,
                    offset=offset,
                    include_vectors=False if not return_fields else "vector" in return_fields,
                    order_by=order_by,
                    pagination=pagination,
                )
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

    async def remove_vectors(
        self,
        artifact_id: str,
        ids: list,
        context: dict = None,
    ):
        user_info = UserInfo.from_context(context)
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
                    await self._remove_vectors_from_collection(
                        f"{artifact.workspace}/{artifact.alias}",
                        ids,
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
        user_info = UserInfo.from_context(context)
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "get_vector", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."
                return await self._get_vector_from_collection(
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
        user_info = UserInfo.from_context(context)
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "list_vectors", session
                )
                assert (
                    artifact.type == "vector-collection"
                ), "Artifact must be a vector collection."
                return await self._list_vectors_in_collection(
                    f"{artifact.workspace}/{artifact.alias}",
                    offset=offset,
                    limit=limit,
                    include_vectors=False if not return_fields else "vector" in return_fields,
                    order_by=order_by,
                    pagination=pagination,
                )
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

    @schema_method
    async def put_file(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to upload file to. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        file_path: Union[str, List[str]] = PydanticField(
            ...,
            description="Path(s) where to store the file(s) within the artifact. Can be a single string or list of strings for batch upload. Use forward slashes for nested paths (e.g., 'data/train.csv', ['models/weights.pt', 'data/train.csv'])."
        ),
        download_weight: Union[float, List[float]] = PydanticField(
            0,
            description="Weight factor(s) for download statistics. Can be a single value or list matching file_path. Higher weights indicate more important files.",
        ),
        use_proxy: Optional[bool] = PydanticField(
            None,
            description="If True, returns proxy URL(s) through Hypha server. If False, returns direct S3 URL(s). Default auto-selects based on setup."
        ),
        use_local_url: bool = PydanticField(
            False,
            description="If True, returns localhost URL(s) instead of public URL(s). Useful for server-side uploads."
        ),
        expires_in: int = PydanticField(
            3600,
            description="URL expiration time in seconds. Default 1 hour. Maximum 7 days (604800 seconds).",
            ge=60,
            le=604800
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Union[str, Dict[str, str]]:
        """Generate presigned URL(s) for uploading file(s) to an artifact.

        This method supports both single file and batch operations:
        - Pass a string for file_path: returns a single URL string
        - Pass a list for file_path: returns a dict mapping paths to URLs (optimized batch mode)

        Batch mode is significantly faster (75-355x) when uploading many files, using a single
        database transaction and S3 client for all operations.

        Returns:
            Single URL string (if file_path is string) or Dict[str, str] (if file_path is list)

        Examples:
            # Single file upload
            url = await put_file("my-dataset", "data.csv")

            # Batch upload (optimized)
            urls = await put_file("my-dataset", ["train.csv", "test.csv", "val.csv"])
            # Returns: {"train.csv": "url1", "test.csv": "url2", "val.csv": "url3"}

            # Batch with download weights
            urls = await put_file(
                "my-dataset",
                ["important.csv", "optional.txt"],
                download_weight=[1.0, 0.1]
            )

        Raises:
            ValueError: If artifact_id invalid, download_weight negative, or list lengths mismatch
            PermissionError: If user lacks write permission
            HTTPException: If artifact not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        # Determine if this is batch mode
        is_batch = isinstance(file_path, list)

        if is_batch:
            # Batch mode: validate inputs
            file_paths = file_path
            if not file_paths:
                raise ValueError("file_path list must contain at least one path")

            # Handle download_weight
            if isinstance(download_weight, list):
                if len(download_weight) != len(file_paths):
                    raise ValueError(f"download_weight list length ({len(download_weight)}) must match file_path list length ({len(file_paths)})")
                download_weights = download_weight
            else:
                # Single value: apply to all files
                download_weights = [download_weight] * len(file_paths)

            # Validate all weights
            for i, weight in enumerate(download_weights):
                if weight < 0:
                    raise ValueError(f"Download weight must be non-negative: {weight} at index {i}")
        else:
            # Single mode: wrap in lists for uniform handling
            file_paths = [file_path]
            download_weights = [download_weight if not isinstance(download_weight, list) else download_weight[0]]

            if download_weights[0] < 0:
                raise ValueError(f"Download weight must be non-negative: {download_weights[0]}")

        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
        session = await self._get_session()

        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )

                # Require artifact to be in staging mode
                assert artifact.staging is not None, "Artifact must be in staging mode."

                # Convert legacy staging format if needed
                artifact.staging = convert_legacy_staging(artifact.staging)

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Handle git-storage artifacts
                    # Note: We don't need row locking here because:
                    # 1. File uploads go directly to S3 staging area
                    # 2. Commit now lists files from S3 directly, not from manifest
                    # The manifest update below is best-effort for metadata tracking
                    presigned_urls = {}
                    staging_dict = artifact.staging
                    staging_files = staging_dict.get("files", [])

                    for fpath, dweight in zip(file_paths, download_weights):
                        url = await self._put_git_file(
                            artifact,
                            parent_artifact,
                            fpath,
                            dweight,
                            expires_in,
                            use_proxy,
                            use_local_url,
                        )
                        presigned_urls[fpath] = url

                        # Add file to staging dict for tracking
                        if not any(f.get("path") == fpath for f in staging_files):
                            file_info = {"path": fpath}
                            if dweight > 0:
                                file_info["download_weight"] = dweight
                            staging_files.append(file_info)
                            staging_dict["files"] = staging_files

                    flag_modified(artifact, "staging")

                    # Save to S3 for git artifacts
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    await self._save_version_to_s3(0, artifact, s3_config)
                    session.add(artifact)
                    await session.commit()

                    logger.info(
                        f"Put {len(file_paths)} file(s) to git artifact with ID: {artifact_id}"
                    )

                    # Return single URL or dict based on input type
                    if is_batch:
                        return presigned_urls
                    else:
                        return presigned_urls[file_paths[0]]

                # Staging mode - files go to staging version
                staging_dict = artifact.staging
                has_new_version_intent = staging_dict.get("_intent") == "new_version"

                if has_new_version_intent:
                    # Creating new version - use next index
                    target_version_index = len(artifact.versions or [])
                else:
                    # Non-staging mode - files go to current version
                    target_version_index = max(0, len(artifact.versions or []) - 1)

                logger.info(
                    f"Uploading {len(file_paths)} file(s) to staging version {target_version_index} (has_new_version_intent: {has_new_version_intent})"
                )

                s3_config = self._get_s3_config(artifact, parent_artifact)

                # Use single S3 client for all URL generations
                async with self._create_client_async(s3_config) as s3_client:
                    presigned_urls = {}

                    for fpath, dweight in zip(file_paths, download_weights):
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{target_version_index}/{fpath}",
                        )
                        presigned_url = await s3_client.generate_presigned_url(
                            "put_object",
                            Params={"Bucket": s3_config["bucket"], "Key": file_key},
                            ExpiresIn=expires_in,
                        )

                        # Use proxy based on use_proxy parameter (None means use server config)
                        if use_proxy is None:
                            use_proxy_resolved = self.s3_controller.enable_s3_proxy
                        else:
                            use_proxy_resolved = use_proxy

                        if use_proxy_resolved:
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
                        elif use_local_url and not use_proxy_resolved:
                            # For S3 direct access with local URL
                            if use_local_url is not True:
                                presigned_url = presigned_url.replace(
                                    s3_config["endpoint_url"], use_local_url
                                )

                        presigned_urls[fpath] = presigned_url

                        # Add file to staging dict
                        staging_files = staging_dict.get("files", [])

                        # Check if file already exists in staging
                        if not any(f["path"] == fpath for f in staging_files):
                            file_info = {"path": fpath}
                            # Only store download_weight if it's greater than 0 to save storage
                            if dweight > 0:
                                file_info["download_weight"] = dweight
                            staging_files.append(file_info)
                            staging_dict["files"] = staging_files

                    flag_modified(artifact, "staging")

                # Save artifact state to S3 once for all files
                await self._save_version_to_s3(target_version_index, artifact, s3_config)
                session.add(artifact)
                await session.commit()
                logger.info(
                    f"Put {len(file_paths)} file(s) to artifact with ID: {artifact_id} (target version: {target_version_index})"
                )

            # Return single URL or dict based on input type
            if is_batch:
                return presigned_urls
            else:
                return presigned_urls[file_paths[0]]

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
        user_info = UserInfo.from_context(context)

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
        user_info = UserInfo.from_context(context)

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

    @schema_method
    async def remove_file(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact to remove file from. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        file_path: str = PydanticField(
            ...,
            description="Path to the file to remove. Use forward slashes for nested paths (e.g., 'data/old.csv', 'temp/cache.bin')."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, str]:
        """Remove a file from an artifact.
        
        This method removes a file from the artifact's storage. The removal is staged
        and must be committed to take permanent effect. File removal updates the
        artifact's file count in the staging manifest.
        
        Returns:
            Dictionary with status message confirming file removal
            
        Examples:
            # Remove a single file
            await remove_file("my-dataset", "obsolete-data.csv")
            await commit("my-dataset", comment="Removed obsolete data")
            
            # Remove file from nested directory
            await remove_file("my-project", "temp/debug.log")
            
            # Remove multiple files (call multiple times)
            await remove_file("cleanup-target", "file1.txt")
            await remove_file("cleanup-target", "file2.txt")
            await commit("cleanup-target")
            
        Raises:
            ValueError: If artifact_id is invalid
            PermissionError: If user lacks write permission
            AssertionError: If artifact not in staging mode
            HTTPException: If artifact or file not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
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

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Handle git-storage artifacts
                    result = await self._remove_git_file(
                        artifact,
                        parent_artifact,
                        file_path,
                    )
                    session.add(artifact)
                    await session.commit()
                    logger.info(
                        f"Staged removal of file '{file_path}' from git artifact {artifact_id}"
                    )
                    return result

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

    @schema_method
    async def get_file(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to download file(s) from. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        file_path: Union[str, List[str]] = PydanticField(
            ...,
            description="Path(s) to the file(s) within the artifact. Can be a single string or list of strings for batch download. Use forward slashes for nested paths."
        ),
        silent: bool = PydanticField(
            False,
            description="If True, does not increment download count statistics. Use for automated/background downloads."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version to download from (e.g., 'v1.0', 'latest'). If not specified, uses current version."
        ),
        stage: bool = PydanticField(
            False,
            description="Download from staged (uncommitted) files. Cannot be used with version parameter."
        ),
        use_proxy: Optional[bool] = PydanticField(
            None,
            description="If True, returns proxy URL(s) through Hypha server. If False, returns direct S3 URL(s). Default auto-selects."
        ),
        use_local_url: Union[bool, str] = PydanticField(
            False,
            description="If True, returns localhost URL(s). If 'absolute', returns absolute file path(s). Default returns public URL(s)."
        ),
        expires_in: int = PydanticField(
            3600,
            description="URL expiration time in seconds. Default 1 hour. Maximum 7 days (604800 seconds).",
            ge=60,
            le=604800
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Union[str, Dict[str, str]]:
        """Generate presigned URL(s) for downloading file(s) from an artifact.

        This method supports both single file and batch operations:
        - Pass a string for file_path: returns a single URL string
        - Pass a list for file_path: returns a dict mapping paths to URLs (optimized batch mode)

        Batch mode is significantly faster (75-355x) when downloading many files, using a single
        database transaction and S3 client for all operations.

        Returns:
            Single URL string (if file_path is string) or Dict[str, str] (if file_path is list)

        Examples:
            # Single file download
            url = await get_file("my-dataset", "data.csv")

            # Batch download (optimized)
            urls = await get_file("my-dataset", ["train.csv", "test.csv", "val.csv"])
            # Returns: {"train.csv": "url1", "test.csv": "url2", "val.csv": "url3"}

            # Download from specific version
            url = await get_file("my-model", "weights.pt", version="v1.0")

        Raises:
            ValueError: If artifact_id invalid or stage used with version
            PermissionError: If user lacks read permission
            FileNotFoundError: If any requested file not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        # Determine if this is batch mode
        is_batch = isinstance(file_path, list)

        if is_batch:
            file_paths = file_path
            if not file_paths:
                raise ValueError("file_path list must contain at least one path")
        else:
            file_paths = [file_path]

        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

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

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Handle git-storage artifacts
                    presigned_urls = {}
                    for fpath in file_paths:
                        url = await self._get_git_file_url(
                            artifact,
                            parent_artifact,
                            fpath,
                            version,
                            expires_in,
                            use_proxy,
                            use_local_url,
                            user_info,
                        )
                        presigned_urls[fpath] = url

                    # Return single URL or dict based on input type
                    if is_batch:
                        return presigned_urls
                    else:
                        return presigned_urls[file_paths[0]]

                version_index = self._get_version_index(artifact, version)
                s3_config = self._get_s3_config(artifact, parent_artifact)

                # Use single S3 client for all URL generations
                async with self._create_client_async(s3_config) as s3_client:
                    presigned_urls = {}

                    for fpath in file_paths:
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{version_index}/{fpath}",
                        )

                        # Check if file exists
                        try:
                            await s3_client.head_object(
                                Bucket=s3_config["bucket"], Key=file_key
                            )
                        except ClientError:
                            raise FileNotFoundError(
                                f"File '{fpath}' does not exist in the artifact."
                            )

                        # Generate presigned URL
                        presigned_url = await s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": s3_config["bucket"], "Key": file_key},
                            ExpiresIn=expires_in,
                        )

                        # Handle proxy/local URL replacement
                        if use_proxy is None:
                            use_proxy_resolved = self.s3_controller.enable_s3_proxy
                        else:
                            use_proxy_resolved = use_proxy

                        if use_proxy_resolved:
                            if use_local_url:
                                local_proxy_url = f"{self.store.local_base_url}/s3" if use_local_url is True else f"{use_local_url}/s3"
                                presigned_url = presigned_url.replace(
                                    s3_config["endpoint_url"], local_proxy_url
                                )
                            elif s3_config["public_endpoint_url"]:
                                presigned_url = presigned_url.replace(
                                    s3_config["endpoint_url"],
                                    s3_config["public_endpoint_url"],
                                )
                        elif use_local_url and not use_proxy_resolved:
                            if use_local_url is not True:
                                presigned_url = presigned_url.replace(
                                    s3_config["endpoint_url"], use_local_url
                                )

                        presigned_urls[fpath] = presigned_url

                # Increment download counts unless silent
                if not silent:
                    if artifact.config and "download_weights" in artifact.config:
                        download_weights_config = artifact.config.get("download_weights", {})
                    else:
                        download_weights_config = {}

                    # Calculate total download weight for all files
                    total_weight = 0
                    for fpath in file_paths:
                        download_weight = download_weights_config.get(fpath) or 0
                        total_weight += download_weight

                    if total_weight > 0:
                        await self._increment_stat(
                            session,
                            artifact.id,
                            "download_count",
                            increment=total_weight,
                        )

                await session.commit()

            # Return single URL or dict based on input type
            if is_batch:
                return presigned_urls
            else:
                return presigned_urls[file_paths[0]]

        except Exception as e:
            raise e
        finally:
            await session.close()

    @schema_method
    async def read_file(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to read file from. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        file_path: str = PydanticField(
            ...,
            description="Path to the file within the artifact. Use forward slashes for nested paths (e.g., 'data/train.csv')."
        ),
        format: str = PydanticField(
            "text",
            description="Output format: 'text' (decoded string), 'binary' (raw bytes), 'base64' (base64-encoded string), or 'json' (parsed JSON object)."
        ),
        encoding: str = PydanticField(
            "utf-8",
            description="Text encoding to use when format='text' or 'json'. Common values: 'utf-8', 'ascii', 'latin-1'."
        ),
        offset: int = PydanticField(
            0,
            description="Byte offset to start reading from. Must be >= 0. Used for partial/chunked reads.",
            ge=0
        ),
        limit: Optional[int] = PydanticField(
            None,
            description="Maximum number of bytes to read. Defaults to 64KB. Maximum allowed is 10MB (10485760)."
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version to read from (e.g., 'v1.0', 'latest', commit SHA for git). Use 'stage' to read staged changes."
        ),
        stage: bool = PydanticField(
            False,
            description="Read from staged (uncommitted) storage. Cannot be used with version parameter."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Read file content from an artifact with flexible encoding options.

        This method reads file content directly without generating presigned URLs,
        making it convenient for AI agents and programmatic access. Supports both
        git-based and raw S3-based artifact storage.

        For git storage:
        - Reads from git commits using version (branch/tag/commit SHA)
        - Use stage=True to read from git staging area
        - Supports LFS (Large File Storage) for large files

        For raw storage:
        - Reads from S3 versioned storage
        - Use stage=True to read from staging version

        Returns:
            Dictionary with 'name' (file path) and 'content' (file data).
            Content type depends on format parameter:
            - text: decoded string using specified encoding
            - binary: raw bytes
            - base64: base64-encoded ASCII string
            - json: parsed Python dict/list from JSON

        Examples:
            # Read text file
            result = await read_file("my-dataset", "README.md")
            text = result["content"]  # string

            # Read JSON file as parsed object
            result = await read_file("my-dataset", "config.json", format="json")
            config = result["content"]  # dict or list

            # Read binary file as base64
            result = await read_file("my-model", "weights.bin", format="base64")
            binary_data = base64.b64decode(result["content"])

            # Read specific version
            result = await read_file("my-dataset", "data.csv", version="v1.0")

            # Partial read (chunked)
            result = await read_file("my-dataset", "large.txt", offset=1000, limit=5000)

        Raises:
            ValueError: If artifact_id invalid, format invalid, or limit exceeded
            PermissionError: If user lacks read permission
            FileNotFoundError: If file not found (via HTTPException 404)
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

        if stage:
            assert version is None or version == "stage", "You cannot specify a version when using stage mode."
            version = "stage"

        # Validate offset/limit
        default_limit = 64 * 1024  # 64KB
        max_limit = 10 * 1024 * 1024  # 10MB
        read_limit = default_limit if limit is None else int(limit)
        if read_limit <= 0:
            raise ValueError("Limit must be a positive integer.")
        if read_limit > max_limit:
            raise ValueError(f"Limit cannot exceed 10MB ({max_limit} bytes).")

        session = await self._get_session(read_only=True)
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "get_file", session
                )

                config = artifact.config or {}

                if config.get("storage") == "git":
                    # Handle git storage
                    data_bytes = await self._read_git_file_content(
                        artifact, parent_artifact, file_path, version, offset, read_limit
                    )
                else:
                    # Handle raw S3 storage
                    version_index = self._get_version_index(artifact, version)
                    s3_config = self._get_s3_config(artifact, parent_artifact)

                    async with self._create_client_async(s3_config) as s3_client:
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{version_index}/{file_path}",
                        )
                        # Use ranged read for safety
                        end_byte = offset + read_limit - 1
                        try:
                            response = await s3_client.get_object(
                                Bucket=s3_config["bucket"],
                                Key=file_key,
                                Range=f"bytes={offset}-{end_byte}",
                            )
                            data_bytes = await response["Body"].read()
                        except ClientError as e:
                            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"File '{file_path}' not found in artifact."
                                )
                            raise
        except Exception:
            raise
        finally:
            await session.close()

        # Format the output
        fmt = (format or "text").lower()
        if fmt == "text":
            content: Any = data_bytes.decode(encoding or "utf-8")
        elif fmt == "base64":
            content = base64.b64encode(data_bytes).decode("ascii")
        elif fmt == "binary":
            content = data_bytes
        elif fmt == "json":
            text_content = data_bytes.decode(encoding or "utf-8")
            content = json.loads(text_content)
        else:
            raise ValueError("Invalid format. Expected 'text', 'binary', 'base64', or 'json'.")

        return {"name": file_path, "content": content}

    async def _read_git_file_content(
        self,
        artifact,
        parent_artifact,
        file_path: str,
        version: Optional[str],
        offset: int,
        limit: int,
    ) -> bytes:
        """Read file content from a git-storage artifact.

        Args:
            artifact: The artifact model instance
            parent_artifact: Parent artifact for S3 config inheritance
            file_path: Path within the git repository
            version: Git ref/commit (defaults to HEAD), or 'stage' for staging area
            offset: Byte offset to start reading from
            limit: Maximum bytes to read

        Returns:
            File content as bytes (sliced according to offset/limit)
        """
        from hypha.git.repo import S3GitRepo
        from hypha.git.lfs import LFSPointer

        s3_config = self._get_s3_config(artifact, parent_artifact)

        # Check if reading from staging area
        if version == "stage":
            # Read from git staging area in S3
            staging_key = safe_join(
                s3_config["prefix"],
                f"{artifact.id}/staging/{file_path}",
            )
            async with self._create_client_async(s3_config) as s3_client:
                try:
                    end_byte = offset + limit - 1
                    response = await s3_client.get_object(
                        Bucket=s3_config["bucket"],
                        Key=staging_key,
                        Range=f"bytes={offset}-{end_byte}",
                    )
                    return await response["Body"].read()
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                        raise HTTPException(
                            status_code=404,
                            detail=f"Staged file '{file_path}' not found."
                        )
                    raise

        # Read from git repository
        s3_client_factory = partial(self._create_client_async, s3_config)
        prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
        repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
        await repo.initialize()

        # Check if repository is empty
        if await repo.is_empty():
            raise HTTPException(
                status_code=404,
                detail="Repository is empty (no commits)",
            )

        # Resolve version to commit SHA
        commit_sha = await self._resolve_git_version(repo, version)

        # Get file content
        content = await repo.get_file_content_async(file_path.strip("/"), commit_sha)
        if content is None:
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_path}' not found in repository."
            )

        # Check if this is an LFS pointer
        lfs_pointer = LFSPointer.parse(content)
        if lfs_pointer is not None:
            # Read from S3 LFS storage
            base_path = f"{s3_config['prefix']}/{artifact.id}"
            lfs_s3_path = lfs_pointer.get_s3_path(base_path)

            async with self._create_client_async(s3_config) as s3_client:
                try:
                    end_byte = offset + limit - 1
                    response = await s3_client.get_object(
                        Bucket=s3_config["bucket"],
                        Key=lfs_s3_path,
                        Range=f"bytes={offset}-{end_byte}",
                    )
                    return await response["Body"].read()
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                        raise HTTPException(
                            status_code=404,
                            detail=f"LFS object not found: {lfs_pointer.oid[:8]}..."
                        )
                    raise

        # Regular git blob content - apply offset/limit
        return content[offset:offset + limit]

    @schema_method
    async def write_file(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to write file to. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        file_path: str = PydanticField(
            ...,
            description="Path where to store the file within the artifact. Use forward slashes for nested paths (e.g., 'data/output.json')."
        ),
        content: Any = PydanticField(
            ...,
            description="File content. Interpreted based on 'format': string for text/base64, bytes for binary, dict/list for json."
        ),
        format: str = PydanticField(
            "text",
            description="Input format: 'text' (string to encode), 'binary' (raw bytes), 'base64' (base64 string to decode), or 'json' (dict/list to serialize)."
        ),
        encoding: str = PydanticField(
            "utf-8",
            description="Text encoding to use when format='text' or 'json'. Common values: 'utf-8', 'ascii', 'latin-1'."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Write file content into an artifact with flexible encoding options.

        This method writes file content directly without requiring presigned URLs,
        making it convenient for AI agents and programmatic access. Supports both
        git-based and raw S3-based artifact storage.

        IMPORTANT: Requires the artifact to be in staging mode (call edit() with stage=True first).
        After writing files, call commit() to finalize changes.

        For git storage:
        - Files are written to the git staging area
        - Call commit() to add files to git repository

        For raw storage:
        - Files are written to the staging version in S3
        - Call commit() to finalize the version

        Returns:
            Dictionary with 'success' (bool) and 'bytes_written' (int).

        Examples:
            # Write text file
            await artifact_manager.edit("my-dataset", stage=True)
            await write_file("my-dataset", "README.md", "# My Dataset\\n...")
            await artifact_manager.commit("my-dataset")

            # Write binary file using base64
            import base64
            binary_data = b"..."
            encoded = base64.b64encode(binary_data).decode()
            await write_file("my-model", "weights.bin", encoded, format="base64")

            # Write JSON data directly (no manual serialization needed)
            data = {"key": "value", "items": [1, 2, 3]}
            await write_file("my-dataset", "config.json", data, format="json")

        Raises:
            ValueError: If artifact_id invalid, format invalid, or artifact not in staging mode
            PermissionError: If user lacks write permission
            AssertionError: If artifact is not in staging mode
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )

                if artifact.staging is None:
                    raise ValueError(
                        "Artifact must be in staging mode. Call edit(artifact_id, stage=True) first."
                    )
                artifact.staging = convert_legacy_staging(artifact.staging)

                # Convert content to bytes based on format
                fmt = (format or "text").lower()
                if fmt == "text":
                    if not isinstance(content, str):
                        raise ValueError("When format='text', content must be a string.")
                    body = content.encode(encoding or "utf-8")
                elif fmt == "base64":
                    if not isinstance(content, str):
                        raise ValueError("When format='base64', content must be a base64 string.")
                    body = base64.b64decode(content)
                elif fmt == "binary":
                    if isinstance(content, (bytes, bytearray)):
                        body = bytes(content)
                    else:
                        raise ValueError("When format='binary', content must be bytes or bytearray.")
                elif fmt == "json":
                    if isinstance(content, str):
                        # Already serialized JSON string
                        body = content.encode(encoding or "utf-8")
                    else:
                        # Serialize dict/list to JSON
                        body = json.dumps(content, indent=2, ensure_ascii=False).encode(encoding or "utf-8")
                else:
                    raise ValueError("Invalid format. Expected 'text', 'binary', 'base64', or 'json'.")

                config = artifact.config or {}
                s3_config = self._get_s3_config(artifact, parent_artifact)

                if config.get("storage") == "git":
                    # For git storage, write to staging area
                    staging_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/staging/{file_path}",
                    )
                    async with self._create_client_async(s3_config) as s3_client:
                        content_type, _ = mimetypes.guess_type(file_path)
                        await s3_client.put_object(
                            Bucket=s3_config["bucket"],
                            Key=staging_key,
                            Body=body,
                            **({"ContentType": content_type} if content_type else {}),
                        )

                    # Track file in staging manifest
                    staging_dict = artifact.staging or {}
                    staged_files = staging_dict.get("files", [])
                    # Remove existing entry for this path if present
                    staged_files = [f for f in staged_files if f.get("path") != file_path]
                    staged_files.append({"path": file_path})
                    staging_dict["files"] = staged_files
                    artifact.staging = staging_dict
                    flag_modified(artifact, "staging")

                else:
                    # For raw S3 storage, write to staging version
                    staging_dict = artifact.staging or {}
                    has_new_version_intent = staging_dict.get("_intent") == "new_version"
                    if has_new_version_intent:
                        target_version_index = len(artifact.versions or [])
                    else:
                        target_version_index = max(0, len(artifact.versions or []) - 1)

                    async with self._create_client_async(s3_config) as s3_client:
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{target_version_index}/{file_path}",
                        )
                        content_type, _ = mimetypes.guess_type(file_path)
                        await s3_client.put_object(
                            Bucket=s3_config["bucket"],
                            Key=file_key,
                            Body=body,
                            **({"ContentType": content_type} if content_type else {}),
                        )

                    # Track file in staging manifest
                    staged_files = staging_dict.get("files", [])
                    if not any(f.get("path") == file_path for f in staged_files):
                        staged_files.append({"path": file_path})
                        staging_dict["files"] = staged_files
                        artifact.staging = staging_dict
                        flag_modified(artifact, "staging")

                    # Save artifact state to S3
                    await self._save_version_to_s3(target_version_index, artifact, s3_config)

                session.add(artifact)
                await session.commit()

                return {"success": True, "bytes_written": len(body)}
        except Exception:
            raise
        finally:
            await session.close()

    @schema_method
    async def list_files(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact identifier to list files from. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        dir_path: Optional[str] = PydanticField(
            None,
            description="Subdirectory path within the artifact to list. Use forward slashes (e.g., 'data/images'). If None, lists root directory."
        ),
        limit: int = PydanticField(
            1000,
            description="Maximum number of files to return. Range: 1-10000.",
            ge=1,
            le=10000
        ),
        version: Optional[str] = PydanticField(
            None,
            description="Version to list files from (e.g., 'v1.0'). If not specified, lists from current version."
        ),
        stage: bool = PydanticField(
            False,
            description="List files from staged (uncommitted) storage. Cannot be used with version parameter."
        ),
        include_pending: bool = PydanticField(
            False,
            description="Include pending file operations in the listing (uploads in progress, scheduled deletions)."
        ),
        offset: int = PydanticField(
            0,
            description="Number of files to skip before returning results. Used for pagination. Default: 0.",
            ge=0
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """List files stored in an artifact.
        
        This method lists all files within an artifact or a specific directory. Returns
        file metadata including names, sizes, and modification times. Supports listing
        from specific versions or staged storage. Results include both regular files
        and directories.
        
        Returns:
            Dictionary containing:
            - items: List of file/directory objects with metadata
            - truncated: Boolean indicating if results were truncated due to limit
            
        Examples:
            # List all files in artifact
            files = await list_files("my-dataset")
            
            # List files in subdirectory
            files = await list_files("my-project", dir_path="data/processed")
            
            # List files from specific version
            files = await list_files("my-model", version="v1.0")
            
            # List staged files
            files = await list_files("work-in-progress", stage=True)
            
            # List with pagination
            files = await list_files("large-dataset", limit=100)
            if files['truncated']:
                print("More files available")

            # List next page with offset
            next_page = await list_files("large-dataset", limit=100, offset=100)
            
        Raises:
            ValueError: If artifact_id invalid or stage used with version
            PermissionError: If user lacks read permission
            HTTPException: If artifact not found
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)

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

                # Check if this is a git-storage artifact
                config = artifact.config or {}
                if config.get("storage") == "git":
                    # Handle git-storage artifacts
                    items = await self._list_git_files(
                        artifact,
                        parent_artifact,
                        dir_path,
                        limit,
                        offset,
                        version,
                    )
                    return items

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
                        offset=offset,
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

    @schema_method
    async def search(
        self,
        parent_id: Optional[str] = PydanticField(
            None,
            description="Parent collection ID to list children from. Format: 'workspace/collection-alias'. If None, lists all artifacts in workspace (use filters={'parent_id': None} for root artifacts only)."
        ),
        keywords: Optional[List[str]] = PydanticField(
            None,
            description="Keywords to search for in artifact names and descriptions. Performs text search across manifest fields."
        ),
        filters: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Filter criteria for artifacts. Examples: {'type': 'dataset'}, {'manifest.status': 'published'}, {'created_by': 'user123'}."
        ),
        mode: str = PydanticField(
            "AND",
            description="How to combine multiple filters. 'AND' requires all conditions to match, 'OR' requires any condition to match."
        ),
        offset: int = PydanticField(
            0,
            description="Number of items to skip for pagination. Use with limit to implement paging.",
            ge=0
        ),
        limit: int = PydanticField(
            100,
            description="Maximum number of items to return. Range: 1-1000.",
            ge=1,
            le=1000
        ),
        order_by: Optional[str] = PydanticField(
            None,
            description="Field to sort results by. Format: 'field' or '-field' for descending. Examples: 'created_at', '-download_count', 'manifest.name'."
        ),
        pagination: bool = PydanticField(
            False,
            description="If True, returns pagination metadata (total count, has_next, has_previous) along with items."
        ),
        stage: Union[bool, str] = PydanticField(
            False,
            description="Control which artifacts to return: False=committed only (default), True=staged only, 'all'=both staged and committed."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search artifacts within a collection or across the workspace.

        This method searches artifacts within a parent collection, at the workspace root,
        or across all artifacts in the workspace. Supports filtering, keyword search,
        sorting, and pagination. Returns artifact summaries with key metadata fields.

        Returns:
            If pagination=False: List of artifact dictionaries
            If pagination=True: Dictionary with 'items' list and pagination metadata

        Examples:
            # Search all artifacts in workspace (no parent_id restriction)
            items = await search()

            # List children of a specific collection
            items = await search("public/datasets")

            # Search with keywords across workspace
            items = await search(keywords=["neural", "network"])

            # Filter by type across workspace
            items = await search(filters={"type": "dataset"})

            # Search root artifacts only (no parent)
            items = await search(filters={"parent_id": None})

            # Combined: search within collection with filters
            items = await search(
                "my-collection",
                filters={"type": "dataset", "manifest.status": "validated"}
            )

            # Paginated results
            result = await search(
                "large-collection",
                offset=20,
                limit=10,
                pagination=True
            )
            print(f"Showing {len(result['items'])} of {result['total']} items")

            # Sort by download count
            popular = await search(
                "public/apps",
                order_by="-download_count",
                limit=10
            )

        Raises:
            ValueError: If parent_id is invalid or filters malformed
            PermissionError: If user lacks read permission on parent or workspace
            HTTPException: If parent collection not found
        """
        logger.info(f"search: stage parameter value = {stage}")

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

                # Prepare base query for artifacts
                if parent_artifact:
                    # Search within a specific parent collection
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
                    # No parent_id specified - search ALL artifacts in workspace
                    # Use filters={"parent_id": None} to search root artifacts only
                    query = select(ArtifactModel).where(
                        ArtifactModel.workspace == context["ws"],
                    )
                    count_query = select(func.count()).where(
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
                                "created_by": ArtifactModel.created_by,
                            }
                            range_fields = {
                                "created_at": ArtifactModel.created_at,
                                "last_modified": ArtifactModel.last_modified,
                                "download_count": ArtifactModel.download_count,
                                "view_count": ArtifactModel.view_count,
                            }

                            # Special handling for parent_id filter
                            if key == "parent_id":
                                if value is None:
                                    # Search root artifacts only (no parent)
                                    condition = ArtifactModel.parent_id == None
                                else:
                                    # Validate and resolve parent_id alias to internal ID
                                    filter_parent_id = self._validate_artifact_id(value, context)
                                    filter_parent = await self._get_artifact(session, filter_parent_id)
                                    condition = ArtifactModel.parent_id == filter_parent.id
                                conditions.append(condition)
                                continue

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
                        description=f"search count query for '{parent_id or context['ws']}'",
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
                    description=f"search main query for '{parent_id or context['ws']}'",
                )
                artifacts = result.scalars().all()

                # Compile summary results for each artifact
                results = []
                for artifact in artifacts:
                    _artifact_data = self._generate_artifact_data(
                        artifact, parent_artifact
                    )
                    results.append(_artifact_data)

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

    @schema_method
    async def list(
        self,
        parent_id: Optional[str] = PydanticField(
            None,
            description="Parent collection ID to list children from. Format: 'workspace/collection-alias'. If None, lists root artifacts (artifacts without a parent)."
        ),
        keywords: Optional[List[str]] = PydanticField(
            None,
            description="Keywords to search for in artifact names and descriptions."
        ),
        filters: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Filter criteria for artifacts. Examples: {'type': 'dataset'}, {'manifest.status': 'published'}."
        ),
        mode: str = PydanticField(
            "AND",
            description="How to combine multiple filters. 'AND' requires all conditions to match, 'OR' requires any condition to match."
        ),
        offset: int = PydanticField(
            0,
            description="Number of items to skip for pagination.",
            ge=0
        ),
        limit: int = PydanticField(
            100,
            description="Maximum number of items to return. Range: 1-1000.",
            ge=1,
            le=1000
        ),
        order_by: Optional[str] = PydanticField(
            None,
            description="Field to sort results by. Format: 'field' or '-field' for descending."
        ),
        silent: bool = PydanticField(
            False,
            description="If True, does not increment view count for the parent collection."
        ),
        pagination: bool = PydanticField(
            False,
            description="If True, returns pagination metadata along with items."
        ),
        stage: Union[bool, str] = PydanticField(
            False,
            description="Control which artifacts to return: False=committed only (default), True=staged only, 'all'=both."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """List children of a collection or root artifacts in the workspace.

        This is a convenience wrapper around search() that provides backward-compatible
        behavior: when parent_id is None, it lists only root artifacts (artifacts
        without a parent), rather than all artifacts in the workspace.

        For workspace-wide search across all artifacts, use search() directly.
        Note: This method increments the view_count of the parent collection when
        silent=False, while search() never increments view_count.

        Returns:
            If pagination=False: List of artifact dictionaries
            If pagination=True: Dictionary with 'items' list and pagination metadata

        Examples:
            # List root artifacts (no parent)
            items = await list()

            # List children of a specific collection
            items = await list("public/datasets")

            # Filter root artifacts by type
            items = await list(filters={"type": "dataset"})
        """
        # If no parent_id specified, list root artifacts only (parent_id=None)
        if parent_id is None:
            filters = filters or {}
            if "parent_id" not in filters:
                filters["parent_id"] = None

        results = await self.search(
            parent_id=parent_id,
            keywords=keywords,
            filters=filters,
            mode=mode,
            offset=offset,
            limit=limit,
            order_by=order_by,
            pagination=pagination,
            stage=stage,
            context=context,
        )

        # Increment view count for parent collection if not silent
        if not silent and parent_id:
            validated_parent_id = self._validate_artifact_id(parent_id, context)
            session = await self._get_session(read_only=False)
            try:
                async with session.begin():
                    parent_artifact = await self._get_artifact(session, validated_parent_id)
                    await self._increment_stat(
                        session, parent_artifact.id, "view_count"
                    )
                    await session.commit()
            finally:
                await session.close()

        return results

    @schema_method
    async def publish(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact to publish. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        to: Optional[str] = PydanticField(
            None,
            description="Target platform for publishing. Currently supports: 'zenodo' for Zenodo archive. More platforms coming soon."
        ),
        metadata: Optional[Dict[str, Any]] = PydanticField(
            None,
            description="Additional metadata for the publication. Platform-specific fields like DOI metadata for Zenodo."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, Any]:
        """Publish an artifact to an external archive or repository.
        
        This method publishes artifacts to external platforms like Zenodo for permanent
        archival and DOI assignment. The artifact must have complete metadata including
        name and description. Publishing creates a permanent, citable record of the artifact.
        
        Returns:
            Dictionary containing publication details including:
            - status: Publication status
            - doi: Digital Object Identifier (if assigned)
            - url: Public URL of the published artifact
            
        Examples:
            # Publish to Zenodo
            result = await publish(
                "my-dataset",
                to="zenodo",
                metadata={
                    "creators": [{"name": "Smith, John"}],
                    "license": "CC-BY-4.0"
                }
            )
            print(f"Published with DOI: {result['doi']}")
            
            # Simple publication with minimal metadata
            result = await publish("research-data", to="zenodo")
            
        Raises:
            ValueError: If manifest missing required fields (name, description)
            PermissionError: If user lacks publish permission
            HTTPException: If artifact not found or publication fails
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
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
                            await self._load_vector_collection(
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
                        await self._delete_vector_collection(
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

    @schema_method
    async def discard(
        self,
        artifact_id: str = PydanticField(
            ...,
            description="Artifact with staged changes to discard. Format: 'workspace/alias' or just 'alias' (uses current workspace)."
        ),
        context: Optional[dict] = PydanticField(
            None,
            description="Context containing user info and workspace. Usually provided automatically by the system."
        ),
    ) -> Dict[str, str]:
        """Discard all staged changes and revert to last committed state.
        
        This method removes all uncommitted changes including staged files, manifest edits,
        and configuration changes. The artifact reverts to its last committed state.
        This operation cannot be undone - staged changes are permanently lost.
        
        Returns:
            Dictionary with status message confirming changes were discarded
            
        Examples:
            # Discard all staged changes
            await edit("my-doc", manifest={"status": "draft"}, stage=True)
            await put_file("my-doc", "draft.txt", stage=True)
            # Changed mind - discard everything
            await discard("my-doc")
            
            # Discard after reviewing staged changes
            staged_data = await read("experiment", stage=True)
            if not validate(staged_data):
                await discard("experiment")
            
        Raises:
            ValueError: If artifact_id is invalid
            PermissionError: If user lacks write permission
            HTTPException: If artifact not found or no staged changes exist
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
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
        user_info = UserInfo.from_context(context)
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
        user_info = UserInfo.from_context(context)
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

    async def set_parent(
        self,
        artifact_id: str,
        new_parent_id: Optional[str] = None,
        context: dict = None,
    ):
        """Set or clear the parent ID of an artifact. 
        
        Args:
            artifact_id: The ID of the artifact to modify
            new_parent_id: The new parent artifact ID, or None to make it an orphan artifact
            context: Context containing user and workspace information
            
        Returns:
            The updated artifact information
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.from_context(context)
        
        session = await self._get_session()
        try:
            async with session.begin():
                # Get the artifact - only need read permission to move it
                # The actual permission check is on detach/attach
                artifact, old_parent = await self._get_artifact_with_permission(
                    user_info, artifact_id, "read", session
                )
                
                # Check detach permission from old parent if it exists
                if old_parent:
                    # Check if user has detach permission on the old parent
                    # Even artifact creators need detach permission
                    has_detach = await self._check_permissions(old_parent, user_info, "detach")
                    if not has_detach:
                        # Check workspace-level permission as fallback
                        # Workspace owners always have highest permission
                        if not user_info.check_permission(artifact.workspace, UserPermission.admin):
                            raise PermissionError(
                                f"User does not have permission to detach artifact from the collection '{old_parent.alias}'."
                            )
                
                # If setting a new parent, validate it and check attach permission
                new_parent = None
                if new_parent_id:
                    new_parent_id = self._validate_artifact_id(new_parent_id, context)
                    # First get the new parent with read permission
                    new_parent, _ = await self._get_artifact_with_permission(
                        user_info, new_parent_id, "read", session
                    )
                    
                    # Check attach permission on the new parent
                    has_attach = await self._check_permissions(new_parent, user_info, "attach")
                    if not has_attach:
                        # Check workspace-level permission as fallback
                        # Workspace owners always have highest permission
                        if not user_info.check_permission(artifact.workspace, UserPermission.read_write):
                            raise PermissionError(
                                f"User does not have permission to attach artifact to the collection '{new_parent.alias}'."
                            )
                    
                    # Mark the new parent as a collection if it isn't already
                    new_parent_config = new_parent.config or {}
                    if not new_parent_config.get("is_collection"):
                        new_parent_config["is_collection"] = True
                        new_parent.config = new_parent_config
                        flag_modified(new_parent, "config")
                        session.add(new_parent)

                    # Check that new parent is in the same workspace
                    if new_parent.workspace != artifact.workspace:
                        raise ValueError(
                            f"Cannot move artifact to a different workspace. "
                            f"Artifact is in '{artifact.workspace}' but parent is in '{new_parent.workspace}'"
                        )

                    # Prevent circular dependencies
                    if new_parent_id == artifact_id:
                        raise ValueError("An artifact cannot be its own parent")

                    # Check if moving would create a cycle (if artifact has children / is_collection)
                    artifact_config = artifact.config or {}
                    if artifact_config.get("is_collection") or artifact.type == "collection":
                        # Check if new_parent is a descendant of artifact
                        current = new_parent
                        while current and current.parent_id:
                            if current.parent_id == artifact.id:
                                raise ValueError(
                                    "Cannot set parent: this would create a circular dependency"
                                )
                            # Get the parent of current
                            current = await self._get_artifact(session, current.parent_id)
                    
                    artifact.parent_id = new_parent.id
                else:
                    # Making it an orphan - check workspace permissions
                    if not user_info.check_permission(
                        artifact.workspace, UserPermission.read_write
                    ):
                        raise PermissionError(
                            f"User does not have permission to create orphan artifacts in workspace '{artifact.workspace}'"
                        )
                    artifact.parent_id = None
                
                artifact.last_modified = int(time.time())
                session.add(artifact)
                await session.commit()
                
                logger.info(
                    f"Updated parent of artifact {artifact_id} to {new_parent_id}"
                )
                
                # Return updated artifact data
                return self._generate_artifact_data(artifact, new_parent)
                
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
        user_info = UserInfo.from_context(context)
        
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
            "duplicate": self.duplicate,
            "put_file": self.put_file,
            "put_file_start_multipart": self.put_file_start_multipart,
            "put_file_complete_multipart": self.put_file_complete_multipart,
            "remove_file": self.remove_file,
            "get_file": self.get_file,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list": self.list,
            "search": self.search,
            "list_files": self.list_files,
            "add_vectors": self.add_vectors,
            "search_vectors": self.search_vectors,
            "remove_vectors": self.remove_vectors,
            "get_vector": self.get_vector,
            "list_vectors": self.list_vectors,
            "publish": self.publish,
            "get_secret": self.get_secret,
            "set_secret": self.set_secret,
            "set_parent": self.set_parent,
            "set_download_weight": self.set_download_weight,
        }
