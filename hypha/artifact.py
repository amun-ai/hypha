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

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import (
    RedirectResponse,
    StreamingResponse,
    JSONResponse,
    Response,
)
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

# Logger setup
LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(LOGLEVEL)


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
    staging: Optional[list] = Field(default=None, sa_column=Column(JSON, nullable=True))
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
        @router.get("/{workspace}/artifacts/{artifact_alias}")
        async def get_artifact(
            workspace: str,
            artifact_alias: str,
            silent: bool = False,
            version: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Get artifact metadata, manifest, and config (excluding secrets)."""
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
                                    # Add to total weight unless silent
                                    if not silent:
                                        download_weight = (
                                            download_weights.get(path) or 1
                                        )  # Default to 1 if no weight specified
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
            token: str = None,
            limit: int = 1000,
            use_proxy: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Retrieve a file within an artifact or list files in a directory."""
            try:
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
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
                            )
                            if s3_config["public_endpoint_url"]:
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
            "delete": UserPermission.admin,
            "reset_stats": UserPermission.admin,
            "publish": UserPermission.admin,
        }
        required_perm = operation_map.get(operation)
        if required_perm is None:
            raise ValueError(f"Operation '{operation}' is not supported.")

        # Fetch the primary artifact
        artifact, parent_artifact = await self._get_artifact_with_parent(
            session, artifact_id
        )
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")

        # Check permissions on the artifact itself
        if await self._check_permissions(artifact, user_info, operation):
            return artifact, parent_artifact

        # Finally, check workspace-level permission
        if user_info.check_permission(artifact.workspace, required_perm):
            return artifact, parent_artifact

        raise PermissionError(
            f"User does not have permission to perform the operation '{operation}' on the artifact."
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
            ArtifactModel.workspace == workspace,
            ArtifactModel.alias.in_(aliases),
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
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, parent_id, "create", session
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
                is_staging = stage or (version == "stage")
                versions = []
                if not is_staging:  # Only create initial version if not in staging mode
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
                new_artifact = ArtifactModel(
                    id=id,
                    workspace=workspace,
                    parent_id=parent_id,
                    alias=alias,
                    staging=(
                        [] if is_staging else None
                    ),  # Set staging based on is_staging
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
                    len(versions) if is_staging else 0
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

                if is_staging:
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
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "edit", session
                )

                # Check if artifact is in staging mode or if stage=True is passed
                is_staging = artifact.staging is not None or stage
                if is_staging:
                    logger.info(f"Artifact {artifact_id} is in staging mode")
                    version = (
                        "stage"  # Force version to be "stage" when in staging mode
                    )

                artifact.type = type or artifact.type
                if manifest:
                    artifact.manifest = manifest
                    flag_modified(artifact, "manifest")

                # Determine target version ('stage', None for latest, or specific)
                target_version = None
                if artifact.staging is not None or stage:
                    target_version = "stage"
                elif version is not None:
                    target_version = version  # Could be "new" or specific

                logger.info(f"Target version for edit: {target_version}")

                # Calculate version_index based on target_version
                if target_version == "stage":
                    version_index = self._get_version_index(artifact, "stage")
                    logger.info(f"Using staging version index: {version_index}")
                    # Ensure staging list exists if putting into staging
                    if artifact.staging is None:
                        artifact.staging = []
                        flag_modified(artifact, "staging")
                elif target_version is not None:  # Specific version or "new"
                    versions = artifact.versions or []
                    if target_version == "new":
                        target_version = f"v{len(versions)}"  # Resolve "new"

                    if parent_artifact and not await self._check_permissions(
                        parent_artifact, user_info, "commit"
                    ):
                        raise PermissionError(
                            f"User does not have permission to commit an artifact to collection '{parent_artifact.alias}'."
                        )
                    versions.append(
                        {
                            "version": target_version,
                            "comment": comment,
                            "created_at": int(time.time()),
                        }
                    )
                    artifact.versions = versions
                    flag_modified(artifact, "versions")
                    version_index = self._get_version_index(
                        artifact, "latest"
                    )  # Index of the newly added version
                    logger.info(
                        f"Created new version {target_version} with index {version_index}"
                    )
                else:  # target_version is None (default: edit latest committed)
                    version_index = self._get_version_index(
                        artifact, None
                    )  # Index of the latest committed version
                    logger.info(
                        f"Using latest committed version index: {version_index}"
                    )

                if config is not None:
                    if parent_artifact:
                        parent_permissions = (
                            parent_artifact.config["permissions"]
                            if parent_artifact
                            else {}
                        )
                        permissions = config.get("permissions", {}) if config else {}
                        permissions[user_info.id] = "*"
                        permissions.update(parent_permissions)
                        config["permissions"] = permissions
                    artifact.config = config
                    flag_modified(artifact, "config")
                if secrets is not None:
                    artifact.secrets = secrets
                    flag_modified(artifact, "secrets")
                artifact.last_modified = int(time.time())

                # Commit to DB if not staging
                if target_version != "stage":
                    artifact.staging = None
                    flag_modified(artifact, "staging")
                    session.add(artifact)
                    await session.commit()
                    logger.info(
                        f"Edited artifact with ID: {artifact_id} (committed), alias: {artifact.alias}, version: {target_version or 'latest'}"  # Log the target version
                    )
                else:
                    logger.info(
                        f"Edited artifact with ID: {artifact_id} (staged), alias: {artifact.alias}"
                    )

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
        context: dict = None,
    ):
        """Read the artifact's data including manifest and config."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "read", session
                )
                if not silent:
                    await self._increment_stat(session, artifact_id, "view_count")

                version_index = self._get_version_index(artifact, version)
                if version_index == self._get_version_index(artifact, None):
                    artifact_data = self._generate_artifact_data(
                        artifact, parent_artifact
                    )
                else:
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    artifact_data = await self._load_version_from_s3(
                        artifact, parent_artifact, version_index, s3_config
                    )
                    artifact_data = self._generate_artifact_data(
                        ArtifactModel(**artifact_data), parent_artifact
                    )

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
                # the staging version index
                staging_version_index = self._get_version_index(artifact, "stage")
                logger.info(
                    f"Committing from staging version index: {staging_version_index}"
                )

                # Load the staged version
                s3_config = self._get_s3_config(artifact, parent_artifact)
                artifact_data = await self._load_version_from_s3(
                    artifact, parent_artifact, staging_version_index, s3_config
                )
                artifact = ArtifactModel(**artifact_data)

                versions = artifact.versions or []
                artifact.config = artifact.config or {}

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    download_weights = {}
                    for file_info in artifact.staging or []:
                        # Verify file exists in target version
                        target_version = file_info.get("target_version", len(versions))
                        file_key = safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{target_version}/{file_info['path']}",
                        )
                        try:
                            await s3_client.head_object(
                                Bucket=s3_config["bucket"], Key=file_key
                            )
                        except ClientError:
                            raise FileNotFoundError(
                                f"File '{file_info['path']}' does not exist in the artifact."
                            )

                        if (
                            file_info.get("download_weight") is not None
                            and file_info["download_weight"] > 0
                        ):
                            download_weights[file_info["path"]] = file_info[
                                "download_weight"
                            ]

                    if download_weights:
                        artifact.config["download_weights"] = download_weights
                        flag_modified(artifact, "config")

                    # Count files in the target version
                    artifact.file_count = await self._count_files_in_prefix(
                        s3_client,
                        s3_config["bucket"],
                        safe_join(
                            s3_config["prefix"],
                            f"{artifact.id}/v{len(versions)}",
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

                # Always create a new version when committing from staging mode
                # Ignore any provided version parameter and use sequential versioning
                assigned_version = f"v{len(versions)}"
                versions.append(
                    {
                        "version": assigned_version,  # Use the calculated sequential version
                        "comment": comment,
                        "created_at": int(time.time()),
                    }
                )

                logger.info(
                    f"Creating new version {assigned_version} for artifact {artifact_id}"
                )

                artifact.versions = versions
                flag_modified(artifact, "versions")

                # Save the version manifest to S3
                await self._save_version_to_s3(
                    len(versions) - 1,
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
                version_index = self._get_version_index(artifact, version)
                artifact.versions.pop(version_index)
                flag_modified(artifact, "versions")
                session.add(artifact)
                # Delete the version from S3
                self._delete_version_files_from_s3(
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
        self, artifact_id, file_path, download_weight: float = 0, context: dict = None
    ):
        """Generate a pre-signed URL to upload a file to an artifact in S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        assert download_weight >= 0, "Download weight must be a non-negative number."
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )
                assert artifact.staging is not None, "Artifact must be in staging mode."
                # The new version is not committed yet, so we use a new version index
                version_index = self._get_version_index(artifact, "stage")

                # Calculate target version index - this is where files will actually be stored
                target_version_index = len(artifact.versions or [])
                logger.info(
                    f"Uploading file '{file_path}' to version {target_version_index}"
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
                        ExpiresIn=3600,
                    )
                    if s3_config["public_endpoint_url"]:
                        # replace the endpoint with the proxy base URL
                        presigned_url = presigned_url.replace(
                            s3_config["endpoint_url"], s3_config["public_endpoint_url"]
                        )
                    artifact.staging = artifact.staging or []
                    if not any(f["path"] == file_path for f in artifact.staging):
                        artifact.staging.append(
                            {
                                "path": file_path,
                                "download_weight": download_weight,
                                "target_version": target_version_index,  # Track which version this file belongs to
                            }
                        )
                        flag_modified(artifact, "staging")
                    # save the artifact to s3
                    await self._save_version_to_s3(version_index, artifact, s3_config)
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

    async def remove_file(self, artifact_id, file_path, context: dict):
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
                artifact.staging = [
                    f for f in artifact.staging if f["path"] != file_path
                ]
                flag_modified(artifact, "staging")
                session.add(artifact)
                await session.commit()

                # The new version is not committed yet, so we use a new version index
                version_index = self._get_version_index(artifact, "stage")
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        f"{artifact.id}/v{version_index}/{file_path}",
                    )
                    await s3_client.delete_object(
                        Bucket=s3_config["bucket"], Key=file_key
                    )

                logger.info(
                    f"Removed file '{file_path}' from artifact with ID: {artifact_id}"
                )
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_file(
        self, artifact_id, file_path, silent=False, version=None, context: dict = None
    ):
        """Generate a pre-signed URL to download a file from an artifact in S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
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
                        ExpiresIn=3600,
                    )
                    if s3_config["public_endpoint_url"]:
                        # replace the endpoint with the proxy base URL
                        presigned_url = presigned_url.replace(
                            s3_config["endpoint_url"], s3_config["public_endpoint_url"]
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
        context: dict = None,
    ):
        """List files in the specified artifact's S3 path."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "list_files", session
                )
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
                                ArtifactModel.staging.isnot(None),
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
                                ArtifactModel.staging.isnot(None),
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
                order_field_map = {
                    "id": ArtifactModel.id,
                    "view_count": ArtifactModel.view_count,
                    "download_count": ArtifactModel.download_count,
                    "last_modified": ArtifactModel.last_modified,
                    "created_at": ArtifactModel.created_at,
                }
                order_by = order_by or "id"
                order_field = order_field_map.get(
                    order_by.split("<")[0] if order_by else "id"
                )
                ascending = "<" in order_by if order_by else True
                query = (
                    query.order_by(
                        order_field.asc() if ascending else order_field.desc()
                    )
                    .limit(limit)
                    .offset(offset)
                )

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
                deposition_info = (
                    config.get("zenodo") or await zenodo_client.create_deposition()
                )
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

                async def upload_files(dir_path=""):
                    files = await self.list_files(
                        artifact.id, dir_path=dir_path, context=context
                    )
                    for file_info in files:
                        name = file_info["name"]
                        if file_info["type"] == "directory":
                            await upload_files(safe_join(dir_path, name))
                        else:
                            url = await self.get_file(
                                artifact.id, safe_join(dir_path, name), context=context
                            )
                            await zenodo_client.import_file(deposition_info, name, url)

                await upload_files()
                record = await zenodo_client.publish(deposition_info)

                artifact.config["zenodo"] = record
                flag_modified(artifact, "config")
                session.add(artifact)
                await session.commit()
                return record
        except Exception as e:
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
            "delete": self.delete,
            "put_file": self.put_file,
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
        }
