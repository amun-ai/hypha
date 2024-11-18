import logging
import time
import sys
import uuid_utils as uuid
import random
import re
import json
from sqlalchemy import (
    event,
    Column,
    JSON,
    UniqueConstraint,
    select,
    text,
    and_,
    or_,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified
from hypha.utils import remove_objects_async, list_objects_async, safe_join
from hypha.utils.zenodo import ZenodoClient
from botocore.exceptions import ClientError
from hypha.s3 import FSFileResponse
from aiobotocore.session import get_session
from sqlalchemy import update
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import relationship  # For parent-child relationships
from fastapi import APIRouter, Depends, HTTPException
from hypha.core import (
    UserInfo,
    UserPermission,
    Artifact,
    CollectionArtifact,
)
from hypha_rpc.utils import ObjectProxy
from jsonschema import validate
from typing import Union, List
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint
from typing import Optional, List

# Logger setup
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(logging.INFO)


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
    staging: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    download_count: float = Field(default=0.0)
    view_count: float = Field(default=0.0)
    file_count: int = Field(default=0)
    created_at: int = Field()
    created_by: Optional[str] = Field(default=None)
    last_modified: int = Field()
    versions: Optional[dict] = Field(
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
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
                session = await self._get_session(read_only=True)
                async with session.begin():
                    # Fetch artifact along with parent
                    artifact, _ = await self._get_artifact_with_permission(
                        user_info, artifact_id, "read", session
                    )

                    version_index = self._get_version_index(artifact, version)

                    if version_index == self._get_version_index(artifact, None):
                        # Prepare artifact representation
                        artifact_data = self._generate_artifact_data(artifact)
                    else:
                        s3_config = self._get_s3_config(artifact)
                        artifact_data = await self._load_version_from_s3(
                            artifact, version_index, s3_config
                        )

                    # Increment view count unless silent
                    if not silent:
                        await self._increment_stat(session, artifact.id, "view_count")
                        await session.commit()

                    return artifact_data

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
                logger.error(f"Unhandled exception: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred."
                )

            finally:
                await session.close()

        # HTTP endpoint for listing child artifacts
        @router.get("/{workspace}/artifacts/{artifact_alias}/children")
        async def list_children(
            workspace: str,
            artifact_alias: str,
            page: int = 0,
            page_size: int = 100,
            order_by: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """List child artifacts of a specified artifact."""
            try:
                artifact_id = self._validate_artifact_id(
                    artifact_alias, {"ws": workspace}
                )
                session = await self._get_session(read_only=True)
                async with session.begin():
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, artifact_id, "list", session
                    )
                    # Query child artifacts
                    query = (
                        select(ArtifactModel)
                        .where(
                            ArtifactModel.workspace == workspace,
                            ArtifactModel.parent_id == parent_artifact.id,
                        )
                        .limit(page_size)
                        .offset(page * page_size)
                    )

                    if order_by:
                        if order_by.endswith("<"):
                            query = query.order_by(
                                getattr(ArtifactModel, order_by[:-1]).asc()
                            )
                        else:
                            query = query.order_by(
                                getattr(ArtifactModel, order_by).desc()
                            )

                    result = await session.execute(query)
                    children = result.scalars().all()

                    # Return children data, excluding secrets
                    return [self._generate_artifact_data(child) for child in children]

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
                logger.error(f"Unhandled exception: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred."
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
                        workspace,
                        f"{self._artifacts_dir}/{artifact.id}/v{version_index}",
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
                                max_length=1000,
                            )
                            if not items:
                                raise HTTPException(
                                    status_code=404,
                                    detail="File or directory not found",
                                )
                            return items

                    s3_client = self._create_client_async(s3_config)
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
                    return FSFileResponse(s3_client, s3_config["bucket"], file_key)

            except KeyError:
                raise HTTPException(status_code=404, detail="Artifact not found")
            except PermissionError:
                raise HTTPException(status_code=403, detail="Permission denied")
            except HTTPException as e:
                raise e  # Re-raise HTTPExceptions to be handled by FastAPI
            except Exception as e:
                logger.error(f"Unhandled exception in get_file: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal server error")
            finally:
                await session.close()

        # Register public services
        artifact_service = self.get_artifact_service()
        self.store.register_public_service(artifact_service)
        self.store.set_artifact_manager(self)
        self.store.register_router(router)

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
        else:
            return ArtifactModel.id == artifact_id

    async def _get_artifact(self, session, artifact_id):
        """
        Retrieve an artifact by its unique ID or by parent ID and alias.
        """
        query = select(ArtifactModel).where(
            self._get_artifact_id_cond(artifact_id),
        )
        result = await session.execute(query)
        artifact = result.scalar_one_or_none()
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        return artifact

    async def _get_artifact_with_parent(self, session, artifact_id):
        """
        Retrieve an artifact and its parent artifact in a single SQL query.
        """
        query = select(ArtifactModel).where(self._get_artifact_id_cond(artifact_id))

        result = await session.execute(query)
        artifact = result.scalar_one_or_none()
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        parent_artifact = None
        if artifact and artifact.parent_id:
            parent_query = select(ArtifactModel).where(
                self._get_artifact_id_cond(artifact.parent_id)
            )
            parent_result = await session.execute(parent_query)
            parent_artifact = parent_result.scalar_one_or_none()
        return artifact, parent_artifact

    def _generate_artifact_data(self, artifact):
        artifact_data = model_to_dict(artifact)
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
                "l": ["list"],
                "l+": ["list", "create", "commit"],
                "r": ["read", "get_file", "list_files", "list"],
                "r+": [
                    "read",
                    "get_file",
                    "put_file",
                    "list_files",
                    "list",
                    "create",
                    "commit",
                ],
                "rw": [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "remove_file",
                ],
                "rw+": [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "remove_file",
                    "create",
                ],
                "*": [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "remove_file",
                    "create",
                    "reset_stats",
                    "publish",
                ],
            }
            return permission_map.get(permission, [])
        else:
            return permission  # Assume it's already a list

    async def _check_permissions(self, artifact, user_info, operation):
        """Check whether a user has permission to perform an operation on an artifact."""
        # Check specific artifact permissions
        permissions = artifact.config.get("permissions", {}) if artifact.config else {}

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
            "get_file": UserPermission.read,
            "list_files": UserPermission.read,
            "create": UserPermission.read_write,
            "edit": UserPermission.read_write,
            "commit": UserPermission.read_write,
            "put_file": UserPermission.read_write,
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

    def _get_s3_config(self, artifact, parent_artifact=None):
        """
        Get S3 configuration using credentials from the artifact's or parent artifact's secrets.
        """
        # merge secrets from parent and artifact
        if parent_artifact and parent_artifact.secrets:
            secrets = parent_artifact.secrets.copy()
        else:
            secrets = {}
        secrets.update(artifact.secrets or {})

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
                "prefix": "",
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
        )

    async def _generate_candidate_aliases(
        self,
        alias_pattern: str,
        id_parts: dict,
        additional_parts: dict = None,
        max_candidates: int = 10,
    ):
        """
        Generate a list of candidate aliases based on the alias pattern and id_parts.
        """
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
        result = await session.execute(query)
        existing_aliases = set(row[0] for row in result.fetchall())
        return existing_aliases

    async def _generate_unique_alias(
        self,
        session,
        workspace,
        alias_pattern,
        id_parts,
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

    async def _handle_existing_artifact(
        self, session, workspace, parent_id, alias, overwrite
    ):
        """Remove existing artifact if overwrite is enabled."""
        try:
            existing_artifact = await self._get_artifact(
                session, f"{workspace}/{alias}"
            )
            if parent_id != existing_artifact.parent_id:
                raise FileExistsError(
                    f"Artifact with alias '{alias}' already exists in a different parent artifact, please remove it first."
                )
            if not overwrite:
                raise FileExistsError(
                    f"Artifact with alias '{alias}' already exists. Use overwrite=True to overwrite."
                )
            logger.info(f"Overwriting artifact with alias: {alias}")
            await session.delete(existing_artifact)
            await session.flush()
            return existing_artifact.id
        except KeyError:
            return None

    async def _save_version_to_s3(self, version_index: int, artifact, s3_config):
        """
        Save the staged version to S3 and update the artifact's manifest.
        """
        assert version_index >= 0, "Version index must be non-negative."
        async with self._create_client_async(s3_config) as s3_client:
            version_key = safe_join(
                s3_config["prefix"],
                artifact.workspace,
                f"{self._artifacts_dir}",
                artifact.id,
                f"v{version_index}.json",
            )
            await s3_client.put_object(
                Bucket=s3_config["bucket"],
                Key=version_key,
                Body=json.dumps(model_to_dict(artifact)),
            )

    async def _load_version_from_s3(
        self, artifact, version_index: int, s3_config, include_secrets=False
    ):
        """
        Load the version from S3.
        """
        assert version_index >= 0, "Version index must be non-negative."
        s3_config = self._get_s3_config(artifact)
        version_key = safe_join(
            s3_config["prefix"],
            artifact.workspace,
            f"{self._artifacts_dir}",
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
                artifact.workspace,
                f"{self._artifacts_dir}",
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
        overwrite=False,
        type="generic",
        config: dict = None,
        secrets: dict = None,
        publish_to=None,
        version: str = None,
        comment: str = None,
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

        if alias:
            if re.match(
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                alias,
            ):
                raise ValueError(
                    "Alias should be a human readable string, it cannot be a UUID."
                )

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
                if not winfo.persistent:
                    raise PermissionError(
                        f"Cannot create artifact in a non-persistent workspace `{workspace}`, you may need to login to a persistent workspace."
                    )

                config = config or {}
                if alias and "{" in alias and "}" in alias:
                    id_parts = {}
                    additional_parts = {
                        "uuid": str(uuid.uuid7()),
                        "timestamp": str(int(time.time())),
                        "user_id": user_info.id,
                    }
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

                    if parent_artifact and parent_artifact.config:
                        id_parts.update(parent_artifact.config.get("id_parts", {}))

                    alias = await self._generate_unique_alias(
                        session,
                        workspace,
                        alias,
                        id_parts,
                        additional_parts=additional_parts,
                    )
                id = str(uuid.uuid7())
                # Modified main section:
                if alias:
                    existing_id = await self._handle_existing_artifact(
                        session, workspace, parent_id, alias, overwrite
                    )
                    if existing_id:
                        id = existing_id
                else:
                    alias = id

                parent_permissions = (
                    parent_artifact.config["permissions"] if parent_artifact else {}
                )
                permissions = config.get("permissions", {}) if config else {}
                permissions[user_info.id] = "*"
                permissions.update(parent_permissions)
                config["permissions"] = permissions

                versions = []
                if version != "stage" and version is not None:
                    if version == "new":
                        version = f"v{len(versions)}"
                    comment = comment or f"Initial version"
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
                    staging=[] if version == "stage" else None,
                    manifest=manifest,
                    created_by=user_info.id,
                    created_at=int(time.time()),
                    last_modified=int(time.time()),
                    config=config,
                    secrets=secrets,
                    versions=versions,
                    type=type,
                )
                version_index = self._get_version_index(new_artifact, version)
                session.add(new_artifact)
                await session.commit()
                await self._save_version_to_s3(
                    version_index,
                    new_artifact,
                    self._get_s3_config(new_artifact, parent_artifact),
                )
                if version != "stage":
                    logger.info(
                        f"Created artifact with ID: {id} (committed), alias: {alias}, parent: {parent_id}"
                    )
                else:
                    logger.info(
                        f"Created artifact with ID: {id} (staged), alias: {alias}, parent: {parent_id}"
                    )
                return self._generate_artifact_data(new_artifact)
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
        context: dict = None,
    ):
        """Edit the artifact's manifest and save it in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        user_info = UserInfo.model_validate(context["user"])
        artifact_id = self._validate_artifact_id(artifact_id, context)
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "edit", session
                )
                artifact.type = type or artifact.type
                if manifest:
                    if artifact.type == "collection":
                        CollectionArtifact.model_validate(manifest)
                    elif artifact.type == "generic":
                        Artifact.model_validate(manifest)
                    if isinstance(manifest, ObjectProxy):
                        manifest = ObjectProxy.toDict(manifest)

                if manifest:
                    artifact.manifest = manifest
                    flag_modified(artifact, "manifest")

                if version is not None:
                    if version != "stage":
                        # Increase the version number
                        versions = artifact.versions or []
                        if version == "new":
                            version = f"v{len(versions)}"
                        versions.append(
                            {
                                "version": version,
                                "comment": comment,
                                "created_at": int(time.time()),
                            }
                        )
                        artifact.versions = versions
                        flag_modified(artifact, "versions")
                        version_index = self._get_version_index(artifact, "latest")
                    else:
                        version_index = self._get_version_index(artifact, "stage")
                else:
                    version_index = self._get_version_index(artifact, None)

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
                if version != "stage":
                    artifact.staging = None
                    flag_modified(artifact, "staging")
                    session.add(artifact)
                    await session.commit()
                    logger.info(
                        f"Edited artifact with ID: {artifact_id} (committed), alias: {artifact.alias}, version: {version}"
                    )
                else:
                    artifact.staging = artifact.staging or []
                    logger.info(
                        f"Edited artifact with ID: {artifact_id} (staged), alias: {artifact.alias}"
                    )
                await self._save_version_to_s3(
                    version_index,
                    artifact,
                    self._get_s3_config(artifact, parent_artifact),
                )
                return self._generate_artifact_data(artifact)
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
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "read", session
                )
                if not silent:
                    await self._increment_stat(session, artifact_id, "view_count")

                version_index = self._get_version_index(artifact, version)
                if version_index == self._get_version_index(artifact, None):
                    artifact_data = self._generate_artifact_data(artifact)
                else:
                    s3_config = self._get_s3_config(artifact)
                    artifact_data = await self._load_version_from_s3(
                        artifact, version_index, s3_config
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
                # the new version index
                version_index = self._get_version_index(artifact, "stage")
                # Load the staged version
                s3_config = self._get_s3_config(artifact)
                artifact_data = await self._load_version_from_s3(
                    artifact, version_index, s3_config
                )
                artifact = ArtifactModel(**artifact_data)

                versions = artifact.versions or []
                artifact.config = artifact.config or {}
                if artifact.staging:
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    async with self._create_client_async(
                        s3_config,
                    ) as s3_client:
                        download_weights = {}
                        for file_info in artifact.staging:
                            file_key = safe_join(
                                s3_config["prefix"],
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact.id}/v{version_index}/{file_info['path']}",
                            )
                            try:
                                await s3_client.head_object(
                                    Bucket=s3_config["bucket"], Key=file_key
                                )
                            except ClientError:
                                raise FileNotFoundError(
                                    f"File '{file_info['path']}' does not exist in the artifact."
                                )
                            if file_info.get("download_weight") is not None:
                                download_weights[file_info["path"]] = file_info[
                                    "download_weight"
                                ]
                        if download_weights:
                            artifact.config["download_weights"] = download_weights
                            flag_modified(artifact, "config")
                        artifact.file_count = len(artifact.staging)

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

                if version is not None:
                    if version == "new":
                        version = f"v{len(versions)}"
                    versions.append(
                        {
                            "version": version,
                            "comment": comment,
                            "created_at": int(time.time()),
                        }
                    )
                artifact.versions = versions
                flag_modified(artifact, "versions")
                artifact.staging = None
                artifact.last_modified = int(time.time())
                flag_modified(artifact, "manifest")
                flag_modified(artifact, "staging")
                await session.merge(artifact)
                await session.commit()
                await self._save_version_to_s3(
                    self._get_version_index(artifact, None),
                    artifact,
                    self._get_s3_config(artifact, parent_artifact),
                )
                logger.info(
                    f"Committed artifact with ID: {artifact_id}, alias: {artifact.alias}, version: {version}"
                )
                return self._generate_artifact_data(artifact)
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
                if delete_files and artifact.file_count > 0:
                    artifact_path = safe_join(
                        s3_config["prefix"],
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact.id}",
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

    async def put_file(
        self, artifact_id, file_path, download_weight: float = 0, context: dict = None
    ):
        """Generate a pre-signed URL to upload a file to an artifact in S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "put_file", session
                )
                assert artifact.staging is not None, "Artifact must be in staging mode."
                # The new version is not committed yet, so we use a new version index
                version_index = self._get_version_index(artifact, "stage")

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact.id}/v{version_index}/{file_path}",
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
                            }
                        )
                        # flag_modified(artifact, "staging")
                    # save the artifact to s3
                    await self._save_version_to_s3(version_index, artifact, s3_config)
                    session.add(artifact)
                    await session.commit()
                    logger.info(
                        f"Put file '{file_path}' to artifact with ID: {artifact_id}"
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
                # The new version is not committed yet, so we use a new version index
                version_index = self._get_version_index(artifact, "stage")

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact.id}/v{version_index}/{file_path}",
                    )
                    await s3_client.delete_object(
                        Bucket=s3_config["bucket"], Key=file_key
                    )

                if artifact.staging:
                    artifact.staging = [
                        f for f in artifact.staging if f["path"] != file_path
                    ]
                    flag_modified(artifact, "staging")

                session.add(artifact)
                await session.commit()
                logger.info(
                    f"Removed file '{file_path}' from artifact with ID: {artifact_id}"
                )
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_file(
        self, artifact_id, path, silent=False, version=None, context: dict = None
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
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact.id}/v{version_index}/{path}",
                    )
                    try:
                        await s3_client.head_object(
                            Bucket=s3_config["bucket"], Key=file_key
                        )
                    except ClientError:
                        raise FileNotFoundError(
                            f"File '{path}' does not exist in the artifact."
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
                    download_weight = download_weights.get(path) or 0
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
        max_length: int = 1000,
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
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact.id}/v{version_index}/{dir_path}",
                            )
                            + "/"
                        )
                    else:
                        full_path = (
                            safe_join(
                                s3_config["prefix"],
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact.id}/v{version_index}",
                            )
                            + "/"
                        )
                    items = await list_objects_async(
                        s3_client,
                        s3_config["bucket"],
                        full_path,
                        max_length=max_length,
                    )
            return items
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def list_children(
        self,
        parent_id=None,
        keywords=None,
        filters=None,
        mode="AND",
        page: int = 0,
        page_size: int = 100,
        order_by=None,
        silent=False,
        context: dict = None,
    ):
        """
        List artifacts within a collection under a specific artifact_id.
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if parent_id:
            parent_id = self._validate_artifact_id(parent_id, context)
        user_info = UserInfo.model_validate(context["user"])

        session = await self._get_session(read_only=True)
        stage = False

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
                    else:
                        # If list_fields is empty or not specified, select all columns
                        query = select(ArtifactModel).where(
                            ArtifactModel.parent_id == parent_artifact.id
                        )
                else:
                    query = select(ArtifactModel).where(ArtifactModel.parent_id == None)
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
                        if key == "stage":
                            stage = bool(value)
                            continue

                        if key == "manifest" and isinstance(value, dict):
                            # Nested search within manifest fields
                            for manifest_key, manifest_value in value.items():
                                if "*" in manifest_value:  # Fuzzy matching in manifest
                                    if backend == "postgresql":
                                        condition = text(
                                            f"manifest->>'{manifest_key}' ILIKE '{manifest_value.replace('*', '%')}'"
                                        )
                                    else:
                                        condition = text(
                                            f"json_extract(manifest, '$.{manifest_key}') LIKE '{manifest_value.replace('*', '%')}'"
                                        )
                                else:  # Exact matching in manifest
                                    if backend == "postgresql":
                                        condition = text(
                                            f"manifest->>'{manifest_key}' = '{manifest_value}'"
                                        )
                                    else:
                                        condition = text(
                                            f"json_extract(manifest, '$.{manifest_key}') = '{manifest_value}'"
                                        )
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
                                        model_field >= value[0]
                                        if value[0] is not None
                                        else True,
                                        model_field <= value[1]
                                        if value[1] is not None
                                        else True,
                                    )
                                else:
                                    condition = model_field >= value
                                conditions.append(condition)
                                continue
                            else:
                                raise ValueError(f"Invalid filter key: {key}")

                if backend == "sqlite":
                    if stage:
                        stage_condition = and_(
                            ArtifactModel.staging.isnot(None), text("staging != 'null'")
                        )
                    else:
                        stage_condition = or_(
                            ArtifactModel.staging.is_(None), text("staging = 'null'")
                        )
                else:
                    if stage:
                        stage_condition = and_(
                            ArtifactModel.staging.isnot(None),
                            text("staging::text != 'null'"),
                        )
                    else:
                        stage_condition = or_(
                            ArtifactModel.staging.is_(None),
                            text("staging::text = 'null'"),
                        )

                query = query.where(stage_condition)

                # Combine conditions based on mode (AND/OR)
                if conditions:
                    query = (
                        query.where(or_(*conditions))
                        if mode == "OR"
                        else query.where(and_(*conditions))
                    )

                # Pagination and ordering
                offset = page * page_size
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
                    .limit(page_size)
                    .offset(offset)
                )

                # Execute the query
                result = await session.execute(query)
                artifacts = result.scalars().all()

                # Compile summary results for each artifact
                results = []
                for artifact in artifacts:
                    _artifact_data = self._generate_artifact_data(artifact)
                    results.append(_artifact_data)

                # Increment view count for parent artifact if not in stage mode
                if not silent and parent_artifact:
                    await self._increment_stat(
                        session, parent_artifact.id, "view_count"
                    )
                    await session.commit()

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
            "publish": self.publish,
        }
