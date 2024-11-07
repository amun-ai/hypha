import logging
import time
import sys
import uuid_utils as uuid
import random
import re
from sqlalchemy import (
    event,
    Column,
    String,
    Integer,
    Float,
    JSON,
    UniqueConstraint,
    select,
    text,
    ForeignKey,  # For linking to parent artifact ID
    and_,
    or_,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import flag_modified
from hypha.utils import remove_objects_async, list_objects_async, safe_join
from hypha.utils.zenodo import ZenodoClient
from botocore.exceptions import ClientError
from aiobotocore.session import get_session
from sqlalchemy import update
from sqlalchemy.ext.declarative import declarative_base
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

# Logger setup
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(logging.INFO)

Base = declarative_base()


# SQLAlchemy model for storing artifacts
class ArtifactModel(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid7()))
    type = Column(String, nullable=True)
    workspace = Column(String, nullable=False, index=True)
    parent_id = Column(String, ForeignKey("artifacts.id"), nullable=True)
    alias = Column(String, nullable=True)
    manifest = Column(JSON, nullable=True)  # Store committed manifest
    stage_manifest = Column(JSON, nullable=True)  # Store staged manifest
    stage_files = Column(JSON, nullable=True)  # Store staged files during staging
    download_count = Column(Float, nullable=False, default=0.0)  # New counter field
    view_count = Column(Float, nullable=False, default=0.0)  # New counter field
    file_count = Column(Integer, nullable=False, default=0)  # New counter field
    created_at = Column(Integer, nullable=False)
    created_by = Column(String, nullable=True)
    last_modified = Column(Integer, nullable=False)
    config = Column(
        JSON, nullable=True
    )  # Store additional configuration and permissions
    secrets = Column(JSON, nullable=True)  # Store secrets (write-only, flat dictionary)

    # Relationships
    parent = relationship("ArtifactModel", remote_side=[id], backref="children")

    # Table constraints
    __table_args__ = (
        UniqueConstraint(
            "workspace", "alias", name="_workspace_alias_uc"
        ),  # Ensure alias is unique within a workspace
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
        @router.get("/{workspace}/artifacts/{artifact_id}")
        async def get_artifact(
            workspace: str,
            artifact_id: str,
            stage: bool = False,
            silent: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Get artifact metadata, manifest, and config (excluding secrets)."""
            try:
                artifact_id = self._validate_artifact_id(artifact_id, {"ws": workspace})
                session = await self._get_session(read_only=True)
                async with session.begin():
                    # Fetch artifact along with parent
                    artifact, _ = await self._get_artifact_with_permission(
                        user_info, artifact_id, "read", session
                    )

                    # Prepare artifact representation
                    artifact_data = self._generate_artifact_data(artifact, stage)

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
        @router.get("/{workspace}/artifacts/{artifact_id}/children")
        async def list_children(
            workspace: str,
            artifact_id: str,
            page: int = 0,
            page_size: int = 100,
            order_by: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """List child artifacts of a specified artifact."""
            try:
                artifact_id = self._validate_artifact_id(artifact_id, {"ws": workspace})
                session = await self._get_session(read_only=True)
                async with session.begin():
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, artifact_id, "list", session
                    )
                    # Query child artifacts
                    query = (
                        select(ArtifactModel)
                        .filter(
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
        @router.get("/{workspace}/artifacts/{artifact_id}/files/{path:path}")
        async def get_file(
            workspace: str,
            artifact_id: str,
            path: str = "",
            silent: bool = False,
            token: str = None,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Retrieve a file within an artifact or list files in a directory."""
            try:
                artifact_id = self._validate_artifact_id(artifact_id, {"ws": workspace})
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
                    base_key = safe_join(
                        workspace, f"{self._artifacts_dir}/{artifact.id}"
                    )
                    file_key = safe_join(base_key, path)
                    if path.endswith("/") or path == "":
                        s3_config = self._get_s3_config(artifact, parent_artifact)
                        async with self._create_client_async(
                            s3_config,
                        ) as s3_client:
                            items = await list_objects_async(
                                s3_client,
                                self.workspace_bucket,
                                file_key.rstrip("/") + "/",
                                max_length=1000,
                            )
                            if not items:
                                raise HTTPException(
                                    status_code=404,
                                    detail="File or directory not found",
                                )
                            return items

                    from hypha.s3 import FSFileResponse

                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    s3_client = self._create_client_async(s3_config)
                    # Increment download count unless silent
                    if not silent:
                        await self._increment_stat(
                            session, artifact.id, "download_count"
                        )
                    return FSFileResponse(s3_client, self.workspace_bucket, file_key)

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
            await conn.run_sync(Base.metadata.create_all)
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

    async def _get_artifact(self, session, artifact_id):
        """
        Retrieve an artifact by its unique ID or by parent ID and alias.
        """
        if "/" in artifact_id:
            assert len(artifact_id.split("/")) == 2, "Invalid artifact ID format."
            ws, alias = artifact_id.split("/")
            query = select(ArtifactModel).filter(
                ArtifactModel.workspace == ws,
                ArtifactModel.alias == alias,
            )
        else:
            query = select(ArtifactModel).filter(
                ArtifactModel.id == artifact_id,
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
        if "/" in artifact_id:
            assert (
                len(artifact_id.split("/")) == 2
            ), "Invalid artifact ID format, it should be `workspace/alias`."
            ws, alias = artifact_id.split("/")
            query = select(ArtifactModel).filter(
                ArtifactModel.workspace == ws,
                ArtifactModel.alias == alias,
            )
        else:
            query = select(ArtifactModel).filter(
                ArtifactModel.id == artifact_id,
            )

        result = await session.execute(query)
        artifact = result.scalar_one_or_none()
        if not artifact:
            raise KeyError(f"Artifact with ID '{artifact_id}' does not exist.")
        parent_artifact = None
        if artifact and artifact.parent_id:
            parent_query = select(ArtifactModel).filter(
                ArtifactModel.id == artifact.parent_id,
            )
            parent_result = await session.execute(parent_query)
            parent_artifact = parent_result.scalar_one_or_none()
        return artifact, parent_artifact

    def _generate_artifact_data(self, artifact, stage=False):
        artifact_data = {
            "id": artifact.id,
            "type": artifact.type,
            "workspace": artifact.workspace,
            "parent_id": artifact.parent_id,
            "alias": artifact.alias,
            "created_by": artifact.created_by,
            "created_at": int(artifact.created_at or 0),
            "last_modified": int(artifact.last_modified or 0),
            "download_count": int(artifact.download_count or 0),
            "view_count": int(artifact.view_count or 0),
            "manifest": artifact.stage_manifest if stage else artifact.manifest,
            "config": artifact.config or {},
        }
        # Exclude 'secrets' from artifact_data to prevent exposure
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
            .where(ArtifactModel.id == artifact_id)
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

        if not access_key_id or not secret_access_key:
            return {
                "endpoint_url": self.s3_controller.endpoint_url,
                "access_key_id": self.s3_controller.access_key_id,
                "secret_access_key": self.s3_controller.secret_access_key,
                "region_name": self.s3_controller.region_name,
                "bucket": self.workspace_bucket,
                "prefix": "",
            }
        else:
            return {
                "endpoint_url": endpoint_url,
                "access_key_id": access_key_id,
                "secret_access_key": secret_access_key,
                "region_name": region_name,
                "bucket": bucket or self.workspace_bucket,
                "prefix": prefix or "",
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
        query = select(ArtifactModel.alias).filter(
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
                    f"Artifact with alias '{alias}' already exists in a different parent artifact."
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

    async def create(
        self,
        alias: str = None,
        workspace: str = None,
        parent_id: str = None,
        manifest: dict = None,
        overwrite=False,
        stage=False,
        type="generic",
        config: dict = None,
        secrets: dict = None,
        publish_to=None,
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
                    parent_artifact, _ = await self._get_artifact_with_permission(
                        user_info, parent_id, "create", session
                    )
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
                        if config is None:
                            config = {"zenodo": deposition_info}
                        else:
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

                parent_permissions = (
                    parent_artifact.config["permissions"] if parent_artifact else {}
                )
                permissions = config.get("permissions", {}) if config else {}
                permissions[user_info.id] = "*"
                permissions.update(parent_permissions)
                if config:
                    config["permissions"] = permissions
                else:
                    config = {"permissions": permissions}

                assert manifest, "Manifest must be provided."
                new_artifact = ArtifactModel(
                    id=id,
                    workspace=workspace,
                    parent_id=parent_id,
                    alias=alias,
                    manifest=None if stage else manifest,
                    stage_manifest=manifest if stage else None,
                    stage_files=[] if stage else None,
                    created_by=user_info.id,
                    created_at=int(time.time()),
                    last_modified=int(time.time()),
                    config=config,
                    secrets=secrets,
                    type=type,
                )

                session.add(new_artifact)
                await session.commit()
                return self._generate_artifact_data(new_artifact, stage=stage)
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
        stage: bool = False,
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

                if stage:
                    artifact.stage_manifest = manifest
                    flag_modified(artifact, "stage_manifest")
                elif manifest:
                    artifact.manifest = manifest
                    flag_modified(artifact, "manifest")
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
                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()
        return self._generate_artifact_data(artifact, stage=stage)

    async def read(
        self,
        artifact_id,
        stage=False,
        silent=False,
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
                if not silent:
                    await self._increment_stat(session, artifact_id, "view_count")
                artifact, _ = await self._get_artifact_with_permission(
                    user_info, artifact_id, "read", session
                )
                artifact_data = self._generate_artifact_data(artifact, stage=stage)

                if not silent:
                    await session.commit()

                return artifact_data
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def commit(self, artifact_id, context: dict):
        """Commit the artifact by finalizing the staged manifest and files."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        artifact_id = self._validate_artifact_id(artifact_id, context)
        user_info = UserInfo.model_validate(context["user"])
        session = await self._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self._get_artifact_with_permission(
                    user_info, artifact_id, "commit", session
                )

                artifact.config = artifact.config or {}
                if artifact.stage_files:
                    s3_config = self._get_s3_config(artifact, parent_artifact)
                    async with self._create_client_async(
                        s3_config,
                    ) as s3_client:
                        download_weights = {}
                        for file_info in artifact.stage_files:
                            file_key = safe_join(
                                s3_config["prefix"],
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact.id}/{file_info['path']}",
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

                parent_artifact_config = (
                    parent_artifact.config if parent_artifact else {}
                )
                if (
                    parent_artifact_config
                    and "collection_schema" in parent_artifact_config
                ):
                    collection_schema = parent_artifact_config["collection_schema"]
                    try:
                        validate(
                            instance=artifact.stage_manifest, schema=collection_schema
                        )
                    except Exception as e:
                        raise ValueError(f"ValidationError: {str(e)}")
                assert (
                    artifact.stage_manifest
                ), "Artifact must be in staging mode to commit."

                artifact.file_count = (
                    len(artifact.stage_files) if artifact.stage_files else 0
                )
                artifact.manifest = artifact.stage_manifest
                artifact.stage_manifest = None
                artifact.stage_files = None
                artifact.last_modified = int(time.time())
                flag_modified(artifact, "stage_manifest")
                flag_modified(artifact, "manifest")
                flag_modified(artifact, "stage_files")
                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def delete(
        self, artifact_id, delete_files=False, recursive=False, context: dict = None
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
                    artifact.workspace, f"{self._artifacts_dir}/{artifact_id}"
                )
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    await remove_objects_async(
                        s3_client, self.workspace_bucket, artifact_path + "/"
                    )

            # Delete the artifact
            await session.refresh(artifact)
            artifact.parent_id = None
            await session.flush()
            await session.delete(artifact)
            await session.commit()

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
                assert (
                    artifact.stage_manifest
                ), "Artifact must be in staging mode to put files."

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact_id}/{file_path}",
                    )
                    presigned_url = await s3_client.generate_presigned_url(
                        "put_object",
                        Params={"Bucket": self.workspace_bucket, "Key": file_key},
                        ExpiresIn=3600,
                    )

                artifact.stage_files = artifact.stage_files or []
                if not any(f["path"] == file_path for f in artifact.stage_files):
                    artifact.stage_files.append(
                        {
                            "path": file_path,
                            "download_weight": download_weight,
                        }
                    )
                    flag_modified(artifact, "stage_files")

                session.add(artifact)
                await session.commit()
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
                assert (
                    artifact.stage_manifest
                ), "Artifact must be in staging mode to remove files."

                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact_id}/{file_path}",
                    )
                    await s3_client.delete_object(
                        Bucket=s3_config["bucket"], Key=file_key
                    )

                if artifact.stage_files:
                    artifact.stage_files = [
                        f for f in artifact.stage_files if f["path"] != file_path
                    ]
                    flag_modified(artifact, "stage_files")

                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def get_file(self, artifact_id, path, silent=False, context: dict = None):
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
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    file_key = safe_join(
                        s3_config["prefix"],
                        artifact.workspace,
                        f"{self._artifacts_dir}/{artifact_id}/{path}",
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
                        Params={"Bucket": self.workspace_bucket, "Key": file_key},
                        ExpiresIn=3600,
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
                    base_query = select(ArtifactModel).filter(
                        ArtifactModel.parent_id == parent_artifact.id,
                    )
                else:
                    base_query = select(ArtifactModel).filter(
                        ArtifactModel.parent_id == None
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

                # Filter by stage mode
                if stage:
                    query = base_query.where(
                        ArtifactModel.stage_manifest != None
                    )  # Use `!= None` instead of `isnot(None)`
                else:
                    query = base_query.where(
                        ArtifactModel.manifest != None
                    )  # Use `!= None` instead of `isnot(None)`

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
                    order_by.split("<")[0] if order_by else "artifact_id"
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
                    manifest = artifact.stage_manifest if stage else artifact.manifest
                    if not manifest:
                        continue
                    _artifact_data = self._generate_artifact_data(artifact, stage=stage)
                    results.append(_artifact_data)

                # Increment view count for parent artifact if not in stage mode
                if not silent and parent_artifact:
                    await self._increment_stat(
                        session, parent_artifact.id, "view_count"
                    )
                    await session.commit()

                return results
        except Exception as e:
            raise ValueError(
                f"An error occurred while executing the search query: {str(e)}"
            )
        finally:
            await session.close()

    async def list_files(
        self,
        artifact_id: str,
        dir_path: str = None,
        max_length: int = 1000,
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
                s3_config = self._get_s3_config(artifact, parent_artifact)
                async with self._create_client_async(
                    s3_config,
                ) as s3_client:
                    if dir_path:
                        full_path = (
                            safe_join(
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact_id}/{dir_path}",
                            )
                            + "/"
                        )
                    else:
                        full_path = (
                            safe_join(
                                artifact.workspace,
                                f"{self._artifacts_dir}/{artifact_id}",
                            )
                            + "/"
                        )
                    items = await list_objects_async(
                        s3_client,
                        self.workspace_bucket,
                        full_path,
                        max_length=max_length,
                    )
            return items
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
                        artifact_id, dir_path=dir_path, context=context
                    )
                    for file_info in files:
                        name = file_info["name"]
                        if file_info["type"] == "directory":
                            await upload_files(safe_join(dir_path, name))
                        else:
                            url = await self.get_file(
                                artifact_id, safe_join(dir_path, name), context=context
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
