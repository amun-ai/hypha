import logging
import time
import sys
import json
import httpx
import traceback
import uuid
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
    and_,
    or_,
)
from hypha.utils import remove_objects_async, list_objects_async, safe_join
from hypha.utils.zenodo import ZenodoClient, Deposition
from botocore.exceptions import ClientError
from sqlalchemy import update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm.attributes import flag_modified
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from hypha.core import (
    UserInfo,
    UserPermission,
    Artifact,
    CollectionArtifact,
    ApplicationArtifact,
    WorkspaceInfo,
)
from hypha_rpc.utils import ObjectProxy
from jsonschema import validate
from typing import Union, List

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(logging.INFO)

Base = declarative_base()


# SQLAlchemy model for storing artifacts
class ArtifactModel(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(String, nullable=False)
    workspace = Column(String, nullable=False, index=True)
    prefix = Column(String, nullable=False)
    manifest = Column(JSON, nullable=True)  # Store committed manifest
    stage_manifest = Column(JSON, nullable=True)  # Store staged manifest
    stage_files = Column(JSON, nullable=True)  # Store staged files during staging
    download_weights = Column(
        JSON, nullable=True
    )  # Store the weights for counting downloads; a dictionary of file paths and their weights 0-1
    download_count = Column(Float, nullable=False, default=0.0)  # New counter field
    view_count = Column(Float, nullable=False, default=0.0)  # New counter field
    created_at = Column(Integer, nullable=False)
    created_by = Column(String, nullable=True)
    last_modified = Column(Integer, nullable=False)
    permissions = Column(
        JSON, nullable=True
    )  # New field for storing permissions for the artifact
    config = Column(JSON, nullable=True)
    __table_args__ = (
        UniqueConstraint("workspace", "prefix", name="_workspace_prefix_uc"),
    )


DEFAULT_SUMMARY_FIELDS = ["type", "name"]


async def generate_unique_prefix(
    id_parts,
    pattern,
    session,
    workspace,
    batch_size=10,
    max_attempts=10,
    additional_parts=None,
):
    placeholder_pattern = re.compile(r"\{(\w+)\}")
    placeholders = placeholder_pattern.findall(pattern)

    # Ensure placeholders in pattern exist in id_parts
    if not all(placeholder in id_parts for placeholder in placeholders):
        raise ValueError("Some placeholders in the pattern are missing in id_parts")

    for _ in range(max_attempts):
        # Generate a batch of candidate prefixes
        candidate_prefixes = set()
        for _ in range(batch_size):
            generated_parts = {
                placeholder: random.choice(id_parts[placeholder])
                for placeholder in placeholders
            }
            if additional_parts:
                generated_parts.update(additional_parts)
            candidate_prefix = pattern
            for placeholder, value in generated_parts.items():
                candidate_prefix = candidate_prefix.replace(f"{{{placeholder}}}", value)
            candidate_prefixes.add(candidate_prefix)

        # Check which prefixes already exist in a single query
        existing_prefixes = await _batch_prefix_exists(
            session, workspace, candidate_prefixes
        )

        # Find a unique prefix in the batch
        unique_prefixes = candidate_prefixes - existing_prefixes
        if unique_prefixes:
            return unique_prefixes.pop()  # Return any unique prefix found

    raise RuntimeError("Could not generate a unique prefix within the maximum attempts")


async def _batch_prefix_exists(session, workspace, prefixes):
    """Check which prefixes already exist in the database."""
    query = select(ArtifactModel.prefix).filter(
        ArtifactModel.workspace == workspace, ArtifactModel.prefix.in_(prefixes)
    )
    result = await session.execute(query)
    return set(row[0] for row in result.fetchall())


class ArtifactController:
    """Artifact Controller using SQLAlchemy for database backend and S3 for file storage."""

    def __init__(
        self,
        store,
        s3_controller,
        workspace_bucket="hypha-workspaces",
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

        @router.get("/{workspace}/artifacts/{prefix:path}")
        async def get_artifact(
            workspace: str,
            prefix: str,
            # The following are used for reading the artifact
            stage: bool = False,
            include_metadata: bool = False,
            # The following are used for listing children
            keywords: str = None,
            filters: str = None,
            mode: str = "AND",
            page: int = 0,
            page_size: int = 100,
            order_by: str = None,
            summary_fields: str = None,
            silent: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Get artifact manifest or file."""
            try:
                if prefix.endswith("/__files__"):
                    prefix = prefix.replace("/__files__", "")
                    return await self.list_files(
                        prefix,
                        context={"ws": workspace, "user": user_info.model_dump()},
                    )

                if "/__files__/" in prefix:
                    prefix, path = prefix.split("/__files__/")
                    try:
                        url = await self.get_file(
                            prefix,
                            path,
                            options={"silent": silent},
                            context={"ws": workspace, "user": user_info.model_dump()},
                        )
                        # Redirect to the pre-signed URL
                        return RedirectResponse(url=url)
                    except FileNotFoundError as e:
                        return await self.list_files(
                            prefix,
                            path,
                            context={"ws": workspace, "user": user_info.model_dump()},
                        )

                if prefix.endswith("/__children__"):
                    assert not stage, "Cannot list children of a staged artifact."
                    prefix = prefix.replace("/__children__", "")
                    if keywords:
                        keywords = keywords.split(",")
                    if filters:
                        filters = json.loads(filters)
                    if summary_fields:
                        summary_fields = summary_fields.split(",")
                    return await self.list_children(
                        prefix,
                        page=page,
                        page_size=page_size,
                        filters=filters,
                        mode=mode,
                        order_by=order_by,
                        keywords=keywords,
                        summary_fields=summary_fields,
                        silent=silent,
                        context={"ws": workspace, "user": user_info.model_dump()},
                    )

                artifact = await self._get_artifact_with_permission(
                    workspace, user_info, prefix, "read"
                )
                manifest = await self._read_manifest(
                    artifact,
                    stage=stage,
                    increment_view_count=not silent,
                    include_metadata=include_metadata,
                )

                return manifest
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(traceback.format_exc()))

        self.store.set_artifact_manager(self)
        artifact_service = self.get_artifact_service()
        self.store.register_public_service(artifact_service)
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
            sync_session = session.sync_session  # Access the synchronous session object

            @event.listens_for(sync_session, "before_flush")
            def prevent_flush(session, flush_context, instances):
                """Prevent any flush operations to keep the session read-only."""
                raise RuntimeError("This session is read-only.")

            @event.listens_for(sync_session, "after_transaction_end")
            def end_transaction(session, transaction):
                """Ensure rollback after a transaction in a read-only session."""
                if not transaction._parent:
                    session.rollback()

        return session

    async def _get_artifact(self, session, workspace, prefix):
        query = select(ArtifactModel).filter(
            ArtifactModel.workspace == workspace,
            ArtifactModel.prefix == prefix,
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def _get_session(self, read_only=False):
        """Return an SQLAlchemy async session. If read_only=True, ensure no modifications are allowed."""
        session = self.SessionLocal()

        if read_only:
            sync_session = session.sync_session  # Access the synchronous session object

            @event.listens_for(sync_session, "before_flush")
            def prevent_flush(session, flush_context, instances):
                """Prevent any flush operations to keep the session read-only."""
                raise RuntimeError("This session is read-only.")

            @event.listens_for(sync_session, "after_transaction_end")
            def end_transaction(session, transaction):
                """Ensure rollback after a transaction in a read-only session."""
                if not transaction._parent:
                    session.rollback()

        return session

    def _generate_metadata(self, artifact):
        metadata = {
            "type": artifact.type,
            "workspace": artifact.workspace,
            "prefix": artifact.prefix,
            "created_by": artifact.created_by,
            "created_at": int(artifact.created_at or 0),
            "modified_at": int(artifact.last_modified or 0),
            "download_count": int(artifact.download_count or 0),
            "view_count": int(artifact.view_count or 0),
            "permissions": artifact.permissions or {},
            "download_weights": artifact.download_weights or {},
            "stage": artifact.stage_manifest is not None,
            "stage_files": artifact.stage_files or [],
            "config": artifact.config or {},
        }
        return {f".{k}": v for k, v in metadata.items()}

    async def _read_manifest(
        self, artifact, stage=False, increment_view_count=False, include_metadata=False
    ):
        manifest = artifact.stage_manifest if stage else artifact.manifest
        prefix = artifact.prefix
        if not manifest:
            raise KeyError(f"No manifest found for artifact '{prefix}'.")
        try:
            session = await self._get_session()
            async with session.begin():
                if stage:
                    manifest["_stage_files"] = artifact.stage_files
                elif increment_view_count:
                    # increase view count
                    stmt = (
                        update(ArtifactModel)
                        .where(ArtifactModel.id == artifact.id)
                        # atomically increment the view count
                        .values(view_count=ArtifactModel.view_count + 1)
                        .execution_options(synchronize_session="fetch")
                    )
                    await session.execute(stmt)
                    await session.commit()
                    artifact.view_count += 1
                if include_metadata:
                    manifest.update(self._generate_metadata(artifact))
                return manifest
        except Exception as e:
            raise e
        finally:
            await session.close()

    def _expand_permission(self, permission):
        """Get the alias for the operation."""
        if isinstance(permission, str):
            if permission == "n":
                return []
            elif permission == "l":
                return ["list"]
            elif permission == "l+":
                return ["list", "create", "commit"]
            elif permission == "r":
                return ["read", "get_file", "list_files", "list"]
            elif permission == "r+":
                return [
                    "read",
                    "get_file",
                    "put_file",
                    "list_files",
                    "list",
                    "create",
                    "commit",
                ]
            elif permission == "rw":
                return [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "remove_file",
                ]
            elif permission == "rw+":
                return [
                    "read",
                    "get_file",
                    "list_files",
                    "list",
                    "edit",
                    "commit",
                    "put_file",
                    "remove_file",
                    "create",
                ]
            elif permission == "*":
                return [
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
                ]
            else:
                raise ValueError(f"Permission '{permission}' is not supported.")
        else:
            return permission

    async def _get_artifact_with_permission(
        self,
        workspace: str,
        user_info: UserInfo,
        prefix: str,
        operation: Union[str, List[str]],
    ):
        """
        Check whether a user has the required permission to perform an operation on an artifact.
        :param workspace: The workspace where the artifact is located.
        :param user_info: The user info object, including user permissions.
        :param prefix: The prefix of the artifact being accessed.
        :param operation: The operation being checked ('read', 'write', etc.).
        :return: True if the user has permission, False otherwise.
        """

        # Map the operation to the corresponding UserPermission enum
        if operation in ["list", "read", "get_file", "list_files"]:  # list
            perm = UserPermission.read
        elif operation in ["create", "edit", "commit", "put_file", "remove_file"]:
            perm = UserPermission.read_write
        elif operation in ["delete", "reset_stats", "publish"]:
            perm = UserPermission.admin
        else:
            raise ValueError(f"Operation '{operation}' is not supported.")

        session = await self._get_session(read_only=True)
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, workspace, prefix)
                if not artifact:
                    raise KeyError(f"Artifact with prefix '{prefix}' does not exist.")
                else:
                    # Step 1: Check artifact-specific permissions
                    # Check if there are specific permissions for this artifact
                    if artifact.permissions:
                        # Check if this specific user is in the permissions dictionary
                        if user_info.id in artifact.permissions and (
                            operation
                            in self._expand_permission(
                                artifact.permissions[user_info.id]
                            )
                        ):
                            return artifact

                        # check if the user is logged in and has permission
                        if (
                            not user_info.is_anonymous
                            and "@" in artifact.permissions
                            and (
                                operation
                                in self._expand_permission(artifact.permissions["@"])
                            )
                        ):
                            return artifact

                        # If all users are allowed for this operation, return True
                        if "*" in artifact.permissions and (
                            operation
                            in self._expand_permission(artifact.permissions["*"])
                        ):
                            return artifact

                # Step 2: Check workspace-level permission
                if user_info.check_permission(workspace, perm):
                    return artifact

                # Step 3: Check parent artifact's permissions if no explicit permission defined
                parent_prefix = "/".join(prefix.split("/")[:-1])
                depth = 5
                while parent_prefix:
                    parent_artifact = await self._get_artifact(
                        session, workspace, parent_prefix
                    )
                    if parent_artifact:
                        # Check parent artifact permissions if they exist
                        if parent_artifact.permissions:
                            # Check if this specific user is in the parent artifact permissions dictionary
                            if user_info.id in parent_artifact.permissions and (
                                operation
                                in self._expand_permission(
                                    parent_artifact.permissions[user_info.id]
                                )
                            ):
                                return artifact

                            # check if the user is logged in and has permission
                            if (
                                not user_info.is_anonymous
                                and "@" in parent_artifact.permissions
                                and (
                                    operation
                                    in self._expand_permission(
                                        parent_artifact.permissions["@"]
                                    )
                                )
                            ):
                                return artifact

                            # If all users are allowed for this operation, return True
                            if "*" in parent_artifact.permissions and (
                                operation
                                in self._expand_permission(
                                    parent_artifact.permissions["*"]
                                )
                            ):
                                return artifact
                    else:
                        break
                    # Move up the hierarchy to check the next parent
                    parent_prefix = "/".join(parent_prefix.split("/")[:-1])
                    depth -= 1
                    if depth <= 0:
                        break

        except Exception as e:
            raise e
        finally:
            await session.close()

        raise PermissionError(
            f"User does not have permission to perform the operation '{operation}' on the artifact."
        )

    def _get_zenodo_client(self, parent_artifact, publish_to):
        archives_config = parent_artifact.config.get("archives")
        if publish_to == "zenodo":
            if (
                "zenodo" not in archives_config
                or "access_token" not in archives_config["zenodo"]
            ):
                raise ValueError("Zenodo access token is not configured.")
            zenodo_token = archives_config["zenodo"]["access_token"]
            zenodo_client = ZenodoClient(zenodo_token, "https://zenodo.org")
            assert zenodo_client, "Zenodo access token is not configured."
        elif publish_to == "sandbox_zenodo":
            if (
                "sandbox_zenodo" not in archives_config
                or "access_token" not in archives_config["sandbox_zenodo"]
            ):
                raise ValueError("Sandbox Zenodo access token is not configured.")
            sandbox_zenodo_token = archives_config["sandbox_zenodo"]["access_token"]
            zenodo_client = ZenodoClient(
                sandbox_zenodo_token, "https://sandbox.zenodo.org"
            )
        else:
            raise ValueError(f"Publishing to '{publish_to}' is not supported.")
        return zenodo_client

    async def create(
        self,
        prefix,
        manifest: dict,
        overwrite=False,
        stage=False,
        orphan=False,
        permissions: dict = None,
        config: dict = None,
        publish_to=None,
        context: dict = None,
    ):
        """Create a new artifact and store its manifest in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")

        assert "__files__" not in prefix.split(
            "/"
        ), "Artifact prefix cannot contain '__files__'."

        assert "__children__" not in prefix.split(
            "/"
        ), "Artifact prefix cannot contain '__children__'."

        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])

        # Validate based on the type of manifest
        if manifest["type"] == "collection":
            CollectionArtifact.model_validate(manifest)
        elif manifest["type"] == "application":
            ApplicationArtifact.model_validate(manifest)
        else:
            Artifact.model_validate(manifest)

        # Convert ObjectProxy to dict if necessary
        if isinstance(manifest, ObjectProxy):
            manifest = ObjectProxy.toDict(manifest)
        session = await self._get_session()
        try:
            async with session.begin():
                # if permissions is not set, try to use the parent artifact's permissions setting
                parent_prefix = "/".join(prefix.split("/")[:-1])
                assert (
                    "{" not in parent_prefix and "}" not in parent_prefix
                ), "Curly braces are not allowed in artifact prefix."
                if parent_prefix:
                    try:
                        parent_artifact = await self._get_artifact_with_permission(
                            ws, user_info, parent_prefix, "create"
                        )
                        if not parent_artifact.manifest:
                            raise ValueError(
                                f"Parent artifact under prefix '{parent_prefix}' must be committed before creating a child artifact."
                            )

                        if permissions is None:
                            permissions = parent_artifact.permissions
                        else:
                            # Merge the parent artifact's permissions with the new permissions
                            for key, value in parent_artifact.permissions.items():
                                if key not in permissions:
                                    permissions[key] = value
                    except KeyError:
                        parent_artifact = None
                else:
                    parent_artifact = None

                if not orphan and not parent_artifact:
                    raise ValueError(
                        f"Parent artifact not found (prefix: {parent_prefix}) for non-orphan artifact, please create the parent artifact first or set orphan=True."
                    )

                if parent_artifact and orphan:
                    raise ValueError(
                        f"Parent artifact found (prefix: {parent_prefix}) for orphan artifact, please set orphan=False."
                    )

                if orphan:
                    # Make sure the user has permission to the workspace
                    if not user_info.check_permission(ws, UserPermission.read_write):
                        raise PermissionError(
                            f"User does not have permission to create an orphan artifact in the workspace '{ws}'."
                        )

                if publish_to:
                    assert parent_artifact, "A parent artifact is required"
                    zenodo_client = self._get_zenodo_client(parent_artifact, publish_to)
                    deposition_info = await zenodo_client.create_deposition()
                    if config:
                        assert (
                            "zenodo" not in config
                        ), "Zenodo config already exists in the artifact."
                        config["zenodo"] = deposition_info
                    else:
                        config = {"zenodo": deposition_info}
                else:
                    deposition_info = None

                if "{" in prefix and "}" in prefix:
                    default_parts = {
                        "uuid": str(uuid.uuid4()),
                        "timestamp": str(time.time()),
                    }
                    if deposition_info:
                        default_parts["zenodo_id"] = deposition_info["id"]
                        default_parts["zenodo_conceptrecid"] = deposition_info[
                            "conceptrecid"
                        ]
                    # check if parent artifact is a collection
                    if (
                        parent_artifact
                        and parent_artifact.type == "collection"
                        and parent_artifact.config
                    ):
                        id_parts = parent_artifact.config.get("id_parts")
                        if id_parts:
                            # id parts example: {"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}
                            # say if the prefix is "collections/{animals}-{colors}"
                            # then the prefix will be generate as "collections/dog-red", "collections/dog-blue", ...
                            try:
                                prefix = await generate_unique_prefix(
                                    id_parts,
                                    prefix,
                                    session,
                                    ws,
                                    additional_parts=default_parts,
                                )
                            except Exception as e:
                                raise e
                        else:
                            prefix = prefix.format(**default_parts)
                    else:
                        prefix = prefix.format(**default_parts)
                    existing_artifact = await self._get_artifact(session, ws, prefix)
                    if existing_artifact:
                        raise RuntimeError(
                            f"Failed to generate unique prefix for artifact '{prefix}'."
                        )
                else:
                    existing_artifact = await self._get_artifact(session, ws, prefix)

                if existing_artifact:
                    if not overwrite:
                        raise FileExistsError(
                            f"Artifact under prefix '{prefix}' already exists. Use overwrite=True to overwrite."
                        )
                    else:
                        logger.info(f"Overwriting artifact under prefix: {prefix}")
                        await session.delete(existing_artifact)
                        await session.commit()

            async with session.begin():
                permissions = permissions or {}
                permissions[user_info.id] = "*"

                new_artifact = ArtifactModel(
                    workspace=ws,
                    prefix=prefix,
                    manifest=None if stage else manifest,
                    stage_manifest=manifest if stage else None,
                    stage_files=[] if stage else None,
                    download_weights=None,
                    created_by=user_info.id,
                    created_at=int(time.time()),
                    last_modified=int(time.time()),
                    permissions=permissions,
                    config=config,
                    type=manifest["type"],
                )
                session.add(new_artifact)
                await session.commit()
            logger.info(f"Created artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()
        return self._generate_metadata(new_artifact)

    async def reset_stats(self, prefix, context: dict):
        """Reset the artifact's manifest's download count and view count."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "reset_stats"
        )
        session = await self._get_session()
        try:
            async with session.begin():
                stmt = (
                    update(ArtifactModel)
                    .where(ArtifactModel.id == artifact.id)
                    .values(download_count=0, view_count=0)
                    .execution_options(synchronize_session="fetch")
                )
                await session.execute(stmt)
                await session.commit()
            logger.info(f"Reset artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def read(
        self,
        prefix,
        stage=False,
        silent=False,
        include_metadata=False,
        context: dict = None,
    ):
        """Read the artifact's manifest from the database and populate collections dynamically."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "read"
        )
        manifest = await self._read_manifest(
            artifact,
            stage=stage,
            increment_view_count=not silent,
            include_metadata=include_metadata,
        )
        return manifest

    async def edit(
        self,
        prefix,
        manifest=None,
        permissions: dict = None,
        config: dict = None,
        stage: bool = False,
        context: dict = None,
    ):
        """Edit the artifact's manifest and save it in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "edit"
        )

        if manifest:
            # Validate the manifest
            if manifest["type"] == "collection":
                CollectionArtifact.model_validate(manifest)
            elif manifest["type"] == "application":
                ApplicationArtifact.model_validate(manifest)
            elif manifest["type"] == "workspace":
                WorkspaceInfo.model_validate(manifest)

            # Convert ObjectProxy to dict if necessary
            if isinstance(manifest, ObjectProxy):
                manifest = ObjectProxy.toDict(manifest)

        session = await self._get_session()
        try:
            async with session.begin():
                if stage:
                    artifact.stage_manifest = manifest
                    flag_modified(
                        artifact, "stage_manifest"
                    )  # Mark JSON field as modified
                elif manifest:
                    artifact.manifest = manifest
                    flag_modified(artifact, "manifest")
                if config:
                    artifact.config = config
                    flag_modified(artifact, "config")
                artifact.last_modified = int(time.time())
                if permissions is not None:
                    artifact.permissions = permissions
                    flag_modified(artifact, "permissions")
                session.add(artifact)
                await session.commit()
            logger.info(f"Edited artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def commit(self, prefix, context: dict):
        """Commit the artifact by finalizing the staged manifest and files."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "commit"
        )

        session = await self._get_session()
        try:
            async with session.begin():
                manifest = artifact.stage_manifest

                download_weights = {}
                # Validate files exist in S3 if the staged files list is present
                if artifact.stage_files:
                    async with self.s3_controller.create_client_async() as s3_client:
                        for file_info in artifact.stage_files:
                            file_key = safe_join(ws, f"{prefix}/{file_info['path']}")
                            try:
                                await s3_client.head_object(
                                    Bucket=self.workspace_bucket, Key=file_key
                                )
                            except ClientError:
                                raise FileNotFoundError(
                                    f"File '{file_info['path']}' does not exist in the artifact."
                                )
                            if file_info.get("download_weight") is not None:
                                download_weights[file_info["path"]] = file_info[
                                    "download_weight"
                                ]
                # Validate the schema if the artifact belongs to a collection
                parent_prefix = "/".join(prefix.split("/")[:-1])
                if parent_prefix:
                    parent_artifact = await self._get_artifact(
                        session, ws, parent_prefix
                    )
                    if (
                        parent_artifact
                        and parent_artifact.config
                        and "collection_schema" in parent_artifact.config
                    ):
                        collection_schema = parent_artifact.config["collection_schema"]
                        try:
                            validate(instance=manifest, schema=collection_schema)
                        except Exception as e:
                            raise ValueError(f"ValidationError: {str(e)}")

                # Finalize the manifest
                artifact.manifest = manifest
                artifact.stage_manifest = None
                artifact.stage_files = None
                artifact.download_weights = download_weights
                artifact.last_modified = int(time.time())
                flag_modified(artifact, "manifest")
                flag_modified(artifact, "stage_files")
                flag_modified(artifact, "download_weights")
                session.add(artifact)
                await session.commit()
            logger.info(f"Committed artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def delete(
        self, prefix, delete_files=False, recursive=False, context: dict = None
    ):
        """Delete an artifact from the database and S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        if delete_files and not recursive:
            raise ValueError("Delete files requires recursive=True.")

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "delete"
        )

        if recursive:
            # Remove all child artifacts
            children = await self.list_children(
                prefix, summary_fields=[".prefix"], context=context
            )
            for child in children:
                await self.delete(
                    child[".prefix"], delete_files=delete_files, context=context
                )

        if delete_files:
            # Remove all files in the artifact's S3 prefix
            artifact_path = safe_join(ws, f"{prefix}") + "/"
            async with self.s3_controller.create_client_async() as s3_client:
                await remove_objects_async(
                    s3_client, self.workspace_bucket, artifact_path
                )

        session = await self._get_session()
        try:
            async with session.begin():
                await session.delete(artifact)
                await session.commit()
            logger.info(f"Deleted artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def list_files(
        self,
        prefix: str,
        dir_path: str = None,
        max_length: int = 1000,
        context: dict = None,
    ):
        """List files in the specified S3 prefix."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        await self._get_artifact_with_permission(ws, user_info, prefix, "list_files")
        async with self.s3_controller.create_client_async() as s3_client:
            if dir_path:
                full_path = safe_join(ws, prefix, dir_path) + "/"
            else:
                full_path = safe_join(ws, prefix) + "/"
            items = await list_objects_async(
                s3_client, self.workspace_bucket, full_path, max_length=max_length
            )
            # TODO: If stage should we return only the staged files?
            return items

    async def list_children(
        self,
        prefix,
        keywords=None,
        filters=None,
        mode="AND",
        page: int = 0,
        page_size: int = 100,
        order_by=None,
        summary_fields=None,
        silent=False,
        context: dict = None,
    ):
        """
        List artifacts within a collection under a specific prefix.
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        parent_artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "list"
        )
        session = await self._get_session(read_only=True)
        stage = False
        try:
            async with session.begin():
                # Get the database backend from the SQLAlchemy engine
                backend = (
                    self.engine.dialect.name
                )  # This will return 'postgresql', 'sqlite', etc.

                base_query = select(ArtifactModel).filter(
                    ArtifactModel.workspace == ws,
                    ArtifactModel.prefix.like(f"{prefix}/%"),
                )

                # Handle keyword-based search (fuzzy search across all manifest fields)
                conditions = []
                if keywords:
                    for keyword in keywords:
                        if backend == "postgresql":
                            condition = text(f"manifest::text ILIKE '%{keyword}%'")
                        else:
                            condition = text(
                                f"json_extract(manifest, '$') LIKE '%{keyword}%'"
                            )
                        conditions.append(condition)

                # Handle filter-based search (specific key-value matching)
                if filters:
                    for key, value in filters.items():
                        if key == ".stage":
                            if value:
                                condition = ArtifactModel.stage_manifest != None
                                stage = True
                            else:
                                condition = ArtifactModel.manifest != None
                            conditions.append(condition)
                            continue

                        elif key == ".type":
                            # check ArtifactModel.type
                            condition = ArtifactModel.type == value
                            conditions.append(condition)
                            continue

                        elif key == ".created_by":
                            # check ArtifactModel.created_by
                            condition = ArtifactModel.created_by == value
                            conditions.append(condition)
                            continue

                        elif key == ".created_at":
                            # if the value is a list, it should be a range
                            # otherwise, it should be a single value which is the lower bound
                            if isinstance(value, list):
                                assert (
                                    len(value) == 2
                                ), "created_at range should have 2 values"
                                if value[0] is None:
                                    condition = ArtifactModel.created_at <= value[1]
                                elif value[1] is None:
                                    condition = ArtifactModel.created_at >= value[0]
                                else:
                                    condition = and_(
                                        ArtifactModel.created_at >= value[0],
                                        ArtifactModel.created_at <= value[1],
                                    )
                            else:
                                condition = ArtifactModel.created_at >= value

                            conditions.append(condition)
                            continue

                        elif key == ".last_modified":
                            if isinstance(value, list):
                                assert (
                                    len(value) == 2
                                ), "last_modified range should have 2 values"
                                if value[0] is None:
                                    condition = ArtifactModel.last_modified <= value[1]
                                elif value[1] is None:
                                    condition = ArtifactModel.last_modified >= value[0]
                                else:
                                    condition = and_(
                                        ArtifactModel.last_modified >= value[0],
                                        ArtifactModel.last_modified <= value[1],
                                    )
                            else:
                                condition = ArtifactModel.last_modified >= value

                            conditions.append(condition)
                            continue

                        elif key == ".download_count":
                            if isinstance(value, list):
                                assert (
                                    len(value) == 2
                                ), "download_count range should have 2 values"
                                if value[0] is None:
                                    condition = ArtifactModel.download_count <= value[1]
                                elif value[1] is None:
                                    condition = ArtifactModel.download_count >= value[0]
                                else:
                                    condition = and_(
                                        ArtifactModel.download_count >= value[0],
                                        ArtifactModel.download_count <= value[1],
                                    )
                            else:
                                condition = ArtifactModel.download_count >= value

                            conditions.append(condition)
                            continue

                        elif key == ".view_count":
                            if isinstance(value, list):
                                assert (
                                    len(value) == 2
                                ), "view_count range should have 2 values"
                                if value[0] is None:
                                    condition = ArtifactModel.view_count <= value[1]
                                elif value[1] is None:
                                    condition = ArtifactModel.view_count >= value[0]
                                else:
                                    condition = and_(
                                        ArtifactModel.view_count >= value[0],
                                        ArtifactModel.view_count <= value[1],
                                    )
                            else:
                                condition = ArtifactModel.view_count >= value

                            conditions.append(condition)
                            continue
                        elif key.startswith(".permissions/"):
                            # format: permissions/user.id=permission
                            user_id = key.split("/")[1]
                            if backend == "postgresql":
                                condition = text(
                                    f"permissions->>'{user_id}' = '{value}'"
                                )
                            else:
                                condition = text(
                                    f"json_extract(permissions, '$.{user_id}') = '{value}'"
                                )
                            conditions.append(condition)
                            continue
                        elif key.startswith("."):
                            raise ValueError(f"Invalid filter key: {key}")

                        if "*" in value:  # Fuzzy matching
                            if backend == "postgresql":
                                condition = text(
                                    f"manifest->>'{key}' ILIKE '{value.replace('*', '%')}'"
                                )
                            else:
                                condition = text(
                                    f"json_extract(manifest, '$.{key}') LIKE '{value.replace('*', '%')}'"
                                )
                        else:  # Exact matching
                            if backend == "postgresql":
                                condition = text(f"manifest->>'{key}' = '{value}'")
                            else:
                                condition = text(
                                    f"json_extract(manifest, '$.{key}') = '{value}'"
                                )
                        conditions.append(condition)

                # Combine conditions using AND/OR mode
                if conditions:
                    if mode == "OR":
                        query = base_query.where(or_(*conditions))
                    else:  # Default is AND
                        query = base_query.where(and_(*conditions))
                else:
                    query = base_query

                offset = page * page_size
                if order_by is None:
                    query = query.order_by(ArtifactModel.prefix.asc())
                elif order_by.startswith("view_count"):
                    if order_by.endswith("<"):
                        query = query.order_by(ArtifactModel.view_count.asc())
                    else:
                        query = query.order_by(ArtifactModel.view_count.desc())
                elif order_by.startswith("download_count"):
                    if order_by.endswith("<"):
                        query = query.order_by(ArtifactModel.download_count.asc())
                    else:
                        query = query.order_by(ArtifactModel.download_count.desc())
                elif order_by.startswith("last_modified"):
                    if order_by.endswith("<"):
                        query = query.order_by(ArtifactModel.last_modified.asc())
                    else:
                        query = query.order_by(ArtifactModel.last_modified.desc())
                elif order_by.startswith("created_at"):
                    if order_by.endswith("<"):
                        query = query.order_by(ArtifactModel.created_at.asc())
                    else:
                        query = query.order_by(ArtifactModel.created_at.desc())
                elif order_by.startswith("prefix"):
                    if order_by.endswith("<"):
                        query = query.order_by(ArtifactModel.prefix.asc())
                    else:
                        query = query.order_by(ArtifactModel.prefix.desc())
                else:
                    raise ValueError(f"Invalid order_by field: {order_by}")

                query = query.limit(page_size).offset(offset)
                # Execute the query
                result = await session.execute(query)
                artifacts = result.scalars().all()

                if summary_fields is None:
                    if parent_artifact.manifest.get("type") == "collection":
                        # Generate the results with summary_fields
                        summary_fields = (
                            parent_artifact.config
                            and parent_artifact.config.get("summary_fields")
                            or DEFAULT_SUMMARY_FIELDS
                        )
                    else:
                        summary_fields = DEFAULT_SUMMARY_FIELDS
                results = []
                for artifact in artifacts:
                    manifest = artifact.stage_manifest if stage else artifact.manifest
                    if not manifest:
                        continue
                    summary = {}
                    _metadata = self._generate_metadata(artifact)
                    for field in summary_fields:
                        if field == ".*":
                            summary.update(_metadata)
                            continue
                        if field == "*":
                            summary.update(manifest)
                            continue
                        if field.startswith("."):
                            summary[field] = _metadata.get(field)
                        else:
                            summary[field] = manifest.get(field)

                    results.append(summary)

                if not stage:
                    # Increment the view count for the parent artifact
                    await self._read_manifest(
                        parent_artifact,
                        stage=False,
                        increment_view_count=not silent,
                        include_metadata=False,
                    )

                return results
        except Exception as e:
            raise ValueError(
                f"An error occurred while executing the search query: {str(e)}"
            )
        finally:
            await session.close()

    async def put_file(
        self, prefix, file_path, options: dict = None, context: dict = None
    ):
        """Generate a pre-signed URL to upload a file to an artifact in S3 and update the manifest."""
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "put_file"
        )
        assert artifact.stage_manifest, "Artifact must be in staging mode to put files."

        options = options or {}

        async with self.s3_controller.create_client_async() as s3_client:
            file_key = safe_join(ws, f"{prefix}/{file_path}")
            presigned_url = await s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.workspace_bucket, "Key": file_key},
                ExpiresIn=3600,
            )
            if self.s3_controller.enable_s3_proxy:
                presigned_url = f"{self.store.public_base_url}/s3/{self.workspace_bucket}/{file_key}?{presigned_url.split('?')[1]}"

        session = await self._get_session()
        try:
            async with session.begin():
                artifact.stage_files = artifact.stage_files or []
                artifact.last_modified = int(time.time())

                if not any(f["path"] == file_path for f in artifact.stage_files):
                    artifact.stage_files.append(
                        {
                            "path": file_path,
                            "download_weight": options.get("download_weight"),
                        }
                    )
                    flag_modified(artifact, "stage_files")

                session.add(artifact)
                await session.commit()
            logger.info(f"Generated pre-signed URL for file upload: {file_path}")
        except Exception as e:
            raise e
        finally:
            await session.close()

        return presigned_url

    async def get_file(self, prefix, path, options: dict = None, context: dict = None):
        """Generate a pre-signed URL to download a file from an artifact in S3."""
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "get_file"
        )
        async with self.s3_controller.create_client_async() as s3_client:
            file_key = safe_join(ws, f"{prefix}/{path}")
            # check if the file exists
            try:
                await s3_client.head_object(Bucket=self.workspace_bucket, Key=file_key)
            except ClientError:
                raise FileNotFoundError(
                    f"File '{path}' does not exist in the artifact."
                )
            presigned_url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.workspace_bucket, "Key": file_key},
                ExpiresIn=3600,
            )
            if self.s3_controller.enable_s3_proxy:
                presigned_url = f"{self.store.public_base_url}/s3/{self.workspace_bucket}/{file_key}?{presigned_url.split('?')[1]}"

        logger.info(f"Generated pre-signed URL for file download: {path}")

        if options is None or not options.get("silent"):
            session = await self._get_session()
            try:
                async with session.begin():
                    artifact = await self._get_artifact(session, ws, prefix)
                    if artifact.download_weights and path in artifact.download_weights:
                        # if it has download_weights, increment the download count by the weight
                        stmt = (
                            update(ArtifactModel)
                            .where(ArtifactModel.id == artifact.id)
                            # atomically increment the download count by the weight
                            .values(
                                download_count=ArtifactModel.download_count
                                + artifact.download_weights[path]
                            )
                            .execution_options(synchronize_session="fetch")
                        )
                        await session.execute(stmt)
                        await session.commit()
            except Exception as e:
                raise e
            finally:
                await session.close()
        return presigned_url

    async def remove_file(self, prefix, file_path, context: dict):
        """Remove a file from the artifact and update the staged manifest."""
        if prefix.startswith("/"):
            ws = prefix.split("/")[1]
            prefix = "/".join(prefix.split("/")[2:])
        else:
            ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "remove_file"
        )
        assert (
            artifact.stage_manifest
        ), "Artifact must be in staging mode to remove files."

        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
                artifact.last_modified = int(time.time())
                if not artifact or not artifact.stage_manifest:
                    raise KeyError(
                        f"Artifact under prefix '{prefix}' is not in staging mode."
                    )
                if artifact.stage_files:
                    # remove the file from the staged files list
                    artifact.stage_files = [
                        f for f in artifact.stage_files if f["path"] != file_path
                    ]
                    flag_modified(artifact, "stage_files")
                if artifact.download_weights:
                    # remove the file from download_weights if it's there
                    artifact.download_weights = {
                        k: v
                        for k, v in artifact.download_weights.items()
                        if k != file_path
                    }
                    flag_modified(artifact, "download_weights")
                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()

        async with self.s3_controller.create_client_async() as s3_client:
            file_key = safe_join(ws, f"{prefix}/{file_path}")
            await s3_client.delete_object(Bucket=self.workspace_bucket, Key=file_key)

        logger.info(f"Removed file from artifact: {file_key}")

    async def publish(
        self, prefix: str, to: str = None, metadata=None, context: dict = None
    ):
        """Publish the artifact to public archive."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        # read the artifact
        artifact = await self._get_artifact_with_permission(
            ws, user_info, prefix, "publish"
        )
        manifest = await self._read_manifest(
            artifact,
            stage=False,
            increment_view_count=False,
            include_metadata=True,
        )

        parent_prefix = "/".join(prefix.split("/")[:-1])
        assert parent_prefix, "Parent artifact not found."
        session = await self._get_session()
        try:
            async with session.begin():
                parent_artifact = await self._get_artifact(session, ws, parent_prefix)
                zenodo_client = self._get_zenodo_client(parent_artifact, to)
                if not parent_artifact:
                    raise ValueError(
                        f"Parent artifact under prefix '{parent_prefix}' not found."
                    )
        except Exception as e:
            raise e
        finally:
            await session.close()

        config = artifact.config or {}
        if config and "zenodo" in config:
            # Infer the target Zenodo instance from the existing config
            deposition_info = config["zenodo"]
            if deposition_info["links"]["self"].startswith(
                "https://sandbox.zenodo.org"
            ):
                assert (
                    to is None or to == "sandbox_zenodo"
                ), "Cannot publish to Zenodo from Sandbox Zenodo."
                to = "sandbox_zenodo"
            else:
                assert (
                    to is None or to == "zenodo"
                ), "Cannot publish to Sandbox Zenodo from Zenodo."
                to = "zenodo"

        if "zenodo" in config:
            deposition_id = config["zenodo"]["id"]
            deposition_info = await zenodo_client.load_deposition(deposition_id)
        else:
            deposition_info = await zenodo_client.create_deposition()
        deposition = Deposition(zenodo_client, deposition_info)

        assert manifest, "Manifest is empty or not committed."
        assert "name" in manifest, "Manifest must have a name."
        assert "type" in manifest, "Manifest must have a type."
        assert "description" in manifest, "Manifest must have a description."

        # Map authors to creators format required by Zenodo
        creators = []
        if "authors" in manifest:
            creators = [
                {
                    "name": author.get("name"),
                    "affiliation": author.get("affiliation", ""),
                }
                for author in manifest["authors"]
            ]
        else:
            creators = [{"name": user_info.id}]
        # Update the metadata
        _metadata = {
            "title": manifest.get("name", "Untitled"),
            "upload_type": "dataset" if manifest.get("type") == "dataset" else "other",
            "description": manifest.get("description", "No description provided."),
            "creators": creators,
            "access_right": "open",
            "license": manifest.get("license", "cc-by"),
            "keywords": manifest.get("tags", []),
            "notes": f"Published automatically from Hypha ({self.store.public_base_url}).",
        }
        if metadata:
            _metadata.update(metadata)
        await deposition.update_metadata(_metadata)

        # Add a file to the deposition
        # list the files in the artifact then upload one by one
        async def upload_files(dir_path=""):
            files = await self.list_files(
                prefix, dir_path=dir_path, max_length=100, context=context
            )
            for file_info in files:
                name = file_info["name"]
                if file_info["type"] == "directory":
                    await upload_files(safe_join(dir_path, name))
                else:
                    url = await self.get_file(
                        prefix, safe_join(dir_path, name), context=context
                    )
                    async with httpx.AsyncClient() as client:
                        async with client.stream("GET", url) as response:

                            async def s3_response_chunk_reader(
                                response, chunk_size: int = 2048
                            ):
                                async for chunk in response.aiter_bytes(chunk_size):
                                    yield chunk

                            put_response = await zenodo_client.client.put(
                                f"{deposition.bucket_url}/{name}",
                                params=zenodo_client.params,
                                headers={"Content-Type": "application/octet-stream"},
                                data=s3_response_chunk_reader(response),
                            )
                            put_response.raise_for_status()
                            logger.info(f"Uploaded file: {name} to Zenodo.")

        await upload_files()
        # Publish the deposition
        record = await deposition.publish()
        logger.info(
            f"Published artifact under prefix: {prefix} to Zenodo, DOI: {record.get('doi')}"
        )

        # Save the record to the artifact's config for future reference
        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
                artifact.config = artifact.config or {}
                artifact.config["zenodo"] = record
                flag_modified(artifact, "config")
                session.add(artifact)
                await session.commit()
        except Exception as e:
            raise e
        finally:
            await session.close()
        return record

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
