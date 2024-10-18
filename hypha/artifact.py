import logging
import sys
import copy
from sqlalchemy import (
    event,
    Column,
    String,
    Integer,
    Boolean,
    Float,
    JSON,
    UniqueConstraint,
    select,
    text,
    and_,
    or_,
)
from hypha.utils import remove_objects_async, list_objects_async, safe_join
from botocore.exceptions import ClientError
from sqlalchemy import update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm.attributes import flag_modified
from fastapi import APIRouter, Depends, HTTPException
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
    public = Column(
        Boolean, nullable=True
    )  # New field for public artifacts, None means it follows the parent artifact
    __table_args__ = (
        UniqueConstraint("workspace", "prefix", name="_workspace_prefix_uc"),
    )


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

        @router.get("/{workspace}/artifacts/{path:path}")
        async def get_artifact(
            workspace: str,
            path: str,
            stage: bool = False,
            user_info: self.store.login_optional = Depends(self.store.login_optional),
        ):
            """Get artifact from the database."""
            try:
                artifact, manifest = await self._read_manifest(
                    workspace, path, stage=stage
                )
                if not artifact.public:
                    if not user_info.check_permission(workspace, UserPermission.read):
                        raise PermissionError(
                            "User does not have read permission to the non-public artifact."
                        )
                return manifest
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except PermissionError as e:
                raise HTTPException(status_code=403, detail=str(e))

        self.store.set_artifact_manager(self)
        self.store.register_public_service(self.get_artifact_service())
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

    async def _read_manifest(self, workspace, prefix, stage=False):
        session = await self._get_session()
        try:
            async with session.begin():
                query = select(ArtifactModel).filter(
                    ArtifactModel.workspace == workspace,
                    ArtifactModel.prefix == prefix,
                )
                result = await session.execute(query)
                artifact = result.scalar_one_or_none()

                if not artifact:
                    raise KeyError(f"Artifact under prefix '{prefix}' does not exist.")

                manifest = artifact.stage_manifest if stage else artifact.manifest
                if not manifest:
                    raise KeyError(f"No manifest found for artifact '{prefix}'.")

                # If the artifact is a collection, dynamically populate the 'collection' field
                if manifest.get("type") == "collection":
                    sub_prefix = f"{prefix}/"
                    query = select(ArtifactModel).filter(
                        ArtifactModel.workspace == workspace,
                        ArtifactModel.prefix.like(f"{sub_prefix}%"),
                    )
                    result = await session.execute(query)
                    sub_artifacts = result.scalars().all()

                    # Populate the 'collection' field with summary_fields for each sub-artifact
                    summary_fields = manifest.get(
                        "summary_fields", ["id", "name", "description"]
                    )
                    collection = []
                    for artifact in sub_artifacts:
                        sub_manifest = artifact.manifest
                        summary = {"_prefix": artifact.prefix}
                        for field in summary_fields:
                            value = sub_manifest.get(field)
                            if value is not None:
                                summary[field] = value
                        collection.append(summary)

                    manifest["collection"] = collection

                if stage:
                    manifest["stage_files"] = artifact.stage_files
                else:
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
                manifest["_stats"] = {
                    "download_count": artifact.download_count,
                    "view_count": artifact.view_count,
                }
                if manifest.get("type") == "collection":
                    manifest["_stats"]["child_count"] = len(collection)
                return artifact, manifest
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def _get_artifact(self, session, workspace, prefix):
        query = select(ArtifactModel).filter(
            ArtifactModel.workspace == workspace,
            ArtifactModel.prefix == prefix,
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def create(
        self,
        prefix,
        manifest: dict,
        overwrite=False,
        stage=False,
        public=None,
        context: dict = None,
    ):
        """Create a new artifact and store its manifest in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

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
                if public is None:
                    # if public is not set, try to use the parent artifact's public setting
                    parent_prefix = "/".join(prefix.split("/")[:-1])
                    if parent_prefix:
                        parent_artifact = await self._get_artifact(
                            session, ws, parent_prefix
                        )
                        if parent_artifact:
                            public = parent_artifact.public

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
                new_artifact = ArtifactModel(
                    workspace=ws,
                    prefix=prefix,
                    manifest=None if stage else manifest,
                    stage_manifest=manifest if stage else None,
                    stage_files=[] if stage else None,
                    download_weights=None,
                    public=public,
                    type=manifest["type"],
                )
                session.add(new_artifact)
                await session.commit()
            logger.info(f"Created artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

        return manifest

    async def reset_stats(self, prefix, context: dict):
        """Reset the artifact's manifest's download count and view count."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
                if not artifact:
                    raise KeyError(f"Artifact under prefix '{prefix}' does not exist.")
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

    async def read(self, prefix, stage=False, context: dict = None):
        """Read the artifact's manifest from the database and populate collections dynamically."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact, manifest = await self._read_manifest(ws, prefix, stage=stage)

        if not artifact.public and not user_info.check_permission(
            ws, UserPermission.read
        ):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )

        return manifest

    async def edit(self, prefix, manifest=None, context: dict = None):
        """Edit the artifact's manifest and save it in the database."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
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
                artifact = await self._get_artifact(session, ws, prefix)
                if not artifact:
                    raise KeyError(f"Artifact under prefix '{prefix}' does not exist.")
                if manifest is None:
                    manifest = copy.deepcopy(artifact.manifest)
                artifact.stage_manifest = manifest
                flag_modified(artifact, "stage_manifest")  # Mark JSON field as modified
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
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
                if not artifact or not artifact.stage_manifest:
                    raise KeyError(
                        f"No staged manifest to commit for artifact '{prefix}'."
                    )

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
                        and "collection_schema" in parent_artifact.manifest
                    ):
                        collection_schema = parent_artifact.manifest[
                            "collection_schema"
                        ]
                        try:
                            validate(instance=manifest, schema=collection_schema)
                        except Exception as e:
                            raise ValueError(f"ValidationError: {str(e)}")

                # Finalize the manifest
                artifact.manifest = manifest
                artifact.stage_manifest = None
                artifact.stage_files = None
                artifact.download_weights = download_weights
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

    async def delete(self, prefix, context: dict):
        """Delete an artifact from the database and S3."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
                if artifact:
                    await session.delete(artifact)
                    await session.commit()
            logger.info(f"Deleted artifact under prefix: {prefix}")
        except Exception as e:
            raise e
        finally:
            await session.close()

        # Remove files from S3
        await self._delete_s3_files(ws, prefix)

    async def _delete_s3_files(self, ws, prefix):
        """Helper method to delete files associated with an artifact in S3."""
        artifact_path = safe_join(ws, f"{prefix}") + "/"
        async with self.s3_controller.create_client_async() as s3_client:
            await remove_objects_async(s3_client, self.workspace_bucket, artifact_path)

    async def list_files(
        self, prefix, max_length=1000, stage=False, context: dict = None
    ):
        """List files in the specified S3 prefix."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        artifact, _ = await self._read_manifest(ws, prefix, stage=stage)
        if not artifact.public and not user_info.check_permission(
            ws, UserPermission.read
        ):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )
        async with self.s3_controller.create_client_async() as s3_client:
            full_path = safe_join(ws, prefix) + "/"
            items = await list_objects_async(
                s3_client, self.workspace_bucket, full_path, max_length=max_length
            )
            # TODO: If stage should we return only the staged files?
            return items

    async def list_artifacts(self, prefix="", context: dict = None):
        """List all artifacts under a certain prefix."""
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        try:
            artifact, _ = await self._read_manifest(ws, prefix)
            if not artifact.public and not user_info.check_permission(
                ws, UserPermission.read
            ):
                raise PermissionError(
                    "User does not have read permission to the workspace."
                )
        except KeyError:
            # the prefix is not a valid artifact, check if the user has permission to list the artifacts
            if not user_info.check_permission(ws, UserPermission.read):
                raise PermissionError(
                    "User does not have read permission to the workspace."
                )

        session = await self._get_session()
        try:
            async with session.begin():
                query = select(ArtifactModel).filter(
                    ArtifactModel.workspace == ws,
                    ArtifactModel.prefix.like(f"{prefix}/%"),
                )
                result = await session.execute(query)
                sub_artifacts = result.scalars().all()

                collection = []
                for artifact in sub_artifacts:
                    if artifact.prefix == prefix:
                        continue
                    name = artifact.prefix[len(prefix) + 1 :]
                    name = name.split("/")[0]
                    collection.append(name)
                return collection
        except Exception as e:
            raise e
        finally:
            await session.close()

    async def search(
        self, prefix, keywords=None, filters=None, mode="AND", context: dict = None
    ):
        """
        Search artifacts within a collection under a specific prefix.
        Supports:
        - `keywords`: list of fuzzy search terms across all manifest fields.
        - `filters`: dictionary of exact or fuzzy match for specific fields.
        - `mode`: either 'AND' or 'OR' to combine conditions.
        """
        if context is None or "ws" not in context:
            raise ValueError("Context must include 'ws' (workspace).")
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact, _ = await self._read_manifest(ws, prefix)
        if not artifact.public and not user_info.check_permission(
            ws, UserPermission.read
        ):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )

        session = await self._get_session(read_only=True)
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

                # Execute the query
                result = await session.execute(query)
                artifacts = result.scalars().all()

                # Generate the results with summary_fields
                summary_fields = []
                for artifact in artifacts:
                    sub_manifest = artifact.manifest
                    summary_fields.append({"_prefix": artifact.prefix, **sub_manifest})

                return summary_fields
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
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

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
                artifact = await self._get_artifact(session, ws, prefix)
                if not artifact or not artifact.stage_manifest:
                    raise KeyError(f"No staged manifest found for artifact '{prefix}'.")

                artifact.stage_files = artifact.stage_files or []

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
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        artifact, _ = await self._read_manifest(ws, prefix)
        if not artifact.public and not user_info.check_permission(
            ws, UserPermission.read
        ):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )

        async with self.s3_controller.create_client_async() as s3_client:
            file_key = safe_join(ws, f"{prefix}/{path}")
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
        ws = context["ws"]

        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        session = await self._get_session()
        try:
            async with session.begin():
                artifact = await self._get_artifact(session, ws, prefix)
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
            "read": self.read,  # accessible to public if the artifact is public
            "commit": self.commit,
            "delete": self.delete,
            "put_file": self.put_file,
            "remove_file": self.remove_file,
            "get_file": self.get_file,  # accessible to public if the artifact is public
            "list": self.list_artifacts,  # accessible to public if the artifact is public
            "search": self.search,  # accessible to public if the artifact is public
            "list_files": self.list_files,  # accessible to public if the artifact is public
        }
