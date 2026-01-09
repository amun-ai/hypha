"""Integration between Git repositories and Hypha artifacts.

This module provides the bridge between Hypha's artifact system and
Git repositories, allowing artifacts to be accessed via Git protocol.

Uses S3 client factory pattern for reliable async operations:
Each S3 operation creates a fresh client context to avoid connection
pool exhaustion and hanging issues with aiobotocore.
"""

import logging
from contextlib import asynccontextmanager
from functools import partial

from hypha.git.repo import S3GitRepo
from hypha.git.http import create_git_router
from hypha.git.lfs import create_lfs_router, GitLFSHandler

logger = logging.getLogger(__name__)


class GitArtifactManager:
    """Manager for Git-enabled artifacts.

    This class handles:
    - Providing access to Git repos for HTTP protocol handlers
    - Managing Git-related artifact configuration

    Git is automatically enabled for artifacts with config.storage == "git".

    Uses S3 client factory pattern for reliable async operations.
    """

    def __init__(self, artifact_controller):
        """Initialize Git artifact manager.

        Args:
            artifact_controller: The ArtifactController instance
        """
        self.artifact_controller = artifact_controller
        self._repo_cache = {}

    def _create_s3_client_factory(self, s3_config):
        """Create an S3 client factory for the given config.

        Args:
            s3_config: S3 configuration dictionary

        Returns:
            Factory function that returns an async context manager for S3 client
        """
        return partial(self.artifact_controller._create_client_async, s3_config)

    @asynccontextmanager
    async def get_repo(
        self,
        workspace: str,
        alias: str,
        user_info,
        write: bool = False,
    ):
        """Get or create a Git repository for an artifact.

        Args:
            workspace: Workspace name
            alias: Artifact alias
            user_info: User info for permission checking
            write: Whether write access is needed

        Yields:
            S3GitRepo instance

        Raises:
            KeyError: If artifact not found or Git storage not enabled
            PermissionError: If user lacks required permissions
        """
        artifact_id = f"{workspace}/{alias}"

        # Get artifact and check permissions
        session = await self.artifact_controller._get_session()
        try:
            async with session.begin():
                artifact, parent_artifact = await self.artifact_controller._get_artifact_with_permission(
                    user_info,
                    artifact_id,
                    "read" if not write else "commit",
                    session,
                )

                # Check if Git storage is enabled
                config = artifact.config or {}
                if config.get("storage") != "git":
                    raise KeyError(f"Git storage not enabled for artifact {artifact_id}")

                # Get S3 config
                s3_config = self.artifact_controller._get_s3_config(
                    artifact, parent_artifact
                )

                # Create S3 client factory and repo
                s3_client_factory = self._create_s3_client_factory(s3_config)
                prefix = f"{s3_config['prefix']}/{artifact.id}/.git"

                repo = S3GitRepo(
                    s3_client_factory,
                    s3_config["bucket"],
                    prefix,
                    s3_config=s3_config,
                )
                await repo.initialize()

                yield repo

        finally:
            await session.close()

    def get_router(self):
        """Get FastAPI router for Git HTTP protocol.

        Returns:
            FastAPI router with Git endpoints
        """

        async def get_repo_callback(workspace, alias, user_info, write=False):
            """Callback to get repo for HTTP handlers."""
            artifact_id = f"{workspace}/{alias}"
            logger.info(f"Git repo request: workspace={workspace}, alias={alias}, write={write}")

            session = await self.artifact_controller._get_session()
            try:
                async with session.begin():
                    try:
                        artifact, parent_artifact = await self.artifact_controller._get_artifact_with_permission(
                            user_info,
                            artifact_id,
                            "read" if not write else "commit",
                            session,
                        )
                    except Exception as e:
                        logger.error(f"Git repo error: artifact not found or permission denied: {artifact_id}, error: {e}")
                        raise KeyError(f"Artifact not found: {artifact_id}") from e

                    config = artifact.config or {}
                    storage_type = config.get("storage")
                    logger.info(f"Git repo: artifact {artifact_id} has storage type: {storage_type}")
                    if storage_type != "git":
                        raise KeyError(f"Git storage not enabled for artifact {artifact_id} (storage={storage_type})")

                    s3_config = self.artifact_controller._get_s3_config(
                        artifact, parent_artifact
                    )

                    # Create S3 client factory for the repo
                    s3_client_factory = self._create_s3_client_factory(s3_config)

                    prefix = f"{s3_config['prefix']}/{artifact.id}/.git"
                    repo = S3GitRepo(s3_client_factory, s3_config["bucket"], prefix, s3_config=s3_config)
                    await repo.initialize()

                    # Store session for cleanup later
                    repo._session = session

                    return repo

            except Exception:
                await session.close()
                raise

        return create_git_router(
            get_repo_callback,
            self.artifact_controller.store.login_optional,
            self.artifact_controller.store.login_required,
            self.artifact_controller.store.parse_user_token,
        )

    def get_lfs_router(self):
        """Get FastAPI router for Git LFS protocol.

        Returns:
            FastAPI router with Git LFS endpoints
        """

        async def get_lfs_handler_callback(workspace, alias, user_info, write=False):
            """Callback to get LFS handler for HTTP handlers."""
            artifact_id = f"{workspace}/{alias}"
            logger.info(f"Git LFS handler request: workspace={workspace}, alias={alias}, write={write}")

            session = await self.artifact_controller._get_session()
            try:
                async with session.begin():
                    try:
                        artifact, parent_artifact = await self.artifact_controller._get_artifact_with_permission(
                            user_info,
                            artifact_id,
                            "read" if not write else "commit",
                            session,
                        )
                    except Exception as e:
                        logger.error(f"Git LFS handler error: artifact not found or permission denied: {artifact_id}, error: {e}")
                        raise KeyError(f"Artifact not found: {artifact_id}") from e

                    config = artifact.config or {}
                    storage_type = config.get("storage")
                    logger.info(f"Git LFS: artifact {artifact_id} has storage type: {storage_type}")
                    if storage_type != "git":
                        raise KeyError(f"Git storage not enabled for artifact {artifact_id} (storage={storage_type})")

                    s3_config = self.artifact_controller._get_s3_config(
                        artifact, parent_artifact
                    )

                    # Create S3 client factory for LFS handler
                    s3_client_factory = self._create_s3_client_factory(s3_config)

                    # Base path for LFS objects (same as git repo prefix but without /git)
                    base_path = f"{s3_config['prefix']}/{artifact.id}"

                    # Get public base URL from store
                    public_base_url = self.artifact_controller.store.public_base_url
                    enable_s3_proxy = self.artifact_controller.s3_controller.enable_s3_proxy

                    handler = GitLFSHandler(
                        s3_client_factory=s3_client_factory,
                        bucket=s3_config["bucket"],
                        base_path=base_path,
                        public_base_url=public_base_url,
                        enable_s3_proxy=enable_s3_proxy,
                    )

                    # Store session for cleanup later
                    handler._session = session

                    return handler

            except Exception:
                await session.close()
                raise

        return create_lfs_router(
            get_lfs_handler_callback,
            self.artifact_controller.store.login_optional,
            self.artifact_controller.store.login_required,
            self.artifact_controller.store.parse_user_token,
        )
