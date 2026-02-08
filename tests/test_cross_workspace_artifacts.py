"""Test suite for cross-workspace artifact access control.

This test file verifies that artifact access control properly prevents
users from accessing artifacts in workspaces they don't have permission to access.
"""

import pytest
import uuid
from hypha_rpc import connect_to_server

from . import WS_SERVER_URL

pytestmark = pytest.mark.asyncio


class TestCrossWorkspaceArtifactAccess:
    """Test cross-workspace artifact access control."""

    async def test_cannot_read_artifact_from_unauthorized_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that users cannot read artifacts from workspaces they don't have access to."""
        # Create first user with their workspace
        api1 = await connect_to_server({
            "client_id": "artifact-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create a second workspace (victim's workspace)
        ws2_info = await api1.create_workspace({
            "name": f"victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace with private artifacts",
        })
        ws2 = ws2_info["id"]

        # Connect to ws2 and create a private artifact
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "artifact-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        artifact_manager = await api2.get_service("public/artifact-manager")

        # Create a private artifact in ws2
        artifact_info = await artifact_manager.create(
            alias="private-data",
            manifest={
                "name": "Private Dataset",
                "description": "Secret data that should not be accessible",
                "sensitive_info": "This is confidential",
            },
            config={
                "permissions": {}  # No explicit permissions - workspace isolation should protect
            }
        )

        artifact_id = artifact_info["id"]
        assert artifact_id.startswith(f"{ws2}/"), f"Artifact should be in ws2: {artifact_id}"

        # Now try to access this artifact from ws1 (attacker's workspace)
        # Create a new connection as attacker
        ws1_token = await api1.generate_token({"workspace": ws1})
        api_attacker = await connect_to_server({
            "client_id": "attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": ws1_token,
        })

        artifact_manager_attacker = await api_attacker.get_service("public/artifact-manager")

        # Attempt 1: Try to read using full artifact ID (workspace/alias)
        with pytest.raises(Exception) as exc_info:
            await artifact_manager_attacker.read(f"{ws2}/private-data")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        # Attempt 2: Try to read using just the UUID
        with pytest.raises(Exception) as exc_info:
            # Extract UUID from artifact_id (format: workspace/alias or UUID)
            if "/" in artifact_id:
                # This is already workspace/alias format
                await artifact_manager_attacker.read(artifact_id)
            else:
                # This is UUID format
                await artifact_manager_attacker.read(artifact_id)

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg or "not" in error_msg, \
            f"Expected permission or not found error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()

    async def test_cannot_edit_artifact_in_unauthorized_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that users cannot edit artifacts in workspaces they don't control."""
        api1 = await connect_to_server({
            "client_id": "edit-test-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create victim's workspace
        ws2_info = await api1.create_workspace({
            "name": f"edit-victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace",
        })
        ws2 = ws2_info["id"]

        # Create artifact in ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "edit-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        artifact_manager2 = await api2.get_service("public/artifact-manager")
        artifact_info = await artifact_manager2.create(
            alias="editable-data",
            manifest={"name": "Dataset", "version": "1.0"}
        )

        # Try to edit from ws1
        api_attacker = await connect_to_server({
            "client_id": "edit-attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": await api1.generate_token({"workspace": ws1}),
        })

        artifact_manager_attacker = await api_attacker.get_service("public/artifact-manager")

        with pytest.raises(Exception) as exc_info:
            await artifact_manager_attacker.edit(
                artifact_id=f"{ws2}/editable-data",
                manifest={"name": "Hijacked!", "version": "666"}
            )

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()

    async def test_cannot_delete_artifact_in_unauthorized_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that users cannot delete artifacts in other workspaces."""
        api1 = await connect_to_server({
            "client_id": "delete-test-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create victim's workspace
        ws2_info = await api1.create_workspace({
            "name": f"delete-victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace",
        })
        ws2 = ws2_info["id"]

        # Create artifact in ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "delete-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        artifact_manager2 = await api2.get_service("public/artifact-manager")
        artifact_info = await artifact_manager2.create(
            alias="important-data",
            manifest={"name": "Critical Dataset", "important": True}
        )

        # Try to delete from ws1
        api_attacker = await connect_to_server({
            "client_id": "delete-attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": await api1.generate_token({"workspace": ws1}),
        })

        artifact_manager_attacker = await api_attacker.get_service("public/artifact-manager")

        with pytest.raises(Exception) as exc_info:
            await artifact_manager_attacker.delete(f"{ws2}/important-data")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        # Verify artifact still exists by reading it from ws2
        artifact = await artifact_manager2.read("important-data")
        assert artifact["manifest"]["important"] is True

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()

    async def test_cannot_access_artifact_files_cross_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that users cannot access artifact files from other workspaces."""
        api1 = await connect_to_server({
            "client_id": "file-test-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create victim's workspace
        ws2_info = await api1.create_workspace({
            "name": f"file-victim-ws-{uuid.uuid4().hex[:8]}",
            "description": "Victim's workspace",
        })
        ws2 = ws2_info["id"]

        # Create artifact with file in ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "file-creator",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        artifact_manager2 = await api2.get_service("public/artifact-manager")
        artifact_info = await artifact_manager2.create(
            alias="data-with-files",
            manifest={"name": "Dataset with Files"}
        )

        # Upload a file to the artifact
        await artifact_manager2.put_file(
            artifact_id="data-with-files",
            file_path="secret.txt",
            file_content="This is secret data that should not be accessible"
        )

        # Try to access the file from ws1
        api_attacker = await connect_to_server({
            "client_id": "file-attacker",
            "workspace": ws1,
            "server_url": WS_SERVER_URL,
            "token": await api1.generate_token({"workspace": ws1}),
        })

        artifact_manager_attacker = await api_attacker.get_service("public/artifact-manager")

        # Attempt to get the file
        with pytest.raises(Exception) as exc_info:
            await artifact_manager_attacker.get_file(
                artifact_id=f"{ws2}/data-with-files",
                path="secret.txt"
            )

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        # Attempt to list files
        with pytest.raises(Exception) as exc_info:
            await artifact_manager_attacker.list_files(f"{ws2}/data-with-files")

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied error, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
        await api_attacker.disconnect()

    async def test_public_artifacts_accessible_cross_workspace(
        self, fastapi_server, test_user_token
    ):
        """Test that artifacts with public permissions are accessible across workspaces."""
        api1 = await connect_to_server({
            "client_id": "public-test-owner",
            "server_url": WS_SERVER_URL,
            "token": test_user_token,
        })
        ws1 = api1.config.workspace

        # Create a public artifact in ws1
        artifact_manager1 = await api1.get_service("public/artifact-manager")
        artifact_info = await artifact_manager1.create(
            alias="public-dataset",
            manifest={"name": "Public Dataset", "public": True},
            config={
                "permissions": {
                    "*": "read"  # Grant read permission to everyone
                }
            }
        )

        # Create second workspace
        ws2_info = await api1.create_workspace({
            "name": f"public-reader-ws-{uuid.uuid4().hex[:8]}",
            "description": "Reader's workspace",
        })
        ws2 = ws2_info["id"]

        # Access the public artifact from ws2
        ws2_token = await api1.generate_token({"workspace": ws2})
        api2 = await connect_to_server({
            "client_id": "public-reader",
            "workspace": ws2,
            "server_url": WS_SERVER_URL,
            "token": ws2_token,
        })

        artifact_manager2 = await api2.get_service("public/artifact-manager")

        # Should be able to read the public artifact
        artifact = await artifact_manager2.read(f"{ws1}/public-dataset")
        assert artifact["manifest"]["public"] is True

        # But should NOT be able to edit or delete it
        with pytest.raises(Exception) as exc_info:
            await artifact_manager2.edit(
                artifact_id=f"{ws1}/public-dataset",
                manifest={"name": "Hijacked!"}
            )

        error_msg = str(exc_info.value).lower()
        assert "permission" in error_msg or "denied" in error_msg, \
            f"Expected permission denied for edit, got: {exc_info.value}"

        await api1.disconnect()
        await api2.disconnect()
