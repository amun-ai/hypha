"""Provide an s3 interface."""
import logging
import sys
import yaml
from botocore.exceptions import ClientError

from hypha.core import UserInfo, UserPermission, Artifact
from hypha.utils import (
    list_objects_async,
    remove_objects_async,
    safe_join,
)

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("artifact")
logger.setLevel(logging.INFO)

MANIFEST_FILENAME = "manifest.yaml"
EDIT_MANIFEST_FILENAME = "_manifest.yaml"


class ArtifactController:
    """Represent an artifact controller."""

    def __init__(self, store, s3_controller=None, workspace_bucket="hypha-workspaces"):
        """Set up controller."""
        self.s3_controller = s3_controller
        self.workspace_bucket = workspace_bucket
        store.register_public_service(self.get_artifact_service())

    async def create(
        self, prefix, manifest: dict, overwrite=False, context: dict = None
    ):
        """Create a new artifact with a manifest file."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        
        Artifact.model_validate(manifest)

        manifest_key = f"{ws}/{prefix}/{MANIFEST_FILENAME}"
        edit_manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"

        async with self.s3_controller.create_client_async() as s3_client:
            # Check if manifest or _manifest.yaml already exists
            try:
                await s3_client.head_object(
                    Bucket=self.workspace_bucket, Key=manifest_key
                )
                manifest_exists = True
            except ClientError:
                manifest_exists = False

            try:
                await s3_client.head_object(
                    Bucket=self.workspace_bucket, Key=edit_manifest_key
                )
                edit_manifest_exists = True
            except ClientError:
                edit_manifest_exists = False

            if (manifest_exists or edit_manifest_exists) and not overwrite:
                raise FileExistsError(
                    f"Artifact under prefix '{prefix}' already exists. Use overwrite=True to overwrite."
                )

            # Check if the artifact is a collection and initialize it
            if manifest.get("type") == "collection":
                collection = await self._build_collection_index(manifest, ws, prefix, s3_client)
                manifest["collection"] = collection

            # If overwrite=True, remove any existing manifest or _manifest.yaml
            if overwrite:
                if manifest_exists:
                    await s3_client.delete_object(
                        Bucket=self.workspace_bucket, Key=manifest_key
                    )
                if edit_manifest_exists:
                    await s3_client.delete_object(
                        Bucket=self.workspace_bucket, Key=edit_manifest_key
                    )

            if manifest["type"] == "collection":
                manifest["collection"] = await self._build_collection_index(manifest, ws, prefix, s3_client)
            # Upload the _manifest.yaml to indicate it's in edit mode
            response = await s3_client.put_object(
                Body=yaml.dump(manifest),
                Bucket=self.workspace_bucket,
                Key=edit_manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to create artifact under prefix: {prefix}"

            

        return manifest

    async def index(self, prefix, context: dict = None):
        """Update the index of the current collection."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        collection_key = f"{ws}/{prefix}/{MANIFEST_FILENAME}"
        async with self.s3_controller.create_client_async() as s3_client:
            # check if the collection manifest file exits
            try:
                await s3_client.head_object(Bucket=self.workspace_bucket, Key=collection_key)
                try:
                    collection_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"
                    await s3_client.head_object(Bucket=self.workspace_bucket, Key=collection_key)
                except ClientError:
                    raise KeyError(f"Collection manifest file does not exist for prefix '{prefix}'")
            except ClientError:
                raise KeyError(f"Collection manifest file does not exist for prefix '{prefix}'")

            # Check if this is a valid collection
            manifest_obj = await s3_client.get_object(
                Bucket=self.workspace_bucket, Key=collection_key
            )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())
            collection = await self._build_collection_index(manifest, ws, prefix, s3_client)
            # Update the collection field
            manifest["collection"] = collection

            # Save the updated manifest
            response = await s3_client.put_object(
                Body=yaml.dump(manifest),
                Bucket=self.workspace_bucket,
                Key=collection_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to update collection index for '{prefix}'"


    async def _build_collection_index(self, manifest, ws, prefix, s3_client):
        if manifest.get("type") != "collection":
            raise ValueError(
                f"The prefix '{prefix}' does not point to a valid collection."
            )

        # List all sub-artifacts under the collection prefix
        sub_prefix = f"{ws}/{prefix}/"
        sub_artifacts = await list_objects_async(
            s3_client, self.workspace_bucket, prefix=sub_prefix
        )
        collection = []

        # Extract summary fields or default to 'name' and 'description'
        summary_fields = manifest.get("summary_fields", ["name", "description"])

        for obj in sub_artifacts:
            if obj["type"] == "directory":
                sub_prefix = obj["name"]
                try:
                    sub_manifest_key = f"{ws}/{sub_prefix}/{MANIFEST_FILENAME}"
                    sub_manifest_obj = await s3_client.get_object(
                        Bucket=self.workspace_bucket, Key=sub_manifest_key
                    )
                    sub_manifest = yaml.safe_load(
                        (await sub_manifest_obj["Body"].read()).decode()
                    )

                    # Extract summary information
                    summary = {}
                    for field in summary_fields:
                        keys = field.split(".")
                        value = sub_manifest
                        for key in keys:
                            if key in value:
                                value = value[key]
                            else:
                                value = None
                                break
                        if value is not None:
                            summary[field] = value
                    summary["_id"] = sub_prefix
                    collection.append(summary)
                except ClientError:
                    pass

        return collection

    async def read(self, prefix, context: dict = None):
        """Read the artifact's manifest. Fallback to _manifest.yaml if manifest.yaml doesn't exist."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )
        manifest_key = f"{ws}/{prefix}/{MANIFEST_FILENAME}"
        edit_manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"

        async with self.s3_controller.create_client_async() as s3_client:
            try:
                # Try to read the finalized manifest.yaml
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=manifest_key
                )
            except ClientError:
                # Fallback to _manifest.yaml if manifest.yaml does not exist
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=edit_manifest_key
                )

            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())
            return manifest

    async def validate(self, prefix, context: dict = None):
        """Validate the artifact, ensure all files are uploaded, and rename _manifest.yaml to manifest.yaml."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"
        final_manifest_key = f"{ws}/{prefix}/{MANIFEST_FILENAME}"

        # Get the _manifest.yaml
        async with self.s3_controller.create_client_async() as s3_client:
            manifest_obj = await s3_client.get_object(
                Bucket=self.workspace_bucket, Key=manifest_key
            )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())

            # Validate if all files in the manifest exist in S3
            for file_info in manifest["files"]:
                file_key = f"{ws}/{prefix}/{file_info['path']}"
                try:
                    await s3_client.head_object(
                        Bucket=self.workspace_bucket, Key=file_key
                    )
                except ClientError:
                    raise FileNotFoundError(
                        f"File '{file_info['path']}' does not exist in the artifact."
                    )

            # Rename _manifest.yaml to manifest.yaml
            response = await s3_client.copy_object(
                Bucket=self.workspace_bucket,
                CopySource={"Bucket": self.workspace_bucket, "Key": manifest_key},
                Key=final_manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to finalize manifest for artifact under prefix: {prefix}"

            # Delete the _manifest.yaml file
            await s3_client.delete_object(
                Bucket=self.workspace_bucket, Key=manifest_key
            )

            # If this artifact is part of a collection, update the collection index
            parent_prefix = "/".join(prefix.split("/")[:-1])
            try:
                await self.index(parent_prefix, context=context)
            except KeyError:
                pass

    async def list_artifacts(self, prefix="", pending=False, context: dict = None):
        """List all artifacts under a certain prefix."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )
        prefix_path = safe_join(ws, prefix) + "/"

        async with self.s3_controller.create_client_async() as s3_client:
            artifacts = await list_objects_async(
                s3_client, self.workspace_bucket, prefix=prefix_path
            )
            artifact_list = []

            try:
                # Check if the prefix is a collection
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket,
                    Key=f"{prefix_path}{MANIFEST_FILENAME}",
                )
                manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())
                if manifest.get("type") == "collection":
                    return manifest.get("collection", [])
            except ClientError:
                pass

            # Otherwise, return a dynamic list of artifacts
            for obj in artifacts:
                if obj["type"] == "directory":
                    name = obj["name"]
                    try:
                        # head the manifest file to check if it's finalized
                        if pending:
                            await s3_client.head_object(
                                Bucket=self.workspace_bucket,
                                Key=f"{prefix_path}{name}/{EDIT_MANIFEST_FILENAME}",
                            )
                            artifact_list.append({"name": f"{name}", "pending": True})
                        else:
                            await s3_client.head_object(
                                Bucket=self.workspace_bucket,
                                Key=f"{prefix_path}{name}/{MANIFEST_FILENAME}",
                            )
                            artifact_list.append({"name": f"{name}"})
                    except ClientError:
                        pass
            return artifact_list

    async def put_file(self, prefix, file_path, context: dict = None):
        """Put a file to an artifact and return the pre-signed URL."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"

        # Get the _manifest.yaml
        async with self.s3_controller.create_client_async() as s3_client:
            manifest_obj = await s3_client.get_object(
                Bucket=self.workspace_bucket, Key=manifest_key
            )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())

            # Generate a pre-signed URL for the file
            file_key = f"{ws}/{prefix}/{file_path}"
            presigned_url = await s3_client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.workspace_bucket, "Key": file_key},
                ExpiresIn=3600,
            )

            # Add the file to the manifest
            file_info = {"path": file_path}
            manifest["files"].append(file_info)

            # Update the _manifest.yaml in S3
            response = await s3_client.put_object(
                Body=yaml.dump(manifest),
                Bucket=self.workspace_bucket,
                Key=manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to update manifest for artifact under prefix: {prefix}"

        return presigned_url

    async def remove_file(self, prefix, file_path, context: dict = None):
        """Remove a file from the artifact and update the _manifest.yaml."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"

        # Get the _manifest.yaml
        async with self.s3_controller.create_client_async() as s3_client:
            manifest_obj = await s3_client.get_object(
                Bucket=self.workspace_bucket, Key=manifest_key
            )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())

            # Remove the file from the manifest
            manifest["files"] = [f for f in manifest["files"] if f["path"] != file_path]

            # Remove the file from S3
            file_key = f"{ws}/{prefix}/{file_path}"
            response = await s3_client.delete_object(
                Bucket=self.workspace_bucket, Key=file_key
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 204
            ), f"Failed to delete file: {file_path}"

            # Update the _manifest.yaml in S3
            response = await s3_client.put_object(
                Body=yaml.dump(manifest),
                Bucket=self.workspace_bucket,
                Key=manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to update manifest after removing file: {file_path}"

    async def get_file(self, prefix, path, context: dict = None):
        """Return a pre-signed URL for a file."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read):
            raise PermissionError(
                "User does not have read permission to the workspace."
            )
        file_key = f"{ws}/{prefix}/{path}"

        # Perform a head_object check before generating the URL
        async with self.s3_controller.create_client_async() as s3_client:
            await s3_client.head_object(Bucket=self.workspace_bucket, Key=file_key)

            # Generate a pre-signed URL for the file
            presigned_url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.workspace_bucket, "Key": file_key},
                ExpiresIn=3600,
            )

        return presigned_url

    async def delete(self, prefix, context: dict = None):
        """Delete an entire artifact including all its files and the manifest."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        artifact_path = f"{ws}/{prefix}/"

        # Remove all objects under the artifact's path
        async with self.s3_controller.create_client_async() as s3_client:
            await remove_objects_async(s3_client, self.workspace_bucket, artifact_path)

    def get_artifact_service(self):
        """Get artifact controller."""
        return {
            "id": "artifact",
            "config": {"visibility": "public", "require_context": True},
            "name": "Artifact",
            "create": self.create,
            "read": self.read,
            "validate": self.validate,
            "delete": self.delete,
            "put_file": self.put_file,
            "remove_file": self.remove_file,
            "get_file": self.get_file,
            "list": self.list_artifacts,
            "index": self.index,  # New index function
        }
