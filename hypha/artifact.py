"""Provide an s3 interface."""
import logging
import sys
import yaml
from botocore.exceptions import ClientError

from hypha.core import (
    UserInfo,
    UserPermission,
    Artifact,
    CollectionArtifact,
    ApplicationArtifact,
)
from hypha.utils import (
    list_objects_async,
    remove_objects_async,
    safe_join,
)
from hypha_rpc.utils import ObjectProxy
from jsonschema import validate

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
        store.set_artifact_manager(self)

    async def create(
        self, prefix, manifest: dict, overwrite=False, stage=False, context: dict = None
    ):
        """Create a new artifact with a manifest file."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )

        if manifest["type"] == "collection":
            # Validate the collection manifest
            CollectionArtifact.model_validate(manifest)
        elif manifest["type"] == "application":
            ApplicationArtifact.model_validate(manifest)
        else:
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

            if manifest_exists or edit_manifest_exists:
                if not overwrite:
                    raise FileExistsError(
                        f"Artifact under prefix '{prefix}' already exists. Use overwrite=True to overwrite."
                    )
                else:
                    logger.info(f"Overwriting artifact under prefix: {prefix}")
                    await self.delete(prefix, context=context)

            # Check if the artifact is a collection and initialize it
            if manifest.get("type") == "collection":
                collection = await self._build_collection_index(
                    manifest, ws, prefix, s3_client
                )
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

            if not stage and manifest["type"] == "collection":
                manifest["collection"] = await self._build_collection_index(
                    manifest, ws, prefix, s3_client
                )

            # Update the parent collection index if the artifact is part of a collection
            if not stage and "/" in prefix:
                parent_prefix = "/".join(prefix.split("/")[:-1])
                # check if the collection manifest file exits
                try:
                    collection_key = f"{ws}/{parent_prefix}/{MANIFEST_FILENAME}"
                    await s3_client.head_object(
                        Bucket=self.workspace_bucket, Key=collection_key
                    )
                    parent_collection = True
                except ClientError:
                    parent_collection = False
            else:
                parent_collection = False
            # Upload the _manifest.yaml to indicate it's in edit mode
            response = await s3_client.put_object(
                Body=yaml.dump(ObjectProxy.toDict(manifest)),
                Bucket=self.workspace_bucket,
                Key=edit_manifest_key if stage else manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to create artifact under prefix: {prefix}"
            if parent_collection:
                await self.index(parent_prefix, context=context)

        return manifest

    async def edit(self, prefix, manifest=None, context: dict = None):
        """Edit the artifact's manifest."""
        ws = context["ws"]
        user_info = UserInfo.model_validate(context["user"])
        if not user_info.check_permission(ws, UserPermission.read_write):
            raise PermissionError(
                "User does not have write permission to the workspace."
            )
        # copy the manifest to _manifest.yaml
        manifest_key = f"{ws}/{prefix}/{MANIFEST_FILENAME}"
        edit_manifest_key = f"{ws}/{prefix}/{EDIT_MANIFEST_FILENAME}"

        async with self.s3_controller.create_client_async() as s3_client:
            if not manifest:
                # Get the manifest
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=manifest_key
                )
                manifest_str = (await manifest_obj["Body"].read()).decode()
                manifest = yaml.safe_load(manifest_str)
            Artifact.model_validate(manifest)
            # Upload the _manifest.yaml to indicate it's in edit mode
            response = await s3_client.put_object(
                Body=yaml.dump(manifest),
                Bucket=self.workspace_bucket,
                Key=edit_manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to edit artifact under prefix: {prefix}"

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
                # Check if this is a valid collection
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=collection_key
                )
            except ClientError:
                raise KeyError(
                    f"Collection manifest file does not exist for prefix '{prefix}'"
                )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())

            collection = await self._build_collection_index(
                manifest, ws, prefix, s3_client
            )
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
                name = obj["name"]
                try:
                    sub_manifest_key = f"{sub_prefix}{name}/{MANIFEST_FILENAME}"
                    sub_manifest_obj = await s3_client.get_object(
                        Bucket=self.workspace_bucket, Key=sub_manifest_key
                    )
                    manifest_str = (await sub_manifest_obj["Body"].read()).decode()
                    sub_manifest = yaml.safe_load(manifest_str)

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
                    summary["_id"] = name
                    collection.append(summary)
                except ClientError:
                    pass
                except Exception as e:
                    logger.error(f"Error while building collection index: {e}")
                    raise e

        return collection

    async def read(self, prefix, stage=False, context: dict = None):
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
                # Read the manifest depending on the stage
                if stage:
                    manifest_obj = await s3_client.get_object(
                        Bucket=self.workspace_bucket, Key=edit_manifest_key
                    )
                else:
                    manifest_obj = await s3_client.get_object(
                        Bucket=self.workspace_bucket, Key=manifest_key
                    )
            except ClientError:
                # If manifest.yaml does not exist and it's not in edit mode, raise an error
                raise KeyError(f"Artifact under prefix '{prefix}' does not exist.")

            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())
            return manifest

    async def commit(self, prefix, context: dict = None):
        """Commit the artifact, ensure all files are uploaded, and rename _manifest.yaml to manifest.yaml."""
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
            try:
                await s3_client.head_object(
                    Bucket=self.workspace_bucket, Key=final_manifest_key
                )
            except ClientError:
                pass

            if "/" in prefix:
                # If this artifact is part of a collection, update the collection index
                parent_prefix = "/".join(prefix.split("/")[:-1])
                # check if the collection manifest file exits
                try:
                    collection_key = f"{ws}/{parent_prefix}/{MANIFEST_FILENAME}"
                    await s3_client.head_object(
                        Bucket=self.workspace_bucket, Key=collection_key
                    )
                    parent_collection = True
                except ClientError:
                    parent_collection = False
            else:
                parent_collection = False

            try:
                manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=manifest_key
                )
            except ClientError:
                raise FileNotFoundError(
                    f"Artifact under prefix '{prefix}' does not exist."
                )
            manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())

            # Validate if all files in the manifest exist in S3
            if "files" in manifest:
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
            # if parent_collection is True, check if collection_schema exists in the parent schema
            if parent_collection:
                parent_manifest_obj = await s3_client.get_object(
                    Bucket=self.workspace_bucket, Key=collection_key
                )
                parent_manifest = yaml.safe_load(
                    (await parent_manifest_obj["Body"].read()).decode()
                )
                if parent_manifest.get("collection_schema"):
                    collection_schema = parent_manifest.get("collection_schema")
                    if collection_schema:
                        validate(instance=manifest, schema=collection_schema)
            # Rename _manifest.yaml to manifest.yaml
            response = await s3_client.copy_object(
                Bucket=self.workspace_bucket,
                CopySource={"Bucket": self.workspace_bucket, "Key": manifest_key},
                Key=final_manifest_key,
            )
            assert (
                "ResponseMetadata" in response
                and response["ResponseMetadata"]["HTTPStatusCode"] == 200
            ), f"Failed to commit manifest for artifact under prefix: {prefix}"

            # Delete the _manifest.yaml file
            await s3_client.delete_object(
                Bucket=self.workspace_bucket, Key=manifest_key
            )

            if parent_collection:
                await self.index(parent_prefix, context=context)

    async def list_artifacts(self, prefix="", stage=False, context: dict = None):
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
                    Key=f"{prefix_path}{MANIFEST_FILENAME}"
                    if not stage
                    else f"{prefix_path}{EDIT_MANIFEST_FILENAME}",
                )
                manifest = yaml.safe_load((await manifest_obj["Body"].read()).decode())
                if manifest.get("type") == "collection":
                    return [item["_id"] for item in manifest.get("collection", [])]
            except ClientError:
                pass

            # Otherwise, return a dynamic list of artifacts
            for obj in artifacts:
                if obj["type"] == "directory":
                    name = obj["name"]
                    try:
                        # head the manifest file to check if it's finalized
                        if stage:
                            await s3_client.head_object(
                                Bucket=self.workspace_bucket,
                                Key=f"{prefix_path}{name}/{EDIT_MANIFEST_FILENAME}",
                            )
                            artifact_list.append(name)
                        else:
                            await s3_client.head_object(
                                Bucket=self.workspace_bucket,
                                Key=f"{prefix_path}{name}/{MANIFEST_FILENAME}",
                            )
                            artifact_list.append(name)
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
            manifest["files"] = manifest.get("files", [])
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
            manifest["files"] = manifest.get("files", [])
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

        async with self.s3_controller.create_client_async() as s3_client:
            # Remove all objects under the artifact's path
            await remove_objects_async(s3_client, self.workspace_bucket, artifact_path)

            if "/" in prefix:
                # If this artifact is part of a collection, re-index the parent collection
                parent_prefix = "/".join(prefix.split("/")[:-1])
                collection_key = f"{ws}/{parent_prefix}/{MANIFEST_FILENAME}"
                # check if the collection manifest file exits
                try:
                    await s3_client.head_object(
                        Bucket=self.workspace_bucket, Key=collection_key
                    )
                    await self.index(parent_prefix, context=context)
                except ClientError:
                    pass

    def get_artifact_service(self):
        """Get artifact controller."""
        return {
            "id": "artifact-manager",
            "config": {"visibility": "public", "require_context": True},
            "name": "Artifact Manager",
            "description": "Manage artifacts in a workspace.",
            "create": self.create,
            "edit": self.edit,
            "read": self.read,
            "commit": self.commit,
            "delete": self.delete,
            "put_file": self.put_file,
            "remove_file": self.remove_file,
            "get_file": self.get_file,
            "list": self.list_artifacts,
            "index": self.index,  # New index function
        }
