# Artifact Manager

The `Artifact Manager` is an essential Hypha service for managing resources such as datasets, AI models, and applications. It allows for structured resource management, providing APIs to create collections, manage datasets, track download statistics, enforce schemas, and control access permissions. The `Artifact Manager` can be used as a backend for web applications, supporting complex operations like indexing, searching, and schema validation.

**Note:** The `Artifact Manager` is available only when S3 storage is enabled on your Hypha server.

---

## Getting Started

### Step 1: Connecting to the Artifact Manager Service

To use the `Artifact Manager`, start by connecting to the Hypha server. This connection allows you to interact with the artifact registry, including creating, editing, and deleting datasets, as well as managing permissions and tracking download statistics.

```python
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"  # Replace with your server URL

# Connect to the Artifact Manager API
server = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
artifact_manager = await server.get_service("public/artifact-manager")
```

### Step 2: Creating a Dataset Gallery Collection

Once connected, you can create a collection that organizes datasets, providing metadata and access permissions for each.

```python
# Define metadata for the dataset gallery
gallery_manifest = {
    "name": "Dataset Gallery",
    "description": "A collection for organizing datasets",
}

# Create the collection with read access for everyone and create access for authenticated users
collection = await artifact_manager.create(
    alias="dataset-gallery",
    type="collection",
    manifest=gallery_manifest,
    config={"permissions": {"*": "r", "@": "r+"}}
)
print("Dataset Gallery created with ID:", collection.id)
```

**Tips: The returned `collection.id` is the unique identifier for the collection with the format `workspace_id/alias`. You can use this ID to refer to the collection in subsequent operations. You can also use only `alias` as a shortcut, however, this only works in the same workspace.**

### Step 3: Adding a Dataset to the Gallery

After creating the gallery, you can add datasets to it, with each dataset having its own metadata and permissions.

```python
# Define metadata for the new dataset
dataset_manifest = {
    "name": "Example Dataset",
    "description": "A dataset containing example data",
}

# Add the dataset to the gallery and stage it for review
dataset = await artifact_manager.create(
    parent_id=collection.id,
    alias="example-dataset",
    manifest=dataset_manifest,
    stage=True
)
print("Dataset added to the gallery.")
```

**Tips: The `stage=True` parameter stages the dataset for review and commit. You can edit the dataset before committing it to the collection.**

### Step 4: Uploading Files to the Dataset with Download Statistics

Each dataset can contain multiple files. Use pre-signed URLs for secure uploads and set `download_weight` to track file downloads.

**Important:** Adding files to a staged artifact does NOT automatically create a new version. You must explicitly specify `version="new"` when editing if you want to create a new version upon commit.

```python
# Generate a pre-signed URL to upload a file with a specific download weight
put_url = await artifact_manager.put_file(
    dataset.id, file_path="data.csv", download_weight=0.5
)

# Upload the file using an HTTP PUT request
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

print("File uploaded to the dataset.")
```

### Step 5: Committing the Dataset

After uploading files, commit the dataset to finalize its status in the collection. Since no `version="new"` was specified when staging or adding files, this will create the initial version v0.

```python
# Commit the dataset to finalize its status
await artifact_manager.commit(dataset.id)
print("Dataset committed.")
```

### Step 6: Listing All Datasets in the Gallery

Retrieve a list of all datasets in the collection for display or further processing.

```python
# List all datasets in the gallery
datasets = await artifact_manager.list(collection.id)
print("Datasets in the gallery:", datasets)
```

---

## Full Example: Creating and Managing a Dataset Gallery

Here's a complete example showing how to connect to the service, create a dataset gallery, add a dataset, upload files, and commit the dataset.

```python
import asyncio
import requests
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Artifact Manager API
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for the Dataset Gallery
    gallery_manifest = {
        "name": "Dataset Gallery",
        "description": "A collection for organizing datasets",
    }
    collection = await artifact_manager.create(
        type="collection",
        alias="dataset-gallery",
        manifest=gallery_manifest,
        config={"permissions": {"*": "r+", "@": "r+"}}
    )
    print("Dataset Gallery created.")

    # Add a dataset to the gallery
    dataset_manifest = {
        "name": "Example Dataset",
        "description": "A dataset containing example data",
    }
    dataset = await artifact_manager.create(
        parent_id=collection.id,
        alias="example-dataset",
        manifest=dataset_manifest,
        stage=True
    )
    print("Dataset added to the gallery.")

    # Upload a file to the dataset
    put_url = await artifact_manager.put_file(dataset.id, file_path="data.csv", download_weight=0.5)
    with open("path/to/local/data.csv", "rb") as f:
        response = requests.put(put_url, data=f)
        assert response.ok, "File upload failed"
    print("File uploaded to the dataset.")

    # Commit the dataset
    await artifact_manager.commit(dataset.id)
    print("Dataset committed.")

    # List all datasets in the gallery
    datasets = await artifact_manager.list(collection.id)
    print("Datasets in the gallery:", datasets)

asyncio.run(main())
```

---

## Advanced Usage: Version Management and Performance Optimization

### Understanding the New Version Control System

The Artifact Manager has been optimized for better performance and more predictable version management. Key improvements include:

#### 1. Explicit Version Control
- **No Automatic Versioning**: Simply adding files to a staged artifact does not create a new version
- **Explicit Intent Required**: You must use `version="new"` when editing to create new versions
- **Predictable Costs**: Prevents unexpected version proliferation that can lead to high storage costs

#### 2. Direct File Placement
- **Optimized Upload**: Files are placed directly in their final S3 destination based on version intent
- **No File Copying**: Eliminates expensive file copying operations during commit
- **Fast Commits**: Commit operations are now metadata-only and complete quickly regardless of file size

#### 3. Version Creation Examples

```python
# Example 1: Update existing version (no new version created)
await artifact_manager.edit(artifact_id=dataset.id, manifest=updated_manifest, stage=True)
# Add files - these go to existing version
put_url = await artifact_manager.put_file(artifact_id=dataset.id, file_path="update.csv")
await artifact_manager.commit(artifact_id=dataset.id)  # Updates existing version

# Example 2: Create new version explicitly
await artifact_manager.edit(
    artifact_id=dataset.id, 
    manifest=updated_manifest, 
    stage=True,
    version="new"  # Explicit new version intent
)
# Add files - these go to new version location
put_url = await artifact_manager.put_file(artifact_id=dataset.id, file_path="new_data.csv")
await artifact_manager.commit(artifact_id=dataset.id, version="v2.0")  # Creates new version
```

#### 4. Performance Benefits
- **Large File Support**: Adding large files is much faster since they're placed directly in final location
- **Reduced S3 Costs**: No duplicate file storage during staging process
- **Fast Commits**: Commit operations complete in seconds, not minutes, regardless of data size
- **Efficient Storage**: Only one copy of each file is stored per version

---

## Advanced Usage: Enforcing Schema for Dataset Validation

The `Artifact Manager` allows schema enforcement to ensure datasets meet specific requirements. This is helpful when maintaining consistency across multiple datasets.

### Step 1: Create a Collection with a Schema

Define a schema for datasets to specify required fields.

```python
# Define a schema for datasets in the gallery
dataset_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "record_type": {"type": "string", "enum": ["dataset", "model", "application"]},
    },
    "required": ["name", "description", "record_type"]
}

# Create a collection with schema validation
gallery_manifest = {
    "name": "Schema Dataset Gallery",
    "description": "A gallery with schema-enforced datasets",
}
collection = await artifact_manager.create(
    type="collection",
    alias="schema-dataset-gallery",
    manifest=gallery_manifest,
    config={"collection_schema": dataset_schema, "permissions": {"*": "r+", "@": "r+"}}
)
print("Schema-based Dataset Gallery created.")
```

### Step 2: Validate Dataset Against Schema

Datasets in the collection will be validated against the defined schema during creation and commit.

```python
# Define a valid dataset that conforms to the schema
valid_dataset_manifest = {
    "name": "Valid Dataset",
    "description": "A valid dataset meeting schema requirements",
    "record_type": "dataset",
}

dataset = await artifact_manager.create(
    parent_id=collection.id,
    alias="valid-dataset",
    manifest=valid_dataset_manifest,
    stage=True
)
print("Valid dataset created.")

# Commit the dataset (will pass schema validation)
await artifact_manager.commit(dataset.id)
print("Valid dataset committed.")
```


---

## API References

### `create(parent_id: str, alias: str, type: str, manifest: dict, permissions: dict=None, config: dict=None, version: str = None, stage: bool = False, comment: str = None, overwrite: bool = False) -> None`

Creates a new artifact or collection with the specified manifest. The artifact is staged until committed. For collections, the `collection` field should be an empty list.

**Parameters:**

- `alias`: A human readable name for indexing the artifact, it can be a text with lower case letters and numbers. You can set it to absolute alias in the format of `"workspace_id/alias"` or just `"alias"`. In the alias itself, `/` is not allowed, you should use `:` instead in the alias. If the alias already exists in the workspace, it will raise an error and you need to either delete the existing artifact or use a different alias.
  To generate an auto-id, you can use patterns like `"{uuid}"` or `"{timestamp}"`. The following patterns are supported:
  - `{uuid}`: Generates a random UUID.
  - `{timestamp}`: Generates a timestamp in milliseconds.
  - `{user_id}`: Generate the user id of the creator.
  - `{zenodo_id}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo deposit ID (which changes when a new version is created).
  - `{zenodo_conceptrecid}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo concept id (which does not change when a new version is created).
  - **Id Parts**: You can also use id parts stored in the parent collection's config['id_parts'] to generate an id. For example, if the parent collection has `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`, you can use `"{colors}-{animals}"` to generate an id like `red-dog`.
- `workspace`: Optional. The workspace id where the artifact will be created. If not set, it will be created in the default workspace. If specified, it should match the workspace in the alias and also the parent_id.
- `parent_id`: The id of the parent collection where the artifact will be created. If the artifact is a top-level collection, leave this field empty or set to None.
- `type`: The type of the artifact. Supported values are `collection`, `vector-collection`, `generic` and any other custom type. By default, it's set to `generic` which contains fields tailored for displaying the artifact as cards on a webpage.
- `manifest`: The manifest of the new artifact. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `config`: Optional. A dictionary containing additional configuration options for the artifact (shared for both staged and committed). For collections, the config can contain the following special fields:
  - `collection_schema`: Optional. A JSON schema that defines the structure of child artifacts in the collection. This schema is used to validate child artifacts when they are created or edited. If a child artifact does not conform to the schema, the creation or edit operation will fail.
  - `id_parts`: Optional. A dictionary of id name parts to be used in generating the id for child artifacts. For example: `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`. This can be used for creating child artifacts with auto-generated ids based on the id parts. For example, when calling `create`, you can specify the alias as `my-pet-{colors}-{animals}`, and the id will be generated based on the id parts, e.g., `my-pet-red-dog`.
  - `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
  - `list_fields`: Optional. A list of fields to be collected when calling ``list`` function. By default, it collects all fields in the artifacts. If you want to collect only specific fields, you can set this field to a list of field names, e.g. `["manifest", "download_count"]`.
  - `publish_to`: Optional. A string specifying the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. If set, the artifact will be published to the specified platform. The artifact must have a valid Zenodo metadata schema to be published.
- `version`: Optional. **Version Creation Behavior**: Controls initial version creation for the artifact.
  - `None` or `"new"` (default): Creates version "v0" immediately (unless `stage=True`)
  - `"stage"`: Creates the artifact in staging mode (equivalent to `stage=True`)
  
  **Important**: If `stage=True` is specified, any version parameter is ignored and the artifact starts in staging mode. When a staged artifact is committed, it will create version "v0" by default.
- `stage`: Optional. If `True`, the artifact will be created in staging mode regardless of the version parameter. Default is `False`. When in staging mode, you must call `commit()` to finalize the artifact with its first version.
- `comment`: Optional. A comment to describe the changes made to the artifact.
- `secrets`: Optional. A dictionary containing secrets to be stored with the artifact. Secrets are encrypted and can only be accessed by the artifact owner or users with appropriate permissions. The following keys can be used:
  - `ZENODO_ACCESS_TOKEN`: The Zenodo access token to publish the artifact to Zenodo.
  - `SANDBOX_ZENODO_ACCESS_TOKEN`: The Zenodo access token to publish the artifact to the Zenodo sandbox.
  - `S3_ENDPOINT_URL`: The endpoint URL of the S3 storage for the artifact.
  - `S3_ACCESS_KEY_ID`: The access key ID for the S3 storage for the artifact.
  - `S3_SECRET_ACCESS_KEY`: The secret access key for the S3 storage for the artifact.
  - `S3_REGION_NAME`: The region name of the S3 storage for the artifact.
  - `S3_BUCKET`: The bucket name of the S3 storage for the artifact. Default to the hypha workspaces bucket.
  - `S3_PREFIX`: The prefix of the S3 storage for the artifact. Default: `""`.
  - `S3_PUBLIC_ENDPOINT_URL`: The public endpoint URL of the S3 storage for the artifact. If the S3 server is not public, you can set this to the public endpoint URL. Default: `None`.
- `overwrite`: Optional. A boolean flag to overwrite the existing artifact with the same alias. Default is `False`.


**Note 1: If you set `stage=True`, you must call `commit()` to finalize the artifact.**

**Example:**

```python
# Assuming we have already created a dataset-gallery collection, we can add a new dataset to it
await artifact_manager.create(artifact_id="dataset-gallery", alias="example-dataset", manifest=dataset_manifest, stage=True)
```

### Permissions

In the `Artifact Manager`, permissions allow users to control access to artifacts and collections. The system uses a flexible permission model that supports assigning different levels of access to individual users, authenticated users, or all users. Permissions are stored in the `permissions` field of an artifact manifest, and they define which operations a user can perform, such as reading, writing, creating, or committing changes.

Permissions can be set both at the artifact level and the workspace level. In the absence of explicit artifact-level permissions, workspace-level permissions are checked to determine access. Here's how permissions work in detail:

**Permission Levels:**

The following permission levels are supported:

- **n**: No access to the artifact.
- **l**: List-only access (includes `list`).
- **l+**: List and create access (includes `list`, `create`, and `commit`).
- **lv**: List and list vectors access (includes `list` and `list_vectors`).
- **lv+**: List, list vectors, create, and commit access (includes `list`, `list_vectors`, `create`, `commit`, `add_vectors`, and `add_documents`).
- **lf**: List and list files access (includes `list` and `list_files`).
- **lf+**: List, list files, create, and commit access (includes `list`, `list_files`, `create`, `commit`, and `put_file`).
- **r**: Read-only access (includes `read`, `get_file`, `list_files`, `list`, `search_vectors`, and `get_vector`).
- **r+**: Read, write, and create access (includes `read`, `get_file`, `put_file`, `list_files`, `list`, `search_vectors`, `get_vector`, `create`, `commit`, `add_vectors`, and `add_documents`).
- **rw**: Read, write, and create access with file management (includes `read`, `get_file`, `get_vector`, `search_vectors`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, and `remove_vectors`).
- **rw+**: Read, write, create, and manage access (includes `read`, `get_file`, `get_vector`, `search_vectors`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, `remove_vectors`, and `create`).
- **\***: Full access to all operations (includes `read`, `get_file`, `get_vector`, `search_vectors`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, `remove_vectors`, `create`, and `reset_stats`).

**Shortcut Permission Notation:**

- **\***: Refers to all users, both authenticated and anonymous. For example, `{"*": "r+"}` gives read and create access to everyone.
- **@**: Refers to authenticated (logged-in) users only. For instance, `{"@": "rw"}` allows all logged-in users to read and write, but restricts anonymous users.
- **user_id**: Specifies permissions for a specific user, identified by their `user_id`. For example, `{"user_id_1": "r+"}` grants read and create permissions to the user with ID `user_id_1`.

When multiple notations are used together, the most specific permission takes precedence. For example, if a user has both `{"*": "r"}` and `{"user_id_1": "n"}` permissions, the user will have no access because the specific permission `{"user_id_1": "n"}` overrides the general permission `{"*": "r"}`.

**Example Permissions:**

1. **Public Read-Only Access:**
   ```json
   {
     "permissions": {
       "*": "r"
     }
   }
   ```
   This grants read-only access to everyone, including anonymous users.

2. **Read, Write and Create Access for Logged-In Users:**
   ```json
   {
     "permissions": {
       "@": "rw+"
     }
   }
   ```
   This allows all authenticated users to read, write and create, but restricts access for anonymous users.

3. **Full Access for a Specific User:**
   ```json
   {
     "permissions": {
       "user_id_1": "rw+"
     }
   }
   ```
   This grants a specific user full access to the artifact, including reading, writing, creating, and managing files.

**Permission Hierarchy:**

1. **Artifact-Level Permissions:** If permissions are explicitly defined for an artifact, those will take precedence over workspace-level permissions.
2. **Workspace-Level Permissions:** If no artifact-specific permissions are set, the system checks the workspace-level permissions.
3. **Parent Artifact Permissions:** If an artifact does not define permissions explicitly, the permissions of its parent artifact in the collection hierarchy are checked.

**Permission Expansion:**

Permissions can be expanded to cover multiple operations. For example, a permission of `"r"` will automatically include operations like reading files, listing contents, and performing searches. Similarly, a permission of `"rw+"` covers all operations, including creating, editing, committing, and managing files.

The following list shows how permission expansions work:

- `"n"`: No operations allowed.
- `"r"`: Includes `read`, `get_file`, `list_files`, and `list`.
- `"r+"`: Includes all operations in `"r"`, plus `put_file`, `create`, and `commit`.
- `"rw"`: Includes all operations in `"r+"`, plus `edit`, `commit`, and `remove_file`.
- `"rw+"`: Includes all operations in `"rw"`, plus `reset_stats`.

---

### `edit(artifact_id: str, manifest: dict = None, type: str = None,  permissions: dict = None, config: dict = None, secrets: dict = None, version: str = None, stage: bool = False, comment: str = None) -> None`

Edits an existing artifact's manifest. The new manifest is staged until committed if staging mode is enabled.

**Parameters:**

- `artifact_id`: The id of the artifact to edit. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `manifest`: The updated manifest. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `type`: Optional. The type of the artifact. Supported values are `collection`, `generic` and any other custom type. By default, it's set to `generic` which contains fields tailored for displaying the artifact as cards on a webpage.
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
- `secrets`: Optional. A dictionary containing secrets to be stored with the artifact. Secrets are encrypted and can only be accessed by the artifact owner or users with appropriate permissions. See the `create` function for a list of supported secrets.
- `config`: Optional. A dictionary containing additional configuration options for the artifact.
- `version`: Optional. **Strict Validation Applied**: Must be `None`, `"new"`, or an existing version name from the artifact's versions array. Custom version names are NOT allowed during edit - they can only be specified during `commit()`.
  - `None` (default): Updates the latest existing version
  - `"new"`: Creates a new version immediately (unless `stage=True`)
  - `"stage"`: Enters staging mode (equivalent to `stage=True`)
  - `"existing_version_name"`: Updates the specified existing version (must already exist in the versions array)
  
  **Special Staging Behavior**: When `stage=True` and `version="new"` are used together, it stores an intent marker for creating a new version during commit, and allows custom version names to be specified later during the `commit()` operation.
- `stage`: Optional. If `True`, the artifact will be edited in staging mode regardless of the version parameter. Default is `False`. When in staging mode, the artifact must be committed to finalize changes.
- `comment`: Optional. A comment to describe the changes made to the artifact.

**Important Notes:**
- **Version Validation**: The system now strictly validates the `version` parameter. You cannot specify arbitrary custom version names during edit.
- **Staging Intent System**: When you use `version="new"` with `stage=True`, the system stores an "intent" to create a new version, which allows you to specify a custom version name later during `commit()`.
- **Historical Version Editing**: You can edit specific existing versions by providing their exact version name (e.g., `version="v1"`), but the version must already exist.

**Example:**

```python
# Edit an artifact and update the latest version
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest
)

# Edit an artifact and create a new version immediately
await artifact_manager.edit(
    artifact_id="example-dataset", 
    manifest=updated_manifest,
    version="new"
)

# Edit an artifact in staging mode with intent to create new version on commit
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest,
    stage=True,
    version="new"  # This sets intent for new version creation during commit
)

# Edit a specific existing version (version must already exist)
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest,
    version="v1"  # Must be an existing version name from the versions array
)

# INVALID: This will raise an error
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest,
    version="custom-v2.0"  # ERROR: Custom version names not allowed during edit
)
```

---

### `commit(artifact_id: str, version: str = None, comment: str = None) -> dict`

Commits staged changes to an artifact, creating a new version or updating the existing latest version depending on the staging mode and version intent.

**Parameters:**

- `artifact_id`: The id of the artifact to commit. Can be a UUID or alias.
- `version`: Optional custom version name. If not provided, a sequential version name will be generated.
- `comment`: Optional comment describing the changes in this version.

**Returns:**

A dictionary containing the committed artifact with updated versions and no staging data.

**Behavior:**

- **Version Creation:** Only creates a new version if `version="new"` was explicitly specified during edit operations
- **Version Update:** If no new version intent exists, updates the existing latest version (or creates v0 if no versions exist)
- **Metadata-Only Operation:** Commit is now a fast, metadata-only operation with no file copying
- **File Placement:** Files are already in their final locations (placed there during upload based on intent)
- **Staging Cleanup:** The staging area is cleared and metadata is finalized

**Performance Note:** The commit operation is highly optimized and performs no file copying, making it fast even for artifacts with large files.

### `discard(artifact_id: str) -> dict`

Discards all staged changes for an artifact, reverting to the last committed state.

**Parameters:**

- `artifact_id`: The id of the artifact to discard changes for. Can be a UUID or alias.

**Returns:**

A dictionary containing the artifact reverted to its last committed state.

**Behavior:**

- Clears all staged manifest changes, reverting to the last committed manifest
- Removes all staged files from S3 storage
- Clears the staging area (sets `staging` to `None`)
- Preserves all committed versions and their files
- Raises an error if no staged changes exist

**Example:**
```python
# Create an artifact and make some changes
artifact = await artifact_manager.create(type="dataset", manifest={"name": "Test"}, stage=True)
await artifact_manager.put_file(artifact_id=artifact.id, file_path="test.txt")

# Discard all staged changes
reverted = await artifact_manager.discard(artifact_id=artifact.id)
assert reverted["staging"] is None  # No staged changes
```

---

### `delete(artifact_id: str, delete_files: bool = False, recursive: bool = False) -> None`

Deletes an artifact, its manifest, and all associated files from both the database and S3 storage.

**Parameters:**

- `artifact_id`: The id of the artifact to delete. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `delete_files`: Optional. A boolean flag to delete all files associated with the artifact. Default is `False`.
- `recursive`: Optional. A boolean flag to delete all child artifacts recursively. Default is `False`.

**Warning: If `delete_files` is set to `True`, `recursive` must be set to `True`, all child artifacts will be deleted, and all files associated with the child artifacts will be permanently deleted from the S3 storage. This operation is irreversible.**

**Example:**

```python
await artifact_manager.delete(artifact_id=artifact.id, delete_files=True)

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.delete(artifact_id="example-dataset", delete_files=True)

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.delete(artifact_id="other_workspace/example-dataset", delete_files=True)
```

---

### `add_vectors(artifact_id: str, vectors: list, embedding_models: Optional[Dict[str, str]] = None, update: bool = False, context: dict = None) -> None`

Adds vectors to a vector collection artifact. See [RAG](./rag.md) for more detailed use of vector collections.

**Parameters:**

- `artifact_id`: The ID of the artifact to which vectors will be added. This must be a vector-collection artifact.
- `vectors`: A list of vectors to add to the collection.
- `embedding_models`: (Optional) A dictionary specifying embedding models to be used. If not provided, the default models from the artifact's configuration will be used.
- `update`: (Optional) A boolean flag to update existing vectors if they have the same ID. Defaults to `False`. If update is set to `True`, the existing vectors will be updated with the new vectors and every vector are expected to have an `id` field. If the vector does not exist, it will be added.
- `context`: A dictionary containing user and session context information.

**Returns:** None.

**Example:**

```python
await artifact_manager.add_vectors(artifact_id="example-id", vectors=[{"id": 1, "vector": [0.1, 0.2]}])
```

---

### `search_vectors(artifact_id: str, query: Optional[Dict[str, Any]] = None, embedding_models: Optional[str] = None, filters: Optional[dict[str, Any]] = None, limit: Optional[int] = 5, offset: Optional[int] = 0, return_fields: Optional[List[str]] = None, order_by: Optional[str] = None, pagination: Optional[bool] = False, context: dict = None) -> list`

Searches vectors in a vector collection artifact based on a query.

**Parameters:**

- `artifact_id`: The ID of the artifact to search within. This must be a vector-collection artifact.
- `query`: (Optional) A dictionary representing the query vector or conditions.
- `embedding_models`: (Optional) Specifies which embedding model to use. Defaults to the artifact's configuration.
- `filters`: (Optional) Filters for refining the search results.
- `limit`: (Optional) Maximum number of results to return. Defaults to 5.
- `offset`: (Optional) Number of results to skip. Defaults to 0.
- `return_fields`: (Optional) A list of fields to include in the results.
- `order_by`: (Optional) Field to order the results by.
- `pagination`: (Optional) Whether to include pagination metadata. Defaults to `False`.
- `context`: A dictionary containing user and session context information.

**Returns:** A list of search results.

**Example:**

```python
results = await artifact_manager.search_vectors(artifact_id="example-id", query={"vector": [0.1, 0.2]}, limit=10)
```

---

### `remove_vectors(artifact_id: str, ids: list, context: dict = None) -> None`

Removes vectors from a vector collection artifact. See [RAG](./rag.md) for more detailed use of vector collections.

**Parameters:**

- `artifact_id`: The ID of the artifact from which vectors will be removed. This must be a vector-collection artifact.
- `ids`: A list of vector IDs to remove.
- `context`: A dictionary containing user and session context information.

**Returns:** None.

**Example:**

```python
await artifact_manager.remove_vectors(artifact_id="example-id", ids=[1, 2, 3])
```

---

### `get_vector(artifact_id: str, id: int, context: dict = None) -> dict`

Fetches a specific vector by its ID from a vector collection artifact.

**Parameters:**

- `artifact_id`: The ID of the artifact to fetch the vector from. This must be a vector-collection artifact.
- `id`: The ID of the vector to fetch.
- `context`: A dictionary containing user and session context information.

**Returns:** A dictionary containing the vector data.

**Example:**

```python
vector = await artifact_manager.get_vector(artifact_id="example-id", id=123)
```

---

### `list_vectors(artifact_id: str, offset: int = 0, limit: int = 10, return_fields: List[str] = None, order_by: str = None, pagination: bool = False, context: dict = None) -> list`

Lists vectors in a vector collection artifact. See [RAG](./rag.md) for more detailed use of vector collections.

**Parameters:**

- `artifact_id`: The ID of the artifact to list vectors from. This must be a vector-collection artifact.
- `offset`: (Optional) Number of results to skip. Defaults to 0.
- `limit`: (Optional) Maximum number of results to return. Defaults to 10.
- `return_fields`: (Optional) A list of fields to include in the results.
- `order_by`: (Optional) Field to order the results by.
- `pagination`: (Optional) Whether to include pagination metadata. Defaults to `False`.
- `context`: A dictionary containing user and session context information.

**Returns:** A list of vectors.

**Example:**

```python
vectors = await artifact_manager.list_vectors(artifact_id="example-id", limit=20)
```

---

### `put_file(artifact_id: str, file_path: str, download_weight: float = 0) -> str`

Generates a pre-signed URL to upload a file to the artifact in S3. The URL can be used with an HTTP `PUT` request to upload the file. 

**File Placement Optimization:** Files are placed directly in their final destination based on version intent:
- If the artifact has `new_version` intent (set via `version="new"` during edit), files are uploaded to the new version location
- Otherwise, files are uploaded to the existing latest version location
- This eliminates the need for expensive file copying during commit operations

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"data.csv"`).
- `download_weight`: A float value representing the file's impact on download count when downloaded via any endpoint. Defaults to `0`. Files with weight `0` don't increment download count when accessed.

**Returns:** A pre-signed URL for uploading the file.

**Important Note:** Adding files to staging does NOT automatically create new version intent. You must explicitly use `version="new"` during edit operations if you want to create a new version when committing.

**Example:**

```python
put_url = await artifact_manager.put_file(artifact.id, file_path="data.csv", download_weight=1.0)

# If "example-dataset" is an alias of the artifact under the current workspace
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
put_url = await artifact_manager.put_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")

# Upload the file using an HTTP PUT request
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"
```

---

### `remove_file(artifact_id: str, file_path: str) -> None`

Removes a file from the artifact and updates the staged manifest. The file is also removed from the S3 storage.

**Parameters:**

- `artifact_id`: The id of the artifact to remove the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to be removed (e.g., `"data.csv"`).

**Example:**

```python
await artifact_manager.remove_file(artifact_id=artifact.id, file_path="data.csv")

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.remove_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.remove_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")
```

---

### `get_file(artifact_id: str, file_path: str, silent: bool = False) -> str`

Generates a pre-signed URL to download a file from the artifact stored in S3.

**Parameters:**

- `artifact_id`: The id of the artifact to download the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to download (e.g., `"data.csv"`).
- `silent`: A boolean to suppress the download count increment. Default is `False`.

**Returns:** A pre-signed URL for downloading the file.

**Example:**

```python
get_url = await artifact_manager.get_file(artifact_id=artifact.id, path="data.csv")

# If "example-dataset" is an alias of the artifact under the current workspace
get_url = await artifact_manager.get_file(artifact_id="example-dataset", path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
get_url = await artifact_manager.get_file(artifact_id="other_workspace/example-dataset", path="data.csv")
```

---

### `list_files(artifact_id: str, dir_path: str=None) -> list`

Lists all files in the artifact.

**Parameters:**

- `artifact_id`: The id of the artifact to list files from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `dir_path`: Optional. The directory path within the artifact to list files. Default is `None`.

**Returns:** A list of files in the artifact.

**Example:**

```python
files = await artifact_manager.list_files(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
files = await artifact_manager.list_files(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
files = await artifact_manager.list_files(artifact_id="other_workspace/example-dataset")
```

---

### `read(artifact_id: str, version: str = None, silent: bool = False, include_metadata: bool = False) -> dict`

Reads and returns the manifest of an artifact or collection. If in staging mode, reads the staged manifest.

**Parameters:**

- `artifact_id`: The id of the artifact to read. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `version`: Optional. The version of the artifact to read. By default, it reads the latest version. If you want to read a staged version, you can set it to `"stage"`.
- `silent`: Optional. If `True`, suppresses the view count increment. Default is `False`.
- `include_metadata`: Optional. If `True`, includes metadata such as download statistics in the manifest (fields starting with `"."`). Default is `False`.

**Returns:** The manifest as a dictionary, including view/download statistics and collection details (if applicable).

**Example:**

```python
artifact = await artifact_manager.read(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
artifact = await artifact_manager.read(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
artifact = await artifact_manager.read(artifact_id="other_workspace/example-dataset")


# Read a specific version
artifact = await artifact_manager.read(artifact_id="example-dataset", version="v1")

# Read the staged version
artifact = await artifact_manager.read(artifact_id="example-dataset", version="stage")

```

---

### `list(parent_id: str=None, keywords: List[str] = None, filters: dict = None, mode: str = "AND", offset: int = 0, limit: int = 100, order_by: str = None, silent: bool = False, stage: bool = False) -> list`

Retrieve a list of child artifacts within a specified collection, supporting keyword-based fuzzy search, field-specific filters, and flexible ordering. This function allows detailed control over the search and pagination of artifacts in a collection, including staged artifacts if specified.

**Parameters:**

- `artifact_id` (str): The id of the parent artifact or collection to list children from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`. If not specified, the function lists all top-level artifacts in the current workspace.
- `keywords` (List[str], optional): A list of search terms used for fuzzy searching across all manifest fields. Each term is searched independently, and results matching any term will be included. For example, `["sample", "dataset"]` returns artifacts containing either "sample" or "dataset" in any field of the manifest.

- `filters` (dict, optional): A dictionary where each key is a manifest field name and each value specifies the match for that field. Filters support both exact and range-based matching, depending on the field. You can filter based on the keys inside the manifest, as well as internal fields like permissions and view/download statistics. Supported internal fields include:
  - **`type`**: Matches the artifact type exactly, e.g., `{"type": "application"}`.
  - **`created_by`**: Matches the exact creator ID, e.g., `{"created_by": "user123"}`.
  - **`created_at`** and **`last_modified`**: Accept a single timestamp (lower bound) or a range for filtering. For example, `{"created_at": [1620000000, 1630000000]}` filters artifacts created between the two timestamps.
  - **`view_count`** and **`download_count`**: Accept a single value or a range for filtering, as with date fields. For example, `{"view_count": [10, 100]}` filters artifacts viewed between 10 and 100 times.
  - **`permissions/<user_id>`**: Searches for artifacts with specific permissions assigned to the given `user_id`.
  - **`version`**: Matches the exact version of the artifact, it only support `"stage"`, `"committed"` or `"*"` (both staged or committed). If `stage` is specified, this filter should align with the `stage` parameter.
  - **`manifest`**: Matches the exact value of the field, e.g., `"manifest": {"name": "example-dataset"}`. These filters also support fuzzy matching if a value contains a wildcard (`*`), e.g., `"manifest": {"name": "dataset*"}`, depending on the SQL backend.
  - **`config`**: Matches the exact value of the field in the config, e.g., `"config": {"collection_schema": {"type": "object"}}`.
- `mode` (str, optional): Defines how multiple conditions (from keywords and filters) are combined. Use `"AND"` to ensure all conditions must match, or `"OR"` to include artifacts meeting any condition. Default is `"AND"`.

- `offset` (int, optional): The number of artifacts to skip before listing results. Default is `0`.

- `limit` (int, optional): The maximum number of artifacts to return. Default is `100`.

- `order_by` (str, optional): The field used to order results. Options include:
  - Built-in fields: `view_count`, `download_count`, `last_modified`, `created_at`, and `id`.
  - Custom JSON fields: `manifest.<field_name>` or `config.<field_name>` (e.g., `manifest.likes`, `config.priority`).
  - Use a suffix `<` or `>` to specify ascending or descending order, respectively (e.g., `view_count<` for ascending, `manifest.likes>` for descending).
  - Default ordering is ascending by id if not specified.

- `silent` (bool, optional): If `True`, prevents incrementing the view count for the parent artifact when listing children. Default is `False`.

- `stage`: Controls which artifacts to return based on their staging status:
  - `True`: Return only staged artifacts
  - `False`: Return only committed artifacts (default)
  - `'all'`: Return both staged and committed artifacts

**Returns:**
A list of artifacts that match the search criteria, each represented by a dictionary containing all the fields.

**Example Usage:**

```python
# Retrieve artifacts in the 'dataset-gallery' collection, filtering by creator and keywords,
# with results ordered by descending view count.
results = await artifact_manager.list(
    artifact_id=collection.id, # Or full alias: "my_workspace/dataset-gallery"
    keywords=["example"],
    filters={"created_by": "user123", "stage": False},
    order_by="view_count>",
    mode="AND",
    offset=0,
    limit=50
)

# Order by custom JSON fields in the manifest
results = await artifact_manager.list(
    artifact_id=collection.id,
    order_by="manifest.likes>",  # Order by likes field in manifest (descending)
    limit=20
)

# Order by custom JSON fields in the config
results = await artifact_manager.list(
    artifact_id=collection.id,
    order_by="config.priority<",  # Order by priority field in config (ascending)
    limit=20
)
```

**Example: Return both staged and committed artifacts:**
```python
# Retrieve all artifacts regardless of staging status
all_artifacts = await artifact_manager.list(
    artifact_id=collection.id,
    stage='all'  # This returns both staged and committed artifacts
)
```

---

### `reset_stats(artifact_id: str) -> None`

Resets the download and view counts for the artifact.

**Parameters:**

- `artifact_id`: The id of the artifact to reset statistics for. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.

**Example:**

```python
await artifact_manager.reset_stats(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.reset_stats(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.reset_stats(artifact_id="other_workspace/example-dataset")
```

---

### `publish(artifact_id: str, to: str = None, metadata: dict = None) -> dict`

Publishes the artifact to a public archive platform such as Zenodo or Zenodo Sandbox. This function uploads all files from the artifact and creates a public record with appropriate metadata.

**Prerequisites:**

1. **Committed Artifact**: The artifact must be committed (not in staging mode) with a valid manifest.
2. **Required Manifest Fields**: The artifact's manifest must contain:
   - `name`: The title of the dataset/artifact
   - `description`: A description of the artifact
3. **Authentication Credentials**: Zenodo access tokens must be configured in the artifact's secrets:
   - `ZENODO_ACCESS_TOKEN`: For publishing to production Zenodo
   - `SANDBOX_ZENODO_ACCESS_TOKEN`: For publishing to Zenodo Sandbox
4. **Permissions**: Requires admin (`*`) permission on the artifact.

**Parameters:**

- `artifact_id`: The id of the artifact to publish. It can be a UUID generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `to`: Optional, the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. This parameter should be the same as the `publish_to` parameter in the `create` function or left empty to use the platform specified in the artifact's config.
- `metadata`: Optional dictionary to override or supplement the automatically generated metadata. This will be merged with the metadata derived from the artifact's manifest.

**Returns:**

- A dictionary containing the published record information from Zenodo, including the DOI and other publication details.

**Automatic Metadata Mapping:**

The function automatically creates Zenodo metadata from the artifact's manifest:

- `title` ← `manifest.name`
- `description` ← `manifest.description`  
- `upload_type` ← `"dataset"` if artifact type is "dataset", otherwise `"other"`
- `creators` ← `manifest.authors` (or defaults to the current user)
- `access_right` ← `"open"`
- `license` ← `manifest.license` (defaults to `"cc-by"`)
- `keywords` ← `manifest.tags`

**File Upload:**

All files in the artifact are automatically uploaded to Zenodo, maintaining the original directory structure.

**Example:**

```python
# Basic publish to Zenodo Sandbox
record = await artifact_manager.publish(
    artifact_id=artifact.id, 
    to="sandbox_zenodo"
)
print(f"Published with DOI: {record['doi']}")

# Publish with custom metadata
custom_metadata = {
    "upload_type": "publication",
    "publication_type": "article", 
    "communities": [{"identifier": "my-community"}]
}
record = await artifact_manager.publish(
    artifact_id="example-dataset", 
    to="sandbox_zenodo",
    metadata=custom_metadata
)

# Publish to production Zenodo (requires ZENODO_ACCESS_TOKEN)
record = await artifact_manager.publish(
    artifact_id="other_workspace/example-dataset", 
    to="zenodo"
)
```

**Setting Up Zenodo Credentials:**

Before publishing, you need to configure the appropriate Zenodo access token:

```python
# For Zenodo Sandbox
await artifact_manager.set_secret(
    artifact_id="example-dataset",
    secret_key="SANDBOX_ZENODO_ACCESS_TOKEN", 
    secret_value="your-sandbox-token"
)

# For production Zenodo  
await artifact_manager.set_secret(
    artifact_id="example-dataset",
    secret_key="ZENODO_ACCESS_TOKEN",
    secret_value="your-production-token" 
)
```

**Notes:**

- After successful publication, the artifact's config is updated with the Zenodo record information.
- The function uploads all files from the artifact, so ensure your artifact contains only the files you want to publish.
- For large artifacts, the publishing process may take some time to upload all files.
- Published records on Zenodo are permanent and cannot be deleted, only new versions can be created.

---

### `get_secret(artifact_id: str, secret_key: str = None) -> str | None`

Retrieves a secret value from an artifact. This operation requires admin permissions.

**Parameters:**

- `artifact_id`: The id of the artifact containing the secret. It can be a UUID generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `secret_key`: The key name of the secret to retrieve.
  - If `None`, returns all secrets in the artifact.

**Returns:**

- The secret value as a string if it exists, or `None` if the secret key does not exist.

**Permissions:**

- Requires admin (`*`) permission on the artifact.

**Example:**

```python
# Get a secret value from an artifact
api_key = await artifact_manager.get_secret(
    artifact_id="example-dataset",
    secret_key="API_KEY"
)

if api_key:
    print("Retrieved API key")
else:
    print("API key not found")
```

---
### `set_secret(artifact_id: str, secret_key: str, secret_value: str) -> None`

Sets or removes a secret value for an artifact. This operation requires read_write permissions.

**Parameters:**

- `artifact_id`: The id of the artifact to store the secret in. It can be a UUID generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `secret_key`: The key name for the secret.
- `secret_value`: The secret value to store. Can be a string or `None`. If set to `None`, the secret key will be removed from the artifact.

**Permissions:**

- Requires read_write (`rw` or higher) permission on the artifact.

**Example:**

```python
# Set a secret value for an artifact
await artifact_manager.set_secret(
    artifact_id="example-dataset",
    secret_key="API_KEY",
    secret_value="your-secret-api-key"
)

# Update an existing secret
await artifact_manager.set_secret(
    artifact_id="example-dataset",
    secret_key="API_KEY",
    secret_value="updated-secret-api-key"
)

# Set a secret to None (effectively removing it)
await artifact_manager.set_secret(
    artifact_id="example-dataset",
    secret_key="API_KEY",
    secret_value=None
)
```

---

## Example Usage of API Functions

Here's a simple sequence of API function calls to manage a dataset gallery:

```python
# Step 1: Connect to the API
server = await connect_to_server({"name": "test-client", "server_url": "https://hypha.aicell.io"})
artifact_manager = await server.get_service("public/artifact-manager")

# Step 2: Create a gallery collection
gallery_manifest = {
    "name": "Dataset Gallery",
    "description": "A collection for organizing datasets",
}
# Create the collection with read permission for everyone and create permission for all authenticated users
collection = await artifact_manager.create(type="collection", alias="dataset-gallery", manifest=gallery_manifest, permissions={"*": "r", "@": "r+"})

# Step 3: Add a dataset to the gallery
dataset_manifest = {
    "name": "Example Dataset",
    "description": "A dataset with example data",
    "type": "dataset",
}
dataset = await artifact_manager.create(parent_id=collection.id, alias="example-dataset" manifest=dataset_manifest, stage=True)

# Step 4: Upload a file to the dataset
# Note: Adding files does not automatically create new versions
put_url = await artifact_manager.put_file(dataset.id, file_path="data.csv")
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

# Step 5: Commit the dataset
# This creates v0 since no version="new" was specified
await artifact_manager.commit(dataset.id)

# Step 6: List all datasets in the gallery
datasets = await artifact_manager.list(collection.id)
print("Datasets in the gallery:", datasets)
```

## HTTP API for Accessing Artifacts and Download Counts

The `Artifact Manager` provides an HTTP API for retrieving artifact manifests, data, file statistics, and managing zip files. These endpoints are designed for public-facing web applications that need to interact with datasets, models, or applications.

---
### Artifact Metadata and File Access Endpoints

#### Endpoints:

- `/{workspace}/artifacts/`: List all top-level artifacts in the workspace.
  - **Query Parameters**:
    - `keywords`: (Optional) Comma-separated search terms
    - `filters`: (Optional) JSON-encoded filter conditions
    - `offset`: (Optional) Number of results to skip
    - `limit`: (Optional) Maximum number of results to return
    - `order_by`: (Optional) Field to sort results by
    - `mode`: (Optional) How to combine conditions ("AND" or "OR")
    - `pagination`: (Optional) Whether to include pagination metadata
    - `stage`: (Optional) Filter by staging status:
      - `true`: Return only staged artifacts
      - `false`: Return only committed artifacts (default)
      - `all`: Return both staged and committed artifacts
    - `no_cache`: (Optional) Force refresh of cached results
- `/{workspace}/artifacts/{artifact_alias}`: Fetch the artifact manifest.
- `/{workspace}/artifacts/{artifact_alias}/children`: List all artifacts in a collection.
  - **Query Parameters**:
    - `keywords`: (Optional) Comma-separated search terms
    - `filters`: (Optional) JSON-encoded filter conditions
    - `offset`: (Optional) Number of results to skip
    - `limit`: (Optional) Maximum number of results to return
    - `order_by`: (Optional) Field to sort results by
    - `mode`: (Optional) How to combine conditions ("AND" or "OR")
    - `pagination`: (Optional) Whether to include pagination metadata
    - `silent`: (Optional) Whether to suppress view count increment
    - `stage`: (Optional) Filter by staging status:
      - `true`: Return only staged artifacts
      - `false`: Return only committed artifacts (default)
      - `all`: Return both staged and committed artifacts
- `/{workspace}/artifacts/{artifact_alias}/files`: List all files in the artifact.
- `/{workspace}/artifacts/{artifact_alias}/files/{file_path:path}`: Download a file from the artifact (redirects to a pre-signed URL).

#### Request Format:

- **Method**: `GET`
- **Headers**: 
  - `Authorization`: Optional. The user's token for accessing private artifacts (obtained via login logic or created by `api.generate_token()`). Not required for public artifacts.

#### Path Parameters:

- **workspace**: The workspace in which the artifact is stored.
- **artifact_alias**: The alias or ID of the artifact to access. This can be generated by `create` or `edit` functions or be an alias under the current workspace.
- **file_path**: (Optional) The relative path to a file within the artifact.

#### Response Examples:

- **Artifact Manifest**: 
  ```json
  {
      "manifest": {
          "name": "Example Dataset",
          "description": "A dataset for testing.",
          "version": "1.0.0"
      },
      "view_count": 150,
      "download_count": 25
  }
  ```

- **Files in Artifact**:
  ```json
  [
      {"name": "example.txt", "type": "file"},
      {"name": "nested", "type": "directory"}
  ]
  ```

- **Download File**: A redirect to a pre-signed URL for the file.

#### Authentication for Workspace-Level Endpoint:

The `/{workspace}/artifacts/` endpoint requires authentication to access private artifacts:

- **Public artifacts**: Can be accessed without authentication
- **Private artifacts**: Require an `Authorization: Bearer <token>` header
- **Token generation**: Use `api.generate_token()` to create a token for authenticated requests

**Example with Authentication**:
```python
import requests

# Get a token for authenticated requests
token = await api.generate_token()

# Access workspace artifacts with authentication
response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/",
    headers={"Authorization": f"Bearer {token}"}
)
```

---

### Dynamic Zip File Creation Endpoint

#### Endpoint:

- `/{workspace}/artifacts/{artifact_alias}/create-zip-file`: Stream a dynamically created zip file containing selected or all files in the artifact.

#### Request Format:

- **Method**: `GET`
- **Query Parameters**: 
  - **file**: (Optional) A list of files to include in the zip file. If omitted, all files in the artifact are included.
  - **token**: (Optional) User token for private artifact access.
  - **version**: (Optional) The version of the artifact to fetch files from.

#### Response:

- Streams the zip file back to the client.
- **Headers**:
  - `Content-Disposition`: Attachment with the artifact alias as the filename.

#### Download Count Behavior:

When using the `create-zip-file` endpoint, the artifact's download count is incremented based on the files included in the zip:

- **Per-file increment**: Each file included in the zip contributes to the download count based on its `download_weight`
- **Zero-weight files**: Files with `download_weight=0` are included in the zip but do not increment the download count
- **Multiple requests**: Each request to the endpoint increments the download count independently
- **All files**: When no specific files are requested (downloading all files), the download count is incremented by the sum of all file weights

**Example**: If you create a zip with 3 files having weights [0.5, 1.0, 0], the download count will increase by 1.5 (0.5 + 1.0 + 0).

**Example Usage:

```python
import requests

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
artifact_alias = "example-dataset"
files = ["example.txt", "nested/example2.txt"]

response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-zip-file",
    params={"file": files},
    stream=True,
)
if response.status_code == 200:
    with open("artifact_files.zip", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Zip file created successfully.")
else:
    print(f"Error: {response.status_code}")
```

---

### Zip File Access Endpoints

These endpoints allow direct access to zip file contents stored in the artifact without requiring the entire zip file to be downloaded or extracted.

#### Endpoints:

1. **`/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path:path}?path=...`**  
   - Access the contents of a zip file, specifying the path within the zip file using a query parameter (`?path=`).

2. **`/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path:path}/~/{path:path|}`**  
   - Access the contents of a zip file, separating the zip file path and the internal path using `/~/`.

---

#### Endpoint 1: `/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path:path}?path=...`

##### Functionality:

- **If `path` ends with `/`:** Lists the contents of the directory specified by `path` inside the zip file.
- **If `path` specifies a file:** Streams the file content from the zip.

##### Request Format:

- **Method**: `GET`
- **Path Parameters**:
  - **workspace**: The workspace in which the artifact is stored.
  - **artifact_alias**: The alias or ID of the artifact to access.
  - **zip_file_path**: Path to the zip file within the artifact.
- **Query Parameters**:
  - **path**: (Optional) The relative path inside the zip file. Defaults to the root directory.

##### Response Examples:

1. **Listing Directory Contents**:
    ```json
    [
        {"type": "file", "name": "example.txt", "size": 123, "last_modified": 1732363845.0},
        {"type": "directory", "name": "nested"}
    ]
    ```

2. **Fetching a File**:
    Streams the file content from the zip.

---

#### Endpoint 2: `/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path:path}/~/{path:path}`

##### Functionality:

- **If `path` ends with `/`:** Lists the contents of the directory specified by `path` inside the zip file.
- **If `path` specifies a file:** Streams the file content from the zip.

##### Request Format:

- **Method**: `GET`
- **Path Parameters**:
  - **workspace**: The workspace in which the artifact is stored.
  - **artifact_alias**: The alias or ID of the artifact to access.
  - **zip_file_path**: Path to the zip file within the artifact.
  - **path**: (Optional) The relative path inside the zip file. Defaults to the root directory.

##### Response Examples:

1. **Listing Directory Contents**:
    ```json
    [
        {"type": "file", "name": "example.txt", "size": 123, "last_modified": 1732363845.0},
        {"type": "directory", "name": "nested"}
    ]
    ```

2. **Fetching a File**:
    Streams the file content from the zip.

---

#### Example Usage for Both Endpoints

##### Listing Directory Contents:

```python
import requests

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
artifact_alias = "example-dataset"
zip_file_path = "example.zip"

# Using the query parameter method
response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}",
    params={"path": "nested/"}
)
print(response.json())

# Using the tilde method
response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}/~/nested/"
)
print(response.json())
```

##### Fetching a File:

```python
# Using the query parameter method
response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}",
    params={"path": "nested/example2.txt"},
    stream=True,
)
with open("example2.txt", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Using the tilde method
response = requests.get(
    f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/zip-files/{zip_file_path}/~/nested/example2.txt",
    stream=True,
)
with open("example2.txt", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
```

### Version Handling Rules

The Artifact Manager follows specific rules for version handling during `create`, `edit`, and `commit` operations:

1. **Explicit Version Control**
   - **No Automatic Versioning:** Adding files to staging does NOT automatically create new version intent
   - **Explicit Intent Required:** Users must explicitly specify `version="new"` during edit operations to create new versions
   - **Predictable Behavior:** Version creation is always under explicit user control, preventing unexpected version proliferation
   - **Cost Optimization:** Eliminates costly file copying operations in S3 by requiring deliberate version decisions

2. **Version Parameter Validation**
   - The `version` parameter in `edit()` is strictly validated
   - Valid values are: `None`, `"new"`, `"stage"`, or existing version names from the artifact's versions list
   - Custom version names cannot be specified during `edit()` - they can only be provided during `commit()`
   - This prevents confusion and ensures predictable version management

3. **File Placement Optimization**
   - **Direct Placement:** Files are uploaded directly to their final destination based on version intent
   - **New Version Intent:** Files uploaded when `new_version` intent exists go to the new version location
   - **Existing Version:** Files uploaded without new version intent go to the existing latest version location
   - **No File Copying:** Commit operations perform no file copying, making them fast and efficient
   - **S3 Cost Efficiency:** Eliminates expensive duplicate file storage during staging

4. **Staging Intent System**
   - When `stage=True` and `version="new"` are used together in `edit()`, the system stores an "intent" to create a new version
   - This intent is preserved until commit, allowing custom version names to be specified at commit time
   - The intent system separates the decision of "what to do" (edit vs create) from "what to name it"
   - **Important:** The intent is only set when explicitly requested, never automatically

5. **Version Determination at Creation Time**
   - When creating an artifact, the version handling must be determined at creation time
   - If `stage=True` is specified, any version parameter is ignored and the artifact starts in staging mode
   - Using `stage=True` is equivalent to `version="stage"` (not recommended) for backward compatibility
   - When committing a staged artifact, it will create version "v0" regardless of any version specified during creation

6. **Version Determination at Edit Time**
   - When editing an artifact, the version handling must be determined at edit time
   - If `stage=True` is specified, any version parameter is ignored and the artifact goes into staging mode
   - Direct edits without staging will update the specified version or the latest version if none is specified
   - Custom version names are no longer allowed during edit - only during commit for staged artifacts

7. **Version Handling During Commit**
   - The commit operation is now a fast, metadata-only operation
   - Whether we're editing an existing version or creating a new one is determined by the staging intent
   - **New version creation**: When committing from staging with "new version intent", custom version names are allowed
   - **Existing version update**: When committing updates to existing versions, the version parameter is ignored
   - Version names must be unique - the system validates that specified versions don't already exist

**Example: Version Handling in Practice**

```python
# Create an artifact in staging mode
artifact = await artifact_manager.create(
    type="dataset",
    manifest=manifest,
    stage=True  # This puts the artifact in staging mode
)

# Add files to staging - this does NOT automatically create new version intent
put_url = await artifact_manager.put_file(artifact_id=artifact.id, file_path="data.csv")
response = requests.put(put_url, data="some data")

# Commit without new version intent - creates v0 (initial version)
committed = await artifact_manager.commit(artifact_id=artifact.id)
# Result: v0 created with the uploaded file

# Edit with explicit intent to create new version during commit
await artifact_manager.edit(
    artifact_id=artifact.id,
    manifest=updated_manifest,
    stage=True,
    version="new"  # EXPLICIT intent for new version creation
)

# Add more files - these will go to the new version location
put_url = await artifact_manager.put_file(artifact_id=artifact.id, file_path="new_data.csv")
response = requests.put(put_url, data="new data")

# Commit with custom version name (allowed due to explicit new version intent)
committed = await artifact_manager.commit(
    artifact_id=artifact.id,
    version="v2.0-beta"  # Custom version name allowed here
)

# Edit without new version intent - adds files to existing version
await artifact_manager.edit(
    artifact_id=artifact.id,
    manifest={"name": "Updated Name"},
    stage=True  # No version="new" specified
)

# Add file to existing version
put_url = await artifact_manager.put_file(artifact_id=artifact.id, file_path="update.csv")
response = requests.put(put_url, data="update data")

# Commit updates existing version (v2.0-beta), does not create new version
committed = await artifact_manager.commit(artifact_id=artifact.id)
```

**Key Benefits of the New System:**
- **Explicit Control**: Version creation only happens when explicitly requested with `version="new"`
- **Cost Efficient**: No file copying during commit - files are placed directly in final destinations
- **Predictable**: No surprise version creation from simply adding files
- **Performance**: Fast, metadata-only commit operations
- **S3 Optimized**: Eliminates expensive file duplication in cloud storage
