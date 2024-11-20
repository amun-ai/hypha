
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
    version="stage"
)
print("Dataset added to the gallery.")
```

**Tips: The `version="stage"` parameter stages the dataset for review and commit. You can edit the dataset before committing it to the collection.**

### Step 4: Uploading Files to the Dataset with Download Statistics

Each dataset can contain multiple files. Use pre-signed URLs for secure uploads and set `download_weight` to track file downloads.

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

After uploading files, commit the dataset to finalize its status in the collection. Committed datasets are accessible based on their permissions.

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

Here’s a complete example showing how to connect to the service, create a dataset gallery, add a dataset, upload files, and commit the dataset.

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
        version="stage"
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
    version="stage"
)
print("Valid dataset created.")

# Commit the dataset (will pass schema validation)
await artifact_manager.commit(dataset.id)
print("Valid dataset committed.")
```


---

## API References

### `create(parent_id: str, alias: str, type: str, manifest: dict, permissions: dict=None, config: dict=None, version: str = None, comment: str = None, overwrite: bool = False, publish_to: str = None) -> None`

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
- `type`: The type of the artifact. Supported values are `collection`, `generic` and any other custom type. By default, it's set to `generic` which contains fields tailored for displaying the artifact as cards on a webpage.
- `manifest`: The manifest of the new artifact. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `config`: Optional. A dictionary containing additional configuration options for the artifact (shared for both staged and committed). For collections, the config can contain the following special fields:
  - `collection_schema`: Optional. A JSON schema that defines the structure of child artifacts in the collection. This schema is used to validate child artifacts when they are created or edited. If a child artifact does not conform to the schema, the creation or edit operation will fail.
  - `id_parts`: Optional. A dictionary of id name parts to be used in generating the id for child artifacts. For example: `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`. This can be used for creating child artifacts with auto-generated ids based on the id parts. For example, when calling `create`, you can specify the alias as `my-pet-{colors}-{animals}`, and the id will be generated based on the id parts, e.g., `my-pet-red-dog`.
  - `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
  - `list_fields`: Optional. A list of fields to be collected when calling ``list`` function. By default, it collects all fields in the artifacts. If you want to collect only specific fields, you can set this field to a list of field names, e.g. `["manifest", "download_count"]`.
- `version`: Optional. The version of the artifact to create. By default, it set to None or `"new"`, it will generate a version `v0`. If you want to create a staged version, you can set it to `"stage"`.
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
- `publish_to`: Optional. A string specifying the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. If set, the artifact will be published to the specified platform. The artifact must have a valid Zenodo metadata schema to be published.

**Note 1: If you set `version="stage"`, you must call `commit()` to finalize the artifact.**

**Example:**

```python
# Assuming we have already created a dataset-gallery collection, we can add a new dataset to it
await artifact_manager.create(artifact_id="dataset-gallery", alias="example-dataset", manifest=dataset_manifest, version="stage")
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
- **r**: Read-only access (includes `read`, `get_file`, `list_files`, `list`, `search_by_vector`, `search_by_text`, and `get_vector`).
- **r+**: Read, write, and create access (includes `read`, `get_file`, `put_file`, `list_files`, `list`, `search_by_vector`, `search_by_text`, `get_vector`, `create`, `commit`, `add_vectors`, and `add_documents`).
- **rw**: Read, write, and create access with file management (includes `read`, `get_file`, `get_vector`, `search_by_vector`, `search_by_text`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, and `remove_vectors`).
- **rw+**: Read, write, create, and manage access (includes `read`, `get_file`, `get_vector`, `search_by_vector`, `search_by_text`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, `remove_vectors`, and `create`).
- **\***: Full access to all operations (includes `read`, `get_file`, `get_vector`, `search_by_vector`, `search_by_text`, `list_files`, `list_vectors`, `list`, `edit`, `commit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, `remove_vectors`, `create`, and `reset_stats`).

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

### `edit(artifact_id: str, manifest: dict = None, type: str = None,  permissions: dict = None, config: dict = None, secrets: dict = None, version: str = None, comment: str = None) -> None`

Edits an existing artifact's manifest. The new manifest is staged until committed.

**Parameters:**

- `artifact_id`: The id of the artifact to edit. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `manifest`: The updated manifest. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `type`: Optional. The type of the artifact. Supported values are `collection`, `generic` and any other custom type. By default, it's set to `generic` which contains fields tailored for displaying the artifact as cards on a webpage.
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
- `secrets`: Optional. A dictionary containing secrets to be stored with the artifact. Secrets are encrypted and can only be accessed by the artifact owner or users with appropriate permissions. See the `create` function for a list of supported secrets.
- `config`: Optional. A dictionary containing additional configuration options for the artifact.
- `version`: Optional. The version of the artifact to edit. By default, it set to None, the version will stay the same. If you want to create a staged version, you can set it to `"stage"`. You can set it to any version in text, e.g. `0.1.0` or `v1`. If you set it to `new`, it will generate a version similar to `v0`, `v1`, etc.
- `comment`: Optional. A comment to describe the changes made to the artifact.

**Example:**

```python
await artifact_manager.edit(artifact_id="example-dataset", manifest=updated_manifest)
```

---

### `commit(artifact_id: str, version: str = None, comment: str = None) -> None`

Finalizes and commits an artifact's staged changes. Validates uploaded files and commit the staged manifest. This process also updates view and download statistics.

**Parameters:**

- `artifact_id`: The id of the artifact to commit. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `version`: Optional. The version of the artifact to edit. By default, it set to None, the version will stay the same. If you want to create a staged version, you can set it to `"stage"`. You can set it to any version in text, e.g. `0.1.0` or `v1`. If you set it to `new`, it will generate a version similar to `v0`, `v1`, etc.
- `comment`: Optional. A comment to describe the changes made to the artifact.

**Example:**

```python
await artifact_manager.commit(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.commit(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.commit(artifact_id="other_workspace/example-dataset")
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

### `put_file(artifact_id: str, file_path: str, download_weight: int = 0) -> str`

Generates a pre-signed URL to upload a file to the artifact in S3. The URL can be used with an HTTP `PUT` request to upload the file. The file is staged until the artifact is committed.

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"data.csv"`).
- `download_weight`: A float value representing the file's impact on download count (0-1). Defaults to `None`.

**Returns:** A pre-signed URL for uploading the file.

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

### `get_file(artifact_id: str, path: str, silent: bool = False) -> str`

Generates a pre-signed URL to download a file from the artifact stored in S3.

**Parameters:**

- `artifact_id`: The id of the artifact to download the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `path`: The relative path of the file to download (e.g., `"data.csv"`).
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

### `read(artifact_id: str, stage: bool = False, silent: bool = False, include_metadata: bool = False) -> dict`

Reads and returns the manifest of an artifact or collection. If in staging mode, reads the staged manifest.

**Parameters:**

- `artifact_id`: The id of the artifact to read. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `version`: Optional. The version of the artifact to read. By default, it reads the latest version. If you want to read a staged version, you can set it to `"stage"`.
- `silent`: Optional. If `True`, suppresses the view count increment. Default is `False`.
- `include_metadata`: Optional. If `True`, includes metadata such as download statistics in the manifest (fields starting with `"."`). Default is `False`.

**Returns:** The manifest as a dictionary, including view/download statistics and collection details (if applicable).

**Example:**

```python
manifest = await artifact_manager.read(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
manifest = await artifact_manager.read(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
manifest = await artifact_manager.read(artifact_id="other_workspace/example-dataset")
```

---

### `list(artifact_id: str=None, keywords: List[str] = None, filters: dict = None, mode: str = "AND", offset: int = 0, limit: int = 100, order_by: str = None, silent: bool = False) -> list`

Retrieve a list of child artifacts within a specified collection, supporting keyword-based fuzzy search, field-specific filters, and flexible ordering. This function allows detailed control over the search and pagination of artifacts in a collection, including staged artifacts if specified.

**Parameters:**

- `artifact_id` (str): The id of the parent artifact or collection to list children from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`. If not specified, the function lists all top-level artifacts in the current workspace.
- `keywords` (List[str], optional): A list of search terms used for fuzzy searching across all manifest fields. Each term is searched independently, and results matching any term will be included. For example, `["sample", "dataset"]` returns artifacts containing either "sample" or "dataset" in any field of the manifest.

- `filters` (dict, optional): A dictionary where each key is a manifest field name and each value specifies the match for that field. Filters support both exact and range-based matching, depending on the field. You can filter based on the keys inside the manifest, as well as internal fields like permissions and view/download statistics by adding a dot (`.`) before the field name. Supported internal fields include:
  - **`.type`**: Matches the artifact type exactly, e.g., `{"type": "application"}`.
  - **`.created_by`**: Matches the exact creator ID, e.g., `{"created_by": "user123"}`.
  - **`.created_at`** and **`last_modified`**: Accept a single timestamp (lower bound) or a range for filtering. For example, `{"created_at": [1620000000, 1630000000]}` filters artifacts created between the two timestamps.
  - **`.view_count`** and **`download_count`**: Accept a single value or a range for filtering, as with date fields. For example, `{"view_count": [10, 100]}` filters artifacts viewed between 10 and 100 times.
  - **`.permissions/<user_id>`**: Searches for artifacts with specific permissions assigned to the given `user_id`.
  - **`.stage`**: If `true`, returns only artifacts in staging mode. Example: `{"stage": true}`. By default, .stage filter is always set to `false`.
  - **`any other field in manifest`**: Matches the exact value of the field, e.g., `{"name": "example-dataset"}`. These filters also support fuzzy matching if a value contains a wildcard (`*`), e.g., `{"name": "dataset*"}`, depending on the SQL backend.

- `mode` (str, optional): Defines how multiple conditions (from keywords and filters) are combined. Use `"AND"` to ensure all conditions must match, or `"OR"` to include artifacts meeting any condition. Default is `"AND"`.

- `offset` (int, optional): The number of artifacts to skip before listing results. Default is `0`.

- `limit` (int, optional): The maximum number of artifacts to return. Default is `100`.

- `order_by` (str, optional): The field used to order results. Options include:
  - `view_count`, `download_count`, `last_modified`, `created_at`, and `id`.
  - Use a suffix `<` or `>` to specify ascending or descending order, respectively (e.g., `view_count<` for ascending).
  - Default ordering is ascending by id if not specified.

- `silent` (bool, optional): If `True`, prevents incrementing the view count for the parent artifact when listing children. Default is `False`.

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

### `publish(artifact_id: str, to: str = None) -> None`

Publishes the artifact to a specified platform, such as Zenodo. The artifact must have a valid Zenodo metadata schema to be published.

**Parameters:**

- `artifact_id`: The id of the artifact to publish. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `to`: Optional, the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. This parameter should be the same as the `publish_to` parameter in the `create` function or left empty to use the platform specified in the artifact's metadata.

**Example:**

```python
await artifact_manager.publish(artifact_id=artifact.id, to="sandbox_zenodo")

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.publish(artifact_id="example-dataset", to="sandbox_zenodo")

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.publish(artifact_id="other_workspace/example-dataset", to="sandbox_zenodo")
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
dataset = await artifact_manager.create(parent_id=collection.id, alias="example-dataset" manifest=dataset_manifest, version="stage")

# Step 4: Upload a file to the dataset
put_url = await artifact_manager.put_file(dataset.id, file_path="data.csv")
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

# Step 5: Commit the dataset
await artifact_manager.commit(dataset.id)

# Step 6: List all datasets in the gallery
datasets = await artifact_manager.list(collection.id)
print("Datasets in the gallery:", datasets)
```


## HTTP API for Accessing Artifacts and Download Counts

The `Artifact Manager` provides an HTTP endpoint for retrieving artifact manifests, data, and download statistics. This is useful for public-facing web applications that need to access datasets, models, or applications.

### Endpoints:

 - `/{workspace}/artifacts/{artifact_alias}` for fetching the artifact manifest.
 - `/{workspace}/artifacts/{artifact_alias}/children` for listing all artifacts in a collection.
 - `/{workspace}/artifacts/{artifact_alias}/files` for listing all files in the artifact.
 - `/{workspace}/artifacts/{artifact_alias}/files/{file_path:path}` for downloading a file from the artifact (will be redirected to a pre-signed URL).


### Request Format:

- **Method**: `GET`
- **Headers**:
  - `Authorization`: Optional. The user's token for accessing private artifacts (obtained via the login logic or created by `api.generate_token()`). Not required for public artifacts.

### Path Parameters:

The path parameters are used to specify the artifact or file to access. The following parameters are supported:

- **workspace**: The workspace in which the artifact is stored.
- **artifact_alias**: The alias or id of the artifact to access. This can be an artifact id generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. Note that this artifact_alias should not contain the workspace.
- **file_path**: Optional, the relative path to a file within the artifact. This is optional and only required when downloading a file.

### Query Parameters:

Qury parameters are passed after the `?` in the URL and are used to control the behavior of the API. The following query parameters are supported:

- **stage**: A boolean flag to fetch the staged version of the manifest. Default is `False`.
- **silent**: A boolean flag to suppress the view count increment. Default is `False`.

- **keywords**: A list of search terms used for fuzzy searching across all manifest fields, separated by commas.
- **filters**: A dictionary of filters to apply to the search, in the format of a JSON string.
- **mode**: The mode for combining multiple conditions. Default is `AND`.
- **offset**: The number of artifacts to skip before listing results. Default is `0`.
- **limit**: The maximum number of artifacts to return. Default is `100`.
- **order_by**: The field used to order results. Default is ascending by id.
- **silent**: A boolean flag to prevent incrementing the view count for the parent artifact when listing children, listing files, or reading the artifact. Default is `False`.

### Response:

For `/{workspace}/artifacts/{artifact_alias}`, the response will be a JSON object representing the artifact manifest. For `/{workspace}/artifacts/{artifact_alias}/__files__/{file_path:path}`, the response will be a pre-signed URL to download the file. The artifact manifest will also include any metadata such as download statistics, e.g. `view_count`, `download_count`. For private artifacts, make sure if the user has the necessary permissions.

For `/{workspace}/artifacts/{artifact_alias}/children`, the response will be a list of artifacts in the collection.

For `/{workspace}/artifacts/{artifact_alias}/files`, the response will be a list of files in the artifact, each file is a dictionary with the `name` and `type` fields.

For `/{workspace}/artifacts/{artifact_alias}/files/{file_path:path}`, the response will be a pre-signed URL to download the file.

### Example: Fetching a public artifact with download statistics

```python
import requests

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
response = requests.get(f"{SERVER_URL}/{workspace}/artifacts/example-dataset")
if response.ok:
    artifact = response.json()
    print(artifact["manifest"]["name"])  # Output: Example Dataset
    print(artifact["download_count"])  # Output: Download count for the dataset
else:
    print(f"Error: {response.status_code}")
```
