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

### Step 7: Reading Dataset Information and Files

After the dataset is committed, you can read its metadata and access its files.

```python
# Read the dataset metadata
dataset_info = await artifact_manager.read(dataset.id)
print("Dataset info:", dataset_info["manifest"])
print("View count:", dataset_info.get("view_count", 0))
print("Download count:", dataset_info.get("download_count", 0))

# List all files in the dataset
files = await artifact_manager.list_files(dataset.id)
print("Files in the dataset:", files)

# Get a download URL for a specific file
download_url = await artifact_manager.get_file(dataset.id, file_path="data.csv")
print("Download URL:", download_url)

# Download the file content (example using requests)
import requests
response = requests.get(download_url)
if response.ok:
    print("File content preview:", response.text[:100] + "...")
else:
    print("Failed to download file")
```

### Step 8: Updating the Dataset

You can update the dataset by editing its manifest or adding/removing files.

```python
# Edit the dataset manifest
updated_manifest = {
    "name": "Updated Example Dataset",
    "description": "An updated dataset with additional information",
    "tags": ["example", "updated", "dataset"]
}

# Stage the changes
await artifact_manager.edit(
    artifact_id=dataset.id,
    manifest=updated_manifest,
    stage=True
)

# Add a new file to the staged dataset
put_url = await artifact_manager.put_file(dataset.id, file_path="metadata.json", download_weight=0.1)
metadata_content = '{"version": "1.1", "updated": "2024-01-01"}'
response = requests.put(put_url, data=metadata_content.encode())
assert response.ok, "Metadata upload failed"

# Commit the changes (this updates the existing version)
await artifact_manager.commit(dataset.id)
print("Dataset updated successfully")
```

### Step 9: Creating a New Version

To create a new version of the dataset, you must explicitly specify the intent during editing.

```python
# Edit with explicit intent to create new version
await artifact_manager.edit(
    artifact_id=dataset.id,
    manifest={
        "name": "Example Dataset v2",
        "description": "Version 2 with new features",
    },
    stage=True,
    version="new"  # Explicit intent for new version creation
)

# Add files to the new version
put_url = await artifact_manager.put_file(dataset.id, file_path="data_v2.csv", download_weight=1.0)
with open("path/to/local/data_v2.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

# Commit with custom version name
await artifact_manager.commit(dataset.id)
print("New version v2 created")

# Read specific version
v1_data = await artifact_manager.read(dataset.id, version="v0")
v2_data = await artifact_manager.read(dataset.id, version="v2")
print("Version v0 name:", v1_data["manifest"]["name"])
print("Version v2 name:", v2_data["manifest"]["name"])
```

### Step 10: File Management Operations

Perform various file operations like removing files and managing file access.

```python
# Remove a file from the dataset (requires staging)
await artifact_manager.edit(dataset.id, stage=True)  # Enter staging mode
await artifact_manager.remove_file(dataset.id, file_path="old_file.csv")
await artifact_manager.commit(dataset.id)  # Commit the file removal

# List files from a specific version
v1_files = await artifact_manager.list_files(dataset.id, version="v0")
v2_files = await artifact_manager.list_files(dataset.id, version="v2")
print("Files in v0:", [f["name"] for f in v1_files])
print("Files in v2:", [f["name"] for f in v2_files])

# Get file with specific options
download_url = await artifact_manager.get_file(
    dataset.id, 
    file_path="data_v2.csv",
    version="v2",
    silent=True,  # Don't increment download count
    use_proxy=False  # Get direct S3 URL
)
print("Direct S3 URL:", download_url)
```

### Step 11: Cleanup (Optional)

If needed, you can delete files, datasets, or entire collections.

```python
# Delete a specific version
await artifact_manager.delete(dataset.id, version="v0", delete_files=True)
print("Version v0 deleted")

# Delete the entire dataset (with all files)
await artifact_manager.delete(dataset.id, delete_files=True)
print("Dataset deleted")

# Delete the collection (requires recursive=True for collections with children)
await artifact_manager.delete(collection.id, delete_files=True, recursive=True)
print("Collection deleted")
```

---

## Full Example: Creating and Managing a Dataset Gallery

Here's a complete example showing how to connect to the service, create a dataset gallery, add a dataset, upload files, access data, manage versions, and clean up.

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
        "tags": ["example", "research"]
    }
    dataset = await artifact_manager.create(
        parent_id=collection.id,
        alias="example-dataset",
        manifest=dataset_manifest,
        stage=True
    )
    print("Dataset added to the gallery.")

    # Upload multiple files to the dataset
    files_to_upload = [
        {"path": "data.csv", "content": "name,age,city\nJohn,25,NYC\nJane,30,LA", "weight": 1.0},
        {"path": "metadata.json", "content": '{"version": "1.0", "rows": 2}', "weight": 0.1},
        {"path": "readme.txt", "content": "This is a sample dataset", "weight": 0.0}
    ]
    
    for file_info in files_to_upload:
        put_url = await artifact_manager.put_file(
            dataset.id, 
            file_path=file_info["path"], 
            download_weight=file_info["weight"]
        )
        response = requests.put(put_url, data=file_info["content"].encode())
        assert response.ok, f"Upload failed for {file_info['path']}"
        print(f"Uploaded: {file_info['path']}")

    # Commit the dataset
    await artifact_manager.commit(dataset.id)
    print("Dataset committed as v0.")

    # Read dataset information
    dataset_info = await artifact_manager.read(dataset.id)
    print(f"Dataset name: {dataset_info['manifest']['name']}")
    print(f"View count: {dataset_info.get('view_count', 0)}")
    print(f"Download count: {dataset_info.get('download_count', 0)}")

    # List all files in the dataset
    files = await artifact_manager.list_files(dataset.id)
    print("Files in dataset:")
    for file in files:
        print(f"  - {file['name']} ({file.get('size', 'unknown')} bytes)")

    # Download and read specific files
    # Download the main data file
    download_url = await artifact_manager.get_file(dataset.id, file_path="data.csv")
    response = requests.get(download_url)
    if response.ok:
        print("Main data file content:")
        print(response.text)
    else:
        print("Failed to download main data file")

    # Download metadata without incrementing download count
    metadata_url = await artifact_manager.get_file(
        dataset.id, 
        file_path="metadata.json", 
        silent=True
    )
    response = requests.get(metadata_url)
    if response.ok:
        print("Metadata content:")
        print(response.text)

    # Update the dataset
    updated_manifest = {
        "name": "Updated Example Dataset",
        "description": "A dataset containing example data with updates",
        "tags": ["example", "research", "updated"]
    }
    
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest=updated_manifest,
        stage=True
    )
    
    # Add a new file to the update
    put_url = await artifact_manager.put_file(dataset.id, file_path="changelog.txt", download_weight=0.1)
    changelog_content = "v1.0: Added more data and metadata"
    response = requests.put(put_url, data=changelog_content.encode())
    assert response.ok, "Changelog upload failed"
    
    # Commit the update (updates existing v0)
    await artifact_manager.commit(dataset.id)
    print("Dataset updated successfully")

    # Create a new version
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={
            "name": "Example Dataset v2",
            "description": "Major update with new analysis",
            "tags": ["example", "research", "v2"]
        },
        stage=True,
        version="new"  # Explicit intent to create new version
    )
    
    # Add files specific to v2
    put_url = await artifact_manager.put_file(dataset.id, file_path="analysis.py", download_weight=0.5)
    analysis_content = "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.describe())"
    response = requests.put(put_url, data=analysis_content.encode())
    assert response.ok, "Analysis script upload failed"
    
    # Commit with custom version name
    await artifact_manager.commit(dataset.id, version="v2.0")
    print("New version v2.0 created")

    # Compare versions
    v0_info = await artifact_manager.read(dataset.id, version="v0")
    v2_info = await artifact_manager.read(dataset.id, version="v2.0")
    
    print(f"v0 name: {v0_info['manifest']['name']}")
    print(f"v2.0 name: {v2_info['manifest']['name']}")
    
    v0_files = await artifact_manager.list_files(dataset.id, version="v0")
    v2_files = await artifact_manager.list_files(dataset.id, version="v2.0")
    
    print(f"Files in v0: {[f['name'] for f in v0_files]}")
    print(f"Files in v2.0: {[f['name'] for f in v2_files]}")

    # File management operations
    # Remove a file (requires staging)
    await artifact_manager.edit(dataset.id, stage=True)
    await artifact_manager.remove_file(dataset.id, file_path="readme.txt")
    await artifact_manager.commit(dataset.id, comment="Removed readme file")
    print("File removed from latest version")

    # Search and filter datasets in the collection
    all_datasets = await artifact_manager.list(collection.id)
    print(f"Total datasets in collection: {len(all_datasets)}")

    # Search by keywords
    search_results = await artifact_manager.list(
        collection.id,
        keywords=["example"],
        mode="AND"
    )
    print(f"Datasets matching 'example': {len(search_results)}")

    # Reset download statistics
    await artifact_manager.reset_stats(dataset.id)
    print("Statistics reset")
    
    # Check updated statistics
    updated_info = await artifact_manager.read(dataset.id)
    print(f"Reset download count: {updated_info.get('download_count', 0)}")
    print(f"Reset view count: {updated_info.get('view_count', 0)}")

    # Demonstrate staging and discard
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={"name": "Staged Changes Test"},
        stage=True
    )
    print("Changes staged")
    
    # Discard staged changes
    await artifact_manager.discard(dataset.id)
    print("Staged changes discarded")

    # List all datasets in the gallery
    datasets = await artifact_manager.list(collection.id)
    print("Final datasets in the gallery:", len(datasets))

    # Optional cleanup (commented out for safety)
    # Delete a specific version
    # await artifact_manager.delete(dataset.id, version="v0", delete_files=True)
    # print("Version v0 deleted")

    # Delete the entire dataset
    # await artifact_manager.delete(dataset.id, delete_files=True)
    # print("Dataset deleted")

    # Delete the collection
    # await artifact_manager.delete(collection.id, delete_files=True, recursive=True)
    # print("Collection deleted")

    print("Example completed successfully!")

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
- **rd+**: Read, write, create draft, and manage access (includes `read`, `get_file`, `get_vector`, `search_vectors`, `list_files`, `list_vectors`, `list`, `edit`, `put_file`, `add_vectors`, `add_documents`, `remove_file`, `remove_vectors`, and `create`), but cannot commit.
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
- `version`: Optional. **Strict Validation Applied**: Must be `None`, `"new"`, or `"stage"`. Historical version editing is NOT supported.
  - `None` (default): Updates the latest existing version
  - `"new"`: Creates a new version immediately (unless `stage=True`)
  - `"stage"`: Enters staging mode (equivalent to `stage=True`)
  
  **Special Staging Behavior**: When `stage=True` and `version="new"` are used together, it stores an intent marker for creating a new version during commit, and allows custom version names to be specified later during the `commit()` operation.
- `stage`: Optional. If `True`, the artifact will be edited in staging mode regardless of the version parameter. Default is `False`. When in staging mode, the artifact must be committed to finalize changes.
- `comment`: Optional. A comment to describe the changes made to the artifact.

**Important Notes:**
- **Version Validation**: The system now strictly validates the `version` parameter. You cannot specify arbitrary custom version names during edit.
- **Staging Intent System**: When you use `version="new"` with `stage=True`, the system stores an "intent" to create a new version, which allows you to specify a custom version name later during `commit()`.
- **No Historical Version Editing**: You cannot edit specific existing versions directly. The `version` parameter only supports `None` (edit latest), `"new"` (create new version), or `"stage"` (staging mode).

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

# Edit in staging mode (valid approach)
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest,
    stage=True  # Use staging mode instead of trying to edit historical versions
)

# INVALID: This will raise an error
await artifact_manager.edit(
    artifact_id="example-dataset",
    manifest=updated_manifest,
    version="v1"  # ERROR: Historical version editing not allowed
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

### `delete(artifact_id: str, delete_files: bool = False, recursive: bool = False, version: str = None) -> None`

Deletes an artifact, its manifest, and all associated files from both the database and S3 storage.

**Parameters:**

- `artifact_id`: The id of the artifact to delete. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `delete_files`: Optional. A boolean flag to delete all files associated with the artifact. Default is `False`.
- `recursive`: Optional. A boolean flag to delete all child artifacts recursively. Default is `False`.
- `version`: Optional. Specifies which version to delete:
  - `None` (default): Delete the entire artifact and all its versions
  - `"stage"`: **Error** - Cannot delete staged version. Use `discard()` instead to remove staged changes
  - `"latest"`: Delete the latest committed version
  - `"v1"`, `"v2"`, etc.: Delete a specific version by name (must exist in version history)
  - Integer index: Delete version by index (0-based)

**Version Handling:**

- **Staged Artifacts**: If you try to delete a staged version (`version="stage"`), the system will raise an error and direct you to use the `discard()` function instead, which is the proper way to remove staged changes.
- **Version Validation**: When deleting a specific version, the system validates that the version exists in the artifact's version history. If the version doesn't exist, an error is raised showing available versions.
- **Complete Deletion**: When `version=None`, the entire artifact is deleted including all versions and metadata.

**Warning: If `delete_files` is set to `True`, `recursive` must be set to `True`, all child artifacts will be deleted, and all files associated with the child artifacts will be permanently deleted from the S3 storage. This operation is irreversible.**

**Examples:**

```python
# Delete entire artifact with all versions
await artifact_manager.delete(artifact_id=artifact.id, delete_files=True)

# Delete a specific version by name
await artifact_manager.delete(artifact_id="example-dataset", version="v1")

# Delete the latest version
await artifact_manager.delete(artifact_id="example-dataset", version="latest")

# This will raise an error - use discard() instead
try:
    await artifact_manager.delete(artifact_id="example-dataset", version="stage")
except ValueError as e:
    print(e)  # "Cannot delete staged version. Please use the 'discard' function instead..."
    # Use discard instead
    await artifact_manager.discard(artifact_id="example-dataset")

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

### `put_file(artifact_id: str, file_path: str, download_weight: float = 0, use_proxy: bool = None, use_local_url: bool = False, expires_in: float = 3600) -> str`

Generates a pre-signed URL to upload a file to the artifact in S3. The URL can be used with an HTTP `PUT` request to upload the file. 

**File Placement Optimization:** Files are placed directly in their final destination based on version intent:
- If the artifact has `new_version` intent (set via `version="new"` during edit), files are uploaded to the new version location
- Otherwise, files are uploaded to the existing latest version location
- This eliminates the need for expensive file copying during commit operations

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"data.csv"`).
- `download_weight`: A float value representing the file's impact on download count when downloaded via any endpoint. Defaults to `0`. Files with weight `0` don't increment download count when accessed.
- `use_proxy`: A boolean to control whether to use the S3 proxy for the generated URL. If `None` (default), follows the server configuration. If `True`, forces the use of the proxy. If `False`, bypasses the proxy and returns a direct S3 URL.
- `use_local_url`: A boolean to control whether to generate URLs for local/cluster-internal access. Defaults to `False`. When `True`, generates URLs suitable for access within the cluster:
  - **With proxy (`use_proxy=True`)**: Uses `local_base_url/s3` instead of the public proxy URL for cluster-internal access
  - **Direct S3 (`use_proxy=False`)**: Uses the S3 endpoint URL as-is (already configured for local access)
  - **Use case**: Useful when services within a cluster need to access files but public URLs are not accessible from within the cluster network
- `expires_in`: A float number for the expiration time of the S3 presigned url

**Returns:** A pre-signed URL for uploading the file.

**Important Note:** Adding files to staging does NOT automatically create new version intent. You must explicitly use `version="new"` during edit operations if you want to create a new version when committing.

**Example:**

```python
put_url = await artifact_manager.put_file(artifact.id, file_path="data.csv", download_weight=1.0)

# If "example-dataset" is an alias of the artifact under the current workspace
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
put_url = await artifact_manager.put_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")

# Upload with proxy bypassed (useful for direct S3 access)
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv", use_proxy=False)

# Upload using local/cluster-internal URLs (useful within cluster networks)
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv", use_local_url=True)

# Upload using local proxy URL for cluster-internal access
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv", use_proxy=True, use_local_url=True)

# Upload the file using an HTTP PUT request
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"
```

---

### `remove_file(artifact_id: str, file_path: str) -> None`

Removes a file from the artifact and updates the staged manifest. The file is also removed from the S3 storage if it exists.

**Parameters:**

- `artifact_id`: The id of the artifact to remove the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to be removed (e.g., `"data.csv"`).

**Error Handling:**

- The function gracefully handles cases where files don't exist in S3 storage
- The file will be removed from the staging manifest even if it doesn't exist in storage
- This prevents errors when removing files that were added to the staging manifest but never uploaded
- Only genuine S3 errors (not "file not found") will cause the operation to fail

**Example:**

```python
await artifact_manager.remove_file(artifact_id=artifact.id, file_path="data.csv")

# If "example-dataset" is an alias of the artifact under the current workspace
await artifact_manager.remove_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
await artifact_manager.remove_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")
```

---

### `get_file(artifact_id: str, file_path: str, silent: bool = False, version: str = None, use_proxy: bool = None, use_local_url: bool = False, expires_in: float = 3600) -> str`

Generates a pre-signed URL to download a file from the artifact stored in S3.

**Parameters:**

- `artifact_id`: The id of the artifact to download the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to download (e.g., `"data.csv"`).
- `silent`: A boolean to suppress the download count increment. Default is `False`.
- `version`: The version of the artifact to download the file from. By default, it downloads from the latest version. If you want to download from a staged version, you can set it to `"stage"`.
- `use_proxy`: A boolean to control whether to use the S3 proxy for the generated URL. If `None` (default), follows the server configuration. If `True`, forces the use of the proxy. If `False`, bypasses the proxy and returns a direct S3 URL.
- `use_local_url`: A boolean to control whether to generate URLs for local/cluster-internal access. Defaults to `False`. When `True`, generates URLs suitable for access within the cluster:
  - **With proxy (`use_proxy=True`)**: Uses `local_base_url/s3` instead of the public proxy URL for cluster-internal access
  - **Direct S3 (`use_proxy=False`)**: Uses the S3 endpoint URL as-is (already configured for local access)
  - **Use case**: Useful when services within a cluster need to access files but public URLs are not accessible from within the cluster network
- `expires_in`: A float number for the expiration time of the S3 presigned url

**Returns:** A pre-signed URL for downloading the file.

**Example:**

```python
get_url = await artifact_manager.get_file(artifact_id=artifact.id, file_path="data.csv")

# If "example-dataset" is an alias of the artifact under the current workspace
get_url = await artifact_manager.get_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
get_url = await artifact_manager.get_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")

# Get a file from a specific version
get_url = await artifact_manager.get_file(artifact_id="example-dataset", file_path="data.csv", version="v1")

# Get a file bypassing the S3 proxy (useful for publishing to external services)
get_url = await artifact_manager.get_file(artifact_id="example-dataset", file_path="data.csv", use_proxy=False)

# Get a file using local/cluster-internal URLs (useful within cluster networks)
get_url = await artifact_manager.get_file(artifact_id="example-dataset", file_path="data.csv", use_local_url=True)

# Get a file using local proxy URL for cluster-internal access
get_url = await artifact_manager.get_file(artifact_id="example-dataset", file_path="data.csv", use_proxy=True, use_local_url=True)
```

---

### `put_file_start_multipart(artifact_id: str, file_path: str, part_count: int, expires_in: int = 3600) -> dict`

Initiates a multipart upload for large files and generates pre-signed URLs for uploading each part. This is useful for files larger than 100MB or when you need to upload files in chunks with parallel processing.

**File Placement Optimization:** Like `put_file`, files are placed directly in their final destination based on version intent:
- If the artifact has `new_version` intent (set via `version="new"` during edit), files are uploaded to the new version location
- Otherwise, files are uploaded to the existing latest version location

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"large_dataset.zip"`).
- `part_count`: The number of parts to split the file into. Must be between 1 and 10,000.
- `expires_in`: The expiration time in seconds for the multipart upload session and part URLs. Defaults to 3600 (1 hour).

**Returns:** A dictionary containing:
- `upload_id`: The unique identifier for this multipart upload session
- `parts`: A list of part information, each containing:
  - `part_number`: The sequential part number (1-indexed)
  - `url`: The pre-signed URL for uploading this specific part

**Important Notes:**
- Each part (except the last) must be at least 5MB in size
- Part URLs expire after the specified `expires_in` time
- The multipart upload session must be completed with `put_file_complete_multipart` 
- If not completed within the expiration time, the session will be automatically cleaned up

**Example:**

```python
# Start multipart upload for a large file
multipart_info = await artifact_manager.put_file_start_multipart(
    artifact_id="example-dataset",
    file_path="large_video.mp4", 
    part_count=5,
    expires_in=7200  # 2 hours
)

upload_id = multipart_info["upload_id"]
part_urls = multipart_info["parts"]

# Upload parts in parallel (example with httpx)
import httpx
import asyncio

async def upload_part(part_info, file_data):
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.put(part_info["url"], data=file_data)
        return {
            "part_number": part_info["part_number"],
            "etag": response.headers["ETag"].strip('"')
        }

# Upload all parts concurrently
uploaded_parts = await asyncio.gather(*[
    upload_part(part, get_part_data(part["part_number"])) 
    for part in part_urls
])
```

---

### `put_file_complete_multipart(artifact_id: str, upload_id: str, parts: list) -> dict`

Completes a multipart upload by combining all uploaded parts into the final file. This must be called after all parts have been successfully uploaded using the URLs from `put_file_start_multipart`.

**Parameters:**

- `artifact_id`: The id of the artifact where the multipart upload was initiated. Must match the artifact_id used in `put_file_start_multipart`.
- `upload_id`: The unique upload identifier returned by `put_file_start_multipart`.
- `parts`: A list of dictionaries containing information about uploaded parts. Each part must include:
  - `part_number`: The part number (1-indexed, matching the original part numbers)
  - `etag`: The ETag returned by S3 when the part was uploaded (without quotes)

**Returns:** A dictionary containing:
- `success`: Boolean indicating whether the multipart upload was completed successfully
- `message`: A descriptive message about the operation result

**Important Notes:**
- All parts must be uploaded before calling this function
- Parts must be provided in the correct order with accurate ETags
- ETags should be provided without surrounding quotes
- The function will fail if any parts are missing or have incorrect ETags
- Once completed successfully, the individual parts are automatically cleaned up by S3

**Example:**

```python
# Complete the multipart upload from the previous example
result = await artifact_manager.put_file_complete_multipart(
    artifact_id="example-dataset",
    upload_id=upload_id,
    parts=uploaded_parts  # From the previous upload_part operations
)

if result["success"]:
    print("Multipart upload completed successfully!")
    print(result["message"])
else:
    print(f"Upload failed: {result['message']}")
```

**Complete Multipart Upload Example:**

```python
import asyncio
import httpx
import os
import math

async def upload_large_file_multipart(artifact_manager, artifact_id, file_path, local_file_path):
    """Complete example of uploading a large file using multipart upload."""
    
    # Calculate optimal part count (aim for ~100MB per part)
    file_size = os.path.getsize(local_file_path)
    part_size = 100 * 1024 * 1024  # 100MB
    part_count = max(1, math.ceil(file_size / part_size))
    
    print(f"Uploading {file_size / (1024*1024):.1f}MB file in {part_count} parts")
    
    # Step 1: Start multipart upload
    multipart_info = await artifact_manager.put_file_start_multipart(
        artifact_id=artifact_id,
        file_path=file_path,
        part_count=part_count,
        expires_in=3600
    )
    
    upload_id = multipart_info["upload_id"]
    part_urls = multipart_info["parts"]
    
    # Step 2: Upload all parts
    async def upload_part(part_info):
        part_number = part_info["part_number"]
        start_pos = (part_number - 1) * part_size
        end_pos = min(start_pos + part_size, file_size)
        
        with open(local_file_path, "rb") as f:
            f.seek(start_pos)
            part_data = f.read(end_pos - start_pos)
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(part_info["url"], data=part_data)
            if response.status_code != 200:
                raise Exception(f"Failed to upload part {part_number}")
                
            return {
                "part_number": part_number,
                "etag": response.headers["ETag"].strip('"')
            }
    
    # Upload parts with controlled concurrency
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent uploads
    
    async def upload_with_semaphore(part_info):
        async with semaphore:
            return await upload_part(part_info)
    
    uploaded_parts = await asyncio.gather(*[
        upload_with_semaphore(part) for part in part_urls
    ])
    
    # Step 3: Complete multipart upload
    result = await artifact_manager.put_file_complete_multipart(
        artifact_id=artifact_id,
        upload_id=upload_id,
        parts=uploaded_parts
    )
    
    return result

# Usage example
# result = await upload_large_file_multipart(
#     artifact_manager, "my-dataset", "data/large_file.zip", "/path/to/large_file.zip"
# )
```

---

### `list_files(artifact_id: str, dir_path: str=None, version: str = None, stage: bool = False, include_pending: bool = False) -> list`

Lists all files in the artifact.

**Parameters:**

- `artifact_id`: The id of the artifact to list files from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `dir_path`: Optional. The directory path within the artifact to list files. Default is `None`.
- `version`: Optional. The version of the artifact to list files from. By default, it reads files from the latest version. If you want to read files from a staged version, you can set it to `"stage"`.
- `stage`: Optional. A boolean flag to list files from the staged version. Default is `False`. When `True`, it's equivalent to setting `version="stage"`.
- `include_pending`: Optional. A boolean flag to include files that are pending in the staging manifest but may not yet be uploaded to S3. Only works when `version="stage"` or `stage=True`. Default is `False`. When enabled, the function will show both actual files that exist in storage and files that are registered in the staging manifest but may not yet be uploaded.

**Returns:** A list of files in the artifact. When `include_pending=True`, pending files will have a `"pending": True` property to distinguish them from actual uploaded files.

**Staging Version Behavior:**

- When `version="stage"` is specified, the function consistently uses version index `len(versions)` (the next version that would be created upon commit)
- This ensures predictable behavior regardless of staging intent markers
- The staging version index represents the "working" version where new files are placed during staging

**Example:**

```python
files = await artifact_manager.list_files(artifact_id=artifact.id)

# If "example-dataset" is an alias of the artifact under the current workspace
files = await artifact_manager.list_files(artifact_id="example-dataset")

# If "example-dataset" is an alias of the artifact under another workspace
files = await artifact_manager.list_files(artifact_id="other_workspace/example-dataset")

# List files from staging version including pending files
files = await artifact_manager.list_files(
    artifact_id="example-dataset", 
    version="stage", 
    include_pending=True
)

# Example response with pending files
[
    {"name": "uploaded_file.csv", "type": "file", "size": 1024, "pending": False},
    {"name": "pending_file.txt", "type": "file", "size": 0, "pending": True}
]
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

Here's a comprehensive sequence of API function calls demonstrating common artifact management operations:

### Basic Setup and Collection Creation

```python
import asyncio
import requests
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def comprehensive_example():
    # Step 1: Connect to the API
    server = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await server.get_service("public/artifact-manager")

    # Step 2: Create a gallery collection with schema validation
    dataset_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string", "enum": ["research", "demo", "production"]},
        },
        "required": ["name", "description", "category"]
    }
    
    gallery_manifest = {
        "name": "Dataset Gallery",
        "description": "A collection for organizing datasets with validation",
    }
    
    collection = await artifact_manager.create(
        type="collection", 
        alias="dataset-gallery", 
        manifest=gallery_manifest, 
        config={
            "collection_schema": dataset_schema,
            "permissions": {"*": "r", "@": "r+"}
        }
    )
    print(f"Collection created with ID: {collection.id}")
```

### Dataset Creation and File Management

```python
    # Step 3: Create a dataset with multiple files
    dataset_manifest = {
        "name": "Research Dataset",
        "description": "A comprehensive research dataset",
        "category": "research",
        "authors": ["Dr. Smith", "Dr. Johnson"],
        "license": "MIT"
    }
    
    dataset = await artifact_manager.create(
        parent_id=collection.id, 
        alias="research-dataset", 
        manifest=dataset_manifest, 
        stage=True
    )
    print(f"Dataset created: {dataset.id}")

    # Step 4: Upload multiple files with different weights
    files_to_upload = [
        {"path": "data/main_dataset.csv", "local": "path/to/main_data.csv", "weight": 1.0},
        {"path": "data/metadata.json", "local": "path/to/metadata.json", "weight": 0.1},
        {"path": "docs/README.md", "local": "path/to/readme.md", "weight": 0.0},
        {"path": "scripts/analysis.py", "local": "path/to/analysis.py", "weight": 0.2}
    ]
    
    for file_info in files_to_upload:
        put_url = await artifact_manager.put_file(
            dataset.id, 
            file_path=file_info["path"], 
            download_weight=file_info["weight"]
        )
        with open(file_info["local"], "rb") as f:
            response = requests.put(put_url, data=f)
            assert response.ok, f"Upload failed for {file_info['path']}"
        print(f"Uploaded: {file_info['path']}")

    # Step 5: Commit the initial version
    await artifact_manager.commit(dataset.id)
    print("Dataset v0 committed")
```

### Reading and Accessing Data

```python
    # Step 6: Read dataset information
    dataset_info = await artifact_manager.read(dataset.id, include_metadata=True)
    print(f"Dataset: {dataset_info['manifest']['name']}")
    print(f"Description: {dataset_info['manifest']['description']}")
    print(f"View count: {dataset_info.get('view_count', 0)}")
    print(f"Download count: {dataset_info.get('download_count', 0)}")

    # Step 7: List and access files
    files = await artifact_manager.list_files(dataset.id)
    print("Files in dataset:")
    for file in files:
        print(f"  - {file['name']} ({file.get('size', 'unknown')} bytes)")

    # Step 8: Download specific files
    # Get main dataset file
    main_data_url = await artifact_manager.get_file(dataset.id, file_path="data/main_dataset.csv")
    response = requests.get(main_data_url)
    if response.ok:
        print("Main dataset downloaded successfully")
        print(f"Content preview: {response.text[:200]}...")
    
    # Get metadata without incrementing download count
    metadata_url = await artifact_manager.get_file(
        dataset.id, 
        file_path="data/metadata.json", 
        silent=True
    )
    metadata_response = requests.get(metadata_url)
    if metadata_response.ok:
        import json
        metadata = json.loads(metadata_response.text)
        print(f"Metadata: {metadata}")
```

### Dataset Updates and File Operations

```python
    # Step 9: Update dataset manifest
    updated_manifest = {
        "name": "Research Dataset (Updated)",
        "description": "A comprehensive research dataset with recent updates",
        "category": "research",
        "authors": ["Dr. Smith", "Dr. Johnson", "Dr. Wilson"],
        "license": "MIT",
        "tags": ["research", "updated", "2024"]
    }
    
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest=updated_manifest,
        stage=True
    )
    
    # Step 10: Add new files and remove old ones
    # Add a new analysis file
    put_url = await artifact_manager.put_file(dataset.id, file_path="analysis/results.csv", download_weight=0.5)
    results_data = "column1,column2,column3\nvalue1,value2,value3\n"
    response = requests.put(put_url, data=results_data.encode())
    assert response.ok, "Results upload failed"
    
    # Remove an old file
    await artifact_manager.remove_file(dataset.id, file_path="scripts/analysis.py")
    print("Old analysis script removed")
    
    # Commit the updates (this updates existing version v0)
    await artifact_manager.commit(dataset.id)
    print("Dataset updated successfully")
```

### Version Management

```python
    # Step 11: Create a new version with significant changes
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={
            "name": "Research Dataset v2",
            "description": "Major update with new analysis methods",
            "category": "research",
            "version": "2.0",
            "authors": ["Dr. Smith", "Dr. Johnson", "Dr. Wilson"],
            "license": "MIT"
        },
        stage=True,
        version="new"  # Explicit intent to create new version
    )
    
    # Add files specific to v2
    put_url = await artifact_manager.put_file(dataset.id, file_path="data/enhanced_dataset.csv", download_weight=1.2)
    enhanced_data = "id,feature1,feature2,target\n1,0.5,0.3,1\n2,0.8,0.1,0\n"
    response = requests.put(put_url, data=enhanced_data.encode())
    assert response.ok, "Enhanced data upload failed"
    
    # Commit with custom version name
    await artifact_manager.commit(dataset.id, version="v2.0-beta")
    print("Version v2.0-beta created")
    
    # Compare versions
    v0_info = await artifact_manager.read(dataset.id, version="v0")
    v2_info = await artifact_manager.read(dataset.id, version="v2.0-beta")
    
    print(f"v0 name: {v0_info['manifest']['name']}")
    print(f"v2 name: {v2_info['manifest']['name']}")
    
    v0_files = await artifact_manager.list_files(dataset.id, version="v0")
    v2_files = await artifact_manager.list_files(dataset.id, version="v2.0-beta")
    
    print(f"Files in v0: {[f['name'] for f in v0_files]}")
    print(f"Files in v2: {[f['name'] for f in v2_files]}")
```

### Collection Management and Search

```python
    # Step 12: Create additional datasets for demonstration
    datasets_to_create = [
        {
            "alias": "demo-dataset",
            "manifest": {
                "name": "Demo Dataset", 
                "description": "Dataset for demonstrations",
                "category": "demo"
            }
        },
        {
            "alias": "production-dataset", 
            "manifest": {
                "name": "Production Dataset",
                "description": "Production-ready dataset", 
                "category": "production"
            }
        }
    ]
    
    for dataset_config in datasets_to_create:
        demo_dataset = await artifact_manager.create(
            parent_id=collection.id,
            alias=dataset_config["alias"],
            manifest=dataset_config["manifest"],
            stage=True
        )
        # Add a simple file
        put_url = await artifact_manager.put_file(demo_dataset.id, file_path="data.txt")
        response = requests.put(put_url, data=b"Sample data")
        await artifact_manager.commit(demo_dataset.id)
    
    # Step 13: Search and filter datasets
    # List all datasets in the collection
    all_datasets = await artifact_manager.list(collection.id)
    print(f"Total datasets: {len(all_datasets)}")
    
    # Search by keywords
    research_datasets = await artifact_manager.list(
        collection.id,
        keywords=["research"],
        mode="AND"
    )
    print(f"Research datasets: {len(research_datasets)}")
    
    # Filter by category
    demo_datasets = await artifact_manager.list(
        collection.id,
        filters={"manifest": {"category": "demo"}},
        order_by="view_count>"
    )
    print(f"Demo datasets: {len(demo_datasets)}")
    
    # Advanced filtering with multiple conditions
    recent_research = await artifact_manager.list(
        collection.id,
        keywords=["research", "updated"],
        filters={"manifest": {"category": "research"}},
        mode="AND",
        order_by="last_modified>",
        limit=5
    )
    print(f"Recent research datasets: {len(recent_research)}")
```

### Statistics and Monitoring

```python
    # Step 14: Monitor download statistics
    # Reset statistics for testing
    await artifact_manager.reset_stats(dataset.id)
    print("Statistics reset")
    
    # Simulate downloads and track statistics
    for i in range(3):
        download_url = await artifact_manager.get_file(dataset.id, file_path="data/enhanced_dataset.csv")
        response = requests.get(download_url)
        print(f"Download {i+1} completed")
    
    # Check updated statistics
    updated_info = await artifact_manager.read(dataset.id, include_metadata=True)
    print(f"Updated download count: {updated_info.get('download_count', 0)}")
    print(f"Updated view count: {updated_info.get('view_count', 0)}")
```

### Advanced Features

```python
    # Step 15: Multipart upload for large files (example)
    # This would be used for files > 100MB
    """
    multipart_info = await artifact_manager.put_file_start_multipart(
        dataset.id, 
        file_path="large_data.zip", 
        part_count=3
    )
    # ... upload parts ...
    result = await artifact_manager.put_file_complete_multipart(
        dataset.id,
        upload_id=multipart_info["upload_id"],
        parts=uploaded_parts
    )
    """
    
    # Step 16: Working with staged changes
    # Edit dataset and view staged changes before committing
    await artifact_manager.edit(
        artifact_id=dataset.id,
        manifest={"name": "Staged Update Test"},
        stage=True
    )
    
    # List staged files (includes pending files)
    staged_files = await artifact_manager.list_files(
        dataset.id, 
        version="stage", 
        include_pending=True
    )
    print(f"Staged files: {[f['name'] for f in staged_files]}")
    
    # Discard staged changes if needed
    await artifact_manager.discard(dataset.id)
    print("Staged changes discarded")
```

### Cleanup

```python
    # Step 17: Cleanup (optional)
    # Delete specific version
    # await artifact_manager.delete(dataset.id, version="v0", delete_files=True)
    
    # Delete entire dataset
    # await artifact_manager.delete(dataset.id, delete_files=True)
    
    # Delete collection with all children
    # await artifact_manager.delete(collection.id, delete_files=True, recursive=True)
    
    print("Example completed successfully!")

# Run the comprehensive example
asyncio.run(comprehensive_example())
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
  - **Query Parameters**:
    - `use_proxy`: (Optional) Boolean to control whether to use the S3 proxy
    - `use_local_url`: (Optional) Boolean to generate local/cluster-internal URLs

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

### File Upload Endpoints

The Artifact Manager provides HTTP endpoints for uploading files directly to artifacts. These endpoints support both single file uploads and multipart uploads for large files.

#### Single File Upload Endpoint

**Endpoint**: `PUT /{workspace}/artifacts/{artifact_alias}/files/{file_path:path}`

Upload a single file by streaming it directly to S3 storage. This endpoint is efficient for handling files of any size as it streams the request body directly to S3 without local temporary storage.

**Request Format:**
- **Method**: `PUT`
- **Path Parameters**:
  - **workspace**: The workspace containing the artifact
  - **artifact_alias**: The alias or ID of the artifact
  - **file_path**: The relative path where the file will be stored within the artifact
- **Query Parameters**:
  - **download_weight**: (Optional) Float value representing the file's impact on download count. Defaults to `0`.
- **Headers**:
  - `Authorization`: Optional. Bearer token for private artifact access
  - `Content-Type`: Optional. MIME type of the file being uploaded
  - `Content-Length`: Optional. Size of the file being uploaded
- **Body**: Raw file content

**Response:**
```json
{
    "success": true,
    "message": "File uploaded successfully"
}
```

**Example Usage:**
```python
import asyncio
import httpx

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
artifact_alias = "example-dataset"
file_path = "data/example.csv"

async def upload_file():
    # Upload a file with download weight
    with open("local_file.csv", "rb") as f:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(
                f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/{file_path}",
                data=f,
                params={"download_weight": 1.0},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "text/csv"
                }
            )

    if response.status_code == 200:
        print("File uploaded successfully")
    else:
        print(f"Upload failed: {response.status_code}")

# Run the async function
asyncio.run(upload_file())
```

#### Multipart Upload Endpoints

For large files, the Artifact Manager supports multipart uploads which allow uploading files in chunks. This is useful for files larger than 5GB or when you need to resume interrupted uploads.

##### 1. Create Multipart Upload

**Endpoint**: `POST /{workspace}/artifacts/{artifact_alias}/create-multipart-upload`

Initiate a multipart upload and get presigned URLs for all parts.

**Request Format:**
- **Method**: `POST`
- **Path Parameters**:
  - **workspace**: The workspace containing the artifact
  - **artifact_alias**: The alias or ID of the artifact
- **Query Parameters**:
  - **path**: The relative path where the file will be stored within the artifact
  - **part_count**: The total number of parts for the upload (required, max 10,000)
  - **expires_in**: (Optional) Number of seconds for presigned URLs to expire. Defaults to 3600.

**Response:**
```json
{
    "upload_id": "abc123...",
    "parts": [
        {"part_number": 1, "url": "https://s3.amazonaws.com/..."},
        {"part_number": 2, "url": "https://s3.amazonaws.com/..."},
        ...
    ]
}
```

##### 2. Complete Multipart Upload

**Endpoint**: `POST /{workspace}/artifacts/{artifact_alias}/complete-multipart-upload`

Complete the multipart upload and finalize the file in S3.

**Request Format:**
- **Method**: `POST`
- **Path Parameters**:
  - **workspace**: The workspace containing the artifact
  - **artifact_alias**: The alias or ID of the artifact
- **Body**:
```json
{
    "upload_id": "abc123...",
    "parts": [
        {"part_number": 1, "etag": "etag1"},
        {"part_number": 2, "etag": "etag2"},
        ...
    ]
}
```

**Response:**
```json
{
    "success": true,
    "message": "File uploaded successfully"
}
```

**Complete Multipart Upload Example:**
```python
import asyncio
import httpx
import os

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
artifact_alias = "example-dataset"
file_path = "large_file.zip"

async def upload_large_file():
    # Step 1: Create multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-multipart-upload",
            params={
                "path": file_path,
                "part_count": 3,
                "expires_in": 3600
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Failed to create multipart upload: {response.status_code}")
            return
            
        upload_data = response.json()
        upload_id = upload_data["upload_id"]
        parts = upload_data["parts"]

    # Step 2: Upload all parts in parallel
    file_size = os.path.getsize("large_file.zip")
    chunk_size = file_size // 3
    uploaded_parts = []
    
    async def upload_part(part_info):
        """Upload a single part by reading directly from disk."""
        part_number = part_info["part_number"]
        url = part_info["url"]
        
        # Calculate start and end positions for this chunk
        start_pos = (part_number - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, file_size)
        chunk_size_actual = end_pos - start_pos
        
        # Read chunk directly from disk with seek
        with open("large_file.zip", "rb") as f:
            f.seek(start_pos)
            chunk_data = f.read(chunk_size_actual)
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(url, data=chunk_data)
            if response.status_code != 200:
                raise Exception(f"Failed to upload part {part_number}")
                
            # Save the ETag from the response header
            etag = response.headers["ETag"].strip('"').strip("'")
            return {
                "part_number": part_number,
                "etag": etag
            }
    
    # Upload all parts in parallel using asyncio.gather (equivalent to Promise.all)
    try:
        uploaded_parts = await asyncio.gather(*[upload_part(part) for part in parts])
        print(f"Successfully uploaded {len(uploaded_parts)} parts in parallel")
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    # Step 3: Complete multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/complete-multipart-upload",
            json={
                "upload_id": upload_id,
                "parts": uploaded_parts
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            print("Multipart upload completed successfully")
        else:
            print(f"Failed to complete multipart upload: {response.status_code}")

# Run the async function
asyncio.run(upload_large_file())
```

**Important Notes:**
- **Part Limits**: Maximum 10,000 parts per multipart upload
- **Part Size**: Each part must be at least 5MB (except the last part)
- **ETags**: The ETag returned from S3 when uploading each part must be included in the completion request
- **Session Expiration**: Upload sessions expire after the specified `expires_in` time
- **Permissions**: Requires `put_file` permission on the artifact
- **Parallel Uploads**: Using `asyncio.gather()` for parallel uploads significantly improves performance for large files

**Advanced Example with Optimized Chunking:**
```python
import asyncio
import httpx
import os
import math

async def upload_large_file_optimized(file_path, artifact_alias, workspace, token):
    """Upload a large file with optimized chunking and parallel uploads."""
    
    # Calculate optimal chunk size (5MB minimum, 100MB recommended for large files)
    file_size = os.path.getsize(file_path)
    min_chunk_size = 5 * 1024 * 1024  # 5MB
    recommended_chunk_size = 100 * 1024 * 1024  # 100MB
    chunk_size = max(min_chunk_size, min(recommended_chunk_size, file_size // 10))
    part_count = math.ceil(file_size / chunk_size)
    
    # For very large files, limit concurrent uploads to avoid overwhelming the system
    max_concurrent_uploads = min(10, part_count)  # Max 10 concurrent uploads
    
    print(f"File size: {file_size / (1024*1024):.1f}MB")
    print(f"Chunk size: {chunk_size / (1024*1024):.1f}MB")
    print(f"Number of parts: {part_count}")
    
    # Step 1: Create multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-multipart-upload",
            params={
                "path": os.path.basename(file_path),
                "part_count": part_count,
                "expires_in": 3600
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            print(f"Failed to create multipart upload: {response.status_code}")
            return
            
        upload_data = response.json()
        upload_id = upload_data["upload_id"]
        parts = upload_data["parts"]

    # Step 2: Upload all parts in parallel with progress tracking
    async def upload_part_with_progress(part_info):
        """Upload a single part by reading directly from disk with offset."""
        part_number = part_info["part_number"]
        url = part_info["url"]
        
        # Calculate start and end positions for this chunk
        start_pos = (part_number - 1) * chunk_size
        end_pos = min(start_pos + chunk_size, file_size)
        chunk_size_actual = end_pos - start_pos
        
        print(f"Starting upload of part {part_number}/{part_count} (size: {chunk_size_actual / (1024*1024):.1f}MB)")
        
        # Read chunk directly from disk with seek
        with open(file_path, "rb") as f:
            f.seek(start_pos)
            chunk_data = f.read(chunk_size_actual)
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(url, data=chunk_data)
            if response.status_code != 200:
                raise Exception(f"Failed to upload part {part_number}")
                
            etag = response.headers["ETag"].strip('"').strip("'")
            print(f"Completed upload of part {part_number}/{part_count}")
            return {
                "part_number": part_number,
                "etag": etag
            }
    
    # Upload all parts with controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent_uploads)
    
    async def upload_part_with_semaphore(part_info):
        """Upload a single part with semaphore control."""
        async with semaphore:
            return await upload_part_with_progress(part_info)
    
    try:
        print(f"Starting upload of {part_count} parts with max {max_concurrent_uploads} concurrent uploads...")
        uploaded_parts = await asyncio.gather(*[upload_part_with_semaphore(part) for part in parts])
        print(f"Successfully uploaded all {len(uploaded_parts)} parts")
    except Exception as e:
        print(f"Upload failed: {e}")
        return

    # Step 3: Complete multipart upload
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/complete-multipart-upload",
            json={
                "upload_id": upload_id,
                "parts": uploaded_parts
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code == 200:
            print("Multipart upload completed successfully")
        else:
            print(f"Failed to complete multipart upload: {response.status_code}")

# Usage example
# asyncio.run(upload_large_file_optimized("large_file.zip", "example-dataset", "my-workspace", token))
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

---

## Static Site Hosting

The Artifact Manager supports hosting static websites through special "site" artifacts. These artifacts can serve static files with optional Jinja2 template rendering, making them perfect for hosting documentation, portfolios, or any static web content.

### Site Artifact Features

- **Static File Serving**: Serve HTML, CSS, JavaScript, images, and other static files
- **Template Rendering**: Jinja2 template support for dynamic content
- **Custom Headers**: Configure CORS, caching, and other HTTP headers
- **Version Control**: Support for staging and versioning of site content
- **Access Control**: Permission-based access to site content
- **View Statistics**: Track page views and download counts

### Creating and Deploying a Static Site

#### Step 1: Create a Site Artifact

```python
import asyncio
import httpx
import requests
from hypha_rpc import connect_to_server

async def deploy_static_site():
    # Connect to the Artifact Manager
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace
    token = await api.generate_token()

    # Create a site artifact
    manifest = {
        "name": "My Portfolio",
        "description": "A personal portfolio website",
        "type": "site"
    }
    
    config = {
        "templates": ["index.html", "about.html"],  # Files that use Jinja2 templates
        "template_engine": "jinja2",
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "max-age=3600"
        }
    }
    
    site_artifact = await artifact_manager.create(
        type="site",
        alias="my-portfolio",
        manifest=manifest,
        config=config,
        stage=True
    )
    
    print(f"Site artifact created with ID: {site_artifact.id}")
    return site_artifact, workspace, token
```

#### Step 2: Upload Site Files

```python
async def upload_site_files(artifact_manager, site_artifact, token):
    # Upload regular HTML file (non-templated)
    static_html = """<!DOCTYPE html>
<html>
<head><title>Static Page</title></head>
<body><h1>This is a static page</h1></body>
</html>"""
    
    file_url = await artifact_manager.put_file(site_artifact.id, "static.html")
    response = requests.put(file_url, data=static_html.encode(), headers={"Content-Type": "text/html"})
    assert response.ok
    
    # Upload templated HTML file (with Jinja2)
    template_html = """<!DOCTYPE html>
<html>
<head>
    <title>{{ MANIFEST.name }}</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>{{ MANIFEST.name }}</h1>
        <p>{{ MANIFEST.description }}</p>
    </header>
    
    <main>
        <section>
            <h2>Site Information</h2>
            <ul>
                <li><strong>Base URL:</strong> {{ BASE_URL }}</li>
                <li><strong>Workspace:</strong> {{ WORKSPACE }}</li>
                <li><strong>View Count:</strong> {{ VIEW_COUNT }}</li>
                <li><strong>Download Count:</strong> {{ DOWNLOAD_COUNT }}</li>
            </ul>
        </section>
        
        <section>
            <h2>User Information</h2>
            {% if USER %}
                <p>Welcome, {{ USER.id }}!</p>
            {% else %}
                <p>Welcome, Anonymous user!</p>
            {% endif %}
        </section>
        
        <section>
            <h2>Configuration</h2>
            <pre>{{ CONFIG | tojson(indent=2) }}</pre>
        </section>
    </main>
    
    <footer>
        <p>Powered by Hypha Artifact Manager</p>
    </footer>
</body>
</html>"""
    
    file_url = await artifact_manager.put_file(site_artifact.id, "index.html")
    response = requests.put(file_url, data=template_html.encode(), headers={"Content-Type": "text/html"})
    assert response.ok
    
    # Upload CSS file
    css_content = """
body {
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
}

header {
    background-color: #f4f4f4;
    padding: 20px;
    border-radius: 5px;
    margin-bottom: 20px;
}

section {
    margin-bottom: 30px;
    padding: 20px;
    border: 1px solid #ddd;
    border-radius: 5px;
}

footer {
    text-align: center;
    padding: 20px;
    color: #666;
    border-top: 1px solid #ddd;
    margin-top: 40px;
}
"""
    
    file_url = await artifact_manager.put_file(site_artifact.id, "style.css")
    response = requests.put(file_url, data=css_content.encode(), headers={"Content-Type": "text/css"})
    assert response.ok
    
    # Upload JavaScript file
    js_content = """
console.log('Site loaded successfully!');
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM ready');
});
"""
    
    file_url = await artifact_manager.put_file(site_artifact.id, "script.js")
    response = requests.put(file_url, data=js_content.encode(), headers={"Content-Type": "application/javascript"})
    assert response.ok
    
    print("All site files uploaded successfully")
```

#### Step 3: Commit and Deploy

```python
async def commit_and_deploy(artifact_manager, site_artifact):
    # Commit the site
    await artifact_manager.commit(site_artifact.id)
    print("Site committed and deployed!")
    
    # The site is now accessible at:
    # https://your-hypha-server.com/{workspace}/site/my-portfolio/
    
    return site_artifact
```

#### Step 4: Complete Deployment Example

```python
async def main():
    # Step 1: Create site artifact
    site_artifact, workspace, token = await deploy_static_site()
    
    # Step 2: Upload files
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await api.get_service("public/artifact-manager")
    await upload_site_files(artifact_manager, site_artifact, token)
    
    # Step 3: Deploy
    await commit_and_deploy(artifact_manager, site_artifact)
    
    # Step 4: Test the site
    site_url = f"{SERVER_URL}/{workspace}/site/my-portfolio/"
    print(f"Your site is now live at: {site_url}")
    
    # Test the site
    async with httpx.AsyncClient() as client:
        response = await client.get(site_url, headers={"Authorization": f"Bearer {token}"})
        if response.status_code == 200:
            print("✅ Site is working correctly!")
        else:
            print(f"❌ Site test failed: {response.status_code}")

# Run the deployment
asyncio.run(main())
```

### Site Configuration Options

#### Template Configuration

```python
config = {
    "templates": ["index.html", "about.html", "contact.html"],  # Files to render with Jinja2
    "template_engine": "jinja2",  # Currently only Jinja2 is supported
    "headers": {
        "Access-Control-Allow-Origin": "*",  # CORS headers
        "Cache-Control": "max-age=3600",     # Caching headers
        "X-Frame-Options": "DENY",           # Security headers
        "X-Content-Type-Options": "nosniff"
    }
}
```

#### Available Template Variables

When using Jinja2 templates, the following variables are available:

- **`MANIFEST`**: The artifact's manifest data
- **`CONFIG`**: The artifact's configuration (excluding secrets)
- **`BASE_URL`**: The base URL for the site (e.g., `/workspace/site/alias/`)
- **`PUBLIC_BASE_URL`**: The public base URL of the Hypha server
- **`LOCAL_BASE_URL`**: The local base URL for cluster-internal access
- **`WORKSPACE`**: The workspace name
- **`VIEW_COUNT`**: Number of times the artifact has been viewed
- **`DOWNLOAD_COUNT`**: Number of times files have been downloaded
- **`USER`**: User information (if authenticated, otherwise None)

### Site Serving Endpoints

#### HTTP Endpoint

**URL Pattern**: `GET /{workspace}/site/{artifact_alias}/{file_path:path}`

**Parameters**:
- **workspace**: The workspace containing the site
- **artifact_alias**: The alias of the site artifact
- **file_path**: (Optional) Path to the file within the site. Defaults to `index.html`

**Query Parameters**:
- **stage**: (Optional) Boolean to serve from staged version instead of committed version
- **token**: (Optional) User token for private site access
- **version**: (Optional) Specific version to serve

**Features**:
- **Automatic MIME Type Detection**: Files are served with appropriate Content-Type headers
- **Template Rendering**: Files listed in `templates` config are rendered with Jinja2
- **Custom Headers**: Headers specified in config are added to responses
- **Default Index**: Empty paths or `/` automatically serve `index.html`
- **View Tracking**: Each request increments the artifact's view count

#### Example Usage

```python
import httpx

async def test_site():
    async with httpx.AsyncClient() as client:
        # Access the main page
        response = await client.get(
            f"{SERVER_URL}/{workspace}/site/my-portfolio/",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Main page: {response.status_code}")
        
        # Access a specific file
        response = await client.get(
            f"{SERVER_URL}/{workspace}/site/my-portfolio/style.css",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"CSS file: {response.status_code}")
        
        # Access staged version
        response = await client.get(
            f"{SERVER_URL}/{workspace}/site/my-portfolio/?stage=true",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Staged version: {response.status_code}")
```

### Site Management

#### Updating Site Content

```python
async def update_site(artifact_manager, site_artifact):
    # Edit the site in staging mode
    await artifact_manager.edit(
        artifact_id=site_artifact.id,
        manifest={"name": "Updated Portfolio", "description": "Updated description"},
        stage=True
    )
    
    # Upload new files
    new_content = "<h1>Updated content</h1>"
    file_url = await artifact_manager.put_file(site_artifact.id, "updated.html")
    response = requests.put(file_url, data=new_content.encode(), headers={"Content-Type": "text/html"})
    
    # Commit the changes
    await artifact_manager.commit(site_artifact.id)
    print("Site updated successfully!")
```

#### Version Management

```python
async def manage_site_versions(artifact_manager, site_artifact):
    # Create a new version
    await artifact_manager.edit(
        artifact_id=site_artifact.id,
        manifest={"name": "Portfolio v2"},
        stage=True,
        version="new"
    )
    
    # Upload new content for v2
    v2_content = "<h1>Portfolio Version 2</h1>"
    file_url = await artifact_manager.put_file(site_artifact.id, "index.html")
    response = requests.put(file_url, data=v2_content.encode(), headers={"Content-Type": "text/html"})
    
    # Commit with custom version name
    await artifact_manager.commit(site_artifact.id, version="v2.0")
    print("New version created!")
```

### Best Practices

1. **File Organization**: Organize your site files logically (CSS in `/css/`, JS in `/js/`, etc.)
2. **Template Usage**: Use templates for dynamic content, static files for performance
3. **Caching**: Configure appropriate cache headers for static assets
4. **Security**: Set proper security headers in your site configuration
5. **Testing**: Always test staged versions before committing to production
6. **Backup**: Use versioning to maintain backup copies of your site

### Example: Complete Portfolio Site

Here's a complete example of deploying a portfolio site:

```python
async def deploy_portfolio():
    # Setup
    api = await connect_to_server({"name": "portfolio-deploy", "server_url": SERVER_URL})
    artifact_manager = await api.get_service("public/artifact-manager")
    workspace = api.config.workspace
    token = await api.generate_token()
    
    # Create site artifact
    site = await artifact_manager.create(
        type="site",
        alias="portfolio",
        manifest={
            "name": "John Doe - Portfolio",
            "description": "Software Engineer & Data Scientist",
            "type": "site"
        },
        config={
            "templates": ["index.html", "about.html", "projects.html"],
            "template_engine": "jinja2",
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "max-age=3600"
            }
        },
        stage=True
    )
    
    # Upload files (simplified for brevity)
    files = {
        "index.html": "<h1>{{ MANIFEST.name }}</h1><p>{{ MANIFEST.description }}</p>",
        "style.css": "body { font-family: Arial; }",
        "script.js": "console.log('Portfolio loaded');"
    }
    
    for filename, content in files.items():
        file_url = await artifact_manager.put_file(site.id, filename)
        response = requests.put(file_url, data=content.encode())
        assert response.ok
    
    # Deploy
    await artifact_manager.commit(site.id)
    
    print(f"Portfolio deployed at: {SERVER_URL}/{workspace}/site/portfolio/")
    return site

# Deploy the portfolio
asyncio.run(deploy_portfolio())
```

---

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
