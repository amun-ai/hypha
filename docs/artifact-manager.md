# Artifact Manager

The `Artifact Manager` is a built-in Hypha service for indexing, managing, and storing resources such as datasets, AI models, and applications. It provides a structured way to manage datasets and similar resources, enabling efficient listing, uploading, updating, and deleting of files. It also now supports tracking download statistics for each artifact.

A typical use case for the `Artifact Manager` is as a backend for a single-page web application that displays a gallery of datasets, AI models, applications, or other types of resources. The default metadata of an artifact is designed to render a grid of cards on a webpage. It also supports tracking download statistics.

**Note:** The `Artifact Manager` is only available when your Hypha server has S3 storage enabled.

---

## Getting Started

### Step 1: Connecting to the Artifact Manager Service

To use the `Artifact Manager`, you first need to connect to the Hypha server. This API allows you to create, read, edit, and delete datasets in the artifact registry (stored in an S3 bucket for each workspace).

```python
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"  # Replace with your server URL

# Connect to the Dataset Manager API
server = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
artifact_manager = await server.get_service("public/artifact-manager")
```

### Step 2: Creating a Dataset Gallery Collection

Once connected, you can create a collection to organize datasets in the gallery.

```python
# Create a collection for the Dataset Gallery
gallery_manifest = {
    "id": "dataset-gallery",
    "name": "Dataset Gallery",
    "description": "A collection for organizing datasets",
    "type": "collection",
    "collection": [],
}

# Create the collection with read permission for everyone and create permission for all authenticated users
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r", "@": "r+"})
print("Dataset Gallery created.")
```

### Step 3: Adding a Dataset to the Gallery

After creating the gallery, you can start adding datasets to it. Each dataset will have its own manifest that describes the dataset and any associated files.

```python
# Create a new dataset inside the Dataset Gallery
dataset_manifest = {
    "id": "example-dataset",
    "name": "Example Dataset",
    "description": "A dataset with example data",
    "type": "dataset",
    "files": [],
}

await artifact_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)
print("Dataset added to the gallery.")
```

### Step 4: Uploading Files to the Dataset with Download Statistics

Once you have created a dataset, you can upload files to it by generating a pre-signed URL. This URL allows you to upload the actual files to the artifact's S3 bucket.

Additionally, when uploading files to an artifact, you can specify a `download_weight` for each file. This weight determines how the file impacts the artifact's download count when it is accessed. For example, primary files might have a higher `download_weight`, while secondary files might have no impact. The download count is automatically updated whenever users download files from the artifact.

```python
# Get a pre-signed URL to upload a file, with a download_weight assigned
put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv", options={"download_weight": 0.5})

# Upload the file using an HTTP PUT request
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

print("File uploaded to the dataset.")
```

### Step 5: Committing the Dataset

After uploading the files, commit the dataset to finalize it. This will check that all files have been uploaded and update the collection.

```python
# Finalize and commit the dataset
await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")
print("Dataset committed.")
```

### Step 6: Listing All Datasets in the Gallery

You can retrieve a list of all datasets in the collection to display on a webpage or for further processing.

```python
# List all datasets in the gallery
datasets = await artifact_manager.list(prefix="collections/dataset-gallery")
print("Datasets in the gallery:", datasets)
```

---

## Full Example: Creating and Managing a Dataset Gallery

Here’s a full example that shows how to connect to the service, create a dataset gallery, add a dataset, upload files with download statistics, and commit the dataset.

```python
import asyncio
import requests
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Dataset Manager API
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await api.get_service("public/artifact-manager")

    # Create a collection for the Dataset Gallery
    gallery_manifest = {
        "id": "dataset-gallery",
        "name": "Dataset Gallery",
        "description": "A collection for organizing datasets",
        "type": "collection",
        "collection": [],
    }
    # Create the collection with read permission for everyone and create permission for all authenticated users
    await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r+", "@": "r+"})
    print("Dataset Gallery created.")

    # Create a new dataset inside the Dataset Gallery
    dataset_manifest = {
        "id": "example-dataset",
        "name": "Example Dataset",
        "description": "A dataset with example data",
        "type": "dataset",
        "files": [],
    }
    await artifact_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)
    print("Dataset added to the gallery.")

    # Get a pre-signed URL to upload a file, with a download_weight assigned
    put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv", options={"download_weight": 0.5})

    # Upload the file using an HTTP PUT request
    with open("path/to/local/data.csv", "rb") as f:
        response = requests.put(put_url, data=f)
        assert response.ok, "File upload failed"
    print("File uploaded to the dataset.")

    # Finalize and commit the dataset
    await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")
    print("Dataset committed.")

    # List all datasets in the gallery
    datasets = await artifact_manager.list(prefix="collections/dataset-gallery")
    print("Datasets in the gallery:", datasets)

asyncio.run(main())
```

---

## Advanced Usage: Enforcing Schema for Dataset Validation

The `Artifact Manager` supports enforcing a schema to ensure that datasets conform to a specific structure. This can be useful in scenarios where you want to maintain consistency across multiple datasets.

### Step 1: Create a Collection with a Schema

You can define a schema for the collection that specifies the required fields for each dataset.

```python
# Define a schema for datasets in the gallery
dataset_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {"type": "string", "enum": ["dataset", "model", "application"]},
        "files": {
            "type": "array",
            "items": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
    "required": ["id", "name", "description", "type"]
}

# Create a collection with the schema
gallery_manifest = {
    "id": "schema-dataset-gallery",
    "name": "Schema Dataset Gallery",
    "description": "A gallery of datasets with enforced schema",
    "type": "collection",
    "collection_schema": dataset_schema,
    "collection": [],
}
# Create the collection with read permission for everyone and create permission for all authenticated users
await artifact_manager.create(prefix="collections/schema-dataset-gallery", manifest=gallery_manifest, permissions={"*": "r+", "@": "r+"})
print("Schema-based Dataset Gallery created.")
```

### Step 2: Validate Dataset Against Schema

When you commit a dataset to this collection, it will be validated against the schema.

```python
# Create a valid dataset that conforms to the schema
valid_dataset_manifest = {
    "id": "valid-dataset",
    "name": "Valid Dataset",
    "description": "A valid dataset conforming to the schema",
    "type": "dataset",
}

await artifact_manager.create(prefix="collections/schema-dataset-gallery/valid-dataset", manifest=valid_dataset_manifest, stage=True)
print("Valid dataset created.")

# Commit the valid dataset (this should pass schema validation)
await artifact_manager.commit(prefix="collections/schema-dataset-gallery/valid-dataset")
print("Valid dataset committed.")
```

---

## API References

### `create(prefix: str, manifest: dict, permissions: dict=None, stage: bool = False) -> None`

Creates a new artifact or collection with the specified manifest. The artifact is staged until committed. For collections, the `collection` field should be an empty list.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `manifest`: The manifest of the new artifact. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
- `stage`: Optional. A boolean flag to stage the artifact. Default is `False`.

**Example:**

```python
await artifact_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)
```

### Permissions

In the `Artifact Manager`, permissions allow users to control access to artifacts and collections. The system uses a flexible permission model that supports assigning different levels of access to individual users, authenticated users, or all users. Permissions are stored in the `permissions` field of an artifact manifest, and they define which operations a user can perform, such as reading, writing, creating, or committing changes.

Permissions can be set both at the artifact level and the workspace level. In the absence of explicit artifact-level permissions, workspace-level permissions are checked to determine access. Here's how permissions work in detail:

**Permission Levels:**

- **r**: Read-only access (includes `read`, `get_file`, `list_files`, `list`).
- **r+**: Read and create access (includes `read`, `get_file`, `list_files`, `list`, `put_file`, `create`, `commit`).
- **rw**: Read and write access (includes `read`, `get_file`, `list_files`, `list`, `edit`, `commit`, `put_file`, `remove_file`).
- **rw+**: Full access, including creation, editing, committing, and file removal (includes `read`, `get_file`, `list_files`, `list`, `edit`, `commit`, `put_file`, `remove_file`, `create`, `reset_stats`).
- **\***: Full access for all operations (includes all operations listed above).

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

### `edit(prefix: str, manifest: dict, permissions: dict = None) -> None`

Edits an existing artifact's manifest. The new manifest is staged until committed. The updated manifest is stored temporarily as `_manifest.yaml`.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `manifest`: The updated manifest. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.

**Example:**

```python
await artifact_manager.edit(prefix="collections/dataset-gallery/example-dataset", manifest=updated_manifest)
```

---

### `commit(prefix: str) -> None`

Finalizes and commits an artifact's staged changes. Validates uploaded files and renames `_manifest.yaml` to `manifest.yaml`. This process also updates view and download statistics.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")
```

---

### `delete(prefix: str) -> None`

Deletes an artifact, its manifest, and all associated files from both the database and S3 storage.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.delete(prefix="collections/dataset-gallery/example-dataset")
```

---

### `put_file(prefix: str, file_path: str, options: dict = None) -> str`

Generates a pre-signed URL to upload a file to the artifact in S3. The URL can be used with an HTTP `PUT` request to upload the file. The file is staged until the artifact is committed.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"data.csv"`).
- `options`: Optional. Additional options such as:
  - `download_weight`: A float value representing the file's impact on download count (0-1). Defaults to `None`.

**Returns:** A pre-signed URL for uploading the file.

**Example:**

```python
put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv", options={"download_weight": 1.0})
```

---

### `remove_file(prefix: str, file_path: str) -> None`

Removes a file from the artifact and updates the staged manifest. The file is also removed from the S3 storage.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `file_path`: The relative path of the file to be removed (e.g., `"data.csv"`).

**Example:**

```python
await artifact_manager.remove_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")
```

---

### `get_file(prefix: str, path: str, options: dict = None) -> str`

Generates a pre-signed URL to download a file from the artifact stored in S3.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `path`: The relative path of the file to download (e.g., `"data.csv"`).
- `options`: Optional. Controls for the download behavior, such as:
  - `silent`: A boolean to suppress the download count increment. Default is `False`.

**Returns:** A pre-signed URL for downloading the file.

**Example:**

```python
get_url = await artifact_manager.get_file(prefix="collections/dataset-gallery/example-dataset", path="data.csv")
```

---

### `list_files(prefix: str, dir_path: str=None) -> list`

Lists all files in the artifact.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `dir_path`: Optional. The directory path within the artifact to list files. Default is `None`.

**Returns:** A list of files in the artifact.

**Example:**

```python
files = await artifact_manager.list_files(prefix="collections/dataset-gallery/example-dataset")
```

---

### `read(prefix: str, stage: bool = False, silent: bool = False) -> dict`

Reads and returns the manifest of an artifact or collection. If in staging mode, reads from `_manifest.yaml`.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `stage`: Optional. If `True`, reads the `_manifest.yaml`. Default is `False`.
- `silent`: Optional. If `True`, suppresses the view count increment. Default is `False`.

**Returns:** The manifest as a dictionary, including view/download statistics and collection details (if applicable).

**Example:**

```python
manifest = await artifact_manager.read(prefix="collections/dataset-gallery/example-dataset")
```

---

### `list(prefix: str, keywords: list = None, filters: dict = None, mode: str = "AND", page: int = 0, page_size: int = 100) -> list`

List or search for artifacts within a collection based on keywords or filters, supporting both `AND` and `OR` modes.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `keywords`: Optional. A list of terms to search across all fields in the manifest.
- `filters`: Optional. A dictionary of key-value pairs for exact or fuzzy matching in specific fields.
- `mode`: Either `"AND"` or `"OR"` to combine conditions. Default is `"AND"`.
- `page`: Optional. The page number for paginated results. Default is `0`.
- `page_size`: Optional. The number of items per page. Default is `100`.

**Returns:** A list of matching artifacts with summary fields.

**Example:**

```python
results = await artifact_manager.list(prefix="collections/dataset-gallery", keywords=["example"], mode="AND")
```

---

### `reset_stats(prefix: str) -> None`

Resets the download and view counts for the artifact.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.reset_stats(prefix="collections/dataset-gallery/example-dataset")
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
    "id": "dataset-gallery",
    "name": "Dataset Gallery",
    "description": "A collection for organizing datasets",
    "type": "collection",
    "collection": [],
}
# Create the collection with read permission for everyone and create permission for all authenticated users
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r", "@": "r+"})

# Step 3: Add a dataset to the gallery
dataset_manifest = {
    "id": "example-dataset",
    "name": "Example Dataset",
    "description": "A dataset with example data",
    "type": "dataset",
    "files": [],
}
await artifact_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)

# Step 4: Upload a file to the dataset
put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")
with open("path/to/local/data.csv", "rb") as f:
    response = requests.put(put_url, data=f)
    assert response.ok, "File upload failed"

# Step 5: Commit the dataset
await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")

# Step 6: List all datasets in the gallery
datasets = await artifact_manager.list(prefix="collections/dataset-gallery")
print("Datasets in the gallery:", datasets)
```


### Resetting Download Statistics

You can reset the download statistics of a dataset using the `reset_stats` function.

```python
await artifact_manager.reset_stats(prefix="collections/dataset-gallery/example-dataset")
print("Download statistics reset.")
```

## HTTP API for Accessing Artifacts and Download Counts

The `Artifact Manager` provides an HTTP endpoint for retrieving artifact manifests, data, and download statistics. This is useful for public-facing web applications that need to access datasets, models, or applications.

### Endpoints:

 - `/{workspace}/artifacts/{prefix:path}` for fetching the artifact manifest.
 - `/{workspace}/artifacts/{prefix:path}/__children__` for listing all artifacts in a collection.
 - `/{workspace}/artifacts/{prefix:path}/__files__` for listing all files in the artifact.
 - `/{workspace}/artifacts/{prefix:path}/__files__/{file_path:path}` for downloading a file from the artifact (will be redirected to a pre-signed URL).

### Path Parameters:

- **workspace**: The workspace in which the artifact is stored.
- **prefix**: The relative prefix to the artifact. For private artifacts, it requires proper authentication by passing the user's token in the request headers.
- **file_path**: The relative path to a file within the artifact. This is optional and only required when downloading a file.

### Request Format:

- **Method**: `GET`
- **Headers**:
  - `Authorization`: Optional. The user's token for accessing private artifacts (obtained via the login logic or created by `api.generate_token()`). Not required for public artifacts.
- **Parameters**:
  - `workspace`: The workspace in which the artifact is stored.
  - `prefix`: The relative prefix to the artifact (e.g., `collections/dataset-gallery/example-dataset`). It should not contain the workspace ID.
  - `stage` (optional): A boolean flag to indicate whether to fetch the staged version of the manifest (`_manifest.yaml`). Default is `False`.
  - `silent` (optional): A boolean flag to suppress the view count increment. Default is `False`.

### Response:

For `/{workspace}/artifacts/{prefix:path}`, the response will be a JSON object representing the artifact manifest. For `/{workspace}/artifacts/{prefix:path}/__files__/{file_path:path}`, the response will be a pre-signed URL to download the file. The artifact manifest will also include any metadata such as download statistics in the `_metadata` field. For private artifacts, make sure if the user has the necessary permissions.

For `/{workspace}/artifacts/{prefix:path}/__children__`, the response will be a list of artifacts in the collection.

For `/{workspace}/artifacts/{prefix:path}/__files__`, the response will be a list of files in the artifact, each file is a dictionary with the `name` and `type` fields.

For `/{workspace}/artifacts/{prefix:path}/__files__/{file_path:path}`, the response will be a pre-signed URL to download the file.

### Example: Fetching a public artifact with download statistics

```python
import requests

SERVER_URL = "https://hypha.aicell.io"
workspace = "my-workspace"
response = requests.get(f"{SERVER_URL}/{workspace}/artifacts/collections/dataset-gallery/example-dataset")
if response.ok:
    artifact = response.json()
    print(artifact["name"])  # Output: Example Dataset
    print(artifact["_metadata"]["download_count"])  # Output: Download count for the dataset
else:
    print(f"Error: {response.status_code}")
```


