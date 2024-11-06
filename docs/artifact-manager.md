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
    "name": "Dataset Gallery",
    "description": "A collection for organizing datasets",
    "type": "collection",
    "collection": [],
}

# Create the collection with read permission for everyone and create permission for all authenticated users
# We set orphan=True to create a collection without a parent
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r", "@": "r+"}, orphan=True)
print("Dataset Gallery created.")
```

### Step 3: Adding a Dataset to the Gallery

After creating the gallery, you can start adding datasets to it. Each dataset will have its own manifest that describes the dataset and any associated files.

```python
# Create a new dataset inside the Dataset Gallery
dataset_manifest = {
    "name": "Example Dataset",
    "description": "A dataset with example data",
    "type": "dataset",
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
        "name": "Dataset Gallery",
        "description": "A collection for organizing datasets",
        "type": "collection",
        "collection": [],
    }
    # Create the collection with read permission for everyone and create permission for all authenticated users
    # We set orphan=True to create a collection without a parent
    await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r+", "@": "r+"}, orphan=True)
    print("Dataset Gallery created.")

    # Create a new dataset inside the Dataset Gallery
    dataset_manifest = {
        "name": "Example Dataset",
        "description": "A dataset with example data",
        "type": "dataset",
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
        "name": {"type": "string"},
        "description": {"type": "string"},
        "type": {"type": "string", "enum": ["dataset", "model", "application"]},
    },
    "required": ["name", "description", "type"]
}

# Create a collection with the schema
gallery_manifest = {
    "name": "Schema Dataset Gallery",
    "description": "A gallery of datasets with enforced schema",
    "type": "collection",
}
# Create the collection with read permission for everyone and create permission for all authenticated users
await artifact_manager.create(prefix="collections/schema-dataset-gallery", manifest=gallery_manifest, config={"collection_schema": dataset_schema}, permissions={"*": "r+", "@": "r+"}, orphan=True)
print("Schema-based Dataset Gallery created.")
```

### Step 2: Validate Dataset Against Schema

When you commit a dataset to this collection, it will be validated against the schema.

```python
# Create a valid dataset that conforms to the schema
valid_dataset_manifest = {
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

### `create(prefix: str, manifest: dict, permissions: dict=None, config: dict=None, stage: bool = False, orphan: bool = False, publish_to: str = None) -> None`

Creates a new artifact or collection with the specified manifest. The artifact is staged until committed. For collections, the `collection` field should be an empty list.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`). To generate an auto-id, you can use patterns like `"{uuid}"` or `"{timestamp}"`. The following patterns are supported:
  - `{uuid}`: Generates a random UUID.
  - `{timestamp}`: Generates a timestamp in milliseconds.
  - `{zenodo_id}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo deposit ID (which changes when a new version is created).
  - `{zenodo_conceptrecid}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo concept id (which does not change when a new version is created).
  - **Id Parts**: You can also use id parts stored in the parent collection's config['id_parts'] to generate an id. For example, if the parent collection has `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`, you can use `"{colors}-{animals}"` to generate an id like `red-dog`.
- `manifest`: The manifest of the new artifact. Ensure the manifest follows the required schema if applicable (e.g., for collections). 

- `config`: Optional. A dictionary containing additional configuration options for the artifact (shared for both staged and committed). For collections, the config can contain the following special fields:
  - `collection_schema`: Optional. A JSON schema that defines the structure of child artifacts in the collection. This schema is used to validate child artifacts when they are created or edited. If a child artifact does not conform to the schema, the creation or edit operation will fail.
  - `summary_fields`: Optional. A list of fields to include in the summary for each child artifact when calling `list(prefix)`. If not specified, the default summary fields (`id`, `type`, `name`) are used. To include all the fields in the summary, add `"*"` to the list. If you want to include internal fields such as `.created_at`, `.last_modified`, or other download/view statistics such as `.download_count`, you can also specify them individually in the `summary_fields`. If you want to include all fields, you can add `".*"` to the list.
  - `id_parts`: Optional. A dictionary of id name parts to be used in generating the id for child artifacts. For example: `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`. This can be used for creating child artifacts with auto-generated ids based on the id parts. For example, when calling `create`, you can specify the prefix as `collections/my-collection/{colors}-{animals}`, and the id will be generated based on the id parts, e.g., `collections/my-collection/red-dog`.
  - `archives`: Optional. Configurations for the public archive servers such as Zenodo for the collection. For example, `"archives": {"sandbox_zenodo": {"access_token": "your sandbox zenodo token"}}, "zenodo": {"access_token": "your zenodo token"}}`. This is used for publishing artifacts to Zenodo.
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
- `stage`: Optional. A boolean flag to stage the artifact. Default is `False`. If it's set to `True`, the artifact will be staged and not committed. You will need to call `commit()` to finalize the artifact.
- `orphan`: Optional. A boolean flag to create the artifact without a parent collection. Default is `False`. If `True`, the artifact will not be associated with any collection. This is mainly used for creating top-level collections, and making sure the artifact is not associated with any parent collection (with inheritance of permissions). To create an orphan artifact, you will need write permissions on the workspace.
- `publish_to`: Optional. A string specifying the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. If set, the artifact will be published to the specified platform. The artifact must have a valid Zenodo metadata schema to be published.

**Note 1: If you set `stage=True`, you must call `commit()` to finalize the artifact.**

**Note 2: If you set `orphan=True`, the artifact will not be associated with any collection. An non-orphan artifact must have a parent collection.**

**Example:**

```python
# Assuming we have already created a dataset-gallery collection, otherwise create it first or set orphan=True
await artifact_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True, orphan=False)
```

### Permissions

In the `Artifact Manager`, permissions allow users to control access to artifacts and collections. The system uses a flexible permission model that supports assigning different levels of access to individual users, authenticated users, or all users. Permissions are stored in the `permissions` field of an artifact manifest, and they define which operations a user can perform, such as reading, writing, creating, or committing changes.

Permissions can be set both at the artifact level and the workspace level. In the absence of explicit artifact-level permissions, workspace-level permissions are checked to determine access. Here's how permissions work in detail:

**Permission Levels:**

- **n**: No access to the artifact.
- **l**: List-only access (includes `list`).
- **l+**: List and create access (includes `list`, `create`, `commit`).
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

### `edit(prefix: str, manifest: dict, permissions: dict = None, config: dict = None, stage: bool = False) -> None`

Edits an existing artifact's manifest. The new manifest is staged until committed.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `manifest`: The updated manifest. Ensure the manifest follows the required schema if applicable (e.g., for collections).
- `permissions`: Optional. A dictionary containing user permissions. For example `{"*": "r+"}` gives read and create access to everyone, `{"@": "rw+"}` allows all authenticated users to read/write/create, and `{"user_id_1": "r+"}` grants read and create permissions to a specific user. You can also set permissions for specific operations, such as `{"user_id_1": ["read", "create"]}`. See detailed explanation about permissions below.
- `config`: Optional. A dictionary containing additional configuration options for the artifact.
- `stage`: Optional. A boolean flag to stage the artifact. Default is `False`. If it's set to `True`, the artifact will be staged and not committed. You will need to call `commit()` to finalize the changes.

**Example:**

```python
await artifact_manager.edit(prefix="collections/dataset-gallery/example-dataset", manifest=updated_manifest)
```

---

### `commit(prefix: str) -> None`

Finalizes and commits an artifact's staged changes. Validates uploaded files and commit the staged manifest. This process also updates view and download statistics.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")
```

---

### `delete(prefix: str, delete_files: bool = False, recursive: bool = False) -> None`

Deletes an artifact, its manifest, and all associated files from both the database and S3 storage.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `delete_files`: Optional. A boolean flag to delete all files associated with the artifact. Default is `False`.
- `recursive`: Optional. A boolean flag to delete all child artifacts recursively. Default is `False`.

**Warning: If `delete_files` is set to `True`, `recursive` must be set to `True`, all child artifacts will be deleted, and all files associated with the child artifacts will be permanently deleted from the S3 storage. This operation is irreversible.**

**Example:**

```python
await artifact_manager.delete(prefix="collections/dataset-gallery/example-dataset", delete_files=True)
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

### `read(prefix: str, stage: bool = False, silent: bool = False, include_metadata: bool = False) -> dict`

Reads and returns the manifest of an artifact or collection. If in staging mode, reads the staged manifest.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `stage`: Optional. If `True`, reads the staged manifest. Default is `False`.
- `silent`: Optional. If `True`, suppresses the view count increment. Default is `False`.
- `include_metadata`: Optional. If `True`, includes metadata such as download statistics in the manifest (fields starting with `"."`). Default is `False`.

**Returns:** The manifest as a dictionary, including view/download statistics and collection details (if applicable).

**Example:**

```python
manifest = await artifact_manager.read(prefix="collections/dataset-gallery/example-dataset")
```

---

### `list(prefix: str, keywords: List[str] = None, filters: dict = None, mode: str = "AND", page: int = 0, page_size: int = 100, order_by: str = None, summary_fields: List[str] = None, silent: bool = False) -> list`

Retrieve a list of child artifacts within a specified collection, supporting keyword-based fuzzy search, field-specific filters, and flexible ordering. This function allows detailed control over the search and pagination of artifacts in a collection, including staged artifacts if specified.

**Parameters:**

- `prefix` (str): The path to the artifact, either as a relative prefix within the workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or as an absolute path including the workspace ID (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`). This defines the collection or artifact directory to search within.

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

- `page` (int, optional): The page number for pagination. Used in conjunction with `page_size` to limit results. Default is `0`, which returns the first page of results.

- `page_size` (int, optional): The maximum number of artifacts to return per page. This is capped at 1000 for performance considerations. Default is `100`.

- `order_by` (str, optional): The field used to order results. Options include:
  - `view_count`, `download_count`, `last_modified`, `created_at`, and `prefix`.
  - Use a suffix `<` or `>` to specify ascending or descending order, respectively (e.g., `view_count<` for ascending).
  - Default ordering is ascending by prefix if not specified.

- `summary_fields` (List[str], optional): A list of fields to include in the summary for each artifact. If not specified, it will use the `summary_fields` value from the parent collection, otherwise, the default summary fields (`id`, `type`, `name`) are used. You can add `"*"` to the list if you want to include all the manifest fields. If you want to include internal fields such as `.created_at`, `.last_modified`, or other download/view statistics such as `.download_count`, you can also specify them individually in the `summary_fields`. If you want to include all fields, you can add `".*"` to the list.

- `silent` (bool, optional): If `True`, prevents incrementing the view count for the parent artifact when listing children. Default is `False`.

**Returns:**
A list of artifacts that match the search criteria, each represented by a dictionary containing summary fields. Fields are specified in the `summary_fields` attribute of the parent artifact's manifest; if this attribute is undefined, default summary fields (`id`, `type`, `name`) are used.

**Example Usage:**

```python
# Retrieve artifacts in the 'dataset-gallery' collection, filtering by creator and keywords,
# with results ordered by descending view count.
results = await artifact_manager.list(
    prefix="collections/dataset-gallery",
    keywords=["example"],
    filters={"created_by": "user123", "stage": False},
    order_by="view_count>",
    mode="AND",
    page=1,
    page_size=50
)
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

### `publish(prefix: str, to: str = None) -> None`

Publishes the artifact to a specified platform, such as Zenodo. The artifact must have a valid Zenodo metadata schema to be published.

**Parameters:**

- `prefix`: The path of the artifact, it can be a prefix relative to the current workspace (e.g., `"collections/dataset-gallery/example-dataset"`) or an absolute prefix with the workspace id (e.g., `"/my_workspace_id/collections/dataset-gallery/example-dataset"`).
- `to`: Optional, the target platform to publish the artifact. Supported values are `zenodo` and `sandbox_zenodo`. This parameter should be the same as the `publish_to` parameter in the `create` function or left empty to use the platform specified in the artifact's metadata.

**Example:**

```python
await artifact_manager.publish(prefix="collections/dataset-gallery/example-dataset", to="sandbox_zenodo")
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
    "type": "collection",
    "collection": [],
}
# Create the collection with read permission for everyone and create permission for all authenticated users
# We set orphan=True to create a collection without a parent
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest, permissions={"*": "r", "@": "r+"}, orphan=True)

# Step 3: Add a dataset to the gallery
dataset_manifest = {
    "name": "Example Dataset",
    "description": "A dataset with example data",
    "type": "dataset",
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


### Request Format:

- **Method**: `GET`
- **Headers**:
  - `Authorization`: Optional. The user's token for accessing private artifacts (obtained via the login logic or created by `api.generate_token()`). Not required for public artifacts.

### Path Parameters:

The path parameters are used to specify the artifact or file to access. The following parameters are supported:

- **workspace**: The workspace in which the artifact is stored.
- **prefix**: The relative prefix to the artifact. For private artifacts, it requires proper authentication by passing the user's token in the request headers.
- **file_path**: Optional, the relative path to a file within the artifact. This is optional and only required when downloading a file.

### Query Parameters:

Qury parameters are passed after the `?` in the URL and are used to control the behavior of the API. The following query parameters are supported:

- **stage**: A boolean flag to fetch the staged version of the manifest. Default is `False`.
- **silent**: A boolean flag to suppress the view count increment. Default is `False`.
- **include_metadata**: A boolean flag to include metadata such as download statistics in the manifest. Default is `False`.

- **keywords**: A list of search terms used for fuzzy searching across all manifest fields, separated by commas.
- **filters**: A dictionary of filters to apply to the search, in the format of a JSON string.
- **mode**: The mode for combining multiple conditions. Default is `AND`.
- **page**: The page number for pagination. Default is `0`.
- **page_size**: The maximum number of artifacts to return per page. Default is `100`.
- **order_by**: The field used to order results. Default is ascending by prefix.
- **summary_fields**: A list of fields to include in the summary for each artifact when listing children. Default to the `summary_fields` value from the parent collection, otherwise, the default summary fields (`id`, `type`, `name`) are used. To include all manifest fields, add `"*"` to the list, and to include all internal fields, add `".*"` to the list.
- **silent**: A boolean flag to prevent incrementing the view count for the parent artifact when listing children, listing files, or reading the artifact. Default is `False`.

### Response:

For `/{workspace}/artifacts/{prefix:path}`, the response will be a JSON object representing the artifact manifest. For `/{workspace}/artifacts/{prefix:path}/__files__/{file_path:path}`, the response will be a pre-signed URL to download the file. The artifact manifest will also include any metadata such as download statistics under keys starting with dot (`.`), e.g. `.view_count`, `.download_count`. For private artifacts, make sure if the user has the necessary permissions.

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
    print(artifact[".download_count"])  # Output: Download count for the dataset
else:
    print(f"Error: {response.status_code}")
```

