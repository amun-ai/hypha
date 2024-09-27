# Artifact Manager

The `Artifact Manager` is a builtin hypha service for indexing, managing, and storing resources such as datasets, AI models, and applications. It is designed to provide a structured way to manage datasets and similar resources, enabling efficient listing, uploading, updating, and deleting of files. 

A typical use case for the `Artifact Manager` is as a backend for a single-page web application displaying a gallery of datasets, AI models, applications or other type of resources. The default metadata of an artifact is designed to render a grid of cards on a webpage.

**Note:** The `Artifact Manager` is only available when your hypha server enabled s3 storage.

## Getting Started

### Step 1: Connecting to the Artifact Manager Service

To use the `Artifact Manager`, you first need to connect to the Hypha server. This API allows you to create, read, edit, and delete datasets in the artifact registry (stored in s3 bucket for each workspace).

```python
from hypha_rpc.websocket_client import connect_to_server

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

await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)
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

### Step 4: Uploading Files to the Dataset

Once you have created a dataset, you can upload files to it by generating a pre-signed URL.

```python
# Get a pre-signed URL to upload a file
put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")

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

Here’s a full example that shows how to connect to the service, create a dataset gallery, add a dataset, upload files, and commit the dataset.

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
    await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)
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

    # Get a pre-signed URL to upload a file
    put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")

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

await artifact_manager.create(prefix="collections/schema-dataset-gallery", manifest=gallery_manifest)
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

## API Reference

This section details the core functions provided by the `Artifact Manager` for creating, managing, and validating artifacts such as datasets and collections.

### `create(prefix: str, manifest: dict, overwrite: bool = False, stage: bool = False) -> dict`

Creates a new artifact or collection with the provided manifest. If the artifact already exists, you must set `overwrite=True` to overwrite it.

**Parameters:**

- `prefix`: The path where the artifact or collection will be created (e.g., `"collections/dataset-gallery"`).
- `manifest`: The manifest describing the artifact (must include fields like `id`, `name`, and `type`).
- `overwrite`: Optional. If `True`, it will overwrite an existing artifact. Default is `False`.
- `stage`: Optional. If `True`, it will put the artifact into staging mode. Default is `False`.

**Returns:** The created manifest as a dictionary.

**Example:**

```python
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)
```

---

### `edit(prefix: str, manifest: dict) -> None`

Edits an existing artifact. You provide the new manifest to update the artifact. The updated manifest is stored temporarily as `_manifest.yaml`.

**Parameters:**

- `prefix`: The path of the artifact to edit (e.g., `"collections/dataset-gallery/example-dataset"`).
- `manifest`: The updated manifest.

**Example:**

```python
await artifact_manager.edit(prefix="collections/dataset-gallery/example-dataset", manifest=updated_manifest)
```

---

### `commit(prefix: str) -> None`

Commits an artifact, finalizing it by validating the uploaded files and renaming `_manifest.yaml` to `manifest.yaml`.

**Parameters:**

- `prefix`: The path of the artifact to commit (e.g., `"collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.commit(prefix="collections/dataset-gallery/example-dataset")
```

---

### `delete(prefix: str) -> None`

Deletes an artifact and all associated files.

**Parameters:**

- `prefix`: The path of the artifact to delete (e.g., `"collections/dataset-gallery/example-dataset"`).

**Example:**

```python
await artifact_manager.delete(prefix="collections/dataset-gallery/example-dataset")
```

---

### `put_file(prefix: str, file_path: str) -> str`

Generates a pre-signed URL to upload a file to an artifact. You can then use the URL with an HTTP `PUT` request to upload the file.

**Parameters:**

- `prefix`: The path of the artifact where the file will be uploaded (e.g., `"collections/dataset-gallery/example-dataset"`).
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"data.csv"`).

**Returns:** A pre-signed URL for uploading the file.

**Example:**

```python
put_url = await artifact_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")
```

---

### `remove_file(prefix: str, file_path: str) -> None`

Removes a file from the artifact and updates the manifest.

**Parameters:**

- `prefix`: The path of the artifact from which the file will be removed (e.g., `"collections/dataset-gallery/example-dataset"`).
- `file_path`: The relative path of the file to be removed (e.g., `"data.csv"`).

**Example:**

```python
await artifact_manager.remove_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")
```

---

### `get_file(prefix: str, path: str) -> str`

Generates a pre-signed URL to download a file from the artifact.

**Parameters:**

- `prefix`: The path of the artifact (e.g., `"collections/dataset-gallery/example-dataset"`).
- `path`: The relative path of the file to download (e.g., `"data.csv"`).

**Returns:** A pre-signed URL for downloading the file.

**Example:**

```python
get_url = await artifact_manager.get_file(prefix="collections/dataset-gallery/example-dataset", path="data.csv")
```

---

### `list(prefix: str, stage: bool = False) -> list`

Lists all artifacts or collections under the specified prefix. If a collection is detected, it returns the list of artifacts within the collection.

**Parameters:**

- `prefix`: The path under which the artifacts or collections are listed (e.g., `"collections/dataset-gallery"`).
- `stage`: Optional. If `True`, it will list the artifacts in staging mode. Default is `False`.

**Returns:** A list of artifact names or collection items.

**Example:**

```python
datasets = await artifact_manager.list(prefix="collections/dataset-gallery")
```

---

### `read(prefix: str, stage: bool = False) -> dict`

Reads the manifest of an artifact or collection. If in staging mode, it reads the `_manifest.yaml`.

**Parameters:**

- `prefix`: The path of the artifact to read (e.g., `"collections/dataset-gallery/example-dataset"`).
- `stage`: Optional. If `True`, it reads from `_manifest.yaml`. Default is `False`.

**Returns:** The manifest as a dictionary.

**Example:**

```python
manifest = await artifact_manager.read(prefix="collections/dataset-gallery/example-dataset")
```

---

### `index(prefix: str) -> None`

Updates the index of a collection. It collects summary information for all artifacts in the collection and updates the manifest’s `collection` field.

**Parameters:**

- `prefix`: The path of the collection to index (e.g., `"collections/dataset-gallery"`).

**Example:**

```python
await artifact_manager.index(prefix="collections/dataset-gallery")
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
await artifact_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)

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
