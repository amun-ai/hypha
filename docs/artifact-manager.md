# Artifact Manager

The `Artifact Manager` is a tool for indexing, managing, and storing resources such as datasets, AI models, and applications. It is designed to provide a structured way to manage datasets and similar resources, enabling efficient listing, uploading, updating, and deleting of files. 

A typical use case for the `Artifact Manager` is as a backend for a single-page web application displaying a gallery of datasets, AI models, or applications. The default metadata of an artifact is designed to render a grid of cards on a webpage.

**Note:** The `Artifact Manager` is only available when your hypha server enabled s3 storage.

## Getting Started

### Step 1: Connecting to the Dataset Manager Service

To use the `Artifact Manager`, you first need to connect to the WebSocket API. This API allows you to create, read, edit, and delete datasets in the gallery.

```python
from hypha_rpc.websocket_client import connect_to_server

SERVER_URL = "https://hypha.aicell.io"  # Replace with your server URL

# Connect to the Dataset Manager API
api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
dataset_manager = await api.get_service("public/artifact-manager")
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

await dataset_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)
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

await dataset_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)
print("Dataset added to the gallery.")
```

### Step 4: Uploading Files to the Dataset

Once you have created a dataset, you can upload files to it by generating a pre-signed URL.

```python
# Get a pre-signed URL to upload a file
put_url = await dataset_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")

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
await dataset_manager.commit(prefix="collections/dataset-gallery/example-dataset")
print("Dataset committed.")
```

### Step 6: Listing All Datasets in the Gallery

You can retrieve a list of all datasets in the collection to display on a webpage or for further processing.

```python
# List all datasets in the gallery
datasets = await dataset_manager.list(prefix="collections/dataset-gallery")
print("Datasets in the gallery:", datasets)
```

---

## Full Example: Creating and Managing a Dataset Gallery

Here’s a full example that shows how to connect to the service, create a dataset gallery, add a dataset, upload files, and commit the dataset.

```python
import asyncio
import requests
from hypha_rpc.websocket_client import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Dataset Manager API
    api = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    dataset_manager = await api.get_service("public/artifact-manager")

    # Create a collection for the Dataset Gallery
    gallery_manifest = {
        "id": "dataset-gallery",
        "name": "Dataset Gallery",
        "description": "A collection for organizing datasets",
        "type": "collection",
        "collection": [],
    }
    await dataset_manager.create(prefix="collections/dataset-gallery", manifest=gallery_manifest)
    print("Dataset Gallery created.")

    # Create a new dataset inside the Dataset Gallery
    dataset_manifest = {
        "id": "example-dataset",
        "name": "Example Dataset",
        "description": "A dataset with example data",
        "type": "dataset",
        "files": [],
    }
    await dataset_manager.create(prefix="collections/dataset-gallery/example-dataset", manifest=dataset_manifest, stage=True)
    print("Dataset added to the gallery.")

    # Get a pre-signed URL to upload a file
    put_url = await dataset_manager.put_file(prefix="collections/dataset-gallery/example-dataset", file_path="data.csv")

    # Upload the file using an HTTP PUT request
    with open("path/to/local/data.csv", "rb") as f:
        response = requests.put(put_url, data=f)
        assert response.ok, "File upload failed"
    print("File uploaded to the dataset.")

    # Finalize and commit the dataset
    await dataset_manager.commit(prefix="collections/dataset-gallery/example-dataset")
    print("Dataset committed.")

    # List all datasets in the gallery
    datasets = await dataset_manager.list(prefix="collections/dataset-gallery")
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

await dataset_manager.create(prefix="collections/schema-dataset-gallery", manifest=gallery_manifest)
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

await dataset_manager.create(prefix="collections/schema-dataset-gallery/valid-dataset", manifest=valid_dataset_manifest, stage=True)
print("Valid dataset created.")

# Commit the valid dataset (this should pass schema validation)
await dataset_manager.commit(prefix="collections/schema-dataset-gallery/valid-dataset")
print("Valid dataset committed.")
```
