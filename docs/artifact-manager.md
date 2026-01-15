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

Each dataset can contain multiple files. Use pre-signed URLs for secure uploads and set `download_weight` to track file downloads. You can later modify download weights using the `set_download_weight` function.

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

### Step 6: Managing Download Weights (Optional)

After committing, you can modify download weights for individual files to better track important downloads.

```python
# Update download weight for an important file
await artifact_manager.set_download_weight(
    artifact_id=dataset.id,
    file_path="data.csv", 
    download_weight=2.0  # Increase weight for this important file
)

# Remove download weight for auxiliary files to save storage
await artifact_manager.set_download_weight(
    artifact_id=dataset.id,
    file_path="readme.txt",
    download_weight=0  # This file won't contribute to download stats
)
print("Download weights updated.")
```

### Step 7: Listing All Datasets in the Gallery

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

## Staging and Draft Mode

The Artifact Manager provides a powerful staging system that allows you to create and modify artifacts in a draft state before committing them permanently. This enables review workflows, quality control, and collaborative content creation.

### Understanding Draft vs Committed Artifacts

1. **Draft (Staged) Artifacts**: 
   - Created with `stage=True` parameter
   - Temporary and can be modified multiple times before committing
   - Not visible in normal artifact listings unless explicitly requested
   - Can be reviewed and approved before becoming permanent
   - Automatically cleaned up if not committed within a certain timeframe

2. **Committed Artifacts**:
   - Permanent artifacts in the system
   - Visible in artifact listings
   - Cannot be deleted by non-owners (only with proper permissions)
   - Form part of the collection's permanent record

### Creating Draft Artifacts

```python
# Create a draft artifact for review
draft_artifact = await artifact_manager.create(
    parent_id="collection-id",
    alias="my-draft",
    manifest={"name": "Draft Content", "description": "For review"},
    stage=True  # This creates a draft
)

# The draft can be modified multiple times
await artifact_manager.edit(
    draft_artifact.id,
    manifest={"name": "Updated Draft", "description": "Ready for review"}
)

# Add files to the draft
put_url = await artifact_manager.put_file(draft_artifact.id, "data.csv")
# Upload file to put_url...
```

### Review Workflow Example

```python
# Step 1: Contributor creates a draft (requires 'draft' permission)
draft = await artifact_manager.create(
    parent_id="curated-collection",
    alias="contribution-1",
    manifest={"name": "New Dataset", "author": "contributor"},
    stage=True
)

# Step 2: Reviewer lists drafts for review
drafts = await artifact_manager.list("curated-collection", stage=True)

# Step 3: Reviewer examines the draft
draft_content = await artifact_manager.read(draft.id)

# Step 4: Reviewer approves and commits (requires 'attach' permission)
await artifact_manager.commit(draft.id)
```

### Permission-Based Staging Workflows

Different permission levels enable different staging workflows:

1. **Open Contribution** (`"@": "r+"`):
   - Any authenticated user can create and commit drafts
   - Suitable for open collaborative projects

2. **Moderated Contribution** (`"@": "l+", "reviewer": "rw+"`):
   - Authenticated users can only create drafts
   - Designated reviewers can commit drafts
   - Ensures quality control

3. **Restricted Contribution** (`"contributor": "l+", "reviewer": "rw+"`):
   - Only specific contributors can create drafts
   - Reviewers manage the approval process
   - Maximum control over content

### Best Practices for Staging

1. **Use Staging for User-Generated Content**: Always stage user contributions before making them permanent
2. **Implement Review Processes**: Use permission separation to ensure proper review
3. **Set Expiration for Drafts**: Configure automatic cleanup of uncommitted drafts
4. **Version Control**: Use staging with version management for controlled updates
5. **Atomic Operations**: Commit related changes together for consistency

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

### `create(parent_id: str, alias: str, artifact_id: str, workspace: str, type: str, manifest: dict, permissions: dict=None, config: dict=None, version: str = None, stage: bool = False, comment: str = None, overwrite: bool = False) -> None`

Creates a new artifact or collection with the specified manifest. The artifact is staged until committed. For collections, the `collection` field should be an empty list.

**Parameters:**

- `alias`: A human readable name for indexing the artifact. It should contain only lowercase letters, numbers, and hyphens (no slashes allowed in the alias itself). Examples: `"my-dataset"`, `"model-v2"`.
  To generate an auto-id, you can use patterns like `"{uuid}"` or `"{timestamp}"`. The following patterns are supported:
  - `{uuid}`: Generates a random UUID.
  - `{timestamp}`: Generates a timestamp in milliseconds.
  - `{user_id}`: Generate the user id of the creator.
  - `{zenodo_id}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo deposit ID (which changes when a new version is created).
  - `{zenodo_conceptrecid}`: If `publish_to` is set to `zenodo` or `sandbox_zenodo`, it will use the Zenodo concept id (which does not change when a new version is created).
  - **Id Parts**: You can also use id parts stored in the parent collection's config['id_parts'] to generate an id. For example, if the parent collection has `{"animals": ["dog", "cat", ...], "colors": ["red", "blue", ...]}`, you can use `"{colors}-{animals}"` to generate an id like `red-dog`.

- `artifact_id`: Optional. The full artifact ID in the format `"workspace/alias"`. This is an alternative to providing `alias` and `workspace` separately. When specified:
  - Cannot be used together with `alias` or `workspace` parameters
  - Must contain exactly one slash separating workspace and alias
  - The workspace portion must match the parent's workspace if a parent is specified
  - Example: `artifact_id="ri-scale/model-example1"`

- `workspace`: Optional. The workspace where the artifact will be created. 
  **Workspace Resolution Rules:**
  1. If `artifact_id` is provided, the workspace is extracted from it and this parameter is ignored
  2. If a `parent_id` is provided, the artifact inherits the parent's workspace (this parameter is ignored)
  3. If no parent and no `artifact_id`, uses this parameter or defaults to the current user's workspace
  4. When specified, it must match the parent's workspace if a parent exists

- `parent_id`: The id of the parent collection where the artifact will be created. If the artifact is a top-level collection, leave this field empty or set to None.
  **Important:** When a parent is specified, the child artifact is ALWAYS created in the parent's workspace, regardless of the user's current workspace.
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
# Method 1: Use the returned artifact ID (recommended for dynamic IDs)
draft = await artifact_manager.create(
    parent_id="dataset-gallery", 
    alias="example-dataset", 
    manifest=dataset_manifest, 
    stage=True
)
# Use the returned ID for subsequent operations
put_url = await artifact_manager.put_file(artifact_id=draft['id'], file_path="data.csv")

# Method 2: Specify the full artifact_id upfront (cleaner for known IDs)
await artifact_manager.create(
    parent_id="ri-scale/dataset-gallery",
    artifact_id="ri-scale/example-dataset",  # Full ID specified
    manifest=dataset_manifest,
    stage=True
)
# Use the same full ID for subsequent operations
put_url = await artifact_manager.put_file(artifact_id="ri-scale/example-dataset", file_path="data.csv")

# IMPORTANT: When creating a child artifact in a parent collection:
# - The artifact is created in the PARENT's workspace, not your current workspace
# - Always use the full ID (workspace/alias) for subsequent operations
# - Never use just the alias if the parent is in a different workspace
```

## Permissions and Access Control

The Artifact Manager provides a comprehensive permission system that enables fine-grained access control over artifacts and collections. This system balances security with flexibility, allowing both strict isolation for sensitive data and open collaboration for public resources.

### Core Concepts

#### Permission Hierarchy

The permission system follows a clear hierarchy that determines how access is granted:

1. **Workspace Owner Supremacy**
   - Workspace owners always have ultimate control over ALL resources in their workspace
   - They can bypass collection-level restrictions through workspace permission fallback
   - This is a deliberate design choice to prevent owners from being locked out of their own data
   - Example: Even if a collection is set to read-only, the workspace owner can still modify it

2. **Collection-Level Isolation**
   - For non-owners, collections provide strict access boundaries
   - Collection permissions completely control what external users can do
   - Non-owners CANNOT bypass collection permissions through workspace access
   - This ensures proper data isolation between different user groups

3. **Permission Resolution Order**
   - First: Check explicit artifact-level permissions
   - Second: Check parent collection permissions (inheritance)
   - Third: For workspace owners ONLY, check workspace-level permissions as fallback
   - This order ensures predictable permission behavior

#### Key Permission Operations

The system includes four critical operations that enable sophisticated access control:

| Operation | Purpose | Use Case |
|-----------|---------|----------|
| **draft** | Create staged artifacts in a collection | Users submit content for review |
| **attach** | Commit drafts to make them permanent | Reviewers approve submissions |
| **detach** | Remove artifacts from their parent | Curators reorganize collections |
| **mutate** | Modify already-committed artifacts | Maintainers update existing content |

### Permission Levels

Permission levels are shortcuts that expand to sets of allowed operations:

| Level | Description | Key Operations | Use Case |
|-------|-------------|----------------|----------|
| **n** | No access | None | Explicitly deny access |
| **l** | List only | `list` | Browse collections without reading content |
| **l+** | List + Draft | `list`, `draft` | Submit content for review |
| **r** | Read only | `read`, `list`, `get_file` | View content without modifications |
| **r+** | Read + Contribute | All read ops + `draft`, `attach` | Full contribution rights |
| **rd+** | Read + Draft only | All read ops + `draft`, `edit` | Create drafts but can't commit |
| **rw** | Read-Write | All r+ ops + `mutate`, `detach` | Manage existing content |
| **rw+** | Full Access | All rw ops + `create` | Complete content control |
| **\*** | Admin | All operations | Full administrative access |

**Special List Levels** (for specific content types):
- **lv/lv+**: List vectors (for vector databases)
- **lf/lf+**: List files (for file collections)

### User Specification

Permissions can be assigned to different user categories:

| Notation | Applies To | Example | Description |
|----------|------------|---------|-------------|
| **\*** | All users (public) | `{"*": "r"}` | Everyone can read |
| **@** | Authenticated users | `{"@": "rw"}` | Logged-in users can read/write |
| **user_id** | Specific user | `{"alice": "rw+"}` | Alice has full access |

**Priority Rule**: More specific permissions override general ones:
- User-specific > Authenticated users > All users
- Example: If `{"*": "r", "alice": "n"}`, Alice has no access despite public read

### Common Use Cases and Recommendations

#### 1. Public Repository Accepting Contributions

**Use Case**: Open-source datasets or models where anyone can contribute

```json
{
  "permissions": {
    "*": "r",      // Everyone can read
    "@": "r+"      // Authenticated users can contribute
  }
}
```

**Why**: Allows public viewing while requiring login for contributions, preventing spam

#### 2. Moderated Community Collection

**Use Case**: Curated collections requiring quality control

```json
{
  "permissions": {
    "*": "r",                    // Public can view
    "@": "l+",                   // Users submit drafts
    "moderator_team": "rw+",     // Moderators approve/reject
    "owner": "*"                 // Owner has full control
  }
}
```

**Why**: Creates a review workflow ensuring quality while encouraging contributions

#### 3. Immutable Publications Archive

**Use Case**: Published papers, official releases, permanent records

```json
{
  "permissions": {
    "*": "r",                    // Public read access
    "archive_admin": "*"         // Only admin can manage
    // No write/mutate permissions for others
  }
}
```

**Why**: Ensures content integrity - once published, cannot be modified

#### 4. Team Collaboration Workspace

**Use Case**: Internal team projects with external visibility

```json
{
  "permissions": {
    "*": "l",                    // Public can browse
    "team_member_1": "rw+",      // Team has full access
    "team_member_2": "rw+",
    "external_reviewer": "r"     // Reviewer can only read
  }
}
```

**Why**: Balances transparency with controlled access

#### 5. Staging Environment for Production

**Use Case**: Test changes before deploying to production

```json
{
  "permissions": {
    "*": "n",                    // No public access to staging
    "developers": "rd+",         // Devs can draft but not deploy
    "release_manager": "rw+",    // Manager approves deployments
    "ci_bot": "r+"              // CI can create and attach artifacts
  }
}
```

**Why**: Enforces deployment approval process

#### 6. Personal Sandbox with Sharing

**Use Case**: Individual workspace with selective sharing

```json
{
  "permissions": {
    "*": "n",                    // Private by default
    "owner": "*",                // Owner has full control
    "collaborator_1": "rw",      // Specific people can edit
    "viewer_1": "r"              // Some can only view
  }
}
```

**Why**: Maintains privacy while enabling collaboration

#### 7. Append-Only Audit Log

**Use Case**: Compliance logs, transaction records

```json
{
  "permissions": {
    "*": "n",                    // No public access
    "auditor": "r",              // Auditors can read
    "system": "r+",              // System can add new entries
    // No mutate or detach permissions
  }
}
```

**Why**: Ensures audit trail integrity - entries can be added but never modified/deleted

### Best Practices

1. **Start Restrictive**: Begin with minimal permissions and add as needed
2. **Use Draft Mode**: For user-generated content, always use draft/attach workflow
3. **Separate Concerns**: Different permissions for reading, contributing, and managing
4. **Document Permissions**: Include comments explaining permission choices
5. **Regular Audits**: Periodically review and update permissions
6. **Test Permissions**: Verify permissions work as expected before going live

### Technical Notes

**Permission Inheritance**: Artifacts inherit permissions from their parent collection if not explicitly defined. This cascades through the collection hierarchy.

**Artifact Creator Rights**: Creators automatically receive admin (`*`) permission on artifacts they create, ensuring they maintain control over their content.

**Backward Compatibility**: Legacy permission codes are automatically mapped:

- `create` → `draft` + `attach`
- Existing permission levels remain functional

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

### `add_vectors(artifact_id: str, vectors: list, update: bool = False, context: dict = None) -> None`

Adds vectors to a vector collection artifact. See [RAG](./rag.md) for more detailed use of vector collections.

**Parameters:**

- `artifact_id`: The ID of the artifact to which vectors will be added. This must be a vector-collection artifact.
- `vectors`: A list of vectors to add to the collection.
- `update`: (Optional) A boolean flag to update existing vectors if they have the same ID. Defaults to `False`. If update is set to `True`, the existing vectors will be updated with the new vectors and every vector are expected to have an `id` field. If the vector does not exist, it will be added.
- `context`: A dictionary containing user and session context information.

**Returns:** None.

**Example:**

```python
await artifact_manager.add_vectors(artifact_id="example-id", vectors=[{"id": 1, "vector": [0.1, 0.2]}])
```

---

### `search_vectors(artifact_id: str, query: Optional[Dict[str, Any]] = None, filters: Optional[dict[str, Any]] = None, limit: Optional[int] = 5, offset: Optional[int] = 0, return_fields: Optional[List[str]] = None, order_by: Optional[str] = None, pagination: Optional[bool] = False, context: dict = None) -> list`

Searches vectors in a vector collection artifact based on a query.

**Parameters:**

- `artifact_id`: The ID of the artifact to search within. This must be a vector-collection artifact.
- `query`: (Optional) A dictionary representing the query vector or conditions.
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

### `put_file(artifact_id: str, file_path: Union[str, List[str]], download_weight: Union[float, List[float]] = 0, use_proxy: bool = None, use_local_url: Union[bool, str] = False, expires_in: float = 3600) -> Union[str, Dict[str, str]]`

Generates pre-signed URL(s) to upload file(s) to the artifact in S3. Supports both single file and batch operations. URLs can be used with HTTP `PUT` requests to upload files.

**Batch Performance:** Batch operations provide **75-355x speedup** compared to sequential operations by using a single transaction, single S3 client, and single S3 state save.

**File Placement Optimization:** Files are placed directly in their final destination based on version intent:
- If the artifact has `new_version` intent (set via `version="new"` during edit), files are uploaded to the new version location
- Otherwise, files are uploaded to the existing latest version location
- This eliminates the need for expensive file copying during commit operations

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path(s) of the file(s) to upload within the artifact. Can be:
  - A single string (e.g., `"data.csv"`) for single file upload
  - A list of strings (e.g., `["file1.csv", "file2.csv", "file3.csv"]`) for batch upload
- `download_weight`: A float value or list of floats representing each file's impact on download count when downloaded via any endpoint. Can be:
  - A single float applied to all files. Defaults to `0`
  - A list of floats matching the length of `file_path` list
  - Files with weight `0` don't increment download count when accessed
- `use_proxy`: A boolean to control whether to use the S3 proxy for the generated URL. If `None` (default), follows the server configuration. If `True`, forces the use of the proxy. If `False`, bypasses the proxy and returns a direct S3 URL.
- `use_local_url`: A boolean or string to control whether to generate URLs for local/cluster-internal access. Defaults to `False`. It can be a specified url, which will be used to replace the server url. When `True`, generates URLs suitable for access within the cluster:
  - **With proxy (`use_proxy=True`)**: Uses `local_base_url/s3` instead of the public proxy URL for cluster-internal access
  - **Direct S3 (`use_proxy=False`)**: Uses the S3 endpoint URL as-is (already configured for local access)
  - **Use case**: Useful when services within a cluster need to access files but public URLs are not accessible from within the cluster network
- `expires_in`: A float number for the expiration time of the S3 presigned url

**Returns:**

- For single file: A pre-signed URL string
- For batch operation: A dictionary mapping file paths to their presigned URLs

**Important Note:** Adding files to staging does NOT automatically create new version intent. You must explicitly use `version="new"` during edit operations if you want to create a new version when committing.

**Example:**

```python
put_url = await artifact_manager.put_file(artifact.id, file_path="data.csv", download_weight=1.0)

# If "example-dataset" is an alias of the artifact under the current workspace
put_url = await artifact_manager.put_file(artifact_id="example-dataset", file_path="data.csv")

# If "example-dataset" is an alias of the artifact under another workspace
put_url = await artifact_manager.put_file(artifact_id="other_workspace/example-dataset", file_path="data.csv")

# IMPORTANT: When working with drafts created in another workspace's collection:
# Always use the full ID returned by create(), not just the alias
draft = await artifact_manager.create(
    parent_id="other_workspace/collection",  # Parent in different workspace
    alias="my-draft",
    stage=True
)
# Use the full ID from draft['id'], which will be "other_workspace/my-draft"
put_url = await artifact_manager.put_file(artifact_id=draft['id'], file_path="data.csv")

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

# Batch upload multiple files (75-355x faster than sequential)
file_paths = ["data1.csv", "data2.csv", "data3.csv"]
put_urls = await artifact_manager.put_file(artifact_id=artifact.id, file_path=file_paths)

# put_urls is now a dict: {"data1.csv": "url1", "data2.csv": "url2", "data3.csv": "url3"}
for path, url in put_urls.items():
    with open(f"path/to/local/{path}", "rb") as f:
        response = requests.put(url, data=f)
        assert response.ok, f"File upload failed for {path}"
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

### `get_file(artifact_id: str, file_path: Union[str, List[str]], silent: bool = False, version: str = None, use_proxy: bool = None, use_local_url: Union[bool, str] = False, expires_in: float = 3600) -> Union[str, Dict[str, str]]`

Generates pre-signed URL(s) to download file(s) from the artifact stored in S3. Supports both single file and batch operations.

**Batch Performance:** Batch operations provide **75-355x speedup** compared to sequential operations by using a single transaction, single S3 client, and single S3 state save.

**Parameters:**

- `artifact_id`: The id of the artifact to download the file from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path(s) of the file(s) to download. Can be:
  - A single string (e.g., `"data.csv"`) for single file download
  - A list of strings (e.g., `["file1.csv", "file2.csv", "file3.csv"]`) for batch download
- `silent`: A boolean to suppress the download count increment. Default is `False`.
- `version`: The version of the artifact to download the file from. By default, it downloads from the latest version. If you want to download from a staged version, you can set it to `"stage"`.
- `use_proxy`: A boolean to control whether to use the S3 proxy for the generated URL. If `None` (default), follows the server configuration. If `True`, forces the use of the proxy. If `False`, bypasses the proxy and returns a direct S3 URL.
- `use_local_url`: A boolean or str to control whether to generate URLs for local/cluster-internal access. Defaults to `False`. It can be specified as a string to replace the actual server url. When `True`, generates URLs suitable for access within the cluster:
  - **With proxy (`use_proxy=True`)**: Uses `local_base_url/s3` instead of the public proxy URL for cluster-internal access
  - **Direct S3 (`use_proxy=False`)**: Uses the S3 endpoint URL as-is (already configured for local access)
  - **Use case**: Useful when services within a cluster need to access files but public URLs are not accessible from within the cluster network
- `expires_in`: A float number for the expiration time of the S3 presigned url

**Returns:**

- For single file: A pre-signed URL string
- For batch operation: A dictionary mapping file paths to their presigned URLs

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

# Batch download multiple files (75-355x faster than sequential)
file_paths = ["data1.csv", "data2.csv", "data3.csv"]
get_urls = await artifact_manager.get_file(artifact_id=artifact.id, file_path=file_paths)

# get_urls is now a dict: {"data1.csv": "url1", "data2.csv": "url2", "data3.csv": "url3"}
for path, url in get_urls.items():
    response = requests.get(url)
    with open(f"downloaded_{path}", "wb") as f:
        f.write(response.content)
```

---

### `put_file_start_multipart(artifact_id: str, file_path: str, part_count: int, expires_in: int = 3600, use_proxy: bool = False, use_local_url: Union[bool, str] = False) -> dict`

Initiates a multipart upload for large files and generates pre-signed URLs for uploading each part. This is useful for files larger than 100MB or when you need to upload files in chunks with parallel processing.

**File Placement Optimization:** Like `put_file`, files are placed directly in their final destination based on version intent:
- If the artifact has `new_version` intent (set via `version="new"` during edit), files are uploaded to the new version location
- Otherwise, files are uploaded to the existing latest version location

**Parameters:**

- `artifact_id`: The id of the artifact to upload the file to. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file to upload within the artifact (e.g., `"large_dataset.zip"`).
- `part_count`: The number of parts to split the file into. Must be between 1 and 10,000. Each part (except the last) must be at least 5MB in size.
- `expires_in`: The expiration time in seconds for the multipart upload session and part URLs. Defaults to 3600 (1 hour).
- `use_proxy`: A boolean to control whether to use the S3 proxy for the generated URLs. If `None` (default), follows the server configuration. If `True`, forces the use of the proxy. If `False`, bypasses the proxy and returns direct S3 URLs.
- `use_local_url`: A boolean or string to control whether to generate URLs for local/cluster-internal access. Defaults to `False`. A URL string can be specified to replace the actual server url. When `True`, generates URLs suitable for access within the cluster:
  - **With proxy (`use_proxy=True`)**: Uses `local_base_url/s3` instead of the public proxy URL for cluster-internal access
  - **Direct S3 (`use_proxy=False`)**: Uses the S3 endpoint URL as-is (already configured for local access)
  - **Use case**: Useful when services within a cluster need to access files but public URLs are not accessible from within the cluster network

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

# Start multipart upload with proxy bypassed (useful for direct S3 access)
multipart_info = await artifact_manager.put_file_start_multipart(
    artifact_id="example-dataset",
    file_path="large_video.mp4", 
    part_count=5,
    use_proxy=False
)

# Start multipart upload using local/cluster-internal URLs
multipart_info = await artifact_manager.put_file_start_multipart(
    artifact_id="example-dataset",
    file_path="large_video.mp4", 
    part_count=5,
    use_local_url=True
)

# Start multipart upload using local proxy URL for cluster-internal access
multipart_info = await artifact_manager.put_file_start_multipart(
    artifact_id="example-dataset",
    file_path="large_video.mp4", 
    part_count=5,
    use_proxy=True,
    use_local_url=True
)

upload_id = multipart_info["upload_id"]
part_urls = multipart_info["parts"]

# Complete example: Upload actual file data in parts
import httpx
import asyncio
import os

# Assume we have a large file to upload
local_file_path = "large_video.mp4"
file_size = os.path.getsize(local_file_path)
part_size = file_size // len(part_urls)  # Calculate size per part

async def upload_part(part_info):
    part_number = part_info["part_number"]
    url = part_info["url"]
    
    # Calculate start and end positions for this part
    start_pos = (part_number - 1) * part_size
    end_pos = min(start_pos + part_size, file_size)
    
    # Read the specific part from the file
    with open(local_file_path, "rb") as f:
        f.seek(start_pos)
        part_data = f.read(end_pos - start_pos)
    
    # Upload the part
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.put(url, data=part_data)
        assert response.status_code == 200, f"Failed to upload part {part_number}"
        return {
            "part_number": part_number,
            "etag": response.headers["ETag"].strip('"')
        }

# Upload all parts concurrently
uploaded_parts = await asyncio.gather(*[
    upload_part(part) for part in part_urls
])

# Now complete the multipart upload
result = await artifact_manager.put_file_complete_multipart(
    artifact_id="example-dataset",
    upload_id=upload_id,
    parts=uploaded_parts
)
print(f"Upload completed: {result['success']}")
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
- Each part (except the last) must be at least 5MB in size
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

### `list_files(artifact_id: str, dir_path: str=None, limit: int = 1000, offset: int = 0, version: str = None, stage: bool = False, include_pending: bool = False) -> list`

Lists all files in the artifact with pagination support.

**Parameters:**

- `artifact_id`: The id of the artifact to list files from. It can be an uuid generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `dir_path`: Optional. The directory path within the artifact to list files. Default is `None`.
- `limit`: Optional. Maximum number of files to return. Used for pagination. Default is 1000. Range: 1-10000.
- `offset`: Optional. Number of files to skip before returning results. Used for pagination. Default is 0.
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

# List files with pagination (first page of 100 items)
page1 = await artifact_manager.list_files(
    artifact_id="large-dataset",
    limit=100,
    offset=0
)

# List next page
page2 = await artifact_manager.list_files(
    artifact_id="large-dataset",
    limit=100,
    offset=100
)

# Example response with pending files
[
    {"name": "uploaded_file.csv", "type": "file", "size": 1024, "pending": False},
    {"name": "pending_file.txt", "type": "file", "size": 0, "pending": True}
]
```

---

### `read_file(artifact_id: str, file_path: str, format: str = "text", encoding: str = "utf-8", offset: int = 0, limit: int = None, version: str = None, stage: bool = False) -> dict`

Reads file content directly from an artifact with flexible encoding options. This method reads file content without generating presigned URLs, making it convenient for AI agents and programmatic access. Supports both git-based and raw S3-based artifact storage.

**Parameters:**

- `artifact_id`: The id of the artifact to read the file from. Format: `"workspace/alias"` or just `"alias"` (uses current workspace).
- `file_path`: Path to the file within the artifact. Use forward slashes for nested paths (e.g., `"data/train.csv"`).
- `format`: Output format. Options:
  - `"text"` (default): Returns decoded string using specified encoding
  - `"binary"`: Returns raw bytes
  - `"base64"`: Returns base64-encoded ASCII string
  - `"json"`: Returns parsed Python dict/list from JSON content
- `encoding`: Text encoding to use when `format="text"` or `"json"`. Default is `"utf-8"`. Common values: `"utf-8"`, `"ascii"`, `"latin-1"`.
- `offset`: Byte offset to start reading from. Must be >= 0. Used for partial/chunked reads. Default is `0`.
- `limit`: Maximum number of bytes to read. Default is 64KB. Maximum allowed is 10MB (10485760 bytes).
- `version`: Version to read from (e.g., `"v1.0"`, `"latest"`, commit SHA for git). Use `"stage"` to read staged changes.
- `stage`: If `True`, reads from staged (uncommitted) storage. Equivalent to `version="stage"`. Cannot be used with other version values.

**Returns:** A dictionary with:
- `name`: The file path
- `content`: File content (type depends on `format` parameter)

**Storage-Specific Behavior:**

- **Git storage**: Reads from git commits using version (branch/tag/commit SHA). Use `stage=True` to read from git staging area. Supports LFS (Large File Storage) for large files.
- **Raw storage**: Reads from S3 versioned storage. Use `stage=True` to read from staging version.

**Example:**

```python
# Read text file
result = await artifact_manager.read_file("my-dataset", "README.md")
text = result["content"]  # string

# Read JSON file as parsed object
result = await artifact_manager.read_file("my-dataset", "config.json", format="json")
config = result["content"]  # dict or list

# Read binary file as base64
result = await artifact_manager.read_file("my-model", "weights.bin", format="base64")
import base64
binary_data = base64.b64decode(result["content"])

# Read specific version
result = await artifact_manager.read_file("my-dataset", "data.csv", version="v1.0")

# Partial read (chunked) - read 5000 bytes starting at offset 1000
result = await artifact_manager.read_file("my-dataset", "large.txt", offset=1000, limit=5000)

# Read from staging
result = await artifact_manager.read_file("my-dataset", "draft.md", stage=True)
```

---

### `write_file(artifact_id: str, file_path: str, content: Any, format: str = "text", encoding: str = "utf-8") -> dict`

Writes file content directly into an artifact with flexible encoding options. This method writes file content without requiring presigned URLs, making it convenient for AI agents and programmatic access. Supports both git-based and raw S3-based artifact storage.

**Important:** Requires the artifact to be in staging mode. Call `edit(artifact_id, stage=True)` first. After writing files, call `commit()` to finalize changes.

**Parameters:**

- `artifact_id`: The id of the artifact to write the file to. Format: `"workspace/alias"` or just `"alias"` (uses current workspace).
- `file_path`: Path where to store the file within the artifact. Use forward slashes for nested paths (e.g., `"data/output.json"`).
- `content`: File content. Interpreted based on `format`:
  - For `"text"`: Must be a string
  - For `"binary"`: Must be bytes or bytearray
  - For `"base64"`: Must be a base64-encoded string
  - For `"json"`: Can be dict/list (will be serialized) or already-serialized JSON string
- `format`: Input format. Options:
  - `"text"` (default): String to encode using specified encoding
  - `"binary"`: Raw bytes to write directly
  - `"base64"`: Base64 string to decode before writing
  - `"json"`: Dict/list to serialize as JSON, or already-serialized JSON string
- `encoding`: Text encoding to use when `format="text"` or `"json"`. Default is `"utf-8"`.

**Returns:** A dictionary with:
- `success`: Boolean indicating success
- `bytes_written`: Number of bytes written

**Storage-Specific Behavior:**

- **Git storage**: Files are written to the git staging area. Call `commit()` to add files to git repository.
- **Raw storage**: Files are written to the staging version in S3. Call `commit()` to finalize the version.

**Example:**

```python
# First, put artifact in staging mode
await artifact_manager.edit("my-dataset", stage=True)

# Write text file
result = await artifact_manager.write_file("my-dataset", "README.md", "# My Dataset\n\nDescription here...")
print(f"Wrote {result['bytes_written']} bytes")

# Write JSON data directly (no manual serialization needed)
data = {"key": "value", "items": [1, 2, 3]}
await artifact_manager.write_file("my-dataset", "config.json", data, format="json")

# Write binary file using base64
import base64
binary_data = b"\x00\x01\x02\x03"
encoded = base64.b64encode(binary_data).decode()
await artifact_manager.write_file("my-model", "weights.bin", encoded, format="base64")

# Commit the changes
await artifact_manager.commit("my-dataset")
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

### `set_download_weight(artifact_id: str, file_path: str, download_weight: float) -> None`

Sets or removes the download weight for a specific file in an artifact. Download weights determine how much each file contributes to the artifact's download count when accessed. This operation requires read_write permissions.

**Storage Optimization:** Files with weight `0` or negative values are not stored in the configuration to save storage space. Only positive weights are persisted.

**Parameters:**

- `artifact_id`: The id of the artifact containing the file. It can be a UUID generated by `create` or `edit` function, or it can be an alias of the artifact under the current workspace. If you want to refer to an artifact in another workspace, you should use the full alias in the format of `"workspace_id/alias"`.
- `file_path`: The relative path of the file within the artifact (e.g., `"data.csv"`, `"models/model.pkl"`).
- `download_weight`: A float value representing the file's impact on download count. Must be non-negative. Setting to `0` effectively removes the weight entry to save storage space.

**Permissions:**

- Requires read_write permissions on the artifact

**Example:**

```python
# Set download weight for a file
await artifact_manager.set_download_weight(
    artifact_id="example-dataset",
    file_path="data.csv",
    download_weight=2.5
)

# Set weight for a file in a subdirectory
await artifact_manager.set_download_weight(
    artifact_id="example-dataset", 
    file_path="models/trained_model.pkl",
    download_weight=5.0
)

# Remove download weight (set to 0) to save storage
await artifact_manager.set_download_weight(
    artifact_id="example-dataset",
    file_path="data.csv", 
    download_weight=0
)

# Update existing download weight
await artifact_manager.set_download_weight(
    artifact_id="example-dataset",
    file_path="important_file.txt",
    download_weight=10.0
)
```

**How Download Weights Work:**

- When a file is accessed via `get_file` or included in a zip download, its download weight is added to the artifact's total download count
- Files without download weights (or with weight `0`) don't contribute to download statistics
- This allows you to track downloads of important files while ignoring access to auxiliary files like README files or metadata
- Download weights are also used when creating zip files - each included file contributes its weight to the total download count

**Special Case - "create-zip-file" Weight:**

You can set a special download weight with the file path `"create-zip-file"` to override the default zip download behavior:

```python
# Set a special weight for zip downloads that overrides individual file weights
await artifact_manager.set_download_weight(
    artifact_id="example-dataset",
    file_path="create-zip-file",  # Special key for zip downloads
    download_weight=5.0
)
```

When this special weight is set:
- **Zip downloads**: Use the special weight instead of summing individual file weights
- **Individual file downloads**: Still use their respective individual weights
- **Use case**: Useful when you want consistent download tracking for zip files regardless of which files are included

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
  - **Query Parameters**:
    - `use_proxy`: (Optional) Boolean to control whether to use the S3 proxy
    - `use_local_url`: (Optional) Boolean or string to generate local/cluster-internal/proxy URLs

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
  - **use_proxy**: (Optional) Boolean to control whether to use the S3 proxy for the generated URLs. Defaults to server configuration.
  - **use_local_url**: (Optional) Boolean or str to generate local/cluster-internal/proxy URLs. Defaults to `False`.
  - **expires_in**: (Optional) Number of seconds for the presigned URL to expire. Defaults to 3600.
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

    # Upload with proxy bypassed (useful for direct S3 access)
    with open("local_file.csv", "rb") as f:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(
                f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/{file_path}",
                data=f,
                params={
                    "download_weight": 1.0,
                    "use_proxy": False
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "text/csv"
                }
            )

    # Upload using local/cluster-internal URLs
    with open("local_file.csv", "rb") as f:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.put(
                f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/files/{file_path}",
                data=f,
                params={
                    "download_weight": 1.0,
                    "use_local_url": True
                },
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

For large files, the Artifact Manager supports multipart uploads which allow uploading files in chunks. This is useful for files larger than 5GB or when you need to resume interrupted uploads. Each part (except the last) must be at least 5MB in size as required by S3.

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
  - **part_count**: The total number of parts for the upload (required, max 10,000). Each part (except the last) must be at least 5MB in size.
  - **expires_in**: (Optional) Number of seconds for presigned URLs to expire. Defaults to 3600.
  - **use_proxy**: (Optional) Boolean to control whether to use the S3 proxy for the generated URLs. Defaults to `False`.
  - **use_local_url**: (Optional) Boolean or str to generate local/cluster-internal/proxy URLs. Defaults to `False`.

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

**Important Notes:**
- All parts must be uploaded before calling this endpoint
- Each part (except the last) must be at least 5MB in size
- ETags should be provided without surrounding quotes
- The request will fail if any parts are missing or have incorrect ETags

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

        # Or with proxy/local URL options
        response = await client.post(
            f"{SERVER_URL}/{workspace}/artifacts/{artifact_alias}/create-multipart-upload",
            params={
                "path": file_path,
                "part_count": 3,
                "expires_in": 3600,
                "use_proxy": False,  # Bypass proxy for direct S3 access
                "use_local_url": True  # Use local URLs for cluster-internal access
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
- **Special override**: If a download weight is set for the special path `"create-zip-file"` (using `set_download_weight`), that weight is used instead of summing individual file weights

**Example**: If you create a zip with 3 files having weights [0.5, 1.0, 0], the download count will increase by 1.5 (0.5 + 1.0 + 0).

**Special Weight Example**: If you set `download_weight` for `"create-zip-file"` to 2.0, then every zip download will increment the count by 2.0 regardless of which files are included.

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

The Artifact Manager supports hosting static websites through special "view" endpoint. These artifacts can serve static files with optional Jinja2 template rendering, making them perfect for hosting documentation, portfolios, or any static web content.

By setting the `view_config`, any artifact can be turned into a static site or pages, similar to Github Pages.

### Site Artifact Features

- **Static File Serving**: Serve HTML, CSS, JavaScript, images, and other static files
- **Template Rendering**: Jinja2 template support for dynamic content
- **Custom Headers**: Configure CORS, caching, and other HTTP headers
- **Version Control**: Support for staging and versioning of site content
- **Access Control**: Permission-based access to site content
- **View Statistics**: Track page views and download counts

### Creating and Deploying a Static View

#### Step 1: Create a View Artifact

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

    # Create a view artifact
    manifest = {
        "name": "My Portfolio",
        "description": "A personal portfolio website",
    }
    
    view_config = {
        "root_directory": "/", # it can be a subdirectory too
        "templates": ["index.html", "about.html"],  # Files that use Jinja2 templates
        "template_engine": "jinja2",
        "use_builtin_template": False,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "max-age=3600"
        }
    }
    
    site_artifact = await artifact_manager.create(
        alias="my-portfolio",
        manifest=manifest,
        config={
            "view_config": view_config
        },
        stage=True
    )
    
    print(f"Artifact created with ID: {site_artifact.id}")
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
    # https://your-hypha-server.com/{workspace}/view/my-portfolio/
    
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
    site_url = f"{SERVER_URL}/{workspace}/view/my-portfolio/"
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
view_config = {
    "root_directory": "/",
    "templates": ["index.html", "about.html", "contact.html"],  # Files to render with Jinja2
    "template_engine": "jinja2",  # Currently only Jinja2 is supported
    "use_builtin_template": False,
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
- **`BASE_URL`**: The base URL for the site (e.g., `/workspace/view/alias/`)
- **`PUBLIC_BASE_URL`**: The public base URL of the Hypha server
- **`LOCAL_BASE_URL`**: The local base URL for cluster-internal access
- **`WORKSPACE`**: The workspace name
- **`VIEW_COUNT`**: Number of times the artifact has been viewed
- **`DOWNLOAD_COUNT`**: Number of times files have been downloaded
- **`USER`**: User information (if authenticated, otherwise None)

### Site Serving Endpoints

#### HTTP Endpoint

**URL Pattern**: `GET /{workspace}/view/{artifact_alias}/{file_path:path}`

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
            f"{SERVER_URL}/{workspace}/view/my-portfolio/",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Main page: {response.status_code}")
        
        # Access a specific file
        response = await client.get(
            f"{SERVER_URL}/{workspace}/view/my-portfolio/style.css",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"CSS file: {response.status_code}")
        
        # Access staged version
        response = await client.get(
            f"{SERVER_URL}/{workspace}/view/my-portfolio/?stage=true",
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
        alias="portfolio",
        manifest={
            "name": "John Doe - Portfolio",
            "description": "Software Engineer & Data Scientist",
            "type": "site"
        },
        config={
            "view": {
                "root_directory": "/",
                "templates": ["index.html", "about.html", "projects.html"],
                "template_engine": "jinja2",
                "headers": {
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "max-age=3600"
                }
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
    
    print(f"Portfolio deployed at: {SERVER_URL}/{workspace}/view/portfolio/")
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

### Version Index Behavior

The Artifact Manager uses version indices internally to determine where files should be stored. Understanding this behavior is crucial for working with staging and versions:

1. **Version Index Calculation**
   - **Creating New Version** (`version="new"` + `stage=True`): Version index = `len(artifact.versions)` (the next version)
   - **Editing Current Version** (`stage=True` without `version="new"`): Version index = `len(artifact.versions) - 1` (current version)
   - This index determines the S3 path where files are stored: `v{version_index}/`

2. **File Storage Location**
   - Files are always placed directly in their final destination based on the version intent
   - When creating a new version: Files go to `v{next_index}/` (e.g., if you have v0, files go to `v1/`)
   - When editing current version: Files go to `v{current_index}/` (e.g., if latest is v0, files go to `v0/`)
   - No file copying occurs during commit - the version index ensures files are already in the right place

3. **Practical Examples**
   ```python
   # Scenario 1: Editing current version
   await artifact_manager.edit(artifact_id=artifact.id, stage=True)
   # Version index = len(versions) - 1 (current version)
   # Files uploaded will go to the current version's directory
   
   # Scenario 2: Creating new version
   await artifact_manager.edit(artifact_id=artifact.id, version="new", stage=True)
   # Version index = len(versions) (next version)
   # Files uploaded will go to the new version's directory
   ```

4. **Important Implications**
   - The version index is determined at edit time, not commit time
   - All file operations (put, list, remove) respect this version index
   - This ensures consistency and prevents file duplication
   - The commit operation simply finalizes the metadata without moving files

---

## Python Integration

### hypha-artifact Package

For easier access to artifact files, we provide a Python package `hypha-artifact` that simplifies common operations like listing files, downloading, uploading, reading, and writing files.

**Installation:**
```bash
pip install hypha-artifact
```

**Basic Usage:**
```python
from hypha_artifact import AsyncHyphaArtifact

# Create artifact client
artifact = AsyncHyphaArtifact(
    server_url="https://hypha.aicell.io",
    artifact_id="workspace/my-artifact",
    workspace="workspace",
    token="your-token"
)

# List all files in the artifact
files = await artifact.list_files()
print(f"Available files: {files}")

# Download a file
content = await artifact.get_file("data/example.csv")
print(f"File content: {content}")

# Upload a file
await artifact.upload_file("data/new_file.txt", "Hello, World!")

# Read file content
with await artifact.open_file("data/example.csv") as f:
    content = await f.read()
    print(f"File content: {content}")

# Write to a file
async with await artifact.open_file("data/output.txt", "w") as f:
    await f.write("New content")
```

**Advanced Features:**
- **File Operations**: List, download, upload, read, and write files
- **Directory Support**: Navigate through artifact directories
- **Async/Await**: Full async support for efficient I/O operations
- **Error Handling**: Comprehensive error handling and retry logic
- **Authentication**: Automatic token management and authentication

For more information and advanced usage examples, see the [hypha-artifact repository](https://github.com/aicell-lab/hypha-artifact/).

---

## Git Storage for Artifacts

Hypha supports Git-based storage for artifacts, enabling version-controlled source code management with full Git protocol compatibility. This is ideal for:

- **AI Agent Code Generation**: Allow AI agents to generate, modify, and commit code via standard Git operations
- **Collaborative Development**: Multiple contributors can push/pull changes using familiar Git workflows
- **Large File Support**: Integrated Git LFS (Large File Storage) for binary files like models and datasets
- **Version History**: Full Git commit history with branch and tag support

### Creating a Git-Enabled Artifact

To create an artifact with Git storage, set `storage: "git"` in the config:

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"name": "my-client", "server_url": SERVER_URL})
artifact_manager = await server.get_service("public/artifact-manager")

# Create a git-storage artifact
artifact = await artifact_manager.create(
    alias="my-code-repo",
    manifest={"name": "My Code Repository", "description": "AI-generated code"},
    config={"storage": "git"}
)

# Read the artifact to get the git_url
info = await artifact_manager.read(artifact_id="my-code-repo")
print(f"Git URL: {info['git_url']}")
# Output: Git URL: https://hypha.aicell.io/<workspace>/git/my-code-repo
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `storage` | Set to `"git"` to enable Git storage | `"raw"` |
| `git_default_branch` | Default branch name | `"main"` |

Example with custom default branch:

```python
await artifact_manager.create(
    alias="dev-repo",
    manifest={"name": "Development Repo"},
    config={"storage": "git", "git_default_branch": "develop"}
)
```

### Cloning and Pushing via Git

Once created, you can interact with the repository using standard Git commands:

```bash
# Clone the repository (use your Hypha token for authentication)
git clone http://<your-hypha-server>/<workspace>/git/my-code-repo

# Configure credentials (use "git" as username, token as password)
git config credential.helper store
echo "http://git:<your-token>@<your-hypha-server>" >> ~/.git-credentials

# Or use token in URL directly
git clone http://git:<your-token>@<your-hypha-server>/<workspace>/git/my-code-repo

# Make changes and push
cd my-code-repo
echo "# My Project" > README.md
git add .
git commit -m "Initial commit"
git push origin main
```

### Git LFS for Large Files

Git-storage artifacts automatically support Git LFS for large files. Configure LFS in your repository:

```bash
# Inside your cloned repository
git lfs install
git lfs track "*.bin" "*.h5" "*.pkl" "*.pt"
git add .gitattributes

# Add large files - they'll be stored via LFS
git add model.pt
git commit -m "Add trained model"
git push origin main
```

LFS files are stored in S3 and streamed efficiently for download.

### Programmatic File Access

You can access files in git-storage artifacts via the standard Artifact Manager API:

```python
# List files at repository root
files = await artifact_manager.list_files(artifact_id="my-code-repo")
for f in files:
    print(f"{f['name']} ({f['type']}, {f.get('size', 'N/A')} bytes)")

# List files in a subdirectory
files = await artifact_manager.list_files(
    artifact_id="my-code-repo",
    dir_path="src/components"
)

# Get file content (returns URL for large files, content for small files)
file_info = await artifact_manager.get_file(
    artifact_id="my-code-repo",
    file_path="src/main.py"
)

# Access specific version (branch, tag, or commit SHA)
files = await artifact_manager.list_files(
    artifact_id="my-code-repo",
    version="v1.0"  # Can be branch name, tag, or full 40-char commit SHA
)

# Get file at specific commit
content = await artifact_manager.get_file(
    artifact_id="my-code-repo",
    file_path="config.json",
    version="abc123def456..."  # Full 40-character commit SHA
)
```

### Programmatic File Upload (Staging API)

For AI agents or programmatic workflows, you can use the staging API to upload files without Git:

```python
import requests

# Enter staging mode
await artifact_manager.edit(artifact_id="my-code-repo", stage=True)

# Upload a file
put_url = await artifact_manager.put_file(
    artifact_id="my-code-repo",
    file_path="generated/output.py"
)
requests.put(put_url, data=b'print("Hello, AI!")')

# Commit the changes
await artifact_manager.commit(
    artifact_id="my-code-repo",
    comment="Add AI-generated output"
)
```

### Removing Files

```python
# Enter staging mode first
await artifact_manager.edit(artifact_id="my-code-repo", stage=True)

# Mark file for removal
await artifact_manager.remove_file(
    artifact_id="my-code-repo",
    file_path="obsolete_code.py"
)

# Commit the removal
await artifact_manager.commit(
    artifact_id="my-code-repo",
    comment="Remove obsolete code"
)
```

### Creating ZIP Archives

Download all or selected files as a ZIP archive:

```python
# Download entire repository as ZIP
zip_url = f"{SERVER_URL}/{workspace}/artifacts/my-code-repo/create-zip"
response = requests.get(zip_url, params={"token": token})
with open("repo.zip", "wb") as f:
    f.write(response.content)

# Download specific files
zip_url = f"{SERVER_URL}/{workspace}/artifacts/my-code-repo/create-zip"
response = requests.get(zip_url, params={
    "token": token,
    "files": ["src/main.py", "README.md"]
})
```

### HTTP File Access

Files can be accessed directly via HTTP endpoints:

```bash
# Get a file (with authentication)
curl -H "Authorization: Bearer <token>" \
  https://<server>/<workspace>/artifacts/my-code-repo/files/src/main.py

# Or use query parameter
curl "https://<server>/<workspace>/artifacts/my-code-repo/files/src/main.py?token=<token>"
```

### Permissions

Git-storage artifacts use the same permission system as regular artifacts:

```python
# Create git repo with specific permissions
await artifact_manager.create(
    alias="team-repo",
    manifest={"name": "Team Repository"},
    config={
        "storage": "git",
        "permissions": {
            "*": "r",           # Everyone can clone (read)
            "@": "r+",          # Authenticated users can clone and create branches
            "team-lead-user-id": "rw+"  # Team lead has full access
        }
    }
)
```

Permission levels for git operations:
- `r` (read): Clone and fetch
- `r+` (read + create): Clone, fetch, and push new branches
- `rw` (read-write): Clone, fetch, and push to existing branches
- `rw+` (read-write + create): Full access including creating new branches

### AI Agent Integration Example

Here's a complete example of an AI agent workflow using git-storage:

```python
import subprocess
import tempfile
import os

async def ai_agent_code_generation():
    """AI agent generates code and pushes to Hypha git artifact."""

    # Connect to Hypha
    server = await connect_to_server({
        "name": "ai-code-agent",
        "server_url": SERVER_URL,
        "token": AI_AGENT_TOKEN
    })
    artifact_manager = await server.get_service("public/artifact-manager")

    # Get or create the code repository
    try:
        repo_info = await artifact_manager.read(artifact_id="ai-generated-code")
    except:
        await artifact_manager.create(
            alias="ai-generated-code",
            manifest={"name": "AI Generated Code"},
            config={"storage": "git"}
        )
        repo_info = await artifact_manager.read(artifact_id="ai-generated-code")

    git_url = repo_info["git_url"]
    workspace = server.config.workspace

    # Clone, modify, and push
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, "repo")

        # Clone with token authentication
        auth_url = git_url.replace("https://", f"https://git:{AI_AGENT_TOKEN}@")
        subprocess.run(["git", "clone", auth_url, repo_path], check=True)

        # Generate code (AI would do this)
        code_content = '''
def hello_world():
    """AI-generated function."""
    return "Hello from AI!"

if __name__ == "__main__":
    print(hello_world())
'''

        with open(os.path.join(repo_path, "ai_generated.py"), "w") as f:
            f.write(code_content)

        # Commit and push
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "AI: Add generated code"],
            cwd=repo_path,
            check=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "AI Agent", "GIT_AUTHOR_EMAIL": "ai@example.com"}
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, check=True)

    print("AI agent successfully pushed generated code!")
    await server.disconnect()
```

### Technical Notes

- **Smart HTTP Protocol**: Git operations use the Git Smart HTTP protocol for efficient data transfer
- **S3 Backend**: All git objects and LFS files are stored in S3 for durability and scalability
- **Pack Files**: Git objects are stored as pack files for efficient storage
- **Authentication**: Uses HTTP Basic Auth with `git` as username and Hypha token as password
- **Version Resolution**: Supports branch names, tag names, and full 40-character commit SHAs
