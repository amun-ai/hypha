# Git-Compatible Version Control System for Hypha Artifacts

## Overview

This document describes the design for exposing Hypha artifacts as Git repositories, enabling standard Git clients to interact with them while storing all data in S3 with proper incremental/delta storage.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Git Client (git clone/push/pull)             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Git Smart HTTP Endpoints (FastAPI)              │
│  /{workspace}/artifacts/{alias}/git/info/refs                   │
│  /{workspace}/artifacts/{alias}/git/git-upload-pack             │
│  /{workspace}/artifacts/{alias}/git/git-receive-pack            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    S3GitObjectStore (Async)                      │
│  - Extends dulwich.object_store.BucketBasedObjectStore          │
│  - All operations are async using aiobotocore                    │
│  - Pack files stored in S3 with proper indexing                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         S3 Storage                               │
│  {workspace}/artifacts/{artifact_id}/git/                        │
│    ├── objects/pack/                                             │
│    │   ├── pack-{sha}.pack                                       │
│    │   └── pack-{sha}.idx                                        │
│    ├── info/refs                                                 │
│    └── HEAD                                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Layout in S3

For each artifact with Git enabled:

```
{workspace}/artifacts/{artifact_id}/
├── git/                          # Git repository data
│   ├── objects/
│   │   └── pack/
│   │       ├── pack-{sha1}.pack  # Pack files with delta compression
│   │       └── pack-{sha1}.idx   # Pack index files
│   ├── refs/
│   │   ├── heads/
│   │   │   └── main              # Branch references
│   │   └── tags/
│   │       └── v1.0              # Tag references
│   └── HEAD                      # Default branch reference
├── v0/                           # Existing version storage (can coexist)
├── v1/
└── ...
```

## Implementation Details

### 1. S3GitObjectStore (hypha/git/object_store.py)

Async S3-backed object store extending `dulwich.object_store.BucketBasedObjectStore`:

```python
class S3GitObjectStore(BucketBasedObjectStore):
    """Async S3-backed Git object store using aiobotocore."""

    def __init__(self, s3_client, bucket: str, prefix: str):
        super().__init__()
        self._s3_client = s3_client
        self._bucket = bucket
        self._prefix = prefix
        self._pack_dir = f"{prefix}/objects/pack"

    async def _iter_pack_names_async(self) -> AsyncIterator[str]:
        """List all pack files in S3."""
        paginator = self._s3_client.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=self._bucket, Prefix=self._pack_dir):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.pack'):
                    yield obj['Key'].rsplit('/', 1)[-1].replace('.pack', '')

    async def _upload_pack_async(self, basename: str, pack_data: bytes, index_data: bytes):
        """Upload pack and index files to S3."""
        await asyncio.gather(
            self._s3_client.put_object(
                Bucket=self._bucket,
                Key=f"{self._pack_dir}/{basename}.pack",
                Body=pack_data
            ),
            self._s3_client.put_object(
                Bucket=self._bucket,
                Key=f"{self._pack_dir}/{basename}.idx",
                Body=index_data
            )
        )

    async def _get_pack_data_async(self, name: str) -> bytes:
        """Download pack file from S3."""
        response = await self._s3_client.get_object(
            Bucket=self._bucket,
            Key=f"{self._pack_dir}/{name}.pack"
        )
        return await response['Body'].read()
```

### 2. S3RefsContainer (hypha/git/refs.py)

Store Git references in S3:

```python
class S3RefsContainer(RefsContainer):
    """Store Git refs in S3."""

    async def get_ref_async(self, name: bytes) -> ObjectID | None:
        """Get a reference from S3."""
        key = f"{self._prefix}/refs/{name.decode('utf-8')}"
        try:
            response = await self._s3_client.get_object(Bucket=self._bucket, Key=key)
            return (await response['Body'].read()).strip()
        except ClientError:
            return None

    async def set_ref_async(self, name: bytes, value: ObjectID):
        """Set a reference in S3."""
        key = f"{self._prefix}/refs/{name.decode('utf-8')}"
        await self._s3_client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=value + b'\n'
        )
```

### 3. Git HTTP Smart Protocol (hypha/git/http.py)

FastAPI router implementing Git Smart HTTP protocol:

```python
@router.get("/{workspace}/artifacts/{alias}/git/info/refs")
async def git_info_refs(
    workspace: str,
    alias: str,
    service: str = Query(None),
    user_info: UserInfo = Depends(login_optional)
):
    """Handle Git reference discovery."""
    if service == "git-upload-pack":
        # Return refs for fetch
        return StreamingResponse(
            generate_refs_advertisement(repo, b"git-upload-pack"),
            media_type="application/x-git-upload-pack-advertisement"
        )
    elif service == "git-receive-pack":
        # Return refs for push (requires write permission)
        return StreamingResponse(
            generate_refs_advertisement(repo, b"git-receive-pack"),
            media_type="application/x-git-receive-pack-advertisement"
        )

@router.post("/{workspace}/artifacts/{alias}/git/git-upload-pack")
async def git_upload_pack(
    workspace: str,
    alias: str,
    request: Request,
    user_info: UserInfo = Depends(login_optional)
):
    """Handle Git fetch/clone requests."""
    # Parse want/have lines from client
    # Compute pack with requested objects
    # Stream pack back to client

@router.post("/{workspace}/artifacts/{alias}/git/git-receive-pack")
async def git_receive_pack(
    workspace: str,
    alias: str,
    request: Request,
    user_info: UserInfo = Depends(login_required)
):
    """Handle Git push requests."""
    # Validate permissions
    # Receive pack from client
    # Validate and store pack
    # Update refs
```

### 4. Integration with Artifact System

Add Git configuration to artifact manifest:

```python
# In ArtifactModel, add to config:
{
    "git_enabled": True,
    "git_default_branch": "main",
    "git_auto_commit": False,  # Auto-commit file changes to Git
}
```

New artifact service methods:

```python
@schema_method
async def enable_git(
    self,
    artifact_id: str,
    default_branch: str = "main",
    import_existing_files: bool = True,
    context: dict = None
) -> dict:
    """Enable Git version control for an artifact."""

@schema_method
async def get_git_url(
    self,
    artifact_id: str,
    context: dict = None
) -> str:
    """Get the Git clone URL for an artifact."""
    # Returns: https://{server}/{workspace}/artifacts/{alias}/git
```

## Key Design Decisions

### 1. Use Dulwich Library

**Decision**: Use dulwich for Git internals, extending `BucketBasedObjectStore` for S3.

**Rationale**:
- Pure Python implementation (no native dependencies)
- Already has cloud storage backends (Swift) as reference
- Handles pack file format, delta compression, protocol parsing
- Well-maintained and actively developed

### 2. Pack-Only Storage (No Loose Objects)

**Decision**: Store all objects in pack files, not loose objects.

**Rationale**:
- Better for S3 (fewer API calls)
- Efficient delta compression across related objects
- Matches dulwich's BucketBasedObjectStore design
- Standard pattern for cloud-hosted Git

### 3. Async All the Way

**Decision**: All S3 operations use aiobotocore async client.

**Rationale**:
- Consistent with Hypha's async architecture
- Better scalability under load
- No blocking operations that could affect other requests

### 4. Stateless HTTP Protocol

**Decision**: Use Git Smart HTTP protocol (not SSH).

**Rationale**:
- Works over standard HTTPS
- Stateless (fits Hypha's request model)
- Easy authentication via existing token system
- Can use existing HTTP infrastructure

### 5. Separate Git Storage from File Storage

**Decision**: Git data stored in `git/` subdirectory, coexisting with version storage.

**Rationale**:
- Backward compatible with existing artifacts
- Optional Git enablement per artifact
- Clear separation of concerns
- Can sync between Git and file versions if needed

## Authentication & Authorization

Git operations use standard Hypha authentication:

```bash
# Clone with token
git clone https://server/ws/artifacts/my-artifact/git

# Push requires write permission
git push https://token@server/ws/artifacts/my-artifact/git main
```

Credential handling:
- HTTP Basic Auth with token as password
- Git credential helper integration
- OAuth token support

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] S3GitObjectStore with async operations
- [ ] S3RefsContainer for reference management
- [ ] Pack file read/write to S3

### Phase 2: HTTP Protocol
- [ ] Info/refs endpoint
- [ ] git-upload-pack (clone/fetch)
- [ ] git-receive-pack (push)
- [ ] Streaming responses for large repos

### Phase 3: Artifact Integration
- [ ] enable_git() method
- [ ] get_git_url() method
- [ ] Git URL routing in artifact HTTP endpoints
- [ ] Permission integration

### Phase 4: Advanced Features
- [ ] Shallow clone support
- [ ] LFS integration for large files
- [ ] Webhook notifications on push
- [ ] Git history to artifact versions sync

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
   - Object store operations
   - Reference management
   - Pack file handling

2. **Integration Tests**: Test with real Git client
   - Clone empty repo
   - Push commits
   - Clone with data
   - Push additional commits
   - Branching and tagging

3. **Performance Tests**:
   - Large repository handling
   - Concurrent operations
   - Delta compression efficiency

## Example Usage

```python
# Enable Git for an artifact
await artifact_manager.enable_git("my-dataset")

# Get clone URL
url = await artifact_manager.get_git_url("my-dataset")
# https://hypha.example.com/workspace/artifacts/my-dataset/git

# Client-side usage
# git clone https://hypha.example.com/workspace/artifacts/my-dataset/git
# cd my-dataset
# git add file.txt
# git commit -m "Add file"
# git push origin main
```

## Dependencies

```
dulwich>=0.22.0  # Pure Python Git implementation
aiobotocore>=2.0.0  # Async S3 client (already used by Hypha)
```
