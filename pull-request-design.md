# Pull Request System for Git-Storage Artifacts

## Overview

This document describes the design of the Pull Request (PR) system for Hypha's git-backed artifacts. The system enables **swarms of AI agents** to collaborate on shared code repositories by working on isolated branches, proposing changes via PRs, and having reviewer agents merge approved changes into the main branch.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Agent A     │  │  Agent B     │  │  Agent C     │
│ feature-auth │  │ fix-parser   │  │ add-tests    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
   create_branch     create_branch     create_branch
   write_file(s)     write_file(s)     write_file(s)
   commit            commit            commit
   create_pr         create_pr         create_pr
       │                 │                 │
       └────────┬────────┘─────────────────┘
                ▼
        ┌───────────────┐
        │ Review Agent  │
        │   get_diff    │
        │ submit_review │
        │   merge_pr    │
        └───────┬───────┘
                ▼
            main branch
           moves forward
```

## Motivation

When multiple AI agents work on the same codebase (e.g., solving different parts of a scientific or engineering problem), they need:

1. **Isolation**: Each agent works on its own branch without interfering with others.
2. **Proposal**: Agents submit their changes for review before merging.
3. **Review**: A reviewer agent (or human) can inspect the diff and approve/reject.
4. **Integration**: Approved changes are merged into the main branch, moving the project forward.
5. **Conflict Detection**: When two agents modify the same file, conflicts are detected and reported rather than silently lost.

This is analogous to GitHub's Pull Request workflow, but optimized for programmatic/AI-agent use:
- Pure RPC API (no web UI needed)
- Machine-readable structured diffs (not unified patch format)
- Simple status model: `open` → `merged` | `closed`
- Automated merge with conflict detection

## Architecture

### Where PRs Live

PR metadata is stored in the artifact's `config` JSON field (`artifact.config["pull_requests"]`). This was chosen over a separate SQL table because:

- PRs are scoped to a single artifact (no cross-artifact queries needed)
- Volume is moderate (tens to low hundreds per artifact)
- Atomic updates: PR status and git refs update in the same transaction
- Portability: when artifacts are exported/duplicated, PRs travel with them

### Git Operations

The core merge/diff logic lives in `hypha/git/merge.py`, separate from the RPC layer in `artifact.py`. This module operates on `S3GitRepo` objects using dulwich for tree walking, merge-base computation, and object creation.

### Integration Point

All new methods are added to the existing `artifact-manager` service (not a separate service), maintaining a single coherent API surface for artifact operations.

## Data Model

### PR Storage Structure

```python
# Inside artifact.config
{
    "storage": "git",
    "pull_requests": {
        "next_pr_number": 4,        # Auto-increment counter
        "prs": {
            "1": {
                "pr_number": 1,
                "title": "Add authentication module",
                "description": "Implements JWT-based auth with refresh tokens.",
                "source_branch": "feature-auth",
                "target_branch": "main",
                "source_sha": "abc123...",      # SHA when PR was created
                "target_sha": "def456...",      # SHA when PR was created
                "status": "open",               # "open" | "merged" | "closed"
                "created_by": "agent-id-123",
                "created_at": 1700000000,       # Unix timestamp
                "updated_at": 1700001000,
                "merged_at": null,
                "closed_at": null,
                "merge_sha": null,              # Set on merge
                "merge_type": null,             # "fast-forward" | "merge"
                "reviews": [
                    {
                        "review_id": "r_abc123",
                        "reviewer": "review-agent-id",
                        "verdict": "approve",       # "approve" | "request_changes" | "comment"
                        "comment": "LGTM, clean separation of concerns.",
                        "created_at": 1700000500
                    }
                ]
            }
        }
    }
}
```

### Why Not Git Refs for PRs

Storing PRs as git refs (like GitHub's `refs/pull/NNN/head`) was considered but rejected:
- Refs only store a SHA pointer — we need structured metadata (title, reviews, status)
- Would require a second storage mechanism for metadata anyway
- Git refs are exposed to git clone/push clients, polluting the ref namespace

### Why Not a Separate SQL Table

- PRs are tightly coupled to a single artifact
- No cross-artifact PR queries are needed
- JSON in config handles moderate volumes well
- Avoids schema migrations

## API Design

All methods are added to `ArtifactController` and exposed via `get_artifact_service()`. They follow the existing `@schema_method` pattern with Pydantic `Field` descriptions.

### Branch Operations

#### `create_branch`

```python
@schema_method
async def create_branch(
    self,
    artifact_id: str = Field(
        ...,
        description="Git artifact identifier. Format: 'workspace/alias' or just 'alias'."
    ),
    branch: str = Field(
        ...,
        description="Name of the branch to create (e.g., 'feature-auth', 'fix-bug-123')."
    ),
    source: Optional[str] = Field(
        None,
        description="Source ref to branch from: branch name, tag name, or commit SHA. Defaults to default branch."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Create a new branch on a git-storage artifact.

    Returns: {"branch": "feature-auth", "sha": "abc123...", "artifact_id": "ws/alias"}
    Raises: ValueError if branch already exists or source ref not found.
    """
```

#### `delete_branch`

```python
@schema_method
async def delete_branch(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    branch: str = Field(
        ...,
        description="Branch name to delete. Cannot delete the default branch."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, str]:
    """Delete a branch from a git-storage artifact.

    Returns: {"status": "deleted", "branch": "feature-auth"}
    Raises: ValueError if branch is the default branch or doesn't exist.
    """
```

#### `list_branches`

```python
@schema_method
async def list_branches(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> List[Dict[str, Any]]:
    """List all branches on a git-storage artifact.

    Returns: [{"name": "main", "sha": "abc...", "default": True}, {"name": "feat", "sha": "def..."}]
    """
```

### PR Lifecycle

#### `create_pr`

```python
@schema_method
async def create_pr(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    title: str = Field(..., description="PR title summarizing the changes."),
    source_branch: str = Field(
        ...,
        description="Branch containing the changes to merge."
    ),
    target_branch: str = Field(
        "main",
        description="Branch to merge into. Defaults to 'main'."
    ),
    description: Optional[str] = Field(
        None,
        description="Detailed description of changes, rationale, and context."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Create a pull request for a git-storage artifact.

    Returns: Full PR object dict with pr_number, title, status, branches, etc.
    Raises: ValueError if source branch doesn't exist or has no new commits.
    """
```

#### `get_pr`

```python
@schema_method
async def get_pr(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number."),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Get details of a pull request including reviews and current branch SHAs.

    Returns: Full PR object dict.
    Raises: KeyError if PR not found.
    """
```

#### `list_prs`

```python
@schema_method
async def list_prs(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    status: Optional[str] = Field(
        None,
        description="Filter by status: 'open', 'merged', 'closed'. None returns all."
    ),
    limit: int = Field(50, description="Maximum PRs to return.", ge=1, le=200),
    offset: int = Field(0, description="Pagination offset.", ge=0),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> List[Dict[str, Any]]:
    """List pull requests for a git-storage artifact.

    Returns: List of PR summary dicts ordered by creation time (newest first).
    """
```

#### `update_pr`

```python
@schema_method
async def update_pr(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number to update."),
    title: Optional[str] = Field(None, description="New title."),
    description: Optional[str] = Field(None, description="New description."),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Update a pull request's title or description. Only open PRs can be updated.

    Returns: Updated PR object dict.
    """
```

#### `close_pr`

```python
@schema_method
async def close_pr(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number to close."),
    comment: Optional[str] = Field(None, description="Reason for closing."),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Close a pull request without merging.

    Returns: Updated PR object dict with status='closed'.
    """
```

#### `merge_pr`

```python
@schema_method
async def merge_pr(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number to merge."),
    merge_strategy: str = Field(
        "auto",
        description="Merge strategy: 'auto' (fast-forward if possible, else 3-way merge), "
                    "'fast-forward' (fail if not possible), 'merge' (always create merge commit)."
    ),
    comment: Optional[str] = Field(
        None,
        description="Custom merge commit message. Defaults to 'Merge PR #N: {title}'."
    ),
    delete_branch: bool = Field(
        True,
        description="Delete the source branch after successful merge."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Merge a pull request.

    Returns: {"status": "merged", "merge_sha": "...", "merge_type": "fast-forward"|"merge"}
    Raises: ValueError if PR not open, has conflicts, or ff not possible when required.
    """
```

### Diff / Comparison

#### `get_diff`

```python
@schema_method
async def get_diff(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    base: str = Field(
        ...,
        description="Base ref for comparison: branch name, tag, or commit SHA."
    ),
    head: str = Field(
        ...,
        description="Head ref for comparison: branch name, tag, or commit SHA."
    ),
    include_content: bool = Field(
        False,
        description="Include full file content (old/new) for changed files. Can be large."
    ),
    path_filter: Optional[str] = Field(
        None,
        description="Restrict diff to files matching this path prefix (e.g., 'src/')."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Compute diff between two git refs. Returns structured data optimized for AI agents.

    Returns: See 'Diff Output Format' section below.
    """
```

### Review

#### `submit_review`

```python
@schema_method
async def submit_review(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number to review."),
    verdict: str = Field(
        ...,
        description="Review verdict: 'approve', 'request_changes', or 'comment'."
    ),
    comment: Optional[str] = Field(
        None,
        description="Review comment explaining the verdict."
    ),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> Dict[str, Any]:
    """Submit a review on a pull request. Only open PRs can be reviewed.

    Returns: Review object dict with review_id, verdict, comment, reviewer, created_at.
    """
```

#### `list_reviews`

```python
@schema_method
async def list_reviews(
    self,
    artifact_id: str = Field(..., description="Git artifact identifier."),
    pr_number: int = Field(..., description="PR number."),
    context: Optional[dict] = Field(None, description="Auto-provided context."),
) -> List[Dict[str, Any]]:
    """List all reviews for a pull request, ordered by creation time.

    Returns: List of review dicts.
    """
```

## Diff Output Format

The diff is returned as structured JSON optimized for programmatic/AI-agent consumption (not unified patch format):

```json
{
    "base_sha": "abc123def456...",
    "head_sha": "789abc012def...",
    "files": [
        {
            "path": "src/auth.py",
            "status": "added",
            "new_sha": "blob_sha...",
            "new_size": 2048,
            "new_content": "import jwt\n..."
        },
        {
            "path": "src/parser.py",
            "status": "modified",
            "old_sha": "old_blob_sha...",
            "new_sha": "new_blob_sha...",
            "old_size": 1024,
            "new_size": 1256,
            "old_content": "...",
            "new_content": "..."
        },
        {
            "path": "src/deprecated.py",
            "status": "deleted",
            "old_sha": "blob_sha...",
            "old_size": 512
        }
    ],
    "stats": {
        "files_changed": 3,
        "additions": 1,
        "deletions": 1,
        "modifications": 1
    }
}
```

- `status` values: `"added"`, `"modified"`, `"deleted"`, `"renamed"`
- `old_content` / `new_content` only included when `include_content=True`
- `path_filter` restricts output to files under that prefix

## Merge Strategies

### Fast-Forward

When the target branch is an ancestor of the source branch (i.e., no new commits on target since the branch point), the target ref is simply moved forward to the source SHA. No merge commit is created.

```
Before:        After (fast-forward):
main: A─B      main: A─B─C─D
           \
feature:    C─D
```

### 3-Way Merge

When both branches have diverged, a 3-way merge is performed:

1. **Find merge base**: The common ancestor commit using `dulwich.graph.find_merge_base`
2. **Flatten trees**: Convert base, target (ours), and source (theirs) trees into `{path: blob_sha}` dicts
3. **Compare per-file**:
   - Only source changed → take source's version
   - Only target changed → keep target's version
   - Both changed to same SHA → no conflict (keep either)
   - Both changed to different SHAs → **conflict**
4. **Build merged tree** from resolved files
5. **Create merge commit** with two parents `[target_sha, source_sha]`
6. **Update target branch** ref

```
Before:           After (3-way merge):
main: A─B─E      main: A─B─E─M (merge commit, parents: E, D)
           \           \     /
feature:    C─D         C─D
```

### Conflict Detection

Conflicts are detected at the **file level** (not line level). If both branches modified the same file to different content, it is reported as a conflict. The merge is **rejected** and the caller receives:

```json
{
    "status": "conflict",
    "conflicts": [
        {
            "path": "src/config.py",
            "base_sha": "aaa...",
            "ours_sha": "bbb...",
            "theirs_sha": "ccc..."
        }
    ]
}
```

The calling agent must resolve conflicts by making additional commits to the source branch, then retry the merge.

## Permission Model

| Operation | Required Permission |
|-----------|-------------------|
| `list_branches` | `read` |
| `create_branch` | `edit` |
| `delete_branch` | `edit` |
| `get_diff` | `read` |
| `create_pr` | `edit` |
| `get_pr` | `read` |
| `list_prs` | `read` |
| `update_pr` | `edit` (creator or admin) |
| `close_pr` | `edit` (creator or admin) |
| `merge_pr` | `commit` |
| `submit_review` | `read` |
| `list_reviews` | `read` |

## Branched Commit Support

Currently `_commit_git()` always commits to `refs/heads/main`. This is extended to accept an optional `branch` parameter:

```python
# Before: always commits to main
await repo.set_ref_async(b"refs/heads/main", commit.id)

# After: commits to specified branch (defaults to default branch)
branch_name = branch or config.get("git_default_branch", "main")
await repo.set_ref_async(f"refs/heads/{branch_name}".encode(), commit.id)
```

The `write_file`, `put_file`, and `commit` methods gain an optional `branch` parameter for git artifacts, threading it through to `_commit_git`.

## File Structure

### New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `hypha/git/merge.py` | Core merge/diff logic (flatten tree, compute diff, 3-way merge, perform merge) | ~350 |
| `tests/test_git_pr.py` | Integration tests for PR system | ~500 |
| `docs/pull-request-design.md` | This design document | — |

### Modified Files

| File | Changes |
|------|---------|
| `hypha/git/repo.py` | Add `delete_ref_async()`, `ensure_packs_loaded()` methods |
| `hypha/artifact.py` | Add 14 new API methods, helper methods, update `_commit_git` for branch targeting, update `get_artifact_service()` |

## Implementation Phases

### Phase 1: Git Merge/Diff Primitives

Create `hypha/git/merge.py` with:
- `flatten_tree_async(repo, tree_sha)` — Recursively flatten git tree to `{path: blob_sha}` dict
- `compute_diff_async(repo, base_sha, head_sha, ...)` — Tree-level diff using dulwich
- `three_way_merge_trees(...)` — 3-way tree merge with conflict detection
- `perform_merge_async(repo, target_branch, source_branch, ...)` — Full merge orchestration

Add to `S3GitRepo`:
- `delete_ref_async(name, old_value)` — Delegate to refs container
- `ensure_packs_loaded()` — Pre-load all packs for synchronous dulwich operations

### Phase 2: Branch Operations

- Implement `create_branch`, `delete_branch`, `list_branches`
- Parameterize `_commit_git` to accept `target_branch`
- Update `write_file` / `put_file` / `commit` for branch support
- Add `_get_git_repo()` and `_validate_git_artifact()` helpers

### Phase 3: Diff and PR CRUD

- Implement `get_diff` (delegates to `compute_diff_async`)
- Implement `create_pr`, `get_pr`, `list_prs`, `update_pr`, `close_pr`
- These are mostly metadata operations on `artifact.config["pull_requests"]`

### Phase 4: Merge and Review

- Implement `merge_pr` (delegates to `perform_merge_async`, updates PR metadata)
- Implement `submit_review`, `list_reviews`
- Update `get_artifact_service()` to expose all new methods

### Phase 5: Testing

Unit tests for merge primitives (MinIO only, no Docker):
- `test_flatten_tree`, `test_compute_diff_*`, `test_fast_forward_detection`
- `test_find_merge_base`, `test_three_way_merge_*`

Integration tests for full PR lifecycle (full server):
- `test_create_and_list_branches`, `test_write_to_branch`
- `test_create_pr_basic`, `test_get_diff_between_branches`
- `test_merge_pr_fast_forward`, `test_merge_pr_three_way`
- `test_merge_pr_conflict_detection`, `test_full_pr_lifecycle`

## Example: Agent Workflow

```python
# Agent A: work on feature branch
svc = await server.get_service("artifact-manager")

# Create a branch
await svc.create_branch("my-repo", branch="feature-auth", source="main")

# Write files to the branch
await svc.edit("my-repo", stage=True, branch="feature-auth")
await svc.write_file("my-repo", "src/auth.py", "import jwt\n...", branch="feature-auth")
await svc.commit("my-repo", comment="Add auth module", branch="feature-auth")

# Open a PR
pr = await svc.create_pr(
    "my-repo",
    title="Add authentication module",
    source_branch="feature-auth",
    target_branch="main",
    description="JWT-based auth with refresh token support."
)
print(f"PR #{pr['pr_number']} created")

# Review Agent: inspect and merge
diff = await svc.get_diff("my-repo", base="main", head="feature-auth", include_content=True)
print(f"Files changed: {diff['stats']['files_changed']}")

review = await svc.submit_review("my-repo", pr["pr_number"], verdict="approve", comment="LGTM")

result = await svc.merge_pr("my-repo", pr["pr_number"])
print(f"Merged via {result['merge_type']}: {result['merge_sha']}")
```

## Future Extensions

- **Auto-merge policies**: Merge automatically when N approvals received
- **CI integration**: Run tests on PR branches before allowing merge
- **Rebase merge strategy**: Replay commits on top of target branch
- **Line-level diff**: Text-level diff for modified files (useful for code review agents)
- **PR comments**: Threaded comments on specific files/lines
- **Webhook/event notifications**: Notify agents when PRs are created/merged
- **Cross-artifact PRs**: Merge between different artifacts (fork model)
