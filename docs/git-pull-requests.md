# Git Storage & Pull Requests

Hypha artifacts can use **git-backed storage**, turning each artifact into a full Git repository stored in S3. On top of this, Hypha provides a **Pull Request (PR) system** that enables multiple AI agents (or humans) to collaborate on the same codebase through isolated branches, structured diffs, reviews, and merges.

**Note:** Git storage and the PR system require S3 storage to be enabled on your Hypha server.

---

## Quick Start

### Step 1: Connect to the Artifact Manager

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({
    "name": "my-agent",
    "server_url": "https://hypha.aicell.io",
})
artifact_manager = await server.get_service("public/artifact-manager")
```

### Step 2: Create a Git-Storage Artifact

```python
await artifact_manager.create(
    alias="my-repo",
    manifest={"name": "My Project", "description": "A collaborative codebase"},
    config={"storage": "git"},
)
```

### Step 3: Push an Initial Commit

You can push files programmatically via the Hypha API:

```python
# Enter staging mode
await artifact_manager.edit("my-repo", stage=True)

# Write files
await artifact_manager.write_file("my-repo", "README.md", "# My Project\n")
await artifact_manager.write_file("my-repo", "src/main.py", "print('hello world')\n")

# Commit
await artifact_manager.commit("my-repo", comment="Initial commit")
```

Or use standard Git clients:

```bash
git clone http://git:<TOKEN>@hypha.aicell.io/<workspace>/git/my-repo
cd my-repo
echo "# My Project" > README.md
git add . && git commit -m "Initial commit"
git push origin main
```

---

## Git Storage Features

### Reading Files

```python
# Read from the default branch (main)
result = await artifact_manager.read_file("my-repo", "README.md")
print(result["content"])

# Read from a specific branch
result = await artifact_manager.read_file("my-repo", "src/main.py", version="feature-auth")

# Read from a specific tag or commit SHA
result = await artifact_manager.read_file("my-repo", "config.json", version="v1.0")
```

### Writing Files to a Branch

```python
# Enter staging mode targeting a specific branch
await artifact_manager.edit("my-repo", stage=True, branch="feature-auth")

# Write files (they go to the staging area)
await artifact_manager.write_file("my-repo", "src/auth.py", "import jwt\n...")
await artifact_manager.write_file("my-repo", "tests/test_auth.py", "def test_login(): ...\n")

# Commit to the branch
await artifact_manager.commit("my-repo", comment="Add auth module", branch="feature-auth")
```

### Listing Files

```python
# List files in root directory
files = await artifact_manager.list_files("my-repo")

# List files from a specific branch
files = await artifact_manager.list_files("my-repo", version="feature-auth")
```

---

## Branch Operations

### Create a Branch

```python
# Create a branch from main (default)
await artifact_manager.create_branch("my-repo", branch="feature-auth")

# Create a branch from a specific source (branch, tag, or commit SHA)
await artifact_manager.create_branch("my-repo", branch="hotfix-123", source="v1.0")
```

### List Branches

```python
branches = await artifact_manager.list_branches("my-repo")
# [
#     {"name": "main", "sha": "abc123...", "default": True},
#     {"name": "feature-auth", "sha": "def456..."},
#     {"name": "hotfix-123", "sha": "789abc..."},
# ]
```

### Delete a Branch

```python
await artifact_manager.delete_branch("my-repo", branch="feature-auth")
# Note: The default branch (main) cannot be deleted.
```

---

## Pull Request Workflow

The PR system follows a familiar workflow optimized for AI agents:

```
1. Create branch  ──>  2. Make changes  ──>  3. Create PR
                                                   │
4. Review (get_diff + submit_review)  <────────────┘
       │
       ▼
5. Merge PR  ──>  main branch updated
```

### Create a Pull Request

```python
pr = await artifact_manager.create_pr(
    "my-repo",
    title="Add authentication module",
    source_branch="feature-auth",
    target_branch="main",  # defaults to "main"
    description="Implements JWT-based auth with refresh token support.",
)
print(f"PR #{pr['pr_number']} created")
```

### View a Pull Request

```python
pr = await artifact_manager.get_pr("my-repo", pr_number=1)
print(f"Status: {pr['status']}")          # "open", "merged", or "closed"
print(f"Source: {pr['source_branch']}")    # "feature-auth"
print(f"Target: {pr['target_branch']}")    # "main"
print(f"Created by: {pr['created_by']}")
```

### List Pull Requests

```python
# List all open PRs
open_prs = await artifact_manager.list_prs("my-repo", status="open")

# List all PRs (no filter)
all_prs = await artifact_manager.list_prs("my-repo")

# Paginate
page2 = await artifact_manager.list_prs("my-repo", limit=10, offset=10)
```

### Update a Pull Request

```python
await artifact_manager.update_pr(
    "my-repo",
    pr_number=1,
    title="Add JWT authentication module",
    description="Updated: now includes refresh token rotation.",
)
```

### Close a Pull Request (Without Merging)

```python
await artifact_manager.close_pr(
    "my-repo",
    pr_number=1,
    comment="Superseded by PR #3",
)
```

---

## Comparing Changes (Diff)

The `get_diff` method returns structured JSON optimized for programmatic and AI-agent use, rather than traditional unified patch format.

### Basic Diff

```python
diff = await artifact_manager.get_diff("my-repo", base="main", head="feature-auth")

print(f"Files changed: {diff['stats']['files_changed']}")
print(f"  Additions: {diff['stats']['additions']}")
print(f"  Modifications: {diff['stats']['modifications']}")
print(f"  Deletions: {diff['stats']['deletions']}")

for f in diff["files"]:
    print(f"  {f['status']:10s} {f['path']} ({f.get('new_size', 0)} bytes)")
```

### Diff with File Content

```python
diff = await artifact_manager.get_diff(
    "my-repo",
    base="main",
    head="feature-auth",
    include_content=True,
)

for f in diff["files"]:
    if f["status"] == "modified":
        print(f"--- {f['path']} (old)")
        print(f["old_content"].decode())
        print(f"+++ {f['path']} (new)")
        print(f["new_content"].decode())
```

### Diff with Path Filter

```python
# Only show changes under src/
diff = await artifact_manager.get_diff(
    "my-repo",
    base="main",
    head="feature-auth",
    path_filter="src/",
)
```

### Diff Output Structure

```json
{
    "base_sha": "abc123...",
    "head_sha": "def456...",
    "files": [
        {
            "path": "src/auth.py",
            "status": "added",
            "new_sha": "...",
            "new_size": 2048
        },
        {
            "path": "src/config.py",
            "status": "modified",
            "old_sha": "...",
            "new_sha": "...",
            "old_size": 512,
            "new_size": 540
        },
        {
            "path": "src/deprecated.py",
            "status": "deleted",
            "old_sha": "...",
            "old_size": 128
        }
    ],
    "stats": {
        "files_changed": 3,
        "additions": 1,
        "modifications": 1,
        "deletions": 1
    }
}
```

---

## Code Review

### Submit a Review

```python
# Approve the PR
review = await artifact_manager.submit_review(
    "my-repo",
    pr_number=1,
    verdict="approve",
    comment="Clean implementation, tests look good.",
)

# Request changes
review = await artifact_manager.submit_review(
    "my-repo",
    pr_number=1,
    verdict="request_changes",
    comment="Please add error handling for expired tokens.",
)

# Leave a comment without verdict
review = await artifact_manager.submit_review(
    "my-repo",
    pr_number=1,
    verdict="comment",
    comment="Consider using asymmetric keys for better security.",
)
```

### List Reviews

```python
reviews = await artifact_manager.list_reviews("my-repo", pr_number=1)
for r in reviews:
    print(f"[{r['verdict']}] {r['reviewer']}: {r['comment']}")
```

---

## Merging

### Merge a Pull Request

```python
result = await artifact_manager.merge_pr("my-repo", pr_number=1)
print(f"Merge type: {result['merge_type']}")  # "fast-forward" or "merge"
print(f"Merge SHA: {result['merge_sha']}")
```

### Merge Strategies

| Strategy | Behavior |
|----------|----------|
| `"auto"` (default) | Try fast-forward first; if branches have diverged, create a merge commit |
| `"fast-forward"` | Only fast-forward; fails if branches have diverged |
| `"merge"` | Always create a merge commit, even if fast-forward is possible |

```python
# Force a merge commit
result = await artifact_manager.merge_pr(
    "my-repo",
    pr_number=1,
    merge_strategy="merge",
    comment="Merge feature-auth into main",
)

# Keep the source branch after merge
result = await artifact_manager.merge_pr(
    "my-repo",
    pr_number=1,
    delete_branch=False,
)
```

### Handling Merge Conflicts

When both branches modify the same file differently, the merge returns a conflict:

```python
result = await artifact_manager.merge_pr("my-repo", pr_number=1)

if result["status"] == "conflict":
    print("Conflicts detected:")
    for conflict in result["conflicts"]:
        print(f"  {conflict['path']}")
        print(f"    base:   {conflict['base_sha']}")
        print(f"    ours:   {conflict['ours_sha']}")
        print(f"    theirs: {conflict['theirs_sha']}")

    # To resolve: push new commits to the source branch, then retry
    # The PR stays open and can be merged once conflicts are resolved
```

Conflict detection is at the **file level** (not line level). If both branches modified the same file to different content, it is reported as a conflict. The calling agent must resolve conflicts by making additional commits to the source branch, then retry the merge.

---

## Full Example: AI Agent Swarm Collaboration

This example shows how multiple AI agents can collaborate on a shared codebase through the PR workflow.

```python
from hypha_rpc import connect_to_server

# ──────────────────────────────────────────────────────
# Setup: Create a shared repository
# ──────────────────────────────────────────────────────

server = await connect_to_server({
    "name": "orchestrator",
    "server_url": "https://hypha.aicell.io",
})
svc = await server.get_service("public/artifact-manager")

# Create the shared repo with an initial file
await svc.create(
    alias="science-project",
    manifest={"name": "Collaborative Science Project"},
    config={"storage": "git"},
)
await svc.edit("science-project", stage=True)
await svc.write_file("science-project", "README.md", "# Science Project\n\nA collaborative codebase.\n")
await svc.write_file("science-project", "src/__init__.py", "")
await svc.commit("science-project", comment="Initial project structure")

# ──────────────────────────────────────────────────────
# Agent A: Implement data loading
# ──────────────────────────────────────────────────────

await svc.create_branch("science-project", branch="feature-data-loader")

await svc.edit("science-project", stage=True, branch="feature-data-loader")
await svc.write_file(
    "science-project", "src/data_loader.py",
    "import pandas as pd\n\n"
    "def load_csv(path: str) -> pd.DataFrame:\n"
    "    return pd.read_csv(path)\n\n"
    "def load_parquet(path: str) -> pd.DataFrame:\n"
    "    return pd.read_parquet(path)\n",
)
await svc.write_file(
    "science-project", "tests/test_data_loader.py",
    "from src.data_loader import load_csv\n\n"
    "def test_load_csv(tmp_path):\n"
    "    p = tmp_path / 'test.csv'\n"
    "    p.write_text('a,b\\n1,2\\n')\n"
    "    df = load_csv(str(p))\n"
    "    assert len(df) == 1\n",
)
await svc.commit("science-project", comment="Add data loader module", branch="feature-data-loader")

pr_a = await svc.create_pr(
    "science-project",
    title="Add data loading module",
    source_branch="feature-data-loader",
    description="CSV and Parquet loading with pandas.",
)
print(f"Agent A created PR #{pr_a['pr_number']}")

# ──────────────────────────────────────────────────────
# Agent B: Implement model training (in parallel)
# ──────────────────────────────────────────────────────

await svc.create_branch("science-project", branch="feature-model-training")

await svc.edit("science-project", stage=True, branch="feature-model-training")
await svc.write_file(
    "science-project", "src/model.py",
    "from sklearn.ensemble import RandomForestClassifier\n\n"
    "def train(X, y):\n"
    "    clf = RandomForestClassifier(n_estimators=100)\n"
    "    clf.fit(X, y)\n"
    "    return clf\n",
)
await svc.commit("science-project", comment="Add model training module", branch="feature-model-training")

pr_b = await svc.create_pr(
    "science-project",
    title="Add model training pipeline",
    source_branch="feature-model-training",
    description="Random forest classifier with sklearn.",
)
print(f"Agent B created PR #{pr_b['pr_number']}")

# ──────────────────────────────────────────────────────
# Review Agent: Review and merge both PRs
# ──────────────────────────────────────────────────────

# Review PR #1 (data loader)
diff_a = await svc.get_diff(
    "science-project",
    base="main",
    head="feature-data-loader",
    include_content=True,
)
print(f"\nPR #{pr_a['pr_number']} diff: {diff_a['stats']['files_changed']} files changed")
for f in diff_a["files"]:
    print(f"  {f['status']:10s} {f['path']}")

await svc.submit_review(
    "science-project",
    pr_number=pr_a["pr_number"],
    verdict="approve",
    comment="Data loader looks clean. Good test coverage.",
)

# Merge PR #1 (fast-forward since main hasn't changed)
result_a = await svc.merge_pr("science-project", pr_number=pr_a["pr_number"])
print(f"PR #{pr_a['pr_number']} merged via {result_a['merge_type']}")

# Review PR #2 (model training)
diff_b = await svc.get_diff(
    "science-project",
    base="main",
    head="feature-model-training",
    include_content=True,
)
print(f"\nPR #{pr_b['pr_number']} diff: {diff_b['stats']['files_changed']} files changed")

await svc.submit_review(
    "science-project",
    pr_number=pr_b["pr_number"],
    verdict="approve",
    comment="LGTM. Consider adding hyperparameter tuning later.",
)

# Merge PR #2 (3-way merge since main now has PR #1's changes)
result_b = await svc.merge_pr("science-project", pr_number=pr_b["pr_number"])
print(f"PR #{pr_b['pr_number']} merged via {result_b['merge_type']}")

# ──────────────────────────────────────────────────────
# Verify: main now has both features
# ──────────────────────────────────────────────────────

files = await svc.list_files("science-project")
print("\nFiles in main:")
for f in files:
    print(f"  {f['name']}")

# Read the merged code
data_loader = await svc.read_file("science-project", "src/data_loader.py")
model = await svc.read_file("science-project", "src/model.py")
print("\nData loader:", data_loader["content"][:50], "...")
print("Model:", model["content"][:50], "...")
```

**Expected output:**

```
Agent A created PR #1
Agent B created PR #2

PR #1 diff: 2 files changed
  added      src/data_loader.py
  added      tests/test_data_loader.py
PR #1 merged via fast-forward

PR #2 diff: 1 files changed
  added      src/model.py
PR #2 merged via merge

Files in main:
  README.md
  src
  tests

Data loader: import pandas as pd

def load_csv(path: str)  ...
Model: from sklearn.ensemble import RandomForestClassi ...
```

---

## Handling Conflicts: Agent Resolution Pattern

When two agents modify the same file, the review agent can detect and delegate conflict resolution:

```python
# Attempt merge - returns conflict
result = await svc.merge_pr("science-project", pr_number=3)

if result["status"] == "conflict":
    for conflict in result["conflicts"]:
        # Read both versions of the conflicting file
        ours = await svc.read_file("science-project", conflict["path"], version="main")
        theirs = await svc.read_file("science-project", conflict["path"], version="feature-x")

        # AI agent resolves the conflict by creating a merged version
        resolved_content = resolve_conflict(ours["content"], theirs["content"])

        # Push the resolution to the source branch
        await svc.edit("science-project", stage=True, branch="feature-x")
        await svc.write_file("science-project", conflict["path"], resolved_content)
        await svc.commit("science-project", comment=f"Resolve conflict in {conflict['path']}", branch="feature-x")

    # Retry merge after resolution
    result = await svc.merge_pr("science-project", pr_number=3)
    assert result["status"] == "merged"
```

---

## Using Standard Git Clients

Git-storage artifacts are accessible via standard Git over HTTP. You can clone, push, and pull using any Git client.

### Authentication

Use the Hypha token as the password with `git` as the username:

```bash
# Clone
git clone http://git:<TOKEN>@hypha.aicell.io/<workspace>/git/<artifact-alias>

# Or configure credentials
git remote set-url origin http://git:<TOKEN>@hypha.aicell.io/<workspace>/git/<artifact-alias>
```

### Git LFS

Large files are supported via Git LFS:

```bash
git lfs install
git lfs track "*.bin" "*.h5" "*.pt"
git add .gitattributes
git add model.pt
git commit -m "Add model weights"
git push origin main
```

---

## API Reference

### Branch Operations

| Method | Parameters | Description |
|--------|-----------|-------------|
| `create_branch(artifact_id, branch, source?)` | `source`: branch/tag/SHA to branch from (default: default branch) | Create a new branch |
| `delete_branch(artifact_id, branch)` | Cannot delete default branch | Delete a branch |
| `list_branches(artifact_id)` | Returns `[{name, sha, default?}]` | List all branches |

### Pull Request Operations

| Method | Parameters | Description |
|--------|-----------|-------------|
| `create_pr(artifact_id, title, source_branch, target_branch?, description?)` | `target_branch` defaults to `"main"` | Create a PR |
| `get_pr(artifact_id, pr_number)` | Includes current branch SHAs for open PRs | Get PR details |
| `list_prs(artifact_id, status?, limit?, offset?)` | `status`: `"open"`, `"merged"`, `"closed"` | List PRs |
| `update_pr(artifact_id, pr_number, title?, description?)` | Only open PRs can be updated | Update PR metadata |
| `close_pr(artifact_id, pr_number, comment?)` | Sets status to `"closed"` | Close without merging |
| `merge_pr(artifact_id, pr_number, merge_strategy?, comment?, delete_branch?)` | `merge_strategy`: `"auto"`, `"fast-forward"`, `"merge"` | Merge a PR |

### Diff and Review

| Method | Parameters | Description |
|--------|-----------|-------------|
| `get_diff(artifact_id, base, head, include_content?, path_filter?)` | `base`/`head`: branch, tag, or SHA | Structured diff |
| `submit_review(artifact_id, pr_number, verdict, comment?)` | `verdict`: `"approve"`, `"request_changes"`, `"comment"` | Submit a review |
| `list_reviews(artifact_id, pr_number)` | Ordered by creation time | List reviews |

### Permission Requirements

| Operation | Permission Level |
|-----------|-----------------|
| `list_branches`, `get_diff`, `get_pr`, `list_prs`, `list_reviews` | `read` |
| `create_branch`, `delete_branch`, `create_pr`, `update_pr`, `close_pr` | `edit` |
| `submit_review` | `read` |
| `merge_pr` | `commit` |
