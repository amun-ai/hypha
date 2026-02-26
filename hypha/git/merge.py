"""Merge and diff logic for the S3-backed Git PR system.

This module implements tree-level merge operations, diff computation,
and ref-based merge orchestration on top of S3GitRepo objects.
All operations are fully async and use the repo's async API to read
and write git objects stored in S3.

Key functions:
- flatten_tree_async: Recursively flatten a git tree into a path dict
- compute_diff_async: Structured diff between two commits
- find_merge_base_async: Find the common ancestor of two commits
- three_way_merge_trees: Perform a 3-way merge at tree level
- perform_merge_async: Full merge orchestration (ff, 3-way, auto)
"""

import logging
import time
from collections import deque
from typing import Optional

from dulwich.objects import Blob, Commit, Tree

logger = logging.getLogger(__name__)


async def flatten_tree_async(repo, tree_sha, prefix=""):
    """Recursively flatten a git tree into a dict of {path: {"sha": blob_sha, "mode": mode}}.

    Walks through the tree object and all nested subtrees, producing a flat
    mapping from file paths to their blob SHA and file mode.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        tree_sha: The SHA of the tree object to flatten. Can be 40-char hex
                  bytes or 20-byte binary SHA.
        prefix: Path prefix for nested calls (used during recursion).

    Returns:
        Dict mapping file paths (str) to dicts with keys:
        - "sha": 40-char hex bytes of the blob
        - "mode": integer file mode (e.g. 0o100644)

    Example:
        {"src/main.py": {"sha": b"abc123...", "mode": 0o100644}}
    """
    result = {}

    # Normalize SHA: convert 20-byte binary to 40-char hex bytes
    if isinstance(tree_sha, bytes) and len(tree_sha) == 20:
        tree_sha = tree_sha.hex().encode()

    tree = await repo.get_object_async(tree_sha)

    for entry in tree.items():
        name, mode, sha = entry
        entry_name = name.decode("utf-8", errors="replace")

        if prefix:
            full_path = f"{prefix}/{entry_name}"
        else:
            full_path = entry_name

        # Normalize sha to 40-char hex bytes
        if isinstance(sha, bytes) and len(sha) == 20:
            sha = sha.hex().encode()

        if mode == 0o40000:
            # Subtree: recurse into it
            subtree_entries = await flatten_tree_async(repo, sha, prefix=full_path)
            result.update(subtree_entries)
        else:
            result[full_path] = {"sha": sha, "mode": mode}

    return result


async def resolve_ref_to_sha(repo, ref_str):
    """Resolve a ref string to a commit SHA.

    Tries multiple resolution strategies in order:
    1. refs/heads/{ref_str} (branch name)
    2. refs/tags/{ref_str} (tag name)
    3. Raw SHA hex (if ref_str looks like a valid hex SHA)

    Args:
        repo: An S3GitRepo instance (must be initialized).
        ref_str: A string that is either a branch name, tag name,
                 or 40-char hex SHA.

    Returns:
        40-char hex bytes SHA of the resolved commit.

    Raises:
        ValueError: If the ref cannot be resolved to any known object.
    """
    if isinstance(ref_str, bytes):
        ref_str = ref_str.decode("utf-8")

    refs = await repo.get_refs_async()

    # Try as branch: refs/heads/{ref_str}
    branch_ref = f"refs/heads/{ref_str}".encode()
    if branch_ref in refs:
        value = refs[branch_ref]
        # Resolve symbolic refs
        if value.startswith(b"ref: "):
            target = value[5:]
            if target in refs:
                return refs[target]
        return value

    # Try as tag: refs/tags/{ref_str}
    tag_ref = f"refs/tags/{ref_str}".encode()
    if tag_ref in refs:
        value = refs[tag_ref]
        if value.startswith(b"ref: "):
            target = value[5:]
            if target in refs:
                return refs[target]
        return value

    # Try as raw SHA hex
    ref_bytes = ref_str.encode() if isinstance(ref_str, str) else ref_str
    if len(ref_bytes) == 40:
        # Verify the object exists
        if await repo.has_object_async(ref_bytes):
            return ref_bytes

    raise ValueError(
        f"Cannot resolve ref '{ref_str}': not a known branch, tag, or commit SHA"
    )


async def compute_diff_async(
    repo, base_sha, head_sha, include_content=False, path_filter=None
):
    """Compute a structured diff between two commits.

    Flattens the tree of each commit and compares them path by path
    to identify additions, deletions, and modifications.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        base_sha: 40-char hex bytes SHA of the base commit.
        head_sha: 40-char hex bytes SHA of the head commit.
        include_content: If True, include old/new blob content in results.
        path_filter: If set, only include files whose paths start with
                     this prefix string.

    Returns:
        Dict with keys:
        - "base_sha": hex string of the base commit
        - "head_sha": hex string of the head commit
        - "files": list of file dicts, each with:
            - "path": file path string
            - "status": "added", "modified", or "deleted"
            - "old_sha": hex string or None
            - "new_sha": hex string or None
            - "old_size": int or None
            - "new_size": int or None
            - "old_content": bytes (only when include_content=True)
            - "new_content": bytes (only when include_content=True)
        - "stats": dict with files_changed, additions, deletions, modifications
    """
    # Normalize SHAs to 40-char hex bytes
    if isinstance(base_sha, bytes) and len(base_sha) == 20:
        base_sha = base_sha.hex().encode()
    if isinstance(head_sha, bytes) and len(head_sha) == 20:
        head_sha = head_sha.hex().encode()

    # Get commit objects and flatten their trees
    base_commit = await repo.get_object_async(base_sha)
    head_commit = await repo.get_object_async(head_sha)

    base_tree_flat = await flatten_tree_async(repo, base_commit.tree)
    head_tree_flat = await flatten_tree_async(repo, head_commit.tree)

    base_paths = set(base_tree_flat.keys())
    head_paths = set(head_tree_flat.keys())

    files = []
    additions = 0
    deletions = 0
    modifications = 0

    # Additions: paths in head but not in base
    for path in sorted(head_paths - base_paths):
        if path_filter and not path.startswith(path_filter):
            continue

        entry = head_tree_flat[path]
        new_sha_hex = _sha_to_hex_str(entry["sha"])

        file_info = {
            "path": path,
            "status": "added",
            "old_sha": None,
            "new_sha": new_sha_hex,
            "old_size": None,
            "new_size": None,
        }

        if include_content:
            new_blob = await repo.get_object_async(entry["sha"])
            file_info["new_size"] = len(new_blob.data)
            file_info["old_content"] = None
            file_info["new_content"] = new_blob.data
        else:
            # Fetch size even without content
            new_blob = await repo.get_object_async(entry["sha"])
            file_info["new_size"] = len(new_blob.data)

        files.append(file_info)
        additions += 1

    # Deletions: paths in base but not in head
    for path in sorted(base_paths - head_paths):
        if path_filter and not path.startswith(path_filter):
            continue

        entry = base_tree_flat[path]
        old_sha_hex = _sha_to_hex_str(entry["sha"])

        file_info = {
            "path": path,
            "status": "deleted",
            "old_sha": old_sha_hex,
            "new_sha": None,
            "old_size": None,
            "new_size": None,
        }

        if include_content:
            old_blob = await repo.get_object_async(entry["sha"])
            file_info["old_size"] = len(old_blob.data)
            file_info["old_content"] = old_blob.data
            file_info["new_content"] = None
        else:
            old_blob = await repo.get_object_async(entry["sha"])
            file_info["old_size"] = len(old_blob.data)

        files.append(file_info)
        deletions += 1

    # Modifications: paths in both, with different SHAs
    for path in sorted(base_paths & head_paths):
        if path_filter and not path.startswith(path_filter):
            continue

        base_entry = base_tree_flat[path]
        head_entry = head_tree_flat[path]

        if base_entry["sha"] == head_entry["sha"]:
            continue  # Unchanged

        old_sha_hex = _sha_to_hex_str(base_entry["sha"])
        new_sha_hex = _sha_to_hex_str(head_entry["sha"])

        file_info = {
            "path": path,
            "status": "modified",
            "old_sha": old_sha_hex,
            "new_sha": new_sha_hex,
            "old_size": None,
            "new_size": None,
        }

        if include_content:
            old_blob = await repo.get_object_async(base_entry["sha"])
            new_blob = await repo.get_object_async(head_entry["sha"])
            file_info["old_size"] = len(old_blob.data)
            file_info["new_size"] = len(new_blob.data)
            file_info["old_content"] = old_blob.data
            file_info["new_content"] = new_blob.data
        else:
            old_blob = await repo.get_object_async(base_entry["sha"])
            new_blob = await repo.get_object_async(head_entry["sha"])
            file_info["old_size"] = len(old_blob.data)
            file_info["new_size"] = len(new_blob.data)

        files.append(file_info)
        modifications += 1

    base_sha_str = _sha_to_hex_str(base_sha)
    head_sha_str = _sha_to_hex_str(head_sha)

    return {
        "base_sha": base_sha_str,
        "head_sha": head_sha_str,
        "files": files,
        "stats": {
            "files_changed": additions + deletions + modifications,
            "additions": additions,
            "deletions": deletions,
            "modifications": modifications,
        },
    }


async def find_merge_base_async(repo, commit_sha1, commit_sha2):
    """Find the merge base (common ancestor) of two commits.

    Uses a BFS walk from both commits simultaneously, tracking visited
    ancestors from each side. The first commit found in both ancestor sets
    is the merge base.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        commit_sha1: 40-char hex bytes SHA of the first commit.
        commit_sha2: 40-char hex bytes SHA of the second commit.

    Returns:
        40-char hex bytes SHA of the merge base, or None if the two
        commits have disjoint histories (no common ancestor).
    """
    # Normalize SHAs
    if isinstance(commit_sha1, bytes) and len(commit_sha1) == 20:
        commit_sha1 = commit_sha1.hex().encode()
    if isinstance(commit_sha2, bytes) and len(commit_sha2) == 20:
        commit_sha2 = commit_sha2.hex().encode()

    # BFS queues for each side
    queue1 = deque([commit_sha1])
    queue2 = deque([commit_sha2])

    # Sets of visited ancestors from each side
    visited1 = {commit_sha1}
    visited2 = {commit_sha2}

    # Check trivial case: same commit
    if commit_sha1 == commit_sha2:
        return commit_sha1

    while queue1 or queue2:
        # Expand from side 1
        if queue1:
            current = queue1.popleft()
            commit = await repo.get_object_async(current)
            for parent_sha in commit.parents:
                # Normalize parent SHA
                if isinstance(parent_sha, bytes) and len(parent_sha) == 20:
                    parent_sha = parent_sha.hex().encode()

                if parent_sha in visited2:
                    return parent_sha

                if parent_sha not in visited1:
                    visited1.add(parent_sha)
                    queue1.append(parent_sha)

        # Expand from side 2
        if queue2:
            current = queue2.popleft()
            commit = await repo.get_object_async(current)
            for parent_sha in commit.parents:
                if isinstance(parent_sha, bytes) and len(parent_sha) == 20:
                    parent_sha = parent_sha.hex().encode()

                if parent_sha in visited1:
                    return parent_sha

                if parent_sha not in visited2:
                    visited2.add(parent_sha)
                    queue2.append(parent_sha)

    # No common ancestor found
    return None


async def can_fast_forward_async(repo, target_sha, source_sha):
    """Check if target_sha is an ancestor of source_sha (fast-forward possible).

    Walks backwards from source_sha through parent commits. If target_sha
    is encountered, a fast-forward merge is possible.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        target_sha: 40-char hex bytes SHA of the target (current) branch tip.
        source_sha: 40-char hex bytes SHA of the source branch tip.

    Returns:
        True if target_sha is reachable from source_sha (fast-forward possible),
        False otherwise.
    """
    # Normalize SHAs
    if isinstance(target_sha, bytes) and len(target_sha) == 20:
        target_sha = target_sha.hex().encode()
    if isinstance(source_sha, bytes) and len(source_sha) == 20:
        source_sha = source_sha.hex().encode()

    # Trivial case
    if target_sha == source_sha:
        return True

    # BFS from source back through parents
    queue = deque([source_sha])
    visited = {source_sha}

    while queue:
        current = queue.popleft()
        commit = await repo.get_object_async(current)

        for parent_sha in commit.parents:
            if isinstance(parent_sha, bytes) and len(parent_sha) == 20:
                parent_sha = parent_sha.hex().encode()

            if parent_sha == target_sha:
                return True

            if parent_sha not in visited:
                visited.add(parent_sha)
                queue.append(parent_sha)

    return False


async def three_way_merge_trees(repo, base_tree_flat, ours_tree_flat, theirs_tree_flat):
    """Perform a 3-way merge at the tree level.

    Compares each path across the base, ours, and theirs trees and determines
    the correct merge result or identifies conflicts.

    Merge rules for each path:
    - Only theirs changed (base==ours, theirs differs): take theirs
    - Only ours changed (base==theirs, ours differs): keep ours
    - Both changed to same value: keep either (no conflict)
    - Both changed differently: CONFLICT
    - Added only in theirs: take theirs
    - Added only in ours: keep ours
    - Added in both with same SHA: keep either
    - Added in both with different SHA: CONFLICT
    - Deleted only in theirs: delete (omit from result)
    - Deleted only in ours: delete (omit from result)
    - Deleted in one, modified in other: CONFLICT

    Args:
        repo: An S3GitRepo instance (unused in tree-level merge but kept
              for API consistency and potential future content-level merge).
        base_tree_flat: Flat tree dict from the merge base commit.
        ours_tree_flat: Flat tree dict from our (target) branch.
        theirs_tree_flat: Flat tree dict from their (source) branch.

    Returns:
        Tuple of (merged_flat, conflicts) where:
        - merged_flat: dict of {path: {"sha": ..., "mode": ...}} for the
          merged result (conflicts excluded from this dict)
        - conflicts: list of dicts with keys:
            - "path": conflicting file path
            - "base_sha": SHA in base (hex str or None)
            - "ours_sha": SHA in ours (hex str or None)
            - "theirs_sha": SHA in theirs (hex str or None)
    """
    merged = {}
    conflicts = []

    all_paths = set(base_tree_flat.keys()) | set(ours_tree_flat.keys()) | set(theirs_tree_flat.keys())

    for path in sorted(all_paths):
        base_entry = base_tree_flat.get(path)
        ours_entry = ours_tree_flat.get(path)
        theirs_entry = theirs_tree_flat.get(path)

        base_sha = base_entry["sha"] if base_entry else None
        ours_sha = ours_entry["sha"] if ours_entry else None
        theirs_sha = theirs_entry["sha"] if theirs_entry else None

        # Case 1: Path exists in all three
        if base_entry and ours_entry and theirs_entry:
            if base_sha == ours_sha == theirs_sha:
                # All same: no change
                merged[path] = ours_entry
            elif base_sha == ours_sha and theirs_sha != base_sha:
                # Only theirs changed
                merged[path] = theirs_entry
            elif base_sha == theirs_sha and ours_sha != base_sha:
                # Only ours changed
                merged[path] = ours_entry
            elif ours_sha == theirs_sha:
                # Both changed to the same thing
                merged[path] = ours_entry
            else:
                # Both changed differently: CONFLICT
                conflicts.append({
                    "path": path,
                    "base_sha": _sha_to_hex_str(base_sha),
                    "ours_sha": _sha_to_hex_str(ours_sha),
                    "theirs_sha": _sha_to_hex_str(theirs_sha),
                })

        # Case 2: Path not in base (newly added)
        elif not base_entry:
            if ours_entry and theirs_entry:
                if ours_sha == theirs_sha:
                    # Both added the same file
                    merged[path] = ours_entry
                else:
                    # Both added different content: CONFLICT
                    conflicts.append({
                        "path": path,
                        "base_sha": None,
                        "ours_sha": _sha_to_hex_str(ours_sha),
                        "theirs_sha": _sha_to_hex_str(theirs_sha),
                    })
            elif ours_entry:
                # Added only in ours
                merged[path] = ours_entry
            elif theirs_entry:
                # Added only in theirs
                merged[path] = theirs_entry

        # Case 3: Path in base but missing from one or both sides
        elif base_entry:
            if ours_entry and not theirs_entry:
                # Deleted in theirs
                if ours_sha == base_sha:
                    # Ours unchanged, theirs deleted: delete
                    pass  # Omit from merged
                else:
                    # Ours modified, theirs deleted: CONFLICT
                    conflicts.append({
                        "path": path,
                        "base_sha": _sha_to_hex_str(base_sha),
                        "ours_sha": _sha_to_hex_str(ours_sha),
                        "theirs_sha": None,
                    })
            elif theirs_entry and not ours_entry:
                # Deleted in ours
                if theirs_sha == base_sha:
                    # Theirs unchanged, ours deleted: delete
                    pass  # Omit from merged
                else:
                    # Theirs modified, ours deleted: CONFLICT
                    conflicts.append({
                        "path": path,
                        "base_sha": _sha_to_hex_str(base_sha),
                        "ours_sha": None,
                        "theirs_sha": _sha_to_hex_str(theirs_sha),
                    })
            else:
                # Deleted in both: delete (no conflict)
                pass  # Omit from merged

    return merged, conflicts


async def build_tree_from_flat(repo, flat_tree):
    """Build git tree objects from a flat path-to-entry dict.

    Groups entries by directory, creates Tree objects bottom-up (deepest
    directories first), and stores all new objects in the repo's object store.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        flat_tree: Dict mapping file paths to {"sha": ..., "mode": ...} dicts,
                   as produced by flatten_tree_async or three_way_merge_trees.

    Returns:
        40-char hex bytes SHA of the root tree object.
    """
    if not flat_tree:
        # Empty tree
        tree = Tree()
        objects_to_add = [(tree, None)]
        await repo._object_store.add_objects_async(objects_to_add)
        return tree.id

    # Build a nested directory structure
    # dir_entries[dir_path] = list of (name_bytes, mode, sha)
    dir_entries = {}

    for path, entry in flat_tree.items():
        parts = path.split("/")
        filename = parts[-1]
        dir_path = "/".join(parts[:-1]) if len(parts) > 1 else ""

        if dir_path not in dir_entries:
            dir_entries[dir_path] = []

        sha = entry["sha"]
        # Ensure SHA is 40-char hex bytes
        if isinstance(sha, bytes) and len(sha) == 20:
            sha = sha.hex().encode()

        dir_entries[dir_path].append((filename.encode("utf-8"), entry["mode"], sha))

    # Collect all unique directory paths and sort deepest first
    all_dirs = set(dir_entries.keys())
    # Also add parent directories that might only contain subdirectories
    for path in list(flat_tree.keys()):
        parts = path.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            all_dirs.add(parent)
    # Always include root
    all_dirs.add("")

    # Sort by depth (deepest first) to build bottom-up
    sorted_dirs = sorted(all_dirs, key=lambda d: d.count("/"), reverse=True)

    # Track SHA for each directory tree
    dir_sha = {}
    objects_to_add = []

    for dir_path in sorted_dirs:
        tree = Tree()

        # Add file entries for this directory
        if dir_path in dir_entries:
            for name, mode, sha in sorted(dir_entries[dir_path], key=lambda e: e[0]):
                tree.add(name, mode, sha)

        # Add subdirectory entries
        for other_dir in all_dirs:
            if other_dir == dir_path:
                continue
            # Check if other_dir is a direct child of dir_path
            if dir_path == "":
                # Root: direct children have no "/" in them (after stripping)
                if "/" not in other_dir and other_dir != "" and other_dir in dir_sha:
                    subdir_name = other_dir.encode("utf-8")
                    tree.add(subdir_name, 0o40000, dir_sha[other_dir])
            else:
                # Non-root: check if other_dir starts with dir_path/ and has
                # no further / after the prefix
                prefix = dir_path + "/"
                if other_dir.startswith(prefix):
                    remainder = other_dir[len(prefix):]
                    if "/" not in remainder and other_dir in dir_sha:
                        subdir_name = remainder.encode("utf-8")
                        tree.add(subdir_name, 0o40000, dir_sha[other_dir])

        dir_sha[dir_path] = tree.id
        objects_to_add.append((tree, None))

    # Store all tree objects
    await repo._object_store.add_objects_async(objects_to_add)

    return dir_sha[""]


async def perform_merge_async(
    repo,
    target_branch,
    source_branch,
    strategy="auto",
    author_name="Hypha",
    author_email="hypha@users.noreply.github.com",
    message=None,
):
    """Perform a full merge of source_branch into target_branch.

    Supports three strategies:
    - "auto": try fast-forward first, fall back to 3-way merge
    - "fast-forward": only fast-forward, raise ValueError if not possible
    - "merge": always create a 3-way merge commit

    For fast-forward merges, the target ref is simply updated to point at
    the source commit.

    For 3-way merges, the merge base is found, trees are flattened and merged,
    a new tree is built, and a merge commit with two parents is created.

    If conflicts are detected during 3-way merge, no refs are modified and
    the conflicts are returned.

    Args:
        repo: An S3GitRepo instance (must be initialized).
        target_branch: Name of the target branch (e.g. "main").
        source_branch: Name of the source branch (e.g. "feature-x").
        strategy: Merge strategy - "auto", "fast-forward", or "merge".
        author_name: Name for the merge commit author.
        author_email: Email for the merge commit author.
        message: Commit message. Defaults to "Merge {source} into {target}".

    Returns:
        Dict with keys:
        - "status": "merged" or "conflict"
        - "merge_type": "fast-forward" or "merge"
        - "merge_sha": hex string of the resulting commit SHA (or None on conflict)
        - "conflicts": list of conflict dicts (empty if no conflicts)

    Raises:
        ValueError: If strategy is "fast-forward" and fast-forward is not possible,
                    or if a branch ref cannot be resolved.
    """
    if message is None:
        message = f"Merge {source_branch} into {target_branch}"

    # Resolve branch refs to commit SHAs
    target_sha = await resolve_ref_to_sha(repo, target_branch)
    source_sha = await resolve_ref_to_sha(repo, source_branch)

    logger.info(
        "Merging %s (%s) into %s (%s) with strategy=%s",
        source_branch,
        _sha_to_hex_str(source_sha)[:12],
        target_branch,
        _sha_to_hex_str(target_sha)[:12],
        strategy,
    )

    # Check if already up to date
    if target_sha == source_sha:
        logger.info("Already up to date")
        return {
            "status": "merged",
            "merge_type": "fast-forward",
            "merge_sha": _sha_to_hex_str(target_sha),
            "conflicts": [],
        }

    # Determine the target ref name
    target_ref = f"refs/heads/{target_branch}".encode()

    # Strategy: fast-forward
    if strategy in ("auto", "fast-forward"):
        ff_possible = await can_fast_forward_async(repo, target_sha, source_sha)

        if ff_possible:
            logger.info("Fast-forward merge possible")
            success = await repo.set_ref_async(target_ref, source_sha, target_sha)
            if not success:
                raise ValueError(
                    f"Failed to update ref {target_ref.decode()}: "
                    "concurrent modification detected"
                )
            return {
                "status": "merged",
                "merge_type": "fast-forward",
                "merge_sha": _sha_to_hex_str(source_sha),
                "conflicts": [],
            }

        if strategy == "fast-forward":
            raise ValueError(
                f"Fast-forward not possible: {target_branch} has diverged "
                f"from {source_branch}. Use strategy='auto' or 'merge'."
            )

        logger.info("Fast-forward not possible, falling back to 3-way merge")

    # 3-way merge
    merge_base_sha = await find_merge_base_async(repo, target_sha, source_sha)
    if merge_base_sha is None:
        raise ValueError(
            f"Cannot merge: no common ancestor between {target_branch} and "
            f"{source_branch} (disjoint histories)"
        )

    logger.info("Merge base: %s", _sha_to_hex_str(merge_base_sha)[:12])

    # Flatten the three trees
    base_commit = await repo.get_object_async(merge_base_sha)
    ours_commit = await repo.get_object_async(target_sha)
    theirs_commit = await repo.get_object_async(source_sha)

    base_tree_flat = await flatten_tree_async(repo, base_commit.tree)
    ours_tree_flat = await flatten_tree_async(repo, ours_commit.tree)
    theirs_tree_flat = await flatten_tree_async(repo, theirs_commit.tree)

    # Perform the 3-way merge
    merged_flat, conflicts = await three_way_merge_trees(
        repo, base_tree_flat, ours_tree_flat, theirs_tree_flat
    )

    if conflicts:
        logger.warning("Merge conflicts detected in %d files", len(conflicts))
        return {
            "status": "conflict",
            "merge_type": "merge",
            "merge_sha": None,
            "conflicts": conflicts,
        }

    # Build the merged tree
    merged_tree_sha = await build_tree_from_flat(repo, merged_flat)

    # Create the merge commit
    now = int(time.time())
    author_line = f"{author_name} <{author_email}>".encode("utf-8")

    merge_commit = Commit()
    merge_commit.tree = merged_tree_sha
    merge_commit.parents = [target_sha, source_sha]
    merge_commit.author = author_line
    merge_commit.committer = author_line
    merge_commit.author_time = now
    merge_commit.author_timezone = 0
    merge_commit.commit_time = now
    merge_commit.commit_timezone = 0
    merge_commit.encoding = b"UTF-8"
    merge_commit.message = message.encode("utf-8") if isinstance(message, str) else message

    # Store the merge commit
    await repo._object_store.add_objects_async([(merge_commit, None)])

    # Update the target ref
    success = await repo.set_ref_async(target_ref, merge_commit.id, target_sha)
    if not success:
        raise ValueError(
            f"Failed to update ref {target_ref.decode()}: "
            "concurrent modification detected"
        )

    merge_sha_str = _sha_to_hex_str(merge_commit.id)
    logger.info("Merge commit created: %s", merge_sha_str[:12])

    return {
        "status": "merged",
        "merge_type": "merge",
        "merge_sha": merge_sha_str,
        "conflicts": [],
    }


def _sha_to_hex_str(sha):
    """Convert a SHA value to a hex string.

    Handles bytes (both 20-byte binary and 40-char hex), strings, and None.

    Args:
        sha: SHA value in any format, or None.

    Returns:
        Hex string representation, or None if input is None.
    """
    if sha is None:
        return None
    if isinstance(sha, str):
        return sha
    if isinstance(sha, bytes):
        if len(sha) == 20:
            return sha.hex()
        return sha.decode("ascii")
    return str(sha)
