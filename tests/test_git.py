"""Test Git version control for artifacts.

This module tests the Git-compatible version control system for Hypha artifacts,
including the S3-backed object store, refs container, and HTTP protocol.

All tests use actual dulwich Git operations - no mocking.
"""

import asyncio
import io
import os
import tempfile
import time
import pytest
import pytest_asyncio
import requests
from contextlib import asynccontextmanager

from aiobotocore.session import get_session as get_aiobotocore_session
from botocore.config import Config

from dulwich.objects import Blob, Tree, Commit, Tag, DEFAULT_OBJECT_FORMAT
from dulwich.pack import write_pack_objects

from . import (
    MINIO_SERVER_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    SERVER_URL_SQLITE,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def s3_client(minio_server):
    """Create an async S3 client connected to MinIO.

    Uses aiobotocore.session.get_session() - same pattern as artifact.py.
    """
    session = get_aiobotocore_session()
    async with session.create_client(
        "s3",
        endpoint_url=MINIO_SERVER_URL,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
        region_name="us-east-1",
        config=Config(connect_timeout=60, read_timeout=300),
    ) as client:
        yield client


@pytest_asyncio.fixture
async def git_test_bucket(s3_client):
    """Create a unique test bucket for Git tests."""
    import uuid
    bucket_name = f"git-test-{uuid.uuid4().hex[:8]}"

    try:
        await s3_client.create_bucket(Bucket=bucket_name)
    except Exception:
        pass  # Bucket may exist

    yield bucket_name

    # Cleanup: delete all objects and the bucket
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                await s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        await s3_client.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


class TestS3GitObjectStore:
    """Tests for the S3-backed Git object store."""

    async def test_store_initialization(self, s3_client, git_test_bucket):
        """Test that object store initializes correctly with empty state."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(
            s3_client,
            git_test_bucket,
            "test-repo/git",
        )
        await store.initialize()

        assert store._initialized
        assert store.bucket == git_test_bucket
        assert store.prefix == "test-repo/git"
        assert len(store._pack_cache) == 0

    async def test_add_single_blob(self, s3_client, git_test_bucket):
        """Test adding a single blob object to the store."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(
            s3_client,
            git_test_bucket,
            "blob-test/git",
        )
        await store.initialize()

        # Create a blob
        content = b"Hello, Git on S3!\nThis is a test file."
        blob = Blob.from_string(content)

        # Create pack with the blob
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        pack_data = pack_buffer.read()

        # Add pack to store
        basename, checksum = await store.add_pack_async(pack_data)

        assert basename is not None
        assert basename.startswith("pack-")
        assert checksum is not None

        # Verify object exists
        exists = await store.contains_async(blob.id)
        assert exists

        # Retrieve and verify content
        type_num, data = await store.get_raw_async(blob.id)
        assert type_num == Blob.type_num
        assert data == content

    async def test_add_multiple_objects(self, s3_client, git_test_bucket):
        """Test adding multiple objects including blobs, trees, and commits."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(
            s3_client,
            git_test_bucket,
            "multi-obj/git",
        )
        await store.initialize()

        # Create multiple blobs
        blob1 = Blob.from_string(b"File 1 content\n")
        blob2 = Blob.from_string(b"File 2 content\n")
        blob3 = Blob.from_string(b"File 3 content\n")

        # Create tree
        tree = Tree()
        tree.add(b"file1.txt", 0o100644, blob1.id)
        tree.add(b"file2.txt", 0o100644, blob2.id)
        tree.add(b"file3.txt", 0o100644, blob3.id)

        # Create commit
        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Test commit with multiple files\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        # Create pack with all objects
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob1, blob2, blob3, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        pack_data = pack_buffer.read()

        # Add pack
        basename, checksum = await store.add_pack_async(pack_data)
        assert basename is not None

        # Verify all objects exist
        for obj in [blob1, blob2, blob3, tree, commit]:
            assert await store.contains_async(obj.id), f"Object {obj.id} not found"

        # Verify commit content
        type_num, data = await store.get_raw_async(commit.id)
        assert type_num == Commit.type_num

    async def test_pack_persistence(self, s3_client, git_test_bucket):
        """Test that packs persist across store instances."""
        from hypha.git.object_store import S3GitObjectStore

        prefix = "persist-test/git"

        # Create first store and add objects
        store1 = S3GitObjectStore(s3_client, git_test_bucket, prefix)
        await store1.initialize()

        blob = Blob.from_string(b"Persistent content")
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await store1.add_pack_async(pack_buffer.read())

        blob_id = blob.id

        # Close first store
        store1.close()

        # Create new store instance
        store2 = S3GitObjectStore(s3_client, git_test_bucket, prefix)
        await store2.initialize()

        # Verify object is still accessible
        assert await store2.contains_async(blob_id)
        type_num, data = await store2.get_raw_async(blob_id)
        assert data == b"Persistent content"


class TestS3RefsContainer:
    """Tests for the S3-backed refs container."""

    async def test_refs_initialization(self, s3_client, git_test_bucket):
        """Test refs container initialization."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-test/git",
        )
        await refs.initialize()

        assert refs._initialized

    async def test_set_and_get_ref(self, s3_client, git_test_bucket):
        """Test setting and getting a reference."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-setget/git",
        )
        await refs.initialize()

        # Set a branch ref
        sha = b"a" * 40
        success = await refs.set_ref_async(b"refs/heads/main", sha)
        assert success

        # Get the ref
        value = await refs.get_ref_async(b"refs/heads/main")
        assert value == sha

    async def test_symbolic_ref(self, s3_client, git_test_bucket):
        """Test symbolic references (like HEAD)."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-symbolic/git",
        )
        await refs.initialize()

        # Set up main branch
        sha = b"b" * 40
        await refs.set_ref_async(b"refs/heads/main", sha)

        # Set HEAD as symbolic ref
        await refs.set_symbolic_ref_async(b"HEAD", b"refs/heads/main")

        # Get symbolic target
        target = await refs.get_symref_async(b"HEAD")
        assert target == b"refs/heads/main"

        # Get resolved value through __getitem__
        resolved = refs[b"HEAD"]
        assert resolved == sha

    async def test_compare_and_swap(self, s3_client, git_test_bucket):
        """Test atomic compare-and-swap operations."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-cas/git",
        )
        await refs.initialize()

        sha1 = b"1" * 40
        sha2 = b"2" * 40
        sha3 = b"3" * 40

        # Set initial ref
        await refs.set_ref_async(b"refs/heads/feature", sha1)

        # CAS with wrong old value should fail
        success = await refs.set_ref_async(
            b"refs/heads/feature", sha3, old_value=sha2
        )
        assert not success

        # Value should be unchanged
        assert await refs.get_ref_async(b"refs/heads/feature") == sha1

        # CAS with correct old value should succeed
        success = await refs.set_ref_async(
            b"refs/heads/feature", sha2, old_value=sha1
        )
        assert success

        # Value should be updated
        assert await refs.get_ref_async(b"refs/heads/feature") == sha2

    async def test_delete_ref(self, s3_client, git_test_bucket):
        """Test deleting references."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-delete/git",
        )
        await refs.initialize()

        sha = b"d" * 40
        await refs.set_ref_async(b"refs/heads/to-delete", sha)

        # Verify ref exists
        assert await refs.get_ref_async(b"refs/heads/to-delete") == sha

        # Delete ref
        success = await refs.delete_ref_async(b"refs/heads/to-delete")
        assert success

        # Verify ref is gone
        assert await refs.get_ref_async(b"refs/heads/to-delete") is None

    async def test_multiple_refs(self, s3_client, git_test_bucket):
        """Test managing multiple refs (branches and tags)."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(
            s3_client,
            git_test_bucket,
            "refs-multi/git",
        )
        await refs.initialize()

        # Create multiple branches and tags
        refs_to_create = {
            b"refs/heads/main": b"m" * 40,
            b"refs/heads/develop": b"d" * 40,
            b"refs/heads/feature/test": b"f" * 40,
            b"refs/tags/v1.0": b"1" * 40,
            b"refs/tags/v2.0": b"2" * 40,
        }

        for name, sha in refs_to_create.items():
            await refs.set_ref_async(name, sha)

        # Get all refs
        all_refs = await refs.get_all_refs_async()

        # Verify all refs exist
        for name, sha in refs_to_create.items():
            assert all_refs.get(name) == sha


class TestS3GitRepo:
    """Tests for the complete S3-backed Git repository."""

    async def test_init_bare_repo(self, s3_client, git_test_bucket):
        """Test initializing a bare repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "repo-init/git",
            default_branch=b"main",
        )

        assert repo.bare
        assert await repo.is_empty()

        # Check HEAD points to main
        symref = await repo._refs.get_symref_async(b"HEAD")
        assert symref == b"refs/heads/main"

    async def test_create_commit(self, s3_client, git_test_bucket):
        """Test creating a commit in the repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "repo-commit/git",
        )

        # Create objects
        blob = Blob.from_string(b"Initial content\n")
        tree = Tree()
        tree.add(b"README.md", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test Author <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Initial commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        # Create pack
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)

        # Add pack and set ref
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Verify
        assert not await repo.is_empty()
        head = await repo.head_async()
        assert head == commit.id

    async def test_receive_pack(self, s3_client, git_test_bucket):
        """Test receiving a pack (simulating git push)."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "repo-receive/git",
        )

        # Create objects
        blob = Blob.from_string(b"Push test content\n")
        tree = Tree()
        tree.add(b"test.txt", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Pusher <push@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Pushed commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        # Create pack data
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        pack_data = pack_buffer.read()

        # Simulate push - update refs
        ref_updates = {
            b"refs/heads/main": (b"0" * 40, commit.id),  # Create new ref
        }

        results = await repo.receive_pack_async(pack_data, ref_updates)

        # All updates should succeed
        for ref_name, error in results.items():
            assert error is None, f"Ref {ref_name} failed: {error}"

        # Verify ref was created
        head = await repo.head_async()
        assert head == commit.id

    async def test_commit_chain(self, s3_client, git_test_bucket):
        """Test creating a chain of commits."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "repo-chain/git",
        )

        commits = []
        parent_id = None

        # Create 3 commits
        for i in range(3):
            blob = Blob.from_string(f"Content version {i}\n".encode())
            tree = Tree()
            tree.add(b"file.txt", 0o100644, blob.id)

            commit = Commit()
            commit.tree = tree.id
            commit.author = commit.committer = b"Chain Test <chain@example.com>"
            commit.encoding = b"UTF-8"
            commit.message = f"Commit {i}\n".encode()
            commit.commit_time = commit.author_time = int(time.time()) + i
            commit.commit_timezone = commit.author_timezone = 0

            if parent_id:
                commit.parents = [parent_id]

            # Add objects
            pack_buffer = io.BytesIO()
            write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
            pack_buffer.seek(0)
            await repo._object_store.add_pack_async(pack_buffer.read())

            parent_id = commit.id
            commits.append(commit)

        # Set ref to latest commit
        await repo.set_ref_async(b"refs/heads/main", commits[-1].id)

        # Verify chain
        head = await repo.head_async()
        assert head == commits[-1].id

        # Walk back through commits
        current = await repo.get_object_async(head)
        assert isinstance(current, Commit)
        assert current.parents[0] == commits[1].id


class TestGitHTTPHandler:
    """Tests for Git HTTP Smart protocol handlers."""

    async def test_pkt_line_format(self):
        """Test pkt-line formatting."""
        from hypha.git.http import pkt_line, pkt_flush

        # Empty flush packet
        assert pkt_flush() == b"0000"

        # Simple data
        result = pkt_line(b"hello")
        # Length = 5 + 4 = 9 = 0x0009
        assert result == b"0009hello"

        # Longer data
        data = b"x" * 100
        result = pkt_line(data)
        length = int(result[:4], 16)
        assert length == 104

    async def test_refs_advertisement_empty(self, s3_client, git_test_bucket):
        """Test refs advertisement for empty repository."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "http-empty/git",
        )

        handler = GitHTTPHandler(repo)

        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Should contain service announcement
        assert b"# service=git-upload-pack" in advertisement
        # Should have flush packets
        assert b"0000" in advertisement

    async def test_refs_advertisement_with_refs(self, s3_client, git_test_bucket):
        """Test refs advertisement with actual refs."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "http-refs/git",
        )

        # Create a commit and ref
        blob = Blob.from_string(b"HTTP test\n")
        tree = Tree()
        tree.add(b"test.txt", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"HTTP Test <http@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"HTTP test commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        handler = GitHTTPHandler(repo)

        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Should contain the ref
        assert b"refs/heads/main" in advertisement
        # Should contain capabilities
        assert b"multi_ack" in advertisement or b"side-band" in advertisement


class TestInfoRefsGeneration:
    """Tests for info/refs generation."""

    async def test_generate_info_refs(self, s3_client, git_test_bucket):
        """Test generating info/refs content."""
        from hypha.git.refs import S3InfoRefsContainer

        refs = S3InfoRefsContainer(
            s3_client,
            git_test_bucket,
            "info-refs/git",
        )
        await refs.initialize()

        # Add some refs
        sha1 = b"a" * 40
        sha2 = b"b" * 40
        await refs.set_ref_async(b"refs/heads/main", sha1)
        await refs.set_ref_async(b"refs/tags/v1.0", sha2)

        # Generate info/refs
        info_refs = await refs.generate_info_refs()

        # Should contain both refs
        assert sha1 + b"\trefs/heads/main\n" in info_refs
        assert sha2 + b"\trefs/tags/v1.0\n" in info_refs


class TestRepoWithDulwichClient:
    """Integration tests using dulwich as a Git client."""

    async def test_full_workflow(self, s3_client, git_test_bucket):
        """Test complete workflow: init, add objects, update refs, retrieve."""
        from hypha.git.repo import S3GitRepo

        # Initialize repo
        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "full-workflow/git",
        )

        # Verify empty
        assert await repo.is_empty()

        # Create first commit
        blob1 = Blob.from_string(b"Version 1 of README\n")
        tree1 = Tree()
        tree1.add(b"README.md", 0o100644, blob1.id)

        commit1 = Commit()
        commit1.tree = tree1.id
        commit1.author = commit1.committer = b"Dev <dev@example.com>"
        commit1.encoding = b"UTF-8"
        commit1.message = b"Initial commit\n"
        commit1.commit_time = commit1.author_time = int(time.time())
        commit1.commit_timezone = commit1.author_timezone = 0

        # Add objects
        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob1, tree1, commit1], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())

        # Update ref
        await repo.set_ref_async(b"refs/heads/main", commit1.id)

        # Verify not empty
        assert not await repo.is_empty()

        # Create second commit with parent
        blob2 = Blob.from_string(b"Version 2 of README\n")
        tree2 = Tree()
        tree2.add(b"README.md", 0o100644, blob2.id)

        commit2 = Commit()
        commit2.tree = tree2.id
        commit2.parents = [commit1.id]
        commit2.author = commit2.committer = b"Dev <dev@example.com>"
        commit2.encoding = b"UTF-8"
        commit2.message = b"Update README\n"
        commit2.commit_time = commit2.author_time = int(time.time()) + 1
        commit2.commit_timezone = commit2.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob2, tree2, commit2], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())

        # Update ref with CAS
        success = await repo.set_ref_async(
            b"refs/heads/main", commit2.id, old_value=commit1.id
        )
        assert success

        # Verify head points to latest commit
        head = await repo.head_async()
        assert head == commit2.id

        # Retrieve and verify commit
        retrieved = await repo.get_object_async(head)
        assert isinstance(retrieved, Commit)
        assert retrieved.message == b"Update README\n"
        assert retrieved.parents == [commit1.id]

        # Create a tag
        await repo._refs.set_ref_async(b"refs/tags/v1.0", commit1.id)

        # Get all refs
        refs = await repo.get_refs_async()
        assert b"refs/heads/main" in refs
        assert b"refs/tags/v1.0" in refs
        assert refs[b"refs/heads/main"] == commit2.id
        assert refs[b"refs/tags/v1.0"] == commit1.id

    async def test_branch_operations(self, s3_client, git_test_bucket):
        """Test branch creation and switching."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "branch-ops/git",
        )

        # Create initial commit on main
        blob = Blob.from_string(b"Main branch content\n")
        tree = Tree()
        tree.add(b"main.txt", 0o100644, blob.id)

        commit1 = Commit()
        commit1.tree = tree.id
        commit1.author = commit1.committer = b"Dev <dev@example.com>"
        commit1.encoding = b"UTF-8"
        commit1.message = b"Main branch commit\n"
        commit1.commit_time = commit1.author_time = int(time.time())
        commit1.commit_timezone = commit1.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit1], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit1.id)

        # Create feature branch from main
        await repo.set_ref_async(b"refs/heads/feature", commit1.id)

        # Add commit to feature branch
        blob2 = Blob.from_string(b"Feature content\n")
        tree2 = Tree()
        tree2.add(b"main.txt", 0o100644, blob.id)
        tree2.add(b"feature.txt", 0o100644, blob2.id)

        commit2 = Commit()
        commit2.tree = tree2.id
        commit2.parents = [commit1.id]
        commit2.author = commit2.committer = b"Dev <dev@example.com>"
        commit2.encoding = b"UTF-8"
        commit2.message = b"Feature branch commit\n"
        commit2.commit_time = commit2.author_time = int(time.time()) + 1
        commit2.commit_timezone = commit2.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob2, tree2, commit2], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/feature", commit2.id)

        # Verify branches diverged
        refs = await repo.get_refs_async()
        assert refs[b"refs/heads/main"] == commit1.id
        assert refs[b"refs/heads/feature"] == commit2.id

        # "Merge" feature to main (fast-forward)
        success = await repo.set_ref_async(
            b"refs/heads/main", commit2.id, old_value=commit1.id
        )
        assert success

        # Delete feature branch
        await repo._refs.delete_ref_async(b"refs/heads/feature")

        refs = await repo.get_refs_async()
        assert b"refs/heads/feature" not in refs
        assert refs[b"refs/heads/main"] == commit2.id


class TestGitCLI:
    """End-to-end tests using standard Git CLI.

    These tests verify that the Git HTTP endpoints work correctly
    with actual Git clients, not just dulwich.

    Note: Due to event loop constraints with aiobotocore and FastAPI's TestClient,
    these tests use the real subprocess tests (TestGitCLISubprocess) for actual
    Git client testing. This class tests the HTTP protocol handlers directly.
    """

    async def test_pkt_line_protocol(self):
        """Test packet line formatting for Git protocol."""
        from hypha.git.http import pkt_line, pkt_flush

        # Empty flush packet
        assert pkt_flush() == b"0000"

        # Simple data packet
        result = pkt_line(b"hello")
        assert result == b"0009hello"  # 5 + 4 = 9 = 0x0009

        # Longer data
        data = b"x" * 100
        result = pkt_line(data)
        length = int(result[:4], 16)
        assert length == 104  # 100 + 4

    async def test_refs_advertisement_generation(self, s3_client, git_test_bucket):
        """Test generating refs advertisement for HTTP protocol."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        # Create test repo
        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "http-test/git",
        )

        # Add a commit
        blob = Blob.from_string(b"HTTP protocol test\n")
        tree = Tree()
        tree.add(b"test.txt", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Test commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Generate refs advertisement
        handler = GitHTTPHandler(repo)
        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Verify format
        assert b"# service=git-upload-pack" in advertisement
        assert b"refs/heads/main" in advertisement
        assert commit.id in advertisement
        # Should include capabilities
        assert b"multi_ack" in advertisement or b"side-band" in advertisement

    async def test_empty_repo_advertisement(self, s3_client, git_test_bucket):
        """Test refs advertisement for an empty repository."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(
            s3_client,
            git_test_bucket,
            "empty-http-test/git",
        )

        handler = GitHTTPHandler(repo)
        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Should still have service announcement
        assert b"# service=git-upload-pack" in advertisement
        # Should have capabilities advertised even for empty repo
        assert b"0000" in advertisement  # Flush packets


class TestGitCLISubprocess:
    """Tests using actual git command-line tool via subprocess.

    These tests run a real HTTP server and use the git binary to verify
    full protocol compatibility.
    """

    @pytest.fixture
    def git_http_server(self, minio_server, request):
        """Start an actual HTTP server for Git protocol testing.

        This is a synchronous fixture that creates a complete isolated test environment
        including its own S3 client to avoid event loop issues.
        """
        import uvicorn
        import threading
        import socket
        import asyncio
        from fastapi import FastAPI
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import create_git_router
        from aiobotocore.session import get_session as get_aiobotocore_session
        from botocore.config import Config
        import uuid

        # Find an available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        # Create unique bucket for this test
        test_id = request.node.name.replace("[", "_").replace("]", "_")
        bucket_name = f"git-test-{uuid.uuid4().hex[:8]}"
        prefix = f"git-subprocess-{test_id}/git"

        # Store state that will be initialized in the server's event loop
        state = {
            "repo": None,
            "s3_client": None,
            "bucket": bucket_name,
            "prefix": prefix,
            "initialized": False,
        }

        async def init_repo():
            """Initialize S3 client and repo in the server's event loop."""
            if state["initialized"]:
                return state["repo"]

            session = get_aiobotocore_session()
            state["s3_client"] = await session.create_client(
                "s3",
                endpoint_url=MINIO_SERVER_URL,
                aws_access_key_id=MINIO_ROOT_USER,
                aws_secret_access_key=MINIO_ROOT_PASSWORD,
                region_name="us-east-1",
                config=Config(connect_timeout=60, read_timeout=300),
            ).__aenter__()

            # Create bucket
            try:
                await state["s3_client"].create_bucket(Bucket=bucket_name)
            except Exception:
                pass

            # Initialize repo
            state["repo"] = await S3GitRepo.init_bare(
                state["s3_client"],
                bucket_name,
                prefix,
            )
            state["initialized"] = True
            return state["repo"]

        async def get_repo_callback(workspace, alias, user_info, write=False):
            return await init_repo()

        def login_optional():
            return {"user_id": "test-user"}

        def login_required():
            return {"user_id": "test-user"}

        app = FastAPI()
        git_router = create_git_router(
            get_repo_callback,
            login_optional,
            login_required,
        )
        app.include_router(git_router)

        # Start server in background thread
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to start
        for _ in range(50):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/")
                break
            except requests.ConnectionError:
                time.sleep(0.1)

        base_url = f"http://127.0.0.1:{port}/test-ws/git/test-artifact"

        yield {
            "url": base_url,
            "port": port,
            "state": state,
            "bucket": bucket_name,
        }

        # Cleanup
        server.should_exit = True
        # Give server time to stop
        time.sleep(0.2)

    @pytest.fixture
    def temp_git_dir(self):
        """Create a temporary directory for Git operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_git_clone_empty_repo(self, git_http_server, temp_git_dir):
        """Test cloning an empty repository with real git command."""
        import subprocess

        url = git_http_server["url"]
        clone_dir = os.path.join(temp_git_dir, "clone")

        # Git clone of empty repo should succeed (or give warning about empty)
        result = subprocess.run(
            ["git", "clone", url, clone_dir],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Empty repo clone might give warning but should not fail fatally
        # Git returns 0 for empty repo clone, or 128 with "empty" warning
        if result.returncode != 0:
            # Some git versions warn about empty repos
            assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                f"Unexpected error: {result.stderr}"

    def test_git_ls_remote_empty(self, git_http_server):
        """Test git ls-remote on an empty repository."""
        import subprocess

        url = git_http_server["url"]

        # Run git ls-remote on empty repo
        result = subprocess.run(
            ["git", "ls-remote", url],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Empty repo should return 0 with no refs (or just HEAD)
        assert result.returncode == 0, f"ls-remote failed: {result.stderr}"

    def test_git_protocol_discovery(self, git_http_server):
        """Test that git can discover the repo via HTTP protocol."""
        url = git_http_server["url"]

        # Test info/refs endpoint directly
        info_refs_url = f"{url}/info/refs?service=git-upload-pack"
        response = requests.get(info_refs_url)

        assert response.status_code == 200
        assert "application/x-git-upload-pack-advertisement" in response.headers.get("content-type", "")

        # Should have service announcement
        assert b"# service=git-upload-pack" in response.content


class TestGitBasicAuth:
    """Tests for Git HTTP Basic Authentication."""

    def test_extract_token_from_basic_auth_bearer(self):
        """Test extracting token from Bearer auth header."""
        from hypha.git.http import extract_token_from_basic_auth

        # Test Bearer token
        token = extract_token_from_basic_auth("Bearer my-secret-token")
        assert token == "my-secret-token"

        # Test with extra whitespace
        token = extract_token_from_basic_auth("Bearer   my-token-with-spaces  ")
        assert token == "my-token-with-spaces"

    def test_extract_token_from_basic_auth_basic(self):
        """Test extracting token from Basic auth header."""
        import base64
        from hypha.git.http import extract_token_from_basic_auth

        # Test Basic auth: base64(username:password)
        # Git uses the password as the token
        credentials = base64.b64encode(b"git:my-hypha-token").decode()
        token = extract_token_from_basic_auth(f"Basic {credentials}")
        assert token == "my-hypha-token"

        # Test with any username
        credentials = base64.b64encode(b"user@example.com:another-token").decode()
        token = extract_token_from_basic_auth(f"Basic {credentials}")
        assert token == "another-token"

        # Test with empty username
        credentials = base64.b64encode(b":token-only").decode()
        token = extract_token_from_basic_auth(f"Basic {credentials}")
        assert token == "token-only"

    def test_extract_token_from_basic_auth_invalid(self):
        """Test invalid auth headers."""
        from hypha.git.http import extract_token_from_basic_auth

        # Test None
        assert extract_token_from_basic_auth(None) is None

        # Test empty string
        assert extract_token_from_basic_auth("") is None

        # Test invalid Basic auth (not base64)
        assert extract_token_from_basic_auth("Basic not-valid-base64!!!") is None

        # Test Basic auth without colon (no password)
        import base64
        credentials = base64.b64encode(b"username-only").decode()
        assert extract_token_from_basic_auth(f"Basic {credentials}") is None

    def test_extract_token_password_with_colon(self):
        """Test that passwords containing colons are handled correctly."""
        import base64
        from hypha.git.http import extract_token_from_basic_auth

        # Password with colons: "token:with:colons"
        credentials = base64.b64encode(b"git:token:with:colons").decode()
        token = extract_token_from_basic_auth(f"Basic {credentials}")
        assert token == "token:with:colons"

    def test_git_receive_pack_requires_auth(self, minio_server):
        """Test that git-receive-pack endpoint returns 401 without auth."""
        import uvicorn
        import threading
        import socket
        import uuid
        import base64
        from fastapi import FastAPI
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import create_git_router
        from aiobotocore.session import get_session as get_aiobotocore_session
        from botocore.config import Config

        # Find available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        bucket_name = f"git-auth-test-{uuid.uuid4().hex[:8]}"
        prefix = "git-auth-test/git"

        state = {"repo": None, "s3_client": None, "initialized": False}

        async def init_repo():
            if state["initialized"]:
                return state["repo"]
            session = get_aiobotocore_session()
            state["s3_client"] = await session.create_client(
                "s3",
                endpoint_url=MINIO_SERVER_URL,
                aws_access_key_id=MINIO_ROOT_USER,
                aws_secret_access_key=MINIO_ROOT_PASSWORD,
                region_name="us-east-1",
                config=Config(connect_timeout=60, read_timeout=300),
            ).__aenter__()
            try:
                await state["s3_client"].create_bucket(Bucket=bucket_name)
            except Exception:
                pass
            state["repo"] = await S3GitRepo.init_bare(
                state["s3_client"], bucket_name, prefix
            )
            state["initialized"] = True
            return state["repo"]

        async def get_repo_callback(workspace, alias, user_info, write=False):
            return await init_repo()

        def login_optional():
            return None  # Anonymous user

        def login_required():
            return {"user_id": "test-user"}

        # Mock token parser that validates tokens
        async def parse_user_token(token):
            if token == "valid-test-token":
                # Return a mock UserInfo object
                class MockScope:
                    current_workspace = None
                class MockUserInfo:
                    id = "test-user"
                    scope = MockScope()
                    def get_workspace(self):
                        return "test-ws"
                return MockUserInfo()
            raise ValueError("Invalid token")

        app = FastAPI()
        git_router = create_git_router(
            get_repo_callback,
            login_optional,
            login_required,
            parse_user_token,  # Pass token parser for Basic Auth
        )
        app.include_router(git_router)

        # Add a health check endpoint
        @app.get("/health")
        def health():
            return {"status": "ok"}

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server using health endpoint
        server_ready = False
        for _ in range(100):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.1)

        assert server_ready, "Server failed to start"

        base_url = f"http://127.0.0.1:{port}/test-ws/git/test-artifact"

        # Test 1: git-receive-pack without auth should return 401
        receive_pack_url = f"{base_url}/git-receive-pack"
        response = requests.post(receive_pack_url, data=b"", timeout=10)
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"
        assert "WWW-Authenticate" in response.headers
        assert "Basic" in response.headers["WWW-Authenticate"]

        # Test 2: git-receive-pack with invalid Basic auth should return 401
        invalid_creds = base64.b64encode(b"git:invalid-token").decode()
        response = requests.post(
            receive_pack_url,
            data=b"",
            headers={"Authorization": f"Basic {invalid_creds}"},
            timeout=10
        )
        assert response.status_code == 401

        # Test 3: git-receive-pack with valid Basic auth should pass auth (200 status from stream)
        # Note: We don't check the full response body because receive_pack processing
        # with empty pack data may hang or have other issues - we just verify auth passes
        valid_creds = base64.b64encode(b"git:valid-test-token").decode()
        response = requests.post(
            receive_pack_url,
            data=b"",  # Empty push data
            headers={"Authorization": f"Basic {valid_creds}"},
            timeout=5,
            stream=True,  # Use streaming to just check status without reading body
        )
        # With valid auth, we should get 200 status code (auth passed)
        # even if the body can't be fully read due to empty pack data
        assert response.status_code == 200
        response.close()  # Close without reading full body to avoid timeout

        # Cleanup
        server.should_exit = True
        time.sleep(0.2)

    def test_git_clone_add_commit_push(self, minio_server):
        """Test full git workflow: clone empty repo, add file, commit, and push.

        This test reproduces the real-world scenario where a user:
        1. Clones an empty repository
        2. Adds a new file
        3. Commits the change
        4. Pushes to the remote

        This specifically tests the streaming implementation in handle_receive_pack.
        """
        import uvicorn
        import threading
        import socket
        import uuid
        import subprocess
        import base64
        from fastapi import FastAPI
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import create_git_router
        from aiobotocore.session import get_session as get_aiobotocore_session
        from botocore.config import Config

        # Find available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()

        bucket_name = f"git-push-test-{uuid.uuid4().hex[:8]}"
        prefix = "git-push-test/git"

        state = {"repo": None, "s3_client": None, "initialized": False}

        async def init_repo():
            if state["initialized"]:
                return state["repo"]
            session = get_aiobotocore_session()
            state["s3_client"] = await session.create_client(
                "s3",
                endpoint_url=MINIO_SERVER_URL,
                aws_access_key_id=MINIO_ROOT_USER,
                aws_secret_access_key=MINIO_ROOT_PASSWORD,
                region_name="us-east-1",
                config=Config(connect_timeout=60, read_timeout=300),
            ).__aenter__()
            try:
                await state["s3_client"].create_bucket(Bucket=bucket_name)
            except Exception:
                pass
            state["repo"] = await S3GitRepo.init_bare(
                state["s3_client"], bucket_name, prefix
            )
            state["initialized"] = True
            return state["repo"]

        async def get_repo_callback(workspace, alias, user_info, write=False):
            return await init_repo()

        def login_optional():
            return None  # Anonymous user for clone

        def login_required():
            return {"user_id": "test-user"}

        # Mock token parser that validates tokens
        async def parse_user_token(token):
            if token == "test-push-token":
                class MockScope:
                    current_workspace = None
                class MockUserInfo:
                    id = "test-user"
                    scope = MockScope()
                    def get_workspace(self):
                        return "test-ws"
                return MockUserInfo()
            raise ValueError("Invalid token")

        app = FastAPI()
        git_router = create_git_router(
            get_repo_callback,
            login_optional,
            login_required,
            parse_user_token,
        )
        app.include_router(git_router)

        @app.get("/health")
        def health():
            return {"status": "ok"}

        # Enable logging for hypha.git
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("hypha.git").setLevel(logging.DEBUG)

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="debug")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server
        server_ready = False
        for _ in range(100):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.1)

        assert server_ready, "Server failed to start"

        # Create a temporary directory for git operations
        with tempfile.TemporaryDirectory() as tmpdir:
            clone_dir = os.path.join(tmpdir, "repo")

            # URL with embedded credentials for push
            base_url = f"http://127.0.0.1:{port}/test-ws/git/test-artifact"
            push_url = f"http://git:test-push-token@127.0.0.1:{port}/test-ws/git/test-artifact"

            # Step 1: Clone the empty repository
            result = subprocess.run(
                ["git", "clone", base_url, clone_dir],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Empty repo clone might give warning but should not fail fatally
            if result.returncode != 0:
                # Some git versions warn about empty repos
                assert "empty" in result.stderr.lower() or "warning" in result.stderr.lower(), \
                    f"Clone failed unexpectedly: {result.stderr}"

            # Step 2: Configure git user (required for commit)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=clone_dir,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=clone_dir,
                capture_output=True,
                timeout=10,
            )

            # Step 3: Create a new file
            test_file = os.path.join(clone_dir, "README.md")
            with open(test_file, "w") as f:
                f.write("# Test Repository\n\nThis is a test file.\n")

            # Step 4: Add the file
            result = subprocess.run(
                ["git", "add", "README.md"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"git add failed: {result.stderr}"

            # Step 5: Commit
            result = subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"git commit failed: {result.stderr}"

            # Verify the server advertises capabilities for receive-pack
            info_refs_url = f"{base_url}/info/refs?service=git-receive-pack"
            response = requests.get(info_refs_url)
            assert response.status_code == 200
            # Server must advertise capabilities, including report-status
            assert b"report-status" in response.content, \
                f"Server should advertise report-status capability, got: {response.content!r}"

            # Step 6: Push with authentication
            # Use a timeout to catch hanging issues
            # Use GIT_CURL_VERBOSE to see what's happening
            env = os.environ.copy()
            env["GIT_CURL_VERBOSE"] = "1"
            result = subprocess.run(
                ["git", "push", "-v", push_url, "main"],
                cwd=clone_dir,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout - should complete much faster
                env=env,
            )

            # Check push succeeded
            assert result.returncode == 0, \
                f"git push failed: stdout={result.stdout}, stderr={result.stderr}"

            # Step 7: Clone the repository again to verify it's not empty
            # This tests the upload-pack (fetch) functionality with real content
            clone_dir2 = os.path.join(tmpdir, "repo2")
            result = subprocess.run(
                ["git", "clone", base_url, clone_dir2],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, \
                f"git clone (after push) failed: stdout={result.stdout}, stderr={result.stderr}"

            # Verify the cloned repository has the file we pushed
            cloned_file = os.path.join(clone_dir2, "README.md")
            assert os.path.exists(cloned_file), \
                f"Cloned repo should contain README.md, dir contents: {os.listdir(clone_dir2)}"

            with open(cloned_file, "r") as f:
                content = f.read()
            assert "Test Repository" in content, \
                f"Cloned file should have expected content, got: {content}"

            # Verify git log shows our commit
            result = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=clone_dir2,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, f"git log failed: {result.stderr}"
            assert "Initial commit" in result.stdout, \
                f"Git log should show 'Initial commit', got: {result.stdout}"

        # Cleanup
        server.should_exit = True
        time.sleep(0.2)
