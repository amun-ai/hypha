"""Unit tests for Git S3-backed storage (object store, refs, repo).

These tests use direct S3 access with dulwich Git operations - no Hypha server needed.

Uses S3 client factory pattern: Each test creates a factory that provides
fresh async context managers for S3 operations, matching the production pattern.
"""

import io
import time
import pytest
import pytest_asyncio
from contextlib import asynccontextmanager

from aiobotocore.session import get_session as get_aiobotocore_session
from botocore.config import Config

from dulwich.objects import Blob, Tree, Commit, DEFAULT_OBJECT_FORMAT
from dulwich.pack import write_pack_objects

from . import (
    MINIO_SERVER_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
)

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


def create_s3_config():
    """Create S3 config dict for sync boto3 client."""
    return {
        "endpoint_url": MINIO_SERVER_URL,
        "access_key_id": MINIO_ROOT_USER,
        "secret_access_key": MINIO_ROOT_PASSWORD,
        "region_name": "us-east-1",
    }


def create_s3_client_factory():
    """Create an S3 client factory for tests.

    Returns a callable that returns an async context manager for S3 client.
    Each call creates a fresh client context for reliable async operations.
    """
    session = get_aiobotocore_session()

    @asynccontextmanager
    async def s3_client_factory():
        async with session.create_client(
            "s3",
            endpoint_url=MINIO_SERVER_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
            region_name="us-east-1",
            config=Config(connect_timeout=60, read_timeout=300),
        ) as client:
            yield client

    return s3_client_factory


@pytest_asyncio.fixture
async def s3_config(minio_server):
    """Create S3 config dict for sync boto3 client."""
    return create_s3_config()


@pytest_asyncio.fixture
async def s3_client_factory(minio_server):
    """Create an S3 client factory connected to MinIO.

    Returns a callable that produces async context managers for S3 clients.
    This matches the production pattern used in artifact_integration.py.
    """
    return create_s3_client_factory()


@pytest_asyncio.fixture
async def git_test_bucket(s3_client_factory):
    """Create a unique test bucket for Git tests."""
    import uuid

    bucket_name = f"git-test-{uuid.uuid4().hex[:8]}"

    async with s3_client_factory() as s3_client:
        try:
            await s3_client.create_bucket(Bucket=bucket_name)
        except Exception:
            pass  # Bucket may exist

    yield bucket_name

    # Cleanup: delete all objects and the bucket
    async with s3_client_factory() as s3_client:
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

    async def test_store_initialization(self, s3_client_factory, s3_config, git_test_bucket):
        """Test that object store initializes correctly with empty state."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(s3_client_factory, git_test_bucket, "test-repo/.git", s3_config=s3_config)
        await store.initialize()

        assert store._initialized
        assert store.bucket == git_test_bucket
        assert store.prefix == "test-repo/.git"
        assert len(store._pack_cache) == 0

    async def test_add_single_blob(self, s3_client_factory, s3_config, git_test_bucket):
        """Test adding a single blob object to the store."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(s3_client_factory, git_test_bucket, "blob-test/.git", s3_config=s3_config)
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

    async def test_add_multiple_objects(self, s3_client_factory, s3_config, git_test_bucket):
        """Test adding multiple objects including blobs, trees, and commits."""
        from hypha.git.object_store import S3GitObjectStore

        store = S3GitObjectStore(s3_client_factory, git_test_bucket, "multi-obj/.git", s3_config=s3_config)
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

    async def test_pack_persistence(self, s3_client_factory, s3_config, git_test_bucket):
        """Test that packs persist across store instances."""
        from hypha.git.object_store import S3GitObjectStore

        prefix = "persist-test/.git"

        # Create first store and add objects
        store1 = S3GitObjectStore(s3_client_factory, git_test_bucket, prefix, s3_config=s3_config)
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
        store2 = S3GitObjectStore(s3_client_factory, git_test_bucket, prefix, s3_config=s3_config)
        await store2.initialize()

        # Verify object is still accessible
        assert await store2.contains_async(blob_id)
        type_num, data = await store2.get_raw_async(blob_id)
        assert data == b"Persistent content"


class TestS3RefsContainer:
    """Tests for the S3-backed refs container."""

    async def test_refs_initialization(self, s3_client_factory, s3_config, git_test_bucket):
        """Test refs container initialization."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-test/.git", s3_config=s3_config)
        await refs.initialize()

        assert refs._initialized

    async def test_set_and_get_ref(self, s3_client_factory, s3_config, git_test_bucket):
        """Test setting and getting a reference."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-setget/.git", s3_config=s3_config)
        await refs.initialize()

        # Set a branch ref
        sha = b"a" * 40
        success = await refs.set_ref_async(b"refs/heads/main", sha)
        assert success

        # Get the ref
        value = await refs.get_ref_async(b"refs/heads/main")
        assert value == sha

    async def test_symbolic_ref(self, s3_client_factory, s3_config, git_test_bucket):
        """Test symbolic references (like HEAD)."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-symbolic/.git", s3_config=s3_config)
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

    async def test_compare_and_swap(self, s3_client_factory, s3_config, git_test_bucket):
        """Test atomic compare-and-swap operations."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-cas/.git", s3_config=s3_config)
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

    async def test_delete_ref(self, s3_client_factory, s3_config, git_test_bucket):
        """Test deleting references."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-delete/.git", s3_config=s3_config)
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

    async def test_multiple_refs(self, s3_client_factory, s3_config, git_test_bucket):
        """Test managing multiple refs (branches and tags)."""
        from hypha.git.refs import S3RefsContainer

        refs = S3RefsContainer(s3_client_factory, git_test_bucket, "refs-multi/.git", s3_config=s3_config)
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

    async def test_init_bare_repo(self, s3_client_factory, s3_config, git_test_bucket):
        """Test initializing a bare repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "repo-init/.git", default_branch=b"main", s3_config=s3_config)

        assert repo.bare
        assert await repo.is_empty()

        # Check HEAD points to main
        symref = await repo._refs.get_symref_async(b"HEAD")
        assert symref == b"refs/heads/main"

    async def test_create_commit(self, s3_client_factory, s3_config, git_test_bucket):
        """Test creating a commit in the repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "repo-commit/.git", s3_config=s3_config)

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

    async def test_receive_pack(self, s3_client_factory, s3_config, git_test_bucket):
        """Test receiving a pack (simulating git push)."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "repo-receive/.git", s3_config=s3_config)

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

    async def test_commit_chain(self, s3_client_factory, s3_config, git_test_bucket):
        """Test creating a chain of commits."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "repo-chain/.git", s3_config=s3_config)

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

    async def test_refs_advertisement_empty(self, s3_client_factory, s3_config, git_test_bucket):
        """Test refs advertisement for empty repository."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "http-empty/.git", s3_config=s3_config)

        handler = GitHTTPHandler(repo)

        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Should contain service announcement
        assert b"# service=git-upload-pack" in advertisement
        # Should have flush packets
        assert b"0000" in advertisement

    async def test_refs_advertisement_with_refs(self, s3_client_factory, s3_config, git_test_bucket):
        """Test refs advertisement with actual refs."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "http-refs/.git", s3_config=s3_config)

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

    async def test_generate_info_refs(self, s3_client_factory, s3_config, git_test_bucket):
        """Test generating info/refs content."""
        from hypha.git.refs import S3InfoRefsContainer

        refs = S3InfoRefsContainer(s3_client_factory, git_test_bucket, "info-refs/.git", s3_config=s3_config)
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

    async def test_full_workflow(self, s3_client_factory, s3_config, git_test_bucket):
        """Test complete workflow: init, add objects, update refs, retrieve."""
        from hypha.git.repo import S3GitRepo

        # Initialize repo
        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "full-workflow/.git", s3_config=s3_config)

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

    async def test_branch_operations(self, s3_client_factory, s3_config, git_test_bucket):
        """Test branch creation and switching."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "branch-ops/.git", s3_config=s3_config)

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


class TestGitHTTPProtocol:
    """Tests for Git HTTP protocol handlers (unit tests, no server needed)."""

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

    async def test_refs_advertisement_generation(self, s3_client_factory, s3_config, git_test_bucket):
        """Test generating refs advertisement for HTTP protocol."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        # Create test repo
        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "http-test/.git", s3_config=s3_config)

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

    async def test_empty_repo_advertisement(self, s3_client_factory, s3_config, git_test_bucket):
        """Test refs advertisement for an empty repository."""
        from hypha.git.repo import S3GitRepo
        from hypha.git.http import GitHTTPHandler

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "empty-http-test/.git", s3_config=s3_config)

        handler = GitHTTPHandler(repo)
        chunks = []
        async for chunk in handler.generate_refs_advertisement(b"git-upload-pack"):
            chunks.append(chunk)

        advertisement = b"".join(chunks)

        # Should still have service announcement
        assert b"# service=git-upload-pack" in advertisement
        # Should have capabilities advertised even for empty repo
        assert b"0000" in advertisement  # Flush packets


class TestGitBasicAuthUnit:
    """Tests for Git HTTP Basic Authentication (unit tests only)."""

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

        assert extract_token_from_basic_auth(None) is None
        assert extract_token_from_basic_auth("") is None
        assert extract_token_from_basic_auth("Basic not-valid-base64!!!") is None

        import base64
        credentials = base64.b64encode(b"username-only").decode()
        assert extract_token_from_basic_auth(f"Basic {credentials}") is None

    def test_extract_token_password_with_colon(self):
        """Test that passwords containing colons are handled correctly."""
        import base64
        from hypha.git.http import extract_token_from_basic_auth

        credentials = base64.b64encode(b"git:token:with:colons").decode()
        token = extract_token_from_basic_auth(f"Basic {credentials}")
        assert token == "token:with:colons"


class TestGitRepoFileAccess:
    """Tests for reading files and directory listings from Git repositories."""

    async def test_list_tree_root(self, s3_client_factory, s3_config, git_test_bucket):
        """Test listing files in the root directory."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-list-root/.git", s3_config=s3_config)

        # Create a simple repo structure
        blob1 = Blob.from_string(b"README content\n")
        blob2 = Blob.from_string(b"print('hello')\n")

        tree = Tree()
        tree.add(b"README.md", 0o100644, blob1.id)
        tree.add(b"main.py", 0o100644, blob2.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Initial commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob1, blob2, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # List root directory
        items = await repo.list_tree_async("")
        assert len(items) == 2

        names = [item["name"] for item in items]
        assert "README.md" in names
        assert "main.py" in names

        # Check types and sizes
        for item in items:
            assert item["type"] == "blob"
            assert item["size"] is not None
            assert item["sha"] is not None

    async def test_list_tree_subdirectory(self, s3_client_factory, s3_config, git_test_bucket):
        """Test listing files in a subdirectory."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-list-subdir/.git", s3_config=s3_config)

        # Create repo with subdirectory
        blob1 = Blob.from_string(b"Root file\n")
        blob2 = Blob.from_string(b"Source code\n")
        blob3 = Blob.from_string(b"Test code\n")

        # Create src/ subdirectory tree
        src_tree = Tree()
        src_tree.add(b"app.py", 0o100644, blob2.id)
        src_tree.add(b"test.py", 0o100644, blob3.id)

        # Create root tree with subdirectory
        root_tree = Tree()
        root_tree.add(b"README.md", 0o100644, blob1.id)
        root_tree.add(b"src", 0o40000, src_tree.id)

        commit = Commit()
        commit.tree = root_tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Initial commit\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(
            pack_buffer,
            [blob1, blob2, blob3, src_tree, root_tree, commit],
            DEFAULT_OBJECT_FORMAT,
        )
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # List root - should have file and directory
        root_items = await repo.list_tree_async("")
        assert len(root_items) == 2

        readme = [i for i in root_items if i["name"] == "README.md"][0]
        assert readme["type"] == "blob"
        assert readme["size"] == len(b"Root file\n")

        src = [i for i in root_items if i["name"] == "src"][0]
        assert src["type"] == "tree"
        assert src["size"] is None  # Directories don't have size

        # List src/ subdirectory
        src_items = await repo.list_tree_async("src")
        assert len(src_items) == 2

        names = [item["name"] for item in src_items]
        assert "app.py" in names
        assert "test.py" in names

    async def test_get_file_content(self, s3_client_factory, s3_config, git_test_bucket):
        """Test reading file content from the repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-content/.git", s3_config=s3_config)

        file_content = b"Hello, Git storage!\nThis is a test file.\n"
        blob = Blob.from_string(file_content)

        tree = Tree()
        tree.add(b"greeting.txt", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Add greeting\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Read file content
        content = await repo.get_file_content_async("greeting.txt")
        assert content == file_content

    async def test_get_file_content_nested(self, s3_client_factory, s3_config, git_test_bucket):
        """Test reading file content from a nested path."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-content-nested/.git", s3_config=s3_config)

        file_content = b"Deeply nested content\n"
        blob = Blob.from_string(file_content)

        # Create nested tree structure: a/b/c/file.txt
        c_tree = Tree()
        c_tree.add(b"file.txt", 0o100644, blob.id)

        b_tree = Tree()
        b_tree.add(b"c", 0o40000, c_tree.id)

        a_tree = Tree()
        a_tree.add(b"b", 0o40000, b_tree.id)

        root_tree = Tree()
        root_tree.add(b"a", 0o40000, a_tree.id)

        commit = Commit()
        commit.tree = root_tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Nested structure\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(
            pack_buffer,
            [blob, c_tree, b_tree, a_tree, root_tree, commit],
            DEFAULT_OBJECT_FORMAT,
        )
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Read deeply nested file
        content = await repo.get_file_content_async("a/b/c/file.txt")
        assert content == file_content

        # Test with leading slash
        content = await repo.get_file_content_async("/a/b/c/file.txt")
        assert content == file_content

    async def test_get_file_not_found(self, s3_client_factory, s3_config, git_test_bucket):
        """Test that reading non-existent file returns None."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-not-found/.git", s3_config=s3_config)

        blob = Blob.from_string(b"Content\n")
        tree = Tree()
        tree.add(b"exists.txt", 0o100644, blob.id)

        commit = Commit()
        commit.tree = tree.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Test\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, tree, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Non-existent file
        content = await repo.get_file_content_async("does-not-exist.txt")
        assert content is None

        # Non-existent directory in path
        content = await repo.get_file_content_async("nonexistent/path/file.txt")
        assert content is None

    async def test_get_file_info(self, s3_client_factory, s3_config, git_test_bucket):
        """Test getting file information."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "file-info/.git", s3_config=s3_config)

        file_content = b"File content for info test\n"
        blob = Blob.from_string(file_content)

        subdir = Tree()
        subdir.add(b"nested.txt", 0o100644, blob.id)

        root = Tree()
        root.add(b"root.txt", 0o100644, blob.id)
        root.add(b"subdir", 0o40000, subdir.id)

        commit = Commit()
        commit.tree = root.id
        commit.author = commit.committer = b"Test <test@example.com>"
        commit.encoding = b"UTF-8"
        commit.message = b"Test\n"
        commit.commit_time = commit.author_time = int(time.time())
        commit.commit_timezone = commit.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob, subdir, root, commit], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit.id)

        # Get file info
        info = await repo.get_file_info_async("root.txt")
        assert info["name"] == "root.txt"
        assert info["type"] == "blob"
        assert info["size"] == len(file_content)
        assert info["sha"] is not None

        # Get directory info
        info = await repo.get_file_info_async("subdir")
        assert info["name"] == "subdir"
        assert info["type"] == "tree"
        assert info["size"] is None

        # Non-existent path
        info = await repo.get_file_info_async("nonexistent")
        assert info is None

    async def test_list_empty_repo(self, s3_client_factory, s3_config, git_test_bucket):
        """Test listing files in an empty repository."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "empty-list/.git", s3_config=s3_config)

        # List should return empty list for empty repo
        items = await repo.list_tree_async("")
        assert items == []

    async def test_get_file_from_specific_commit(self, s3_client_factory, s3_config, git_test_bucket):
        """Test reading file from a specific commit SHA."""
        from hypha.git.repo import S3GitRepo

        repo = await S3GitRepo.init_bare(s3_client_factory, git_test_bucket, "commit-specific/.git", s3_config=s3_config)

        # Create first commit
        blob1 = Blob.from_string(b"Version 1\n")
        tree1 = Tree()
        tree1.add(b"file.txt", 0o100644, blob1.id)

        commit1 = Commit()
        commit1.tree = tree1.id
        commit1.author = commit1.committer = b"Test <test@example.com>"
        commit1.encoding = b"UTF-8"
        commit1.message = b"First\n"
        commit1.commit_time = commit1.author_time = int(time.time())
        commit1.commit_timezone = commit1.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob1, tree1, commit1], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit1.id)

        # Create second commit
        blob2 = Blob.from_string(b"Version 2\n")
        tree2 = Tree()
        tree2.add(b"file.txt", 0o100644, blob2.id)

        commit2 = Commit()
        commit2.tree = tree2.id
        commit2.parents = [commit1.id]
        commit2.author = commit2.committer = b"Test <test@example.com>"
        commit2.encoding = b"UTF-8"
        commit2.message = b"Second\n"
        commit2.commit_time = commit2.author_time = int(time.time()) + 1
        commit2.commit_timezone = commit2.author_timezone = 0

        pack_buffer = io.BytesIO()
        write_pack_objects(pack_buffer, [blob2, tree2, commit2], DEFAULT_OBJECT_FORMAT)
        pack_buffer.seek(0)
        await repo._object_store.add_pack_async(pack_buffer.read())
        await repo.set_ref_async(b"refs/heads/main", commit2.id)

        # Read from HEAD (should be version 2)
        content = await repo.get_file_content_async("file.txt")
        assert content == b"Version 2\n"

        # Read from first commit
        content = await repo.get_file_content_async("file.txt", commit1.id)
        assert content == b"Version 1\n"

        # Read from second commit
        content = await repo.get_file_content_async("file.txt", commit2.id)
        assert content == b"Version 2\n"
