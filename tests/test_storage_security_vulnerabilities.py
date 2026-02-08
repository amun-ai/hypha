"""
Test suite for storage and S3 security vulnerabilities in Hypha artifact system.

These tests document discovered vulnerabilities in artifact storage, S3 presigned URLs,
and SQL injection in search/filter operations.
"""
import pytest
from hypha.core import UserInfo, UserPermission


class TestArtifactStorageSQLInjection:
    """
    CRITICAL VULNERABILITY: SQL Injection in artifact search/filter operations

    Location: hypha/artifact.py lines 8451-8454, 8199-8201, 8219-8230

    The _build_manifest_condition() and search() methods use text(f"...{user_input}...")
    which directly interpolates user-controlled input into SQL without parameterization.

    Impact: Complete database compromise, data exfiltration, modification, or deletion
    Severity: CRITICAL
    """

    @pytest.mark.asyncio
    async def test_sql_injection_via_keyword_search(self, root_user_client, test_user_token):
        """
        V-STORAGE-01: SQL Injection through keywords parameter

        Location: hypha/artifact.py lines 8451-8456

        Code:
            for keyword in keywords:
                if backend == "postgresql":
                    condition = text(f"manifest::text ILIKE '%{keyword}%'")
                else:
                    condition = text(f"json_extract(manifest, '$') LIKE '%{keyword}%'")

        Attack: Inject malicious SQL through keyword to extract/modify data
        """
        # Create a test artifact
        artifact = await root_user_client.create(
            parent_id="collections/test-collection",
            manifest={"name": "Test Dataset", "secret_data": "sensitive_value"},
            type="dataset"
        )
        artifact_id = artifact["id"]

        # SQL Injection payload - extract all data by breaking out of LIKE
        # This payload attempts to bypass the LIKE and inject a UNION SELECT
        malicious_keyword = "' OR 1=1 --"

        # For SQLite: attempt to extract data
        sqlite_injection = "' OR json_extract(manifest, '$.secret_data') LIKE '%sensitive%' --"

        # For PostgreSQL: UNION-based injection
        postgres_injection = "' UNION SELECT id, alias, workspace, manifest, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL FROM artifacts --"

        try:
            # Attempt the SQL injection
            results = await root_user_client.search(
                keywords=[malicious_keyword],
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            # If this succeeds without error, SQL injection is possible
            # The results may contain unintended data
            print(f"SQL Injection attempt returned {len(results)} results")
            print(f"This should not happen - indicates SQL injection vulnerability")
            assert False, "SQL injection vulnerability confirmed - keywords not sanitized"

        except Exception as e:
            # Check if error reveals SQL structure
            error_str = str(e)
            if "SQL" in error_str or "syntax" in error_str or "manifest" in error_str:
                pytest.fail(f"SQL error leakage reveals structure: {error_str}")

    @pytest.mark.asyncio
    async def test_sql_injection_via_manifest_filter(self, root_user_client):
        """
        V-STORAGE-02: SQL Injection through manifest filters

        Location: hypha/artifact.py lines 8219-8230

        Code:
            if backend == "postgresql":
                return text(f"manifest->>'{manifest_key}' ILIKE '{value.replace('*', '%')}'")
            else:
                return text(f"json_extract(manifest, '$.{manifest_key}') LIKE '{value.replace('*', '%')}'")

        Attack: Inject SQL through manifest filter values
        """
        # Create artifacts with different permissions
        await root_user_client.create(
            parent_id="collections/test-collection",
            manifest={"name": "Public Data", "status": "public"},
            config={"permissions": {"*": "read"}},
            type="dataset"
        )

        await root_user_client.create(
            parent_id="collections/test-collection",
            manifest={"name": "Secret Data", "status": "classified"},
            config={"permissions": {"admin-only": "read"}},
            type="dataset"
        )

        # SQL Injection through manifest filter with wildcard replacement
        # The code does value.replace('*', '%') but doesn't escape quotes
        injection_value = "public' OR '1'='1"

        try:
            results = await root_user_client.search(
                filters={
                    "manifest": {
                        "status": injection_value
                    }
                },
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            # If this returns the "Secret Data" artifact, SQL injection worked
            if len(results) > 1:
                names = [r["manifest"]["name"] for r in results]
                if "Secret Data" in names:
                    pytest.fail("SQL injection via manifest filter - bypassed permission check")

        except Exception as e:
            error_str = str(e)
            if "SQL" in error_str or "syntax" in error_str:
                pytest.fail(f"SQL injection attempt exposed error: {error_str}")

    @pytest.mark.asyncio
    async def test_sql_injection_via_config_permissions_filter(self, root_user_client):
        """
        V-STORAGE-03: SQL Injection through config.permissions filter

        Location: hypha/artifact.py lines 8498-8505

        Code:
            if backend == "postgresql":
                condition = text(f"permissions->>'{user_id}' = '{permission}'")
            else:
                condition = text(f"json_extract(permissions, '$.{user_id}') = '{permission}'")

        Attack: Inject through user_id or permission values in config filter
        """
        # SQL injection through user_id key in permissions filter
        malicious_user_id = "admin' OR '1'='1' --"

        try:
            results = await root_user_client.search(
                filters={
                    "config": {
                        "permissions": {
                            malicious_user_id: "read"
                        }
                    }
                },
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            # If this doesn't raise an error, SQL injection is possible
            pytest.fail("SQL injection through config.permissions filter not sanitized")

        except Exception as e:
            # Should fail safely, but check for info leakage
            if "SQL" in str(e) or "syntax" in str(e):
                pytest.fail(f"SQL error reveals internals: {e}")

    @pytest.mark.asyncio
    async def test_sql_injection_via_order_by_json_field(self, root_user_client):
        """
        V-STORAGE-04: SQL Injection through order_by JSON field paths

        Location: hypha/artifact.py lines 8650-8676

        Code:
            json_key = field_name[9:]  # Remove "manifest." prefix
            order_clause = f'''
                CASE
                    WHEN manifest->>'{json_key}' ~ '^-?[0-9]+\\.?[0-9]*$'
                    THEN (manifest->>'{json_key}')::numeric
                END {'ASC' if ascending else 'DESC'} NULLS LAST
            '''
            query = query.order_by(text(order_clause))

        Attack: Inject SQL through order_by parameter using manifest.* or config.* fields
        """
        # Create some test data
        for i in range(3):
            await root_user_client.create(
                parent_id="collections/test-collection",
                manifest={"name": f"Dataset {i}", "priority": i},
                type="dataset"
            )

        # SQL injection through order_by JSON field
        # Attempt to inject into the json_key variable
        malicious_order = "manifest.priority' OR 1=1 --"

        try:
            results = await root_user_client.search(
                parent_id="collections/test-collection",
                order_by=malicious_order,
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            # If this succeeds, the injection went through
            pytest.fail("SQL injection in order_by JSON field not prevented")

        except Exception as e:
            # Check for SQL error leakage
            if "SQL" in str(e) or "syntax" in str(e) or "CASE" in str(e):
                pytest.fail(f"SQL structure leaked in error: {e}")


class TestS3PresignedURLVulnerabilities:
    """
    Test S3 presigned URL manipulation and path traversal vulnerabilities
    """

    @pytest.mark.asyncio
    async def test_s3_path_traversal_in_file_path(self, root_user_client, test_user_token):
        """
        V-STORAGE-05: Path traversal in S3 file operations

        Location: hypha/artifact.py, hypha/s3.py - safe_join() usage

        Test if file_path parameters are properly validated to prevent accessing
        files outside the artifact directory using ../../../ patterns.
        """
        # Create artifact with a file
        artifact = await root_user_client.create(
            parent_id="collections/test-collection",
            manifest={"name": "Test Artifact"},
            type="dataset"
        )

        await root_user_client.put_file(
            artifact["id"],
            "data/file.txt",
            "normal file content"
        )

        await root_user_client.commit(artifact["id"], "Initial commit")

        # Attempt path traversal to access other workspace's files
        # Format: ../../../other-workspace/secrets.txt
        traversal_paths = [
            "../../../other-workspace/secret.txt",
            "../../../../../../etc/passwd",
            "./../secret-artifact/data.txt",
            "data/../../other-file.txt"
        ]

        for malicious_path in traversal_paths:
            try:
                url = await root_user_client.get_file(
                    artifact["id"],
                    malicious_path,
                    context={"ws": "test-workspace", "user": {"id": "user123"}}
                )

                # If URL is generated, check if it points outside artifact directory
                if "other-workspace" in url or "etc/passwd" in url or ".." in url:
                    pytest.fail(f"Path traversal not prevented: {url}")

            except Exception as e:
                # Should raise an exception for invalid paths
                # But error shouldn't reveal directory structure
                if "hypha" in str(e).lower() or "/Users/" in str(e):
                    pytest.fail(f"Error reveals file system structure: {e}")

    @pytest.mark.asyncio
    async def test_presigned_url_workspace_isolation(self, root_user_client, test_user_token):
        """
        V-STORAGE-06: Presigned URL doesn't enforce workspace isolation

        Test if presigned URLs generated for one workspace can be manipulated
        to access files in another workspace.
        """
        # Create artifact in workspace A
        artifact_a = await root_user_client.create(
            parent_id="workspace-a/collections/test",
            manifest={"name": "Workspace A Secret"},
            type="dataset"
        )

        await root_user_client.put_file(
            artifact_a["id"],
            "secret.txt",
            "workspace A confidential data"
        )

        await root_user_client.commit(artifact_a["id"], "Add secret")

        # Get presigned URL
        url_a = await root_user_client.get_file(
            artifact_a["id"],
            "secret.txt",
            context={"ws": "workspace-a", "user": {"id": "user-a"}}
        )

        # Attempt to manipulate the URL to access workspace B's files
        # This tests if the presigned URL signature only covers the bucket/key
        # and not the workspace prefix
        url_b = url_a.replace("workspace-a", "workspace-b")

        # Try to use the manipulated URL
        # (In a real test, we would make an HTTP request to the URL)
        # For now, check if the URL structure allows such manipulation
        if "workspace-a" in url_a and url_a.replace("workspace-a", "workspace-b") != url_a:
            # This indicates the workspace is part of the path, not the signature
            pytest.fail("Presigned URL may allow workspace manipulation - workspace not in signature")

    @pytest.mark.asyncio
    async def test_multipart_upload_id_manipulation(self, root_user_client):
        """
        V-STORAGE-07: Multipart upload ID manipulation

        Location: hypha/s3.py put_file_complete_multipart

        Test if upload_id from one user can be used to complete/abort
        uploads initiated by another user.
        """
        # User A starts a multipart upload
        result_a = await root_user_client.put_file_start_multipart(
            file_path="large-file.bin",
            part_count=5,
            context={"ws": "workspace-a", "user": {"id": "user-a"}}
        )

        upload_id_a = result_a["upload_id"]

        # User B attempts to complete User A's upload
        # This should be prevented - upload_id should be workspace-scoped
        parts = [
            {"part_number": 1, "etag": "fake-etag-1"},
            {"part_number": 2, "etag": "fake-etag-2"},
        ]

        try:
            await root_user_client.put_file_complete_multipart(
                file_path="large-file.bin",
                upload_id=upload_id_a,
                parts=parts,
                context={"ws": "workspace-b", "user": {"id": "user-b"}}
            )

            pytest.fail("Multipart upload completed across workspaces - missing isolation")

        except Exception as e:
            # Should fail, but check error doesn't leak upload details
            if upload_id_a in str(e):
                pytest.fail(f"Error leaks upload_id details: {e}")


class TestVectorSearchDataLeakage:
    """
    Test vector search data leakage and cross-workspace access
    """

    @pytest.mark.asyncio
    async def test_vector_search_cross_collection_leakage(self, root_user_client):
        """
        V-STORAGE-08: Vector search may leak data across collections

        Location: hypha/vectors.py search_vectors method

        Test if vector search properly isolates data between collections
        and doesn't return vectors from collections user can't access.
        """
        # Create two collections with similar vectors but different permissions
        public_collection = await root_user_client.create(
            parent_id="collections/public-vectors",
            manifest={"name": "Public Embeddings"},
            type="vector-collection",
            config={"permissions": {"*": "read"}}
        )

        private_collection = await root_user_client.create(
            parent_id="collections/private-vectors",
            manifest={"name": "Private Embeddings"},
            type="vector-collection",
            config={"permissions": {"admin": "read"}}
        )

        # Add similar vectors to both
        await root_user_client.add_vectors(
            public_collection["id"],
            vectors=[[1.0, 0.0, 0.0]],
            texts=["public document"],
            ids=["pub-1"]
        )

        await root_user_client.add_vectors(
            private_collection["id"],
            vectors=[[1.0, 0.1, 0.0]],  # Very similar vector
            texts=["confidential document"],
            ids=["priv-1"]
        )

        # Search public collection as regular user
        # Ensure private collection results don't leak
        results = await root_user_client.search_vectors(
            public_collection["id"],
            query=[1.0, 0.05, 0.0],
            k=10,
            context={"ws": "test-workspace", "user": {"id": "regular-user"}}
        )

        # Check if any private results leaked
        for result in results:
            if "confidential" in result.get("text", ""):
                pytest.fail("Vector search leaked private collection data")

    @pytest.mark.asyncio
    async def test_vector_search_filter_sql_injection(self, root_user_client):
        """
        V-STORAGE-09: SQL injection through vector search filters

        Location: hypha/vectors.py _convert_filters_to_hybrid_query

        Test if vector search filters properly sanitize user input
        """
        collection = await root_user_client.create(
            parent_id="collections/test-vectors",
            manifest={"name": "Test Vectors"},
            type="vector-collection"
        )

        # Add vectors with metadata
        await root_user_client.add_vectors(
            collection["id"],
            vectors=[[1.0, 0.0]],
            metadata=[{"category": "public", "sensitive": False}],
            ids=["vec-1"]
        )

        await root_user_client.add_vectors(
            collection["id"],
            vectors=[[0.0, 1.0]],
            metadata=[{"category": "secret", "sensitive": True}],
            ids=["vec-2"]
        )

        # Attempt SQL injection through filter
        malicious_filter = {
            "category": "public' OR sensitive = True --"
        }

        try:
            results = await root_user_client.search_vectors(
                collection["id"],
                query=[0.5, 0.5],
                filters=malicious_filter,
                k=10,
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            # Check if sensitive vector was returned
            for result in results:
                if result.get("id") == "vec-2":
                    pytest.fail("SQL injection in vector search filters bypassed restriction")

        except Exception as e:
            if "SQL" in str(e) or "syntax" in str(e):
                pytest.fail(f"SQL injection attempt revealed error: {e}")


class TestCollectionArtifactTraversal:
    """
    Test collection artifact directory traversal vulnerabilities
    """

    @pytest.mark.asyncio
    async def test_collection_path_traversal(self, root_user_client):
        """
        V-STORAGE-10: Collection artifact path traversal

        Test if collection operations (list_files, get_file) properly validate
        paths to prevent traversal outside the collection directory.
        """
        # Create a collection artifact
        collection = await root_user_client.create(
            parent_id=None,
            manifest={"name": "Test Collection"},
            type="collection",
            config={"is_collection": True}
        )

        # Add some child artifacts
        child = await root_user_client.create(
            parent_id=collection["id"],
            manifest={"name": "Child Artifact"},
            type="dataset"
        )

        # Attempt to access parent using path traversal in list operation
        try:
            results = await root_user_client.search(
                parent_id=collection["id"] + "/../..",
                context={"ws": "test-workspace", "user": {"id": "user123"}}
            )

            pytest.fail("Collection path traversal not prevented in search")

        except Exception as e:
            # Should fail, but not leak paths
            if "/Users/" in str(e) or "workspace/" in str(e):
                pytest.fail(f"Path traversal error leaks directory structure: {e}")
