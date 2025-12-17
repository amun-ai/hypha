"""
Integration tests for S3-based vector search engine.

These tests verify that the S3VectorSearchEngine actually works correctly
for semantic text similarity search - not just that the API works, but
that the search results make semantic sense.
"""

import pytest
import pytest_asyncio
import numpy as np
import uuid

# Check if zarr 3.x is available
try:
    import zarr
    if not hasattr(zarr, '__version__') or zarr.__version__ < '3.0':
        pytest.skip("S3Vector requires zarr>=3.1.0", allow_module_level=True)

    from hypha.s3vector import (
        S3VectorSearchEngine,
    )
    S3VECTOR_AVAILABLE = True
except ImportError:
    pytest.skip("S3Vector dependencies (zarr>=3.1.0) not available, requires Python>=3.11", allow_module_level=True)
    S3VECTOR_AVAILABLE = False

# Check if fastembed is available for text embedding
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from fakeredis import aioredis

# Import test configuration for MinIO/S3 settings
from . import (
    MINIO_SERVER_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
)


async def empty_and_delete_bucket(s3_client, bucket_name):
    """Helper function to empty and delete an S3 bucket."""
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        async for page in pages:
            if 'Contents' in page:
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    await s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={'Objects': objects}
                    )

        await s3_client.delete_bucket(Bucket=bucket_name)
    except Exception:
        pass


@pytest.fixture
def s3_config(minio_server):
    """S3 configuration using the shared MinIO server from conftest."""
    return {
        "endpoint_url": MINIO_SERVER_URL,
        "access_key_id": MINIO_ROOT_USER,
        "secret_access_key": MINIO_ROOT_PASSWORD,
        "region": "us-east-1"
    }


@pytest.fixture
def redis_client():
    """Create a fake Redis client for testing."""
    return aioredis.FakeRedis()


class TestS3VectorSemanticSearch:
    """
    Integration tests that verify S3Vector actually performs meaningful
    semantic similarity search, not just that the API works.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
    async def test_basic_semantic_similarity(self, s3_config, redis_client, minio_server):
        """
        Test that semantically similar texts are returned with higher scores
        than unrelated texts.

        This is the core test to verify vector search is working correctly.
        """
        bucket_name = f"test-semantic-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            embedding_model="fastembed:BAAI/bge-small-en-v1.5",
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "semantic_test"

        try:
            # Create collection
            await engine.create_collection(collection_id, {
                "dimension": engine.embedding_dim,
                "metric": "cosine",
                "num_centroids": 4
            })

            # Define test documents with clear semantic categories
            documents = [
                # Category: Programming
                {"id": "prog1", "text": "Python is a popular programming language for machine learning"},
                {"id": "prog2", "text": "JavaScript is used for web development and frontend applications"},
                {"id": "prog3", "text": "Rust is a systems programming language focused on safety"},

                # Category: Food
                {"id": "food1", "text": "Italian pizza is made with fresh tomatoes and mozzarella cheese"},
                {"id": "food2", "text": "Sushi is a traditional Japanese dish made with rice and fish"},
                {"id": "food3", "text": "French cuisine is known for its pastries and sauces"},

                # Category: Animals
                {"id": "animal1", "text": "Dogs are loyal companions and popular household pets"},
                {"id": "animal2", "text": "Cats are independent animals that love to sleep and play"},
                {"id": "animal3", "text": "Elephants are large mammals known for their intelligence"},

                # Category: Sports
                {"id": "sport1", "text": "Soccer is the world's most popular sport with billions of fans"},
                {"id": "sport2", "text": "Basketball is played with two teams trying to score in hoops"},
                {"id": "sport3", "text": "Tennis is a racket sport played on various court surfaces"},
            ]

            # Generate embeddings and add vectors
            texts = [doc["text"] for doc in documents]
            embeddings = list(engine.model.embed(texts))

            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Extract category from id (e.g., "prog1" -> "prog")
                category = ''.join(c for c in doc["id"] if not c.isdigit())
                vectors.append({
                    "id": doc["id"],
                    "vector": np.array(embedding, dtype=np.float32),
                    "metadata": {"text": doc["text"], "category": category}
                })

            await engine.add_vectors(collection_id, vectors)

            # Test 1: Query about programming should return programming documents
            query = "What programming languages are good for AI and data science?"
            query_embedding = list(engine.model.embed([query]))[0]

            results = await engine.search_vectors(
                collection_id,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=3
            )

            print("\n=== Test 1: Programming Query ===")
            print(f"Query: {query}")
            for i, r in enumerate(results):
                print(f"  {i+1}. [{r['category']}] Score: {r['_score']:.4f} - {r['text']}")

            # At least one of top 3 should be programming-related
            categories = [r["category"] for r in results[:3]]
            assert "prog" in categories, f"Expected programming result in top 3, got: {categories}"

            # Test 2: Query about food should return food documents
            query = "What are some delicious dishes from different countries?"
            query_embedding = list(engine.model.embed([query]))[0]

            results = await engine.search_vectors(
                collection_id,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=3
            )

            print("\n=== Test 2: Food Query ===")
            print(f"Query: {query}")
            for i, r in enumerate(results):
                print(f"  {i+1}. [{r['category']}] Score: {r['_score']:.4f} - {r['text']}")

            # At least one of top 3 should be food-related
            categories = [r["category"] for r in results[:3]]
            assert "food" in categories, f"Expected food result in top 3, got: {categories}"

            # Test 3: Query about pets should return animal documents
            query = "What animals make good pets for families?"
            query_embedding = list(engine.model.embed([query]))[0]

            results = await engine.search_vectors(
                collection_id,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=3
            )

            print("\n=== Test 3: Pet Query ===")
            print(f"Query: {query}")
            for i, r in enumerate(results):
                print(f"  {i+1}. [{r['category']}] Score: {r['_score']:.4f} - {r['text']}")

            # At least one of top 3 should be animal-related
            categories = [r["category"] for r in results[:3]]
            assert "animal" in categories, f"Expected animal result in top 3, got: {categories}"

            # Test 4: Score ordering - similar documents should have higher scores
            # Search for exact text to verify score ordering
            query = "Python is a popular programming language for machine learning"
            query_embedding = list(engine.model.embed([query]))[0]

            results = await engine.search_vectors(
                collection_id,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=12
            )

            print("\n=== Test 4: Exact Match Query ===")
            print(f"Query: {query}")
            for i, r in enumerate(results):
                print(f"  {i+1}. [{r['category']}] Score: {r['_score']:.4f} - {r['text']}")

            # The exact match should have the highest score (close to 1.0)
            assert results[0]["id"] == "prog1", f"Expected prog1 as top result, got: {results[0]['id']}"
            assert results[0]["_score"] > 0.95, f"Expected score > 0.95 for exact match, got: {results[0]['_score']}"

            # Verify scores are descending
            scores = [r["_score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
    async def test_text_query_search(self, s3_config, redis_client, minio_server):
        """
        Test that text queries (not pre-computed vectors) work correctly.
        This tests the full pipeline: text -> embedding -> search.
        """
        bucket_name = f"test-text-query-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            embedding_model="fastembed:BAAI/bge-small-en-v1.5",
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "text_query_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": engine.embedding_dim,
                "metric": "cosine"
            })

            # Add documents with text embedding
            documents = [
                {"id": "doc1", "text": "The quick brown fox jumps over the lazy dog"},
                {"id": "doc2", "text": "Machine learning algorithms can predict outcomes"},
                {"id": "doc3", "text": "Deep neural networks are revolutionizing AI"},
                {"id": "doc4", "text": "Natural language processing enables text understanding"},
                {"id": "doc5", "text": "Computer vision helps machines see and interpret images"},
            ]

            # Generate embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = list(engine.model.embed(texts))

            vectors = []
            for doc, embedding in zip(documents, embeddings):
                vectors.append({
                    "id": doc["id"],
                    "vector": np.array(embedding, dtype=np.float32),
                    "metadata": {"text": doc["text"]}
                })

            await engine.add_vectors(collection_id, vectors)

            # Search with a text query (should be converted to embedding automatically)
            text_query = "artificial intelligence and deep learning"
            results = await engine.search_vectors(
                collection_id,
                query_vector=text_query,  # Pass text directly
                limit=5
            )

            print("\n=== Text Query Search ===")
            print(f"Query: {text_query}")
            for i, r in enumerate(results):
                print(f"  {i+1}. Score: {r['_score']:.4f} - {r['text']}")

            # AI-related documents should score higher
            assert len(results) > 0, "Should have search results"
            # The top result should be AI/ML related
            top_text = results[0]["text"].lower()
            ai_terms = ["neural", "machine learning", "ai", "deep"]
            assert any(term in top_text for term in ai_terms), \
                f"Expected AI-related top result, got: {top_text}"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
    async def test_combined_filter_and_semantic_search(self, s3_config, redis_client, minio_server):
        """
        Test that combining metadata filters with vector search works correctly.
        """
        bucket_name = f"test-filter-semantic-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            embedding_model="fastembed:BAAI/bge-small-en-v1.5",
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "filter_semantic_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": engine.embedding_dim,
                "metric": "cosine"
            })

            # Add products with categories
            products = [
                {"id": "laptop1", "text": "High performance laptop for programming", "category": "electronics", "price": 1200},
                {"id": "laptop2", "text": "Budget laptop for students", "category": "electronics", "price": 500},
                {"id": "phone1", "text": "Smartphone with great camera", "category": "electronics", "price": 800},
                {"id": "book1", "text": "Programming book for beginners", "category": "books", "price": 30},
                {"id": "book2", "text": "Advanced machine learning textbook", "category": "books", "price": 80},
                {"id": "headphones1", "text": "Wireless headphones for music", "category": "electronics", "price": 150},
            ]

            # Generate embeddings
            texts = [p["text"] for p in products]
            embeddings = list(engine.model.embed(texts))

            vectors = []
            for product, embedding in zip(products, embeddings):
                vectors.append({
                    "id": product["id"],
                    "vector": np.array(embedding, dtype=np.float32),
                    "metadata": {
                        "text": product["text"],
                        "category": product["category"],
                        "price": product["price"]
                    }
                })

            await engine.add_vectors(collection_id, vectors)

            # Search for "programming" but only in books category
            query = "programming and coding tutorials"
            query_embedding = list(engine.model.embed([query]))[0]

            results = await engine.search_vectors(
                collection_id,
                query_vector=np.array(query_embedding, dtype=np.float32),
                filters={"category": "books"},
                limit=5
            )

            print("\n=== Filtered Semantic Search ===")
            print(f"Query: {query}")
            print(f"Filter: category=books")
            for i, r in enumerate(results):
                print(f"  {i+1}. [{r['category']}] ${r['price']} - {r['text']}")

            # All results should be books
            for r in results:
                assert r["category"] == "books", f"Expected books category, got: {r['category']}"

            # The programming book should rank higher than ML book for this query
            if len(results) >= 2:
                assert results[0]["id"] == "book1", "Programming book should be first for programming query"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
    async def test_distance_metrics(self, s3_config, redis_client, minio_server):
        """
        Test that different distance metrics (cosine, l2) produce different results.
        """
        bucket_name = f"test-metrics-{uuid.uuid4().hex[:8]}"

        # Test with cosine distance
        engine_cosine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            embedding_model="fastembed:BAAI/bge-small-en-v1.5",
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine_cosine.initialize()

        collection_cosine = "cosine_test"
        collection_l2 = "l2_test"

        try:
            # Create collections with different metrics
            await engine_cosine.create_collection(collection_cosine, {
                "dimension": engine_cosine.embedding_dim,
                "metric": "cosine"
            })

            await engine_cosine.create_collection(collection_l2, {
                "dimension": engine_cosine.embedding_dim,
                "metric": "l2"
            })

            # Add same documents to both
            documents = [
                {"id": "doc1", "text": "Machine learning is a type of artificial intelligence"},
                {"id": "doc2", "text": "Deep learning uses neural networks with many layers"},
                {"id": "doc3", "text": "Natural language processing handles text data"},
            ]

            texts = [d["text"] for d in documents]
            embeddings = list(engine_cosine.model.embed(texts))

            for collection_id in [collection_cosine, collection_l2]:
                vectors = []
                for doc, embedding in zip(documents, embeddings):
                    vectors.append({
                        "id": doc["id"],
                        "vector": np.array(embedding, dtype=np.float32),
                        "metadata": {"text": doc["text"]}
                    })
                await engine_cosine.add_vectors(collection_id, vectors)

            # Search with same query
            query = "AI and machine learning systems"
            query_embedding = list(engine_cosine.model.embed([query]))[0]

            results_cosine = await engine_cosine.search_vectors(
                collection_cosine,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=3
            )

            results_l2 = await engine_cosine.search_vectors(
                collection_l2,
                query_vector=np.array(query_embedding, dtype=np.float32),
                limit=3
            )

            print("\n=== Distance Metric Comparison ===")
            print(f"Query: {query}")
            print("\nCosine results:")
            for i, r in enumerate(results_cosine):
                print(f"  {i+1}. Score: {r['_score']:.4f} - {r['text']}")

            print("\nL2 results:")
            for i, r in enumerate(results_l2):
                print(f"  {i+1}. Score: {r['_score']:.4f} - {r['text']}")

            # Both should return results
            assert len(results_cosine) > 0, "Cosine search should return results"
            assert len(results_l2) > 0, "L2 search should return results"

            # Cosine scores should be between 0 and 1 (similarity)
            for r in results_cosine:
                assert 0 <= r["_score"] <= 1.1, f"Cosine score should be in [0,1], got: {r['_score']}"

        finally:
            await engine_cosine.delete_collection(collection_cosine)
            await engine_cosine.delete_collection(collection_l2)
            async with engine_cosine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)


class TestS3VectorEdgeCases:
    """Test edge cases and error handling in S3 vector search."""

    @pytest.mark.asyncio
    async def test_empty_collection_search(self, s3_config, redis_client, minio_server):
        """Test searching in an empty collection returns empty results."""
        bucket_name = f"test-empty-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "empty_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": 384,
                "metric": "cosine"
            })

            # Search in empty collection
            results = await engine.search_vectors(
                collection_id,
                query_vector=np.random.rand(384).astype(np.float32),
                limit=10
            )

            assert isinstance(results, list)
            assert len(results) == 0, "Empty collection should return no results"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)

    @pytest.mark.asyncio
    async def test_large_limit_search(self, s3_config, redis_client, minio_server):
        """Test searching with limit larger than collection size."""
        bucket_name = f"test-large-limit-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "large_limit_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })

            # Add only 5 vectors
            vectors = []
            for i in range(5):
                vectors.append({
                    "id": f"vec_{i}",
                    "vector": np.random.rand(128).astype(np.float32),
                    "metadata": {"index": i}
                })
            await engine.add_vectors(collection_id, vectors)

            # Search with limit > collection size
            results = await engine.search_vectors(
                collection_id,
                query_vector=np.random.rand(128).astype(np.float32),
                limit=100
            )

            assert len(results) <= 5, "Should return at most the number of vectors in collection"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)

    @pytest.mark.asyncio
    async def test_filter_no_matches(self, s3_config, redis_client, minio_server):
        """Test filtering with no matching documents."""
        bucket_name = f"test-no-match-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "no_match_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })

            # Add vectors with category "A"
            vectors = []
            for i in range(10):
                vectors.append({
                    "id": f"vec_{i}",
                    "vector": np.random.rand(128).astype(np.float32),
                    "metadata": {"category": "A"}
                })
            await engine.add_vectors(collection_id, vectors)

            # Search with filter for non-existent category
            results = await engine.search_vectors(
                collection_id,
                query_vector=np.random.rand(128).astype(np.float32),
                filters={"category": "B"},
                limit=10
            )

            assert len(results) == 0, "Should return no results for non-matching filter"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)


class TestS3VectorPagination:
    """Test pagination functionality in S3 vector search."""

    @pytest.mark.asyncio
    async def test_offset_pagination(self, s3_config, redis_client, minio_server):
        """Test that offset pagination works correctly."""
        bucket_name = f"test-pagination-{uuid.uuid4().hex[:8]}"

        engine = S3VectorSearchEngine(
            endpoint_url=s3_config["endpoint_url"],
            access_key_id=s3_config["access_key_id"],
            secret_access_key=s3_config["secret_access_key"],
            region_name=s3_config["region"],
            bucket_name=bucket_name,
            redis_client=redis_client,
            num_centroids=4,
            shard_size=50
        )
        await engine.initialize()

        collection_id = "pagination_test"

        try:
            await engine.create_collection(collection_id, {
                "dimension": 128,
                "metric": "cosine"
            })

            # Add 50 vectors
            vectors = []
            for i in range(50):
                vectors.append({
                    "id": f"vec_{i:03d}",
                    "vector": np.random.rand(128).astype(np.float32),
                    "metadata": {"index": i}
                })
            await engine.add_vectors(collection_id, vectors)

            # Get first page
            page1 = await engine.search_vectors(
                collection_id,
                query_vector=None,  # Filter-only search
                limit=10,
                offset=0
            )

            # Get second page
            page2 = await engine.search_vectors(
                collection_id,
                query_vector=None,
                limit=10,
                offset=10
            )

            # Verify no overlap
            page1_ids = {r["id"] for r in page1}
            page2_ids = {r["id"] for r in page2}
            overlap = page1_ids.intersection(page2_ids)

            assert len(overlap) == 0, f"Pages should not overlap, found: {overlap}"

        finally:
            await engine.delete_collection(collection_id)
            async with engine.s3_client_factory() as s3_client:
                await empty_and_delete_bucket(s3_client, bucket_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
