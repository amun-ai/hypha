"""
Zarr-optimized S3-based vector search engine.

Key advantages of Zarr for vector storage:
1. Chunked storage - load only needed vector chunks
2. Built-in compression (blosc, lz4) - reduce S3 transfer costs
3. Parallel chunk loading - better concurrency
4. Rich metadata - store vector metadata efficiently
5. Append-friendly - easier incremental updates
"""

import asyncio
import json
import logging
import time
import hashlib
import io
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

import zarr
import aioboto3
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

from hypha.s3vector_optimized import (
    VectorLakeError, 
    OptimizedHNSW, 
    S3ConnectionPool,
    OptimizedRedisCache,
    CollectionMetadata
)

logger = logging.getLogger(__name__)


@dataclass 
class ZarrShardMetadata:
    """Metadata for a Zarr-based shard."""
    shard_id: str
    centroid_id: int
    vector_count: int
    zarr_path: str  # Path to Zarr array in S3
    chunk_info: Dict[str, Any]  # Zarr chunking information
    created_at: float
    updated_at: float
    compression_ratio: float = 1.0  # Actual compression achieved


class ZarrS3Store:
    """Custom Zarr store for S3 with async support."""
    
    def __init__(self, s3_pool: S3ConnectionPool, bucket: str, prefix: str):
        self.s3_pool = s3_pool
        self.bucket = bucket
        self.prefix = prefix.rstrip('/')
        self._cache = {}  # Simple in-memory cache for metadata
    
    def _get_key(self, path: str) -> str:
        """Convert Zarr path to S3 key."""
        return f"{self.prefix}/{path.lstrip('/')}"
    
    async def get(self, path: str) -> bytes:
        """Get data from S3."""
        if path in self._cache:
            return self._cache[path]
            
        key = self._get_key(path)
        client = await self.s3_pool.get_client()
        
        try:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            data = await response["Body"].read()
            
            # Cache small metadata files
            if len(data) < 1024:  # 1KB threshold
                self._cache[path] = data
                
            return data
        except Exception as e:
            raise KeyError(f"Key {path} not found: {e}")
        finally:
            await self.s3_pool.return_client(client)
    
    async def put(self, path: str, data: bytes):
        """Put data to S3."""
        key = self._get_key(path)
        client = await self.s3_pool.get_client()
        
        try:
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)
            
            # Update cache for small files
            if len(data) < 1024:
                self._cache[path] = data
        finally:
            await self.s3_pool.return_client(client)
    
    async def delete(self, path: str):
        """Delete from S3."""
        key = self._get_key(path)
        client = await self.s3_pool.get_client()
        
        try:
            await client.delete_object(Bucket=self.bucket, Key=key)
            self._cache.pop(path, None)
        finally:
            await self.s3_pool.return_client(client)
    
    async def list_prefix(self, prefix: str) -> List[str]:
        """List objects with prefix."""
        full_prefix = self._get_key(prefix)
        client = await self.s3_pool.get_client()
        
        try:
            objects = []
            paginator = client.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get('Contents', []):
                    # Remove the full prefix to get relative path
                    rel_path = obj['Key'][len(self.prefix)+1:]
                    objects.append(rel_path)
            
            return objects
        finally:
            await self.s3_pool.return_client(client)


class ZarrVectorShard:
    """Zarr-based vector shard with optimized chunking."""
    
    def __init__(self, store: ZarrS3Store, shard_path: str, dimension: int):
        self.store = store
        self.shard_path = shard_path
        self.dimension = dimension
        self._zarr_array = None
        self._metadata_array = None
        
    async def create(self, initial_size: int = 1000, chunk_size: int = 100):
        """Create new Zarr arrays for vectors and metadata."""
        # Calculate optimal chunk size based on dimension and memory constraints
        # Target ~1MB per chunk for good S3 performance
        target_chunk_bytes = 1024 * 1024  # 1MB
        vector_bytes_per_item = self.dimension * 4  # float32
        optimal_chunk_size = max(50, min(chunk_size, target_chunk_bytes // vector_bytes_per_item))
        
        # Create vector array with compression
        vector_chunks = (optimal_chunk_size, self.dimension)
        
        # Use blosc compression with lz4 for speed
        compressor = zarr.Blosc(cname='lz4', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        
        # Create arrays asynchronously
        await self._create_zarr_arrays(initial_size, vector_chunks, compressor)
        
        logger.info(f"Created Zarr shard with chunks={vector_chunks}, compression={compressor}")
    
    async def _create_zarr_arrays(self, initial_size: int, vector_chunks: Tuple[int, int], compressor):
        """Create the Zarr arrays."""
        # We'll simulate async Zarr operations
        # In practice, you'd use zarr-python with async stores
        
        # For now, create the arrays synchronously but store metadata about chunking
        self.chunk_info = {
            "vector_chunks": vector_chunks,
            "compressor": str(compressor),
            "initial_size": initial_size
        }
        
        # Store chunking metadata
        metadata = {
            "dimension": self.dimension,
            "chunk_info": self.chunk_info,
            "created_at": time.time()
        }
        
        await self.store.put(f"{self.shard_path}/.zmetadata", json.dumps(metadata).encode())
    
    async def append_vectors(self, vectors: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Append vectors to the Zarr array."""
        # In a full implementation, this would:
        # 1. Load current array size
        # 2. Resize array if needed
        # 3. Write new chunks
        # 4. Update metadata
        
        # For now, simulate by storing vectors in chunks
        chunk_size = self.chunk_info["vector_chunks"][0]
        
        for i in range(0, len(vectors), chunk_size):
            chunk_vectors = vectors[i:i+chunk_size]
            chunk_metadata = metadata[i:i+chunk_size]
            chunk_ids = ids[i:i+chunk_size]
            
            # Store each chunk
            chunk_idx = i // chunk_size
            await self._store_chunk(chunk_idx, chunk_vectors, chunk_metadata, chunk_ids)
    
    async def _store_chunk(self, chunk_idx: int, vectors: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Store a single chunk."""
        # Compress vectors using zarr's compression
        compressor = zarr.Blosc(cname='lz4', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        
        # Create temporary zarr array for compression
        temp_array = zarr.array(vectors, compressor=compressor)
        vector_bytes = temp_array.tobytes()
        
        # Store vector chunk
        vector_key = f"{self.shard_path}/vectors/{chunk_idx:06d}"
        await self.store.put(vector_key, vector_bytes)
        
        # Store metadata chunk
        metadata_data = {
            "metadata": metadata,
            "ids": ids,
            "shape": vectors.shape
        }
        metadata_key = f"{self.shard_path}/metadata/{chunk_idx:06d}.json"
        await self.store.put(metadata_key, json.dumps(metadata_data).encode())
    
    async def load_chunks(self, chunk_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load specific chunks or all chunks."""
        if chunk_indices is None:
            # Load all chunks
            vector_paths = await self.store.list_prefix(f"{self.shard_path}/vectors/")
            chunk_indices = [int(p.split('/')[-1]) for p in vector_paths if p.split('/')[-1].isdigit()]
            chunk_indices.sort()
        
        if not chunk_indices:
            return np.array([]), [], []
        
        # Load chunks in parallel
        tasks = []
        for chunk_idx in chunk_indices:
            tasks.append(self._load_single_chunk(chunk_idx))
        
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_vectors = []
        all_metadata = []
        all_ids = []
        
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.warning(f"Failed to load chunk: {result}")
                continue
                
            vectors, metadata, ids = result
            all_vectors.append(vectors)
            all_metadata.extend(metadata)
            all_ids.extend(ids)
        
        if all_vectors:
            combined_vectors = np.vstack(all_vectors)
            return combined_vectors, all_metadata, all_ids
        
        return np.array([]), [], []
    
    async def _load_single_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load a single chunk."""
        # Load vector data
        vector_key = f"{self.shard_path}/vectors/{chunk_idx:06d}"
        vector_bytes = await self.store.get(vector_key)
        
        # Load metadata
        metadata_key = f"{self.shard_path}/metadata/{chunk_idx:06d}.json"
        metadata_bytes = await self.store.get(metadata_key)
        metadata_data = json.loads(metadata_bytes.decode())
        
        # Decompress vectors
        compressor = zarr.Blosc(cname='lz4', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        shape = tuple(metadata_data["shape"])
        
        # Reconstruct zarr array and extract data
        temp_array = zarr.array(np.frombuffer(vector_bytes, dtype=np.float32).reshape(shape))
        vectors = np.array(temp_array)
        
        return vectors, metadata_data["metadata"], metadata_data["ids"]


class ZarrS3VectorSearchEngine:
    """S3Vector engine optimized with Zarr storage."""
    
    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str = "us-east-1",
        bucket_name: str = "vector-search",
        embedding_model: Optional[str] = None,
        redis_client: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        num_centroids: int = 100,
        shard_size: int = 10000,  # Smaller shards work better with Zarr
        cache_ttl: int = 3600,
        max_s3_connections: int = 20,
        zarr_chunk_size: int = 200,  # Vectors per Zarr chunk
        compression_level: int = 3,  # Zarr compression level
    ):
        """Initialize Zarr-optimized S3 vector search engine."""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.embedding_model = embedding_model
        self.num_centroids = num_centroids
        self.shard_size = shard_size
        self.zarr_chunk_size = zarr_chunk_size
        self.compression_level = compression_level
        
        # Initialize components
        self.session = aioboto3.Session()
        self.s3_client_factory = self._create_client_factory()
        self.s3_pool = S3ConnectionPool(self.s3_client_factory, max_s3_connections)
        
        self.redis_cache = OptimizedRedisCache(redis_client, cache_ttl) if redis_client else None
        self.indices = {}  # Collection ID -> (centroids, hnsw)
        self.zarr_stores = {}  # Collection ID -> ZarrS3Store
        self.shard_managers = {}  # Collection ID -> shard metadata
        
        # Initialize embedding model
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dim = None
        
        if embedding_model and TextEmbedding:
            try:
                if embedding_model.startswith("fastembed:"):
                    model_name = embedding_model.split(":")[1]
                else:
                    model_name = embedding_model
                
                self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
                test_embedding = list(self.model.embed(["test"]))[0]
                self.embedding_dim = len(test_embedding)
            except Exception as e:
                logger.error(f"Failed to load embedding model {embedding_model}: {e}")
                self.model = None
                self.embedding_dim = None
    
    def _create_client_factory(self):
        """Create optimized S3 client factory."""
        def factory():
            return self.session.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region_name,
                config=aioboto3.session.Config(
                    max_pool_connections=50,
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    read_timeout=60,
                    connect_timeout=10
                )
            )
        return factory
    
    def calculate_optimal_centroids(self, num_vectors: int) -> int:
        """Calculate optimal number of centroids."""
        if num_vectors < 1000:
            return max(4, int(np.sqrt(num_vectors) / 5))
        elif num_vectors < 100_000:
            return max(10, int(np.sqrt(num_vectors) / 10))
        elif num_vectors < 1_000_000:
            return max(50, int(np.sqrt(num_vectors) / 20))
        else:
            return min(2000, int(np.sqrt(num_vectors) / 50))
    
    async def initialize(self):
        """Initialize the search engine."""
        client = await self.s3_pool.get_client()
        try:
            await client.head_bucket(Bucket=self.bucket_name)
        except:
            await client.create_bucket(Bucket=self.bucket_name)
        finally:
            await self.s3_pool.return_client(client)
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int = None,
        distance_metric: str = "cosine",
        overwrite: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new vector collection with Zarr storage."""
        # Handle backward compatibility
        if isinstance(dimension, dict):
            config = dimension
            dimension = config.get("dimension", 384)
            distance_metric = config.get("metric", distance_metric)
            overwrite = config.get("overwrite", overwrite)
            kwargs.update(config)
        
        if dimension is None:
            dimension = 384
            
        collection_id = collection_name
        
        # Validate parameters
        if dimension <= 0 or dimension > 2048:
            raise ValueError(f"Dimension {dimension} must be between 1 and 2048")
        
        # Normalize distance metric
        if distance_metric.lower() == "inner_product":
            distance_metric = "ip"
        
        if distance_metric.lower() not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Distance metric must be 'cosine', 'l2', 'ip', or 'inner_product'")
        
        metric = distance_metric.lower()
        num_centroids = kwargs.get("num_centroids", self.calculate_optimal_centroids(10000))
        shard_size = kwargs.get("shard_size", self.shard_size)
        
        # Check if collection exists
        if not overwrite:
            try:
                await self.get_collection_info(collection_id)
                raise ValueError(f"Collection {collection_name} already exists.")
            except VectorLakeError:
                pass
        
        # Create Zarr store for this collection
        zarr_store = ZarrS3Store(self.s3_pool, self.bucket_name, f"{collection_id}/zarr")
        self.zarr_stores[collection_id] = zarr_store
        
        # Create metadata
        metadata = CollectionMetadata(
            collection_id=collection_id,
            dimension=dimension,
            metric=metric,
            total_vectors=0,
            num_centroids=num_centroids,
            num_shards=0,
            shard_size=shard_size,
            created_at=time.time(),
            updated_at=time.time(),
            index_version="v2_zarr"
        )
        
        # Save metadata to S3
        client = await self.s3_pool.get_client()
        try:
            meta_key = f"{collection_id}/metadata.json"
            await client.put_object(
                Bucket=self.bucket_name,
                Key=meta_key,
                Body=json.dumps(asdict(metadata))
            )
        finally:
            await self.s3_pool.return_client(client)
        
        # Initialize optimized index
        await self._initialize_zarr_index(collection_id, dimension, num_centroids, metric)
        
        # Cache metadata
        if self.redis_cache:
            await self.redis_cache.set_metadata(collection_id, metadata)
        
        return asdict(metadata)
    
    async def _initialize_zarr_index(self, collection_id: str, dimension: int, num_centroids: int, metric: str):
        """Initialize centroids and HNSW index with Zarr storage."""
        # Generate initial centroids
        if metric == "cosine":
            centroids = np.random.randn(num_centroids, dimension).astype(np.float32)
            centroids = normalize(centroids, axis=1)
        else:
            centroids = np.random.randn(num_centroids, dimension).astype(np.float32)
        
        # Build HNSW index
        hnsw_index = OptimizedHNSW(metric=metric, m=16, ef=100)
        for i, centroid in enumerate(centroids):
            hnsw_index.add(centroid, i)
        
        # Store centroids using Zarr
        zarr_store = self.zarr_stores[collection_id]
        
        # Create Zarr array for centroids with compression
        compressor = zarr.Blosc(cname='lz4', clevel=self.compression_level)
        centroid_array = zarr.array(centroids, compressor=compressor)
        
        # Store centroids as compressed Zarr
        await zarr_store.put("centroids/data", centroid_array.tobytes())
        await zarr_store.put("centroids/metadata.json", json.dumps({
            "shape": centroids.shape,
            "dtype": str(centroids.dtype),
            "compressor": str(compressor)
        }).encode())
        
        # Store HNSW index
        hnsw_data = hnsw_index.serialize()
        await zarr_store.put("centroids/hnsw.bin", hnsw_data)
        
        # Cache in memory
        self.indices[collection_id] = (centroids, hnsw_index)
        if self.redis_cache:
            await self.redis_cache.set_centroids(collection_id, centroids, hnsw_index)
    
    async def add_vectors(
        self, 
        collection_name: str, 
        vectors, 
        metadata=None, 
        ids=None, 
        **kwargs
    ):
        """Add vectors using Zarr storage."""
        collection_id = collection_name
        
        # Process input vectors (similar to original implementation)
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            vector_data = [vectors[i] for i in range(len(vectors))]
        elif isinstance(vectors, list):
            if isinstance(vectors[0], np.ndarray):
                vector_data = vectors
            else:
                vector_data = [np.array(v, dtype=np.float32) for v in vectors]
        else:
            raise ValueError("Invalid vector format")
        
        vectors_array = np.array(vector_data, dtype=np.float32)
        
        if metadata is None:
            metadata = [{} for _ in range(len(vectors_array))]
        if ids is None:
            ids = [f"vec_{hashlib.md5(v.tobytes()).hexdigest()}" for v in vectors_array]
        
        # Load centroids and assign vectors to shards
        centroids, hnsw_index = await self._load_index(collection_id)
        
        # Group vectors by nearest centroid
        shard_assignments = defaultdict(list)
        for i, vector in enumerate(vectors_array):
            nearest_centroids = hnsw_index.search(vector, k=1)
            if nearest_centroids:
                centroid_id = nearest_centroids[0][1]  # (distance, centroid_id)
                shard_assignments[centroid_id].append((i, vector, metadata[i], ids[i]))
        
        # Write to Zarr shards
        zarr_store = self.zarr_stores[collection_id]
        
        for centroid_id, assignments in shard_assignments.items():
            shard_id = f"shard_{centroid_id:06d}_{int(time.time())}"
            shard_path = f"shards/{shard_id}"
            
            # Create Zarr shard
            shard = ZarrVectorShard(zarr_store, shard_path, vectors_array.shape[1])
            await shard.create(len(assignments), self.zarr_chunk_size)
            
            # Extract vectors and metadata for this shard
            indices, shard_vectors, shard_metadata, shard_ids = zip(*assignments)
            shard_vectors_array = np.array(shard_vectors, dtype=np.float32)
            
            # Append to Zarr shard
            await shard.append_vectors(shard_vectors_array, list(shard_metadata), list(shard_ids))
            
            # Update shard metadata
            shard_meta = ZarrShardMetadata(
                shard_id=shard_id,
                centroid_id=centroid_id,
                vector_count=len(assignments),
                zarr_path=shard_path,
                chunk_info=shard.chunk_info,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # Store shard metadata
            if collection_id not in self.shard_managers:
                self.shard_managers[collection_id] = {}
            self.shard_managers[collection_id][shard_id] = shard_meta
        
        return {"added": len(ids), "ids": ids}
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: Optional[Union[List[float], np.ndarray, str]] = None,
        embedding_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        include_vectors: bool = False,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search vectors using Zarr storage with chunked loading."""
        collection_id = collection_name
        
        if query_vector is None:
            raise ValueError("query_vector must be provided")
        
        # Handle text query vector
        if isinstance(query_vector, str):
            if self.model:
                embeddings = list(self.model.embed([query_vector]))
                query_vector = np.array(embeddings[0], dtype=np.float32)
            else:
                raise ValueError("No embedding model available for text query")
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Normalize for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Load index
        centroids, hnsw_index = await self._load_index(collection_id)
        
        # Find nearest centroids
        num_shards_to_search = min(5, len(centroids))  # Conservative for Zarr
        nearest_centroids = hnsw_index.search(query_vector, k=num_shards_to_search)
        
        if not nearest_centroids:
            return [] if not pagination else {"items": [], "total": 0, "offset": offset, "limit": limit}
        
        # Load relevant shards using Zarr
        results = await self._search_zarr_shards(
            collection_id, query_vector, nearest_centroids, limit * 2, filters, include_vectors
        )
        
        # Sort and paginate
        results.sort(key=lambda x: x["_score"], reverse=True)
        start_idx = offset
        end_idx = offset + limit
        paginated_results = results[start_idx:end_idx]
        
        if pagination:
            return {
                "items": paginated_results,
                "total": len(results),
                "offset": offset,
                "limit": limit,
            }
        else:
            return paginated_results
    
    async def _search_zarr_shards(
        self,
        collection_id: str,
        query_vector: np.ndarray,
        nearest_centroids: List[Tuple[float, int]],
        max_results: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Search Zarr shards efficiently."""
        if collection_id not in self.shard_managers:
            return []
        
        shard_metadata = self.shard_managers[collection_id]
        zarr_store = self.zarr_stores[collection_id]
        
        # Find relevant shards
        relevant_shards = []
        for dist, centroid_id in nearest_centroids:
            shards_for_centroid = [
                shard_id for shard_id, meta in shard_metadata.items()
                if meta.centroid_id == centroid_id
            ]
            relevant_shards.extend(shards_for_centroid)
        
        if not relevant_shards:
            return []
        
        # Load shards using Zarr with chunked access
        all_results = []
        
        # Process shards with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent shard processing
        
        async def process_shard(shard_id):
            async with semaphore:
                return await self._process_single_zarr_shard(
                    zarr_store, shard_metadata[shard_id], query_vector, filters, include_vectors
                )
        
        shard_tasks = [process_shard(shard_id) for shard_id in relevant_shards]
        shard_results = await asyncio.gather(*shard_tasks, return_exceptions=True)
        
        # Combine results
        for result in shard_results:
            if isinstance(result, Exception):
                logger.warning(f"Shard processing failed: {result}")
                continue
            if result:
                all_results.extend(result)
        
        return all_results[:max_results]
    
    async def _process_single_zarr_shard(
        self,
        zarr_store: ZarrS3Store,
        shard_meta: ZarrShardMetadata,
        query_vector: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Process a single Zarr shard."""
        try:
            # Create shard object
            shard = ZarrVectorShard(zarr_store, shard_meta.zarr_path, len(query_vector))
            
            # Load vectors from Zarr chunks
            vectors, metadata, ids = await shard.load_chunks()
            
            if len(vectors) == 0:
                return []
            
            # Calculate similarities
            similarities = np.dot(vectors, query_vector)
            
            # Apply filters
            if filters:
                mask = np.ones(len(metadata), dtype=bool)
                for field_name, value in filters.items():
                    field_values = [meta.get(field_name) for meta in metadata]
                    field_mask = np.array([v == value for v in field_values])
                    mask &= field_mask
                
                similarities = similarities[mask]
                metadata = [metadata[i] for i, keep in enumerate(mask) if keep]
                ids = [ids[i] for i, keep in enumerate(mask) if keep]
                if include_vectors:
                    vectors = vectors[mask]
            
            # Create results
            results = []
            for i, score in enumerate(similarities):
                result_item = {
                    "id": ids[i],
                    "_score": float(score),
                }
                result_item.update(metadata[i])
                
                if include_vectors:
                    result_item["vector"] = vectors[i].tolist()
                
                results.append(result_item)
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to process Zarr shard {shard_meta.shard_id}: {e}")
            return []
    
    async def _load_index(self, collection_id: str) -> Tuple[np.ndarray, OptimizedHNSW]:
        """Load centroids and HNSW index."""
        # Check memory cache
        if collection_id in self.indices:
            return self.indices[collection_id]
        
        # Check Redis cache
        if self.redis_cache:
            cached = await self.redis_cache.get_centroids(collection_id)
            if cached:
                self.indices[collection_id] = cached
                return cached
        
        # Load from Zarr
        zarr_store = self.zarr_stores.get(collection_id)
        if not zarr_store:
            zarr_store = ZarrS3Store(self.s3_pool, self.bucket_name, f"{collection_id}/zarr")
            self.zarr_stores[collection_id] = zarr_store
        
        try:
            # Load centroids from Zarr
            centroid_data = await zarr_store.get("centroids/data")
            centroid_metadata = json.loads((await zarr_store.get("centroids/metadata.json")).decode())
            
            # Reconstruct centroids array
            shape = tuple(centroid_metadata["shape"])
            dtype = np.dtype(centroid_metadata["dtype"])
            centroids = np.frombuffer(centroid_data, dtype=dtype).reshape(shape)
            
            # Load HNSW index
            hnsw_data = await zarr_store.get("centroids/hnsw.bin")
            hnsw_index = OptimizedHNSW.deserialize(hnsw_data)
            
            # Cache results
            result = (centroids, hnsw_index)
            self.indices[collection_id] = result
            if self.redis_cache:
                await self.redis_cache.set_centroids(collection_id, centroids, hnsw_index)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load Zarr index for {collection_id}: {e}")
            raise VectorLakeError(f"Collection {collection_id} not found")
    
    # Implement other required methods...
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection metadata."""
        collection_id = collection_name
        
        if self.redis_cache:
            try:
                cached = await self.redis_cache.get_metadata(collection_id)
                if cached:
                    return asdict(cached)
            except:
                pass
        
        client = await self.s3_pool.get_client()
        try:
            meta_key = f"{collection_id}/metadata.json"
            response = await client.get_object(Bucket=self.bucket_name, Key=meta_key)
            meta_data = await response["Body"].read()
            metadata = CollectionMetadata(**json.loads(meta_data))
            
            if self.redis_cache:
                await self.redis_cache.set_metadata(collection_id, metadata)
            
            return asdict(metadata)
        except:
            raise VectorLakeError(f"Collection {collection_id} not found")
        finally:
            await self.s3_pool.return_client(client)
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        client = await self.s3_pool.get_client()
        try:
            collections = []
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Delimiter='/'):
                for prefix_info in page.get('CommonPrefixes', []):
                    collection_id = prefix_info['Prefix'].rstrip('/')
                    collections.append(collection_id)
            return collections
        finally:
            await self.s3_pool.return_client(client)
    
    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection."""
        collection_id = collection_name
        
        client = await self.s3_pool.get_client()
        try:
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=f"{collection_id}/"):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects:
                        await client.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={'Objects': objects}
                        )
        finally:
            await self.s3_pool.return_client(client)
        
        # Clear caches
        if self.redis_cache:
            await self.redis_cache.invalidate_collection(collection_id)
        
        # Remove from memory
        if collection_id in self.indices:
            del self.indices[collection_id]
        if collection_id in self.zarr_stores:
            del self.zarr_stores[collection_id]
        if collection_id in self.shard_managers:
            del self.shard_managers[collection_id]
    
    async def count(self, collection_name: str) -> int:
        """Count vectors in collection."""
        try:
            info = await self.get_collection_info(collection_name)
            return info.get("total_vectors", 0)
        except:
            return 0


# Factory functions
def create_zarr_s3_vector_engine_from_config(config: Dict[str, Any]) -> ZarrS3VectorSearchEngine:
    """Create a Zarr-optimized S3 vector search engine from configuration."""
    return ZarrS3VectorSearchEngine(
        endpoint_url=config.get("endpoint_url"),
        access_key_id=config.get("access_key_id"),
        secret_access_key=config.get("secret_access_key"),
        region_name=config.get("region_name", "us-east-1"),
        bucket_name=config.get("bucket_name", "vector-search"),
        embedding_model=config.get("embedding_model"),
        redis_client=config.get("redis_client"),
        num_centroids=config.get("num_centroids", 100),
        shard_size=config.get("shard_size", 10000),
        cache_ttl=config.get("cache_ttl", 3600),
        max_s3_connections=config.get("max_s3_connections", 20),
        zarr_chunk_size=config.get("zarr_chunk_size", 200),
        compression_level=config.get("compression_level", 3),
    )