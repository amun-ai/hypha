"""S3-based vector search engine with Zarr chunked storage for Hypha.

Implements a Zarr-optimized architecture with:
- SPANN (Space Partition Approximate Nearest Neighbor) indexing
- Zarr chunked arrays for efficient vector storage
- S3 for persistent storage with compression
- Redis for ephemeral caching
- Sharding for scalability
"""

import asyncio
import json
import logging
import pickle
import time
import hashlib
import heapq
import io
import lz4.frame
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
import threading

import zarr
import numcodecs
from numcodecs import Blosc
import aioboto3
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

from hypha.core.store import RedisStore
from hypha_rpc import RPC
from hypha_rpc.utils.schema import schema_method

logger = logging.getLogger(__name__)


class VectorLakeError(Exception):
    """Custom exception for VectorLake errors."""
    pass


@dataclass
class ShardMetadata:
    """Metadata for a Zarr-based vector shard."""
    shard_id: str
    centroid_id: int
    vector_count: int
    zarr_path: str  # Path to Zarr array in S3
    chunk_info: Dict[str, Any]  # Zarr chunking information
    created_at: float
    updated_at: float
    compression_ratio: float = 1.0  # Actual compression achieved


@dataclass
class CollectionMetadata:
    """Metadata for a vector collection."""
    collection_id: str
    dimension: int
    metric: str
    total_vectors: int
    num_centroids: int
    num_shards: int
    shard_size: int
    created_at: float
    updated_at: float
    index_version: str = "zarr_v1"


class HNSW:
    """Hierarchical Navigable Small World for centroid navigation.
    
    Optimized implementation for centroid search with Zarr storage.
    """
    
    def __init__(self, metric="cosine", m=16, ef=200, max_m=32, max_m0=64):
        self.metric = metric
        self.m = m
        self.max_m = max_m
        self.max_m0 = max_m0
        self.ef = ef
        self.ml = 1 / np.log(2.0)  # Level generation factor
        self.data = {}
        self.levels = {}  # node -> max level
        self.graph = defaultdict(lambda: defaultdict(set))  # level -> node -> connections
        self.entry_point = None
        self._size = 0
        
    def _get_random_level(self):
        """Generate level for new node."""
        level = int(-np.log(np.random.uniform()) * self.ml)
        return level
        
    def add(self, vector: np.ndarray, idx: int):
        """Add vector with improved connectivity."""
        vector = vector.astype(np.float32)
        if self.metric == "cosine":
            vector = vector / np.linalg.norm(vector)
            
        level = self._get_random_level()
        self.data[idx] = vector
        self.levels[idx] = level
        self._size += 1
        
        if self.entry_point is None:
            self.entry_point = idx
            return
            
        # Search from top level down to level 1
        curr_nearest = [self.entry_point]
        entry_level = self.levels.get(self.entry_point, 0)
        for lev in range(max(entry_level, level), level, -1):
            curr_nearest = self._search_layer(vector, curr_nearest, 1, lev)
            
        # Search and connect at each level from level down to 0
        for lev in range(min(level, entry_level), -1, -1):
            candidates = self._search_layer(vector, curr_nearest, self.ef, lev)
            max_conn = self.max_m0 if lev == 0 else self.max_m
            
            # Select diverse neighbors
            selected = self._select_neighbors(vector, candidates, min(max_conn, len(candidates)))
            
            # Add bidirectional connections
            for neighbor_idx in selected:
                self.graph[lev][idx].add(neighbor_idx)
                self.graph[lev][neighbor_idx].add(idx)
                
                # Prune connections if needed
                if len(self.graph[lev][neighbor_idx]) > max_conn:
                    # Keep best connections
                    neighbor_vector = self.data[neighbor_idx]
                    neighbor_connections = list(self.graph[lev][neighbor_idx])
                    scores = []
                    for conn in neighbor_connections:
                        if conn in self.data:
                            score = self._distance(neighbor_vector, self.data[conn])
                            scores.append((score, conn))
                    scores.sort()
                    
                    # Keep best connections
                    self.graph[lev][neighbor_idx] = set(conn for _, conn in scores[:max_conn])
                    
                    # Update reverse connections
                    for _, conn in scores[max_conn:]:
                        if idx in self.graph[lev][conn]:
                            self.graph[lev][conn].discard(neighbor_idx)
            
            curr_nearest = selected
            
        # Update entry point if necessary
        if level > self.levels.get(self.entry_point, 0):
            self.entry_point = idx
    
    def _select_neighbors(self, query: np.ndarray, candidates: List[int], m: int) -> List[int]:
        """Select diverse neighbors using simple heuristic."""
        if len(candidates) <= m:
            return candidates
            
        # Sort by distance
        scored = [(self._distance(query, self.data[idx]), idx) for idx in candidates if idx in self.data]
        scored.sort()
        
        return [idx for _, idx in scored[:m]]
    
    def _search_layer(self, query: np.ndarray, entry_points: List[int], num_closest: int, level: int) -> List[int]:
        """Optimized layer search."""
        visited = set()
        candidates = []
        w = []
        
        # Initialize with entry points
        for ep in entry_points:
            if ep in self.data:
                d = self._distance(query, self.data[ep])
                heapq.heappush(candidates, (-d, ep))
                heapq.heappush(w, (d, ep))
                visited.add(ep)
        
        while candidates:
            lowerBound, c = heapq.heappop(candidates)
            
            if -lowerBound > w[0][0]:
                break
                
            # Explore neighbors
            for neighbor in self.graph[level].get(c, set()):
                if neighbor not in visited and neighbor in self.data:
                    visited.add(neighbor)
                    d = self._distance(query, self.data[neighbor])
                    
                    if d < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-d, neighbor))
                        heapq.heappush(w, (d, neighbor))
                        
                        if len(w) > num_closest:
                            heapq.heappop(w)
        
        return [idx for _, idx in w]
    
    def search(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors."""
        if self.entry_point is None or self._size == 0:
            return []
            
        query = query.astype(np.float32)
        if self.metric == "cosine":
            query = query / np.linalg.norm(query)
            
        # Search from entry point down
        curr_nearest = [self.entry_point]
        entry_level = self.levels.get(self.entry_point, 0)
        for level in range(entry_level, 0, -1):
            curr_nearest = self._search_layer(query, curr_nearest, 1, level)
            
        # Search at level 0
        candidates = self._search_layer(query, curr_nearest, max(self.ef, k), 0)
        
        # Return top k with distances
        results = []
        for candidate in candidates[:k]:
            if candidate in self.data:
                dist = self._distance(query, self.data[candidate])
                results.append((dist, candidate))
                
        results.sort()
        return results[:k]
        
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Optimized distance calculation."""
        if self.metric == "cosine":
            return 1 - np.dot(a, b)
        elif self.metric == "l2":
            diff = a - b
            return np.dot(diff, diff)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
    def serialize(self) -> bytes:
        """Serialize to compressed bytes."""
        data = {
            "metric": self.metric,
            "m": self.m,
            "max_m": self.max_m,
            "max_m0": self.max_m0,
            "ef": self.ef,
            "ml": self.ml,
            "data": {k: v.tolist() for k, v in self.data.items()},
            "levels": self.levels,
            "graph": {
                str(level): {
                    str(node): list(neighbors) 
                    for node, neighbors in level_graph.items()
                } for level, level_graph in self.graph.items()
            },
            "entry_point": self.entry_point,
            "_size": self._size
        }
        return lz4.frame.compress(pickle.dumps(data))
        
    @classmethod
    def deserialize(cls, data: bytes) -> 'HNSW':
        """Deserialize from compressed bytes."""
        data = pickle.loads(lz4.frame.decompress(data))
        
        hnsw = cls(
            metric=data["metric"], 
            m=data["m"], 
            ef=data["ef"],
            max_m=data.get("max_m", 32),
            max_m0=data.get("max_m0", 64)
        )
        hnsw.ml = data.get("ml", 1 / np.log(2.0))
        hnsw.data = {int(k): np.array(v, dtype=np.float32) for k, v in data["data"].items()}
        hnsw.levels = {int(k): v for k, v in data["levels"].items()}
        hnsw.graph = defaultdict(lambda: defaultdict(set))
        
        for level_str, level_graph in data["graph"].items():
            level = int(level_str)
            for node_str, neighbors in level_graph.items():
                node = int(node_str)
                hnsw.graph[level][node] = set(neighbors)
                
        hnsw.entry_point = data["entry_point"]
        hnsw._size = data.get("_size", len(hnsw.data))
        return hnsw


class ZarrS3Store:
    """Custom Zarr store for S3 with async support."""
    
    def __init__(self, s3_pool, bucket: str, prefix: str):
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


class S3ConnectionPool:
    """Connection pool for S3 operations."""
    
    def __init__(self, s3_client_factory, max_connections=20):
        self.s3_client_factory = s3_client_factory
        self.max_connections = max_connections
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._created = 0
        self._lock = asyncio.Lock()
        
    async def get_client(self):
        """Get S3 client from pool."""
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            async with self._lock:
                if self._created < self.max_connections:
                    self._created += 1
                    return self.s3_client_factory()
                else:
                    return await self._pool.get()
    
    async def return_client(self, client):
        """Return client to pool."""
        try:
            self._pool.put_nowait(client)
        except asyncio.QueueFull:
            await client.close()
            async with self._lock:
                self._created -= 1


class RedisCache:
    """Redis caching layer with compression."""
    
    def __init__(self, redis_client, default_ttl=3600, compression_threshold=1024):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self._local_cache = OrderedDict()  # Small local cache
        self._local_cache_size = 100
        self._lock = threading.RLock()
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if above threshold."""
        if len(data) > self.compression_threshold:
            return b'LZ4:' + lz4.frame.compress(data)
        return b'RAW:' + data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compressed."""
        if data.startswith(b'LZ4:'):
            return lz4.frame.decompress(data[4:])
        elif data.startswith(b'RAW:'):
            return data[4:]
        return data
        
    async def get_centroids(self, collection_id: str) -> Optional[Tuple[np.ndarray, HNSW]]:
        """Get cached centroids with local cache."""
        cache_key = f"centroids:{collection_id}"
        
        # Check local cache first
        with self._lock:
            if cache_key in self._local_cache:
                self._local_cache.move_to_end(cache_key)
                return self._local_cache[cache_key]
        
        # Check Redis
        key = f"cache:centroids:{collection_id}"
        data = await self.redis.get(key)
        if data:
            try:
                decompressed = self._decompress_data(data)
                cached = pickle.loads(decompressed)
                centroids = np.array(cached["centroids"], dtype=np.float32)
                hnsw = HNSW.deserialize(cached["hnsw"])
                
                # Store in local cache
                result = (centroids, hnsw)
                with self._lock:
                    self._local_cache[cache_key] = result
                    if len(self._local_cache) > self._local_cache_size:
                        self._local_cache.popitem(last=False)
                        
                return result
            except Exception as e:
                logger.warning(f"Failed to deserialize centroids cache: {e}")
                
        return None
        
    async def set_centroids(self, collection_id: str, centroids: np.ndarray, hnsw: HNSW, ttl: Optional[int] = None):
        """Cache centroids with compression."""
        key = f"cache:centroids:{collection_id}"
        data = {
            "centroids": centroids.tolist(),
            "hnsw": hnsw.serialize()
        }
        
        serialized = pickle.dumps(data)
        compressed = self._compress_data(serialized)
        
        await self.redis.setex(key, ttl or self.default_ttl, compressed)
        
        # Update local cache
        cache_key = f"centroids:{collection_id}"
        with self._lock:
            self._local_cache[cache_key] = (centroids, hnsw)
            if len(self._local_cache) > self._local_cache_size:
                self._local_cache.popitem(last=False)
    
    async def get_shard_batch(self, collection_id: str, shard_ids: List[str]) -> Dict[str, Optional[Tuple[np.ndarray, List[Dict], List[str]]]]:
        """Get multiple shards in batch."""
        if not shard_ids:
            return {}
            
        # Build pipeline for batch get
        pipe = self.redis.pipeline()
        keys = [f"cache:shard:{collection_id}:{shard_id}" for shard_id in shard_ids]
        
        for key in keys:
            pipe.get(key)
            
        results = await pipe.execute()
        
        batch_results = {}
        for shard_id, data in zip(shard_ids, results):
            if data:
                try:
                    decompressed = self._decompress_data(data)
                    cached = pickle.loads(decompressed)
                    vectors = np.array(cached["vectors"], dtype=np.float32)
                    batch_results[shard_id] = (vectors, cached["metadata"], cached["ids"])
                except Exception as e:
                    logger.warning(f"Failed to deserialize shard {shard_id}: {e}")
                    batch_results[shard_id] = None
            else:
                batch_results[shard_id] = None
                
        return batch_results
    
    async def set_shard_batch(self, collection_id: str, shard_data: Dict[str, Tuple[np.ndarray, List[Dict], List[str]]], ttl: Optional[int] = None):
        """Set multiple shards in batch."""
        if not shard_data:
            return
            
        pipe = self.redis.pipeline()
        
        for shard_id, (vectors, metadata, ids) in shard_data.items():
            key = f"cache:shard:{collection_id}:{shard_id}"
            data = {
                "vectors": vectors.tolist(),
                "metadata": metadata,
                "ids": ids
            }
            serialized = pickle.dumps(data)
            compressed = self._compress_data(serialized)
            pipe.setex(key, ttl or (self.default_ttl // 4), compressed)
            
        await pipe.execute()
    
    async def get_metadata(self, collection_id: str) -> Optional[CollectionMetadata]:
        """Get cached collection metadata."""
        key = f"cache:meta:{collection_id}"
        data = await self.redis.get(key)
        if data:
            try:
                decompressed = self._decompress_data(data)
                metadata_dict = pickle.loads(decompressed)
                return CollectionMetadata(**metadata_dict)
            except Exception as e:
                logger.warning(f"Failed to deserialize metadata cache: {e}")
        return None
        
    async def set_metadata(self, collection_id: str, metadata: CollectionMetadata, ttl: Optional[int] = None):
        """Cache collection metadata."""
        key = f"cache:meta:{collection_id}"
        data = asdict(metadata)
        
        serialized = pickle.dumps(data)
        compressed = self._compress_data(serialized)
        
        await self.redis.setex(key, ttl or (self.default_ttl // 2), compressed)
        
    async def invalidate_collection(self, collection_id: str):
        """Invalidate all caches for a collection."""
        pattern = f"cache:*:{collection_id}*"
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break


class ZarrVectorShard:
    """Zarr-based vector shard with optimized chunking."""
    
    def __init__(self, store: ZarrS3Store, shard_path: str, dimension: int, chunk_size: int = 200):
        self.store = store
        self.shard_path = shard_path
        self.dimension = dimension
        self.chunk_size = chunk_size
        self.compressor = Blosc(cname='lz4', clevel=3, shuffle=Blosc.SHUFFLE)
        self._next_chunk_id = 0
        
    async def create(self, initial_size: int = 1000):
        """Create new Zarr arrays for vectors and metadata."""
        # Calculate optimal chunk size based on dimension and memory constraints
        # Target ~1MB per chunk for good S3 performance
        target_chunk_bytes = 1024 * 1024  # 1MB
        vector_bytes_per_item = self.dimension * 4  # float32
        optimal_chunk_size = max(50, min(self.chunk_size, target_chunk_bytes // vector_bytes_per_item))
        
        # Store chunking metadata
        metadata = {
            "dimension": self.dimension,
            "chunk_size": optimal_chunk_size,
            "compressor": str(self.compressor),
            "created_at": time.time(),
            "next_chunk_id": 0
        }
        
        await self.store.put(f"{self.shard_path}/.zmetadata", json.dumps(metadata).encode())
        
        logger.info(f"Created Zarr shard with chunk_size={optimal_chunk_size}, compression={self.compressor}")
    
    async def append_vectors(self, vectors: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Append vectors to the Zarr array."""
        if len(vectors) == 0:
            return
            
        # Load current metadata
        try:
            meta_data = await self.store.get(f"{self.shard_path}/.zmetadata")
            shard_meta = json.loads(meta_data.decode())
            self._next_chunk_id = shard_meta.get("next_chunk_id", 0)
        except:
            await self.create()
            self._next_chunk_id = 0
        
        # Store vectors in chunks
        for i in range(0, len(vectors), self.chunk_size):
            chunk_vectors = vectors[i:i+self.chunk_size]
            chunk_metadata = metadata[i:i+self.chunk_size]
            chunk_ids = ids[i:i+self.chunk_size]
            
            # Store each chunk
            await self._store_chunk(self._next_chunk_id, chunk_vectors, chunk_metadata, chunk_ids)
            self._next_chunk_id += 1
        
        # Update metadata
        shard_meta["next_chunk_id"] = self._next_chunk_id
        await self.store.put(f"{self.shard_path}/.zmetadata", json.dumps(shard_meta).encode())
    
    async def _store_chunk(self, chunk_idx: int, vectors: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Store a single chunk."""
        # Compress vectors using numcodecs directly
        compressed_data = self.compressor.encode(vectors.tobytes())
        
        # Store vector chunk
        vector_key = f"{self.shard_path}/vectors/{chunk_idx:06d}"
        await self.store.put(vector_key, compressed_data)
        
        # Store metadata chunk
        metadata_data = {
            "metadata": metadata,
            "ids": ids,
            "shape": vectors.shape,
            "dtype": str(vectors.dtype)
        }
        metadata_key = f"{self.shard_path}/metadata/{chunk_idx:06d}.json"
        await self.store.put(metadata_key, json.dumps(metadata_data).encode())
    
    async def load_chunks(self, chunk_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load specific chunks or all chunks."""
        if chunk_indices is None:
            # Load all chunks
            try:
                vector_paths = await self.store.list_prefix(f"{self.shard_path}/vectors/")
                chunk_indices = []
                for path in vector_paths:
                    filename = path.split('/')[-1]
                    if filename.isdigit() or (len(filename) == 6 and filename.isdigit()):
                        chunk_indices.append(int(filename))
                chunk_indices.sort()
            except:
                return np.array([]), [], []
        
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
            if len(vectors) > 0:
                all_vectors.append(vectors)
                all_metadata.extend(metadata)
                all_ids.extend(ids)
        
        if all_vectors:
            combined_vectors = np.vstack(all_vectors)
            return combined_vectors, all_metadata, all_ids
        
        return np.array([]), [], []
    
    async def _load_single_chunk(self, chunk_idx: int) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load a single chunk."""
        try:
            # Load vector data
            vector_key = f"{self.shard_path}/vectors/{chunk_idx:06d}"
            vector_bytes = await self.store.get(vector_key)
            
            # Load metadata
            metadata_key = f"{self.shard_path}/metadata/{chunk_idx:06d}.json"
            metadata_bytes = await self.store.get(metadata_key)
            metadata_data = json.loads(metadata_bytes.decode())
            
            # Decompress vectors using numcodecs
            shape = tuple(metadata_data["shape"])
            dtype = np.dtype(metadata_data.get("dtype", "float32"))
            
            # Get compressor from shard metadata or use default
            compressor = Blosc(cname='lz4', clevel=3, shuffle=Blosc.SHUFFLE)
            
            # Decompress and reshape
            decompressed_bytes = compressor.decode(vector_bytes)
            vectors = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
            
            return vectors, metadata_data["metadata"], metadata_data["ids"]
        except Exception as e:
            logger.warning(f"Failed to load chunk {chunk_idx}: {e}")
            return np.array([]), [], []


class S3VectorSearchEngine:
    """S3-based vector search engine with Zarr chunked storage."""

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
        """Initialize Zarr-based S3 vector search engine."""
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
        
        self.redis_cache = RedisCache(redis_client, cache_ttl) if redis_client else None
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
        """Create a factory function for S3 client creation."""

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
        """Calculate optimal number of centroids based on dataset size."""
        if num_vectors < 1000:
            return max(4, int(np.sqrt(num_vectors) / 5))
        elif num_vectors < 100_000:
            return max(10, int(np.sqrt(num_vectors) / 10))
        elif num_vectors < 1_000_000:
            return max(50, int(np.sqrt(num_vectors) / 20))
        else:
            return min(2000, int(np.sqrt(num_vectors) / 50))
    
    def optimize_hnsw_params(self, num_centroids: int) -> Dict[str, int]:
        """Optimize HNSW parameters based on number of centroids."""
        if num_centroids < 50:
            return {"m": 8, "ef": 50, "max_m": 16, "max_m0": 32}
        elif num_centroids < 500:
            return {"m": 16, "ef": 100, "max_m": 32, "max_m0": 64}
        else:
            return {"m": 24, "ef": 200, "max_m": 48, "max_m0": 96}

    async def initialize(self):
        """Initialize the search engine and create bucket if needed."""
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
        dimension: int = None,  # Allow None to support config dict
        distance_metric: str = "cosine",
        overwrite: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new vector collection.
        
        Args:
            collection_name: Collection identifier
            dimension: Dimension of vectors OR config dict for backward compatibility
            distance_metric: Distance metric - 'cosine', 'l2', or 'ip' (default: 'cosine')
            overwrite: Whether to overwrite existing collection
            
        Returns:
            Collection metadata
        """
        # Handle backward compatibility - support passing config as dimension parameter
        if isinstance(dimension, dict):
            config = dimension
            dimension = config.get("dimension", 384)
            distance_metric = config.get("metric", distance_metric)
            overwrite = config.get("overwrite", overwrite)
            # Merge config into kwargs
            for key, value in config.items():
                if key not in ["dimension", "metric", "overwrite"]:
                    kwargs[key] = value
        
        # Set default if still None
        if dimension is None:
            dimension = 384
            
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # Validate parameters
        if dimension <= 0 or dimension > 2048:
            raise ValueError(f"Dimension {dimension} must be between 1 and 2048")
        
        # Normalize distance metric names for consistency with pgvector
        if distance_metric.lower() == "inner_product":
            distance_metric = "ip"
        
        if distance_metric.lower() not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Distance metric must be 'cosine', 'l2', 'ip', or 'inner_product', got {distance_metric}")
        
        # Use normalized metric name
        metric = distance_metric.lower()
        num_centroids = kwargs.get("num_centroids", self.calculate_optimal_centroids(10000))
        shard_size = kwargs.get("shard_size", self.shard_size)
        
        # Check if collection exists
        if not overwrite:
            try:
                await self.get_collection_info(collection_id)
                raise ValueError(f"Collection {collection_name} already exists.")
            except VectorLakeError:
                # Collection doesn't exist, proceed
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
            index_version="zarr_v1"
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
        
        # Initialize index with Zarr storage
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
        
        # Build HNSW index with optimized parameters
        hnsw_params = self.optimize_hnsw_params(num_centroids)
        hnsw_index = HNSW(
            metric=metric,
            m=hnsw_params["m"],
            ef=hnsw_params["ef"],
            max_m=hnsw_params["max_m"],
            max_m0=hnsw_params["max_m0"]
        )
        
        # Add centroids to HNSW
        for i, centroid in enumerate(centroids):
            hnsw_index.add(centroid, i)
        
        # Store centroids using Zarr
        zarr_store = self.zarr_stores[collection_id]
        
        # Create compressor for centroids
        compressor = Blosc(cname='lz4', clevel=self.compression_level)
        
        # Compress centroids data directly
        compressed_centroids = compressor.encode(centroids.tobytes())
        
        # Store centroids as compressed data
        await zarr_store.put("centroids/data", compressed_centroids)
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
        vectors: Union[List[Dict[str, Any]], List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        update: bool = False,
        embedding_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Add vectors to a collection using Zarr storage."""
        collection_id = collection_name
        
        # Process vector documents and extract components (same as original)
        vector_data = []
        metadata_list = []
        ids_list = []
        
        # Handle different input formats
        if isinstance(vectors, np.ndarray):
            # Direct numpy array
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            vector_data = [vectors[i] for i in range(len(vectors))]
            metadata_list = metadata or [{} for _ in range(len(vectors))]
            if ids:
                ids_list = ids
            else:
                ids_list = [str(hashlib.md5(v.tobytes()).hexdigest()) for v in vector_data]
        elif isinstance(vectors, list) and len(vectors) > 0:
            if isinstance(vectors[0], np.ndarray):
                # List of numpy arrays
                vector_data = vectors
                metadata_list = metadata or [{} for _ in range(len(vectors))]
                if ids:
                    ids_list = ids
                else:
                    ids_list = [str(hashlib.md5(v.tobytes()).hexdigest()) for v in vector_data]
            elif isinstance(vectors[0], (list, tuple)):
                # List of lists/tuples
                vector_data = [np.array(v, dtype=np.float32) for v in vectors]
                metadata_list = metadata or [{} for _ in range(len(vectors))]
                if ids:
                    ids_list = ids
                else:
                    ids_list = [str(hashlib.md5(v.tobytes()).hexdigest()) for v in vector_data]
            elif isinstance(vectors[0], dict):
                # List of dictionaries (original format)
                for i, vector_doc in enumerate(vectors):
                    # Extract vector
                    vector_value = vector_doc.get("vector")
                    if vector_value is None:
                        raise ValueError("'vector' field is required")
                    
                    # Handle different vector formats
                    if isinstance(vector_value, str) and (embedding_model or self.model):
                        # Text to embed
                        model_to_use = self.model if self.model else None
                        if model_to_use:
                            embeddings = list(model_to_use.embed([vector_value]))
                            vector_array = np.array(embeddings[0], dtype=np.float32)
                        else:
                            raise ValueError("No embedding model available for text query")
                    elif isinstance(vector_value, (list, tuple)):
                        vector_array = np.array(vector_value, dtype=np.float32)
                    elif isinstance(vector_value, np.ndarray):
                        vector_array = vector_value.astype(np.float32)
                    else:
                        raise ValueError(f"Invalid vector type: {type(vector_value)}")
                    
                    vector_data.append(vector_array)
                    
                    # Extract metadata (try both 'metadata' and 'manifest' keys)
                    meta = vector_doc.get("metadata", vector_doc.get("manifest", {}))
                    metadata_list.append(meta)
                    
                    # Extract or generate ID
                    vector_id = vector_doc.get("id")
                    if vector_id is None:
                        if update:
                            raise ValueError("ID is required for update operation.")
                        vector_id = str(hashlib.md5(vector_array.tobytes()).hexdigest())
                    ids_list.append(str(vector_id))
            else:
                raise ValueError(f"Invalid vectors format: {type(vectors[0])}")
        else:
            # Empty list or invalid format
            if not vectors:
                return {"added": 0, "ids": []}
        
        # Convert to numpy array
        vectors_array = np.array(vector_data, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors_array = vectors_array / norms
        
        # Load centroids and assign vectors to shards
        centroids, hnsw_index = await self._load_index(collection_id)
        
        # Group vectors by nearest centroid
        shard_assignments = defaultdict(list)
        for i, vector in enumerate(vectors_array):
            nearest_centroids = hnsw_index.search(vector, k=1)
            if nearest_centroids:
                centroid_id = nearest_centroids[0][1]  # (distance, centroid_id)
                shard_assignments[centroid_id].append((i, vector, metadata_list[i], ids_list[i]))
        
        # Write to Zarr shards
        zarr_store = self.zarr_stores.get(collection_id)
        if not zarr_store:
            zarr_store = ZarrS3Store(self.s3_pool, self.bucket_name, f"{collection_id}/zarr")
            self.zarr_stores[collection_id] = zarr_store
        
        for centroid_id, assignments in shard_assignments.items():
            shard_id = f"shard_{centroid_id:06d}_{int(time.time())}"
            shard_path = f"shards/{shard_id}"
            
            # Create Zarr shard
            shard = ZarrVectorShard(zarr_store, shard_path, vectors_array.shape[1], self.zarr_chunk_size)
            await shard.create(len(assignments))
            
            # Extract vectors and metadata for this shard
            indices, shard_vectors, shard_metadata, shard_ids = zip(*assignments)
            shard_vectors_array = np.array(shard_vectors, dtype=np.float32)
            
            # Append to Zarr shard
            await shard.append_vectors(shard_vectors_array, list(shard_metadata), list(shard_ids))
            
            # Update shard metadata
            shard_meta = ShardMetadata(
                shard_id=shard_id,
                centroid_id=centroid_id,
                vector_count=len(assignments),
                zarr_path=shard_path,
                chunk_info={"chunk_size": self.zarr_chunk_size},
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # Store shard metadata
            if collection_id not in self.shard_managers:
                self.shard_managers[collection_id] = {}
            self.shard_managers[collection_id][shard_id] = shard_meta
        
        # Invalidate cache
        if self.redis_cache:
            await self.redis_cache.invalidate_collection(collection_id)
        
        return {"added": len(ids_list), "ids": ids_list}

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
        """Search for similar vectors using Zarr chunked storage."""
        collection_id = collection_name
        
        if query_vector is None:
            raise ValueError("query_vector must be provided")
        
        # Handle text query vector
        if isinstance(query_vector, str):
            if embedding_model and self.model:
                embeddings = list(self.model.embed([query_vector]))
                query_vector = embeddings[0]
            elif self.model:
                embeddings = list(self.model.embed([query_vector]))
                query_vector = embeddings[0]
            else:
                raise ValueError("No embedding model available for text query")
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Load index
        centroids, hnsw_index = await self._load_index(collection_id)
        
        # Adaptive shard selection based on dataset size and limit
        total_centroids = len(centroids)
        if limit <= 10:
            num_shards_to_search = min(3, total_centroids)
        elif limit <= 50:
            num_shards_to_search = min(8, total_centroids)
        elif limit <= 200:
            num_shards_to_search = min(15, total_centroids)
        else:
            num_shards_to_search = min(25, total_centroids)
        
        # Find nearest centroids using HNSW
        nearest_centroids = hnsw_index.search(query_vector, k=num_shards_to_search)
        
        if not nearest_centroids:
            return [] if not pagination else {"items": [], "total": 0, "offset": offset, "limit": limit}
        
        # Load shard metadata
        if collection_id not in self.shard_managers:
            await self._load_shard_metadata(collection_id)
        
        shard_metadata = self.shard_managers.get(collection_id, {})
        
        # Find relevant shards for each centroid
        relevant_shards = []
        for dist, centroid_id in nearest_centroids:
            shards_for_centroid = [
                shard_id for shard_id, meta in shard_metadata.items()
                if meta.centroid_id == centroid_id
            ]
            relevant_shards.extend(shards_for_centroid)
        
        if not relevant_shards:
            return [] if not pagination else {"items": [], "total": 0, "offset": offset, "limit": limit}
        
        # Search shards using Zarr
        results = await self._search_zarr_shards(
            collection_id, query_vector, relevant_shards, limit * 3, filters, include_vectors
        )
        
        # Sort by score and apply offset and limit
        results.sort(key=lambda x: x["_score"], reverse=True)
        start_idx = offset
        end_idx = offset + limit
        paginated_results = results[start_idx:end_idx]
        
        # Return with pagination info if requested
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
        shard_ids: List[str],
        max_results: int,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Search Zarr shards efficiently."""
        if not shard_ids:
            return []
        
        shard_metadata = self.shard_managers.get(collection_id, {})
        zarr_store = self.zarr_stores.get(collection_id)
        
        if not zarr_store:
            return []
        
        # Process shards with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit concurrent shard processing
        
        async def process_shard(shard_id):
            async with semaphore:
                return await self._process_single_zarr_shard(
                    zarr_store, shard_metadata.get(shard_id), query_vector, filters, include_vectors
                )
        
        shard_tasks = [process_shard(shard_id) for shard_id in shard_ids if shard_id in shard_metadata]
        shard_results = await asyncio.gather(*shard_tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
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
        shard_meta: Optional[ShardMetadata],
        query_vector: np.ndarray,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """Process a single Zarr shard."""
        if not shard_meta:
            return []
            
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
            logger.warning(f"Failed to process Zarr shard {shard_meta.shard_id if shard_meta else 'unknown'}: {e}")
            return []
    
    async def _load_index(self, collection_id: str) -> Tuple[np.ndarray, HNSW]:
        """Load centroids and HNSW index with caching."""
        # Check memory cache first
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
            compressed_centroid_data = await zarr_store.get("centroids/data")
            centroid_metadata = json.loads((await zarr_store.get("centroids/metadata.json")).decode())
            
            # Decompress centroids
            compressor = Blosc(cname='lz4', clevel=self.compression_level)
            centroid_data = compressor.decode(compressed_centroid_data)
            
            # Reconstruct centroids array
            shape = tuple(centroid_metadata["shape"])
            dtype = np.dtype(centroid_metadata["dtype"])
            centroids = np.frombuffer(centroid_data, dtype=dtype).reshape(shape)
            
            # Load HNSW index
            hnsw_data = await zarr_store.get("centroids/hnsw.bin")
            hnsw_index = HNSW.deserialize(hnsw_data)
            
            # Cache results
            result = (centroids, hnsw_index)
            self.indices[collection_id] = result
            if self.redis_cache:
                await self.redis_cache.set_centroids(collection_id, centroids, hnsw_index)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load Zarr index for {collection_id}: {e}")
            raise VectorLakeError(f"Collection {collection_id} not found")
    
    async def _load_shard_metadata(self, collection_id: str):
        """Load shard metadata for a collection."""
        zarr_store = self.zarr_stores.get(collection_id)
        if not zarr_store:
            zarr_store = ZarrS3Store(self.s3_pool, self.bucket_name, f"{collection_id}/zarr")
            self.zarr_stores[collection_id] = zarr_store
        
        try:
            # List shard directories
            shard_paths = await zarr_store.list_prefix("shards/")
            
            shard_metadata = {}
            for path in shard_paths:
                if path.endswith("/.zmetadata"):
                    shard_dir = path[:-len("/.zmetadata")]
                    shard_id = shard_dir.split('/')[-1]
                    
                    # Load shard metadata
                    try:
                        meta_data = await zarr_store.get(path)
                        shard_meta_dict = json.loads(meta_data.decode())
                        
                        # Extract centroid_id from shard_id
                        centroid_id = int(shard_id.split('_')[1]) if '_' in shard_id else 0
                        
                        shard_metadata[shard_id] = ShardMetadata(
                            shard_id=shard_id,
                            centroid_id=centroid_id,
                            vector_count=0,  # Will be calculated dynamically
                            zarr_path=shard_dir,
                            chunk_info={"chunk_size": shard_meta_dict.get("chunk_size", self.zarr_chunk_size)},
                            created_at=shard_meta_dict.get("created_at", time.time()),
                            updated_at=time.time()
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load shard metadata for {shard_id}: {e}")
            
            self.shard_managers[collection_id] = shard_metadata
            
        except Exception as e:
            logger.debug(f"Failed to load shard metadata for {collection_id}: {e}")
            self.shard_managers[collection_id] = {}

    async def get_collection_info(
        self, collection_name: str
    ) -> Dict[str, Any]:
        """Get information about a collection."""
        collection_id = collection_name
        
        # Try cache first
        if self.redis_cache:
            cached = await self.redis_cache.get_metadata(collection_id)
            if cached:
                return asdict(cached)
        
        # Load from S3
        client = await self.s3_pool.get_client()
        try:
            meta_key = f"{collection_id}/metadata.json"
            response = await client.get_object(
                Bucket=self.bucket_name,
                Key=meta_key
            )
            meta_data = await response["Body"].read()
            metadata = CollectionMetadata(**json.loads(meta_data))
            
            # Cache it
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

    async def remove_vectors(self, collection_name: str, ids: List[str], **kwargs):
        """Remove vectors from a collection."""
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to find and remove vectors from the appropriate shards
        logger.warning(f"remove_vectors not fully implemented for Zarr engine for collection {collection_name}")
        return len(ids)

    async def get_vector(self, collection_name: str, id: str) -> Dict[str, Any]:
        """Get a single vector from a collection."""
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to search through shards to find the vector
        logger.warning(f"get_vector not fully implemented for Zarr engine for collection {collection_name}")
        raise NotImplementedError("get_vector not fully implemented for Zarr engine")

    async def list_vectors(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 10,
        include_vectors: bool = False,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """List vectors in a collection."""
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to iterate through all shards and collect vectors
        logger.warning(f"list_vectors not fully implemented for Zarr engine for collection {collection_name}")
        
        # Return empty results for now
        results = []
        
        if pagination:
            return {
                "items": results,
                "total": 0,
                "offset": offset,
                "limit": limit,
            }
        else:
            return results

    async def count(self, collection_name: str) -> int:
        """Count vectors in a collection."""
        try:
            info = await self.get_collection_info(collection_name)
            return info.get("total_vectors", 0)
        except:
            return 0

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection and all its data."""
        collection_id = collection_name
        
        # Delete from S3
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

    async def _embed_texts(
        self,
        texts: List[str],
        embedding_model: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate embeddings for texts using fastembed."""
        model_to_use = embedding_model or self.embedding_model
        
        if not model_to_use:
            raise ValueError("No embedding model configured")
        
        if not TextEmbedding:
            raise ImportError("fastembed is not installed")
        
        # Use instance model if available and matches
        if self.model and (not embedding_model or embedding_model == self.embedding_model):
            embeddings = list(self.model.embed(texts))
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        
        # Create temporary model for different model name
        if model_to_use.startswith("fastembed:"):
            model_name = model_to_use.split(":")[1]
        else:
            model_name = model_to_use
        
        temp_model = TextEmbedding(
            model_name=model_name,
            cache_dir=self.cache_dir
        )
        embeddings = list(temp_model.embed(texts))
        return [np.array(emb, dtype=np.float32) for emb in embeddings]


def create_s3_vector_engine_from_config(config: Dict[str, Any]) -> S3VectorSearchEngine:
    """Create an S3 vector search engine from configuration."""
    return S3VectorSearchEngine(
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


async def create_s3_vector_engine(store: RedisStore, config: Dict[str, Any]) -> S3VectorSearchEngine:
    """Create an S3 vector search engine with Redis caching from store."""
    config = config.copy()
    config["redis_client"] = store.get_redis() if store else None
    engine = create_s3_vector_engine_from_config(config)
    await engine.initialize()
    return engine