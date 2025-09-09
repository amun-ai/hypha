"""S3-based vector search engine with SPANN indexing for Hypha.

Implements a Turbopuffer-inspired architecture with:
- SPANN (Space Partition Approximate Nearest Neighbor) indexing
- S3 for persistent storage with Zarr format
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
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import aioboto3
import numpy as np

# Try to import zarr 3.x - required for S3Vector
try:
    import zarr
    import zarr.api.asynchronous as zarr_async
    from zarr.abc.store import Store, ByteRequest
    from zarr.core.buffer import default_buffer_prototype
    from zarr.core.common import AccessModeLiteral
    ZARR_AVAILABLE = True
    ZARR_VERSION = zarr.__version__
except ImportError as e:
    ZARR_AVAILABLE = False
    ZARR_VERSION = None
    # Create dummy classes to avoid NameError
    Store = object
    ByteRequest = object
    default_buffer_prototype = None
    AccessModeLiteral = str

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None

from hypha.core.store import RedisStore

logger = logging.getLogger(__name__)


class VectorLakeError(Exception):
    """Custom exception for VectorLake errors."""
    pass


@dataclass
class ShardMetadata:
    """Metadata for a vector shard."""
    shard_id: str
    centroid_id: int
    vector_count: int
    batch_files: List[str]
    created_at: float
    updated_at: float


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


class HNSW:
    """Hierarchical Navigable Small World for centroid navigation.
    
    Simplified implementation optimized for centroid search.
    """
    
    def __init__(self, metric="cosine", m=16, ef=200):
        self.metric = metric
        self.m = m  # Number of connections per layer
        self.ef = ef  # Size of dynamic candidate list
        self.data = []
        self.graph = defaultdict(set)
        self.entry_point = None
        
    def add(self, vector: np.ndarray, idx: int):
        """Add a vector to the HNSW index."""
        # Store vector at the index position
        if len(self.data) <= idx:
            self.data.extend([None] * (idx + 1 - len(self.data)))
        self.data[idx] = vector
        
        if self.entry_point is None:
            self.entry_point = idx
            return
            
        # Find nearest neighbors among existing nodes
        neighbors = self._search_layer(vector, self.ef)
        
        # For small graphs, ensure minimum connectivity
        if len(neighbors) < self.m and len([d for d in self.data if d is not None]) > 1:
            # If we don't have enough neighbors, add more connections
            # This helps with small graphs
            all_nodes = [i for i in range(len(self.data)) 
                        if i != idx and self.data[i] is not None]
            if all_nodes:
                # Calculate distances to all nodes
                distances = [(i, self._distance(vector, self.data[i])) 
                           for i in all_nodes]
                distances.sort(key=lambda x: x[1])
                # Use up to m nearest as neighbors
                neighbors = distances[:self.m]
        
        # Add bidirectional links (excluding self-loops)
        m = min(self.m, len(neighbors))
        for neighbor_idx, _ in neighbors[:m]:
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx != idx:  # Avoid self-loops
                self.graph[idx].add(neighbor_idx)
                self.graph[neighbor_idx].add(idx)
            
    def search(self, query: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors."""
        if self.entry_point is None:
            return []
            
        candidates = self._search_layer(query, max(self.ef, k))
        return candidates[:k]
        
    def _search_layer(self, query: np.ndarray, ef: int) -> List[Tuple[int, float]]:
        """Search a single layer of the graph."""
        visited = set()
        candidates = []
        w = []
        
        # Start from entry point
        d = self._distance(query, self.data[self.entry_point])
        heapq.heappush(candidates, (-d, self.entry_point))
        heapq.heappush(w, (d, self.entry_point))
        visited.add(self.entry_point)
        
        while candidates:
            lowerBound, c = heapq.heappop(candidates)
            if -lowerBound > w[0][0]:
                break
                
            # Check neighbors - if no neighbors, check all points
            neighbors = self.graph.get(c, set())
            if not neighbors:
                # For nodes without connections, consider all other unvisited nodes
                neighbors = set(i for i in range(len(self.data)) if self.data[i] is not None) - visited
                
            for neighbor in neighbors:
                neighbor = int(neighbor)  # Ensure neighbor is an integer
                if neighbor not in visited and neighbor < len(self.data) and self.data[neighbor] is not None:
                    visited.add(neighbor)
                    d = self._distance(query, self.data[neighbor])
                    
                    if d < w[0][0] or len(w) < ef:
                        heapq.heappush(candidates, (-d, neighbor))
                        heapq.heappush(w, (d, neighbor))
                        
                        if len(w) > ef:
                            heapq.heappop(w)
                            
        return sorted(w, key=lambda x: x[0])
        
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        if self.metric == "cosine":
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        elif self.metric == "l2":
            return np.linalg.norm(a - b)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
    def serialize(self) -> Dict:
        """Serialize the HNSW index."""
        return {
            "metric": self.metric,
            "m": self.m,
            "ef": self.ef,
            "data": [v.tolist() if v is not None else None for v in self.data],
            "graph": {str(k): [int(x) for x in v] for k, v in self.graph.items()},
            "entry_point": self.entry_point
        }
        
    @classmethod
    def deserialize(cls, data: Dict) -> 'HNSW':
        """Deserialize an HNSW index."""
        hnsw = cls(metric=data["metric"], m=data["m"], ef=data["ef"])
        hnsw.data = [np.array(v) if v is not None else None for v in data["data"]]
        hnsw.graph = defaultdict(set)
        for k, v in data["graph"].items():
            hnsw.graph[int(k)] = set(int(x) for x in v)
        hnsw.entry_point = data["entry_point"]
        return hnsw


class ZarrS3Store(Store):
    """Custom Zarr v3 store implementation for S3."""
    
    def __init__(self, s3_client_factory, bucket_name, prefix, mode: AccessModeLiteral = "r+"):
        self.s3_client_factory = s3_client_factory
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')
        self._cache = {}  # Simple in-memory cache for metadata
        self._mode = mode
        self._is_open = False
    
    async def _open(self) -> None:
        """Open the store."""
        self._is_open = True
    
    async def close(self) -> None:
        """Close the store."""
        self._is_open = False
        self._cache.clear()
    
    @property
    def supports_writes(self) -> bool:
        """Whether this store supports write operations."""
        return self._mode in ("w", "w+", "r+", "w-", "x")
    
    @property
    def supports_deletes(self) -> bool:
        """Whether this store supports delete operations."""
        return self._mode in ("w", "w+", "r+", "w-", "x")
    
    @property
    def supports_partial_writes(self) -> bool:
        """Whether this store supports partial writes."""
        return False  # S3 doesn't support partial writes easily
    
    @property
    def supports_listing(self) -> bool:
        """Whether this store supports listing operations."""
        return True
    
    @property
    def read_only(self) -> bool:
        """Whether this store is read-only."""
        return self._mode == "r"
    
    def _get_key(self, key: str) -> str:
        """Convert Zarr key to S3 key."""
        if key.startswith('/'):
            key = key[1:]
        return f"{self.prefix}/{key}" if self.prefix else key
    
    async def get(self, key: str, prototype=None, byte_range: ByteRequest | None = None):
        """Get item from S3."""
        if prototype is None:
            prototype = default_buffer_prototype()
            
        # Check cache first for small files
        if key in self._cache and byte_range is None:
            return prototype.buffer.from_bytes(self._cache[key])
            
        s3_key = self._get_key(key)
        async with self.s3_client_factory() as s3_client:
            try:
                kwargs = {"Bucket": self.bucket_name, "Key": s3_key}
                
                # Handle byte range requests
                if byte_range is not None:
                    if isinstance(byte_range, tuple):
                        start, end = byte_range
                        if end is not None:
                            kwargs["Range"] = f"bytes={start}-{end-1}"
                        else:
                            kwargs["Range"] = f"bytes={start}-"
                    
                response = await s3_client.get_object(**kwargs)
                data = await response["Body"].read()
                
                # Cache small metadata files
                if len(data) < 10240 and byte_range is None:  # 10KB threshold
                    self._cache[key] = data
                    
                return prototype.buffer.from_bytes(data)
            except Exception:
                return None
    
    async def get_partial_values(self, prototype, key_ranges):
        """Get multiple partial values."""
        results = []
        for key, byte_range in key_ranges:
            result = await self.get(key, prototype, byte_range)
            results.append(result)
        return results
    
    async def set(self, key: str, value, byte_range=None) -> None:
        """Set item to S3."""
        if byte_range is not None:
            raise NotImplementedError("Partial writes not supported for S3")
            
        s3_key = self._get_key(key)
        
        # Convert buffer to bytes
        if hasattr(value, 'to_bytes'):
            data = value.to_bytes()
        else:
            data = bytes(value)
            
        async with self.s3_client_factory() as s3_client:
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=data
            )
            
            # Update cache for small files
            if len(data) < 10240:
                self._cache[key] = data
    
    async def set_partial_values(self, key_start_values):
        """Set multiple partial values - not supported for S3."""
        raise NotImplementedError("Partial writes not supported for S3")
    
    async def delete(self, key: str) -> None:
        """Delete item from S3."""
        s3_key = self._get_key(key)
        async with self.s3_client_factory() as s3_client:
            await s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            self._cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in S3."""
        if key in self._cache:
            return True
            
        s3_key = self._get_key(key)
        async with self.s3_client_factory() as s3_client:
            try:
                await s3_client.head_object(
                    Bucket=self.bucket_name,
                    Key=s3_key
                )
                return True
            except:
                return False
    
    async def list(self):
        """List all keys in the store."""
        async with self.s3_client_factory() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix if self.prefix else None
            ):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if self.prefix:
                        key = key[len(self.prefix)+1:]
                    yield key
    
    async def list_prefix(self, prefix: str):
        """List keys with given prefix."""
        full_prefix = self._get_key(prefix)
        async with self.s3_client_factory() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=full_prefix
            ):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if self.prefix:
                        key = key[len(self.prefix)+1:]
                    yield key
    
    async def list_dir(self, prefix: str):
        """List directory contents."""
        full_prefix = self._get_key(prefix)
        if full_prefix and not full_prefix.endswith('/'):
            full_prefix += '/'
            
        async with self.s3_client_factory() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=full_prefix,
                Delimiter='/'
            ):
                # Yield files
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if self.prefix:
                        key = key[len(self.prefix)+1:]
                    # Remove the prefix to get relative path
                    if key.startswith(prefix):
                        key = key[len(prefix):].lstrip('/')
                    if key:  # Don't yield empty strings
                        yield key
                # Yield directories
                for prefix_info in page.get('CommonPrefixes', []):
                    key = prefix_info['Prefix']
                    if self.prefix:
                        key = key[len(self.prefix)+1:]
                    # Remove the prefix to get relative path
                    if key.startswith(prefix):
                        key = key[len(prefix):].lstrip('/')
                    if key:  # Don't yield empty strings
                        yield key
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another store."""
        if not isinstance(other, ZarrS3Store):
            return False
        return (self.bucket_name == other.bucket_name and 
                self.prefix == other.prefix)


class S3PersistentIndex:
    """Manages SPANN centroids and index in S3 using Zarr."""
    
    def __init__(self, s3_client_factory, bucket_name, collection_id):
        self.s3_client_factory = s3_client_factory
        self.bucket_name = bucket_name
        self.collection_id = collection_id
        self.centroids = None
        self.hnsw_index = None
        self.zarr_store = None
        
    async def initialize(self, dimension: int, num_centroids: int, metric: str = "cosine"):
        """Initialize a new index with random centroids."""
        # Generate initial centroids
        self.centroids = np.random.randn(num_centroids, dimension).astype(np.float32)
        if metric == "cosine":
            self.centroids = normalize(self.centroids)
            
        # Build HNSW index
        self.hnsw_index = HNSW(metric=metric)
        for i, centroid in enumerate(self.centroids):
            self.hnsw_index.add(centroid, i)
            
        # Save to S3 using Zarr
        await self.save()
        
    async def load(self) -> bool:
        """Load centroids and index from S3 using Zarr."""
        try:
            # Create Zarr store
            self.zarr_store = ZarrS3Store(
                self.s3_client_factory,
                self.bucket_name,
                f"{self.collection_id}",
                mode="r"
            )
            await self.zarr_store._open()
            
            # Open existing Zarr v3 array
            z_centroids = await zarr_async.open_array(
                store=self.zarr_store,
                path="centroids_array",
                zarr_format=3
            )
            # Load centroids data asynchronously
            self.centroids = await z_centroids.getitem(...)
            
            # Load HNSW index
            graph_data = await self.zarr_store.get("hnsw_graph.json")
            if graph_data:
                self.hnsw_index = HNSW.deserialize(json.loads(graph_data.to_bytes().decode('utf-8')))
            else:
                raise ValueError("HNSW graph not found")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load centroids from S3: {e}")
            return False
            
    async def save(self):
        """Save centroids and index to S3 using Zarr."""
        # Create Zarr store if not exists
        if not self.zarr_store:
            self.zarr_store = ZarrS3Store(
                self.s3_client_factory,
                self.bucket_name,
                f"{self.collection_id}",
                mode="w+"
            )
            await self.zarr_store._open()
        
        # Create array using Zarr v3 async API with default compression
        # Store array directly at the root of the store with no name
        z_centroids = await zarr_async.create_array(
            store=self.zarr_store,
            name="centroids_array",
            shape=self.centroids.shape,
            chunks=(min(100, len(self.centroids)), self.centroids.shape[1]),
            dtype=self.centroids.dtype,
            zarr_format=3,
            overwrite=True
        )
        
        # Write centroids data asynchronously
        await z_centroids.setitem(..., self.centroids)
        
        # Save HNSW graph as a separate file
        await self.zarr_store.set(
            "hnsw_graph.json",
            json.dumps(self.hnsw_index.serialize()).encode('utf-8')
        )
            
    def find_nearest_centroids(self, vector: np.ndarray, k: int = 5) -> List[int]:
        """Find k nearest centroids for a vector."""
        if self.hnsw_index is None:
            return []
        results = self.hnsw_index.search(vector, k)
        return [idx for _, idx in results]  # Extract indices (second element)
        
    async def update_centroids(self, vectors: np.ndarray):
        """Update centroids using k-means clustering."""
        if len(vectors) < len(self.centroids):
            return  # Not enough vectors to update
            
        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=len(self.centroids),
            batch_size=min(1000, len(vectors)),
            n_init=1
        )
        kmeans.fit(vectors)
        
        # Update centroids
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Rebuild HNSW index
        metric = self.hnsw_index.metric
        self.hnsw_index = HNSW(metric=metric)
        for i, centroid in enumerate(self.centroids):
            self.hnsw_index.add(centroid, i)
            
        # Save updated index
        await self.save()


class ShardManager:
    """Manages vector shards in S3 using Zarr."""
    
    def __init__(self, s3_client_factory, bucket_name, collection_id, shard_size=100000):
        self.s3_client_factory = s3_client_factory
        self.bucket_name = bucket_name
        self.collection_id = collection_id
        self.shard_size = shard_size
        self.shard_metadata = {}
        self.zarr_stores = {}  # Cache Zarr stores for shards
        
    async def load_metadata(self):
        """Load shard metadata from S3."""
        try:
            # Create a Zarr store for metadata
            meta_store = ZarrS3Store(
                self.s3_client_factory,
                self.bucket_name,
                f"{self.collection_id}/shard_metadata",
                mode="r"
            )
            await meta_store._open()
            
            # Try to load shard index
            index_data = await meta_store.get("index.json")
            if index_data:
                shard_index = json.loads(index_data.to_bytes().decode('utf-8'))
                
                # Load each shard's metadata
                for shard_id in shard_index.get('shards', []):
                    try:
                        shard_meta_data = await meta_store.get(f"{shard_id}.json")
                        if shard_meta_data:
                            self.shard_metadata[shard_id] = ShardMetadata(**json.loads(shard_meta_data.to_bytes().decode('utf-8')))
                    except:
                        pass
            
            await meta_store.close()
        except Exception as e:
            logger.debug(f"Failed to load shard metadata: {e}")
            
    async def save_metadata(self):
        """Save shard metadata to S3."""
        meta_store = ZarrS3Store(
            self.s3_client_factory,
            self.bucket_name,
            f"{self.collection_id}/shard_metadata",
            mode="w+"
        )
        await meta_store._open()
        
        # Save shard index
        shard_index = {'shards': list(self.shard_metadata.keys())}
        await meta_store.set("index.json", json.dumps(shard_index).encode('utf-8'))
        
        # Save each shard's metadata
        for shard_id, shard_meta in self.shard_metadata.items():
            await meta_store.set(
                f"{shard_id}.json",
                json.dumps(asdict(shard_meta)).encode('utf-8')
            )
        
        await meta_store.close()
            
    async def get_shard_for_centroid(self, centroid_id: int) -> Optional[str]:
        """Get or create shard for a centroid."""
        # Find existing shard
        for shard_id, metadata in self.shard_metadata.items():
            if metadata.centroid_id == centroid_id and metadata.vector_count < self.shard_size:
                return shard_id
                
        # Create new shard
        shard_id = f"shard_{centroid_id:06d}_{int(time.time())}"
        self.shard_metadata[shard_id] = ShardMetadata(
            shard_id=shard_id,
            centroid_id=centroid_id,
            vector_count=0,
            batch_files=[],
            created_at=time.time(),
            updated_at=time.time()
        )
        return shard_id
        
    async def write_vectors(
        self,
        shard_id: str,
        vectors: np.ndarray,
        metadata: List[Dict],
        ids: List[str]
    ):
        """Write vectors to a shard using Zarr."""
        # Get or create Zarr store for this shard
        if shard_id not in self.zarr_stores:
            self.zarr_stores[shard_id] = ZarrS3Store(
                self.s3_client_factory,
                self.bucket_name,
                f"{self.collection_id}/shards/{shard_id}",
                mode="w+"
            )
            await self.zarr_stores[shard_id]._open()
        
        zarr_store = self.zarr_stores[shard_id]
        shard_meta = self.shard_metadata[shard_id]
        
        # Calculate optimal chunk size based on vector dimension
        dim = vectors.shape[1]
        target_chunk_bytes = 1024 * 1024  # 1MB chunks
        vectors_per_chunk = max(10, min(1000, target_chunk_bytes // (dim * 4)))
        
        # Get current batch number
        batch_num = len(shard_meta.batch_files)
        batch_id = f"batch_{batch_num:06d}"
        
        # Create Zarr v3 array for this batch with default compression
        z_vectors = await zarr_async.create_array(
            store=zarr_store,
            name=f"{batch_id}/vectors",
            shape=vectors.shape,
            chunks=(vectors_per_chunk, dim),
            dtype=vectors.dtype,
            zarr_format=3,
            overwrite=True
        )
        
        # Write vectors asynchronously
        await z_vectors.setitem(..., vectors)
        
        # Save metadata and IDs as JSON
        meta_json = json.dumps({'metadata': metadata, 'ids': ids})
        await zarr_store.set(f"{batch_id}/metadata.json", meta_json.encode('utf-8'))
        
        # Update shard metadata
        shard_meta.batch_files.append(batch_id)
        shard_meta.vector_count += len(vectors)
        shard_meta.updated_at = time.time()
        
        # Save metadata only periodically to reduce S3 operations
        # This improves write performance significantly
        if len(shard_meta.batch_files) % 5 == 0 or shard_meta.vector_count >= self.shard_size:
            await self.save_metadata()
            
    async def load_shard(
        self,
        shard_id: str
    ) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load all vectors from a shard using Zarr."""
        shard_meta = self.shard_metadata.get(shard_id)
        if not shard_meta:
            return np.array([]), [], []
        
        # Get or create Zarr store for this shard
        if shard_id not in self.zarr_stores:
            self.zarr_stores[shard_id] = ZarrS3Store(
                self.s3_client_factory,
                self.bucket_name,
                f"{self.collection_id}/shards/{shard_id}",
                mode="r"
            )
            await self.zarr_stores[shard_id]._open()
        
        zarr_store = self.zarr_stores[shard_id]
        
        all_vectors = []
        all_metadata = []
        all_ids = []
        
        for batch_id in shard_meta.batch_files:
            try:
                # Open Zarr v3 array
                z_vectors = await zarr_async.open_array(
                    store=zarr_store,
                    path=f"{batch_id}/vectors",
                    zarr_format=3
                )
                # Load vectors asynchronously
                batch_vectors = await z_vectors.getitem(...)
                all_vectors.append(batch_vectors)
                
                # Load metadata
                meta_data = await zarr_store.get(f"{batch_id}/metadata.json")
                if meta_data:
                    meta_dict = json.loads(meta_data.to_bytes().decode('utf-8'))
                    all_metadata.extend(meta_dict['metadata'])
                    all_ids.extend(meta_dict['ids'])
                else:
                    raise ValueError(f"Metadata not found for batch {batch_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load batch {batch_id}: {e}")
        
        if all_vectors:
            return np.vstack(all_vectors), all_metadata, all_ids
        else:
            return np.array([]), [], []


class RedisCache:
    """Redis caching layer for vectors and centroids."""
    
    def __init__(self, redis_client, default_ttl=3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        
    async def get_centroids(self, collection_id: str) -> Optional[Tuple[np.ndarray, HNSW]]:
        """Get cached centroids and HNSW index."""
        key = f"cache:centroids:{collection_id}"
        data = await self.redis.get(key)
        if data:
            cached = pickle.loads(data)
            centroids = np.array(cached["centroids"])
            hnsw = HNSW.deserialize(cached["hnsw"])
            return centroids, hnsw
        return None
        
    async def set_centroids(
        self,
        collection_id: str,
        centroids: np.ndarray,
        hnsw: HNSW,
        ttl: Optional[int] = None
    ):
        """Cache centroids and HNSW index."""
        key = f"cache:centroids:{collection_id}"
        data = {
            "centroids": centroids.tolist(),
            "hnsw": hnsw.serialize()
        }
        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            pickle.dumps(data)
        )
        
    async def get_shard(
        self,
        collection_id: str,
        shard_id: str
    ) -> Optional[Tuple[np.ndarray, List[Dict], List[str]]]:
        """Get cached shard data."""
        key = f"cache:shard:{collection_id}:{shard_id}"
        data = await self.redis.get(key)
        if data:
            cached = pickle.loads(data)
            vectors = np.array(cached["vectors"])
            return vectors, cached["metadata"], cached["ids"]
        return None
        
    async def set_shard(
        self,
        collection_id: str,
        shard_id: str,
        vectors: np.ndarray,
        metadata: List[Dict],
        ids: List[str],
        ttl: Optional[int] = None
    ):
        """Cache shard data."""
        key = f"cache:shard:{collection_id}:{shard_id}"
        data = {
            "vectors": vectors.tolist(),
            "metadata": metadata,
            "ids": ids
        }
        await self.redis.setex(
            key,
            ttl or (self.default_ttl // 4),  # Shorter TTL for shard data
            pickle.dumps(data)
        )
        
    async def get_metadata(self, collection_id: str) -> Optional[CollectionMetadata]:
        """Get cached collection metadata."""
        key = f"cache:meta:{collection_id}"
        data = await self.redis.get(key)
        if data:
            return CollectionMetadata(**json.loads(data))
        return None
        
    async def set_metadata(
        self,
        collection_id: str,
        metadata: CollectionMetadata,
        ttl: Optional[int] = None
    ):
        """Cache collection metadata."""
        key = f"cache:meta:{collection_id}"
        await self.redis.setex(
            key,
            ttl or (self.default_ttl // 2),
            json.dumps(asdict(metadata))
        )
        
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


class S3VectorSearchEngine:
    """S3-based vector search engine with SPANN indexing."""

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
        shard_size: int = 100000,
        cache_ttl: int = 3600,
    ):
        """Initialize the S3 vector search engine.
        
        Args:
            endpoint_url: S3 endpoint URL
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region_name: AWS region name
            bucket_name: S3 bucket name for storing vectors
            embedding_model: Name of the fastembed model (e.g., "fastembed:BAAI/bge-small-en-v1.5")
            redis_client: Redis client for caching (optional)
            cache_dir: Directory for caching embedding models
            num_centroids: Number of SPANN centroids
            shard_size: Maximum vectors per shard
            cache_ttl: Redis cache TTL in seconds
        """
        # Check if zarr 3.x is available
        if not ZARR_AVAILABLE:
            import sys
            raise ImportError(
                f"S3VectorSearchEngine requires zarr>=3.1.0, which requires Python>=3.11. "
                f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}. "
                f"Please upgrade to Python 3.11 or later to use S3Vector functionality."
            )
        
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.embedding_model = embedding_model
        self.num_centroids = num_centroids
        self.shard_size = shard_size
        
        # Initialize components
        self.session = aioboto3.Session()
        self.s3_client_factory = self._create_client_factory()
        self.redis_cache = RedisCache(redis_client, cache_ttl) if redis_client else None
        self.indices = {}  # Collection ID -> S3PersistentIndex
        self.shard_managers = {}  # Collection ID -> ShardManager
        
        # Initialize embedding model if specified
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_dim = None
        
        if embedding_model and TextEmbedding:
            try:
                # Handle fastembed model format
                if embedding_model.startswith("fastembed:"):
                    model_name = embedding_model.split(":")[1]
                else:
                    model_name = embedding_model
                
                self.model = TextEmbedding(
                    model_name=model_name,
                    cache_dir=cache_dir
                )
                # Get embedding dimension by encoding a test text
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
            )

        return factory
    
    def calculate_optimal_centroids(self, num_vectors: int) -> int:
        """Calculate optimal number of centroids based on dataset size.
        
        Args:
            num_vectors: Number of vectors in the dataset
            
        Returns:
            Optimal number of centroids
        """
        if num_vectors < 10_000:
            return max(4, int(np.sqrt(num_vectors) / 10))
        elif num_vectors < 1_000_000:
            return int(np.sqrt(num_vectors) / 10)
        else:
            return min(1000, int(np.sqrt(num_vectors) / 100))
    
    def optimize_hnsw_params(self, num_centroids: int) -> Dict[str, int]:
        """Optimize HNSW parameters based on number of centroids.
        
        Args:
            num_centroids: Number of centroids in the index
            
        Returns:
            Dictionary with optimized m and ef parameters
        """
        if num_centroids < 100:
            return {"m": 8, "ef": 50}
        elif num_centroids < 1000:
            return {"m": 16, "ef": 100}
        else:
            return {"m": 32, "ef": 200}

    async def initialize(self):
        """Initialize the search engine and create bucket if needed."""
        async with self.s3_client_factory() as s3_client:
            try:
                await s3_client.head_bucket(Bucket=self.bucket_name)
            except:
                await s3_client.create_bucket(Bucket=self.bucket_name)

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
        
        # Extract embedding_model from kwargs (if provided from artifact controller)
        embedding_model = kwargs.pop("embedding_model", self.embedding_model)
            
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
        num_centroids = kwargs.get("num_centroids", self.num_centroids)
        shard_size = kwargs.get("shard_size", self.shard_size)
        
        # Check if collection exists
        if not overwrite:
            try:
                await self.get_collection_info(collection_id)
                raise ValueError(f"Collection {collection_name} already exists.")
            except VectorLakeError:
                # Collection doesn't exist, proceed
                pass
        
        # Create metadata with embedding model
        metadata_dict = {
            "collection_id": collection_id,
            "dimension": dimension,
            "metric": metric,
            "total_vectors": 0,
            "num_centroids": num_centroids,
            "num_shards": 0,
            "shard_size": shard_size,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
        # Store embedding model in metadata if provided
        if embedding_model:
            metadata_dict["embedding_model"] = embedding_model
        
        metadata = CollectionMetadata(
            collection_id=collection_id,
            dimension=dimension,
            metric=metric,
            total_vectors=0,
            num_centroids=num_centroids,
            num_shards=0,
            shard_size=shard_size,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Save metadata to S3 (including embedding_model if present)
        async with self.s3_client_factory() as s3_client:
            meta_key = f"{collection_id}/metadata.json"
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=meta_key,
                Body=json.dumps(metadata_dict)
            )
        
        # Initialize index
        index = S3PersistentIndex(self.s3_client_factory, self.bucket_name, collection_id)
        await index.initialize(dimension, num_centroids, metric)
        self.indices[collection_id] = index
        
        # Initialize shard manager
        self.shard_managers[collection_id] = ShardManager(
            self.s3_client_factory, self.bucket_name, collection_id, shard_size
        )
        
        # Cache metadata
        if self.redis_cache:
            await self.redis_cache.set_metadata(collection_id, metadata)
        
        return asdict(metadata)

    async def add_vectors(
        self,
        collection_name: str,
        vectors: Union[List[Dict[str, Any]], List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        update: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Add vectors to a collection.
        
        Args:
            collection_name: Collection identifier
            vectors: Can be:
                - List of vector documents (dicts with 'id', 'vector', 'metadata' keys)
                - List of numpy arrays
                - Single numpy array (2D)
            metadata: Optional list of metadata dicts (used when vectors is array)
            ids: Optional list of IDs (used when vectors is array)
            update: If True, update existing vectors (not implemented in S3Vector)
            
        Returns:
            Added ids list
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # Process vector documents and extract components
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
                    if isinstance(vector_value, str) and self.model:
                        # Text to embed using collection's configured model
                        embeddings = list(self.model.embed([vector_value]))
                        vector_array = np.array(embeddings[0], dtype=np.float32)
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
                return []
        
        # Convert to numpy array
        vectors_array = np.array(vector_data, dtype=np.float32)
        
        # Load index
        if collection_id not in self.indices:
            index = S3PersistentIndex(self.s3_client_factory, self.bucket_name, collection_id)
            if not await index.load():
                raise VectorLakeError(f"Collection {collection_id} not found")
            self.indices[collection_id] = index
        else:
            index = self.indices[collection_id]
        
        # Load shard manager
        if collection_id not in self.shard_managers:
            shard_manager = ShardManager(
                self.s3_client_factory, self.bucket_name, collection_id, self.shard_size
            )
            await shard_manager.load_metadata()
            self.shard_managers[collection_id] = shard_manager
        else:
            shard_manager = self.shard_managers[collection_id]
        
        # Assign vectors to shards based on nearest centroid
        shard_assignments = defaultdict(list)
        for i, vector in enumerate(vectors_array):
            nearest_centroids = index.find_nearest_centroids(vector, k=1)
            if nearest_centroids:
                centroid_id = nearest_centroids[0]
                shard_id = await shard_manager.get_shard_for_centroid(centroid_id)
                shard_assignments[shard_id].append((i, vector, metadata_list[i], ids_list[i]))
        
        # Write vectors to shards
        for shard_id, assignments in shard_assignments.items():
            indices, shard_vectors, shard_metadata, shard_ids = zip(*assignments)
            await shard_manager.write_vectors(
                shard_id,
                np.array(shard_vectors),
                list(shard_metadata),
                list(shard_ids)
            )
        
        # Invalidate cache
        if self.redis_cache:
            await self.redis_cache.invalidate_collection(collection_id)
        
        return ids_list

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: Optional[Union[List[float], np.ndarray, str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        include_vectors: bool = False,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for similar vectors in a collection.
        
        Args:
            collection_name: Collection identifier
            query_vector: Query vector, can be list, numpy array, or text string
            filters: Optional metadata filter
            limit: Number of results to return
            offset: Number of results to skip (for pagination)
            include_vectors: Whether to include vectors in results
            order_by: Field to sort by (if no query_vector)
            pagination: If True, return dict with items and pagination info
            
        Returns:
            List of search results with scores and metadata, or dict with pagination if pagination=True
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # Handle query vector
        if query_vector is not None:
            # Handle text query vector
            if isinstance(query_vector, str):
                if self.model:
                    # Use the collection's configured embedding model
                    embeddings = list(self.model.embed([query_vector]))
                    query_vector = embeddings[0]
                else:
                    raise ValueError("No embedding model configured for text query")
            
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector, dtype=np.float32)
        
        # Load index with caching
        if collection_id not in self.indices:
            index = S3PersistentIndex(self.s3_client_factory, self.bucket_name, collection_id)
            
            # Try cache first
            if self.redis_cache:
                cached = await self.redis_cache.get_centroids(collection_id)
                if cached:
                    index.centroids, index.hnsw_index = cached
                elif not await index.load():
                    raise VectorLakeError(f"Collection {collection_id} not found")
                else:
                    await self.redis_cache.set_centroids(
                        collection_id, index.centroids, index.hnsw_index
                    )
            elif not await index.load():
                raise VectorLakeError(f"Collection {collection_id} not found")
            
            self.indices[collection_id] = index
        else:
            index = self.indices[collection_id]
        
        # Load shard manager
        if collection_id not in self.shard_managers:
            shard_manager = ShardManager(
                self.s3_client_factory, self.bucket_name, collection_id, self.shard_size
            )
            await shard_manager.load_metadata()
            self.shard_managers[collection_id] = shard_manager
        else:
            shard_manager = self.shard_managers[collection_id]
        
        # Determine which shards to search
        if query_vector is not None:
            # Find nearest centroids - adaptive based on limit
            # For small limits, search fewer shards; for larger limits, search more
            if limit <= 10:
                num_shards_to_search = min(2, len(index.centroids))
            elif limit <= 50:
                num_shards_to_search = min(5, len(index.centroids))
            elif limit <= 200:
                num_shards_to_search = min(10, len(index.centroids))
            else:
                num_shards_to_search = min(20, len(index.centroids))
            
            nearest_centroids = index.find_nearest_centroids(query_vector, k=num_shards_to_search)
        else:
            # For filter-only search, we need to search all shards
            nearest_centroids = list(range(len(index.centroids)))
        
        # Load and search relevant shards with early termination
        all_results = []
        
        # Process centroids in order of proximity for early termination (or all for filter-only)
        for centroid_id in nearest_centroids:
            # Find shards for this centroid
            relevant_shards = [
                shard_id for shard_id, meta in shard_manager.shard_metadata.items()
                if meta.centroid_id == centroid_id
            ]
            
            # Parallel shard loading for this centroid
            async def load_shard_with_cache(shard_id):
                """Load a shard with cache support."""
                # Try cache first
                if self.redis_cache:
                    cached = await self.redis_cache.get_shard(collection_id, shard_id)
                    if cached:
                        return cached
                    else:
                        result = await shard_manager.load_shard(shard_id)
                        if result[0] is not None and len(result[0]) > 0:
                            await self.redis_cache.set_shard(
                                collection_id, shard_id, result[0], result[1], result[2]
                            )
                        return result
                else:
                    return await shard_manager.load_shard(shard_id)
            
            # Load shards for this centroid in parallel
            shard_results = await asyncio.gather(
                *[load_shard_with_cache(shard_id) for shard_id in relevant_shards],
                return_exceptions=True
            )
            
            # Process shards from this centroid
            centroid_results = []
            for shard_result in shard_results:
                if isinstance(shard_result, Exception):
                    continue
                
                shard_vectors, shard_metadata, shard_ids = shard_result
                if shard_vectors is None or len(shard_vectors) == 0:
                    continue
                
                # Calculate similarities if we have a query vector
                if query_vector is not None:
                    if index.hnsw_index.metric == "cosine":
                        query_norm = query_vector / np.linalg.norm(query_vector)
                        shard_norms = shard_vectors / np.linalg.norm(shard_vectors, axis=1, keepdims=True)
                        similarities = np.dot(shard_norms, query_norm)
                    else:
                        distances = np.linalg.norm(shard_vectors - query_vector, axis=1)
                        similarities = 1 / (1 + distances)
                else:
                    # For filter-only search, assign equal scores
                    similarities = np.ones(len(shard_vectors))
                
                # Apply metadata filter if provided
                if filters:
                    filtered_indices = []
                    for i, meta in enumerate(shard_metadata):
                        if all(meta.get(k) == v for k, v in filters.items()):
                            filtered_indices.append(i)
                    
                    if filtered_indices:
                        similarities = similarities[filtered_indices]
                        shard_metadata = [shard_metadata[i] for i in filtered_indices]
                        shard_ids = [shard_ids[i] for i in filtered_indices]
                        shard_vectors = shard_vectors[filtered_indices]
                    else:
                        # No matches in this shard, skip it
                        continue
                
                # Add results from this shard
                for i, score in enumerate(similarities):
                    result_item = {
                        "id": shard_ids[i],
                        "_score": float(score),  # Use _score to match PgVector
                    }
                    # Add metadata fields directly (flatten metadata)
                    result_item.update(shard_metadata[i])
                    # Optionally include vectors
                    if include_vectors:
                        result_item["vector"] = shard_vectors[i].tolist()
                    centroid_results.append(result_item)
            
            # Add centroid results to all results
            all_results.extend(centroid_results)
            
            # Early termination: if we have significantly more results than needed, 
            # and this isn't the first centroid, we can stop
            if len(all_results) >= limit * 3 and centroid_id != nearest_centroids[0]:
                break
        
        # Sort by score and apply offset and limit
        all_results.sort(key=lambda x: x["_score"], reverse=True)
        start_idx = offset
        end_idx = offset + limit
        results = all_results[start_idx:end_idx]
        
        # Return with pagination info if requested
        if pagination:
            return {
                "items": results,
                "total": len(all_results),
                "offset": offset,
                "limit": limit,
            }
        else:
            return results

    async def get_collection_info(
        self, collection_name: str
    ) -> Dict[str, Any]:
        """Get information about a collection.
        
        Args:
            collection_name: Collection identifier
            
        Returns:
            Collection metadata
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # Try cache first
        if self.redis_cache:
            cached = await self.redis_cache.get_metadata(collection_id)
            if cached:
                return asdict(cached)
        
        # Load from S3
        async with self.s3_client_factory() as s3_client:
            meta_key = f"{collection_id}/metadata.json"
            try:
                response = await s3_client.get_object(
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

    async def list_collections(self) -> List[str]:
        """List all collections.
        
        Returns:
            List of collection IDs
        """
        collections = []
        async with self.s3_client_factory() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Delimiter='/'):
                for prefix_info in page.get('CommonPrefixes', []):
                    collection_id = prefix_info['Prefix'].rstrip('/')
                    collections.append(collection_id)
        return collections

    async def remove_vectors(self, collection_name: str, ids: List[str], **kwargs):
        """Remove vectors from a collection.
        
        Args:
            collection_name: Collection identifier
            ids: List of vector IDs to remove
            
        Returns:
            Number of vectors removed
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to find and remove vectors from the appropriate shards
        logger.warning(f"remove_vectors not fully implemented for S3Vector engine for collection {collection_name}")
        return len(ids)

    async def get_vector(self, collection_name: str, id: str) -> Dict[str, Any]:
        """Get a single vector from a collection.
        
        Args:
            collection_name: Collection identifier
            id: Vector ID to retrieve
            
        Returns:
            Vector data with metadata
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to search through shards to find the vector
        logger.warning(f"get_vector not fully implemented for S3Vector engine for collection {collection_name}")
        raise NotImplementedError("get_vector not fully implemented for S3Vector engine")

    async def list_vectors(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 10,
        include_vectors: bool = False,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """List vectors in a collection.
        
        Args:
            collection_name: Collection identifier
            offset: Number of results to skip
            limit: Maximum number of results
            include_vectors: Whether to include vector arrays in results
            order_by: Field to sort by
            pagination: If True, return dict with pagination info
            
        Returns:
            List of vectors or paginated results
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # This is a simplified implementation - in a full implementation,
        # we would need to iterate through all shards and collect vectors
        logger.warning(f"list_vectors not fully implemented for S3Vector engine for collection {collection_name}")
        
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
        """Count vectors in a collection.
        
        Args:
            collection_name: Collection identifier
            
        Returns:
            Number of vectors in the collection
        """
        try:
            info = await self.get_collection_info(collection_name)
            return info.get("total_vectors", 0)
        except:
            return 0

    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection and all its data.
        
        Args:
            collection_name: Collection identifier
            **kwargs: Additional keyword arguments for compatibility
        """
        # Handle backward compatibility - use collection_name as collection_id
        collection_id = collection_name
        
        # Delete from S3
        async with self.s3_client_factory() as s3_client:
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=f"{collection_id}/"):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects:
                        await s3_client.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={'Objects': objects}
                        )
        
        # Clear caches
        if self.redis_cache:
            await self.redis_cache.invalidate_collection(collection_id)
        
        # Remove from memory
        if collection_id in self.indices:
            del self.indices[collection_id]
        if collection_id in self.shard_managers:
            del self.shard_managers[collection_id]

    async def _embed_texts(
        self,
        texts: List[str],
        embedding_model: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Generate embeddings for texts using fastembed.
        
        Args:
            texts: List of text strings to embed
            embedding_model: Optional model override (uses instance model if not specified)
            
        Returns:
            List of embedding vectors as numpy arrays
        """
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
        shard_size=config.get("shard_size", 100000),
        cache_ttl=config.get("cache_ttl", 3600),
    )


async def create_s3_vector_engine(store: RedisStore, config: Dict[str, Any]) -> S3VectorSearchEngine:
    """Create an S3 vector search engine with Redis caching from store."""
    config = config.copy()
    config["redis_client"] = store.get_redis() if store else None
    engine = create_s3_vector_engine_from_config(config)
    await engine.initialize()
    return engine


# Check if S3Vector can be used at import time
if not ZARR_AVAILABLE:
    import sys
    
    class S3VectorSearchEngine:
        """Placeholder class that raises ImportError when zarr is not available."""
        def __new__(cls, *args, **kwargs):
            raise ImportError(
                f"S3VectorSearchEngine requires zarr>=3.1.0, which requires Python>=3.11. "
                f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}. "
                f"Please upgrade to Python 3.11 or later to use S3Vector functionality."
            )
        
        def __init__(self, *args, **kwargs):
            # This should never be reached
            pass
    
    # Also override the factory functions
    def create_s3_vector_engine(*args, **kwargs):
        raise ImportError(
            f"S3VectorSearchEngine requires zarr>=3.1.0, which requires Python>=3.11. "
            f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}. "
            f"Please upgrade to Python 3.11 or later to use S3Vector functionality."
        )
    
    def create_s3_vector_engine_from_config(*args, **kwargs):
        raise ImportError(
            f"S3VectorSearchEngine requires zarr>=3.1.0, which requires Python>=3.11. "
            f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}. "
            f"Please upgrade to Python 3.11 or later to use S3Vector functionality."
        )