"""S3-based vector search engine with SPANN indexing for Hypha.

Implements a Turbopuffer-inspired architecture with:
- SPANN (Space Partition Approximate Nearest Neighbor) indexing
- S3 for persistent storage
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
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import aioboto3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


class S3PersistentIndex:
    """Manages SPANN centroids and index in S3."""
    
    def __init__(self, s3_client_factory, bucket_name, collection_id):
        self.s3_client_factory = s3_client_factory
        self.bucket_name = bucket_name
        self.collection_id = collection_id
        self.centroids = None
        self.hnsw_index = None
        
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
            
        # Save to S3
        await self.save()
        
    async def load(self) -> bool:
        """Load centroids and index from S3."""
        try:
            async with self.s3_client_factory() as s3_client:
                # Load centroids
                centroid_key = f"{self.collection_id}/centroids/centroids.parquet"
                response = await s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=centroid_key
                )
                centroid_data = await response["Body"].read()
                table = pq.read_table(io.BytesIO(centroid_data))
                self.centroids = np.vstack([np.array(row) for row in table.column("vector").to_pylist()])
                
                # Load HNSW graph
                graph_key = f"{self.collection_id}/centroids/centroid_graph.json"
                response = await s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=graph_key
                )
                graph_data = await response["Body"].read()
                self.hnsw_index = HNSW.deserialize(json.loads(graph_data))
                
                return True
        except Exception as e:
            logger.debug(f"Failed to load centroids from S3: {e}")
            return False
            
    async def save(self):
        """Save centroids and index to S3."""
        async with self.s3_client_factory() as s3_client:
            # Save centroids as Parquet
            table = pa.table({
                "centroid_id": list(range(len(self.centroids))),
                "vector": self.centroids.tolist()
            })
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            
            centroid_key = f"{self.collection_id}/centroids/centroids.parquet"
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=centroid_key,
                Body=buffer.getvalue()
            )
            
            # Save HNSW graph as JSON
            graph_key = f"{self.collection_id}/centroids/centroid_graph.json"
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=graph_key,
                Body=json.dumps(self.hnsw_index.serialize())
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
    """Manages vector shards in S3."""
    
    def __init__(self, s3_client_factory, bucket_name, collection_id, shard_size=100000):
        self.s3_client_factory = s3_client_factory
        self.bucket_name = bucket_name
        self.collection_id = collection_id
        self.shard_size = shard_size
        self.shard_metadata = {}
        
    async def load_metadata(self):
        """Load shard metadata from S3."""
        try:
            async with self.s3_client_factory() as s3_client:
                # List all shard directories
                paginator = s3_client.get_paginator('list_objects_v2')
                prefix = f"{self.collection_id}/shards/"
                
                async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter='/'):
                    for prefix_info in page.get('CommonPrefixes', []):
                        shard_dir = prefix_info['Prefix']
                        shard_id = shard_dir.split('/')[-2]
                        
                        # Load shard metadata
                        meta_key = f"{shard_dir}metadata.json"
                        try:
                            response = await s3_client.get_object(
                                Bucket=self.bucket_name,
                                Key=meta_key
                            )
                            meta_data = await response["Body"].read()
                            self.shard_metadata[shard_id] = ShardMetadata(**json.loads(meta_data))
                        except:
                            pass  # Shard metadata doesn't exist yet
        except Exception as e:
            logger.debug(f"Failed to load shard metadata: {e}")
            
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
        """Write vectors to a shard."""
        async with self.s3_client_factory() as s3_client:
            # Create batch filename
            batch_id = f"vectors_{int(time.time() * 1000)}"
            batch_key = f"{self.collection_id}/shards/{shard_id}/{batch_id}.parquet"
            
            # Create Parquet table
            table = pa.table({
                "id": ids,
                "vector": vectors.tolist(),
                "metadata": [json.dumps(m) for m in metadata]
            })
            
            # Write to buffer
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            
            # Upload to S3
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=batch_key,
                Body=buffer.getvalue()
            )
            
            # Update metadata
            shard_meta = self.shard_metadata[shard_id]
            shard_meta.batch_files.append(batch_id)
            shard_meta.vector_count += len(vectors)
            shard_meta.updated_at = time.time()
            
            # Save metadata
            meta_key = f"{self.collection_id}/shards/{shard_id}/metadata.json"
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=meta_key,
                Body=json.dumps(asdict(shard_meta))
            )
            
    async def load_shard(
        self,
        shard_id: str
    ) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """Load all vectors from a shard."""
        vectors = []
        metadata = []
        ids = []
        
        shard_meta = self.shard_metadata.get(shard_id)
        if not shard_meta:
            return np.array([]), [], []
            
        async with self.s3_client_factory() as s3_client:
            for batch_file in shard_meta.batch_files:
                batch_key = f"{self.collection_id}/shards/{shard_id}/{batch_file}.parquet"
                
                try:
                    response = await s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=batch_key
                    )
                    batch_data = await response["Body"].read()
                    table = pq.read_table(io.BytesIO(batch_data))
                    
                    vectors.extend([np.array(v) for v in table.column("vector").to_pylist()])
                    metadata.extend([json.loads(m) for m in table.column("metadata").to_pylist()])
                    ids.extend(table.column("id").to_pylist())
                except Exception as e:
                    logger.warning(f"Failed to load batch {batch_file}: {e}")
                    
        return np.array(vectors) if vectors else np.array([]), metadata, ids


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
            updated_at=time.time()
        )
        
        # Save metadata to S3
        async with self.s3_client_factory() as s3_client:
            meta_key = f"{collection_id}/metadata.json"
            await s3_client.put_object(
                Bucket=self.bucket_name,
                Key=meta_key,
                Body=json.dumps(asdict(metadata))
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
        embedding_model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Add vectors to a collection.
        
        Args:
            collection_name: Collection identifier (renamed from collection_id for consistency)
            vectors: Can be:
                - List of vector documents (dicts with 'id', 'vector', 'metadata' keys)
                - List of numpy arrays
                - Single numpy array (2D)
            metadata: Optional list of metadata dicts (used when vectors is array)
            ids: Optional list of IDs (used when vectors is array)
            update: If True, update existing vectors (not implemented in S3Vector)
            embedding_model: Model to use for text embeddings
            
        Returns:
            Dictionary with 'added' count and 'ids' list
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
        """Search for similar vectors in a collection.
        
        Args:
            collection_name: Collection identifier (renamed from collection_id for consistency)
            query_vector: Query vector, can be list, numpy array, or text string
            embedding_model: Model to use for text embeddings if query_vector is text
            filters: Optional metadata filter (renamed from filter_metadata for consistency)
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
        
        # Get query vector
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
        
        # Load and search relevant shards with early termination
        all_results = []
        
        # Process centroids in order of proximity for early termination
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
                
                # Calculate similarities
                if index.hnsw_index.metric == "cosine":
                    query_norm = query_vector / np.linalg.norm(query_vector)
                    shard_norms = shard_vectors / np.linalg.norm(shard_vectors, axis=1, keepdims=True)
                    similarities = np.dot(shard_norms, query_norm)
                else:
                    distances = np.linalg.norm(shard_vectors - query_vector, axis=1)
                    similarities = 1 / (1 + distances)
                
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
            collection_name: Collection identifier (renamed from collection_id for consistency)
            
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
            collection_name: Collection identifier (renamed from collection_id for consistency)
            
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
            collection_name: Collection identifier (renamed from collection_id for consistency)
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