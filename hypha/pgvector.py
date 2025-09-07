"""PostgreSQL-based vector search engine using pgvector extension with partition-by-dimension approach."""

import asyncio
import numpy as np
import uuid
import logging
import os
import sys
import json
import hashlib
from typing import Any, List, Dict, Optional, Union
from sqlalchemy import (
    Column,
    String,
    Integer,
    BigInteger,
    Float,
    Text,
    JSON as SQLJSON,
    select,
    text,
    and_,
    or_,
    create_engine,
    MetaData,
    Table,
    Index,
    func,
    delete,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import expression
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import UserDefinedType
from sqlalchemy.schema import CreateTable, DropTable
from sqlalchemy.exc import ProgrammingError, IntegrityError

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("pgvector")
logger.setLevel(LOGLEVEL)


class Vector(UserDefinedType):
    """pgvector type for SQLAlchemy."""
    
    cache_ok = True
    
    def __init__(self, dim=None):
        self.dim = dim
    
    def get_col_spec(self, **kw):
        if self.dim is None:
            return "VECTOR"
        return f"VECTOR({self.dim})"
    
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, np.ndarray):
                return f"[{','.join(str(float(x)) for x in value.flatten())}]"
            elif isinstance(value, (list, tuple)):
                return f"[{','.join(str(float(x)) for x in value)}]"
            elif isinstance(value, bytes):
                # Convert bytes to numpy array then to string
                arr = np.frombuffer(value, dtype=np.float32)
                return f"[{','.join(str(float(x)) for x in arr)}]"
            return value
        return process
    
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            # Parse the vector string format [1.0,2.0,3.0]
            if isinstance(value, str):
                value = value.strip('[]')
                return np.array([float(x) for x in value.split(',')], dtype=np.float32)
            return value
        return process


class PgVectorSearchEngine:
    """PostgreSQL-based vector search engine using pgvector extension with partition-by-dimension approach."""
    
    def __init__(self, engine, prefix: str = "vec", cache_dir=None):
        """Initialize PgVector search engine.
        
        Args:
            engine: SQLAlchemy async engine
            prefix: Schema prefix (default: "vec")
            cache_dir: Directory for caching models
        """
        self._engine = engine
        self._prefix = prefix
        self._cache_dir = cache_dir
        self._sessionmaker = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        self._initialized = False
        self._created_parent_tables = set()  # Track which parent tables we've created
    
    def _calculate_ivfflat_parameters(self, row_count: int) -> tuple[int, int]:
        """
        Calculate optimal IVFFlat parameters based on dataset size.
        
        Args:
            row_count: Number of rows in the dataset
            
        Returns:
            Tuple of (lists, probes) parameters
        """
        import math
        
        if row_count <= 100:
            # For very small datasets, use fewer lists to avoid empty clusters
            lists = max(1, row_count // 10)
            lists = min(lists, 10)  # Cap at 10 for small datasets
        elif row_count < 1_000_000:
            # For datasets < 1M rows: lists = rows / 1000, minimum 10
            lists = max(10, row_count // 1000)
        else:
            # For datasets >= 1M rows: lists = sqrt(rows)
            lists = int(math.sqrt(row_count))
        
        # Set probes = sqrt(lists) for optimal recall/speed balance
        probes = max(1, int(math.sqrt(lists)))
        
        return lists, probes
    
    async def _ensure_initialized(self):
        """Ensure the pgvector schema and infrastructure is initialized."""
        if self._initialized:
            return
        
        async with self._engine.begin() as conn:
            # Check if pgvector extension is installed
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except ProgrammingError as e:
                logger.warning(f"Could not create pgvector extension: {e}")
            
            # Create the schema if it doesn't exist
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._prefix}"))
            
            # Create collection registry table (without dimension constraints)
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self._prefix}.collection_registry (
                    collection_id BIGINT PRIMARY KEY,
                    collection_name TEXT UNIQUE NOT NULL,
                    dim INT NOT NULL CHECK (dim > 0 AND dim <= 2048),
                    vector_fields JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
        
        self._initialized = True
    
    def _get_collection_id(self, collection_name: str) -> int:
        """Get a deterministic collection ID from the collection name."""
        # Use hash to get a consistent numeric ID from collection name
        return abs(hash(collection_name)) % (2**31)
    
    def _get_parent_table(self, dim: int) -> str:
        """Get the parent table name for a given dimension."""
        return f"{self._prefix}.emb_{dim}_parent"
    
    def _get_child_table(self, dim: int, collection_id: int) -> str:
        """Get the child table name for a collection."""
        return f"{self._prefix}.emb_{dim}_parent_c{collection_id}"
    
    async def _ensure_parent_table_exists(self, dim: int):
        """Ensure parent table exists for the given dimension."""
        if dim in self._created_parent_tables:
            return  # Already created
        
        parent_table = self._get_parent_table(dim)
        
        async with self._engine.begin() as conn:
            # Check if parent table already exists
            result = await conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{self._prefix}' 
                    AND table_name = 'emb_{dim}_parent'
                )
            """))
            exists = result.scalar()
            
            if not exists:
                # Create parent table for this dimension
                await conn.execute(text(f"""
                    CREATE TABLE {parent_table} (
                        collection_id BIGINT NOT NULL,
                        id TEXT NOT NULL,
                        vector vector({dim}) NOT NULL,
                        manifest JSONB,
                        created_at TIMESTAMPTZ DEFAULT now(),
                        PRIMARY KEY (collection_id, id)
                    ) PARTITION BY LIST (collection_id)
                """))
                logger.info(f"Created parent table for dimension {dim}: {parent_table}")
        
        self._created_parent_tables.add(dim)
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int = 384,
        distance_metric: str = "cosine",
        overwrite: bool = False,
        **kwargs
    ):
        """Create a vector collection using partition-by-dimension approach.
        
        Args:
            collection_name: Name of the collection
            dimension: Dimension of the vectors (default: 384)
            distance_metric: Distance metric to use - 'cosine', 'l2', or 'ip' (default: 'cosine')
            overwrite: Whether to overwrite existing collection
        """
        await self._ensure_initialized()
        
        # Validate dimension is reasonable
        if dimension <= 0 or dimension > 2048:
            raise ValueError(f"Dimension {dimension} must be between 1 and 2048")
        
        # Validate distance metric
        if distance_metric.lower() not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Distance metric must be 'cosine', 'l2', or 'ip', got {distance_metric}")
        
        # Ensure parent table exists for this dimension
        await self._ensure_parent_table_exists(dimension)
        
        collection_id = self._get_collection_id(collection_name)
        
        async with self._engine.begin() as conn:
            # Check if collection exists
            result = await conn.execute(text(f"""
                SELECT collection_id FROM {self._prefix}.collection_registry 
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            existing_collection = result.fetchone()
            
            if existing_collection:
                if overwrite:
                    await self.delete_collection(collection_name)
                else:
                    raise ValueError(f"Collection {collection_name} already exists.")
            
            # Store collection metadata with simplified config
            metadata = {
                "dimension": dimension,
                "distance_metric": distance_metric.lower()
            }
            
            # Insert into registry
            await conn.execute(text(f"""
                INSERT INTO {self._prefix}.collection_registry 
                (collection_id, collection_name, dim, vector_fields)
                VALUES (:collection_id, :collection_name, :dim, :metadata)
                ON CONFLICT (collection_id) DO UPDATE SET 
                    collection_name = EXCLUDED.collection_name,
                    dim = EXCLUDED.dim,
                    vector_fields = EXCLUDED.vector_fields
            """), {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "dim": dimension,
                "metadata": json.dumps(metadata)
            })
            
            # Create child partition
            parent_table = self._get_parent_table(dimension)
            child_table = self._get_child_table(dimension, collection_id)
            child_table_simple = child_table.split('.')[-1]  # Get table name without schema
            
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {child_table}
                PARTITION OF {parent_table}
                FOR VALUES IN ({collection_id})
            """))
            
            # Create vector index with optimal parameters
            ops_map = {
                "cosine": "vector_cosine_ops",
                "l2": "vector_l2_ops", 
                "ip": "vector_ip_ops"
            }
            ops = ops_map.get(distance_metric.lower(), "vector_cosine_ops")
            
            # Start with minimal lists for new collection
            lists = 1
            index_name = f"{child_table_simple}_hnsw"
            
            try:
                await conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {child_table} USING ivfflat (vector {ops})
                    WITH (lists = {lists})
                """))
                logger.info(f"Created vector index {index_name} for collection {collection_name} with lists={lists}")
            except ProgrammingError:
                # Fallback to HNSW if available
                try:
                    await conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {child_table} USING hnsw (vector {ops})
                    """))
                    logger.info(f"Created HNSW index {index_name} for collection {collection_name}")
                except ProgrammingError:
                    logger.warning(f"Could not create vector index for {collection_name}")
            
            # Create metadata indexes
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {child_table_simple}_cid 
                ON {child_table} (collection_id)
            """))
            await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {child_table_simple}_created_at 
                ON {child_table} (created_at DESC)
            """))
        
        logger.info(f"Collection {collection_name} created successfully")
    
    async def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection and its partition."""
        await self._ensure_initialized()
        
        async with self._engine.begin() as conn:
            # Get collection info
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                logger.warning(f"Collection {collection_name} not found")
                return
            
            collection_id, dim = collection_info
            
            # Drop the partition
            parent_table = self._get_parent_table(dim)
            child_table = self._get_child_table(dim, collection_id)
            
            try:
                await conn.execute(text(f"ALTER TABLE {parent_table} DETACH PARTITION {child_table}"))
                await conn.execute(text(f"DROP TABLE IF EXISTS {child_table}"))
            except ProgrammingError as e:
                logger.warning(f"Could not drop partition {child_table}: {e}")
            
            # Remove from registry
            await conn.execute(text(f"""
                DELETE FROM {self._prefix}.collection_registry 
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
        
        logger.info(f"Collection {collection_name} deleted successfully")
    
    async def list_collections(self) -> List[Dict[str, str]]:
        """List all collections."""
        await self._ensure_initialized()
        
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_name FROM {self._prefix}.collection_registry
                ORDER BY collection_name
            """))
            rows = result.fetchall()
        
        return [{"name": row[0]} for row in rows]
    
    async def count(self, collection_name: str) -> int:
        """Count vectors in a collection."""
        await self._ensure_initialized()
        
        async with self._engine.connect() as conn:
            # Get collection info
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                return 0
            
            collection_id, dim = collection_info
            parent_table = self._get_parent_table(dim)
            
            # Count rows in the partition
            count_result = await conn.execute(text(f"""
                SELECT COUNT(*) FROM {parent_table} 
                WHERE collection_id = :collection_id
            """), {"collection_id": collection_id})
            
            return count_result.scalar() or 0
    
    async def _get_metadata(self, collection_name: str) -> Dict[str, Any]:
        """Get metadata for a collection."""
        await self._ensure_initialized()
        
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT vector_fields, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Collection {collection_name} not found")
            
            metadata_json, dim = row
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            else:
                metadata = metadata_json
            
            # Ensure backward compatibility
            if "dimension" not in metadata:
                metadata["dimension"] = dim
            if "distance_metric" not in metadata:
                metadata["distance_metric"] = "cosine"
            
            return metadata
    
    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]],
        update: bool = False,
        embedding_model: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """Add vectors to a collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vector documents, each containing:
                - id: Unique identifier (optional, auto-generated if not provided)
                - vector: Vector data as list/array/bytes or text to embed
                - manifest: Dictionary of additional metadata (optional)
            update: If True, update existing vectors
            embedding_model: Model to use for text embeddings (e.g., "fastembed:BAAI/bge-small-en-v1.5")
        """
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim = collection_info
        
        parent_table = self._get_parent_table(dim)
        
        ids = []
        
        async with self._sessionmaker() as session:
            for vector_data in vectors:
                if "id" in vector_data:
                    vector_id = str(vector_data["id"])
                else:
                    if update:
                        raise ValueError("ID is required for update operation.")
                    vector_id = str(uuid.uuid4())
                
                ids.append(vector_id)
                
                # Process vector field
                vector_value = vector_data.get("vector")
                if vector_value is None:
                    raise ValueError("'vector' field is required")
                
                # Convert vector to string format
                if isinstance(vector_value, np.ndarray):
                    vec_list = vector_value.astype("float32").tolist()
                    vector_str = f"[{','.join(map(str, vec_list))}]"
                elif isinstance(vector_value, (list, tuple)):
                    vector_str = f"[{','.join(map(str, vector_value))}]"
                elif isinstance(vector_value, bytes):
                    arr = np.frombuffer(vector_value, dtype=np.float32)
                    vec_list = arr.tolist()
                    vector_str = f"[{','.join(map(str, vec_list))}]"
                elif isinstance(vector_value, str) and embedding_model:
                    # Generate embedding from text
                    embeddings = await self._embed_texts([vector_value], embedding_model)
                    vec_list = embeddings[0] if isinstance(embeddings[0], list) else embeddings[0].tolist()
                    vector_str = f"[{','.join(map(str, vec_list))}]"
                else:
                    if isinstance(vector_value, list):
                        vector_str = f"[{','.join(map(str, vector_value))}]"
                    else:
                        raise ValueError(f"Invalid vector type: {type(vector_value)}")
                
                # Get manifest directly from the input (no longer extract from flat keys)
                manifest = vector_data.get("manifest", {})
                
                if update:
                    # Update existing vector
                    await session.execute(text(f"""
                        UPDATE {parent_table} 
                        SET vector = CAST(:vector AS vector), manifest = :manifest
                        WHERE collection_id = :collection_id AND id = :id
                    """), {
                        "collection_id": collection_id,
                        "id": vector_id,
                        "vector": vector_str,
                        "manifest": json.dumps(manifest) if manifest else None
                    })
                else:
                    # Insert new vector
                    try:
                        await session.execute(text(f"""
                            INSERT INTO {parent_table} (collection_id, id, vector, manifest)
                            VALUES (:collection_id, :id, CAST(:vector AS vector), :manifest)
                        """), {
                            "collection_id": collection_id,
                            "id": vector_id,
                            "vector": vector_str,
                            "manifest": json.dumps(manifest) if manifest else None
                        })
                    except IntegrityError:
                        logger.warning(f"Vector with ID {vector_id} already exists, skipping")
            
            await session.commit()
        
        logger.info(f"Added {len(vectors)} vectors to collection {collection_name}")
        return ids
    
    async def remove_vectors(self, collection_name: str, ids: List[str], **kwargs):
        """Remove vectors from a collection."""
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim = collection_info
        
        parent_table = self._get_parent_table(dim)
        
        async with self._sessionmaker() as session:
            for vector_id in ids:
                await session.execute(text(f"""
                    DELETE FROM {parent_table} 
                    WHERE collection_id = :collection_id AND id = :id
                """), {"collection_id": collection_id, "id": str(vector_id)})
            await session.commit()
        
        logger.info(f"Removed {len(ids)} vectors from collection {collection_name}")
    
    async def get_vector(self, collection_name: str, id: str) -> Dict[str, Any]:
        """Get a single vector by ID."""
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim = collection_info
        
        parent_table = self._get_parent_table(dim)
        
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT id, vector, manifest FROM {parent_table}
                WHERE collection_id = :collection_id AND id = :id
            """), {"collection_id": collection_id, "id": str(id)})
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Vector {id} not found in collection {collection_name}")
            
            vector_id, vector_col, manifest = row
            
            # Build result
            vector_data = {"id": vector_id}
            
            # Add vector field
            if vector_col is not None:
                if isinstance(vector_col, np.ndarray):
                    vector_data["vector"] = vector_col.tolist()
                elif isinstance(vector_col, str):
                    # Parse pgvector string format
                    vector_col = vector_col.strip('[]')
                    if vector_col:
                        vector_data["vector"] = [float(x.strip()) for x in vector_col.split(',')]
                    else:
                        vector_data["vector"] = []
            
            # Add manifest fields
            if manifest:
                if isinstance(manifest, dict):
                    vector_data.update(manifest)
                else:
                    vector_data.update(json.loads(manifest))
            
            return vector_data
    
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
            collection_name: Name of the collection
            offset: Number of items to skip
            limit: Maximum number of items to return
            include_vectors: Whether to include vectors in results
            order_by: Field to sort by (e.g., 'created_at', '-score')
            pagination: If True, return dict with items and pagination info
        """
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim = collection_info
        
        parent_table = self._get_parent_table(dim)
        
        # Determine which fields to select
        if include_vectors:
            select_clause = "id, vector, manifest"
        else:
            select_clause = "id, manifest"
        
        # Build ORDER BY clause
        order_clause = ""
        if order_by:
            order_clause = f" ORDER BY {order_by}"
        else:
            order_clause = " ORDER BY id"
        
        async with self._engine.connect() as conn:
            # Get total count if pagination requested
            total = None
            if pagination:
                count_result = await conn.execute(text(f"""
                    SELECT COUNT(*) FROM {parent_table} 
                    WHERE collection_id = :collection_id
                """), {"collection_id": collection_id})
                total = count_result.scalar()
            
            # Get vectors
            result = await conn.execute(text(f"""
                SELECT {select_clause} FROM {parent_table}
                WHERE collection_id = :collection_id
                {order_clause}
                LIMIT :limit OFFSET :offset
            """), {
                "collection_id": collection_id, 
                "limit": limit, 
                "offset": offset
            })
            rows = result.fetchall()
            
            vectors = []
            for row in rows:
                if include_vectors:
                    vector_id, vector_col, manifest = row
                    vector_data = {"id": vector_id}
                    
                    # Add vector field
                    if vector_col is not None:
                        if isinstance(vector_col, np.ndarray):
                            vector_data["vector"] = vector_col.tolist()
                        elif isinstance(vector_col, str):
                            vector_col = vector_col.strip('[]')
                            if vector_col:
                                vector_data["vector"] = [float(x.strip()) for x in vector_col.split(',')]
                            else:
                                vector_data["vector"] = []
                else:
                    vector_id, manifest = row
                    vector_data = {"id": vector_id}
                
                # Add manifest fields
                if manifest:
                    if isinstance(manifest, dict):
                        vector_data.update(manifest)
                    else:
                        vector_data.update(json.loads(manifest))
                
                vectors.append(vector_data)
        
        if pagination:
            return {
                "items": vectors,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
        else:
            return vectors
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: Optional[Union[List[float], np.ndarray, str]] = None,
        embedding_model: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 5,
        offset: Optional[int] = 0,
        include_vectors: bool = False,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search (optional)
            embedding_model: Model to use for text embeddings if query_vector is text
            filters: Filter criteria on manifest fields. Supports:
                - Exact match: {'field': 'value'}
                - Range filter: {'field': [min, max]}
                - Operators: {'field': {'$gt': 5, '$lt': 10}}
                - In list: {'field': {'$in': ['value1', 'value2']}}
                - Like pattern: {'field': {'$like': '%pattern%'}}
                - Not equal: {'field': {'$ne': 'value'}}
                - Regex pattern: {'field': {'$regex': 'pattern'}}
                - All values: {'field': {'$all': ['value1', 'value2']}}
            limit: Maximum number of results
            offset: Number of results to skip
            include_vectors: Whether to include vectors in results
            order_by: Field to sort by (if no query_vector)
            pagination: If True, return dict with items and pagination info
        """
        await self._ensure_initialized()
        
        # Get collection info and metadata
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim, vector_fields FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim, metadata_json = collection_info
            
            # Parse metadata
            if isinstance(metadata_json, str):
                metadata = json.loads(metadata_json)
            else:
                metadata = metadata_json
            
            distance_metric = metadata.get("distance_metric", "cosine").upper()
        
        parent_table = self._get_parent_table(dim)
        
        # Process query vector
        vector_query = None
        if query_vector is not None:
            # Convert query vector to string format
            if isinstance(query_vector, np.ndarray):
                vec_list = query_vector.astype("float32").tolist()
                vector_query = f"[{','.join(map(str, vec_list))}]"
            elif isinstance(query_vector, (list, tuple)):
                vector_query = f"[{','.join(map(str, query_vector))}]"
            elif isinstance(query_vector, str) and embedding_model:
                # Generate embedding from text
                embeddings = await self._embed_texts([query_vector], embedding_model)
                vec_list = embeddings[0] if isinstance(embeddings[0], list) else embeddings[0].tolist()
                vector_query = f"[{','.join(map(str, vec_list))}]"
            else:
                raise ValueError(f"Invalid query_vector type: {type(query_vector)}")
        
        # Build WHERE clause for manifest filters
        where_clauses = [f"collection_id = {collection_id}"]
        params = {"collection_id": collection_id}
        
        if filters:
            # Process filters on manifest JSONB field similar to artifact.py
            filter_idx = 0
            for field_name, value in filters.items():
                if isinstance(value, dict):
                    # Handle operator-based filters
                    for operator, operand in value.items():
                        filter_idx += 1
                        param_name = f"filter_{field_name}_{filter_idx}"
                        
                        if operator == "$like":
                            # Pattern matching with ILIKE
                            where_clauses.append(f"manifest->>'{field_name}' ILIKE :{param_name}")
                            params[param_name] = str(operand)
                        elif operator == "$in":
                            # Check if value is in list
                            if isinstance(operand, (list, tuple)):
                                in_params = []
                                for i, item in enumerate(operand):
                                    item_param = f"{param_name}_{i}"
                                    in_params.append(f":{item_param}")
                                    params[item_param] = str(item)
                                where_clauses.append(f"manifest->>'{field_name}' IN ({','.join(in_params)})")
                            else:
                                raise ValueError(f"$in operator requires a list, got {type(operand)}")
                        elif operator == "$all":
                            # Check if JSONB array contains all values
                            if isinstance(operand, (list, tuple)):
                                # Use literal JSON for JSONB containment operator
                                json_value = json.dumps(operand).replace("'", "''")
                                where_clauses.append(f"manifest->'{field_name}' @> '{json_value}'::jsonb")
                            else:
                                raise ValueError(f"$all operator requires a list, got {type(operand)}")
                        elif operator == "$gt":
                            # Greater than
                            where_clauses.append(f"(manifest->>'{field_name}')::float > :{param_name}")
                            params[param_name] = float(operand)
                        elif operator == "$gte":
                            # Greater than or equal
                            where_clauses.append(f"(manifest->>'{field_name}')::float >= :{param_name}")
                            params[param_name] = float(operand)
                        elif operator == "$lt":
                            # Less than
                            where_clauses.append(f"(manifest->>'{field_name}')::float < :{param_name}")
                            params[param_name] = float(operand)
                        elif operator == "$lte":
                            # Less than or equal
                            where_clauses.append(f"(manifest->>'{field_name}')::float <= :{param_name}")
                            params[param_name] = float(operand)
                        elif operator == "$ne":
                            # Not equal
                            where_clauses.append(f"manifest->>'{field_name}' != :{param_name}")
                            params[param_name] = str(operand)
                        elif operator == "$regex":
                            # Regular expression matching
                            where_clauses.append(f"manifest->>'{field_name}' ~ :{param_name}")
                            params[param_name] = str(operand)
                        else:
                            # Nested filter (e.g., {'metadata': {'status': 'published'}})
                            json_path = f"manifest->'{field_name}'->>'{operator}'"
                            where_clauses.append(f"{json_path} = :{param_name}")
                            params[param_name] = str(operand)
                elif isinstance(value, (list, tuple)) and len(value) == 2:
                    # Range filter for numeric fields
                    where_clauses.append(
                        f"(manifest->>'{field_name}')::float BETWEEN :min_{field_name} AND :max_{field_name}"
                    )
                    params[f"min_{field_name}"] = value[0]
                    params[f"max_{field_name}"] = value[1]
                else:
                    # Exact match filter or text search
                    if isinstance(value, str) and '%' in value:
                        # Support LIKE patterns for backward compatibility
                        where_clauses.append(f"manifest->>'{field_name}' ILIKE :filter_{field_name}")
                    else:
                        where_clauses.append(f"manifest->>'{field_name}' = :filter_{field_name}")
                    params[f"filter_{field_name}"] = str(value)
        
        where_clause = " AND ".join(where_clauses)
        
        # Build SELECT clause
        select_fields = ["id", "manifest"]
        if include_vectors:
            select_fields.insert(1, "vector")
        
        # Add distance/score calculation if doing vector search
        if vector_query:
            if distance_metric == "COSINE":
                select_fields.append(f"(vector <=> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 - (vector <=> CAST(:query_vector AS vector))) AS _score")
            elif distance_metric == "L2":
                select_fields.append(f"(vector <-> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 / (1 + (vector <-> CAST(:query_vector AS vector)))) AS _score")
            elif distance_metric == "IP":
                select_fields.append(f"(vector <#> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(vector <#> CAST(:query_vector AS vector)) AS _score")
            else:
                select_fields.append(f"(vector <=> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 - (vector <=> CAST(:query_vector AS vector))) AS _score")
        
        select_clause = ", ".join(select_fields)
        
        # Build ORDER BY clause
        if vector_query and not order_by:
            if distance_metric == "COSINE":
                order_clause = f" ORDER BY (vector <=> CAST(:query_vector AS vector)) ASC"
            elif distance_metric == "L2":
                order_clause = f" ORDER BY (vector <-> CAST(:query_vector AS vector)) ASC"
            elif distance_metric == "IP":
                order_clause = f" ORDER BY (vector <#> CAST(:query_vector AS vector)) DESC"
            else:
                order_clause = f" ORDER BY (vector <=> CAST(:query_vector AS vector)) ASC"
        elif order_by:
            order_clause = f" ORDER BY {order_by}"
        else:
            order_clause = " ORDER BY id"
        
        async with self._sessionmaker() as session:
            # Set optimal probes for search if doing vector query
            if vector_query:
                try:
                    # Get current row count for optimization
                    count_result = await session.execute(text(f"""
                        SELECT COUNT(*) FROM {parent_table} WHERE collection_id = :collection_id
                    """), {"collection_id": collection_id})
                    row_count = count_result.scalar() or 0
                    lists, probes = self._calculate_ivfflat_parameters(row_count)
                    
                    await session.execute(text(f"SET ivfflat.probes = {probes}"))
                    logger.debug(f"Set ivfflat.probes = {probes} for query optimization")
                except Exception as e:
                    logger.debug(f"Could not set ivfflat.probes: {e}")
            
            # Get total count if pagination requested
            total = None
            if pagination:
                count_query = f"SELECT COUNT(*) FROM {parent_table} WHERE {where_clause}"
                count_result = await session.execute(text(count_query), params)
                total = count_result.scalar()
            
            # Build and execute search query
            search_query = f"""
                SELECT {select_clause}
                FROM {parent_table}
                WHERE {where_clause}
                {order_clause}
                LIMIT :limit OFFSET :offset
            """
            
            if vector_query:
                params["query_vector"] = vector_query
            params["limit"] = limit
            params["offset"] = offset
            
            result = await session.execute(text(search_query), params)
            rows = result.fetchall()
            
            vectors = []
            for row in rows:
                row_dict = dict(row._mapping)
                vector_data = {"id": row_dict["id"]}
                
                # Add vector field if included
                if "vector" in row_dict:
                    vector_col = row_dict["vector"]
                    if isinstance(vector_col, np.ndarray):
                        vector_data["vector"] = vector_col.tolist()
                    elif isinstance(vector_col, str):
                        vector_col = vector_col.strip('[]')
                        if vector_col:
                            vector_data["vector"] = [float(x.strip()) for x in vector_col.split(',')]
                        else:
                            vector_data["vector"] = []
                
                # Add manifest fields
                manifest = row_dict.get("manifest")
                if manifest:
                    if isinstance(manifest, dict):
                        vector_data.update(manifest)
                    else:
                        vector_data.update(json.loads(manifest))
                
                # Add score if present (from similarity search)
                if "_score" in row_dict:
                    vector_data["_score"] = row_dict["_score"]
                
                vectors.append(vector_data)
        
        if pagination:
            return {
                "items": vectors,
                "total": total,
                "offset": offset,
                "limit": limit,
            }
        else:
            return vectors
    
    async def _embed_texts(
        self,
        texts: List[str],
        embedding_model: str = "fastembed:BAAI/bge-small-en-v1.5",
    ) -> List[np.ndarray]:
        """Generate embeddings for texts."""
        if not embedding_model:
            raise ValueError("Embedding model is not configured.")
        
        if embedding_model.startswith("fastembed:"):
            from fastembed import TextEmbedding
            
            model_name = embedding_model.split(":")[1]
            model = TextEmbedding(
                model_name=model_name, cache_dir=self._cache_dir
            )
            loop = asyncio.get_event_loop()
            embeddings = list(
                await loop.run_in_executor(None, model.embed, texts)
            )
            return embeddings
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")