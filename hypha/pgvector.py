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
    
    # Supported dimensions - can be extended as needed
    SUPPORTED_DIMS = [32, 64, 128, 256, 384, 512, 768, 1024, 1536]
    
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
            
            # Create collection registry table
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self._prefix}.collection_registry (
                    collection_id BIGINT PRIMARY KEY,
                    collection_name TEXT UNIQUE NOT NULL,
                    dim INT NOT NULL CHECK (dim IN ({','.join(map(str, self.SUPPORTED_DIMS))})),
                    vector_fields JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            
            # Create parent tables for each supported dimension
            for dim in self.SUPPORTED_DIMS:
                parent_table = f"{self._prefix}.emb_{dim}_parent"
                await conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {parent_table} (
                        collection_id BIGINT NOT NULL,
                        id TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        payload JSONB,
                        created_at TIMESTAMPTZ DEFAULT now(),
                        PRIMARY KEY (collection_id, id)
                    ) PARTITION BY LIST (collection_id)
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
    
    async def create_collection(
        self,
        collection_name: str,
        vector_fields: List[Dict[str, Any]],
        overwrite: bool = False,
        **kwargs  # Remove s3_client, bucket, prefix from signature
    ):
        """Create a vector collection using partition-by-dimension approach.
        
        Args:
            collection_name: Name of the collection
            vector_fields: List of field definitions
            overwrite: Whether to overwrite existing collection
        """
        await self._ensure_initialized()
        
        # Extract vector field dimension
        vector_field = None
        for field in vector_fields:
            if field.get("type") == "VECTOR":
                if vector_field is not None:
                    raise ValueError("Only one vector field per collection is supported")
                vector_field = field
                break
        
        if not vector_field:
            raise ValueError("At least one VECTOR field is required")
        
        dim = vector_field.get("attributes", {}).get("DIM", 384)
        if dim not in self.SUPPORTED_DIMS:
            raise ValueError(f"Dimension {dim} not supported. Supported dimensions: {self.SUPPORTED_DIMS}")
        
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
            
            # Insert into registry
            await conn.execute(text(f"""
                INSERT INTO {self._prefix}.collection_registry 
                (collection_id, collection_name, dim, vector_fields)
                VALUES (:collection_id, :collection_name, :dim, :vector_fields)
                ON CONFLICT (collection_id) DO UPDATE SET 
                    collection_name = EXCLUDED.collection_name,
                    dim = EXCLUDED.dim,
                    vector_fields = EXCLUDED.vector_fields
            """), {
                "collection_id": collection_id,
                "collection_name": collection_name,
                "dim": dim,
                "vector_fields": json.dumps(vector_fields)
            })
            
            # Create child partition
            parent_table = self._get_parent_table(dim)
            child_table = self._get_child_table(dim, collection_id)
            child_table_simple = child_table.split('.')[-1]  # Get table name without schema
            
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {child_table}
                PARTITION OF {parent_table}
                FOR VALUES IN ({collection_id})
            """))
            
            # Create vector index with optimal parameters
            distance_metric = vector_field.get("attributes", {}).get("DISTANCE_METRIC", "cosine").lower()
            ops_map = {
                "cosine": "vector_cosine_ops",
                "l2": "vector_l2_ops", 
                "ip": "vector_ip_ops"
            }
            ops = ops_map.get(distance_metric, "vector_cosine_ops")
            
            # Start with minimal lists for new collection
            lists = 1
            index_name = f"{child_table_simple}_hnsw"
            
            try:
                await conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {child_table} USING ivfflat (embedding {ops})
                    WITH (lists = {lists})
                """))
                logger.info(f"Created vector index {index_name} for collection {collection_name} with lists={lists}")
            except ProgrammingError:
                # Fallback to HNSW if available
                try:
                    await conn.execute(text(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {child_table} USING hnsw (embedding {ops})
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
    
    async def _get_fields(self, collection_name: str) -> Dict[str, Any]:
        """Get field information for a collection."""
        await self._ensure_initialized()
        
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT vector_fields, dim FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Collection {collection_name} not found")
            
            vector_fields, dim = row
            fields = {}
            
            for field in vector_fields:
                field_name = field["name"]
                field_info = {
                    "identifier": field_name,
                    "type": field["type"],
                    **field.get("attributes", {})
                }
                # Ensure dimension is set for vector fields
                if field["type"] == "VECTOR":
                    field_info["DIM"] = dim
                fields[field_name] = field_info
            
            # Always include id field
            fields["id"] = {"identifier": "id", "type": "TEXT"}
            
            return fields
    
    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]],
        update: bool = False,
        embedding_models: Optional[Dict[str, str]] = None,
        **kwargs  # Remove s3_client parameters
    ) -> List[str]:
        """Add vectors to a collection."""
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim, vector_fields FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim, vector_fields = collection_info
        
        parent_table = self._get_parent_table(dim)
        vector_field_name = None
        
        # Find the vector field name
        for field in vector_fields:
            if field["type"] == "VECTOR":
                vector_field_name = field["name"]
                break
        
        if not vector_field_name:
            raise ValueError("No vector field found in collection")
        
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
                vector_value = vector_data.get(vector_field_name)
                if vector_value is None:
                    raise ValueError(f"Vector field '{vector_field_name}' is required")
                
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
                elif isinstance(vector_value, str) and embedding_models and vector_field_name in embedding_models:
                    # Generate embedding from text
                    embeddings = await self._embed_texts([vector_value], embedding_models[vector_field_name])
                    vec_list = embeddings[0] if isinstance(embeddings[0], list) else embeddings[0].tolist()
                    vector_str = f"[{','.join(map(str, vec_list))}]"
                else:
                    if isinstance(vector_value, list):
                        vector_str = f"[{','.join(map(str, vector_value))}]"
                    else:
                        vector_str = str(vector_value)
                
                # Prepare payload (all non-vector fields)
                payload = {}
                for key, value in vector_data.items():
                    if key not in ["id", vector_field_name]:
                        payload[key] = value
                
                if update:
                    # Update existing vector
                    await session.execute(text(f"""
                        UPDATE {parent_table} 
                        SET embedding = CAST(:embedding AS vector), payload = :payload
                        WHERE collection_id = :collection_id AND id = :id
                    """), {
                        "collection_id": collection_id,
                        "id": vector_id,
                        "embedding": vector_str,
                        "payload": json.dumps(payload) if payload else None
                    })
                else:
                    # Insert new vector
                    try:
                        await session.execute(text(f"""
                            INSERT INTO {parent_table} (collection_id, id, embedding, payload)
                            VALUES (:collection_id, :id, CAST(:embedding AS vector), :payload)
                        """), {
                            "collection_id": collection_id,
                            "id": vector_id,
                            "embedding": vector_str,
                            "payload": json.dumps(payload) if payload else None
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
                SELECT collection_id, dim, vector_fields FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim, vector_fields = collection_info
        
        parent_table = self._get_parent_table(dim)
        vector_field_name = None
        
        # Find the vector field name
        for field in vector_fields:
            if field["type"] == "VECTOR":
                vector_field_name = field["name"]
                break
        
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT id, embedding, payload FROM {parent_table}
                WHERE collection_id = :collection_id AND id = :id
            """), {"collection_id": collection_id, "id": str(id)})
            row = result.fetchone()
            
            if not row:
                raise ValueError(f"Vector {id} not found in collection {collection_name}")
            
            vector_id, embedding, payload = row
            
            # Build result
            vector_data = {"id": vector_id}
            
            # Add vector field
            if embedding is not None and vector_field_name:
                if isinstance(embedding, np.ndarray):
                    vector_data[vector_field_name] = embedding.tolist()
                elif isinstance(embedding, str):
                    # Parse pgvector string format
                    embedding = embedding.strip('[]')
                    if embedding:
                        vector_data[vector_field_name] = [float(x.strip()) for x in embedding.split(',')]
                    else:
                        vector_data[vector_field_name] = []
            
            # Add payload fields
            if payload:
                if isinstance(payload, dict):
                    vector_data.update(payload)
                else:
                    vector_data.update(json.loads(payload))
            
            return vector_data
    
    async def list_vectors(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 10,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """List vectors in a collection."""
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim, vector_fields FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim, vector_fields = collection_info
        
        parent_table = self._get_parent_table(dim)
        vector_field_name = None
        
        # Find the vector field name
        for field in vector_fields:
            if field["type"] == "VECTOR":
                vector_field_name = field["name"]
                break
        
        # Determine which fields to select
        if return_fields is None:
            # Return id and payload fields by default (not the vector)
            select_clause = "id, payload"
        else:
            if vector_field_name in return_fields:
                select_clause = "id, embedding, payload"
            else:
                select_clause = "id, payload"
        
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
                if select_clause.startswith("id, payload"):
                    vector_id, payload = row
                    vector_data = {"id": vector_id}
                else:  # id, embedding, payload
                    vector_id, embedding, payload = row
                    vector_data = {"id": vector_id}
                    
                    # Add vector field if requested
                    if return_fields and vector_field_name in return_fields and embedding is not None:
                        if isinstance(embedding, np.ndarray):
                            vector_data[vector_field_name] = embedding.tolist()
                        elif isinstance(embedding, str):
                            embedding = embedding.strip('[]')
                            if embedding:
                                vector_data[vector_field_name] = [float(x.strip()) for x in embedding.split(',')]
                            else:
                                vector_data[vector_field_name] = []
                
                # Add payload fields
                if payload:
                    if isinstance(payload, dict):
                        vector_data.update(payload)
                    else:
                        vector_data.update(json.loads(payload))
                
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
        query: Optional[Dict[str, Any]] = None,
        embedding_models: Optional[Dict[str, str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        extra_filter: Optional[str] = None,
        limit: Optional[int] = 5,
        offset: Optional[int] = 0,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search vectors in a collection."""
        await self._ensure_initialized()
        
        # Get collection info
        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                SELECT collection_id, dim, vector_fields FROM {self._prefix}.collection_registry
                WHERE collection_name = :collection_name
            """), {"collection_name": collection_name})
            collection_info = result.fetchone()
            
            if not collection_info:
                raise ValueError(f"Collection {collection_name} not found")
            
            collection_id, dim, vector_fields = collection_info
        
        parent_table = self._get_parent_table(dim)
        vector_field_name = None
        distance_metric = "COSINE"
        
        # Find the vector field
        for field in vector_fields:
            if field["type"] == "VECTOR":
                vector_field_name = field["name"]
                distance_metric = field.get("attributes", {}).get("DISTANCE_METRIC", "COSINE").upper()
                break
        
        # Process vector query
        vector_query = None
        if query and vector_field_name in query:
            query_value = query[vector_field_name]
            
            # Convert query vector to string format
            if isinstance(query_value, np.ndarray):
                vec_list = query_value.astype("float32").tolist()
                vector_query = f"[{','.join(map(str, vec_list))}]"
            elif isinstance(query_value, (list, tuple)):
                vector_query = f"[{','.join(map(str, query_value))}]"
            elif isinstance(query_value, bytes):
                arr = np.frombuffer(query_value, dtype=np.float32)
                vec_list = arr.tolist()
                vector_query = f"[{','.join(map(str, vec_list))}]"
            elif isinstance(query_value, str) and embedding_models and vector_field_name in embedding_models:
                # Generate embedding from text
                embeddings = await self._embed_texts([query_value], embedding_models[vector_field_name])
                vec_list = embeddings[0] if isinstance(embeddings[0], list) else embeddings[0].tolist()
                vector_query = f"[{','.join(map(str, vec_list))}]"
            else:
                if isinstance(query_value, list):
                    vector_query = f"[{','.join(map(str, query_value))}]"
                else:
                    vector_query = str(query_value)
        
        # Build WHERE clause for payload filters
        where_clauses = [f"collection_id = {collection_id}"]
        params = {"collection_id": collection_id}
        
        if filters:
            # Filters are applied to payload JSONB field
            for field_name, value in filters.items():
                if field_name == vector_field_name:
                    continue  # Skip vector field in filters
                
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    # Range filter
                    where_clauses.append(f"(payload->>:{field_name}_key)::float BETWEEN :min_{field_name} AND :max_{field_name}")
                    params[f"{field_name}_key"] = field_name
                    params[f"min_{field_name}"] = value[0]
                    params[f"max_{field_name}"] = value[1]
                else:
                    # Exact match filter
                    where_clauses.append(f"payload->>:{field_name}_key = :filter_{field_name}")
                    params[f"{field_name}_key"] = field_name
                    params[f"filter_{field_name}"] = str(value)
        
        if extra_filter:
            where_clauses.append(f"({extra_filter})")
        
        where_clause = " AND ".join(where_clauses)
        
        # Build SELECT clause
        select_fields = ["id", "payload"]
        if return_fields and vector_field_name in return_fields:
            select_fields.insert(1, "embedding")
        
        # Add distance/score calculation if doing vector search
        if vector_query:
            if distance_metric == "COSINE":
                select_fields.append(f"(embedding <=> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 - (embedding <=> CAST(:query_vector AS vector))) AS _score")
            elif distance_metric == "L2":
                select_fields.append(f"(embedding <-> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 / (1 + (embedding <-> CAST(:query_vector AS vector)))) AS _score")
            elif distance_metric == "IP":
                select_fields.append(f"(embedding <#> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(embedding <#> CAST(:query_vector AS vector)) AS _score")
            else:
                select_fields.append(f"(embedding <=> CAST(:query_vector AS vector)) AS distance")
                select_fields.append(f"(1 - (embedding <=> CAST(:query_vector AS vector))) AS _score")
        
        select_clause = ", ".join(select_fields)
        
        # Build ORDER BY clause
        if vector_query and not order_by:
            if distance_metric == "COSINE":
                order_clause = f" ORDER BY (embedding <=> CAST(:query_vector AS vector)) ASC"
            elif distance_metric == "L2":
                order_clause = f" ORDER BY (embedding <-> CAST(:query_vector AS vector)) ASC"
            elif distance_metric == "IP":
                order_clause = f" ORDER BY (embedding <#> CAST(:query_vector AS vector)) DESC"
            else:
                order_clause = f" ORDER BY (embedding <=> CAST(:query_vector AS vector)) ASC"
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
                if "embedding" in row_dict and vector_field_name:
                    embedding = row_dict["embedding"]
                    if isinstance(embedding, np.ndarray):
                        vector_data[vector_field_name] = embedding.tolist()
                    elif isinstance(embedding, str):
                        embedding = embedding.strip('[]')
                        if embedding:
                            vector_data[vector_field_name] = [float(x.strip()) for x in embedding.split(',')]
                        else:
                            vector_data[vector_field_name] = []
                
                # Add payload fields
                payload = row_dict.get("payload")
                if payload:
                    if isinstance(payload, dict):
                        vector_data.update(payload)
                    else:
                        vector_data.update(json.loads(payload))
                
                # Add score if present
                if "_score" in row_dict:
                    vector_data["score"] = row_dict["_score"]
                
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