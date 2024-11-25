import asyncio
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter
from qdrant_client.models import Distance, VectorParams
import uuid
import numpy as np
from typing import Any
import logging
import sys

# Logger setup
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger("vectors")
logger.setLevel(logging.INFO)


class VectorSearchEngine:
    def __init__(self, store):
        self.store = store
        self._vectordb_client = self.store.get_vectordb_client()
        self._cache_dir = self.store.get_cache_dir()

    def _ensure_client(self):
        assert (
            self._vectordb_client
        ), "The server is not configured to use a VectorDB client."

    async def create_collection(self, collection_name: str, vectors_config: dict):
        self._ensure_client()
        await self._vectordb_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vectors_config.get("size", 128),
                distance=Distance(vectors_config.get("distance", "Cosine")),
            ),
        )
        logger.info(f"Collection {collection_name} created.")

    async def count(self, collection_name: str):
        self._ensure_client()
        return (
            await self._vectordb_client.count(collection_name=collection_name)
        ).count

    async def delete_collection(self, collection_name: str):
        self._ensure_client()
        await self._vectordb_client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection {collection_name} deleted.")

    async def _embed_texts(self, embedding_model, texts: list):
        self._ensure_client()
        assert (
            embedding_model
        ), "Embedding model must be provided, e.g. 'fastembed:BAAI/bge-small-en-v1.5', 'openai:text-embedding-3-small' for openai embeddings."
        if embedding_model.startswith("fastembed"):
            from fastembed import TextEmbedding

            assert ":" in embedding_model, "Embedding model must be provided."
            model_name = embedding_model.split(":")[-1]
            embedding_model = TextEmbedding(
                model_name=model_name, cache_dir=self._cache_dir
            )
            loop = asyncio.get_event_loop()
            embeddings = list(
                await loop.run_in_executor(None, embedding_model.embed, texts)
            )
        elif embedding_model.startswith("openai"):
            assert (
                self._openai_client
            ), "The server is not configured to use an OpenAI client."
            assert ":" in embedding_model, "Embedding model must be provided."
            embedding_model = embedding_model.split(":")[-1]
            result = await self._openai_client.embeddings.create(
                input=texts, model=embedding_model
            )
            embeddings = [data.embedding for data in result.data]
        else:
            raise ValueError(
                f"Unsupported embedding model: {embedding_model}, supported models: 'fastembed:*', 'openai:*'"
            )
        return embeddings

    async def add_vectors(self, collection_name: str, vectors: list):
        self._ensure_client()
        assert isinstance(vectors, list), "Vectors must be a list of dictionaries."
        assert all(
            isinstance(v, dict) for v in vectors
        ), "Vectors must be a list of dictionaries."

        _points = []
        for p in vectors:
            p["id"] = p.get("id") or str(uuid.uuid4())
            _points.append(PointStruct(**p))
        await self._vectordb_client.upsert(
            collection_name=collection_name,
            points=_points,
        )
        logger.info(f"Added {len(vectors)} vectors to collection {collection_name}.")

    async def add_documents(self, collection_name, documents, embedding_model):
        self._ensure_client()
        texts = [doc["text"] for doc in documents]
        embeddings = await self._embed_texts(embedding_model, texts)
        points = [
            PointStruct(
                id=doc.get("id") or str(uuid.uuid4()),
                vector=embedding,
                payload=doc,
            )
            for embedding, doc in zip(embeddings, documents)
        ]
        await self._vectordb_client.upsert(
            collection_name=collection_name,
            points=points,
        )
        logger.info(
            f"Added {len(documents)} documents to collection {collection_name}."
        )

    async def search_vectors(
        self,
        collection_name: str,
        embedding_model: str,
        query_text: str,
        query_vector: Any,
        query_filter: dict = None,
        offset: int = 0,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        pagination: bool = False,
    ):
        self._ensure_client()
        # if it's a numpy array, convert it to a list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        from qdrant_client.models import Filter

        if query_filter:
            query_filter = Filter.model_validate(query_filter)

        if query_text:
            assert (
                not query_vector
            ), "Either query_text or query_vector must be provided."
            embeddings = await self._embed_texts(embedding_model, [query_text])
            query_vector = embeddings[0]

        search_results = await self._vectordb_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        if pagination:
            count = await self._vectordb_client.count(collection_name=collection_name)
            return {
                "total": count.count,
                "items": search_results,
                "offset": offset,
                "limit": limit,
            }
        logger.info(f"Performed semantic search in collection {collection_name}.")
        return search_results

    async def remove_vectors(self, collection_name: str, ids: list):
        self._ensure_client()
        await self._vectordb_client.delete(
            collection_name=collection_name,
            points_selector=ids,
        )
        logger.info(f"Removed {len(ids)} vectors from collection {collection_name}.")

    async def get_vector(self, collection_name: str, id: str):
        self._ensure_client()
        points = await self._vectordb_client.retrieve(
            collection_name=collection_name,
            ids=[id],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            raise ValueError(f"Vector {id} not found in collection {collection_name}.")
        logger.info(f"Retrieved vector {id} from collection {collection_name}.")
        return points[0]

    async def list_vectors(
        self,
        collection_name: str,
        query_filter: dict = None,
        offset: int = 0,
        limit: int = 10,
        order_by: str = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        self._ensure_client()
        if query_filter:
            query_filter = Filter.model_validate(query_filter)
        points, _ = await self._vectordb_client.scroll(
            collection_name=collection_name,
            scroll_filter=query_filter,
            limit=limit,
            offset=offset,
            order_by=order_by,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        logger.info(f"Listed vectors in collection {collection_name}.")
        return points
