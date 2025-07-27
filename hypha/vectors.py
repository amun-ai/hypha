import asyncio
import numpy as np
import uuid
import logging
import os
import sys
import re
import json
from typing import Any, List, Dict, Optional
from hypha.utils import list_objects_async, remove_objects_async
from fakeredis import aioredis
from redis.commands.search.field import (
    TagField,
    TextField,
    NumericField,
    GeoField,
    VectorField,
)

try:
    # for new redis-py, e.g. 6.2.0
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    # fallback to old version of redis-py, e.g. 5.2.0
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redis.commands.search.query import Query
from redis.commands.search.query import Query

LOGLEVEL = os.environ.get("HYPHA_LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, stream=sys.stdout)
logger = logging.getLogger("vectors")
logger.setLevel(LOGLEVEL)

FIELD_TYPE_MAPPING = {
    "TAG": TagField,
    "TEXT": TextField,
    "NUMERIC": NumericField,
    "GEO": GeoField,
    "VECTOR": VectorField,
}


def escape_redis_syntax(value: str) -> str:
    """Escape Redis special characters in a query string, except '*'."""
    # Escape all special characters except '*'
    return re.sub(r"([{}|@$\\\-\[\]\(\)\!&~:\"])", r"\\\1", value)


def sanitize_search_value(value: str) -> str:
    """Sanitize a value to prevent injection attacks, allowing '*' for wildcard support."""
    # Allow alphanumeric characters, spaces, underscores, hyphens, dots, slashes, and '*'
    value = re.sub(
        r"[^a-zA-Z0-9 _\-./*]", "", value
    )  # Remove unwanted characters except '*'
    return escape_redis_syntax(value.strip())


def parse_attributes(attributes):
    """
    Parse the attributes list into a structured dictionary.
    """
    parsed_attributes = []
    for attr in attributes:
        attr_dict = {}
        for i in range(0, len(attr), 2):
            key = attr[i].decode("utf-8") if isinstance(attr[i], bytes) else attr[i]
            value = attr[i + 1]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            attr_dict[key] = value
        parsed_attributes.append(attr_dict)
    return {attr["identifier"]: attr for attr in parsed_attributes}


class VectorSearchEngine:
    def __init__(self, redis, prefix: str = "vs", cache_dir=None):
        self._redis: aioredis.FakeRedis = redis
        self._index_name_prefix = prefix
        self._cache_dir = cache_dir
        self._maximum_search_results = None

    def _get_index_name(self, collection_name: str) -> str:
        return (
            f"{self._index_name_prefix}:{collection_name}"
            if self._index_name_prefix
            else collection_name
        )

    async def _get_fields(self, collection_name: str):
        index_name = self._get_index_name(collection_name)
        try:
            info = await self._redis.ft(index_name).info()
        except aioredis.ResponseError:
            raise KeyError(f"Vector collection {collection_name} does not exist.")
        fields = parse_attributes(info["attributes"])
        return fields

    async def create_collection(
        self,
        collection_name: str,
        vector_fields: List[Dict[str, Any]],
        overwrite: bool = False,
        s3_client=None,
        bucket=None,
        prefix=None,
    ):
        """
        Creates a RedisSearch index collection.

        Args:
            collection_name (str): Name of the collection.
            fields (List[Dict[str, Any]]): A list of dictionaries defining fields.
                Example:
                [
                    {"type": "TAG", "name": "id"},
                    {"type": "VECTOR", "name": "vector", "algorithm": "FLAT",
                     "attributes": {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}}
                ]
        """
        index_name = self._get_index_name(collection_name)
        assert vector_fields, "At least one field must be provided."

        redis_fields = []
        for field in vector_fields:
            field_type = field.get("type")
            field_name = field.get("name")
            if not field_type or not field_name:
                raise ValueError(
                    f"Invalid field definition: {field}. Each field must have 'name' and 'type'."
                )

            field_class = FIELD_TYPE_MAPPING.get(field_type)
            if not field_class:
                raise ValueError(
                    f"Unsupported field type: {field_type}. Supported types: {list(FIELD_TYPE_MAPPING.keys())}"
                )

            if field_type == "VECTOR":
                algorithm = field.get("algorithm", "FLAT")
                attributes = field.get("attributes", {})
                redis_fields.append(field_class(field_name, algorithm, attributes))
            else:
                redis_fields.append(field_class(name=field_name))
        # Check if the collection already exists, remove it if overwrite is True
        # otherwise raise an error
        try:
            await self._redis.ft(index_name).info()
            if overwrite:
                # Drop the index
                await self._redis.ft(index_name).dropindex(delete_documents=False)
            else:
                raise ValueError(f"Collection {collection_name} already exists.")
        # ResponseError is raised if the collection does not exist
        except aioredis.ResponseError:
            pass

        # Create the index
        await self._redis.ft(index_name).create_index(
            fields=redis_fields,
            definition=IndexDefinition(
                prefix=[f"{index_name}:"], index_type=IndexType.HASH
            ),
        )
        if s3_client:
            await self._dump_index(collection_name, s3_client, bucket, prefix)
        fields_info = [
            f"{field.name} ({type(field).__name__})" for field in redis_fields
        ]
        logger.info(f"Collection {collection_name} created with fields: {fields_info}.")

    async def count(self, collection_name: str):
        index_name = self._get_index_name(collection_name)
        count_query = Query("*").paging(0, 0).dialect(2)
        results = await self._redis.ft(index_name).search(count_query)
        return results.total

    async def delete_collection(
        self, collection_name: str, s3_client=None, bucket=None, prefix=None
    ):
        index_name = self._get_index_name(collection_name)

        # Drop the index
        await self._redis.ft(index_name).dropindex(delete_documents=True)
        if s3_client:
            if not prefix.endswith("/"):
                prefix = f"{prefix}/"
            await remove_objects_async(s3_client, bucket, prefix)

        logger.info(f"Collection {collection_name} and all associated keys deleted.")

    async def _dump_vectors(
        self,
        vectors: List[Dict[str, Any]],
        fields: Dict[str, Any],
        s3_client,
        bucket: str,
        prefix: str,
    ):
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"
        # Dump vector data for the current page
        for vector in vectors:
            vector_id = vector["id"]
            vector_data = {key: value for key, value in vector.items()}

            # Convert vector bytes to list for JSON serialization
            for key, value in vector_data.items():
                if key not in fields:
                    continue
                if fields[key]["type"] == "VECTOR":
                    vector_data[key] = (
                        np.frombuffer(value, dtype=np.float32).astype(float).tolist()
                    )

            vector_key = f"{prefix}{vector_id}.json"
            await s3_client.put_object(
                Bucket=bucket,
                Key=vector_key,
                Body=json.dumps(vector_data),
            )

    async def _dump_index(
        self, collection_name: str, s3_client, bucket: str, prefix: str
    ):
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"
        # Dump index settings
        fields = await self._get_fields(collection_name)
        dump_fields = []
        for field in fields.values():
            field_data = {
                "type": field["type"],
                "name": field["identifier"],
            }
            if field["type"] == "VECTOR":
                field_data["algorithm"] = field["algorithm"]
                field_data["attributes"] = {
                    "TYPE": field["data_type"],
                    "DIM": field["dim"],
                    "DISTANCE_METRIC": field["distance_metric"],
                }
            dump_fields.append(field_data)
        index_settings = {
            "collection_name": collection_name,
            "index_name": self._get_index_name(collection_name),
            "fields": dump_fields,
        }
        index_key = f"{prefix}_index.json"
        await s3_client.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=json.dumps(index_settings),
        )

    async def dump_collection(
        self,
        collection_name: str,
        s3_client,
        bucket: str,
        prefix: str,
        page_size: int = 100,
    ):
        fields = await self._get_fields(collection_name)
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"
        offset = 0
        while True:
            # Fetch vectors in pages
            vectors = await self.list_vectors(
                collection_name,
                offset=offset,
                limit=page_size,
                return_fields=list(fields.keys()),
            )
            if not vectors:  # Exit loop if no more vectors are found
                break

            if s3_client:
                # Dump vectors to S3
                await self._dump_vectors(vectors, fields, s3_client, bucket, prefix)

            # Move to the next page
            offset += page_size

        if s3_client:
            await self._dump_index(collection_name, s3_client, bucket, prefix)
        logger.info(
            f"Collection {collection_name} dumped to S3 bucket {bucket} under prefix {prefix}."
        )

    async def load_collection(
        self,
        collection_name: str,
        s3_client,
        bucket: str,
        prefix: str,
        overwrite: bool = False,
    ):
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"
        index_key = f"{prefix}_index.json"

        # Load index settings
        index_response = await s3_client.get_object(Bucket=bucket, Key=index_key)
        index_settings = json.loads(await index_response["Body"].read())
        await self.create_collection(
            collection_name, index_settings["fields"], overwrite=overwrite
        )

        # List vector files

        vector_prefix = prefix
        vector_objects = await list_objects_async(s3_client, bucket, vector_prefix)

        # Load vectors
        for obj in vector_objects:
            if obj["name"].endswith(".json") and not obj["name"].endswith(
                "_index.json"
            ):
                if obj["type"] != "file":
                    continue
                key = f"{vector_prefix}{obj['name']}"
                vector_response = await s3_client.get_object(Bucket=bucket, Key=key)
                vector_data = json.loads(await vector_response["Body"].read())

                # Convert vector lists back to bytes
                for key, value in vector_data.items():
                    if isinstance(value, list):
                        vector_data[key] = np.array(value, dtype=np.float32)

                try:
                    # Add vector to Redis
                    await self.add_vectors(
                        collection_name,
                        [{"id": obj["name"].split(".")[0], **vector_data}],
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load vector {obj['name']} from S3 bucket {bucket} under prefix {prefix}: {e}"
                    )
                    raise e

        logger.info(
            f"Collection {collection_name} loaded from S3 bucket {bucket} under prefix {prefix}."
        )

    async def list_collections(self):
        # use redis.ft to list all collections
        collections = await self._redis.execute_command("FT._LIST")

        items = []
        for c in collections:
            # remove the prefix from the collection names
            items.append(
                {"name": c.decode("utf-8").replace(f"{self._index_name_prefix}:", "")}
            )
        return items

    async def _embed_texts(
        self,
        texts: List[str],
        embedding_model: str = "fastembed:BAAI/bge-small-en-v1.5",
    ) -> List[np.ndarray]:
        if not embedding_model:
            raise ValueError("Embedding model is not configured.")
        if embedding_model.startswith("fastembed:"):
            from fastembed import TextEmbedding

            model_name = embedding_model.split(":")[1]
            embedding_model = TextEmbedding(
                model_name=model_name, cache_dir=self._cache_dir
            )
        loop = asyncio.get_event_loop()
        embeddings = list(
            await loop.run_in_executor(None, embedding_model.embed, texts)
        )
        return embeddings

    async def add_vectors(
        self,
        collection_name: str,
        vectors: List[Dict[str, Any]],
        update: bool = False,
        embedding_models: Optional[Dict[str, str]] = None,
        s3_client=None,
        bucket=None,
        prefix=None,
    ):
        index_name = self._get_index_name(collection_name)
        fields = await self._get_fields(collection_name)
        ids = []
        for vector in vectors:
            if "id" in vector:
                _id = vector["id"]
            else:
                if update:
                    raise ValueError("ID is required for update operation.")
                _id = str(uuid.uuid4())
                vector["id"] = _id
            ids.append(_id)
            # convert numpy arrays to bytes
            for key, value in vector.items():
                if key in fields:
                    if fields[key]["type"] == "VECTOR":
                        if isinstance(value, np.ndarray) or isinstance(value, list):
                            vector[key] = np.array(value).astype("float32").tobytes()
                        else:
                            assert (
                                key in embedding_models
                            ), f"Embedding model not provided for field {key}."
                            embeddings = await self._embed_texts(
                                [vector[key]], embedding_models[key]
                            )
                            vector[key] = (
                                np.array(embeddings[0]).astype("float32").tobytes()
                            )

            await self._redis.hset(
                f"{index_name}:{_id}",
                mapping=vector,
            )

        if s3_client:
            assert bucket and prefix, "S3 bucket and prefix must be provided."
            await self._dump_vectors(vectors, fields, s3_client, bucket, prefix)

        logger.info(f"Added {len(vectors)} vectors to collection {collection_name}.")
        return ids

    def _format_docs(self, docs, index_name):
        formated = []
        for doc in docs:
            d = vars(doc)
            if "payload" in d:
                del d["payload"]
            if "id" in d:
                d["id"] = d["id"].replace(f"{index_name}:", "")
            formated.append(d)
        return formated

    def _convert_filters_to_hybrid_query(self, filters: dict, fields: dict) -> str:
        """
        Convert a filter dictionary to a Redis hybrid query string.

        Args:
            filters (dict): Dictionary of filters, e.g., {"type": "my-type", "year": [2011, 2012]}.

        Returns:
            str: Redis hybrid query string, e.g., "(@type:{my-type} @year:[2011 2012])".
        """

        conditions = []

        for field_name, value in filters.items():
            # Find the field type in the schema
            field_type = fields.get(field_name)
            field_type = (
                FIELD_TYPE_MAPPING.get(field_type["type"]) if field_type else None
            )

            if not field_type:
                raise ValueError(
                    f"Unknown field '{field_name}' in filters, available fields: {list(fields.keys())}"
                )

            # Sanitize the field name
            sanitized_field_name = sanitize_search_value(field_name)

            if field_type == TagField:
                # Use `{value}` for TagField
                if not isinstance(value, str):
                    raise ValueError(
                        f"TagField '{field_name}' requires a string value."
                    )
                sanitized_value = sanitize_search_value(value)
                conditions.append(f"@{sanitized_field_name}:{{{sanitized_value}}}")

            elif field_type == NumericField:
                # Use `[min max]` for NumericField
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(
                        f"NumericField '{field_name}' requires a list or tuple with two elements."
                    )
                min_val, max_val = value
                conditions.append(f"@{sanitized_field_name}:[{min_val} {max_val}]")

            elif field_type == TextField:
                # Use `"value"` for TextField
                if not isinstance(value, str):
                    raise ValueError(
                        f"TextField '{field_name}' requires a string value."
                    )
                if "*" in value:
                    assert value.endswith("*"), "Wildcard '*' must be at the end."
                    sanitized_value = sanitize_search_value(value)
                    conditions.append(f"@{sanitized_field_name}:{sanitized_value}")
                else:
                    sanitized_value = escape_redis_syntax(value)
                    conditions.append(f'@{sanitized_field_name}:"{sanitized_value}"')

            else:
                raise ValueError(f"Unsupported field type for '{field_name}'.")

        return " ".join(conditions)

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
    ):
        """
        Search vectors in a collection.
        """
        index_name = self._get_index_name(collection_name)
        collection_fields = await self._get_fields(collection_name)
        fields = {field["identifier"]: field for field in collection_fields.values()}
        all_fields = [field for field in collection_fields.keys()] + ["score"]
        # Generate embedding if text_query is provided
        if query:
            assert (
                len(query) == 1
            ), "Only one of text_query or vector_query can be provided."
            use_vector = list(query.keys())[0]
            assert (
                collection_fields[use_vector]["type"] == "VECTOR"
            ), f"Field {use_vector} is not a vector field."

            vector_query = query[use_vector]
            if isinstance(vector_query, np.ndarray):
                vector_query = vector_query.astype("float32").tobytes()
            elif isinstance(vector_query, list):
                vector_query = np.array(vector_query).astype("float32").tobytes()
            elif vector_query is not None:
                assert (
                    embedding_models and use_vector in embedding_models
                ), f"Embedding model not provided for field {use_vector}."
                embeddings = await self._embed_texts(
                    [vector_query],
                    embedding_models[use_vector],
                )
                vector_query = np.array(embeddings[0]).astype("float32").tobytes()

        # If service_embedding is provided, prepare KNN search query
        if vector_query is not None:
            query_params = {"vector": vector_query}
            knn_query = f"[KNN {limit} @{use_vector} $vector AS score]"
            # Combine filters into the query string
            if filters:
                filter_query = self._convert_filters_to_hybrid_query(
                    filters, collection_fields
                )
                query_string = (
                    f"(({filter_query}) ({extra_filter}))=>{knn_query}"
                    if extra_filter
                    else f"({filter_query})=>{knn_query}"
                )
            else:
                query_string = (
                    f"({extra_filter})=>{knn_query}"
                    if extra_filter
                    else f"*=>{knn_query}"
                )
        else:
            query_params = {}
            if filters:
                filter_query = self._convert_filters_to_hybrid_query(
                    filters, collection_fields
                )
                query_string = (
                    f"({filter_query}) ({extra_filter})"
                    if extra_filter
                    else f"{filter_query}"
                )
            else:
                query_string = extra_filter if extra_filter else "*"

        if return_fields is None:
            # exclude embedding
            return_fields = [
                field
                for field in all_fields
                if field == "score" or collection_fields[field]["type"] != "VECTOR"
            ]
        else:
            for field in return_fields:
                if field not in all_fields:
                    raise ValueError(f"Invalid field: {field}")
        if order_by is None:
            order_by = "score" if vector_query is not None else all_fields[0]
        else:
            if order_by not in all_fields:
                raise ValueError(f"Invalid order_by field: {order_by}")

        # Build the RedisSearch query
        query = (
            Query(query_string)
            .sort_by(order_by, asc=True)
            .paging(offset, limit)
            .dialect(2)
        )
        for field in fields.values():
            if field["identifier"] in return_fields:
                if field["type"] == "VECTOR":
                    query.return_field(field["identifier"], decode_field=False)
                else:
                    query.return_field(
                        field["identifier"], decode_field=True, encoding="utf-8"
                    )
        if "score" in return_fields:
            query.return_field("score")
        # Perform the search using the RedisSearch index
        results = await self._redis.ft(index_name).search(
            query, query_params=query_params
        )

        # Handle pagination
        if pagination:
            count_query = Query(query_string).paging(0, 0).dialect(2)
            count_results = await self._redis.ft(index_name).search(
                count_query, query_params=query_params
            )
            total_count = count_results.total
        else:
            total_count = None

        docs = self._format_docs(results.docs, index_name)

        if pagination:
            return {
                "items": docs,
                "total": total_count,
                "offset": offset,
                "limit": limit,
            }
        else:
            return docs

    async def remove_vectors(
        self,
        collection_name: str,
        ids: List[str],
        s3_client=None,
        bucket=None,
        prefix=None,
    ):
        index_name = self._get_index_name(collection_name)
        for id in ids:
            await self._redis.delete(f"{index_name}:{id}")
            if s3_client:
                key = f"{prefix}{id}.json"
                await s3_client.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Removed {len(ids)} vectors from collection {collection_name}.")

    async def get_vector(self, collection_name: str, id: str):
        index_name = self._get_index_name(collection_name)
        vector = await self._redis.hgetall(f"{index_name}:{id}")
        if not vector:
            raise ValueError(f"Vector {id} not found in collection {collection_name}.")

        return {key.decode("utf-8"): value for key, value in vector.items()}

    async def _get_max_search_results(self) -> int:
        try:
            result = await self._redis.execute_command(
                "FT.CONFIG", "GET", "MAXSEARCHRESULTS"
            )
            result = result[0]
            config = {
                result[i].decode(): result[i + 1].decode()
                for i in range(0, len(result), 2)
            }
            return int(config.get("MAXSEARCHRESULTS", 0))  # Default to 0 if not found
        except Exception as e:
            raise RuntimeError(f"Error retrieving MAXSEARCHRESULTS: {e}")

    async def list_vectors(
        self,
        collection_name: str,
        offset: int = 0,
        limit: int = 10,
        return_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        pagination: Optional[bool] = False,
    ):
        index_name = self._get_index_name(collection_name)
        # read the index settings
        fields = await self._get_fields(collection_name)
        return_fields = return_fields or [
            field["identifier"]
            for field in fields.values()
            if field["type"] != "VECTOR"
        ]
        if order_by is None:
            order_by = list(fields.keys())[0]
        else:
            if order_by not in fields.keys():
                raise ValueError(f"Invalid order_by field: {order_by}")

        query_string = "*"
        if limit is None:
            if not self._maximum_search_results:
                self._maximum_search_results = await self._get_max_search_results()
            limit = self._maximum_search_results
        if offset is None:
            offset = 0

        query = (
            Query(query_string)
            .paging(offset, limit)
            .dialect(2)
            .sort_by(order_by, asc=True)
        )
        for field in fields.values():
            if field["identifier"] in return_fields:
                if field["type"] == "VECTOR":
                    query.return_field(field["identifier"], decode_field=False)
                else:
                    query.return_field(
                        field["identifier"], decode_field=True, encoding="utf-8"
                    )
        results = await self._redis.ft(index_name).search(query)
        docs = self._format_docs(results.docs, index_name)

        if pagination:
            return {
                "items": docs,
                "duration": results.duration,
                "total": results.total,
                "offset": offset,
                "limit": limit,
            }
        else:
            return docs
