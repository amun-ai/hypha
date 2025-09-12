# Tutorial: Vector Search for Retrieval-Augmented Generation

Vector collections are a powerful feature in the `Artifact Manager`, enabling the efficient management and querying of high-dimensional data, such as embeddings for text, images, or other data types. These collections are especially useful in advanced applications like Retrieval-Augmented Generation (RAG) systems.

**Retrieval-Augmented Generation (RAG)** is a technique that enhances language models by providing them with external, task-specific knowledge retrieved from a database of embeddings. In a RAG system, user queries are transformed into embeddings that are compared with stored embeddings in a vector collection. The most relevant results, such as metadata or associated content, are retrieved and combined to form a context. This context is then passed to a language model, such as GPT, to generate a more accurate, knowledge-informed response.

The `Artifact Manager` supports multiple vector engine backends:

- **PgVector**: PostgreSQL-based vector storage (default for PostgreSQL databases)
- **S3Vector**: S3-based vector storage with Redis caching (recommended for SQLite or when using external S3 storage)

These capabilities allow you to integrate various embedding models (e.g., text embeddings from sentence transformers or image embeddings from CLIP) and use them to power applications that require dynamic retrieval and generation of information, such as:

- **Question-Answering Systems**: Retrieve relevant documents or information based on a query and use it to generate precise answers.
- **Knowledge-Augmented Chatbots**: Enhance chatbot responses by retrieving task-specific knowledge stored as embeddings.
- **Content Recommendation**: Retrieve similar items (text, images, or products) to provide personalized recommendations.

This tutorial will guide you through creating a vector collection, adding embeddings, and leveraging these features to build a RAG pipeline for both text-based and image-based applications.

---

## What is a Vector Collection?

A vector collection is a specialized artifact designed to store and manage vector embeddings, along with their associated metadata. These collections are commonly used in machine learning tasks, such as similarity search, clustering, and powering applications like RAG systems for natural language processing.

---

## Step-by-Step Guide to Using Vector Collections

### Step 1: Create a Vector Collection

Start by defining the structure of your vector collection. The configuration varies based on your database backend.

#### Option A: Using PgVector (PostgreSQL)

```python
import asyncio
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Artifact Manager
    server = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await server.get_service("public/artifact-manager")

    # Define the manifest and configuration for the vector collection
    vector_collection_manifest = {
        "name": "Text and Image Embeddings",
        "description": "A collection for managing embeddings of text and images",
    }
    
    # Configuration for PgVector (default for PostgreSQL)
    vector_collection_config = {
        # PgVector is used automatically with PostgreSQL
        # Optional: specify embedding model for automatic embedding generation
        "embedding_model": "all-MiniLM-L6-v2",  # Optional, for automatic text embedding
        "dimension": 384,  # Vector dimension (must match your embedding model output)
        "distance_metric": "cosine",  # Options: "cosine", "l2", "inner_product"
    }

    # Create the vector collection
    vector_collection = await artifact_manager.create(
        type="vector-collection",
        manifest=vector_collection_manifest,
        config=vector_collection_config,
    )
    print(f"Vector Collection created with ID: {vector_collection.id}")

asyncio.run(main())
```

#### Option B: Using S3Vector (SQLite or External S3)

```python
import asyncio
from hypha_rpc import connect_to_server

SERVER_URL = "https://hypha.aicell.io"

async def main():
    # Connect to the Artifact Manager
    server = await connect_to_server({"name": "test-client", "server_url": SERVER_URL})
    artifact_manager = await server.get_service("public/artifact-manager")

    # Define the manifest and configuration for the vector collection
    vector_collection_manifest = {
        "name": "Text and Image Embeddings",
        "description": "A collection for managing embeddings of text and images",
    }
    
    # Configuration for S3Vector (required for SQLite)
    vector_collection_config = {
        "vector_engine": "s3vector",  # Required for SQLite databases
        "embedding_model": "all-MiniLM-L6-v2",  # Optional, for automatic text embedding
        "dimension": 384,  # Vector dimension
        "distance_metric": "cosine",  # Options: "cosine", "l2", "inner_product"
        
        # Optional: Custom S3 configuration
        "s3_config": {
            "endpoint_url": "https://s3.amazonaws.com",  # Optional custom S3 endpoint
            "bucket": "my-vectors",  # Optional custom bucket
            "region_name": "us-east-1",
        }
    }

    # Create the vector collection
    vector_collection = await artifact_manager.create(
        type="vector-collection",
        manifest=vector_collection_manifest,
        config=vector_collection_config,
    )
    print(f"Vector Collection created with ID: {vector_collection.id}")

asyncio.run(main())
```

---

### Step 2: Add Vectors to the Collection

Once the collection is created, you can add vector embeddings. The format depends on whether you provide pre-computed embeddings or let the system generate them.

#### With Pre-computed Embeddings

```python
import numpy as np

async def add_vectors(artifact_manager, vector_collection_id):
    # Add vectors with pre-computed embeddings
    vectors = [
        {
            "id": "doc1",  # Unique identifier
            "vector": np.random.rand(384).tolist(),  # Pre-computed embedding
            "metadata": {
                "source": "example1.txt",
                "type": "text",
                "content": "This is the actual text content"
            }
        },
        {
            "id": "doc2",
            "vector": np.random.rand(384).tolist(),
            "metadata": {
                "source": "example2.txt", 
                "type": "text",
                "content": "Another text document"
            }
        },
    ]

    result = await artifact_manager.add_vectors(
        artifact_id=vector_collection_id,
        vectors=vectors,
    )
    print(f"Added {len(result)} vectors to the collection")
```

#### With Automatic Embedding Generation

If you configured an `embedding_model`, you can provide text directly:

```python
async def add_text_vectors(artifact_manager, vector_collection_id):
    # Add text that will be automatically embedded
    vectors = [
        {
            "id": "doc1",
            "text": "This is the actual text that will be embedded",  # Text for embedding
            "metadata": {
                "source": "document1.pdf",
                "page": 1,
                "type": "pdf"
            }
        },
        {
            "id": "doc2",
            "text": "Another piece of text to be embedded automatically",
            "metadata": {
                "source": "document2.pdf",
                "page": 5,
                "type": "pdf"
            }
        },
    ]

    result = await artifact_manager.add_vectors(
        artifact_id=vector_collection_id,
        vectors=vectors,
    )
    print(f"Added {len(result)} vectors with automatic embedding")
```

---

### Step 3: Search for Similar Embeddings

You can perform vector searches using either pre-computed query vectors or text queries (if embedding model is configured).

#### Search with Pre-computed Query Vector

```python
async def search_vectors(artifact_manager, vector_collection_id):
    query_vector = np.random.rand(384).tolist()  # Your query embedding
    
    results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query_vector=query_vector,
        limit=5,
        include_vectors=False,  # Set to True to include vectors in results
    )
    
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}")
        print(f"Metadata: {result.get('metadata', {})}")
```

#### Search with Text Query (Automatic Embedding)

```python
async def search_by_text(artifact_manager, vector_collection_id):
    # If embedding_model is configured, you can search with text directly
    results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query_text="Find documents about machine learning",  # Text query
        limit=5,
    )
    
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}")
        print(f"Content: {result.get('metadata', {}).get('content', 'N/A')}")
```

#### Search with Metadata Filters

```python
async def search_with_filters(artifact_manager, vector_collection_id):
    # Combine vector search with metadata filtering
    results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query_vector=np.random.rand(384).tolist(),
        filters={"type": "pdf", "page": {"$gte": 5}},  # Filter by metadata
        limit=5,
    )
    print("Filtered search results:", results)
```

---

### Step 4: Build a Retrieval-Augmented Generation (RAG) System

**RAG systems** enhance language models by integrating external knowledge. Below are two examples of building a RAG pipeline using vector collections: one for **text-based search** and another for **image-based search**.

---

#### Example 1: Text Embedding Search for RAG

This example demonstrates how to retrieve relevant text-based results and use them as context for a language model.

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer

async def rag_pipeline_text(artifact_manager, vector_collection_id, query_text):
    # Initialize OpenAI API client
    openai = OpenAI(api_key="your_openai_api_key")
    
    # Option 1: Generate embedding manually
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_text).tolist()
    
    # Search for similar documents
    search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query_vector=query_embedding,
        limit=3,
    )
    
    # OR Option 2: If embedding_model is configured, use text directly
    # search_results = await artifact_manager.search_vectors(
    #     artifact_id=vector_collection_id,
    #     query_text=query_text,
    #     limit=3,
    # )

    # Combine retrieved results into context
    context_parts = []
    for result in search_results:
        metadata = result.get("metadata", {})
        content = metadata.get("content", "")
        source = metadata.get("source", "Unknown")
        context_parts.append(f"[Source: {source}]\n{content}")
    
    context = "\n\n".join(context_parts)

    # Use OpenAI GPT model to generate a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
        {"role": "user", "content": f"Query: {query_text}\n\nContext:\n{context}\n\nPlease answer the query based on the context above."},
    ]

    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract and return the response
    response = completion.choices[0].message.content
    return {
        "query": query_text,
        "context": context,
        "response": response,
        "sources": [r.get("metadata", {}).get("source") for r in search_results]
    }

# Example usage
# result = await rag_pipeline_text(artifact_manager, "vector-collection-id", "What is machine learning?")
```

---

#### Example 2: Multi-Modal Fusion for Combined Text and Image Search

Vector collections support a single vector field per collection. To handle multi-modal data (text + images), you need to fuse embeddings into a single vector. Here's how to properly combine different modalities:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

class MultiModalEmbedder:
    """Fuse text and image embeddings into a single vector for multi-modal search."""
    
    def __init__(self, text_dim=384, image_dim=512, text_weight=0.7):
        """
        Initialize multi-modal embedder.
        
        Args:
            text_dim: Dimension of text embeddings
            image_dim: Dimension of image embeddings  
            text_weight: Weight for text (0-1), image weight = 1 - text_weight
        """
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.total_dim = text_dim + image_dim
        
        # Weights for combining modalities
        self.text_weight = text_weight
        self.image_weight = 1 - text_weight
        
    def normalize(self, vector):
        """L2 normalize a vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def fuse_embeddings(self, text=None, image=None):
        """
        Fuse text and image embeddings using weighted concatenation.
        
        The formula: fused = [√α · t̂ ; √(1-α) · î]
        where t̂ and î are unit-normalized embeddings, α is text weight.
        """
        # Initialize with zeros
        fused = np.zeros(self.total_dim)
        
        # Process text embedding
        if text is not None:
            text_emb = self.text_model.encode(text)
            text_emb = self.normalize(text_emb)  # Unit normalize
            # Scale by square root of weight to preserve cosine contribution
            text_emb = np.sqrt(self.text_weight) * text_emb
            fused[:self.text_dim] = text_emb
        
        # Process image embedding
        if image is not None:
            inputs = self.image_processor(images=image, return_tensors="pt")
            image_features = self.image_model.get_image_features(**inputs)
            image_emb = image_features[0].detach().numpy()
            image_emb = self.normalize(image_emb)  # Unit normalize
            # Scale by square root of weight
            image_emb = np.sqrt(self.image_weight) * image_emb
            fused[self.text_dim:] = image_emb[:self.image_dim]
        
        return fused.tolist()

async def create_multimodal_collection(artifact_manager):
    """Create a collection for multi-modal embeddings."""
    
    # Calculate total dimension for fused embeddings
    text_dim = 384  # all-MiniLM-L6-v2
    image_dim = 512  # CLIP
    total_dim = text_dim + image_dim  # 896
    
    collection = await artifact_manager.create(
        type="vector-collection",
        manifest={
            "name": "Multi-Modal Collection",
            "description": "Fused text and image embeddings"
        },
        config={
            "dimension": total_dim,
            "distance_metric": "cosine",
            # Note: No embedding_model since we handle fusion manually
        }
    )
    
    return collection["id"], MultiModalEmbedder(text_dim, image_dim)

async def add_multimodal_documents(artifact_manager, collection_id, embedder):
    """Add documents with fused text/image embeddings."""
    
    documents = []
    
    # Text-only document
    documents.append({
        "id": "doc1",
        "vector": embedder.fuse_embeddings(
            text="Introduction to machine learning algorithms",
            image=None  # No image for this doc
        ),
        "metadata": {
            "type": "text",
            "source": "ml_intro.txt",
            "has_image": False
        }
    })
    
    # Image-only document  
    # documents.append({
    #     "id": "img1",
    #     "vector": embedder.fuse_embeddings(
    #         text=None,  # No text
    #         image=image_data  # Your image data
    #     ),
    #     "metadata": {
    #         "type": "image",
    #         "source": "diagram.png",
    #         "has_text": False
    #     }
    # })
    
    # Combined text + image document
    # documents.append({
    #     "id": "multi1",
    #     "vector": embedder.fuse_embeddings(
    #         text="Neural network architecture diagram",
    #         image=image_data  # Corresponding image
    #     ),
    #     "metadata": {
    #         "type": "multimodal",
    #         "source": "nn_architecture.pdf",
    #         "has_text": True,
    #         "has_image": True
    #     }
    # })
    
    await artifact_manager.add_vectors(
        artifact_id=collection_id,
        vectors=documents
    )

async def search_multimodal(artifact_manager, collection_id, embedder, query_text=None, query_image=None):
    """Search using fused query embedding."""
    
    # Create fused query embedding
    query_vector = embedder.fuse_embeddings(
        text=query_text,
        image=query_image
    )
    
    # Search with the fused vector
    results = await artifact_manager.search_vectors(
        artifact_id=collection_id,
        query_vector=query_vector,
        limit=5
    )
    
    return results

# Example usage
# collection_id, embedder = await create_multimodal_collection(artifact_manager)
# await add_multimodal_documents(artifact_manager, collection_id, embedder)
# results = await search_multimodal(
#     artifact_manager, collection_id, embedder,
#     query_text="machine learning",
#     query_image=None
# )
```

**Key Points for Multi-Modal Fusion:**

1. **Fixed Dimension**: Choose a single vector dimension that combines both modalities (e.g., 384 + 512 = 896)

2. **Normalization**: Always unit-normalize each embedding before fusion: `ê = e/||e||`

3. **Weighted Fusion**: Scale by square root of weights to preserve cosine similarity:
   - Formula: `fused = [√α · text_normalized ; √(1-α) · image_normalized]`
   - Example with α=0.7: text gets √0.7 ≈ 0.84, image gets √0.3 ≈ 0.55

4. **Consistency**: Apply the same transformation at both indexing and query time

5. **Missing Modalities**: Fill missing parts with zeros (text-only, image-only, or both)


### Tips for Implementing Vector Collections and RAG Systems

When using vector collections in the `Artifact Manager`, it's important to consider the following best practices and tips to optimize memory usage and performance:

#### 1. **Choose the Right Vector Engine**

   - **PgVector (PostgreSQL)**: 
     - Best for production deployments with PostgreSQL
     - Supports native SQL queries alongside vector operations
     - Efficient for moderate-sized collections (millions of vectors)
     - Automatic with PostgreSQL databases
   
   - **S3Vector (S3-based storage)**:
     - Required for SQLite databases
     - Excellent for very large collections (billions of vectors)
     - Lower memory footprint (stores vectors in S3)
     - Good for cost-sensitive deployments
     - Supports custom S3 endpoints (MinIO, AWS, etc.)

#### 2. **Design Efficient Vector Collections**

   - **Keep Metadata Lightweight**: 
     - Store only essential searchable fields in metadata
     - For PgVector: Metadata is stored in PostgreSQL (more flexible)
     - For S3Vector: Metadata is cached in Redis (keep it small)
   
   - **Use File Storage for Large Data**:
     - Add a unique `id` for each vector
     - Store large content (documents, images) as files using `put_file()`
     - Reference files by ID in vector metadata
     - This keeps vector operations fast and memory-efficient

#### 3. **Optimize Searches and Filters**

   - **Use Appropriate Search Parameters**:
     - `limit`: Control number of results (default: 10)
     - `offset`: For pagination through large result sets
     - `include_vectors`: Set to False unless you need the actual vectors
     - `filters`: Use metadata filters to narrow search scope
   
   - **Batch Operations**: 
     - Add multiple vectors in a single call for better performance
     - Use list operations to retrieve multiple vectors by ID

#### 4. **Embedding Model Configuration**

   - **Automatic Embedding Generation**:
     - Configure `embedding_model` in collection config
     - Supports models from Sentence Transformers library
     - Common models: `all-MiniLM-L6-v2` (384d), `all-mpnet-base-v2` (768d)
   
   - **Consistent Dimensions**:
     - Ensure `dimension` parameter matches your embedding model output
     - All vectors in a collection must have the same dimension

#### 5. **Manage Memory and Performance**

   - **For PgVector**:
     - Vectors stored in PostgreSQL database
     - Good for collections up to millions of vectors
     - Supports HNSW index for faster searches
   
   - **For S3Vector**:
     - Vectors stored in S3, indexed in memory when needed
     - Excellent for very large collections
     - Automatic caching with Redis for frequently accessed vectors
     - Consider S3 egress costs for frequent searches

#### 6. **Workspace Management**

   - **Collection Persistence**:
     - PgVector: Data persists in PostgreSQL
     - S3Vector: Data persists in S3, index rebuilt on workspace activation
   
   - **Wait for Readiness** (especially important for S3Vector):
     ```python
     # Wait for collections to be loaded
     import time
     timeout = 30
     start_time = time.time()
     
     while True:
         try:
             # Try to access the collection
             count = await artifact_manager.count_vectors(
                 artifact_id=vector_collection_id
             )
             if count >= 0:
                 break
         except Exception:
             pass
         
         if time.time() - start_time > timeout:
             raise TimeoutError("Collection failed to load")
         await asyncio.sleep(1)
     ```

#### 7. **Search Best Practices**

   - **Hybrid Search**: Combine vector similarity with metadata filtering
   - **Result Caching**: Cache frequent queries to reduce computation
   - **Incremental Indexing**: Add new vectors without rebuilding entire index
   - **Monitoring**: Track search latency and adjust collection size/configuration

### Example: Complete RAG Implementation

Here's a complete example showing how to build an efficient RAG system:

```python
import asyncio
from hypha_rpc import connect_to_server
from sentence_transformers import SentenceTransformer
import numpy as np

async def create_rag_system():
    # Connect to Hypha
    server = await connect_to_server({
        "name": "rag-client",
        "server_url": "https://hypha.aicell.io"
    })
    artifact_manager = await server.get_service("public/artifact-manager")
    
    # Step 1: Create collection with appropriate engine
    collection_config = {
        "vector_engine": "s3vector",  # For SQLite or large-scale deployments
        "embedding_model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "distance_metric": "cosine"
    }
    
    collection = await artifact_manager.create(
        type="vector-collection",
        manifest={
            "name": "Knowledge Base",
            "description": "RAG knowledge base"
        },
        config=collection_config
    )
    
    # Step 2: Add documents with lightweight metadata
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence...",
            "metadata": {
                "title": "Introduction to ML",
                "category": "tutorial",
                "doc_id": "doc1"
            }
        },
        {
            "id": "doc2", 
            "text": "Neural networks are inspired by biological neurons...",
            "metadata": {
                "title": "Neural Networks",
                "category": "tutorial",
                "doc_id": "doc2"
            }
        }
    ]
    
    # Add vectors (embeddings generated automatically)
    await artifact_manager.add_vectors(
        artifact_id=collection["id"],
        vectors=documents
    )
    
    # Store full documents as files (for large content)
    for doc in documents:
        await artifact_manager.put_file(
            artifact_id=collection["id"],
            file_path=f"{doc['id']}.txt",
            content=doc["text"]  # Full document content
        )
    
    # Step 3: Search and retrieve
    async def search_and_generate(query):
        # Search using text query (embedding generated automatically)
        results = await artifact_manager.search_vectors(
            artifact_id=collection["id"],
            query_text=query,
            limit=3,
            filters={"category": "tutorial"}  # Optional filtering
        )
        
        # Retrieve full documents
        full_texts = []
        for result in results:
            doc_id = result["metadata"]["doc_id"]
            file_url = await artifact_manager.get_file(
                artifact_id=collection["id"],
                file_path=f"{doc_id}.txt"
            )
            # In production, fetch content from file_url
            full_texts.append({
                "title": result["metadata"]["title"],
                "score": result["score"],
                "content": "..."  # Retrieved from file_url
            })
        
        return full_texts
    
    # Use the system
    results = await search_and_generate("What is machine learning?")
    print(f"Found {len(results)} relevant documents")
    
    return collection["id"]

# Run the example
# asyncio.run(create_rag_system())
```

### Performance Comparison

| Aspect | PgVector | S3Vector |
|--------|----------|----------|
| **Storage Location** | PostgreSQL | S3 + Redis cache |
| **Max Collection Size** | Millions | Billions |
| **Memory Usage** | Higher | Lower |
| **Query Speed** | Faster | Good with caching |
| **Setup Complexity** | Simple (auto) | Requires config |
| **Best For** | Production with PostgreSQL | SQLite or very large collections |
| **Cost** | Database storage | S3 storage + egress |

By following these guidelines and choosing the appropriate vector engine for your use case, you can build efficient, scalable RAG systems using the Hypha Artifact Manager.
