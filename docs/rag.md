# Tutorial: Using Vector Collections for Retrieval-Augmented Generation

Vector collections are a powerful feature in the `Artifact Manager`, enabling the efficient management and querying of high-dimensional data, such as embeddings for text, images, or other data types. These collections are especially useful in advanced applications like Retrieval-Augmented Generation (RAG) systems.

**Retrieval-Augmented Generation (RAG)** is a technique that enhances language models by providing them with external, task-specific knowledge retrieved from a database of embeddings. In a RAG system, user queries are transformed into embeddings that are compared with stored embeddings in a vector collection. The most relevant results, such as metadata or associated content, are retrieved and combined to form a context. This context is then passed to a language model, such as GPT, to generate a more accurate, knowledge-informed response.

The `Artifact Manager` makes implementing RAG straightforward by offering robust APIs for creating and managing vector collections, adding embeddings, and performing similarity searches. These capabilities allow you to integrate various embedding models (e.g., text embeddings from sentence transformers or image embeddings from CLIP) and use them to power applications that require dynamic retrieval and generation of information, such as:

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

Start by defining the structure of your vector collection, including vector fields and metadata.

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
    # Define the fields for the vector collection
    # This can include multiple vector fields and metadata fields.
    vector_collection_config = {
        "vector_fields": [
            {
                "type": "VECTOR",
                "name": "text_vector",
                "algorithm": "FLAT",
                "attributes": {
                    "TYPE": "FLOAT32",
                    "DIM": 768,
                    "DISTANCE_METRIC": "COSINE",
                },
            },
            {
                "type": "VECTOR",
                "name": "image_vector",
                "algorithm": "FLAT",
                "attributes": {
                    "TYPE": "FLOAT32",
                    "DIM": 512,
                    "DISTANCE_METRIC": "COSINE",
                },
            },
            {
                "type": "STRING",
                "name": "source",
            },
            {
                "type": "TAG",
                "name": "type",
            },
        ],
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

Once the collection is created, you can add vector embeddings generated from text or images.

```python
import numpy as np

async def add_vectors(artifact_manager, vector_collection_id):
    vectors = [
        {
            "text_vector": np.random.rand(768).tolist(),
            "image_vector": np.random.rand(512).tolist(),
            "source": "example1",
            "type": "text",
        },
        {
            "text_vector": np.random.rand(768).tolist(),
            "image_vector": np.random.rand(512).tolist(),
            "source": "example2",
            "type": "image",
        },
    ]

    await artifact_manager.add_vectors(
        artifact_id=vector_collection_id,
        vectors=vectors,
    )
    print("Vectors added to the collection.")
```

---

### Step 3: Search for Similar Embeddings

You can perform vector searches to find the most similar embeddings to a query vector.

```python
async def search_vectors(artifact_manager, vector_collection_id):
    query_vector = np.random.rand(768).tolist()  # Replace with your query embedding
    results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query={"text_vector": query_vector},
        limit=5,
    )
    print("Search results:", results)
```

---

### Step 4: Build a Retrieval-Augmented Generation (RAG) System

**RAG systems** enhance language models by integrating external knowledge. Below are two examples of building a RAG pipeline using vector collections: one for **text-based search** and another for **image-based search**.

---

#### Example 1: Text Embedding Search for RAG

This example demonstrates how to retrieve relevant text-based results and use them as context for a language model.

```python
import numpy as np
from openai import OpenAI

async def rag_pipeline_text(artifact_manager, vector_collection_id, query_text):
    # Initialize OpenAI API client
    openai = OpenAI(api_key="your_openai_api_key")

    # Generate a text embedding for the query (Replace with actual embedding generation logic)
    text_embedding = np.random.rand(768).tolist()

    # Search for similar text embeddings in the vector collection
    search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query={"text_vector": text_embedding},
        limit=3,
    )

    # Combine retrieved results into a context for RAG
    context = "\n".join(
        [result.get("source", "Unknown source") for result in search_results]
    )

    # Use OpenAI GPT model to generate a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Query: {query_text}\nContext:\n{context}"},
    ]

    completion = await openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract and print the response
    response = completion.choices[0].message.content
    print(f"Query: {query_text}\nContext:\n{context}\nResponse:\n{response}")

# Example usage for text
# await rag_pipeline_text(artifact_manager, "vector-collection-id", "Explain recursion.")
```

---

#### Example 2: Image Embedding Search for RAG

This example focuses on searching for images based on a query image embedding and retrieving relevant metadata for RAG.

```python
import numpy as np
from openai import OpenAI

async def rag_pipeline_image(artifact_manager, vector_collection_id, query_image_embedding):
    # Initialize OpenAI API client
    openai = OpenAI(api_key="your_openai_api_key")

    # Query the vector collection using the image embedding
    search_results = await artifact_manager.search_vectors(
        artifact_id=vector_collection_id,
        query={"image_vector": query_image_embedding},
        limit=3,
    )

    # Combine retrieved image metadata into a context
    context = "\n".join(
        [
            f"Image Source: {result.get('source', 'Unknown source')}, "
            f"Description: {result.get('description', 'No description')}"
            for result in search_results
        ]
    )

    # Use OpenAI GPT model to generate a response based on image metadata
    messages = [
        {"role": "system", "content": "You are a helpful assistant analyzing image contexts."},
        {"role": "user", "content": f"Query Image Context:\n{context}"},
    ]

    completion = await openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    # Extract and print the response
    response = completion.choices[0].message.content
    print(f"Query Image Context:\n{context}\nResponse:\n{response}")

```


### Tips for Implementing Vector Collections and RAG Systems

When using vector collections in the `Artifact Manager`, it’s important to consider the following best practices and tips to optimize memory usage and performance:

#### 1. **Design Efficient Vector Collections**

   - **Avoid Storing Large Data Directly in Vectors**: Vector collections are powered by Redis, which is an in-memory database. Storing large fields (e.g., images or lengthy text) as metadata in the vector collection can become prohibitively expensive.
   - **Use File Storage for Large Data**:
     - Add a unique `id` for each vector.
     - Use this `id` as the filename or reference for additional data (e.g., an image or document) and upload it to the same collection using the `put_file()` function.
     - This approach keeps your vector metadata lightweight while maintaining links to larger associated data.

#### 2. **Optimize Searches and Filters**

   - **Leverage Metadata Fields**: Use metadata fields to store searchable tags, categories, or attributes. These fields can be used to filter results and create subcollections.
   - **Efficient Retrieval**: Once vectors are retrieved via search, use their `id` to fetch associated files or additional data from the collection using the `get_file()` function.

#### 3. **Manage Memory and Workspace Activity**

   - **S3 Dumping for Inactive Workspaces**:
     - To save memory, the `Artifact Manager` dumps all vector data into S3 when a workspace becomes inactive. 
     - If the workspace is reactivated, the vector data will be reloaded into the vector engine (Redis). This process may take time.
   - **Wait for Readiness**:
     - Ensure that all vector collections are fully loaded before interacting with them by using:
       ```python
       await server.wait_until_ready()
       ```

#### 4. **Search Tips**

   - Use vector embeddings for similarity searches to find relevant results efficiently.
   - Combine vector-based retrieval with metadata filters for more refined results (e.g., filter by `type`, `tags`, or other fields).
   - When performing searches, fetch associated data or files using the `id` of the retrieved vectors.

#### Example: Lightweight and Efficient Design

1. **Add a Vector with an ID**:
   ```python
   vector = {
       "vector": np.random.rand(768).tolist(),
       "metadata": {"id": "image_123", "category": "nature"}
   }
   await artifact_manager.add_vectors(artifact_id=vector_collection_id, vectors=[vector])
   ```

2. **Upload Associated File**:
   ```python
   await artifact_manager.put_file(
       artifact_id=vector_collection_id,
       file_path="image_123.jpg",
       download_weight=0.1,
   )
   ```

3. **Search and Fetch File**:
   ```python
   search_results = await artifact_manager.search_vectors(
       artifact_id=vector_collection_id,
       query={"vector": query_vector},
       limit=1,
   )
   result_id = search_results[0]["metadata"]["id"]
   file_url = await artifact_manager.get_file(
       artifact_id=vector_collection_id,
       file_path=f"{result_id}.jpg"
   )
   print(f"File URL: {file_url}")
   ```

By following these tips, you can efficiently design and manage vector collections in the `Artifact Manager`, enabling high-performance RAG systems while optimizing memory usage and ensuring scalability.
