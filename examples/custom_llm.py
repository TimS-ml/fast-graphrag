"""Example usage of GraphRAG with custom LLM and Embedding services compatible with the OpenAI API.

This module demonstrates how to configure and use fast-graphrag with any LLM and embedding service
that provides OpenAI API-compatible interfaces. This is useful for:
- Using Azure OpenAI instead of standard OpenAI
- Using open-source models through OpenAI-compatible endpoints (like Ollama, vLLM, etc.)
- Using any other API service that implements the OpenAI API specification
- Separating LLM and embedding services from different providers

Module Overview:
    The example shows how to instantiate a GraphRAG instance with custom OpenAI API-compatible
    services for both language model and text embedding tasks. This allows flexibility in choosing
    which models and providers to use for knowledge graph construction and querying.

Setup Instructions:
    1. Install required dependencies: pip install fast-graphrag python-dotenv instructor
    2. Create a .env file with your API credentials
    3. Update the configuration parameters (model names, API URLs, API keys, etc.)
    4. Configure DOMAIN, QUERIES, and ENTITY_TYPES according to your use case
    5. Place your document/text data in the working directory or use grag.insert() to add data
    6. Run the script to build the knowledge graph and perform queries

Configuration:
    - working_dir: Directory where the knowledge graph index will be stored
    - domain: A description of the domain/subject matter for context-aware processing
    - example_queries: Sample queries that help optimize the entity extraction process
    - entity_types: List of entity types relevant to your domain
"""

from typing import List

import instructor
from dotenv import load_dotenv

from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

# Load environment variables from .env file (contains API keys and credentials)
load_dotenv()

# === Configuration Parameters ===
# DOMAIN: Describes the subject matter or context for the knowledge graph
# This helps the LLM understand what types of information to extract and focus on
DOMAIN = ""

# QUERIES: Example queries that represent typical use cases for this domain
# These are used to optimize the entity and relationship extraction process
QUERIES: List[str] = []

# ENTITY_TYPES: List of entity types to extract from documents
# Examples: ["Person", "Organization", "Location", "Product", etc.]
ENTITY_TYPES: List[str] = []

# === Working Directory Setup ===
# This directory will store the vector index, entities, relationships, and other graph data
working_dir = "./examples/ignore/hp"

# === Initialize GraphRAG with Custom OpenAI-Compatible Services ===
grag = GraphRAG(
    working_dir=working_dir,
    domain=DOMAIN,
    example_queries="\n".join(QUERIES),
    entity_types=ENTITY_TYPES,
    config=GraphRAG.Config(
        # LLM Service: Handles language understanding and generation tasks
        # This can be any OpenAI API-compatible service (Azure, Ollama, vLLM, etc.)
        llm_service=OpenAILLMService(
            model="your-llm-model",  # Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            base_url="llm.api.url.com",  # API endpoint URL for the LLM service
            api_key="your-api-key",  # API key for authentication
            mode=instructor.Mode.JSON,  # Response format mode for structured outputs
            api_version="your-llm-api_version",  # API version (required for Azure)
            client="openai or azure"  # Client type: "openai" for standard OpenAI, "azure" for Azure
        ),
        # Embedding Service: Handles text-to-vector conversion for semantic search
        # This can be any OpenAI API-compatible embedding service
        embedding_service=OpenAIEmbeddingService(
            model="your-embedding-model",  # Embedding model identifier (e.g., "text-embedding-3-small")
            base_url="emb.api.url.com",  # API endpoint URL for the embedding service
            api_key="your-api-key",  # API key for authentication
            embedding_dim=512,  # Dimensionality of the output embeddings (must match the model)
            api_version="your-llm-api_version",  # API version (required for Azure)
            client="openai or azure"  # Client type: "openai" or "azure"
        ),
    ),
)

# === Usage Example ===
# After setting up the configuration:
# 1. Insert documents: grag.insert("your document text here")
# 2. Query the knowledge graph: result = grag.query("your question")
# 3. Access the response: print(result.response)
