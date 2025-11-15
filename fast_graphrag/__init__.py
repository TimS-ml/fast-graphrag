"""Fast-GraphRAG: A high-performance Graph-based Retrieval-Augmented Generation system.

This package provides a comprehensive implementation of GraphRAG, which combines knowledge
graphs with retrieval-augmented generation to enable intelligent information extraction,
storage, and querying from unstructured text data.

The system processes documents through the following pipeline:
    1. Text chunking: Breaking documents into manageable pieces
    2. Information extraction: Identifying entities and relationships using LLMs
    3. Knowledge graph construction: Building a graph from extracted information
    4. Vector embedding: Creating embeddings for semantic search
    5. Storage: Persisting graphs, vectors, and chunks for retrieval
    6. Query processing: Answering questions using graph traversal and LLM generation

Main Components:
    GraphRAG: The primary class for creating and managing a GraphRAG system.
        Provides methods for inserting documents and querying the knowledge graph.

    QueryParam: Configuration parameters for controlling query behavior,
        including retrieval depth, ranking policies, and generation settings.

Typical Usage:
    >>> from fast_graphrag import GraphRAG
    >>> grag = GraphRAG(working_dir="./my_graphrag")
    >>> await grag.insert("Your document text here")
    >>> response = await grag.query("What is this document about?")

Exports:
    GraphRAG: Main class for GraphRAG operations
    QueryParam: Query configuration dataclass
"""

__all__ = ["GraphRAG", "QueryParam"]

from dataclasses import dataclass, field
from typing import Type

from fast_graphrag._llm import DefaultEmbeddingService, DefaultLLMService
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._base import BaseGraphUpsertPolicy
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._policies._ranking import RankingPolicy_TopK, RankingPolicy_WithThreshold
from fast_graphrag._services import (
    BaseChunkingService,
    BaseInformationExtractionService,
    BaseStateManagerService,
    DefaultChunkingService,
    DefaultInformationExtractionService,
    DefaultStateManagerService,
)
from fast_graphrag._storage import (
    DefaultGraphStorage,
    DefaultGraphStorageConfig,
    DefaultIndexedKeyValueStorage,
    DefaultVectorStorage,
    DefaultVectorStorageConfig,
)
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import TChunk, TEmbedding, TEntity, THash, TId, TIndex, TRelation

from ._graphrag import BaseGraphRAG, QueryParam


@dataclass
class GraphRAG(BaseGraphRAG[TEmbedding, THash, TChunk, TEntity, TRelation, TId]):
    """A Graph-based Retrieval-Augmented Generation system for intelligent document processing.

    GraphRAG combines knowledge graph construction with LLM-powered question answering to
    provide a powerful system for extracting, storing, and querying information from documents.

    The system automatically:
        - Chunks documents into processable segments
        - Extracts entities and relationships using LLMs
        - Builds and maintains a knowledge graph
        - Creates vector embeddings for semantic search
        - Answers questions by combining graph traversal with LLM generation

    Type Parameters:
        TEmbedding: Type for vector embeddings (e.g., list of floats)
        THash: Type for content hashing
        TChunk: Type for document chunks
        TEntity: Type for graph entities/nodes
        TRelation: Type for graph relationships/edges
        TId: Type for entity identifiers

    Attributes:
        config (Config): Configuration object containing all services, storage backends,
            and policies for the GraphRAG system. See Config class for details.
        llm_service (BaseLLMService): Service for LLM operations (text generation)
        embedding_service (BaseEmbeddingService): Service for creating vector embeddings
        chunking_service (BaseChunkingService): Service for splitting text into chunks
        information_extraction_service (BaseInformationExtractionService): Service for
            extracting entities and relationships from text
        state_manager (BaseStateManagerService): Manages graph state, storage, and retrieval

    Example:
        >>> from fast_graphrag import GraphRAG
        >>> # Create a GraphRAG instance with default configuration
        >>> grag = GraphRAG(working_dir="./my_data")
        >>>
        >>> # Insert documents into the knowledge graph
        >>> await grag.insert("Einstein developed the theory of relativity.")
        >>> await grag.insert("Marie Curie won two Nobel Prizes.")
        >>>
        >>> # Query the knowledge graph
        >>> response = await grag.query("Who were famous scientists?")
        >>> print(response)
        >>>
        >>> # Use custom configuration
        >>> custom_config = GraphRAG.Config(
        ...     llm_service=MyCustomLLM(),
        ...     embedding_service=MyCustomEmbedding()
        ... )
        >>> grag = GraphRAG(working_dir="./my_data", config=custom_config)
    """

    @dataclass
    class Config:
        """Configuration for the GraphRAG system.

        This class contains all configurable components of the GraphRAG pipeline, including
        services for processing, storage backends, and policies for ranking and graph updates.

        Service Classes (Process Pipeline):
            chunking_service_cls: Service class for splitting documents into chunks.
                Default: DefaultChunkingService (splits by token count with overlap)
            information_extraction_service_cls: Service class for extracting entities and
                relationships from text using LLMs.
                Default: DefaultInformationExtractionService
            state_manager_cls: Service class for managing graph state, storage, and retrieval.
                Default: DefaultStateManagerService

        Core Services (LLM and Embeddings):
            llm_service: Instance for LLM operations (text generation, entity extraction).
                Default: DefaultLLMService (uses OpenAI-compatible API)
            embedding_service: Instance for creating vector embeddings from text.
                Default: DefaultEmbeddingService (uses OpenAI-compatible embeddings)

        Storage Backends:
            graph_storage: Storage backend for the knowledge graph (nodes and edges).
                Default: DefaultGraphStorage with NetworkX in-memory graph
            entity_storage: Vector storage for entity embeddings (semantic search).
                Default: DefaultVectorStorage with in-memory FAISS index
            chunk_storage: Key-value storage for document chunks.
                Default: DefaultIndexedKeyValueStorage with in-memory dict

        Upsert Policies (Graph Construction):
            information_extraction_upsert_policy: Policy for upserting entities/relations
                during information extraction phase.
                Default: NodeUpsertPolicy_SummarizeDescription with
                    EdgeUpsertPolicy_UpsertIfValidNodes
            node_upsert_policy: Policy for handling node conflicts during graph updates.
                Default: NodeUpsertPolicy_SummarizeDescription (merges descriptions)
            edge_upsert_policy: Policy for handling edge conflicts during graph updates.
                Default: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM
                    (uses LLM to merge similar edges)

        Ranking Policies (Retrieval):
            entity_ranking_policy: Policy for ranking/filtering retrieved entities.
                Default: RankingPolicy_WithThreshold(threshold=0.005)
                    - Filters entities by similarity score threshold
            relation_ranking_policy: Policy for ranking/filtering retrieved relations.
                Default: RankingPolicy_TopK(top_k=64)
                    - Returns top 64 most relevant relations
            chunk_ranking_policy: Policy for ranking/filtering retrieved chunks.
                Default: RankingPolicy_TopK(top_k=8)
                    - Returns top 8 most relevant chunks

        Example:
            >>> # Use default configuration
            >>> config = GraphRAG.Config()
            >>>
            >>> # Customize specific components
            >>> config = GraphRAG.Config(
            ...     llm_service=MyCustomLLM(),
            ...     entity_ranking_policy=RankingPolicy_TopK(
            ...         RankingPolicy_TopK.Config(top_k=100)
            ...     )
            ... )
        """

        # Service classes for the processing pipeline
        # Handles text chunking: splits documents into overlapping segments for processing
        chunking_service_cls: Type[BaseChunkingService[TChunk]] = field(default=DefaultChunkingService)

        # Extracts entities and relationships from text chunks using LLM prompts
        information_extraction_service_cls: Type[BaseInformationExtractionService[TChunk, TEntity, TRelation, TId]] = (
            field(default=DefaultInformationExtractionService)
        )

        # Policy for inserting extracted entities/relations during information extraction
        # Default: Summarizes node descriptions and only upserts edges with valid nodes
        information_extraction_upsert_policy: BaseGraphUpsertPolicy[TEntity, TRelation, TId] = field(
            default_factory=lambda: DefaultGraphUpsertPolicy(
                config=NodeUpsertPolicy_SummarizeDescription.Config(),
                nodes_upsert_cls=NodeUpsertPolicy_SummarizeDescription,
                edges_upsert_cls=EdgeUpsertPolicy_UpsertIfValidNodes,
            )
        )

        # Manages the overall state including graph storage, embeddings, and checkpointing
        state_manager_cls: Type[BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]] = field(
            default=DefaultStateManagerService
        )

        # Core LLM service for text generation (extraction prompts, answer generation)
        llm_service: BaseLLMService = field(default_factory=lambda: DefaultLLMService())

        # Embedding service for converting text to vectors for semantic search
        embedding_service: BaseEmbeddingService = field(default_factory=lambda: DefaultEmbeddingService())

        # Storage backend for the knowledge graph (nodes=entities, edges=relations)
        # Default: NetworkX-based in-memory graph with serialization support
        graph_storage: BaseGraphStorage[TEntity, TRelation, TId] = field(
            default_factory=lambda: DefaultGraphStorage(DefaultGraphStorageConfig(node_cls=TEntity, edge_cls=TRelation))
        )

        # Vector storage for entity embeddings (used for semantic entity search)
        # Default: FAISS-based in-memory index with cosine similarity
        entity_storage: DefaultVectorStorage[TIndex, TEmbedding] = field(
            default_factory=lambda: DefaultVectorStorage(
                DefaultVectorStorageConfig()
            )
        )

        # Key-value storage for document chunks (indexed by content hash)
        # Default: In-memory dictionary storage
        chunk_storage: DefaultIndexedKeyValueStorage[THash, TChunk] = field(
            default_factory=lambda: DefaultIndexedKeyValueStorage(None)
        )

        # Ranking policy for entities: filters by similarity threshold (default: 0.005)
        # Only entities with similarity score >= threshold are retrieved
        entity_ranking_policy: RankingPolicy_WithThreshold = field(
            default_factory=lambda: RankingPolicy_WithThreshold(RankingPolicy_WithThreshold.Config(threshold=0.005))
        )

        # Ranking policy for relations: returns top K most similar (default: 64)
        # Limits the number of relationships retrieved during queries
        relation_ranking_policy: RankingPolicy_TopK = field(
            default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=64))
        )

        # Ranking policy for chunks: returns top K most similar (default: 8)
        # Limits the number of document chunks used for answer generation
        chunk_ranking_policy: RankingPolicy_TopK = field(
            default_factory=lambda: RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=8))
        )

        # Node upsert policy: how to handle duplicate entities
        # Default: Summarizes multiple descriptions into one using LLM
        node_upsert_policy: NodeUpsertPolicy_SummarizeDescription = field(
            default_factory=lambda: NodeUpsertPolicy_SummarizeDescription()
        )

        # Edge upsert policy: how to handle duplicate relationships
        # Default: Merges similar edges using LLM to combine their descriptions
        edge_upsert_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM = field(
            default_factory=lambda: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM()
        )

        def __post_init__(self):
            """Initialize the Config after dataclass initialization.

            This method is automatically called after the dataclass __init__ to perform
            additional setup. It configures the entity storage to use the correct embedding
            dimension from the embedding service.

            The embedding dimension must match between:
                - The embedding service (which generates vectors of a specific size)
                - The entity storage (which indexes vectors for similarity search)

            This ensures that entity embeddings generated by the embedding service can
            be properly stored and searched in the vector storage backend.
            """
            # Configure entity storage to match the embedding service's vector dimension
            self.entity_storage.embedding_dim = self.embedding_service.embedding_dim

    # Configuration object containing all services, storage, and policies
    config: Config = field(default_factory=Config)

    def __post_init__(self):
        """Initialize the GraphRAG instance after dataclass initialization.

        This method is automatically called after the dataclass __init__ to set up all
        the components of the GraphRAG system. It performs the following initialization:

        1. Service Initialization:
            - Sets up LLM service for text generation
            - Sets up embedding service for vector creation
            - Instantiates chunking service from the configured class
            - Instantiates information extraction service with upsert policy

        2. State Manager Setup:
            - Creates a workspace for data persistence (graph, vectors, chunks)
            - Configures checkpointing (keeps last n_checkpoints versions)
            - Wires together all storage backends (graph, entity, chunk)
            - Applies all ranking and upsert policies

        The state manager coordinates:
            - Graph storage: Persisting and querying the knowledge graph
            - Entity storage: Storing and searching entity embeddings
            - Chunk storage: Storing original document chunks
            - Ranking policies: Controlling retrieval quality and quantity
            - Upsert policies: Managing conflicts during graph updates

        Note:
            This method should not be called directly. It's automatically invoked
            during GraphRAG instantiation.
        """
        # Initialize core services from config
        self.llm_service = self.config.llm_service
        self.embedding_service = self.config.embedding_service

        # Instantiate chunking service (converts documents to chunks)
        self.chunking_service = self.config.chunking_service_cls()

        # Instantiate information extraction service with its upsert policy
        # This service extracts entities/relations and applies the policy during insertion
        self.information_extraction_service = self.config.information_extraction_service_cls(
            graph_upsert=self.config.information_extraction_upsert_policy
        )

        # Create the state manager: the central coordinator for all storage and retrieval
        # - workspace: Manages file-based persistence and checkpointing
        # - embedding_service: Used to generate embeddings for entities
        # - *_storage: Backend storage systems for graph, entities, and chunks
        # - *_ranking_policy: Controls what gets retrieved during queries
        # - *_upsert_policy: Controls how conflicts are resolved during updates
        self.state_manager = self.config.state_manager_cls(
            workspace=Workspace.new(self.working_dir, keep_n=self.n_checkpoints),
            embedding_service=self.embedding_service,
            graph_storage=self.config.graph_storage,
            entity_storage=self.config.entity_storage,
            chunk_storage=self.config.chunk_storage,
            entity_ranking_policy=self.config.entity_ranking_policy,
            relation_ranking_policy=self.config.relation_ranking_policy,
            chunk_ranking_policy=self.config.chunk_ranking_policy,
            node_upsert_policy=self.config.node_upsert_policy,
            edge_upsert_policy=self.config.edge_upsert_policy,
        )
