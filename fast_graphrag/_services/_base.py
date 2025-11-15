"""Base service classes for the fast-graphrag document processing pipeline.

This module defines the abstract base classes for the three core services that power
the GraphRAG system:

1. BaseChunkingService: Handles document segmentation into manageable chunks
2. BaseInformationExtractionService: Extracts entities and relationships using LLMs
3. BaseStateManagerService: Orchestrates storage, retrieval, and ranking operations

These base classes establish the contract that concrete implementations must follow,
enabling a modular and extensible architecture for the knowledge graph construction
and query pipeline.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, List, Optional, Type

from scipy.sparse import csr_matrix

from fast_graphrag._llm import BaseEmbeddingService, BaseLLMService
from fast_graphrag._policies._base import (
    BaseEdgeUpsertPolicy,
    BaseGraphUpsertPolicy,
    BaseNodeUpsertPolicy,
    BaseRankingPolicy,
)
from fast_graphrag._storage import BaseBlobStorage, BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import (
    GTChunk,
    GTEdge,
    GTEmbedding,
    GTHash,
    GTId,
    GTNode,
    TContext,
    TDocument,
    TIndex,
)


@dataclass
class BaseChunkingService(Generic[GTChunk]):
    """Base class for document chunking services.

    This service is responsible for the first stage of the document processing pipeline:
    breaking down raw documents into smaller, manageable chunks. Chunking is essential
    for both LLM context window limitations and for creating granular retrieval units.

    The chunking strategy affects downstream processing:
    - Smaller chunks: More precise retrieval but may lose context
    - Larger chunks: Better context preservation but less precise matching
    - Overlapping chunks: Ensures no information is lost at boundaries

    Type Parameters:
        GTChunk: The generic chunk type that will be produced by this service.
    """

    def __post__init__(self):
        """Post-initialization hook for subclasses to set up internal state."""
        pass

    async def extract(self, data: Iterable[TDocument]) -> Iterable[Iterable[GTChunk]]:
        """Extract and deduplicate chunks from raw documents.

        This method processes input documents and segments them into chunks according
        to the implementation's chunking strategy. Each document produces a sequence
        of chunks, and duplicates are removed to avoid redundant processing.

        Args:
            data: An iterable of TDocument objects containing raw text and metadata.
                  Each document will be independently chunked.

        Returns:
            An iterable of iterables, where each inner iterable contains the chunks
            extracted from a single document. The structure preserves document boundaries,
            allowing downstream services to track chunk provenance.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError


@dataclass
class BaseInformationExtractionService(Generic[GTChunk, GTNode, GTEdge, GTId]):
    """Base class for LLM-powered entity and relationship extraction.

    This service represents the second stage of the document processing pipeline:
    converting unstructured text chunks into structured knowledge graph elements
    (entities and relationships). It leverages Large Language Models to identify
    and extract meaningful information.

    The extraction process typically involves:
    1. Sending chunks to an LLM with specialized prompts
    2. Parsing structured responses containing entities and relationships
    3. Optional "gleaning" steps for iterative refinement
    4. Merging extracted graphs from multiple chunks
    5. Resolving conflicts and duplicates via graph upsert policies

    Type Parameters:
        GTChunk: The chunk type to extract from.
        GTNode: The node/entity type to extract.
        GTEdge: The edge/relationship type to extract.
        GTId: The identifier type for graph elements.

    Attributes:
        graph_upsert: Policy for merging and deduplicating entities and relationships.
        max_gleaning_steps: Number of iterative refinement passes the LLM can make
            to extract additional entities it may have missed initially.
    """

    graph_upsert: BaseGraphUpsertPolicy[GTNode, GTEdge, GTId]
    max_gleaning_steps: int = 0

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[GTChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[GTNode, GTEdge, GTId]]]]:
        """Extract entities and relationships from chunked documents using an LLM.

        This method processes chunks in parallel, extracting structured knowledge
        graph elements from each document. It returns futures to allow for
        asynchronous processing of multiple documents concurrently.

        Args:
            llm: The language model service to use for extraction.
            documents: Nested iterable where each inner iterable contains chunks
                from a single document. Document boundaries are preserved.
            prompt_kwargs: Template variables for the extraction prompts (e.g.,
                domain-specific instructions, examples, entity type descriptions).
            entity_types: List of valid entity types to extract (e.g., "PERSON",
                "ORGANIZATION", "LOCATION"). Entities not matching these types
                may be filtered or marked as "UNKNOWN".

        Returns:
            A list of futures, one per document. Each future resolves to either:
            - A BaseGraphStorage containing extracted entities and relationships
            - None if extraction failed for that document

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from a user query for retrieval purposes.

        This method identifies entities mentioned in a search query, enabling
        entity-focused retrieval from the knowledge graph. It categorizes entities
        as either "named" (specific named entities) or "generic" (conceptual entities).

        Args:
            llm: The language model service to use for entity extraction.
            query: The user's search query string.
            prompt_kwargs: Template variables for the query entity extraction prompt.

        Returns:
            A dictionary with two keys:
            - "named": List of specific named entities (e.g., "Albert Einstein")
            - "generic": List of conceptual/generic entities (e.g., "physicist")

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError


@dataclass
class BaseStateManagerService(Generic[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding]):
    """Orchestrates storage, retrieval, and ranking operations for the knowledge graph.

    This service is the central coordinator of the GraphRAG system, managing the entire
    lifecycle of knowledge graph construction and query processing. It acts as the bridge
    between extraction services and storage backends, handling:

    1. **Insertion Pipeline**:
       - Deduplicates chunks to avoid redundant processing
       - Upserts entities and relationships into graph storage
       - Computes and stores entity embeddings for semantic search
       - Builds mapping matrices (entities→relationships, relationships→chunks)
       - Performs entity deduplication via similarity matching

    2. **Query Pipeline**:
       - Extracts entities from user queries
       - Performs multi-stage retrieval and ranking:
         a. Entity retrieval via vector similarity
         b. Entity ranking via graph traversal
         c. Relationship ranking via entity scores
         d. Chunk ranking via relationship scores
       - Returns scored context for answer generation

    3. **Storage Coordination**:
       - Manages three primary storage backends (graph, vector, key-value)
       - Maintains auxiliary blob storage for mapping matrices
       - Handles transactional operations (start/done phases)
       - Supports workspace isolation for multi-tenant scenarios

    Type Parameters:
        GTNode: The node/entity type stored in the graph.
        GTEdge: The edge/relationship type stored in the graph.
        GTHash: The hash type for chunk identification.
        GTChunk: The chunk type stored in key-value storage.
        GTId: The identifier type for graph elements.
        GTEmbedding: The embedding vector type for semantic search.

    Attributes:
        workspace: Isolated namespace for storage operations, enabling multi-tenancy.
        graph_storage: Stores the knowledge graph (entities and relationships).
        entity_storage: Vector database for entity embeddings (enables semantic search).
        chunk_storage: Key-value store for text chunks indexed by hash.
        embedding_service: Service for generating vector embeddings from text.
        node_upsert_policy: Strategy for merging duplicate entities.
        edge_upsert_policy: Strategy for merging duplicate relationships.
        entity_ranking_policy: Scoring policy for entity relevance.
        relation_ranking_policy: Scoring policy for relationship relevance.
        chunk_ranking_policy: Scoring policy for chunk relevance.
        node_specificity: If True, weight entities by document frequency (IDF-like).
        blob_storage_cls: Storage class for sparse matrices (mappings).
    """

    workspace: Optional[Workspace] = field()

    # Primary storage backends for different data types
    graph_storage: BaseGraphStorage[GTNode, GTEdge, GTId] = field()
    entity_storage: BaseVectorStorage[TIndex, GTEmbedding] = field()
    chunk_storage: BaseIndexedKeyValueStorage[GTHash, GTChunk] = field()

    # Embedding generation service
    embedding_service: BaseEmbeddingService = field()

    # Policies for handling duplicates and conflicts
    node_upsert_policy: BaseNodeUpsertPolicy[GTNode, GTId] = field()
    edge_upsert_policy: BaseEdgeUpsertPolicy[GTEdge, GTId] = field()

    # Policies for ranking retrieved elements
    entity_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))
    relation_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))
    chunk_ranking_policy: BaseRankingPolicy = field(default_factory=lambda: BaseRankingPolicy(None))

    # Advanced retrieval options
    node_specificity: bool = field(default=False)

    # Storage for mapping matrices (entities→relationships, relationships→chunks)
    blob_storage_cls: Type[BaseBlobStorage[csr_matrix]] = field(default=BaseBlobStorage)

    async def insert_start(self) -> None:
        """Prepare all storage backends for insertion operations.

        This method initializes the storage backends for a new batch of data insertion.
        It typically:
        - Opens database connections or file handles
        - Loads existing indices into memory for deduplication
        - Sets up transactional contexts
        - Creates checkpoints for rollback capability

        Must be called before any upsert operations.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def insert_done(self) -> None:
        """Finalize and commit all insertion operations.

        This method completes the insertion phase by:
        - Building and persisting mapping matrices (entities→relationships, relationships→chunks)
        - Committing transactions to all storage backends
        - Flushing in-memory indices to disk
        - Releasing resources and closing connections

        Must be called after all upsert operations are complete.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def query_start(self) -> None:
        """Prepare all storage backends for query operations.

        This method initializes the storage backends for retrieval operations. It typically:
        - Opens database connections in read-only mode
        - Loads indices and mapping matrices into memory
        - Restores from checkpoints if needed
        - Sets up read-optimized caching

        Must be called before any retrieval operations.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def query_done(self) -> None:
        """Finalize and cleanup after query operations.

        This method completes the query phase by:
        - Committing any read transactions
        - Releasing cached data from memory
        - Closing connections
        - Freeing resources

        Must be called after all retrieval operations are complete.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def filter_new_chunks(self, chunks_per_data: Iterable[Iterable[GTChunk]]) -> List[List[GTChunk]]:
        """Filter out chunks that already exist in storage to avoid redundant processing.

        This method efficiently identifies which chunks are new by checking their hashes
        against the chunk storage. Only new chunks are returned, reducing computational
        cost for repeated insertions of the same documents.

        The chunk structure (document boundaries) is preserved in the output, which is
        important for tracking which chunks belong to which documents.

        Args:
            chunks_per_data: Nested iterable where each inner iterable contains chunks
                from a single document.

        Returns:
            A list of lists with the same structure as input, but containing only chunks
            that are not already present in storage.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[GTNode, GTEdge, GTId]]]],
        documents: Iterable[Iterable[GTChunk]],
        show_progress: bool = True
    ) -> None:
        """Insert or update entities, relationships, and chunks in the knowledge graph.

        This is the core insertion method that coordinates the entire upsert pipeline:

        1. **Extract Subgraphs**: Awaits extraction futures and collects entities/relationships
        2. **Upsert Graph Elements**: Merges entities and relationships using upsert policies
        3. **Generate Embeddings**: Computes vector embeddings for all entities
        4. **Entity Deduplication**: Identifies similar entities via embedding similarity
        5. **Create Identity Edges**: Links duplicate entities with special "is" relationships
        6. **Store Chunks**: Persists all chunks to key-value storage

        Args:
            llm: Language model service (used by upsert policies for entity resolution).
            subgraphs: Futures that will resolve to extracted subgraphs, one per document.
                Each may be None if extraction failed.
            documents: The original chunks grouped by document, used for chunk storage.
            show_progress: If True, displays a progress bar during processing.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def get_context(
        self, query: str, entities: Dict[str, List[str]]
    ) -> Optional[TContext[GTNode, GTEdge, GTHash, GTChunk]]:
        """Retrieve relevant context from the knowledge graph for answering a query.

        This method implements the multi-stage retrieval and ranking pipeline:

        1. **Entity Retrieval**: Find entities semantically similar to query entities
        2. **Entity Ranking**: Score entities using graph structure (e.g., PageRank)
        3. **Relationship Ranking**: Score relationships connected to high-scoring entities
        4. **Chunk Ranking**: Score chunks associated with high-scoring relationships

        Each stage filters and ranks elements, propagating relevance scores through
        the graph structure to identify the most pertinent information.

        Args:
            query: The user's search query string.
            entities: Entities extracted from the query, categorized as:
                - "named": Specific named entities (e.g., "Albert Einstein")
                - "generic": Conceptual entities (e.g., "relativity theory")

        Returns:
            A TContext object containing:
            - entities: List of (entity, score) tuples
            - relations: List of (relationship, score) tuples
            - chunks: List of (chunk, score) tuples
            Returns None if no relevant context is found.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def get_num_entities(self) -> int:
        """Get the total number of entities in the knowledge graph.

        Returns:
            The count of entities (nodes) stored in the graph.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def get_num_relations(self) -> int:
        """Get the total number of relationships in the knowledge graph.

        Returns:
            The count of relationships (edges) stored in the graph.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def get_num_chunks(self) -> int:
        """Get the total number of text chunks stored.

        Returns:
            The count of chunks in the chunk storage.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    async def save_graphml(self, output_path: str) -> None:
        """Export the knowledge graph to GraphML format for visualization or analysis.

        GraphML is an XML-based format that can be imported into graph visualization
        tools like Gephi, Cytoscape, or NetworkX.

        Args:
            output_path: Filesystem path where the GraphML file should be written.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError
