"""Main GraphRAG implementation module.

This module contains the core implementation of the Graph-based Retrieval-Augmented
Generation (GraphRAG) system. It provides the BaseGraphRAG class which orchestrates
the entire RAG pipeline including:
- Document chunking and ingestion
- Entity and relationship extraction from text
- Knowledge graph construction and storage
- Query processing and context retrieval
- LLM-based response generation

The module supports both synchronous and asynchronous operations for flexibility
in different application contexts.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._models import TAnswer
from fast_graphrag._policies._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._services._chunk_extraction import BaseChunkingService
from fast_graphrag._services._information_extraction import BaseInformationExtractionService
from fast_graphrag._services._state_manager import BaseStateManagerService
from fast_graphrag._storage._base import BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage
from fast_graphrag._types import GTChunk, GTEdge, GTEmbedding, GTHash, GTId, GTNode, TContext, TDocument, TQueryResponse
from fast_graphrag._utils import TOKEN_TO_CHAR_RATIO, get_event_loop, logger
from pydantic import BaseModel


@dataclass
class InsertParam:
    """Parameters for document insertion operations.

    This dataclass is currently a placeholder for future extension. It can be used
    to pass additional parameters to control the document insertion behavior,
    such as extraction settings, batch sizes, or processing options.
    """
    pass


@dataclass
class QueryParam:
    """Parameters for controlling query behavior and response generation.

    Attributes:
        with_references (bool): If True, the LLM response will include references
            to source entities and relationships used in generating the answer.
            Defaults to False.
        only_context (bool): If True, only returns the retrieved context without
            generating an LLM response. Useful for debugging or when you want
            to handle response generation separately. Defaults to False.
        entities_max_tokens (int): Maximum number of tokens to include from
            retrieved entities in the context. Defaults to 4000.
        relations_max_tokens (int): Maximum number of tokens to include from
            retrieved relationships in the context. Defaults to 3000.
        chunks_max_tokens (int): Maximum number of tokens to include from
            retrieved text chunks in the context. Defaults to 9000.
    """
    with_references: bool = field(default=False)
    only_context: bool = field(default=False)
    entities_max_tokens: int = field(default=4000)
    relations_max_tokens: int = field(default=3000)
    chunks_max_tokens: int = field(default=9000)


@dataclass
class BaseGraphRAG(Generic[GTEmbedding, GTHash, GTChunk, GTNode, GTEdge, GTId]):
    """Core implementation of the Graph-based Retrieval-Augmented Generation system.

    This class orchestrates the entire GraphRAG pipeline, managing the flow from
    document ingestion to query answering. It combines several key components:

    1. Document Processing Pipeline:
       - Chunking: Splits documents into manageable pieces
       - Information Extraction: Extracts entities and relationships from chunks
       - Graph Construction: Builds a knowledge graph from extracted information

    2. Query Pipeline:
       - Entity Recognition: Identifies key entities in user queries
       - Context Retrieval: Finds relevant entities, relationships, and chunks
       - Response Generation: Uses LLM to generate answers based on context

    The class uses generic types to support different storage implementations and
    data structures, making it flexible and extensible.

    Attributes:
        working_dir (str): Directory path for storing temporary files and checkpoints.
        domain (str): Domain description to guide entity extraction (e.g., "scientific research").
        example_queries (str): Example queries to help the LLM understand query patterns.
        entity_types (List[str]): List of entity types to extract (e.g., ["PERSON", "ORGANIZATION"]).
        n_checkpoints (int): Number of checkpoints to maintain for rollback. Defaults to 0.
        llm_service (BaseLLMService): Service for LLM interactions.
        chunking_service (BaseChunkingService): Service for splitting documents into chunks.
        information_extraction_service (BaseInformationExtractionService): Service for
            extracting entities and relationships from text.
        state_manager (BaseStateManagerService): Service managing graph storage, embeddings,
            and state persistence.

    Generic Type Parameters:
        GTEmbedding: Type for embedding vectors
        GTHash: Type for chunk hashing
        GTChunk: Type for text chunks
        GTNode: Type for graph nodes (entities)
        GTEdge: Type for graph edges (relationships)
        GTId: Type for node/edge identifiers
    """

    working_dir: str = field()
    domain: str = field()
    example_queries: str = field()
    entity_types: List[str] = field()
    n_checkpoints: int = field(default=0)

    llm_service: BaseLLMService = field(init=False, default_factory=lambda: BaseLLMService(model=""))
    chunking_service: BaseChunkingService[GTChunk] = field(init=False, default_factory=lambda: BaseChunkingService())
    information_extraction_service: BaseInformationExtractionService[GTChunk, GTNode, GTEdge, GTId] = field(
        init=False,
        default_factory=lambda: BaseInformationExtractionService(
            graph_upsert=BaseGraphUpsertPolicy(
                config=None,
                nodes_upsert_cls=BaseNodeUpsertPolicy,
                edges_upsert_cls=BaseEdgeUpsertPolicy,
            )
        ),
    )
    state_manager: BaseStateManagerService[GTNode, GTEdge, GTHash, GTChunk, GTId, GTEmbedding] = field(
        init=False,
        default_factory=lambda: BaseStateManagerService(
            workspace=None,
            graph_storage=BaseGraphStorage[GTNode, GTEdge, GTId](config=None),
            entity_storage=BaseVectorStorage[GTId, GTEmbedding](config=None),
            chunk_storage=BaseIndexedKeyValueStorage[GTHash, GTChunk](config=None),
            embedding_service=BaseEmbeddingService(),
            node_upsert_policy=BaseNodeUpsertPolicy(config=None),
            edge_upsert_policy=BaseEdgeUpsertPolicy(config=None),
        ),
    )

    def insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True
    ) -> Tuple[int, int, int]:
        """Insert documents into the knowledge graph (synchronous version).

        This is a synchronous wrapper around async_insert(). It processes documents
        through the complete ingestion pipeline: chunking, entity/relationship extraction,
        and graph storage. Use this method in synchronous contexts; for async contexts,
        use async_insert() directly.

        The insertion process:
        1. Chunks documents into smaller pieces
        2. Filters out duplicate chunks
        3. Extracts entities and relationships from new chunks
        4. Updates the knowledge graph with new information
        5. Commits changes to storage

        Args:
            content (Union[str, List[str]]): Document(s) to insert. Can be a single
                string or a list of strings. Each string is treated as a separate document.
            metadata (Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]], optional):
                Metadata associated with the document(s). Can be a single dict applied
                to all documents, or a list of dicts (one per document). Defaults to None.
            params (InsertParam, optional): Additional parameters for insertion control.
                Currently unused but available for future extensions. Defaults to None.
            show_progress (bool, optional): Whether to display a progress bar during
                insertion. Defaults to True.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - Number of entities in the graph after insertion
                - Number of relationships in the graph after insertion
                - Number of chunks in the graph after insertion

        Raises:
            Exception: If any error occurs during the insertion pipeline, it will be
                logged and re-raised.

        Example:
            >>> grag = BaseGraphRAG(...)
            >>> num_entities, num_relations, num_chunks = grag.insert(
            ...     "Albert Einstein developed the theory of relativity.",
            ...     metadata={"source": "physics.txt"}
            ... )
        """
        return get_event_loop().run_until_complete(self.async_insert(content, metadata, params, show_progress))

    async def async_insert(
        self,
        content: Union[str, List[str]],
        metadata: Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]] = None,
        params: Optional[InsertParam] = None,
        show_progress: bool = True
    ) -> Tuple[int, int, int]:
        """Insert documents into the knowledge graph (asynchronous version).

        This is the core async implementation of document insertion. It processes documents
        through the complete ingestion pipeline using async/await for better performance
        with I/O operations. The method is transactional - changes are only committed
        if the entire pipeline completes successfully.

        The insertion workflow:
        1. Prepare and normalize input documents with metadata
        2. Chunk documents into smaller, processable pieces
        3. Filter out chunks that already exist (deduplication)
        4. Extract entities and relationships from new chunks using LLM
        5. Merge extracted information into the knowledge graph
        6. Compute and store embeddings for semantic search
        7. Commit all changes atomically

        Args:
            content (Union[str, List[str]]): Document(s) to insert. Can be a single
                string or a list of strings. Each string is treated as a separate document.
            metadata (Union[List[Optional[Dict[str, Any]]], Optional[Dict[str, Any]]], optional):
                Metadata associated with the document(s). Can be a single dict applied
                to all documents, or a list of dicts (one per document). Defaults to None.
            params (InsertParam, optional): Additional parameters for insertion control.
                Currently unused but available for future extensions. Defaults to None.
            show_progress (bool, optional): Whether to display a progress bar during
                insertion. Defaults to True.

        Returns:
            Tuple[int, int, int]: A tuple containing:
                - Total number of entities in the graph after insertion
                - Total number of relationships in the graph after insertion
                - Total number of chunks in the graph after insertion

        Raises:
            Exception: If any error occurs during the insertion pipeline, changes are
                not committed and the exception is logged and re-raised.

        Note:
            This method uses the state_manager's transactional interface:
            - insert_start(): Begins a transaction
            - insert_done(): Commits the transaction
            If an exception occurs, the transaction is implicitly rolled back.
        """
        # Initialize parameters with defaults if not provided
        if params is None:
            params = InsertParam()

        # Normalize content to always be a list for uniform processing
        if isinstance(content, str):
            content = [content]
        if isinstance(metadata, dict):
            metadata = [metadata]

        # Create document objects by pairing content with metadata
        # If metadata is a single dict or None, apply it to all documents
        # Otherwise, zip content with corresponding metadata entries
        if metadata is None or isinstance(metadata, dict):
            data = (TDocument(data=c, metadata=metadata or {}) for c in content)
        else:
            data = (TDocument(data=c, metadata=m or {}) for c, m in zip(content, metadata))

        try:
            # Begin insertion transaction - prepares state manager for updates
            await self.state_manager.insert_start()

            # Step 1: Chunk the documents into smaller pieces for processing
            # This makes extraction more accurate and manageable
            chunked_documents = await self.chunking_service.extract(data=data)

            # Step 2: Filter out duplicate chunks to avoid reprocessing
            # Deduplication is based on content hashing
            new_chunks_per_data = await self.state_manager.filter_new_chunks(chunks_per_data=chunked_documents)

            # Step 3: Extract entities and relationships from new chunks only
            # Uses LLM with domain-specific prompts to identify graph elements
            subgraphs = self.information_extraction_service.extract(
                llm=self.llm_service,
                documents=new_chunks_per_data,
                prompt_kwargs={
                    "domain": self.domain,  # Domain context for extraction
                    "example_queries": self.example_queries,  # Example queries to guide extraction
                    "entity_types": ",".join(self.entity_types),  # Expected entity types
                },
                entity_types=self.entity_types,
            )
            if len(subgraphs) == 0:
                logger.info("No new entities or relationships extracted from the data.")

            # Step 4: Merge extracted subgraphs into the main knowledge graph
            # This includes entity resolution, relationship merging, and embedding computation
            await self.state_manager.upsert(
                llm=self.llm_service, subgraphs=subgraphs, documents=new_chunks_per_data, show_progress=show_progress
            )

            # Step 5: Retrieve final counts of graph elements
            r = (
                await self.state_manager.get_num_entities(),
                await self.state_manager.get_num_relations(),
                await self.state_manager.get_num_chunks(),
            )

            # Step 6: Commit the transaction - all changes are now persisted
            await self.state_manager.insert_done()

            # Return the statistics about the updated graph
            return r
        except Exception as e:
            # Log error and re-raise - transaction will be rolled back
            logger.error(f"Error during insertion: {e}")
            raise e

    def query(self, query: str, params: Optional[QueryParam] = None, response_model = None) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        """Query the knowledge graph and generate a response (synchronous version).

        This is a synchronous wrapper around async_query(). It retrieves relevant
        information from the knowledge graph and uses an LLM to generate a natural
        language response. Use this method in synchronous contexts; for async contexts,
        use async_query() directly.

        The query process:
        1. Extract entities from the query text
        2. Retrieve relevant entities, relationships, and chunks from the graph
        3. Construct context from retrieved information
        4. Generate response using LLM with the context

        The method manages query transactions to ensure read consistency, especially
        important when concurrent insertions might be occurring.

        Args:
            query (str): The natural language question to answer.
            params (QueryParam, optional): Parameters controlling query behavior,
                such as token limits and whether to include references. Defaults to None.
            response_model (BaseModel, optional): Pydantic model for structured output.
                If provided, the LLM response will be parsed into this model format.
                Defaults to None (returns TAnswer).

        Returns:
            TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]: Object containing:
                - response: The generated answer (string or structured model)
                - context: Retrieved entities, relationships, and chunks used for the answer

        Raises:
            Exception: If any error occurs during query processing, it will be
                logged and re-raised. The query transaction is properly closed
                via a finally block.

        Example:
            >>> grag = BaseGraphRAG(...)
            >>> result = grag.query("Who developed the theory of relativity?")
            >>> print(result.response)
            "Albert Einstein developed the theory of relativity."
            >>> print(f"Used {len(result.context.entities)} entities")
        """
        async def _query() -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
            # Begin query transaction - ensures consistent read state
            await self.state_manager.query_start()
            try:
                # Execute the actual query logic
                answer = await self.async_query(query, params, response_model)
                return answer
            except Exception as e:
                # Log any errors that occur during query processing
                logger.error(f"Error during query: {e}")
                raise e
            finally:
                # Always close the query transaction, even if an error occurred
                await self.state_manager.query_done()

        # Run the async query in the event loop and return the result
        return get_event_loop().run_until_complete(_query())

    async def async_query(
        self, query: Optional[str], params: Optional[QueryParam] = None, response_model = None
    ) -> TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]:
        """Query the knowledge graph and generate a response (asynchronous version).

        This is the core async implementation of query processing. It implements a
        sophisticated retrieval pipeline that:
        1. Extracts entities mentioned in the query
        2. Performs semantic search to find relevant graph elements
        3. Retrieves connected entities, relationships, and source chunks
        4. Generates a contextually-grounded response using an LLM

        The retrieval-augmented generation (RAG) approach ensures responses are
        grounded in the actual knowledge stored in the graph, reducing hallucinations.

        Args:
            query (Optional[str]): The natural language question to answer.
                If None or empty, returns a failure response.
            params (QueryParam, optional): Parameters controlling query behavior:
                - with_references: Include source references in response
                - only_context: Return only context without generating response
                - entities_max_tokens: Max tokens for entity context
                - relations_max_tokens: Max tokens for relationship context
                - chunks_max_tokens: Max tokens for chunk context
                Defaults to None (uses QueryParam defaults).
            response_model (BaseModel, optional): Pydantic model for structured output.
                If provided, the LLM response will be parsed into this model format.
                If None, returns a TAnswer object. Defaults to None.

        Returns:
            TQueryResponse[GTNode, GTEdge, GTHash, GTChunk]: Contains:
                - response: Generated answer (string or structured model) or empty if only_context=True
                - context: TContext object with retrieved entities, relationships, and chunks

        Note:
            This method does NOT manage query transactions - the caller (query() method)
            is responsible for calling query_start() and query_done() on the state_manager.

        Example:
            >>> grag = BaseGraphRAG(...)
            >>> # Get context only for debugging
            >>> result = await grag.async_query(
            ...     "What is relativity?",
            ...     params=QueryParam(only_context=True)
            ... )
            >>> print(f"Retrieved {len(result.context.entities)} entities")
            >>>
            >>> # Get full response with references
            >>> result = await grag.async_query(
            ...     "What is relativity?",
            ...     params=QueryParam(with_references=True)
            ... )
        """
        # Handle empty or None query - return failure response
        if query is None or len(query) == 0:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )

        # Initialize parameters with defaults if not provided
        if params is None:
            params = QueryParam()

        # Step 1: Extract entities mentioned in the query using LLM
        # This helps identify what the query is asking about
        extracted_entities = await self.information_extraction_service.extract_entities_from_query(
            llm=self.llm_service, query=query, prompt_kwargs={}
        )

        # Step 2: Retrieve relevant context from the knowledge graph
        # Combines semantic search on query + entity-based retrieval
        # Returns entities, relationships, and chunks relevant to the query
        context = await self.state_manager.get_context(query=query, entities=extracted_entities)
        if context is None:
            return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](
                response=PROMPTS["fail_response"], context=TContext([], [], [])
            )

        # Step 3: Truncate context to fit within token limits
        # Converts token limits to character limits and formats context as string
        context_str = context.truncate(
            max_chars={
                "entities": params.entities_max_tokens * TOKEN_TO_CHAR_RATIO,
                "relations": params.relations_max_tokens * TOKEN_TO_CHAR_RATIO,
                "chunks": params.chunks_max_tokens * TOKEN_TO_CHAR_RATIO,
            },
            output_context_str=not params.only_context  # Skip string conversion if only context needed
        )

        # Step 4: Generate response using LLM or return empty if only context requested
        if params.only_context:
            # User only wants the context, not a generated response
            answer = ""
        else:
            # Determine response model - use TAnswer if none specified
            response_model = TAnswer if response_model is None else response_model

            # Generate response using appropriate prompt template
            # Different prompts are used based on whether references are requested
            llm_response, _ = await format_and_send_prompt(
                prompt_key="generate_response_query_with_references"
                if params.with_references
                else "generate_response_query_no_references",
                llm=self.llm_service,
                format_kwargs={
                    "query": query,  # User's question
                    "context": context_str  # Retrieved information from graph
                },
                response_model=response_model,
            )

            # Extract answer from response based on model type
            if response_model is None:
                answer = llm_response.answer
            else:
                # Structured response - return the entire model
                answer = llm_response

        # Return response paired with the context used to generate it
        return TQueryResponse[GTNode, GTEdge, GTHash, GTChunk](response=answer, context=context)

    def save_graphml(self, output_path: str) -> None:
        """Export the knowledge graph to GraphML format.

        GraphML is a comprehensive and easy-to-use file format for graphs. It's an
        XML-based format supported by many graph visualization and analysis tools
        such as Gephi, Cytoscape, and yEd.

        This method exports the entire knowledge graph including all entities (nodes),
        relationships (edges), and their associated attributes. The export is done
        within a query transaction to ensure a consistent snapshot of the graph.

        Args:
            output_path (str): File path where the GraphML file will be saved.
                Should have a .graphml or .xml extension. The directory must exist.

        Example:
            >>> grag = BaseGraphRAG(...)
            >>> grag.save_graphml("/path/to/output/my_graph.graphml")
            >>> # Can now visualize the graph in tools like Gephi

        Note:
            This is a synchronous operation that uses the query transaction mechanism
            to ensure the exported graph is in a consistent state, even if concurrent
            insertions are occurring.
        """
        async def _save_graphml() -> None:
            # Begin query transaction to get consistent snapshot of graph
            await self.state_manager.query_start()
            try:
                # Delegate to state manager to perform the actual export
                await self.state_manager.save_graphml(output_path)
            finally:
                # Always close the query transaction
                await self.state_manager.query_done()

        # Run the async save operation in the event loop
        get_event_loop().run_until_complete(_save_graphml())
