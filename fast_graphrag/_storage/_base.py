"""Base storage classes for the fast-graphrag library.

This module defines abstract base classes for all storage backends used in the knowledge graph system.
It provides a unified interface for different types of storage:

1. **Blob Storage**: Simple key-less storage for single large objects (e.g., entire graphs)
2. **Key-Value Storage**: Indexed storage for entities, chunks, and other keyed data
3. **Vector Storage**: Specialized storage for embeddings with similarity search capabilities
4. **Graph Storage**: Storage for graph structures with nodes and edges

Mode Switching Pattern:
    All storage backends implement a mode switching pattern that ensures data consistency
    during bulk operations. Storage can be in one of two modes:

    - **Insert Mode**: Optimized for bulk write operations. Implementations may batch writes,
      disable indexing temporarily, or use other optimizations for faster insertions.
    - **Query Mode**: Optimized for read operations. Ensures all data is committed and
      indexes are built for fast retrieval.

    The pattern enforces that:
    - Switching modes automatically commits any pending operations from the previous mode
    - Operations cannot be mixed within a single transaction
    - Explicit start/done calls bracket each mode's operations

    Example:
        ```python
        # Insert data
        await storage.insert_start()
        await storage.upsert(keys, values)
        await storage.insert_done()

        # Query data
        await storage.query_start()
        results = await storage.get(keys)
        await storage.query_done()
        ```

Type Parameters:
    GTBlob: Generic type for blob data
    GTKey: Generic type for storage keys
    GTValue: Generic type for storage values
    GTId: Generic type for identifiers (nodes, edges)
    GTNode: Generic type for graph nodes
    GTEdge: Generic type for graph edges
    GTEmbedding: Generic type for vector embeddings
"""
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    final,
)

from scipy.sparse import csr_matrix  # type: ignore

from fast_graphrag._types import GTBlob, GTEdge, GTEmbedding, GTId, GTKey, GTNode, GTValue, THash, TIndex, TScore
from fast_graphrag._utils import logger

from ._namespace import Namespace


@dataclass
class BaseStorage:
    """Abstract base class for all storage backends.

    This class implements the mode switching pattern that all storage backends inherit.
    It manages transitions between insert and query modes, ensuring that operations
    from one mode are properly committed before switching to another mode.

    The mode switching mechanism prevents data corruption and ensures consistency by:
    - Tracking the current operational mode (insert or query)
    - Managing transaction state with the _in_progress flag
    - Automatically committing pending operations when switching modes
    - Logging warnings and errors for improper usage

    Attributes:
        config: Backend-specific configuration object. The structure depends on the
            concrete storage implementation.
        namespace: Optional namespace for isolating storage operations. Allows multiple
            independent storage contexts within the same backend.
        _mode: Current operational mode, either "insert" or "query". None if no mode
            has been set yet.
        _in_progress: Tracks whether a transaction is currently in progress. True if
            operations have started but not been committed, False if committed, None
            if never initialized.
    """
    config: Optional[Any] = field()
    namespace: Optional[Namespace] = field(default=None)
    _mode: Optional[Literal["insert", "query"]] = field(init=False, default=None)
    _in_progress: Optional[bool] = field(init=False, default=None)

    def set_in_progress(self, in_progress: bool) -> None:
        """Set the in-progress flag to track transaction state.

        Args:
            in_progress: True if a transaction is in progress, False if committed.
        """
        self._in_progress = in_progress

    @final
    async def insert_start(self):
        """Begin an insert mode transaction.

        This method switches the storage to insert mode, committing any pending query
        operations if necessary. If already in insert mode and a transaction is in
        progress, this method does nothing (idempotent).

        The method ensures that:
        - Any pending query operations are committed before switching modes
        - The storage backend's _insert_start() hook is called to prepare for insertions
        - The _in_progress flag is properly managed

        Note:
            This method is marked @final and cannot be overridden. Subclasses should
            implement _insert_start() instead to customize insertion preparation logic.
        """
        if self._mode == "query":
            logger.info("Switching from query to insert mode.")
            if self._in_progress is not False:
                t = (
                    f"[{self.__class__.__name__}] Cannot being insert before committing query operations."
                    "Committing query operations now."
                )
                logger.error(t)
                await self._query_done()
                self._in_progress = False
        self._mode = "insert"

        if self._in_progress is not True:
            await self._insert_start()

    @final
    async def query_start(self):
        """Begin a query mode transaction.

        This method switches the storage to query mode, committing any pending insert
        operations if necessary. If already in query mode and a transaction is in
        progress, this method does nothing (idempotent).

        The method ensures that:
        - Any pending insert operations are committed before switching modes
        - The storage backend's _query_start() hook is called to prepare for queries
        - The _in_progress flag is properly managed

        Note:
            This method is marked @final and cannot be overridden. Subclasses should
            implement _query_start() instead to customize query preparation logic.
        """
        if self._mode == "insert":
            logger.info("Switching from insert to query mode.")
            if self._in_progress is not False:
                t = (
                    f"[{self.__class__.__name__}] Cannot being query before commiting insert operations."
                    "Committing insert operations now."
                )
                logger.error(t)
                await self._insert_done()
                self._in_progress = False
        self._mode = "query"

        if self._in_progress is not True:
            await self._query_start()

    @final
    async def insert_done(self) -> None:
        """Commit and finalize insert mode operations.

        This method should be called after all insert operations are complete to
        ensure data is properly persisted. It calls the storage backend's _insert_done()
        hook to perform any necessary finalization (e.g., flushing buffers, rebuilding
        indexes, committing transactions).

        The method validates that:
        - The storage is currently in insert mode (logs error if in query mode)
        - There are pending operations to commit (logs warning if none)

        Note:
            This method is marked @final and cannot be overridden. Subclasses should
            implement _insert_done() instead to customize commit logic.
        """
        if self._mode == "query":
            t = f"[{self.__class__.__name__}] Trying to commit insert operations in query mode."
            logger.error(t)
        else:
            if self._in_progress is not False:
                await self._insert_done()
            else:
                logger.warning(f"[{self.__class__.__name__}] No insert operations to commit.")

    @final
    async def query_done(self) -> None:
        """Finalize query mode operations and release resources.

        This method should be called after all query operations are complete to
        release any resources held during the query session. It calls the storage
        backend's _query_done() hook to perform cleanup (e.g., closing cursors,
        releasing locks, cleaning up temporary data).

        The method validates that:
        - The storage is currently in query mode (logs error if in insert mode)
        - There are pending operations to finalize (logs warning if none)

        Note:
            This method is marked @final and cannot be overridden. Subclasses should
            implement _query_done() instead to customize cleanup logic.
        """
        if self._mode == "insert":
            t = f"[{self.__class__.__name__}] Trying to commit query operations in insert mode."
            logger.error(t)
        else:
            if self._in_progress is not False:
                await self._query_done()
            else:
                logger.warning(f"[{self.__class__.__name__}] No query operations to commit.")

    async def _insert_start(self):
        """Hook for subclasses to prepare storage for insert operations.

        This protected method is called by insert_start() to allow subclasses to
        perform any necessary preparation before insert operations begin. Common
        implementations might:
        - Open a database transaction
        - Disable indexes for faster bulk insertions
        - Allocate buffers for batching operations
        - Initialize write caches

        Note:
            Subclasses should override this method to implement backend-specific
            preparation logic. The default implementation does nothing.
        """
        pass

    async def _insert_done(self):
        """Hook for subclasses to commit and finalize insert operations.

        This protected method is called by insert_done() to allow subclasses to
        perform any necessary finalization after insert operations complete. Common
        implementations might:
        - Commit database transactions
        - Rebuild or re-enable indexes
        - Flush write buffers to disk
        - Update metadata or statistics

        Note:
            Subclasses should override this method to implement backend-specific
            commit logic. The default implementation validates the current mode.
        """
        if self._mode == "query":
            logger.error("Trying to commit insert operations in query mode.")

    async def _query_start(self):
        """Hook for subclasses to prepare storage for query operations.

        This protected method is called by query_start() to allow subclasses to
        perform any necessary preparation before query operations begin. Common
        implementations might:
        - Ensure all indexes are built and optimized
        - Load frequently-accessed data into memory
        - Open read-only database connections
        - Initialize query caches

        Note:
            Subclasses should override this method to implement backend-specific
            preparation logic. The default implementation does nothing.
        """
        pass

    async def _query_done(self):
        """Hook for subclasses to finalize and clean up after query operations.

        This protected method is called by query_done() to allow subclasses to
        perform any necessary cleanup after query operations complete. Common
        implementations might:
        - Close database cursors or connections
        - Release locks or resources
        - Clear temporary query caches
        - Log query statistics

        Note:
            Subclasses should override this method to implement backend-specific
            cleanup logic. The default implementation validates the current mode.
        """
        if self._mode == "insert":
            logger.error("Trying to commit query operations in insert mode.")


####################################################################################################
# Blob Storage
####################################################################################################


@dataclass
class BaseBlobStorage(BaseStorage, Generic[GTBlob]):
    """Abstract base class for blob storage backends.

    Blob storage provides simple key-less storage for single large objects. Unlike
    key-value storage, blob storage is designed for scenarios where you need to store
    and retrieve a single large data structure (e.g., an entire serialized graph,
    a complete index, or a large binary object).

    This is the simplest storage interface, with only get and set operations.
    Implementations might use:
    - File system storage (single file)
    - Object storage (S3, Azure Blob, etc.)
    - Database BLOB columns
    - In-memory storage for testing

    Type Parameters:
        GTBlob: The type of blob data to store. This can be any serializable type
            (e.g., bytes, pickled objects, JSON-serializable dicts).

    Example:
        ```python
        # Store a complete graph structure
        await blob_storage.insert_start()
        await blob_storage.set(serialized_graph)
        await blob_storage.insert_done()

        # Retrieve the graph structure
        await blob_storage.query_start()
        graph = await blob_storage.get()
        await blob_storage.query_done()
        ```
    """
    async def get(self) -> Optional[GTBlob]:
        """Retrieve the stored blob.

        Returns:
            The stored blob data, or None if no blob has been stored yet.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def set(self, blob: GTBlob) -> None:
        """Store a blob, replacing any existing blob.

        Args:
            blob: The blob data to store. This completely replaces any existing
                blob data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


####################################################################################################
# Key-Value Storage
####################################################################################################


@dataclass
class BaseIndexedKeyValueStorage(BaseStorage, Generic[GTKey, GTValue]):
    """Abstract base class for indexed key-value storage backends.

    This storage type provides a key-value interface with automatic indexing. Each
    key-value pair is assigned a unique integer index for efficient access. This
    dual-access pattern (by key or by index) enables both hash-based lookups and
    array-based operations.

    Indexed key-value storage is used for storing entities, chunks, and other data
    where both keyed access (for lookups) and indexed access (for bulk operations
    or array-based algorithms) are needed.

    Type Parameters:
        GTKey: The type of keys used to identify values. Typically strings or hashes.
        GTValue: The type of values to store.

    Key Features:
        - Dual access: Retrieve values by key or by integer index
        - Upsert semantics: Insert or update in a single operation
        - Index mapping: Convert between keys and integer indices
        - Batch operations: Support for bulk get/upsert/delete operations
        - New key detection: Identify which keys don't exist yet

    Example:
        ```python
        # Insert entities
        await storage.insert_start()
        keys = ["entity1", "entity2", "entity3"]
        values = [Entity(...), Entity(...), Entity(...)]
        await storage.upsert(keys, values)
        await storage.insert_done()

        # Query entities by key
        await storage.query_start()
        entities = await storage.get(["entity1", "entity2"])

        # Or query by index for array operations
        entities = await storage.get_by_index([0, 1, 2])
        await storage.query_done()
        ```
    """
    async def size(self) -> int:
        """Get the total number of key-value pairs stored.

        Returns:
            The number of key-value pairs currently in storage.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get(self, keys: Iterable[GTKey]) -> Iterable[Optional[GTValue]]:
        """Retrieve values by their keys.

        Args:
            keys: An iterable of keys to look up.

        Returns:
            An iterable of values corresponding to the keys. Returns None for
            keys that don't exist in storage. The order matches the input keys.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_by_index(self, indices: Iterable[TIndex]) -> Iterable[Optional[GTValue]]:
        """Retrieve values by their integer indices.

        Args:
            indices: An iterable of integer indices to look up.

        Returns:
            An iterable of values corresponding to the indices. Returns None for
            indices that don't exist in storage. The order matches the input indices.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_index(self, keys: Iterable[GTKey]) -> Iterable[Optional[TIndex]]:
        """Get the integer indices for the given keys.

        This method provides the mapping from keys to their assigned integer indices,
        enabling conversion between key-based and index-based operations.

        Args:
            keys: An iterable of keys to look up.

        Returns:
            An iterable of integer indices corresponding to the keys. Returns None
            for keys that don't exist in storage. The order matches the input keys.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert(self, keys: Iterable[GTKey], values: Iterable[GTValue]) -> None:
        """Insert new key-value pairs or update existing ones.

        If a key already exists, its value is updated. If a key doesn't exist,
        a new key-value pair is inserted and assigned the next available index.

        Args:
            keys: An iterable of keys to insert or update.
            values: An iterable of values corresponding to the keys. Must have
                the same length as keys.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_by_index(self, indices: Iterable[TIndex], values: Iterable[GTValue]) -> None:
        """Update values at the given indices.

        This method updates existing values by their integer indices. It's typically
        used for bulk updates in array-based operations.

        Args:
            indices: An iterable of integer indices to update.
            values: An iterable of new values corresponding to the indices. Must
                have the same length as indices.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def delete(self, keys: Iterable[GTKey]) -> None:
        """Delete key-value pairs by their keys.

        Args:
            keys: An iterable of keys to delete. Non-existent keys are silently
                ignored.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def delete_by_index(self, indices: Iterable[TIndex]) -> None:
        """Delete key-value pairs by their indices.

        Args:
            indices: An iterable of integer indices to delete. Non-existent indices
                are silently ignored.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def mask_new(self, keys: Iterable[GTKey]) -> Iterable[bool]:
        """Identify which keys are new (don't exist in storage).

        This method is useful for optimizing bulk insertions by identifying which
        keys need to be inserted versus updated.

        Args:
            keys: An iterable of keys to check.

        Returns:
            An iterable of boolean values, where True indicates the key is new
            (doesn't exist) and False indicates it already exists. The order
            matches the input keys.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


####################################################################################################
# Vector Storage
####################################################################################################


@dataclass
class BaseVectorStorage(BaseStorage, Generic[GTId, GTEmbedding]):
    """Abstract base class for vector storage backends.

    Vector storage provides specialized storage for high-dimensional embeddings with
    efficient similarity search capabilities. This is a critical component for semantic
    search, entity matching, and other AI-powered features in the knowledge graph.

    Vector storage backends typically implement:
    - Approximate nearest neighbor (ANN) search algorithms (e.g., HNSW, IVF, LSH)
    - Distance metrics (cosine similarity, euclidean distance, dot product)
    - Index structures optimized for high-dimensional vectors
    - Batch operations for efficient bulk processing

    Type Parameters:
        GTId: The type of identifiers for vectors. Typically strings or integers that
            reference entities, chunks, or other graph elements.
        GTEmbedding: The type of embedding vectors. Usually numpy arrays or lists of floats.

    Attributes:
        embedding_dim: The dimensionality of the embedding vectors. All vectors stored
            in this backend must have this same dimensionality. Set to 0 if not yet
            initialized.

    Example:
        ```python
        # Create vector storage with 384-dimensional embeddings
        storage = VectorStorage(embedding_dim=384)

        # Insert embeddings
        await storage.insert_start()
        ids = ["chunk1", "chunk2", "chunk3"]
        embeddings = [embed("text1"), embed("text2"), embed("text3")]
        await storage.upsert(ids, embeddings)
        await storage.insert_done()

        # Find similar vectors
        await storage.query_start()
        query_embedding = embed("search query")
        similar_ids, scores = await storage.get_knn([query_embedding], top_k=5)
        await storage.query_done()
        ```
    """
    embedding_dim: int = field(default=0)

    @property
    def size(self) -> int:
        """Get the total number of vectors stored.

        Returns:
            The number of embedding vectors currently in storage.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int
    ) -> Tuple[Iterable[Iterable[GTId]], Iterable[Iterable[TScore]]]:
        """Find the k-nearest neighbors for each query embedding.

        This is the core similarity search operation. For each query embedding,
        it finds the top_k most similar vectors in storage using the backend's
        configured distance metric (typically cosine similarity).

        Args:
            embeddings: An iterable of query embedding vectors to search for.
                Each embedding must have dimensionality matching embedding_dim.
            top_k: The number of nearest neighbors to return for each query.

        Returns:
            A tuple of two iterables:
                - First element: For each query, an iterable of the k nearest neighbor IDs
                - Second element: For each query, an iterable of the k similarity scores

            Both iterables have the same structure and ordering. For example:
                ([["id1", "id2"], ["id3", "id4"]], [[0.95, 0.87], [0.91, 0.83]])

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        """Insert or update embedding vectors.

        If an ID already exists, its embedding (and metadata) is updated. If an ID
        doesn't exist, a new vector is inserted into the index.

        Args:
            ids: An iterable of identifiers for the embeddings. These typically
                reference chunks, entities, or other graph elements.
            embeddings: An iterable of embedding vectors corresponding to the IDs.
                Must have the same length as ids. Each embedding must have
                dimensionality matching embedding_dim.
            metadata: Optional metadata dictionaries for each embedding. If provided,
                must have the same length as ids. Metadata can be used for filtering
                or storing additional information alongside vectors.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, threshold: Optional[float] = None
    ) -> csr_matrix:
        """Score all stored embeddings against the given query embeddings.

        Unlike get_knn which returns only the top-k matches, this method computes
        similarity scores for all stored embeddings, returning a sparse matrix of
        scores. This is useful for global ranking operations or when you need to
        consider all possible matches.

        Args:
            embeddings: An iterable of query embedding vectors. Each embedding must
                have dimensionality matching embedding_dim.
            top_k: Maximum number of non-zero scores to keep per query. Defaults to 1.
                Setting higher values includes more matches but increases memory usage.
            threshold: Optional minimum similarity score. Scores below this threshold
                are set to zero (not included in sparse matrix). Helps reduce memory
                usage and computation by filtering low-relevance matches.

        Returns:
            A sparse CSR matrix of shape (num_queries, num_stored_embeddings) where
            entry [i, j] contains the similarity score between query i and stored
            embedding j. Low scores (below threshold or outside top_k) are zero.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


####################################################################################################
# Graph Storage
####################################################################################################


@dataclass
class BaseGraphStorage(BaseStorage, Generic[GTNode, GTEdge, GTId]):
    """Abstract base class for graph storage backends.

    Graph storage is the core storage type for the knowledge graph. It manages the
    storage and retrieval of nodes (entities) and edges (relationships) that form
    the graph structure. This class provides a comprehensive interface for graph
    operations including creation, retrieval, update, and graph algorithms.

    The graph storage maintains:
    - Nodes: Typically representing entities extracted from documents
    - Edges: Representing relationships between entities
    - Node/Edge indices: Integer indices for efficient array-based operations
    - Edge attributes: Metadata about relationships (e.g., source chunks, weights)
    - Graph structure: Adjacency information for traversal and algorithms

    Type Parameters:
        GTNode: The type of graph nodes. Typically contains entity information.
        GTEdge: The type of graph edges. Typically contains relationship information.
        GTId: The type of identifiers for nodes and edges. Usually strings or hashes.

    Example:
        ```python
        # Build a graph
        await storage.insert_start()

        # Add nodes (entities)
        idx1 = await storage.upsert_node(Entity("Python"), None)
        idx2 = await storage.upsert_node(Entity("Programming"), None)

        # Add edges (relationships)
        edge = Relationship(source="Python", target="Programming", type="is_a")
        await storage.upsert_edge(edge, None)

        await storage.insert_done()

        # Query the graph
        await storage.query_start()
        node, idx = await storage.get_node("Python")
        neighbors = await storage.get_edges(idx1, idx2)
        await storage.query_done()
        ```
    """
    async def save_graphml(self, path: str) -> None:
        """Export the graph to GraphML format.

        GraphML is a standard XML-based format for representing graphs. This method
        exports the entire graph structure (nodes, edges, and attributes) to a file
        that can be visualized or analyzed with standard graph tools.

        Args:
            path: File path where the GraphML file should be written.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def node_count(self) -> int:
        """Get the total number of nodes in the graph.

        Returns:
            The number of nodes currently stored in the graph.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def edge_count(self) -> int:
        """Get the total number of edges in the graph.

        Returns:
            The number of edges currently stored in the graph.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_node(self, node: Union[GTNode, GTId]) -> Union[Tuple[GTNode, TIndex], Tuple[None, None]]:
        """Retrieve a node and its index by node object or ID.

        Args:
            node: Either a node object or a node ID to look up.

        Returns:
            A tuple of (node, index) if the node exists, or (None, None) if not found.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_all_edges(self) -> Iterable[GTEdge]:
        """Retrieve all edges in the graph.

        Returns:
            An iterable of all edge objects in the graph. The order is implementation-
            dependent.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_edges(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[Tuple[GTEdge, TIndex]]:
        """Get all edges between a source and target node.

        This method retrieves all edges connecting the specified source and target
        nodes. There may be multiple edges between the same pair of nodes (multigraph).

        Args:
            source_node: The source node ID or index.
            target_node: The target node ID or index.

        Returns:
            An iterable of (edge, index) tuples for all edges from source to target.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def _get_edge_indices(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[TIndex]:
        """Get indices of all edges between a source and target node.

        This is a protected helper method for retrieving just the indices of edges,
        without loading the full edge objects.

        Args:
            source_node: The source node ID or index.
            target_node: The target node ID or index.

        Returns:
            An iterable of edge indices for all edges from source to target.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_node_by_index(self, index: TIndex) -> Union[GTNode, None]:
        """Retrieve a node by its integer index.

        Args:
            index: The integer index of the node to retrieve.

        Returns:
            The node object if found, None otherwise.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_edge_by_index(self, index: TIndex) -> Union[GTEdge, None]:
        """Retrieve an edge by its integer index.

        Args:
            index: The integer index of the edge to retrieve.

        Returns:
            The edge object if found, None otherwise.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_node(self, node: GTNode, node_index: Union[TIndex, None]) -> TIndex:
        """Insert a new node or update an existing one.

        If node_index is None, a new node is inserted and assigned the next available
        index. If node_index is provided, the node at that index is updated.

        Args:
            node: The node object to insert or update.
            node_index: The index of an existing node to update, or None to insert new.

        Returns:
            The index of the inserted or updated node.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def upsert_edge(self, edge: GTEdge, edge_index: Union[TIndex, None]) -> TIndex:
        """Insert a new edge or update an existing one.

        If edge_index is None, a new edge is inserted and assigned the next available
        index. If edge_index is provided, the edge at that index is updated.

        Args:
            edge: The edge object to insert or update.
            edge_index: The index of an existing edge to update, or None to insert new.

        Returns:
            The index of the inserted or updated edge.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def insert_edges(
        self,
        edges: Optional[Iterable[GTEdge]] = None,
        indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None,
        attrs: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> List[TIndex]:
        """Bulk insert edges with optional attributes.

        This method provides efficient bulk insertion of multiple edges. Edges can
        be specified either as edge objects or as index pairs (source, target).
        Additional attributes can be attached to the edges.

        Args:
            edges: Optional iterable of edge objects to insert.
            indices: Optional iterable of (source_index, target_index) tuples. Used
                when you have node indices but not full edge objects.
            attrs: Optional mapping of attribute names to sequences of values. Each
                attribute sequence must have the same length as edges/indices.
                Example: {"weight": [0.5, 0.8], "type": ["related", "similar"]}

        Returns:
            A list of indices for the newly inserted edges, in the same order as
            the input edges/indices.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def are_neighbours(self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]) -> bool:
        """Check if two nodes are connected by an edge.

        Args:
            source_node: The source node ID or index.
            target_node: The target node ID or index.

        Returns:
            True if there is at least one edge from source to target, False otherwise.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def delete_edges_by_index(self, indices: Iterable[TIndex]) -> None:
        """Delete edges by their indices.

        Args:
            indices: An iterable of edge indices to delete. Non-existent indices
                are silently ignored.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_entities_to_relationships_map(self) -> csr_matrix:
        """Get a sparse matrix mapping entities to their relationships.

        This method returns a matrix that represents which entities participate in
        which relationships. It's used for graph algorithms and analysis.

        Returns:
            A sparse CSR matrix of shape (num_entities, num_relationships) where
            entry [i, j] is non-zero if entity i participates in relationship j.
            The exact non-zero values depend on the implementation (often 1.0 or
            relationship weights).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_relationships_to_chunks_map(self) -> dict[int, List[THash]]:
        """Get a mapping from relationship indices to source chunk hashes.

        This method provides traceability from relationships back to the original
        document chunks where they were extracted. This is crucial for citation,
        verification, and context retrieval.

        Returns:
            A dictionary mapping relationship indices to lists of chunk hashes. Each
            relationship may be mentioned in multiple chunks.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def get_relationships_attrs(self, key: str) -> List[List[Any]]:
        """Get attribute values for all relationships.

        This method retrieves a specific attribute from all relationships in the
        graph, organized by entity pairs.

        Args:
            key: The attribute key to retrieve (e.g., "weight", "type", "source_chunks").

        Returns:
            A list of lists, where each inner list contains the attribute values for
            all relationships between a specific pair of entities. The structure
            matches the graph's entity-to-entity organization.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    async def score_nodes(self, initial_weights: Optional[csr_matrix]) -> csr_matrix:
        """Score nodes based on graph structure and initial weights.

        This method implements graph-based ranking algorithms (e.g., PageRank-like
        algorithms) to score nodes based on their position in the graph and optional
        initial weights. It's used for identifying important entities and ranking
        search results.

        Args:
            initial_weights: Optional sparse matrix of initial node weights/scores.
                If provided, these weights influence the final scoring. If None,
                all nodes start with equal weight.

        Returns:
            A sparse CSR matrix containing the computed node scores. The shape and
            structure depend on the specific algorithm implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
