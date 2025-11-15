"""HNSW (Hierarchical Navigable Small World) vector database implementation.

This module implements approximate nearest neighbor search using the HNSW algorithm,
which builds a multi-layer graph structure for efficient similarity search.

HNSW Algorithm Overview:
    HNSW creates a hierarchical graph where:
    - Each vector is a node in the graph
    - Nodes are connected to nearby neighbors (by similarity)
    - Multiple layers form a hierarchy (like a skip list)
    - Search starts at the top layer and descends, getting more precise

    Layer Structure:
        Layer 2: [sparse, long-range connections]
        Layer 1: [medium density connections]
        Layer 0: [dense, short-range connections to all nodes]

    Search Process:
        1. Start at a random entry point in the top layer
        2. Greedily navigate to the nearest neighbor
        3. Descend to the next layer and repeat
        4. At layer 0, find the exact k-nearest neighbors

Performance Characteristics:
    - Query Time: O(log(n)) - logarithmic in dataset size
    - Build Time: O(n*log(n)*d) where n=vectors, d=dimension
    - Space Complexity: O(n*M) where M is max connections per node
    - Suitable for large datasets (millions of vectors)
    - ~95-99% recall (configurable via ef_search parameter)

Comparison with Brute-Force:
    HNSW:
        - Approximate results (95-99% recall)
        - Much faster query time O(log n) vs O(n)
        - Requires index building time
        - Higher memory for graph structure
        - Best for large datasets

    Brute-force:
        - Exact results (100% recall)
        - Slower query time for large datasets
        - No build time
        - Lower memory overhead
        - Best for small datasets

Key Parameters:
    - M: Max number of connections per node (higher = better recall, more memory)
    - ef_construction: Size of dynamic candidate list during construction (higher = better quality)
    - ef_search: Size of dynamic candidate list during search (higher = better recall)
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import hnswlib
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEmbedding, GTId, TScore
from fast_graphrag._utils import logger

from ._base import BaseVectorStorage


@dataclass
class HNSWVectorStorageConfig:
    """Configuration parameters for HNSW index.

    These parameters control the trade-off between speed, accuracy, and memory usage.

    Attributes:
        ef_construction: Size of the dynamic candidate list during index construction.
                        Higher values improve index quality but slow down construction.
                        Typical range: 100-500. Default: 128.

        M: Maximum number of bi-directional links per node in the graph.
           Higher values improve recall but increase memory usage and construction time.
           Typical range: 16-64. Default: 64.
           Memory usage is roughly: (M * 2 * 4 bytes * num_vectors)

        ef_search: Size of the dynamic candidate list during search.
                  Higher values improve recall but slow down queries.
                  Should be >= k (number of neighbors requested).
                  Typical range: 50-500. Default: 96.

        num_threads: Number of threads for parallel operations.
                    -1 uses all available CPU cores. Default: -1.

    Guidelines:
        - For higher recall: increase M and ef_search
        - For faster construction: decrease ef_construction
        - For faster queries: decrease ef_search
        - For less memory: decrease M
    """
    ef_construction: int = field(default=128)
    M: int = field(default=64)
    ef_search: int = field(default=96)
    num_threads: int = field(default=-1)


@dataclass
class HNSWVectorStorage(BaseVectorStorage[GTId, GTEmbedding]):
    """HNSW-based vector storage for fast approximate nearest neighbor search.

    This class uses the hnswlib library to build a hierarchical navigable small world
    graph for efficient similarity search in high-dimensional spaces.

    How It Works:
        Indexing:
            1. Initialize HNSW index with max capacity and graph parameters
            2. For each vector inserted:
               - Find M nearest neighbors in each layer
               - Create bi-directional links to neighbors
               - Probabilistically assign to higher layers
            3. Build time: O(n*log(n)*d) for n vectors of dimension d

        Querying:
            1. Start at entry point in top layer
            2. Greedily navigate to nearest neighbor using graph edges
            3. Descend to next layer and repeat
            4. At bottom layer, explore ef_search candidates
            5. Return top-k by similarity
            6. Query time: O(log(n)*d)

    Performance Characteristics:
        - Query Time: O(log(n)) - logarithmic growth with dataset size
        - Memory: O(n*M*d) - proportional to connections per node
        - Recall: ~95-99% depending on ef_search parameter
        - Best for: Large datasets (100k+ vectors)

    Attributes:
        RESOURCE_NAME: Filename pattern for persisted HNSW index.
        RESOURCE_METADATA_NAME: Filename for persisted metadata.
        INITIAL_MAX_ELEMENTS: Default initial capacity (can grow dynamically).
        config: HNSW configuration parameters.
        _index: The hnswlib Index object containing the graph structure.
        _metadata: Dictionary mapping vector IDs to metadata.
    """
    RESOURCE_NAME = "hnsw_index_{}.bin"
    RESOURCE_METADATA_NAME = "hnsw_metadata.pkl"
    INITIAL_MAX_ELEMENTS = 128000
    config: HNSWVectorStorageConfig = field()  # type: ignore
    _index: Any = field(init=False, default=None)  # type: ignore
    _metadata: Dict[GTId, Dict[str, Any]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Return the current number of vectors in the index.

        Returns:
            Number of indexed vectors.
        """
        return self._index.get_current_count()

    @property
    def max_size(self) -> int:
        """Return the maximum capacity of the index.

        Returns:
            Maximum number of vectors the index can hold before resizing.
        """
        return self._index.get_max_elements() or self.INITIAL_MAX_ELEMENTS

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        """Insert or update vectors in the HNSW index.

        Adds vectors to the graph structure, creating connections to nearby neighbors.
        Automatically resizes the index if capacity is exceeded.

        Indexing Process:
            1. Check if resize is needed (double capacity as needed)
            2. For each vector:
               - Insert into bottom layer (layer 0)
               - Select random layer level
               - Create M connections to nearest neighbors in each layer
               - Update existing connections to maintain graph quality

        Args:
            ids: Iterable of unique identifiers for the vectors.
            embeddings: Iterable of embedding vectors to index.
            metadata: Optional iterable of metadata dictionaries.

        Raises:
            AssertionError: If lengths of ids, embeddings, and metadata don't match.
        """
        ids = list(ids)
        embeddings = np.array(list(embeddings), dtype=np.float32)
        metadata = list(metadata) if metadata else None

        assert (len(ids) == len(embeddings)) and (
            metadata is None or (len(metadata) == len(ids))
        ), "ids, embeddings, and metadata (if provided) must have the same length"

        # Resize index if needed (doubles capacity)
        if self.size + len(embeddings) >= self.max_size:
            new_size = self.max_size * 2
            while self.size + len(embeddings) >= new_size:
                new_size *= 2
            self._index.resize_index(new_size)
            logger.info("Resizing HNSW index.")

        if metadata:
            self._metadata.update(dict(zip(ids, metadata)))
        # Add items to HNSW graph structure
        self._index.add_items(data=embeddings, ids=ids, num_threads=self.config.num_threads)

    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int
    ) -> Tuple[Iterable[Iterable[GTId]], npt.NDArray[TScore]]:
        """Get k-nearest neighbors using HNSW approximate search.

        Performs logarithmic-time search through the HNSW graph to find approximate
        nearest neighbors.

        Query Process:
            1. Start at entry point in highest layer
            2. Greedily navigate graph edges to nearest neighbor
            3. Descend to next layer when local minimum is reached
            4. At layer 0, maintain ef_search candidates
            5. Return top-k from final candidate set

        Args:
            embeddings: Query embedding vectors.
            top_k: Number of nearest neighbors to return per query.

        Returns:
            Tuple of (ids, scores) where:
                - ids: List of lists of neighbor IDs for each query
                - scores: 2D numpy array of similarity scores (range 0-1)
                  Note: HNSW returns distances in [0, 2], converted to similarities [0, 1]
        """
        if self.size == 0:
            empty_list: List[List[GTId]] = []
            logger.info("Querying knns in empty index.")
            return empty_list, np.array([], dtype=TScore)

        top_k = min(top_k, self.size)

        # Increase ef if top_k exceeds configured ef_search
        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        # HNSW returns cosine distances in range [0, 2] where 0=identical, 2=opposite
        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        # Convert distances [0, 2] to similarities [1, 0]
        return ids, 1.0 - np.array(distances, dtype=TScore) * 0.5

    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, threshold: Optional[float] = None
    ) -> csr_matrix:
        """Compute similarity scores for all stored vectors, returning sparse matrix.

        Similar to get_knn but returns results as a sparse matrix. Only the top-k
        scores per query are stored; the rest are implicit zeros.

        Args:
            embeddings: Query embedding vectors.
            top_k: Number of top scores to keep per query. Default is 1.
            threshold: Optional minimum score threshold. Scores below this are set to 0.

        Returns:
            Sparse CSR matrix of shape (num_queries, num_stored_vectors) containing
            similarity scores for the top-k matches per query.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(list(embeddings), dtype=np.float32)

        if embeddings.size == 0 or self.size == 0:
            logger.warning(f"No provided embeddings ({embeddings.size}) or empty index ({self.size}).")
            return csr_matrix((0, self.size))

        top_k = min(top_k, self.size)
        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        # HNSW returns cosine distances in range [0, 2] where 0=identical, 2=opposite
        ids, distances = self._index.knn_query(data=embeddings, k=top_k, num_threads=self.config.num_threads)

        ids = np.array(ids)
        # Convert distances [0, 2] to similarities [1, 0]
        scores = 1.0 - np.array(distances, dtype=TScore) * 0.5

        if threshold is not None:
            scores[scores < threshold] = 0

        # Create sparse distance matrix with shape (num_queries, num_all_vectors)
        # Only top-k scores per query are non-zero
        flattened_ids = ids.ravel()
        flattened_scores = scores.ravel()

        scores = csr_matrix(
            (flattened_scores, (np.repeat(np.arange(len(ids)), top_k), flattened_ids)),
            shape=(len(ids), self.size),
        )

        return scores

    async def _insert_start(self):
        """Initialize HNSW index for inserting data.

        Creates a new HNSW index with cosine similarity metric. Loads existing index
        from disk if available, otherwise initializes empty index with default capacity.
        """
        self._index = hnswlib.Index(space="cosine", dim=self.embedding_dim)  # type: ignore

        if self.namespace:
            index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)

            if index_file_name and metadata_file_name:
                try:
                    # Load the HNSW graph structure from disk
                    self._index.load_index(index_file_name, allow_replace_deleted=True)
                    with open(metadata_file_name, "rb") as f:
                        self._metadata = pickle.load(f)
                        logger.debug(
                            f"Loaded {self.size} elements from vectordb storage '{index_file_name}'."
                        )
                    return  # All good
                except Exception as e:
                    t = f"Error loading metadata file for vectordb storage '{metadata_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
        else:
            logger.debug("Creating new volatile vectordb storage.")

        # Initialize new HNSW index with configuration parameters
        self._index.init_index(
            max_elements=self.INITIAL_MAX_ELEMENTS,
            ef_construction=self.config.ef_construction,  # Quality during construction
            M=self.config.M,  # Max connections per node
            allow_replace_deleted=True
        )
        self._index.set_ef(self.config.ef_search)  # Search quality parameter
        self._metadata = {}

    async def _insert_done(self):
        """Save HNSW index and metadata to disk after inserting data.

        Persists the entire HNSW graph structure (including all layers and connections)
        and metadata using hnswlib's native serialization format.
        """
        if self.namespace:
            index_file_name = self.namespace.get_save_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_save_path(self.RESOURCE_METADATA_NAME)

            try:
                # Save HNSW graph structure to disk
                self._index.save_index(index_file_name)
                with open(metadata_file_name, "wb") as f:
                    pickle.dump(self._metadata, f)
                logger.debug(f"Saving {self.size} elements from vectordb storage '{index_file_name}'.")
            except Exception as e:
                t = f"Error saving vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        """Initialize HNSW index for querying.

        Loads the HNSW graph structure and metadata from disk. The loaded index
        is ready for immediate querying without requiring re-construction.

        Raises:
            AssertionError: If namespace is not set.
            InvalidStorageError: If loading from disk fails.
        """
        assert self.namespace, "Loading a vectordb requires a namespace."
        self._index = hnswlib.Index(space="cosine", dim=self.embedding_dim)  # type: ignore

        index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
        metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)
        if index_file_name and metadata_file_name:
            try:
                # Load the complete HNSW graph structure from disk
                self._index.load_index(index_file_name, allow_replace_deleted=True)
                with open(metadata_file_name, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.debug(f"Loaded {self.size} elements from vectordb storage '{index_file_name}'.")

                return # All good
            except Exception as e:
                t = f"Error loading vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")

        # Initialize empty index if no data file found
        self._index.init_index(
            max_elements=self.INITIAL_MAX_ELEMENTS,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
            allow_replace_deleted=True
        )
        self._index.set_ef(self.config.ef_search)
        self._metadata = {}

    async def _query_done(self):
        """Clean up after querying.

        For HNSW storage, no cleanup is needed as the index can be reused for
        multiple queries.
        """
        pass
