"""Default storage implementations for the fast-graphrag framework.

This module provides concrete storage class implementations that combine various
storage backends with the framework's standard type system. These classes serve
as the default storage providers for vectors, graphs, blobs, and key-value data.

The default implementations use:
- HNSW (Hierarchical Navigable Small World) for vector similarity search
- IGraph for graph storage with adjacency list structures
- Pickle serialization for blob and key-value storage

Attributes:
    __all__: List of public classes exported from this module.
"""

__all__ = [
    "DefaultVectorStorage",
    "DefaultVectorStorageConfig",
    "DefaultBlobStorage",
    "DefaultIndexedKeyValueStorage",
    "DefaultGraphStorage",
    "DefaultGraphStorageConfig",
]

from fast_graphrag._storage._blob_pickle import PickleBlobStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage
from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig
from fast_graphrag._types import GTBlob, GTEdge, GTEmbedding, GTId, GTKey, GTNode, GTValue


class DefaultVectorStorage(HNSWVectorStorage[GTId, GTEmbedding]):
    """Default vector storage using HNSW indexing with framework types.

    This class specializes HNSWVectorStorage to use the framework's standard
    types for identifiers (GTId) and embeddings (GTEmbedding). It provides
    efficient similarity search capabilities for embedding vectors using the
    Hierarchical Navigable Small World algorithm.

    The vector storage maintains indexes that enable fast approximate nearest
    neighbor searches, making it suitable for semantic search and retrieval
    operations in graph RAG applications.
    """
    pass


class DefaultVectorStorageConfig(HNSWVectorStorageConfig):
    """Default configuration for HNSW vector storage.

    This class extends HNSWVectorStorageConfig to provide configuration
    parameters for the default vector storage implementation. It handles
    settings such as index dimensions, similarity metrics, and performance
    tuning parameters for the HNSW algorithm.

    Configuration can be used to customize vector storage behavior, such as
    adjusting index granularity, similarity thresholds, or memory usage.
    """
    pass


class DefaultBlobStorage(PickleBlobStorage[GTBlob]):
    """Default blob storage using pickle serialization.

    This class specializes PickleBlobStorage to use the framework's standard
    blob type (GTBlob) for storing arbitrary binary data. It provides a simple
    file-based storage mechanism suitable for persisting unstructured data
    like documents, images, or other binary objects.

    The blob storage uses pickle for serialization, enabling efficient storage
    and retrieval of Python objects. It's optimized for scenarios where data
    persistence matters more than complex querying.
    """
    pass


class DefaultIndexedKeyValueStorage(PickleIndexedKeyValueStorage[GTKey, GTValue]):
    """Default indexed key-value storage with pickle serialization.

    This class specializes PickleIndexedKeyValueStorage to use the framework's
    standard types for keys (GTKey) and values (GTValue). It provides fast
    key-based lookup with optional indexing capabilities for efficient range
    queries and filtering operations.

    The storage uses pickle for serialization and supports indexed retrieval,
    making it suitable for storing metadata, properties, and other auxiliary
    data associated with graph nodes and edges.
    """
    pass


class DefaultGraphStorage(IGraphStorage[GTNode, GTEdge, GTId]):
    """Default graph storage using IGraph data structures.

    This class specializes IGraphStorage to use the framework's standard types:
    - GTNode: Node data type for vertices in the graph
    - GTEdge: Edge data type for relationships between nodes
    - GTId: Identifier type for addressing nodes

    The graph storage maintains the structural relationships between entities,
    supporting operations like traversal, pathfinding, and neighborhood queries.
    It uses igraph internally for efficient graph algorithms and storage.
    """
    pass


class DefaultGraphStorageConfig(IGraphStorageConfig[GTNode, GTEdge]):
    """Default configuration for IGraph-based graph storage.

    This class extends IGraphStorageConfig to provide configuration parameters
    for the default graph storage implementation. It handles settings specific
    to graph structure, such as directed/undirected settings, node and edge
    attributes, and graph persistence options.

    Configuration allows customization of graph storage behavior, including
    memory optimization strategies and algorithm parameters.
    """
    pass
