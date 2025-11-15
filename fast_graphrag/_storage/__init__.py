"""Storage module for the fast-graphrag framework.

This module provides the abstract base classes and default implementations for
various storage backends used in the graph RAG framework. It includes support for:

1. Vector Storage: Efficient storage and retrieval of embedding vectors with
   similarity search capabilities using HNSW indexing.

2. Graph Storage: Storage and management of knowledge graphs using igraph
   structures, supporting complex relational data.

3. Blob Storage: Generic storage for unstructured binary data such as documents,
   images, or other arbitrary objects.

4. Indexed Key-Value Storage: Fast key-based lookup with optional indexing for
   range queries and filtering operations on metadata and properties.

The module is organized into two main layers:
- Base Classes: Abstract interfaces (BaseBlobStorage, BaseVectorStorage, etc.)
  that define the contract for implementing storage backends.
- Default Implementations: Concrete implementations that combine specific
  backend technologies with the framework's type system.

Attributes:
    __all__: List of public classes exported from the storage module.
"""

__all__ = [
    'Namespace',
    'BaseBlobStorage',
    'BaseIndexedKeyValueStorage',
    'BaseVectorStorage',
    'BaseGraphStorage',
    'DefaultBlobStorage',
    'DefaultIndexedKeyValueStorage',
    'DefaultVectorStorage',
    'DefaultGraphStorage',
    'DefaultGraphStorageConfig',
    'DefaultVectorStorageConfig',
]

# Import base storage classes that define the abstract interface
from ._base import BaseBlobStorage, BaseGraphStorage, BaseIndexedKeyValueStorage, BaseVectorStorage, Namespace

# Import default storage implementations that combine backends with framework types
from ._default import (
    DefaultBlobStorage,
    DefaultGraphStorage,
    DefaultGraphStorageConfig,
    DefaultIndexedKeyValueStorage,
    DefaultVectorStorage,
    DefaultVectorStorageConfig,
)
