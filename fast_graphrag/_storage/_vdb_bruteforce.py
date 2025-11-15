"""Brute-force vector database implementation.

This module implements a simple brute-force vector similarity search approach that computes
exact cosine similarities between query vectors and all stored vectors. While slower than
approximate methods like HNSW for large datasets, it guarantees exact results.

Approach:
    - Stores all vectors in a numpy array
    - Computes dot products between normalized query and database vectors
    - Returns exact top-k matches sorted by similarity

Performance Characteristics:
    - Time Complexity: O(n*d) where n=number of vectors, d=dimension
    - Space Complexity: O(n*d)
    - Suitable for small to medium datasets (< 100k vectors)
    - 100% recall (exact results)
    - Linear search time that grows with dataset size

Comparison with HNSW:
    Brute-force:
        - Exact results (100% recall)
        - Slower query time, especially for large datasets
        - No index building time
        - Lower memory overhead

    HNSW:
        - Approximate results (configurable recall)
        - Much faster query time for large datasets
        - Requires index building time
        - Higher memory overhead for graph structure
"""

import enum
import heapq
import json
import os
import pickle
import base64
import hashlib
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import TypedDict, Literal, Union, Callable, List, Tuple, Optional, Dict, Any, Iterable
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEmbedding, GTId, TScore
from fast_graphrag._utils import logger
from fast_graphrag._storage._base import BaseVectorStorage


@dataclass
class BruteForceVectorStorageConfig:
    """Configuration for BruteForce Vector Storage.

    Attributes:
        num_threads: Number of threads to use for computation. Default -1 uses all available.
                     Kept for compatibility with HNSW configuration.
        similarity_cutoff: Minimum similarity threshold. Vectors with similarity below this
                          threshold will be filtered out. Range: [0.0, 1.0].
    """
    num_threads: int = field(default=-1)
    similarity_cutoff: float = field(default=0.0)


# Field names for internal data structures
f_ID = "__id__"        # Unique identifier for each vector
f_VECTOR = "__vector__"  # The embedding vector itself
f_METRICS = "__metrics__"  # Similarity score/metrics

# Type definitions for NanoVectorDB
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)
Float = np.float32
ConditionLambda = Callable[[Data], bool]


def array_to_buffer_string(array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded string for JSON serialization.

    Args:
        array: Numpy array to convert.

    Returns:
        Base64-encoded string representation of the array.
    """
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    """Convert base64-encoded string back to numpy array.

    Args:
        base64_str: Base64-encoded string representation.
        dtype: Data type for the resulting array. Defaults to Float (np.float32).

    Returns:
        Reconstructed numpy array.
    """
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    """Load vector database from JSON file.

    Args:
        file_name: Path to the JSON storage file.

    Returns:
        Loaded database dictionary or None if file doesn't exist.
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    # Reconstruct the matrix from base64 string
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    """Generate MD5 hash of numpy array for unique identification.

    Args:
        a: Numpy array to hash.

    Returns:
        Hexadecimal MD5 hash string.
    """
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length for cosine similarity computation.

    Cosine similarity between normalized vectors can be computed as a simple dot product.

    Args:
        a: Array of vectors with shape (n, dim) or (dim,).

    Returns:
        L2-normalized vectors with the same shape.
    """
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    """Lightweight in-memory vector database with brute-force search.

    This class provides a simple vector database that stores embeddings in memory
    and performs exact similarity searches using brute-force computation.

    Attributes:
        embedding_dim: Dimensionality of the embedding vectors.
        metric: Similarity metric to use. Currently only "cosine" is supported.
        storage_file: Path to JSON file for persistence.

    How Similarity Search Works:
        1. All vectors are normalized to unit length on insertion
        2. Query vectors are also normalized
        3. Cosine similarity = dot product of normalized vectors
        4. Results are sorted by similarity score in descending order
    """
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"

    def pre_process(self):
        """Pre-process stored vectors according to the selected metric.

        For cosine similarity, normalizes all vectors to unit length.
        """
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        """Initialize the vector database after dataclass construction.

        Loads existing data from storage file if available, otherwise creates empty storage.
        Validates that embedding dimensions match and prepares the similarity metric.
        """
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        """Retrieve additional metadata stored in the database.

        Returns:
            Dictionary of additional data stored alongside vectors.
        """
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        """Store additional metadata in the database.

        Args:
            **kwargs: Key-value pairs to store as additional data.
        """
        self.__storage["additional_data"] = kwargs

    def upsert(self, datas: list[Data]):
        """Insert or update vectors in the database.

        This method performs an "upsert" operation: existing vectors (by ID) are updated,
        while new vectors are inserted.

        Indexing Operation:
            1. Generate IDs for vectors (use provided ID or hash of vector)
            2. Normalize vectors if using cosine similarity
            3. Update existing vectors with matching IDs
            4. Append new vectors to the storage matrix
            5. Maintain metadata alongside vectors

        Args:
            datas: List of data dictionaries, each containing:
                   - f_VECTOR: The embedding vector (required)
                   - f_ID: Unique identifier (optional, will be generated if missing)
                   - Other metadata fields (optional)

        Returns:
            Dictionary with 'update' and 'insert' keys, each containing lists of IDs
            that were updated or inserted.
        """
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}

        # Update existing vectors
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])

        if len(_index_datas) == 0:
            return report_return

        # Insert new vectors
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        """Retrieve vectors by their IDs.

        Args:
            ids: List of vector IDs to retrieve.

        Returns:
            List of data dictionaries for the matching vectors.
        """
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def delete(self, ids: list[str]):
        """Delete vectors from the database by their IDs.

        Args:
            ids: List of vector IDs to delete.
        """
        ids = set(ids)
        left_data = []
        delete_index = []
        for i, data in enumerate(self.__storage["data"]):
            if data[f_ID] in ids:
                delete_index.append(i)
                ids.remove(data[f_ID])
            else:
                left_data.append(data)
        self.__storage["data"] = left_data
        self.__storage["matrix"] = np.delete(
            self.__storage["matrix"], delete_index, axis=0
        )

    def save(self):
        """Persist the vector database to disk as JSON.

        The matrix is converted to base64 string for JSON serialization.
        """
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, ensure_ascii=False)

    def __len__(self):
        """Return the number of vectors in the database."""
        return len(self.__storage["data"])

    def query(
        self,
        query: np.ndarray,
        top_k: int = 10,
        better_than_threshold: float = None,
        filter_lambda: ConditionLambda = None,
    ) -> list[dict]:
        """Query the database for similar vectors.

        Querying Operation (Brute-force):
            1. Normalize the query vector
            2. Optionally filter vectors using filter_lambda
            3. Compute dot products with all (filtered) vectors - O(n*d)
            4. Sort by similarity score - O(n*log(n))
            5. Return top-k results above threshold

        Args:
            query: Query embedding vector.
            top_k: Number of top results to return. Default is 10.
            better_than_threshold: Optional minimum similarity threshold.
                                   Results below this will be filtered out.
            filter_lambda: Optional function to filter vectors before search.

        Returns:
            List of dictionaries containing vector data and similarity scores,
            sorted by descending similarity.
        """
        return self.usable_metrics[self.metric](
            query, top_k, better_than_threshold, filter_lambda=filter_lambda
        )

    def _cosine_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float,
        filter_lambda: ConditionLambda = None,
    ):
        """Internal method for cosine similarity-based search.

        Performs brute-force computation of cosine similarity between query and all vectors.

        Args:
            query: Query embedding vector.
            top_k: Number of results to return.
            better_than_threshold: Minimum similarity threshold.
            filter_lambda: Optional filter function for pre-filtering vectors.

        Returns:
            List of result dictionaries with similarity scores.
        """
        query = normalize(query)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            # Apply filter to reduce search space
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]

        # Compute cosine similarity via dot product (vectors are already normalized)
        scores = np.dot(use_matrix, query)

        # Sort and get top-k indices
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]  # Reverse to get descending order
        sort_abs_index = filter_index[sort_index]

        # Build results, filtering by threshold
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            if (
                better_than_threshold is not None
                and scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
        return results


@dataclass
class MultiTenantNanoVDB:
    """Multi-tenant vector database with LRU cache management.

    This class manages multiple isolated vector databases (one per tenant) with an
    in-memory LRU cache. When the cache reaches capacity, least recently used databases
    are persisted to disk and evicted from memory.

    Attributes:
        embedding_dim: Dimensionality of embedding vectors.
        metric: Similarity metric (currently only "cosine").
        max_capacity: Maximum number of tenant databases to keep in memory.
        storage_dir: Directory for persisting tenant databases.
    """
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    max_capacity: int = 1000
    storage_dir: str = "./nano_multi_tenant_storage"

    @staticmethod
    def jsonfile_from_id(tenant_id):
        """Generate storage filename for a tenant.

        Args:
            tenant_id: Unique tenant identifier.

        Returns:
            JSON filename for the tenant's database.
        """
        return f"nanovdb_{tenant_id}.json"

    def __post_init__(self):
        """Initialize multi-tenant storage with empty cache."""
        if self.max_capacity < 1:
            raise ValueError("max_capacity should be greater than 0")
        self.__storage: dict[str, NanoVectorDB] = {}
        self.__cache_queue: list[str] = []

    def contain_tenant(self, tenant_id: str) -> bool:
        """Check if a tenant exists (in memory or on disk).

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant exists, False otherwise.
        """
        return tenant_id in self.__storage or os.path.exists(
            f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"
        )

    def __load_tenant_in_cache(
        self, tenant_id: str, in_memory_tenant: NanoVectorDB
    ) -> NanoVectorDB:
        """Load a tenant database into the LRU cache.

        If cache is full, evicts the least recently used tenant (first in queue)
        and saves it to disk.

        Args:
            tenant_id: Tenant identifier.
            in_memory_tenant: NanoVectorDB instance to cache.

        Returns:
            The cached NanoVectorDB instance.
        """
        print(len(self.__storage), self.max_capacity)
        if len(self.__storage) >= self.max_capacity:
            # Evict least recently used tenant
            vdb = self.__storage.pop(self.__cache_queue.pop(0))
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
            vdb.save()
        self.__storage[tenant_id] = in_memory_tenant
        self.__cache_queue.append(tenant_id)
        pass

    def __load_tenant(self, tenant_id: str) -> NanoVectorDB:
        """Load a tenant's vector database from cache or disk.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            NanoVectorDB instance for the tenant.

        Raises:
            ValueError: If tenant doesn't exist.
        """
        if tenant_id in self.__storage:
            return self.__storage[tenant_id]
        if not self.contain_tenant(tenant_id):
            raise ValueError(f"Tenant {tenant_id} not in storage")

        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return in_memory_tenant

    def create_tenant(self) -> str:
        """Create a new tenant with a unique ID.

        Returns:
            UUID string identifying the new tenant.
        """
        tenant_id = str(uuid4())
        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return tenant_id

    def delete_tenant(self, tenant_id: str):
        """Delete a tenant and its data (from cache and disk).

        Args:
            tenant_id: Tenant identifier to delete.
        """
        if tenant_id in self.__storage:
            self.__storage.pop(tenant_id)
            self.__cache_queue.remove(tenant_id)
        if os.path.exists(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"):
            os.remove(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}")

    def get_tenant(self, tenant_id: str) -> NanoVectorDB:
        """Get a tenant's vector database.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            NanoVectorDB instance for the tenant.
        """
        return self.__load_tenant(tenant_id)

    def save(self):
        """Save all in-memory tenant databases to disk."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        for db in self.__storage.values():
            db.save()


@dataclass
class BruteForceVectorStorage(BaseVectorStorage[GTId, GTEmbedding]):
    """Brute-force vector storage implementation for exact nearest neighbor search.

    This class implements a simple but exact vector similarity search using brute-force
    computation. All query vectors are compared against all stored vectors using cosine
    similarity.

    How It Works:
        Indexing:
            1. Vectors are stored in a single numpy matrix (n_vectors x embedding_dim)
            2. IDs and metadata are stored in separate parallel lists
            3. No index structure is built - just raw vectors

        Querying:
            1. Normalize both query and database vectors
            2. Compute cosine similarity via matrix multiplication
            3. Sort all similarities and return top-k
            4. Time complexity: O(n * d) for n vectors of dimension d

    Performance Characteristics:
        - Query Time: O(n*d) - linear in dataset size
        - Memory: O(n*d) - just the vectors themselves
        - Accuracy: 100% (exact results)
        - Best for: Small datasets (< 100k vectors)

    Attributes:
        RESOURCE_NAME: Filename pattern for persisted index.
        RESOURCE_METADATA_NAME: Filename for persisted metadata.
        config: Configuration parameters.
        _vectors: Numpy matrix of all embedding vectors (n x d).
        _ids: List of integer IDs corresponding to each vector.
        _metadata: Dictionary mapping IDs to metadata dictionaries.
    """
    RESOURCE_NAME = "bruteforce_index_{}.bin"
    RESOURCE_METADATA_NAME = "bruteforce_metadata.pkl"
    config: BruteForceVectorStorageConfig = field()
    _vectors: Optional[np.ndarray] = field(init=False, default=None)
    _ids: List[GTId] = field(init=False, default_factory=list)
    _metadata: Dict[GTId, Dict[str, Any]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Return the number of vectors stored.

        Returns:
            Number of vectors in the database.
        """
        return len(self._ids) if self._ids is not None else 0

    async def upsert(
        self,
        ids: Iterable[GTId],
        embeddings: Iterable[GTEmbedding],
        metadata: Union[Iterable[Dict[str, Any]], None] = None,
    ) -> None:
        """Insert or update vectors in the storage.

        Args:
            ids: Iterable of unique identifiers for the vectors.
            embeddings: Iterable of embedding vectors to store.
            metadata: Optional iterable of metadata dictionaries.
        """
        # Ensure ids are integer type
        ids = [int(id_) for id_ in ids]
        embeddings = np.array(list(embeddings), dtype=np.float32)
        metadata = list(metadata) if metadata else None

        if self._vectors is None:
            self._vectors = embeddings
            self._ids = ids
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
            self._ids.extend(ids)

        if metadata:
            self._metadata.update(dict(zip(ids, metadata)))

    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int
    ) -> Tuple[Iterable[Iterable[GTId]], npt.NDArray[TScore]]:
        """Get k-nearest neighbors for query embeddings.

        Performs brute-force search by computing cosine similarity between query vectors
        and all stored vectors, then returning the top-k most similar.

        Args:
            embeddings: Query embedding vectors.
            top_k: Number of nearest neighbors to return per query.

        Returns:
            Tuple of (ids, scores) where:
                - ids: List of lists of neighbor IDs for each query
                - scores: 2D numpy array of similarity scores (range 0-1)
        """
        if self.size == 0:
            empty_list: List[List[GTId]] = []
            return empty_list, np.array([], dtype=TScore)

        query_embeddings = np.array(list(embeddings), dtype=np.float32)

        # Normalize vectors to compute cosine similarity
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        vectors_normalized = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)

        # Compute cosine similarity via dot product
        similarities = np.dot(query_embeddings, vectors_normalized.T)

        top_k = min(top_k, self.size)
        # Get indices of top-k scores (argsort puts smallest first, so take last k and reverse)
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        # Ensure returned IDs are integer type
        result_ids = [[int(self._ids[idx]) for idx in indices] for indices in top_indices]
        return result_ids, top_scores

    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, threshold: Optional[float] = None
    ) -> csr_matrix:
        """Compute similarity scores for all stored vectors, returning sparse matrix.

        Similar to get_knn but returns results as a sparse matrix where only the top-k
        scores per query are stored (rest are implicit zeros).

        Args:
            embeddings: Query embedding vectors.
            top_k: Number of top scores to keep per query. Default is 1.
            threshold: Optional minimum score threshold. Scores below this are set to 0.

        Returns:
            Sparse CSR matrix of shape (num_queries, num_stored_vectors) containing
            similarity scores.
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(list(embeddings), dtype=np.float32)

        if embeddings.size == 0 or self.size == 0:
            return csr_matrix((0, self.size))

        # Compute cosine similarity
        similarities = np.dot(embeddings, self._vectors.T) / (
            np.linalg.norm(embeddings, axis=1)[:, np.newaxis] *
            np.linalg.norm(self._vectors, axis=1)
        )

        top_k = min(top_k, self.size)
        top_indices = np.argsort(similarities, axis=1)[:, -top_k:][:, ::-1]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        if threshold is not None:
            top_scores[top_scores < threshold] = 0

        # Create sparse matrix (only top-k scores per query are non-zero)
        rows = np.repeat(np.arange(len(embeddings)), top_k)
        cols = top_indices.ravel()
        scores = csr_matrix(
            (top_scores.ravel(), (rows, cols)),
            shape=(len(embeddings), self.size)
        )

        return scores

    async def _insert_start(self):
        """Initialize storage for inserting data.

        Loads existing vectors and metadata from disk if available, otherwise
        initializes empty storage.
        """
        if self.namespace:
            index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)

            if index_file_name and metadata_file_name:
                try:
                    with open(index_file_name, "rb") as f:
                        data = pickle.load(f)
                        self._vectors = data.get('vectors')
                        # Ensure IDs are integer type
                        self._ids = [int(id_) for id_ in data.get('ids', [])]
                    with open(metadata_file_name, "rb") as f:
                        self._metadata = pickle.load(f)
                    logger.debug(f"Loaded {self.size} elements from vectordb storage '{index_file_name}'.")
                    return
                except Exception as e:
                    logger.warning(f"Error loading vectordb storage from {index_file_name}: {e}")

            logger.info(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
        else:
            logger.debug("Creating new volatile vectordb storage.")

        self._vectors = None
        self._ids = []
        self._metadata = {}

    async def _insert_done(self):
        """Save storage after inserting data.

        Persists vectors, IDs, and metadata to disk using pickle format.
        """
        if self.namespace:
            index_file_name = self.namespace.get_save_path(self.RESOURCE_NAME.format(self.embedding_dim))
            metadata_file_name = self.namespace.get_save_path(self.RESOURCE_METADATA_NAME)

            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(index_file_name), exist_ok=True)

                with open(index_file_name, "wb") as f:
                    pickle.dump({'vectors': self._vectors, 'ids': self._ids}, f)
                with open(metadata_file_name, "wb") as f:
                    pickle.dump(self._metadata, f)
                logger.debug(f"Saved {self.size} elements to vectordb storage '{index_file_name}'.")
            except Exception as e:
                t = f"Error saving vectordb storage to {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        """Initialize storage for querying.

        Loads vectors and metadata from disk. Raises error if namespace is not set
        or if loading fails.
        """
        assert self.namespace, "Loading a vectordb requires a namespace."
        
        index_file_name = self.namespace.get_load_path(self.RESOURCE_NAME.format(self.embedding_dim))
        metadata_file_name = self.namespace.get_load_path(self.RESOURCE_METADATA_NAME)

        if index_file_name and metadata_file_name:
            try:
                with open(index_file_name, "rb") as f:
                    data = pickle.load(f)
                    self._vectors = data.get('vectors')
                    self._ids = data.get('ids', [])
                with open(metadata_file_name, "rb") as f:
                    self._metadata = pickle.load(f)
                logger.debug(f"Loaded {self.size} elements from vectordb storage '{index_file_name}'.")
                return
            except Exception as e:
                t = f"Error loading vectordb storage from {index_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for vectordb storage '{index_file_name}'. Loading empty vectordb.")
            self._vectors = None
            self._ids = []
            self._metadata = {}

    async def _query_done(self):
        """Clean up after querying.

        For brute-force storage, no cleanup is needed as there's no index to dispose of.
        """
        pass