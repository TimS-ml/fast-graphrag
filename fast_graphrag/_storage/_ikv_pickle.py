"""Pickle-based indexed key-value storage implementation.

This module provides an indexed key-value storage implementation using Python's pickle
serialization. It maintains a mapping between keys and integer indices for efficient
lookups and supports operations like upsert, delete, and batch queries.

The storage uses index recycling to efficiently reuse indices from deleted entries,
and caches keys in a NumPy array for fast membership checks.
"""

import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import numpy.typing as npt

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTKey, GTValue, TIndex
from fast_graphrag._utils import logger

from ._base import BaseIndexedKeyValueStorage


@dataclass
class PickleIndexedKeyValueStorage(BaseIndexedKeyValueStorage[GTKey, GTValue]):
    """A pickle-based storage implementation for indexed key-value pairs.

    This class provides efficient storage and retrieval of key-value pairs with
    automatic index management. It assigns a unique integer index to each key and
    maintains a free list for recycling indices from deleted entries.

    The implementation uses NumPy arrays for efficient batch operations and caches
    keys for fast membership checks.

    Attributes:
        RESOURCE_NAME: The default filename used for storing the pickle file.
        _data: Dictionary mapping indices to values. Contains the actual stored data.
        _key_to_index: Dictionary mapping keys to their assigned indices.
        _free_indices: List of indices that have been freed by delete operations
            and can be reused for new entries.
        _np_keys: Cached NumPy array of keys for efficient membership checks.
            Invalidated on insert/delete operations.
    """

    RESOURCE_NAME = "kv_data.pkl"
    _data: Dict[Union[None, TIndex], GTValue] = field(init=False, default_factory=dict)
    _key_to_index: Dict[GTKey, TIndex] = field(init=False, default_factory=dict)
    _free_indices: List[TIndex] = field(init=False, default_factory=list)
    _np_keys: Optional[npt.NDArray[np.object_]] = field(init=False, default=None)

    async def size(self) -> int:
        """Get the number of key-value pairs in storage.

        Returns:
            The count of stored key-value pairs.
        """
        return len(self._data)

    async def get(self, keys: Iterable[GTKey]) -> Iterable[Optional[GTValue]]:
        """Retrieve values for the given keys.

        Args:
            keys: An iterable of keys to look up.

        Returns:
            An iterable of values corresponding to the keys. Returns None for
            keys that don't exist in storage.
        """
        return (self._data.get(self._key_to_index.get(key, None), None) for key in keys)

    async def get_by_index(self, indices: Iterable[TIndex]) -> Iterable[Optional[GTValue]]:
        """Retrieve values by their internal indices.

        Args:
            indices: An iterable of internal storage indices.

        Returns:
            An iterable of values at the given indices. Returns None for
            indices that don't exist.
        """
        return (self._data.get(index, None) for index in indices)

    async def get_index(self, keys: Iterable[GTKey]) -> Iterable[Optional[TIndex]]:
        """Get the internal indices for the given keys.

        Args:
            keys: An iterable of keys to look up.

        Returns:
            An iterable of indices corresponding to the keys. Returns None for
            keys that don't exist in storage.
        """
        return (self._key_to_index.get(key, None) for key in keys)

    async def upsert(self, keys: Iterable[GTKey], values: Iterable[GTValue]) -> None:
        """Insert or update key-value pairs.

        For each key-value pair:
        - If the key already exists, updates its value
        - If the key is new, assigns it an index (recycling from free list if
          available) and stores the value

        The NumPy key cache is invalidated when new keys are added.

        Args:
            keys: An iterable of keys to upsert.
            values: An iterable of values corresponding to the keys.
        """
        for key, value in zip(keys, values):
            index = self._key_to_index.get(key, None)
            if index is None:
                # New key: assign an index
                if len(self._free_indices) > 0:
                    # Recycle an index from a deleted entry
                    index = self._free_indices.pop()
                else:
                    # Allocate a new index
                    index = TIndex(len(self._data))
                self._key_to_index[key] = index

                # Invalidate cache since we added a new key
                self._np_keys = None
            # Store or update the value
            self._data[index] = value

    async def delete(self, keys: Iterable[GTKey]) -> None:
        """Delete key-value pairs from storage.

        Removes the specified keys and their associated values. The freed indices
        are added to the free list for recycling. The NumPy key cache is invalidated.

        Args:
            keys: An iterable of keys to delete.
        """
        for key in keys:
            index = self._key_to_index.pop(key, None)
            if index is not None:
                # Add the freed index to the free list for reuse
                self._free_indices.append(index)
                self._data.pop(index, None)

                # Invalidate cache since we removed a key
                self._np_keys = None
            else:
                logger.warning(f"Key '{key}' not found in indexed key-value storage.")

    async def mask_new(self, keys: Iterable[GTKey]) -> Iterable[bool]:
        """Create a boolean mask indicating which keys are new (not in storage).

        This method uses a cached NumPy array of existing keys for efficient
        membership checks. The cache is built on first use and invalidated on
        insert/delete operations.

        Args:
            keys: An iterable of keys to check.

        Returns:
            A boolean array where True indicates the key is new (not in storage)
            and False indicates the key already exists.
        """
        keys = list(keys)

        if len(keys) == 0:
            return np.array([], dtype=bool)

        # Build or use cached NumPy array of existing keys
        if self._np_keys is None:
            self._np_keys = np.fromiter(
                self._key_to_index.keys(),
                count=len(self._key_to_index),
                dtype=type(keys[0]),
            )
        keys_array = np.array(keys, dtype=type(keys[0]))

        # Return mask: True for keys NOT in storage (new keys)
        return ~np.isin(keys_array, self._np_keys)

    async def _insert_start(self):
        """Initialize the storage for insert operations.

        This method is called at the beginning of an insert operation. It loads
        existing data from disk if a namespace is configured, or initializes empty
        storage structures for volatile (in-memory) mode.

        The method loads three data structures from the pickle file:
        - _data: The index-to-value mapping
        - _free_indices: The list of recycled indices
        - _key_to_index: The key-to-index mapping

        Raises:
            InvalidStorageError: If there's an error loading the data file.
        """
        if self.namespace:
            # Persistent mode: attempt to load existing data from disk
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)

            if data_file_name:
                try:
                    with open(data_file_name, "rb") as f:
                        self._data, self._free_indices, self._key_to_index = pickle.load(f)
                        logger.debug(
                            f"Loaded {len(self._data)} elements from indexed key-value storage '{data_file_name}'."
                        )
                except Exception as e:
                    t = f"Error loading data file for key-vector storage '{data_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for key-vector storage '{data_file_name}'. Loading empty storage.")
                self._data = {}
                self._free_indices = []
                self._key_to_index = {}
        else:
            # Volatile mode: initialize empty in-memory storage
            self._data = {}
            self._free_indices = []
            self._key_to_index = {}
            logger.debug("Creating new volatile indexed key-value storage.")
        # Reset the key cache
        self._np_keys = None

    async def _insert_done(self):
        """Finalize the storage after insert operations.

        This method is called after insert operations complete. If a namespace is
        configured, it persists all storage structures to disk using pickle.

        The method saves three data structures to the pickle file:
        - _data: The index-to-value mapping
        - _free_indices: The list of recycled indices
        - _key_to_index: The key-to-index mapping

        Raises:
            InvalidStorageError: If there's an error saving the data file.
        """
        if self.namespace:
            # Save all data structures to disk in persistent mode
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                with open(data_file_name, "wb") as f:
                    pickle.dump((self._data, self._free_indices, self._key_to_index), f)
                    logger.debug(f"Saving {len(self._data)} elements to indexed key-value storage '{data_file_name}'.")
            except Exception as e:
                t = f"Error saving data file for key-vector storage '{data_file_name}': {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        """Initialize the storage for query operations.

        This method is called at the beginning of a query operation. It loads all
        storage structures from disk. A namespace must be configured for query
        operations.

        The method loads three data structures from the pickle file:
        - _data: The index-to-value mapping
        - _free_indices: The list of recycled indices
        - _key_to_index: The key-to-index mapping

        Raises:
            AssertionError: If no namespace is configured.
            InvalidStorageError: If there's an error loading the data file.
        """
        assert self.namespace, "Loading a kv storage requires a namespace."
        data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if data_file_name:
            try:
                with open(data_file_name, "rb") as f:
                    self._data, self._free_indices, self._key_to_index = pickle.load(f)
                    logger.debug(
                        f"Loaded {len(self._data)} elements from indexed key-value storage '{data_file_name}'."
                    )
            except Exception as e:
                t = f"Error loading data file for key-vector storage {data_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for key-vector storage '{data_file_name}'. Loading empty storage.")
            self._data = {}
            self._free_indices = []
            self._key_to_index = {}
        # Reset the key cache
        self._np_keys = None

    async def _query_done(self):
        """Finalize the storage after query operations.

        This method is called after query operations complete. For indexed
        key-value storage, no cleanup is needed after queries.
        """
        pass
