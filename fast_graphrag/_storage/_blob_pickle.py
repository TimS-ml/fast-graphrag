"""Pickle-based blob storage implementation.

This module provides a simple blob storage implementation that uses Python's pickle
serialization to persist data to disk. It stores a single blob object that can be
loaded from and saved to a pickle file.

The storage supports both persistent (with namespace) and volatile (in-memory only)
modes of operation.
"""

import pickle
from dataclasses import dataclass, field
from typing import Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTBlob
from fast_graphrag._utils import logger

from ._base import BaseBlobStorage


@dataclass
class PickleBlobStorage(BaseBlobStorage[GTBlob]):
    """A pickle-based storage implementation for blob data.

    This class provides a simple storage mechanism for a single blob object using
    Python's pickle serialization format. The blob can be persisted to disk or kept
    in memory depending on whether a namespace is configured.

    Attributes:
        RESOURCE_NAME: The default filename used for storing the pickle file.
        _data: Internal storage for the blob object. Initialized to None.
    """

    RESOURCE_NAME = "blob_data.pkl"
    _data: Optional[GTBlob] = field(init=False, default=None)

    async def get(self) -> Optional[GTBlob]:
        """Retrieve the stored blob data.

        Returns:
            The stored blob object, or None if no data has been set.
        """
        return self._data

    async def set(self, blob: GTBlob) -> None:
        """Store a blob object.

        Args:
            blob: The blob object to store.
        """
        self._data = blob

    async def _insert_start(self):
        """Initialize the storage for insert operations.

        This method is called at the beginning of an insert operation. It loads
        existing data from disk if a namespace is configured, or initializes an
        empty storage for volatile (in-memory) mode.

        Raises:
            InvalidStorageError: If there's an error loading the data file.
        """
        if self.namespace:
            # Persistent mode: attempt to load existing data from disk
            data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
            if data_file_name:
                try:
                    with open(data_file_name, "rb") as f:
                        self._data = pickle.load(f)
                except Exception as e:
                    t = f"Error loading data file for blob storage {data_file_name}: {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.info(f"No data file found for blob storage {data_file_name}. Loading empty storage.")
                self._data = None
        else:
            # Volatile mode: initialize empty in-memory storage
            self._data = None
            logger.debug("Creating new volatile blob storage.")

    async def _insert_done(self):
        """Finalize the storage after insert operations.

        This method is called after insert operations complete. If a namespace is
        configured, it persists the blob data to disk using pickle serialization.
        """
        if self.namespace:
            # Save data to disk in persistent mode
            data_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                with open(data_file_name, "wb") as f:
                    pickle.dump(self._data, f)
                logger.debug(
                    f"Saving blob storage '{data_file_name}'."
                )
            except Exception as e:
                logger.error(f"Error saving data file for blob storage {data_file_name}: {e}")

    async def _query_start(self):
        """Initialize the storage for query operations.

        This method is called at the beginning of a query operation. It loads the
        blob data from disk. A namespace must be configured for query operations.

        Raises:
            AssertionError: If no namespace is configured.
            InvalidStorageError: If there's an error loading the data file.
        """
        assert self.namespace, "Loading a blob storage requires a namespace."

        # Load data from the namespaced file path
        data_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if data_file_name:
            try:
                with open(data_file_name, "rb") as f:
                    self._data = pickle.load(f)
            except Exception as e:
                t = f"Error loading data file for blob storage {data_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            logger.warning(f"No data file found for blob storage {data_file_name}. Loading empty blob.")
            self._data = None

    async def _query_done(self):
        """Finalize the storage after query operations.

        This method is called after query operations complete. For blob storage,
        no cleanup is needed after queries.
        """
        pass
