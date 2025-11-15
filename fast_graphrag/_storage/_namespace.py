"""Workspace and namespace management for storage resources.

This module provides classes for managing storage workspaces and namespaces with
support for checkpointing and rollback. Workspaces manage checkpoint directories
and provide isolation between different storage sessions, while namespaces provide
logical separation of resources within a workspace.

Key features:
- Automatic checkpoint management with configurable retention
- Rollback support for failed checkpoints
- Namespace-based resource organization
- Error checkpoint preservation for debugging
"""

import os
import shutil
import time
from typing import Any, Callable, List, Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._utils import logger


class Workspace:
    """Manages storage workspace with checkpoint support.

    A workspace provides a directory structure for storing data with optional
    checkpoint management. Checkpoints are timestamped directories that allow
    for versioning and rollback of storage state.

    Checkpoint behavior:
    - When keep_n > 0: Creates timestamped checkpoint directories
    - When keep_n = 0: Uses the working directory directly (no checkpoints)
    - Failed checkpoints are renamed with "0__err_" prefix for debugging

    Attributes:
        working_dir: The base directory for this workspace.
        keep_n: Number of checkpoints to retain. Older checkpoints are deleted.
        checkpoints: Sorted list of available checkpoint timestamps (newest first).
        current_load_checkpoint: The checkpoint currently being loaded from.
        save_checkpoint: The checkpoint being saved to (lazily initialized).
        failed_checkpoints: List of checkpoint names that failed to load.
    """

    @staticmethod
    def new(working_dir: str, checkpoint: int = 0, keep_n: int = 0) -> "Workspace":
        """Factory method to create a new Workspace instance.

        Args:
            working_dir: The base directory for the workspace.
            checkpoint: The checkpoint to load. 0 means use working_dir directly,
                None means use the latest checkpoint.
            keep_n: Number of checkpoints to retain. 0 means no checkpointing.

        Returns:
            A new Workspace instance.
        """
        return Workspace(working_dir, checkpoint, keep_n)

    @staticmethod
    def get_path(working_dir: str, checkpoint: Optional[int] = None) -> Optional[str]:
        """Get the filesystem path for a checkpoint.

        Args:
            working_dir: The base directory of the workspace.
            checkpoint: The checkpoint number. None returns None, 0 returns
                working_dir, other values return working_dir/checkpoint.

        Returns:
            The full path to the checkpoint directory, or None if checkpoint is None.
        """
        if checkpoint is None:
            return None
        elif checkpoint == 0:
            return working_dir
        return os.path.join(working_dir, str(checkpoint))

    def __init__(self, working_dir: str, checkpoint: int = 0, keep_n: int = 0):
        """Initialize a workspace.

        Args:
            working_dir: The base directory for the workspace. Created if it
                doesn't exist.
            checkpoint: The checkpoint to load. 0 means use working_dir directly,
                otherwise uses the specified or latest checkpoint.
            keep_n: Number of checkpoints to retain. 0 disables checkpointing.
        """
        self.working_dir: str = working_dir
        self.keep_n: int = keep_n
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        # Scan for existing checkpoints (exclude error checkpoints)
        self.checkpoints = sorted(
            (int(x.name) for x in os.scandir(self.working_dir) if x.is_dir() and not x.name.startswith("0__err_")),
            reverse=True,
        )
        if self.checkpoints:
            # Use specified checkpoint or the most recent one
            self.current_load_checkpoint = checkpoint if checkpoint else self.checkpoints[0]
        else:
            self.current_load_checkpoint = checkpoint
        self.save_checkpoint: Optional[int] = None
        self.failed_checkpoints: List[str] = []

    def __del__(self):
        """Cleanup method called when the workspace is destroyed.

        Performs two cleanup operations:
        1. Renames failed checkpoints with "0__err_" prefix for debugging
        2. Removes old checkpoints exceeding the retention limit (keep_n)
        """
        # Mark failed checkpoints with error prefix
        for checkpoint in self.failed_checkpoints:
            old_path = os.path.join(self.working_dir, checkpoint)
            new_path = os.path.join(self.working_dir, f"0__err_{checkpoint}")
            os.rename(old_path, new_path)

        # Remove old checkpoints beyond retention limit
        if self.keep_n > 0:
            checkpoints = sorted((x.name for x in os.scandir(self.working_dir) if x.is_dir()), reverse=True)
            for checkpoint in checkpoints[self.keep_n + 1 :]:
                shutil.rmtree(os.path.join(self.working_dir, str(checkpoint)))

    def make_for(self, namespace: str) -> "Namespace":
        """Create a namespace within this workspace.

        Args:
            namespace: The namespace identifier.

        Returns:
            A Namespace instance associated with this workspace.
        """
        return Namespace(self, namespace)

    def get_load_path(self) -> Optional[str]:
        """Get the path to load data from.

        Returns:
            The path to the current checkpoint directory, or None if the
            working directory is empty (no checkpoint files exist).
        """
        load_path = self.get_path(self.working_dir, self.current_load_checkpoint)
        # Return None if using working_dir and it's empty
        if load_path == self.working_dir and len([x for x in os.scandir(load_path) if x.is_file()]) == 0:
            return None
        return load_path


    def get_save_path(self) -> str:
        """Get the path to save data to.

        Lazily initializes the save checkpoint on first call. If checkpointing
        is enabled (keep_n > 0), creates a timestamp-based checkpoint directory.
        Otherwise, uses the working directory directly.

        Returns:
            The path to the save checkpoint directory.
        """
        if self.save_checkpoint is None:
            if self.keep_n > 0:
                # Create timestamp-based checkpoint
                self.save_checkpoint = int(time.time())
            else:
                # Use working directory directly
                self.save_checkpoint = 0
        save_path = self.get_path(self.working_dir, self.save_checkpoint)

        assert save_path is not None, "Save path cannot be None."

        # Ensure the directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path)

    def _rollback(self) -> bool:
        """Roll back to the next older checkpoint.

        Attempts to roll back to the next checkpoint in the sorted checkpoint
        list. Marks the current checkpoint as failed if applicable.

        Returns:
            True if rollback was attempted (even if no checkpoint was found),
            False if there's no current checkpoint to roll back from.
        """
        if self.current_load_checkpoint is None:
            return False
        # Find the next older checkpoint (checkpoints are sorted newest-first)
        try:
            self.current_load_checkpoint = next(x for x in self.checkpoints if x < self.current_load_checkpoint)
            logger.warning("Rolling back to checkpoint: %s", self.current_load_checkpoint)
        except (StopIteration, ValueError):
            self.current_load_checkpoint = None
            logger.warning("No checkpoints to rollback to. Last checkpoint tried: %s", self.current_load_checkpoint)

        return True

    async def with_checkpoints(self, fn: Callable[[], Any]) -> Any:
        """Execute a function with automatic checkpoint rollback on failure.

        Attempts to execute the given function. If it fails, automatically rolls
        back to the next older checkpoint and retries. Continues until either
        the function succeeds or no more checkpoints are available.

        Args:
            fn: An async callable to execute. Should load data from the current
                checkpoint.

        Returns:
            The return value of the function on success.

        Raises:
            InvalidStorageError: If all checkpoints fail and no valid storage
                can be created.
        """
        while True:
            try:
                return await fn()
            except Exception as e:
                logger.warning("Error occurred loading checkpoint: %s", e)
                # Mark current checkpoint as failed
                if self.current_load_checkpoint is not None:
                    self.failed_checkpoints.append(str(self.current_load_checkpoint))
                # Try to roll back to an older checkpoint
                if self._rollback() is False:
                    break
        raise InvalidStorageError("No valid checkpoints to load or default storages cannot be created.")


class Namespace:
    """Provides namespace-based resource management within a workspace.

    A namespace creates a logical separation of storage resources by prefixing
    resource filenames with the namespace identifier. This allows multiple storage
    instances to coexist within the same workspace without name conflicts.

    Resource file naming pattern: {namespace}_{resource_name}

    Attributes:
        namespace: The namespace identifier, or None for unnamespaced resources.
        workspace: The workspace instance this namespace belongs to.
    """

    def __init__(self, workspace: Workspace, namespace: Optional[str] = None):
        """Initialize a namespace.

        Args:
            workspace: The workspace instance to associate with.
            namespace: The namespace identifier. Can be None for unnamespaced
                operations.
        """
        self.namespace = namespace
        self.workspace = workspace

    def get_load_path(self, resource_name: str) -> Optional[str]:
        """Get the full path to load a namespaced resource.

        Combines the workspace's load path with the namespace and resource name
        to create a fully qualified file path.

        Args:
            resource_name: The name of the resource file (e.g., "data.pkl").

        Returns:
            The full path to the resource file, or None if the workspace has
            no load path available.

        Raises:
            AssertionError: If namespace is not set.
        """
        assert self.namespace is not None, "Namespace must be set to get resource load path."
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        return os.path.join(load_path, f"{self.namespace}_{resource_name}")

    def get_save_path(self, resource_name: str) -> str:
        """Get the full path to save a namespaced resource.

        Combines the workspace's save path with the namespace and resource name
        to create a fully qualified file path.

        Args:
            resource_name: The name of the resource file (e.g., "data.pkl").

        Returns:
            The full path to the resource file.

        Raises:
            AssertionError: If namespace is not set.
        """
        assert self.namespace is not None, "Namespace must be set to get resource save path."
        return os.path.join(self.workspace.get_save_path(), f"{self.namespace}_{resource_name}")
