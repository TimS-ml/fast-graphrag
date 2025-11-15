"""Utility functions and decorators for the fast-graphrag library.

This module provides:
- Performance measurement decorators (timeit)
- Async function throttling and concurrency control
- Event loop management utilities
- Sparse matrix manipulation functions for efficient graph operations
- Embedding similarity calculations

These utilities support the core RAG (Retrieval-Augmented Generation) operations
by providing efficient data structures and async execution patterns.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from fast_graphrag._types import TIndex

# Logger for GraphRAG operations
logger = logging.getLogger("graphrag")

# Approximate ratio for converting token counts to character counts
# Used for estimating LLM context window usage
TOKEN_TO_CHAR_RATIO = 4


def timeit(func: Callable[..., Any]):
    """Decorator to measure and record execution time of async functions.

    This decorator wraps async functions to track their execution times,
    useful for performance profiling and optimization. All execution times
    are stored in a list attribute on the wrapper function.

    Args:
        func (Callable): The async function to be timed

    Returns:
        Callable: Wrapped function that records execution times

    Example:
        @timeit
        async def expensive_operation():
            await asyncio.sleep(1)

        # Access execution times via wrapper.execution_times
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        # Store execution time for later analysis
        wrapper.execution_times.append(duration)  # type: ignore
        return result

    # Initialize the execution times list
    wrapper.execution_times = []  # type: ignore
    return wrapper


def throttle_async_func_call(
    max_concurrent: int = 2048,
    stagger_time: Optional[float] = None,
    waiting_time: float = 0.001,
):
    """Decorator factory to throttle concurrent execution of async functions.

    This decorator limits the number of concurrent executions using a semaphore
    and optionally staggers calls to prevent overwhelming external services
    (e.g., LLM API rate limits).

    Args:
        max_concurrent (int): Maximum number of concurrent function executions.
            Defaults to 2048.
        stagger_time (Optional[float]): Optional delay in seconds before each
            function execution. Useful for rate limiting. Defaults to None.
        waiting_time (float): Reserved for future use. Currently unused.
            Defaults to 0.001.

    Returns:
        Callable: Decorator function that wraps async functions with throttling

    Example:
        @throttle_async_func_call(max_concurrent=10, stagger_time=0.1)
        async def call_llm_api(prompt):
            return await llm.complete(prompt)
    """
    _wrappedFn = TypeVar("_wrappedFn", bound=Callable[..., Any])

    def decorator(func: _wrappedFn) -> _wrappedFn:
        # Create a semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)

        @wraps(func)
        async def wait_func(*args: Any, **kwargs: Any) -> Any:
            # Acquire semaphore before executing function
            async with semaphore:
                try:
                    # Optional stagger to space out calls over time
                    if stagger_time:
                        await asyncio.sleep(stagger_time)
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in throttled function {func.__name__}: {e}")
                    raise e

        return wait_func  # type: ignore

    return decorator


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an asyncio event loop for the current thread.

    This function handles event loop acquisition in different contexts:
    - Main thread: Returns existing event loop
    - Sub-threads: Creates and sets a new event loop

    This is particularly useful when running async code in multi-threaded
    environments or when the event loop state is uncertain.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop

    Example:
        loop = get_event_loop()
        loop.run_until_complete(async_function())
    """
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        # This is necessary because sub-threads don't have a default event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def extract_sorted_scores(
    row_vector: csr_matrix,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Extract and sort non-zero scores from a sparse row vector.

    This function is used for ranking operations in the GraphRAG system,
    such as sorting entities or chunks by their relevance scores.

    Args:
        row_vector (csr_matrix): A sparse CSR matrix with at most one row.
            Typically represents similarity scores or relevance weights.

    Returns:
        Tuple containing:
            - NDArray[int64]: Indices of non-zero elements, sorted by score (descending)
            - NDArray[float32]: Corresponding scores, sorted in descending order

    Raises:
        AssertionError: If the input matrix has more than one row

    Example:
        >>> scores = csr_matrix([[0, 0.9, 0, 0.5, 0]])
        >>> indices, values = extract_sorted_scores(scores)
        >>> # indices: [1, 3], values: [0.9, 0.5]
    """
    assert row_vector.shape[0] <= 1, "The input matrix must be a row vector."

    # Handle empty matrix edge case
    if row_vector.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Step 1: Get the indices of non-zero elements
    non_zero_indices = row_vector.nonzero()[1]

    # Step 2: Extract the probabilities/scores of these indices
    probabilities = row_vector.data

    # Step 3: Use NumPy to create arrays for indices and probabilities
    indices_array = np.array(non_zero_indices)
    probabilities_array = np.array(probabilities)

    # Step 4: Sort the probabilities in descending order and get the sorted indices
    sorted_indices = np.argsort(probabilities_array)[::-1]

    # Step 5: Create sorted arrays for indices and probabilities
    sorted_indices_array = indices_array[sorted_indices]
    sorted_probabilities_array = probabilities_array[sorted_indices]

    return sorted_indices_array, sorted_probabilities_array


def csr_from_indices_list(
    data: List[List[Union[int, TIndex]]], shape: Tuple[int, int]
) -> csr_matrix:
    """Create a CSR (Compressed Sparse Row) matrix from a list of column indices.

    This function converts a list-of-lists representation of sparse matrix indices
    into an efficient CSR matrix format. Each inner list represents the column
    indices where the row has a value of 1. This is useful for representing
    relationships or connections in the knowledge graph.

    Args:
        data (List[List[Union[int, TIndex]]]): List where each element is a list
            of column indices for that row. Row i has values of 1 at columns data[i].
        shape (Tuple[int, int]): The desired shape (num_rows, num_cols) of the
            resulting sparse matrix.

    Returns:
        csr_matrix: A sparse matrix in CSR format with 1s at specified positions

    Example:
        >>> # Create a 3x5 matrix with connections:
        >>> # Row 0: columns [1, 3]
        >>> # Row 1: column [2]
        >>> # Row 2: columns [0, 4]
        >>> data = [[1, 3], [2], [0, 4]]
        >>> matrix = csr_from_indices_list(data, shape=(3, 5))
    """
    num_rows = len(data)

    # Flatten the list of lists and create corresponding row indices
    # Each row index is repeated for the number of connections it has
    row_indices = np.repeat(np.arange(num_rows), [len(row) for row in data])
    col_indices = np.concatenate(data) if num_rows > 0 else np.array([], dtype=np.int64)

    # Data values (all ones in this case, representing binary connections)
    values = np.broadcast_to(1, len(row_indices))

    # Create the CSR matrix with specified shape
    return csr_matrix((values, (row_indices, col_indices)), shape=shape)
