"""VoyageAI Embedding Service Module.

This module provides integration with VoyageAI's embedding API for generating
dense vector representations of text. It implements rate limiting, batching, and
async request handling to efficiently embed large volumes of text data.

The VoyageAIEmbeddingService class handles:
- Asynchronous API requests to VoyageAI endpoints
- Request batching to stay within API limits (max 128 elements per batch)
- Rate limiting at multiple levels (concurrent, per-minute, per-second)
- Error handling and retry logic
"""

import asyncio
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Optional

import numpy as np
from aiolimiter import AsyncLimiter
from voyageai import client_async
from voyageai.object.embeddings import EmbeddingsObject

from fast_graphrag._utils import logger

from ._base import BaseEmbeddingService, NoopAsyncContextManager


@dataclass
class VoyageAIEmbeddingService(BaseEmbeddingService):
  """VoyageAI Embedding Service for generating dense vector representations.

  This class provides async methods to interact with VoyageAI's embedding API.
  It inherits from BaseEmbeddingService and implements rate limiting and batching
  to optimize API usage and stay within quota constraints.

  Attributes:
      embedding_dim (int): The dimensionality of the embedding vectors. Default is 1024.
      max_elements_per_request (int): Maximum number of texts to embed in a single API
          request. VoyageAI API limit is 128. Default is 128.
      model (Optional[str]): The name of the VoyageAI embedding model to use.
          Default is "voyage-3".
      api_version (Optional[str]): API version to use. Default is None.
      max_requests_concurrent (int): Maximum number of concurrent API requests allowed.
          Controlled by CONCURRENT_TASK_LIMIT environment variable. Default is 1024.
      max_requests_per_minute (int): Rate limit for requests per minute. Default is 1800.
      max_requests_per_second (int): Rate limit for requests per second. Default is 100.
      rate_limit_per_second (bool): Whether to enable per-second rate limiting.
          Default is False.
  """

  # Output dimensionality of embedding vectors
  embedding_dim: int = field(default=1024)
  # Maximum texts per batch request (VoyageAI API constraint)
  max_elements_per_request: int = field(default=128)
  # Model name for embeddings (e.g., "voyage-3")
  model: Optional[str] = field(default="voyage-3")
  # Optional API version specification
  api_version: Optional[str] = field(default=None)
  # Concurrent request semaphore limit (from env var or default to 1024)
  max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
  # Rate limit for requests per minute
  max_requests_per_minute: int = field(default=1800)
  # Rate limit for requests per second
  max_requests_per_second: int = field(default=100)
  # Flag to enable second-level rate limiting
  rate_limit_per_second: bool = field(default=False)

  def __post_init__(self):
    """Initialize the VoyageAI embedding service after dataclass initialization.

    This method is called automatically after the dataclass __init__ method.
    It sets up rate limiters and the async VoyageAI client with the following:
    - Concurrent request semaphore for limiting parallel API calls
    - Per-minute rate limiter to respect API quotas
    - Per-second rate limiter for fine-grained rate control
    - Async HTTP client with retry logic (4 max retries)
    """
    # Set up concurrent request semaphore or use no-op context manager if disabled
    self.embedding_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    # Set up per-minute rate limiter (60 second window) or use no-op context manager if disabled
    self.embedding_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    # Set up per-second rate limiter (1 second window) or use no-op context manager if disabled
    self.embedding_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )
    # Initialize VoyageAI async client with API key and retry configuration
    self.embedding_async_client: client_async.AsyncClient = client_async.AsyncClient(
      api_key=self.api_key, max_retries=4
    )
    logger.debug("Initialized VoyageAIEmbeddingService.")

  async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Get the embedding representation of input texts using VoyageAI API.

    This method handles batching of large text lists (respecting the 128 element
    per-batch limit) and makes concurrent async requests to the VoyageAI API.
    All embeddings are gathered and returned as a single numpy array.

    Args:
        texts (list[str]): List of text strings to embed.
        model (Optional[str]): The name of the VoyageAI model to use for embeddings.
            If not provided, defaults to the model specified in the configuration.

    Returns:
        np.ndarray: A 2D numpy array of shape (len(texts), embedding_dim) containing
            the embedding vectors. Data type is float32.

    Raises:
        ValueError: If model name is not provided and no default model is configured.
        Exception: Any exception from the VoyageAI API client (network errors, auth failures, etc.)
    """
    try:
      logger.debug(f"Getting embedding for texts: {texts}")
      # Use provided model or fall back to default model
      model = model or self.model
      if model is None:
        raise ValueError("Model name must be provided.")

      # Split texts into batches respecting the max_elements_per_request limit
      # This ensures we don't exceed VoyageAI API constraints
      batched_texts = [
        texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
        for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
      ]
      # Execute all embedding requests concurrently using asyncio.gather
      response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])

      # Extract embeddings from all responses and flatten into a single list
      data = chain(*[r.embeddings for r in response])
      # Convert to numpy array for efficient numerical operations
      embeddings = np.array(list(data))
      logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

      return embeddings
    except Exception:
      logger.exception("An error occurred:", exc_info=True)
      raise

  async def _embedding_request(self, input: List[str], model: str) -> EmbeddingsObject:
    """Make a single embedding request to VoyageAI API with rate limiting.

    This is a private helper method that handles a single batch of text embeddings.
    It applies all configured rate limiters (concurrent, per-minute, per-second)
    before making the actual API request. Rate limiters are applied as nested
    context managers to ensure proper resource management.

    Args:
        input (List[str]): A batch of text strings to embed. Should not exceed
            max_elements_per_request (128 elements).
        model (str): The VoyageAI model name to use for embedding.

    Returns:
        EmbeddingsObject: The response object from VoyageAI containing the embedding
            vectors for the input texts.

    Note:
        This method applies rate limiting in the following order:
        1. Concurrent request limit (semaphore)
        2. Per-minute request limit
        3. Per-second request limit
    """
    # Acquire concurrent request semaphore to limit parallel API calls
    async with self.embedding_max_requests_concurrent:
      # Apply per-minute rate limiting to respect API quotas
      async with self.embedding_per_minute_limiter:
        # Apply per-second rate limiting for fine-grained control
        async with self.embedding_per_second_limiter:
          # Make the actual async API call to VoyageAI with specified embedding dimension
          return await self.embedding_async_client.embed(model=model, texts=input, output_dimension=self.embedding_dim)
