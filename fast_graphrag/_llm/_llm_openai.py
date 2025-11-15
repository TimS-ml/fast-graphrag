"""OpenAI-compatible LLM Services module.

This module provides asynchronous integrations with OpenAI's APIs for language model
interactions and text embeddings. It supports both the OpenAI API and Azure OpenAI
deployments with rate limiting, retry logic, and structured output capabilities.

Key Components:
    - OpenAILLMService: Asynchronous LLM service for text generation with support for
      structured outputs using Pydantic models.
    - OpenAIEmbeddingService: Asynchronous embedding service for generating vector
      representations of text.

Features:
    - Async/await support for non-blocking API calls
    - Rate limiting (per-minute, per-second, concurrent request limits)
    - Exponential backoff retry logic for transient failures
    - Token counting using tiktoken for cost estimation
    - Support for both standard OpenAI and Azure OpenAI endpoints
    - Structured output parsing with Pydantic models via instructor
"""

import asyncio
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Literal, Optional, Tuple, Type, cast

import instructor
import numpy as np
import tiktoken
from aiolimiter import AsyncLimiter
from openai import APIConnectionError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
  AsyncRetrying,
  retry,
  retry_if_exception_type,
  stop_after_attempt,
  wait_exponential,
)

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BaseModelAlias
from fast_graphrag._utils import logger

from ._base import TOKEN_PATTERN, BaseEmbeddingService, BaseLLMService, NoopAsyncContextManager, T_model

# Default timeout for OpenAI API requests in seconds
TIMEOUT_SECONDS = 180.0


@dataclass
class OpenAILLMService(BaseLLMService):
  """Asynchronous LLM service for OpenAI and Azure OpenAI models.

  This class provides a high-level interface to OpenAI's chat completion API
  with support for both standard OpenAI and Azure OpenAI endpoints. It includes
  structured output parsing via the instructor library, automatic token counting,
  and comprehensive rate limiting to prevent API quota exhaustion.

  Attributes:
      model (str): The name of the language model to use (default: "gpt-4o-mini").
      mode (instructor.Mode): The response parsing mode for instructor library
          (default: JSON mode for structured outputs).
      client (Literal["openai", "azure"]): The API provider to use (default: "openai").
      api_version (Optional[str]): Azure API version string, required when client="azure".
  """

  model: str = field(default="gpt-4o-mini")
  mode: instructor.Mode = field(default=instructor.Mode.JSON)
  client: Literal["openai", "azure"] = field(default="openai")
  api_version: Optional[str] = field(default=None)

  def __post_init__(self):
    """Initialize the OpenAI LLM service with rate limiting and async client setup.

    This method initializes:
    - Token encoding for the specified model (via tiktoken)
    - Rate limiting semaphores and limiters for concurrent requests
    - The instructor-wrapped OpenAI or Azure OpenAI async client

    Raises:
        ValueError: If client type is neither "openai" nor "azure".
        AssertionError: If Azure is selected but base_url or api_version is missing.
    """
    # Initialize tokenizer for accurate token counting and cost estimation
    self.encoding = None
    try:
      # Attempt to load the official tokenizer for the specified model
      self.encoding = tiktoken.encoding_for_model(self.model)
    except Exception as e:
      # Fall back to naive tokenization (splitting by whitespace/punctuation) if model not recognized
      logger.info(f"LLM: failed to load tokenizer for model '{self.model}' ({e}). Falling back to naive tokenization.")
      self.encoding = None

    # Initialize rate limiting controls to prevent API quota exhaustion
    # Concurrent request semaphore limits total simultaneous API calls
    self.llm_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    # Per-minute rate limit: limits requests within a 60-second window
    self.llm_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    # Per-second rate limit: limits requests within a 1-second window
    self.llm_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )

    # Initialize the appropriate OpenAI client based on configuration
    if self.client == "azure":
      # Azure OpenAI requires both a base URL (endpoint) and API version
      assert (
        self.base_url is not None and self.api_version is not None
      ), "Azure OpenAI requires a base url and an api version."
      # Create AsyncAzureOpenAI client and wrap it with instructor for structured outputs
      self.llm_async_client = instructor.from_openai(
        AsyncAzureOpenAI(
          azure_endpoint=self.base_url,
          api_key=self.api_key,
          api_version=self.api_version,
          timeout=TIMEOUT_SECONDS,
        ),
        mode=self.mode,
      )
    elif self.client == "openai":
      # Create standard OpenAI client and wrap it with instructor for structured outputs
      self.llm_async_client = instructor.from_openai(
        AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_SECONDS), mode=self.mode
      )
    else:
      raise ValueError("Invalid client type. Must be 'openai' or 'azure'")

    logger.debug("Initialized OpenAILLMService with patched OpenAI client.")

  def count_tokens(self, text: str) -> int:
    """Count the number of tokens in the given text.

    Uses the official tokenizer for the model if available, otherwise falls back
    to a simple regex-based tokenization method. Token counts are useful for
    estimating API costs and managing context window limits.

    Args:
        text (str): The text to tokenize.

    Returns:
        int: The total number of tokens in the text.
    """
    if self.encoding:
      # Use model-specific tokenizer for accurate token counting
      return len(self.encoding.encode(text))
    else:
      # Fallback: use regex pattern to split on whitespace and punctuation
      return len(TOKEN_PATTERN.findall(text))

  async def send_message(
    self,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    response_model: Type[T_model] | None = None,
    **kwargs: Any,
  ) -> Tuple[T_model, list[dict[str, str]]]:
    """Send a message to the OpenAI language model and receive a response.

    Sends a message to the configured OpenAI or Azure OpenAI model with optional
    system context and conversation history. Supports structured output parsing
    by accepting a Pydantic model for response validation. All API calls are
    subject to configured rate limiting and retry logic.

    Args:
        prompt (str): The user message to send to the language model.
        system_prompt (str, optional): System prompt to set the AI's behavior and context.
            Defaults to None (no system prompt).
        history_messages (list[dict[str, str]], optional): Previous messages in the conversation
            with 'role' and 'content' keys (e.g., [{"role": "user", "content": "..."}, ...]).
            Defaults to None (no history).
        response_model (Type[T_model], optional): A Pydantic model class to parse and validate
            the model's response. If provided, the response will be structured according to
            this model. Defaults to None (returns raw string response).
        **kwargs: Additional arguments passed to the OpenAI API (e.g., temperature, top_p,
            max_tokens, presence_penalty, frequency_penalty).

    Returns:
        Tuple[T_model, list[dict[str, str]]]: A tuple containing:
            - The parsed response (structured according to response_model if provided,
              otherwise a string response)
            - The complete message history including the system prompt, user message,
              and assistant response

    Raises:
        LLMServiceNoResponseError: If the model returns an empty or None response.
        Exception: Any exceptions from the OpenAI API after exhausting retry attempts.
    """
    # Apply all three rate limiting controls in nested context managers
    async with self.llm_max_requests_concurrent:
      async with self.llm_per_minute_limiter:
        async with self.llm_per_second_limiter:
          try:
            logger.debug(f"Sending message with prompt: {prompt}")
            model = self.model
            # Build message list following OpenAI API format
            messages: list[dict[str, str]] = []

            # Add system prompt if provided to guide model behavior
            if system_prompt:
              messages.append({"role": "system", "content": system_prompt})
              logger.debug(f"Added system prompt: {system_prompt}")

            # Add conversation history if provided for context
            if history_messages:
              messages.extend(history_messages)
              logger.debug(f"Added history messages: {history_messages}")

            # Add current user message as the last message
            messages.append({"role": "user", "content": prompt})

            # Make the API call to OpenAI with instructor wrapping for structured outputs
            # The instructor library intercepts this call to handle response_model validation
            llm_response: T_model = await self.llm_async_client.chat.completions.create(
              model=model,
              messages=messages,  # type: ignore
              # Handle both standard Pydantic models and BaseModelAlias wrapper models
              response_model=response_model.Model
              if response_model and issubclass(response_model, BaseModelAlias)
              else response_model,
              **kwargs,
              # Configure exponential backoff retry: max 3 attempts with delays from 4-10 seconds
              max_retries=AsyncRetrying(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)),
            )

            # Validate that we received a response from the API
            if not llm_response:
              logger.error("No response received from the language model.")
              raise LLMServiceNoResponseError("No response received from the language model.")

            # Append assistant response to message history for conversation continuity
            messages.append(
              {
                "role": "assistant",
                # Serialize response to JSON if it's a Pydantic model, otherwise convert to string
                "content": llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response),
              }
            )
            logger.debug(f"Received response: {llm_response}")

            # Convert BaseModelAlias responses back to dataclass for compatibility
            if response_model and issubclass(response_model, BaseModelAlias):
              llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

            # Return both the parsed response and the complete message history
            return llm_response, messages
          except Exception:
            logger.exception("An error occurred:", exc_info=True)
            raise


@dataclass
class OpenAIEmbeddingService(BaseEmbeddingService):
  """Asynchronous embedding service for OpenAI and Azure OpenAI models.

  This class provides a high-level interface to OpenAI's text embedding API
  for generating dense vector representations of text. It supports both standard
  OpenAI and Azure OpenAI endpoints with batching, rate limiting, and retry logic
  to handle API quotas efficiently.

  Attributes:
      embedding_dim (int): The dimensionality of the embedding vectors
          (default: 1536 for text-embedding-3-small).
      max_elements_per_request (int): Maximum number of texts to embed in a single
          API request (default: 32). Texts are batched to respect this limit.
      model (Optional[str]): The name of the embedding model to use
          (default: "text-embedding-3-small").
      client (Literal["openai", "azure"]): The API provider to use (default: "openai").
      api_version (Optional[str]): Azure API version string, required when client="azure".
  """

  embedding_dim: int = field(default=1536)
  max_elements_per_request: int = field(default=32)
  model: Optional[str] = field(default="text-embedding-3-small")
  client: Literal["openai", "azure"] = field(default="openai")
  api_version: Optional[str] = field(default=None)

  def __post_init__(self):
    """Initialize the OpenAI embedding service with rate limiting and async client setup.

    This method initializes:
    - Rate limiting semaphores and limiters for concurrent API requests
    - The appropriate OpenAI or Azure OpenAI async client for embeddings

    Raises:
        ValueError: If client type is neither "openai" nor "azure".
        AssertionError: If Azure is selected but base_url or api_version is missing.
    """
    # Initialize rate limiting controls to prevent API quota exhaustion
    # Concurrent request semaphore limits total simultaneous API calls
    self.embedding_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    # Per-minute rate limit: limits requests within a 60-second window
    self.embedding_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    # Per-second rate limit: limits requests within a 1-second window
    self.embedding_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )

    # Initialize the appropriate OpenAI client based on configuration
    if self.client == "azure":
      # Azure OpenAI requires both a base URL (endpoint) and API version
      assert (
        self.base_url is not None and self.api_version is not None
      ), "Azure OpenAI requires a base url and an api version."
      # Create AsyncAzureOpenAI client for embeddings
      self.embedding_async_client = AsyncAzureOpenAI(
        azure_endpoint=self.base_url, api_key=self.api_key, api_version=self.api_version
      )
    elif self.client == "openai":
      # Create standard OpenAI client for embeddings
      self.embedding_async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
    else:
      raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
    logger.debug("Initialized OpenAIEmbeddingService with OpenAI client.")

  async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Get embedding vector representations for the input texts.

    Converts a list of text strings into dense vector representations (embeddings)
    using the OpenAI embedding API. Texts are automatically batched to respect
    max_elements_per_request limits, and batches are processed concurrently
    for efficiency.

    Args:
        texts (list[str]): A list of text strings to embed. Each string will be
            converted to a vector in the embedding space.
        model (Optional[str]): The name of the embedding model to use. If not provided,
            defaults to the model specified during service initialization. Defaults to None.

    Returns:
        np.ndarray[Any, np.dtype[np.float32]]: A 2D numpy array where each row is an
            embedding vector of shape (len(texts), embedding_dim). Each row corresponds
            to the embedding of the text at the same index in the input list.

    Raises:
        ValueError: If model name is not provided and no default model is configured.
        Exception: Any exceptions from the OpenAI API after exhausting retry attempts.
    """
    try:
      logger.debug(f"Getting embedding for texts: {texts}")
      # Use provided model or fall back to default model
      model = model or self.model
      if model is None:
        raise ValueError("Model name must be provided.")

      # Split texts into batches respecting the API's max_elements_per_request limit
      # This prevents exceeding API payload limits and improves request efficiency
      batched_texts = [
        texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
        for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
      ]
      # Process all batches concurrently for faster embedding generation
      response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])

      # Flatten the list of responses from multiple batch requests
      data = chain(*[r.data for r in response])
      # Extract embedding vectors from response objects and convert to numpy array
      embeddings = np.array([dp.embedding for dp in data])
      logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

      return embeddings
    except Exception:
      logger.exception("An error occurred:", exc_info=True)
      raise

  @retry(
    # Retry up to 3 times on failure
    stop=stop_after_attempt(3),
    # Use exponential backoff: 4-10 seconds between retries
    wait=wait_exponential(multiplier=1, min=4, max=10),
    # Only retry on rate limit, connection, and timeout errors (not on validation errors)
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, TimeoutError)),
  )
  async def _embedding_request(self, input: List[str], model: str) -> Any:
    """Make a single embedding request to the OpenAI API with rate limiting.

    This is an internal method that handles a single batch of texts. It applies
    all configured rate limiting controls before making the API call. The method
    is decorated with @retry to automatically handle transient failures with
    exponential backoff.

    Args:
        input (List[str]): A batch of text strings to embed. Should not exceed
            max_elements_per_request.
        model (str): The name of the embedding model to use.

    Returns:
        Any: The response object from the OpenAI API containing embeddings data.

    Raises:
        RateLimitError: If the API rate limit is exceeded (will be retried).
        APIConnectionError: If there's a connection error (will be retried).
        TimeoutError: If the request times out (will be retried).
    """
    # Apply all three rate limiting controls in nested context managers
    async with self.embedding_max_requests_concurrent:
      async with self.embedding_per_minute_limiter:
        async with self.embedding_per_second_limiter:
          # Make the embedding API call with specified dimensions and float format
          return await self.embedding_async_client.embeddings.create(
            model=model,
            input=input,
            # Set embedding dimensions (can be different from model's native dimension)
            dimensions=self.embedding_dim,
            # Request embeddings as floats for better precision
            encoding_format="float"
          )
