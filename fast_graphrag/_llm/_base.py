"""Base classes and utilities for Language Model and Embedding services.

This module provides the foundational abstract classes and utility functions for
integrating different Language Model (LLM) and embedding service providers into
the fast-graphrag framework.

The main components include:
    - BaseLLMService: Abstract base class for Language Model implementations
    - BaseEmbeddingService: Abstract base class for embedding model implementations
    - format_and_send_prompt: Helper function to format prompts and send to LLM
    - NoopAsyncContextManager: A no-operation async context manager for compatibility

Classes:
    BaseLLMService: Abstract base class defining the interface for LLM services
    BaseEmbeddingService: Abstract base class defining the interface for embedding services
    NoopAsyncContextManager: A context manager that performs no operations

Functions:
    format_and_send_prompt: Retrieves a prompt from the PROMPTS registry, formats it,
        and sends it to an LLM service with optional system prompts
    count_tokens: Counts the number of tokens in a text string
    is_within_token_limit: Checks if text fits within a specified token limit
    encode: Gets embedding representations for input texts
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel

from fast_graphrag._models import BaseModelAlias
from fast_graphrag._prompt import PROMPTS

T_model = TypeVar("T_model", bound=Union[BaseModel, BaseModelAlias])
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


async def format_and_send_prompt(
  prompt_key: str,
  llm: "BaseLLMService",
  format_kwargs: dict[str, Any],
  response_model: Type[T_model],
  **args: Any,
) -> Tuple[T_model, list[dict[str, str]]]:
  """Get a prompt, format it with the supplied args, and send it to the LLM.

  If a system prompt is provided (i.e. PROMPTS contains a key named
  '{prompt_key}_system'), it will use both the system and prompt entries:
      - System prompt: PROMPTS[prompt_key + '_system']
      - Message prompt: PROMPTS[prompt_key + '_prompt']

  Otherwise, it will default to using the single prompt defined by:
      - PROMPTS[prompt_key]

  Args:
      prompt_key (str): The key for the prompt in the PROMPTS dictionary.
      llm (BaseLLMService): The LLM service to use for sending the message.
      response_model (Type[T_model]): The expected response model.
      format_kwargs (dict[str, Any]): Dictionary of arguments to format the prompt.
      model (str | None): The model to use for the LLM. Defaults to None.
      max_tokens (int | None): The maximum number of tokens for the response. Defaults to None.
      **args (Any): Additional keyword arguments to pass to the LLM.

  Returns:
      Tuple[T_model, list[dict[str, str]]]: The response from the LLM.
  """
  system_key = prompt_key + "_system"

  if system_key in PROMPTS:
    # Use separate system and prompt entries
    system = PROMPTS[system_key]
    prompt = PROMPTS[prompt_key + "_prompt"]
    formatted_system = system.format(**format_kwargs)
    formatted_prompt = prompt.format(**format_kwargs)
    return await llm.send_message(
      system_prompt=formatted_system, prompt=formatted_prompt, response_model=response_model, **args
    )
  else:
    # Default: use the single prompt entry
    prompt = PROMPTS[prompt_key]
    formatted_prompt = prompt.format(**format_kwargs)
    return await llm.send_message(prompt=formatted_prompt, response_model=response_model, **args)


@dataclass
class BaseLLMService:
  """Abstract base class for Language Model service implementations.

  This class defines the interface and common functionality for integrating
  different Language Model providers (e.g., OpenAI, Anthropic, local LLMs)
  into the fast-graphrag framework. Subclasses must implement the send_message
  method to provide specific LLM integration logic.

  The class includes built-in token counting and rate limiting capabilities
  to help manage API usage and prevent throttling.

  Attributes:
      model (str): The name of the model to use (e.g., "gpt-4", "claude-3").
          This is a required field that must be provided when initializing.
      base_url (Optional[str]): The base URL for the API endpoint. Defaults to None,
          which typically means using the default endpoint for the provider.
      api_key (Optional[str]): The API key for authentication. Defaults to None,
          which typically means using an environment variable for credentials.
      llm_async_client (Any): An async client instance for making API requests.
          This is initialized separately and not provided during dataclass init.
          Defaults to None.
      max_requests_concurrent (int): The maximum number of concurrent requests
          allowed. Defaults to the CONCURRENT_TASK_LIMIT environment variable
          (1024 if not set).
      max_requests_per_minute (int): The maximum number of requests allowed per
          minute. Defaults to 500 to stay within typical API rate limits.
      max_requests_per_second (int): The maximum number of requests allowed per
          second. Defaults to 60 to prevent sudden request spikes.
      rate_limit_concurrency (bool): Whether to enforce concurrent request limits.
          Defaults to True.
      rate_limit_per_minute (bool): Whether to enforce per-minute rate limits.
          Defaults to False.
      rate_limit_per_second (bool): Whether to enforce per-second rate limits.
          Defaults to False.
  """

  model: str = field()
  base_url: Optional[str] = field(default=None)
  api_key: Optional[str] = field(default=None)
  llm_async_client: Any = field(init=False, default=None)
  max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
  max_requests_per_minute: int = field(default=500)
  max_requests_per_second: int = field(default=60)
  rate_limit_concurrency: bool = field(default=True)
  rate_limit_per_minute: bool = field(default=False)
  rate_limit_per_second: bool = field(default=False)

  def count_tokens(self, text: str) -> int:
    """Counts the number of tokens in the provided text.

    This method uses a simple Unicode-aware regex pattern to count tokens,
    treating each word and punctuation mark as a separate token. This provides
    a quick approximate token count without requiring model-specific tokenizers.

    Args:
        text (str): The input text to tokenize and count.

    Returns:
        int: The total number of tokens in the text.

    Note:
        This is a simplified tokenization that may not exactly match the actual
        token count from specific LLM providers. For precise token counting,
        consider using the model provider's official tokenizer.
    """
    return len(TOKEN_PATTERN.findall(text))

  def is_within_token_limit(self, text: str, token_limit: int) -> Union[int, bool]:
    """Checks if the provided text fits within the specified token limit.

    This is a lightweight utility method for validating that text will not exceed
    the model's context length or other token constraints. It performs an early
    exit if the token count exceeds the limit.

    Args:
        text (str): The input text to check.
        token_limit (int): The maximum number of tokens allowed.

    Returns:
        Union[int, bool]: The token count (int) if the text is within the limit
            and token_count <= token_limit, otherwise False. This allows calling
            code to both verify the limit and retrieve the count in one call.

    Example:
        >>> llm = BaseLLMService(model="gpt-4")
        >>> result = llm.is_within_token_limit("Hello world", 100)
        >>> if result:  # Returns token count if within limit
        ...     print(f"Text uses {result} tokens")
        >>> else:  # Returns False if exceeds limit
        ...     print("Text exceeds token limit")
    """
    token_count = self.count_tokens(text)
    return token_count if token_count <= token_limit else False

  async def send_message(
    self,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    response_model: Type[T_model] | None = None,
    **kwargs: Any,
  ) -> Tuple[T_model, list[dict[str, str]]]:
    """Sends a message to the language model and receives a structured response.

    This is the primary abstract method that subclasses must implement. It handles
    the core communication with the LLM service, including sending prompts,
    maintaining conversation history, and parsing responses into structured models.

    Args:
        prompt (str): The main input message or query to send to the language model.
            This will typically be the user's question or instruction. When combined
            with a system_prompt, it becomes the complete message context.
        system_prompt (str, optional): An optional system-level prompt that sets the
            behavior, context, or personality for the conversation. This is typically
            used to guide the model's response style. Defaults to None.
        history_messages (list[dict[str, str]], optional): A list of previous messages
            in the conversation, maintaining context from prior interactions. Each
            message should be a dictionary with 'role' and 'content' keys. Defaults to None.
        response_model (Type[T_model], optional): A Pydantic BaseModel or BaseModelAlias
            class that defines the expected structure of the response. If provided,
            the LLM's response will be parsed and validated against this model.
            Defaults to None, in which case a string response may be returned.
        **kwargs (Any): Additional keyword arguments specific to the LLM implementation.
            Common examples might include: max_tokens (int), temperature (float),
            top_p (float), model (str), or any other provider-specific parameters.

    Returns:
        Tuple[T_model, list[dict[str, str]]]: A tuple containing:
            - The response from the language model, parsed into the response_model
              type if one was provided
            - An updated list of conversation messages including the current exchange

    Raises:
        NotImplementedError: This method must be implemented by subclasses.
            Concrete implementations should raise provider-specific exceptions
            for API errors, timeout, authentication failures, etc.

    Example:
        Subclasses would implement this like:
        >>> class OpenAILLM(BaseLLMService):
        ...     async def send_message(self, prompt, system_prompt=None, ...):
        ...         # Implementation specific to OpenAI API
        ...         response = await self.llm_async_client.create(...)
        ...         return parsed_response, messages
    """
    raise NotImplementedError


@dataclass
class BaseEmbeddingService:
  """Abstract base class for embedding model service implementations.

  This class defines the interface and common functionality for integrating
  different embedding model providers (e.g., OpenAI, Hugging Face, Cohere)
  into the fast-graphrag framework. Subclasses must implement the encode
  method to provide specific embedding service integration logic.

  Embedding services convert text input into high-dimensional vector
  representations that capture semantic meaning. These embeddings are
  essential for similarity search, clustering, and retrieval operations
  in knowledge graphs and RAG systems.

  The class includes rate limiting capabilities to manage API usage and
  prevent throttling when working with external embedding services.

  Attributes:
      embedding_dim (int): The dimensionality of the embedding vectors produced
          by the model. Defaults to 1536, which matches OpenAI's text-embedding-3
          models. Other models may use different dimensions (e.g., 768, 1024).
      model (Optional[str]): The name of the embedding model to use.
          Defaults to "text-embedding-3-small" (OpenAI). Examples of other
          models include "text-embedding-3-large", "all-MiniLM-L6-v2", etc.
      base_url (Optional[str]): The base URL for the API endpoint. Defaults to None,
          which typically means using the default endpoint for the provider.
          Useful for self-hosted embedding services or proxy configurations.
      api_key (Optional[str]): The API key for authentication. Defaults to None,
          which typically means using an environment variable for credentials.
      max_requests_concurrent (int): The maximum number of concurrent embedding
          requests allowed. Defaults to the CONCURRENT_TASK_LIMIT environment
          variable (1024 if not set).
      max_requests_per_minute (int): The maximum number of requests allowed per
          minute. Defaults to 500 (Tier 1 OpenAI RPM limit). Adjust based on
          your service tier or rate limit requirements.
      max_requests_per_second (int): The maximum number of requests allowed per
          second. Defaults to 100 to prevent sudden request spikes.
      rate_limit_concurrency (bool): Whether to enforce concurrent request limits.
          Defaults to True.
      rate_limit_per_minute (bool): Whether to enforce per-minute rate limits.
          Defaults to True (recommended for API-based services).
      rate_limit_per_second (bool): Whether to enforce per-second rate limits.
          Defaults to False (per-minute limit is usually sufficient).
      embedding_async_client (Any): An async client instance for making API requests.
          This is initialized separately and not provided during dataclass init.
          Defaults to None.
  """

  embedding_dim: int = field(default=1536)
  model: Optional[str] = field(default="text-embedding-3-small")
  base_url: Optional[str] = field(default=None)
  api_key: Optional[str] = field(default=None)
  max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
  max_requests_per_minute: int = field(default=500)  # Tier 1 OpenAI RPM
  max_requests_per_second: int = field(default=100)
  rate_limit_concurrency: bool = field(default=True)
  rate_limit_per_minute: bool = field(default=True)
  rate_limit_per_second: bool = field(default=False)

  embedding_async_client: Any = field(init=False, default=None)

  async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Encodes input texts into high-dimensional embedding vectors.

    This is the primary abstract method that subclasses must implement. It converts
    a list of text strings into their semantic vector representations using the
    configured embedding model. The resulting embeddings can be used for similarity
    search, clustering, and other vector-based operations.

    Args:
        texts (list[str]): A list of text strings to encode. Each string will be
            converted into an embedding vector. The method should handle:
            - Empty lists (return empty array)
            - Single-item lists
            - Large lists (though batch processing should respect rate limits)
        model (Optional[str]): The specific embedding model to use for this request.
            If provided, this overrides the default model configured in the service.
            Defaults to None, which uses the model specified during initialization.
            Useful for A/B testing or trying different embedding models.

    Returns:
        np.ndarray[Any, np.dtype[np.float32]]: A NumPy array of shape (N, embedding_dim)
            containing the embedding vectors, where:
            - N is the number of input texts
            - embedding_dim is the dimensionality of each embedding (e.g., 1536)
            - dtype is float32 for memory efficiency

            The array can be accessed as:
            >>> embeddings = await embedding_service.encode(["hello", "world"])
            >>> embeddings.shape  # (2, 1536)
            >>> embeddings[0]     # First embedding vector
            >>> embeddings[1]     # Second embedding vector

    Raises:
        NotImplementedError: This method must be implemented by subclasses.
            Concrete implementations should raise provider-specific exceptions
            for API errors, timeout, invalid input, authentication failures, etc.
            Common exceptions might include:
            - ValueError: For invalid input texts or model names
            - asyncio.TimeoutError: For service timeouts
            - aiohttp.ClientError: For network-related errors

    Example:
        Subclasses would implement this like:
        >>> class OpenAIEmbedding(BaseEmbeddingService):
        ...     async def encode(self, texts, model=None):
        ...         response = await self.embedding_async_client.create(
        ...             input=texts,
        ...             model=model or self.model
        ...         )
        ...         vectors = [item.embedding for item in response.data]
        ...         return np.array(vectors, dtype=np.float32)

    Note:
        - Batch size and rate limiting should be handled by subclass implementations
        - The returned embeddings are typically normalized or unnormalized depending
          on the model (check model documentation)
        - For large-scale encoding, subclasses should implement appropriate batching
          and caching strategies
    """
    raise NotImplementedError


class NoopAsyncContextManager:
  """A no-operation async context manager for compatibility and flexibility.

  This context manager performs no operations on entry or exit. It's useful in
  scenarios where:
  - A context manager is required by the API but no cleanup is needed
  - Conditional context manager usage (e.g., use a real context manager only
    when needed, otherwise use this no-op version)
  - Wrapping operations that may or may not require resource management

  Example:
      >>> async with NoopAsyncContextManager() as noop:
      ...     # This block executes without any special setup or cleanup
      ...     await some_async_operation()

  This is typically used in resource management where cleanup might be optional
  or conditional based on configuration or runtime state.
  """

  async def __aenter__(self) -> "NoopAsyncContextManager":
    """Enter the async context manager.

    Returns:
        NoopAsyncContextManager: Returns itself to allow assignment in 'as' clause.
    """
    return self

  async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
    """Exit the async context manager.

    Performs no cleanup operations on exit. Any exceptions that occurred
    within the context block are not suppressed.

    Args:
        exc_type (Any): The exception type if an exception occurred, None otherwise.
        exc (Any): The exception instance if an exception occurred, None otherwise.
        tb (Any): The traceback if an exception occurred, None otherwise.
    """
    pass
