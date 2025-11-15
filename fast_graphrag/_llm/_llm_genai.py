"""Google Gemini AI LLM Services module for Gemini and Vertex AI endpoints.

This module provides comprehensive integration with Google's Gemini AI models and
Vertex AI platform, offering both language model generation and embedding services.

Main Components:
    GeminiLLMService: Service for sending messages and generating responses using
        Gemini models. Supports structured output with Pydantic models, system
        prompts, conversation history, and advanced safety configurations.
    GeminiEmbeddingService: Service for obtaining dense embedding vectors for
        input texts using specialized embedding models.

Key Features:
    - Asynchronous processing with full asyncio support
    - Retry handling via tenacity library for resilience against transient errors
    - Advanced rate limiting with per-second, per-minute, and concurrent request limits
    - Gemini-specific safety settings and content generation configurations
    - JSON response validation with automatic repair capabilities
    - Support for both Google Gemini API and Vertex AI endpoints
    - Local tokenizer integration for accurate token counting
    - Structured output with Pydantic model validation (instructor mode)

Safety and Compliance:
    - Configurable safety settings for various content categories
    - Support for Gemini's harm blocking thresholds
    - Error handling for rate limits and API constraints

Dependencies:
    - google-genai: Google Gemini SDK
    - instructor: Structured output validation library
    - tenacity: Retry mechanism library
    - aiolimiter: Asynchronous rate limiting
    - vertexai: Vertex AI tokenizer for token counting
"""

import asyncio
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple, Type, cast

import instructor
import numpy as np
import requests  # used to catch requests.exceptions.ConnectionError
from aiolimiter import AsyncLimiter
from google import genai  # type: ignore
from google.genai import errors, types  # type: ignore
from json_repair import repair_json
from pydantic import BaseModel, TypeAdapter, ValidationError
from tenacity import (
  retry,
  retry_if_exception_type,
  stop_after_attempt,
  wait_exponential,
)
from vertexai.preview.tokenization import get_tokenizer_for_model  # type: ignore

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._llm._base import BaseEmbeddingService, BaseLLMService, NoopAsyncContextManager, T_model
from fast_graphrag._models import BaseModelAlias
from fast_graphrag._utils import logger


def default_safety_settings() -> List[types.SafetySetting]:
  """Provides default Gemini safety settings with all categories set to BLOCK_NONE.

  Gemini models include built-in safety filters to prevent generation of harmful content.
  This function configures permissive safety settings across all content categories,
  allowing the model to respond to a broader range of inputs. This is useful for
  applications that require less restrictive content filtering.

  The BLOCK_NONE threshold means the model will not block any content based on
  these harm categories. Safety filtering can still occur at other levels.

  Safety Categories Configured:
      - HARM_CATEGORY_DANGEROUS_CONTENT: Information about weapons or dangerous activities
      - HARM_CATEGORY_HARASSMENT: Content targeting individuals or groups
      - HARM_CATEGORY_HATE_SPEECH: Derogatory language or discrimination
      - HARM_CATEGORY_SEXUALLY_EXPLICIT: Adult sexual content
      - HARM_CATEGORY_CIVIC_INTEGRITY: Content affecting democratic processes

  Returns:
      List[types.SafetySetting]: A list of SafetySetting objects for all harm categories
          with BLOCK_NONE thresholds.

  Example:
      >>> settings = default_safety_settings()
      >>> config = types.GenerateContentConfig(safety_settings=settings)
  """
  return [
    types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
      category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
      threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
  ]


def _execute_with_inner_retries(
  operation: Callable[[], Awaitable[Any]],
  validate: Callable[[Any, int, int], bool],
  max_attempts: int = 4,
  short_sleep: float = 0.01,
  error_sleep: float = 0.2,
) -> Any:
  """Executes an asynchronous operation with inner retry logic for Gemini API calls.

  This helper function implements an inner retry loop that complements the outer
  retry mechanism (via tenacity decorator). It is specifically designed for Gemini
  API operations that may experience transient failures or incomplete responses.

  The function allows custom validation logic to determine if a response is acceptable,
  enabling flexible retry criteria based on response content (e.g., ensuring JSON
  parsing success or non-empty text fields).

  Retry Strategy:
      - Attempts the operation up to max_attempts times
      - Uses custom validation function to check response quality
      - Applies exponential backoff with different delays for connection errors
      - Catches both Gemini ClientErrors and general connection errors

  Args:
      operation: An async callable that performs the Gemini API request. Should
          return the response object or raise an exception.
      validate: A validation function with signature (result, attempt, max_attempts) -> bool.
          Should return True if the result is acceptable and satisfies requirements.
          Receives the current attempt number and total attempts for conditional logic.
      max_attempts: Maximum number of retry attempts (default: 4).
      short_sleep: Sleep duration in seconds for minor issues or normal retries
          (default: 0.01 seconds).
      error_sleep: Sleep duration in seconds for connection-related errors
          (default: 0.2 seconds, longer due to network issues).

  Returns:
      The valid result from operation if validation passes.

  Raises:
      Exception: The last encountered exception if all max_attempts are exhausted
          and no valid response is obtained.

  Example:
      >>> async def call_gemini():
      ...     return await client.aio.models.generate_content(model=model, contents=prompt)
      >>> def validate_response(resp, attempt, max_attempts):
      ...     return resp and hasattr(resp, 'text') and resp.text
      >>> response = await _execute_with_inner_retries(call_gemini, validate_response)
  """
  # Track the last encountered exception to raise if all retries are exhausted
  last_exception: Exception = Exception("Unknown error")
  for attempt in range(max_attempts):
    try:
      # Execute the Gemini API operation asynchronously
      result = await operation()
      # Validate the result using the provided validation function
      if validate(result, attempt, max_attempts):
        return result
    except (errors.ClientError, ConnectionResetError, requests.exceptions.ConnectionError) as e:
      # Catch Gemini-specific ClientError and connection-related exceptions
      last_exception = e
    except Exception as e:
      # Catch any other unexpected exceptions
      last_exception = e
    # Delay before next attempt; use longer sleep duration for connection errors
    # since network issues typically require more time to resolve
    await asyncio.sleep(
      error_sleep
      if isinstance(last_exception, (ConnectionResetError, requests.exceptions.ConnectionError))
      else short_sleep
    )
  raise last_exception


@dataclass
class GeminiLLMService(BaseLLMService):
  """Service for generating text responses using Google Gemini AI models.

  This class provides a comprehensive interface for interacting with Google's Gemini
  language models (via Google Gemini API or Vertex AI). It supports both unstructured
  text generation and structured output with Pydantic models through the instructor
  library integration.

  Key Capabilities:
      - Streaming and non-streaming text generation
      - Structured output validation with Pydantic models
      - Conversation history management
      - System prompt support for role-based instructions
      - Concurrent request handling with rate limiting
      - Automatic retry with exponential backoff
      - Support for both Gemini API and Vertex AI endpoints
      - Configurable safety settings for content filtering

  Gemini-Specific Features:
      - Uses types.GenerateContentConfig for fine-grained control over generation
      - Supports JSON mode response format for structured outputs
      - Candidate count configuration for controlling output diversity
      - Built-in safety settings for harmful content categories
      - Local tokenizer integration for accurate token counting
      - Response parsing with JSON repair for malformed outputs

  Attributes:
      model: The Gemini model to use (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
      mode: Instructor mode for response parsing (default: JSON mode).
      client: API endpoint to use - "gemini" for Gemini API or "vertex" for Vertex AI.
      api_key: API key for authentication. Required for Gemini API, optional for Vertex AI.
      temperature: Controls output randomness (0.0-2.0). Lower = more deterministic.
      candidate_count: Number of response variations to generate (default: 1).
      max_requests_concurrent: Maximum concurrent requests allowed (default: 1024).
      max_requests_per_minute: Rate limit for requests per minute (default: 2000 for Flash 2.0).
      max_requests_per_second: Rate limit for requests per second (default: 500).
      project_id: GCP project ID (required for Vertex AI endpoint).
      location: GCP region (required for Vertex AI endpoint).
      safety_settings: Gemini safety filter configurations for various harm categories.
  """

  # Core fields required to interact with the Gemini API endpoint.
  model: str = field(default="gemini-2.0-flash")
  mode: instructor.Mode = field(default=instructor.Mode.JSON)
  client: Literal["gemini", "vertex"] = field(default="gemini")
  api_key: Optional[str] = field(default=None)
  temperature: float = field(default=0.7)
  candidate_count: int = field(default=1)
  max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 1024)))
  # Gemini Flash 2.0 has a paid developer API limit of 2000 requests per minute
  max_requests_per_minute: int = field(default=2000)
  max_requests_per_second: int = field(default=500)
  project_id: Optional[str] = field(default=None)
  location: Optional[str] = field(default=None)
  # Gemini safety settings define content filtering thresholds for various harm categories
  safety_settings: list[types.SafetySetting] = field(default_factory=default_safety_settings)

  def __post_init__(self):
    """Post-initialization setup for Gemini LLM service configuration.

    This method is called after the dataclass fields are initialized. It performs
    critical setup tasks for the Gemini LLM service:

    1. Creates rate limiting and concurrency control mechanisms:
        - Semaphore for concurrent request limits
        - AsyncLimiter for per-minute rate limiting
        - AsyncLimiter for per-second rate limiting
    2. Initializes the appropriate Gemini API client:
        - Vertex AI client: Uses project ID and location from GCP, or express API key
        - Gemini API client: Uses API key for direct authentication
    3. Sets up the local tokenizer for accurate token counting

    The rate limiters are conditionally created based on the parent class's
    configuration flags (rate_limit_concurrency, rate_limit_per_minute, rate_limit_per_second).
    If rate limiting is disabled, NoopAsyncContextManager instances are used instead.

    Raises:
        AssertionError: If Vertex AI client mode is selected but required parameters
            (project_id, location, or api_key) are not properly provided.
        ValueError: If an invalid client type is specified (must be 'gemini' or 'vertex').

    Note:
        The tokenizer is initialized with "gemini-1.5-flash-002" as a stable reference
        model. Token counts from this tokenizer should be accurate for most Gemini models.
    """
    # Initialize concurrency control semaphore if rate limiting is enabled.
    # This limits the maximum number of simultaneous requests to Gemini API.
    self.llm_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    # Initialize per-minute rate limiter to comply with Gemini API quotas.
    # Tracks requests over a 60-second window.
    self.llm_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    # Initialize per-second rate limiter for fine-grained request throttling.
    # Helps prevent burst requests that may trigger API rate limiting.
    self.llm_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )

    # Instantiate the appropriate Gemini client based on the configured endpoint.
    # Gemini SDK supports both direct API and Vertex AI authentication methods.
    if self.client == "vertex":
      # Vertex AI authentication requires either:
      # (1) project_id and location (uses Application Default Credentials), OR
      # (2) express API key (Bearer token authentication)
      assert (self.project_id is not None and self.location is not None and self.api_key is None) or (
        self.project_id is None and self.location is None and self.api_key is not None
      ), "Vertex AI requires a project id and location, or an express API key."
      if self.api_key is not None:
        # Initialize Vertex AI client with express API key authentication
        self.llm_async_client: genai.Client = genai.Client(vertexai=True, api_key=self.api_key)
      else:
        # Initialize Vertex AI client with default credentials (project and location based)
        self.llm_async_client = genai.Client(vertexai=True, project=self.project_id, location=self.location)
    elif self.client == "gemini":
      # Initialize Gemini API client with direct API key authentication
      # Requires a valid API key from Google Cloud Console
      self.llm_async_client: genai.Client = genai.Client(api_key=self.api_key)
    else:
      raise ValueError("Invalid client type. Must be 'gemini' or 'vertex'")

    # Initialize the local tokenizer for accurate token counting.
    # Uses Vertex AI's tokenizer service to count tokens in the same way as Gemini.
    # The reference model is stable and provides consistent token counts across Gemini versions.
    self.tokenizer = get_tokenizer_for_model("gemini-1.5-flash-002")
    logger.debug("Initialized GeminiLLMService.")

  def count_tokens(self, text: str) -> int:
    """Count the number of tokens in the provided text utilizing the local Gemini tokenizer.

    Args:
        text (str): The input text whose tokens are to be counted.
        model (Optional[str]): An optional model override (not used in current implementation).
        **kwargs: Additional keyword arguments (currently unused).

    Returns:
        int: Total token count.
    """
    return self.tokenizer.count_tokens(contents=text).total_tokens

  @retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((TimeoutError, Exception)),
  )
  async def send_message(
    self,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    response_model: Type[T_model] | None = None,
    **kwargs: Any,
  ) -> Tuple[T_model, list[dict[str, str]]]:
    """Sends a message to the Gemini AI language model with full retry and validation.

    This method is the primary interface for generating responses from Gemini models.
    It handles all aspects of the request lifecycle including concurrency control,
    rate limiting, response validation, and structured output parsing.

    The method uses a two-level retry strategy:
      1. Outer retry: Tenacity decorator retries the entire method up to 8 times
      2. Inner retry: _execute_with_inner_retries provides fast local validation

    Request Flow:
      1. Applies rate limiters (per-second, per-minute, concurrent)
      2. Builds message history including the current prompt
      3. Configures Gemini's GenerateContentConfig with safety and generation settings
      4. Executes the API call with inner validation loop
      5. Parses and validates the response (with JSON repair if needed)
      6. Converts structured output to the response model if provided
      7. Appends the response to conversation history

    Gemini-Specific Features:
      - Response JSON schema: When response_model is provided, configures Gemini's
        JSON mode to ensure output conforms to the schema
      - Candidate count: Generates multiple response options (controlled by self.candidate_count)
      - System instruction: Uses Gemini's native system instruction feature
      - Safety settings: Applies content filtering thresholds for various harm categories
      - BaseModelAlias support: Automatically converts between Pydantic models and dataclasses

    Args:
        prompt: The main user message to send to Gemini.
        system_prompt: Optional system-level instructions for role-based behavior or task definition.
        history_messages: Optional list of prior messages in conversation format
            (dict with 'role' and 'content' keys).
        response_model: Optional Pydantic model (or BaseModelAlias) that structures the response.
            When provided, configures JSON mode and response schema validation.
        **kwargs: Additional generation parameters (currently unused but accepted for compatibility).

    Returns:
        Tuple containing:
        - The parsed response (as response_model type if provided, otherwise raw text)
        - The updated message history including the new exchange

    Raises:
        ValueError: If the model name is missing or invalid.
        LLMServiceNoResponseError: If no valid response is obtained after all retry attempts.
        errors.APIError: For unrecoverable API errors (400, 403, 404) that should not be retried.
        ValidationError: If response parsing or schema validation fails despite JSON repair.

    Example:
        >>> service = GeminiLLMService(api_key="your-key")
        >>> response, history = await service.send_message(
        ...     prompt="Summarize quantum computing",
        ...     system_prompt="You are a physics expert",
        ...     response_model=SummaryModel
        ... )
    """
    # Apply the concurrency and rate limiters.
    # These context managers ensure we stay within Gemini API quotas and limits.
    async with self.llm_max_requests_concurrent:
      async with self.llm_per_minute_limiter:
        async with self.llm_per_second_limiter:
          # Use the configured Gemini model for this request.
          model = self.model

          # Build message history including the current user prompt.
          # Gemini accepts messages in role/content format for multi-turn conversations.
          messages: List[Dict[str, str]] = []
          if history_messages:
            messages.extend(history_messages)
          messages.append({"role": "user", "content": prompt})
          # Combine all messages into a single prompt string for this API call.
          # Gemini will treat the entire string as the input to generate_content.
          combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

          try:

            def validate_generate_content(response: Any, attempt: int, max_attempts: int) -> bool:
              """Validates Gemini's generate_content response during inner retry loop.

              Checks that the response has required fields and that parsed output
              (if using structured mode) is available before the final retry attempt.
              """
              # Require that a response exists and that the "text" field is non-empty.
              # Gemini returns text content in the response.text attribute.
              if not response or not getattr(response, "text", ""):
                return False
              # If a response model is expected, then require a parsed response on non-final attempts.
              # On the final attempt, we'll attempt to parse the raw text if parsed is unavailable.
              if response_model is not None and attempt != (max_attempts - 1):
                if not getattr(response, "parsed", ""):
                  return False
              return True

            # Configure Gemini's generation parameters using types.GenerateContentConfig.
            # This Gemini-specific configuration object controls model behavior and output format.
            # The config is different for structured vs unstructured responses.
            generate_config = (
              types.GenerateContentConfig(
                # System instruction: Gemini's native way to provide system-level context.
                # This is more efficient than prepending to the message history.
                system_instruction=system_prompt,
                # Response MIME type: Set to JSON when using structured output mode.
                # This tells Gemini to return responses in JSON format.
                response_mime_type="application/json",
                # Response schema: Provides the Pydantic model schema for structured output.
                # Gemini uses this to constrain generation to match the schema.
                # Convert BaseModelAlias to its underlying Pydantic Model if needed.
                response_schema=(
                  (response_model.Model) if issubclass(response_model, BaseModelAlias) else (response_model)
                ),
                # Candidate count: Number of response variations to generate.
                # Higher values may increase latency but allow selection of best responses.
                candidate_count=self.candidate_count,
                # Temperature: Controls randomness in responses (0.0 = deterministic, 2.0 = very random).
                temperature=self.temperature,
                # Safety settings: Content filtering thresholds for various harm categories.
                # These are Gemini-specific safety controls for responsible AI.
                safety_settings=self.safety_settings,
              )
              if response_model
              # Configuration for unstructured text responses (no JSON schema constraint).
              else types.GenerateContentConfig(
                system_instruction=system_prompt,
                candidate_count=self.candidate_count,
                temperature=self.temperature,
                safety_settings=self.safety_settings,
              )
            )

            # Use the helper function to perform inner retries with custom validation.
            # This provides fast, localized retry logic before the outer tenacity retry kicks in.
            # The operation calls Gemini's aio.models.generate_content async method.
            response = await _execute_with_inner_retries(
              operation=lambda: self.llm_async_client.aio.models.generate_content(  # type: ignore
                model=model,
                contents=combined_prompt,
                config=generate_config,
              ),
              validate=validate_generate_content,
              max_attempts=4,
              short_sleep=0.01,
              error_sleep=0.2,
            )

            # Final check: Ensure we have a valid response with text content.
            # Even after inner retries, we validate the response one more time.
            if not response or not getattr(response, "text", ""):
              raise LLMServiceNoResponseError("Failed to obtain a valid response for content.")

            # Parse and validate the response according to the response model.
            # Gemini's response includes both raw text and parsed structured data (if JSON mode).
            try:
              if response_model:
                # Structured output mode: Parse the response to match the Pydantic model schema.
                # Gemini provides response.parsed which contains the structured output.
                if response.parsed:
                  # The response was successfully parsed by Gemini into the response schema.
                  # Now validate it as a Pydantic model using TypeAdapter.
                  if issubclass(response_model, BaseModelAlias):
                    # BaseModelAlias wraps a Pydantic model; validate against the wrapped model.
                    llm_response = TypeAdapter(response_model.Model).validate_python(response.parsed)
                  else:
                    # Direct Pydantic model validation.
                    llm_response = TypeAdapter(response_model).validate_python(response.parsed)
                else:
                  # Fallback: Gemini didn't provide parsed output (rare).
                  # Attempt to repair potentially malformed JSON from response.text
                  # and parse it manually using json_repair utility.
                  fixed_json = cast(str, repair_json(response.parsed))
                  if issubclass(response_model, BaseModelAlias):
                    llm_response = TypeAdapter(response_model.Model).validate_json(fixed_json)
                  else:
                    llm_response = TypeAdapter(response_model).validate_json(fixed_json)
              else:
                # Unstructured output mode: Return the raw text response from Gemini.
                llm_response = response.text
            except ValidationError as e:
              # If Pydantic validation fails, raise an error with details.
              raise LLMServiceNoResponseError(f"Invalid JSON response: {str(e)}") from e

            # Append the AI model's response to the conversation history.
            # This maintains context for multi-turn conversations.
            # The role "model" is Gemini's convention for model-generated responses.
            messages.append(
              {
                "role": "model",
                "content": (
                  # If the response is a Pydantic model, serialize to JSON string.
                  # Otherwise, convert to string directly.
                  llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response)
                ),
              }
            )

            # If working with a BaseModelAlias, convert the Pydantic model back to dataclass.
            # BaseModelAlias provides interoperability between Pydantic and dataclass representations.
            if response_model and issubclass(response_model, BaseModelAlias):
              llm_response = cast(
                T_model,
                cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response),
              )

            return cast(T_model, llm_response), messages

          except errors.APIError as e:
            # Handle Gemini API error responses with appropriate logging and retry strategy.
            # Different error codes require different handling approaches.
            if e.code == 429 or (e.details and e.details.get("code") == 429):  # type: ignore
              # Rate limit error (HTTP 429): Gemini has throttled the request.
              # Let the outer tenacity retry mechanism handle this with exponential backoff.
              logger.warning(f"Rate limit error encountered: {e.code} - {e.message}. Attempting retry.")
              raise
            elif e.code in (400, 403, 404):
              # Client errors that should not be retried:
              # - 400: Bad Request (invalid parameters or malformed request)
              # - 403: Forbidden (permission denied or quota exceeded)
              # - 404: Not Found (model doesn't exist)
              # These indicate issues with the request itself, not transient API issues.
              logger.error(
                f"Client error encountered: {e.code} - {e.message}. Check your request parameters or API key."
              )
              raise
            elif e.code in (500, 503, 504):
              # Server-side errors that may be transient:
              # - 500: Internal Server Error
              # - 503: Service Unavailable
              # - 504: Gateway Timeout
              # Let the outer retry mechanism attempt recovery.
              logger.error(f"Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.")
              raise
            else:
              # Unexpected error code: Log with full details and raise.
              logger.exception(f"Unexpected API error encountered: {e.code} - {e.message}")
              raise
          except Exception as e:
            # Catch any other unexpected exceptions during the API call.
            logger.exception(f"Unexpected error: {e}")
            raise


@dataclass
class GeminiEmbeddingService(BaseEmbeddingService):
  """Service for generating dense embedding vectors using Google Gemini embedding models.

  This class provides a comprehensive interface for obtaining vector embeddings from
  text inputs using Google's Gemini embedding API. It handles batching, concurrent
  requests, and rate limiting to efficiently process large volumes of texts.

  Embeddings are dense vector representations (typically 768 dimensions) that capture
  semantic meaning of text. They can be used for similarity search, clustering, and
  other vector-based operations.

  Key Capabilities:
      - Batch processing with configurable batch sizes
      - Concurrent request handling with rate limiting
      - Automatic retry with exponential backoff
      - Per-minute, per-second, and concurrent request limits
      - Efficient numpy array output for downstream processing

  Gemini Embedding-Specific Features:
      - Uses text-embedding-004 model by default (768-dimensional embeddings)
      - Supports batches up to 100 items per request (max_elements_per_request=99)
      - Google Cloud enforces strict rate limits on embedding endpoints
      - Returns ContentEmbedding objects with vector values
      - Proper error handling for malformed inputs or quota exceeded

  Request Rate Limits:
      - Per-minute limit: 80 requests (Google Cloud enforces strict limits)
      - Per-second limit: 20 requests (fine-grained throttling)
      - Concurrent limit: 150 simultaneous requests (default)
      These defaults are tuned for stable operation without hitting quotas.

  Attributes:
      embedding_dim: Dimension of embedding vectors (default: 768 for text-embedding-004).
      max_elements_per_request: Maximum items per batch request. Gemini API limit is 100,
          so default is 99 to provide a safety margin.
      model: Embedding model to use (default: "text-embedding-004").
      api_version: Optional API version parameter (currently unused).
      max_requests_concurrent: Maximum simultaneous requests allowed (default: 150).
      max_requests_per_minute: Rate limit for requests per minute (default: 80).
      max_requests_per_second: Rate limit for requests per second (default: 20).
  """

  embedding_dim: int = field(default=768)
  # Gemini API has a hard limit of 100 items per batch; use 99 as a safety margin
  max_elements_per_request: int = field(default=99)
  model: Optional[str] = field(default="text-embedding-004")
  api_version: Optional[str] = field(default=None)
  max_requests_concurrent: int = field(default=int(os.getenv("CONCURRENT_TASK_LIMIT", 150)))
  # Google Cloud enforces strict rate limits on batch requests to Gemini embedding endpoints
  max_requests_per_minute: int = field(default=80)
  max_requests_per_second: int = field(default=20)

  def __post_init__(self):
    """Post-initialization setup for Gemini embedding service configuration.

    This method is called after the dataclass fields are initialized. It performs
    critical setup tasks for the Gemini embedding service:

    1. Creates rate limiting and concurrency control mechanisms:
        - Semaphore for concurrent request limits
        - AsyncLimiter for per-minute rate limiting
        - AsyncLimiter for per-second rate limiting
    2. Initializes the asynchronous Gemini API client for embedding requests

    The rate limiters are conditionally created based on the parent class's
    configuration flags (rate_limit_concurrency, rate_limit_per_minute, rate_limit_per_second).
    If rate limiting is disabled, NoopAsyncContextManager instances are used instead.

    Note:
        Embedding requests have strict Google Cloud rate limits. The defaults
        (80 per minute, 20 per second) are tuned to avoid quota exceeded errors.
    """
    # Initialize concurrency control semaphore if rate limiting is enabled.
    # This limits the maximum number of simultaneous embedding requests to Gemini API.
    self.embedding_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    # Initialize per-minute rate limiter for embedding requests.
    # Google Cloud enforces strict per-minute limits on embedding endpoints.
    # Tracks requests over a 60-second window.
    self.embedding_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    # Initialize per-second rate limiter for fine-grained request throttling.
    # Helps prevent burst embedding requests that trigger API rate limiting.
    self.embedding_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )
    # Initialize the Gemini API client for embedding requests using the provided API key.
    self.embedding_async_client: genai.Client = genai.Client(api_key=self.api_key)
    logger.debug("Initialized GeminiEmbeddingService.")

  async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Obtain dense embedding vectors for provided input texts from Gemini.

    This method is the primary interface for embedding text inputs. It handles batch
    splitting to stay within Gemini API limits, concurrent request execution, and
    response aggregation into a single numpy array.

    The embedding process:
      1. Splits texts into batches (max 99 items per batch)
      2. Concurrently sends embedding requests for all batches
      3. Aggregates ContentEmbedding responses from all batches
      4. Extracts vector values and combines into a single numpy array

    Concurrency:
        Batches are processed concurrently using asyncio.gather() to maximize throughput
        while respecting rate limits set by the rate limiters in _embedding_request().

    Gemini Embedding-Specific Details:
        - Uses the text-embedding-004 model by default (768-dimensional vectors)
        - Returns float32 numpy arrays for memory efficiency
        - Batches are limited to 99 items (Gemini API hard limit is 100)
        - Each ContentEmbedding object contains a vector in the .values attribute

    Args:
        texts: List of input text strings to be embedded. Can be any size;
            batching is handled automatically.
        model: Optional model override. If not provided, uses the service's default model.

    Returns:
        np.ndarray with shape (len(texts), 768): Array of embedding vectors, one per input text.
            Vectors are 32-bit floating point values.

    Raises:
        ValueError: If no model is available (both model parameter and self.model are None).
        LLMServiceNoResponseError: If embedding requests fail after all retries.
        Exception: Propagates any other exception encountered during embedding.

    Example:
        >>> service = GeminiEmbeddingService(api_key="your-key")
        >>> texts = ["Hello world", "How are you?"]
        >>> embeddings = await service.encode(texts)
        >>> print(embeddings.shape)  # Output: (2, 768)
    """
    try:
      logger.debug(f"Getting embedding for texts: {texts}")
      # Use the provided model or fall back to the service's default.
      model = model or self.model
      if model is None:
        raise ValueError("Model name must be provided.")

      # Split texts into batches that respect Gemini API limits.
      # The max_elements_per_request is 99 by default (API hard limit is 100).
      # This ensures no batch exceeds the API constraint.
      batched_texts = [
        texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
        for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
      ]
      # Execute embedding requests concurrently for all batches.
      # asyncio.gather() waits for all concurrent tasks and returns results in order.
      response = await asyncio.gather(*[self._embedding_request(batch, model) for batch in batched_texts])

      # Flatten the list of responses from all batches.
      # Each response is a list of ContentEmbedding objects.
      # chain() combines all batch responses into a single iterable.
      data = chain(*list(response))
      # Extract the vector values from each ContentEmbedding object and create numpy array.
      # Each embedding is a vector of dimension 768 (for text-embedding-004).
      embeddings = np.array([dp.values for dp in data])
      logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

      return embeddings
    except Exception:
      logger.exception("An error occurred during embedding encoding:", exc_info=True)
      raise

  @retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((TimeoutError, Exception)),
  )
  async def _embedding_request(self, input: list[Any], model: str) -> list[types.ContentEmbedding]:
    """Makes an embedding request for a batch of input texts with retry and rate limiting.

    This private method handles a single batch of embedding requests with full retry and
    rate limiting support. It is called by the encode() method for each batch.

    The method uses a two-level retry strategy:
      1. Outer retry: Tenacity decorator retries up to 8 times with exponential backoff
      2. Inner retry: _execute_with_inner_retries provides fast validation loop

    Rate Limiting:
        Applies three levels of rate limiting via context managers:
        - Concurrent limit: Restricts simultaneous requests
        - Per-minute limit: Complies with Google Cloud quota
        - Per-second limit: Prevents burst requests

    Gemini Embedding-Specific Details:
        - Calls embed_content() with aio (async) interface
        - Returns list of ContentEmbedding objects, each with a .values attribute
        - Validates that response.embeddings is non-empty before returning
        - Handles Gemini-specific error codes and retry logic

    Args:
        input: A batch of text items to be embedded (max 99 items recommended).
        model: The embedding model name (e.g., "text-embedding-004").

    Returns:
        list[types.ContentEmbedding]: List of ContentEmbedding objects containing
            the embedding vectors for each input text, in the same order.

    Raises:
        LLMServiceNoResponseError: If a valid response is not obtained after retries.
        errors.APIError: For unrecoverable API errors (400, 403, 404).
        TimeoutError: If the request times out.

    Note:
        This is a private method; use encode() instead for the public interface.
    """
    # Apply all three levels of rate limiting to stay within Gemini API quotas.
    async with self.embedding_max_requests_concurrent:
      async with self.embedding_per_minute_limiter:
        async with self.embedding_per_second_limiter:
          try:

            def validate_embedding_response(response: Any, attempt: int, max_attempts: int) -> bool:
              """Validates Gemini's embed_content response during inner retry loop.

              Checks that the response contains non-empty embeddings list.
              """
              # Require that response exists and has a non-empty embeddings list.
              # Gemini returns embeddings as a list of ContentEmbedding objects.
              if not response or not getattr(response, "embeddings", None) or response.embeddings == []:
                return False
              return True

            # Execute the embedding request with inner retry loop.
            # The validation function checks for non-empty embeddings in the response.
            response = await _execute_with_inner_retries(
              operation=lambda: self.embedding_async_client.aio.models.embed_content(model=model, contents=input),  # type: ignore
              validate=validate_embedding_response,
              max_attempts=4,
              short_sleep=0.01,
              error_sleep=0.2,
            )

            # Final validation: Ensure we have a valid embeddings list.
            # Even after inner retries, check that response contains data.
            if not response or not getattr(response, "embeddings", None) or response.embeddings == []:
              raise LLMServiceNoResponseError("Failed to obtain a valid response for embeddings.")

            # Return the list of ContentEmbedding objects from Gemini.
            # Each object contains a vector in the .values attribute.
            return response.embeddings

          except errors.APIError as e:
            # Handle Gemini API errors with appropriate retry strategy.
            # Different error codes indicate different problems and recovery approaches.
            if e.code == 429 or (e.details and e.details.get("code") == 429):  # type: ignore
              # Rate limit error (HTTP 429): Gemini has throttled embedding requests.
              # Let the outer tenacity retry mechanism handle recovery.
              logger.warning(f"Rate limit error encountered: {e.code} - {e.message}. Delegating to outer retry.")
              raise
            elif e.code in (400, 403, 404):
              # Client errors that should not be retried:
              # - 400: Bad Request (malformed input or invalid batch format)
              # - 403: Forbidden (API key invalid or quota exceeded)
              # - 404: Not Found (model doesn't exist)
              logger.error(
                f"Client error encountered: {e.code} - {e.message}. Check your request parameters or API key."
              )
              raise
            elif e.code in (500, 503, 504):
              # Server-side errors that may be transient:
              # - 500: Internal Server Error
              # - 503: Service Unavailable (overloaded)
              # - 504: Gateway Timeout
              logger.error(f"Server error encountered: {e.code} - {e.message}. Consider retrying after a short delay.")
              raise
            else:
              # Unexpected error code: Log and raise for outer retry mechanism.
              logger.exception(f"Unexpected API error encountered: {e.code} - {e.message}")
              raise
          except Exception as e:
            # Catch any other unexpected exceptions during the embedding request.
            logger.exception(f"Unexpected error during embedding request: {e}")
            raise
