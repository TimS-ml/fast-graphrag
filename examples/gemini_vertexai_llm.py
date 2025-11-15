"""Example usage of GraphRAG with Gemini LLM and Embeddings from Google Cloud VertexAI.

This module demonstrates how to use fast-graphrag with Google Cloud's VertexAI platform,
which provides Gemini models and text embedding services. It includes custom implementations
of LLM and embedding services that integrate with VertexAI's APIs.

Key Features:
    - Custom VertexAILLMService: Handles Gemini model interactions with structured JSON responses
    - Custom VertexAIEmbeddingService: Converts text to embeddings using VertexAI's Gecko model
    - Async/await pattern for non-blocking API calls
    - Comprehensive retry logic with exponential backoff for rate limits and transient errors
    - Throttling mechanism to prevent overwhelming the API with concurrent requests
    - Schema validation and error handling for structured outputs
    - Support for both Vertex AI direct API and OpenAI-compatible endpoints

Requirements:
    - Google Cloud Project with VertexAI API enabled
    - Service account with VertexAI permissions
    - pip install fast-graphrag google-cloud-aiplatform

Setup Instructions:
    1. Install dependencies:
       pip install fast-graphrag google-cloud-aiplatform

    2. Authenticate with Google Cloud:
       gcloud auth application-default login
       (This creates credentials that will be used automatically)

    3. Set your Google Cloud project (optional, can be set in code):
       gcloud config set project YOUR_PROJECT_ID

    4. In the __main__ block, optionally uncomment and set:
       - project_id: Your Google Cloud project ID
       - location: Your desired region (e.g., "us-central1")

    5. Prepare your input file:
       - Ensure "../mock_data.txt" exists relative to the script location
       - The file should contain text to analyze

    6. Run the script:
       python gemini_vertexai_llm.py

VertexAI vs AI Studio:
    - VertexAI: Enterprise solution with VPC support, IAM control, and multiple regions
    - AI Studio: Simpler API access with higher rate limits but fewer enterprise features
    - Use VertexAI if you need production-grade infrastructure and multi-region support
    - Use AI Studio (gemini_example.py) if you prioritize simplicity and higher rate limits

Performance Considerations:
    - The Gecko embedding model outputs 768-dimensional vectors
    - Batching is limited to 32 requests per batch to match OpenAI's behavior
    - Concurrent requests are throttled to 50 maximum to avoid rate limit issues
    - Temperature is set to 0.6 for balanced creativity and determinism
"""

import asyncio
import re
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast, Literal, TypeVar, Callable
from functools import wraps
import logging
import instructor

import numpy as np
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel

from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService
from fast_graphrag._models import BaseModelAlias
from fast_graphrag._models import _json_schema_slim

# Configure logging - set to ERROR to reduce verbosity
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Type variable for generic response models
T_model = TypeVar("T_model")


def throttle_async_func_call(
    max_concurrent: int = 2048, stagger_time: Optional[float] = None, waiting_time: float = 0.001
):
    """Decorator to limit concurrent executions of an async function.

    This decorator prevents overwhelming the API by limiting the number of concurrent
    requests. It acts as a rate limiter to ensure we don't exceed API quotas.

    Args:
        max_concurrent: Maximum number of concurrent function executions (default: 2048)
        stagger_time: Time to wait between function calls (optional, currently disabled)
        waiting_time: Time to sleep when at max capacity (default: 0.001 seconds)

    Returns:
        Decorator function that wraps the target async function
    """
    _wrappedFn = TypeVar("_wrappedFn", bound=Callable[..., Any])

    def decorator(func: _wrappedFn) -> _wrappedFn:
        # Counters for tracking concurrent executions
        __current_exes = 0
        __current_queued = 0

        @wraps(func)
        async def wait_func(*args: Any, **kwargs: Any) -> Any:
            nonlocal __current_exes, __current_queued
            # Block until there's room for another concurrent execution
            while __current_exes >= max_concurrent:
                await asyncio.sleep(waiting_time)

            # Optional: Stagger requests over time (currently disabled)
            # __current_queued += 1
            # await asyncio.sleep(stagger_time * (__current_queued - 1))
            # __current_queued -= 1

            # Increment counter and execute the function
            __current_exes += 1
            try:
                result = await func(*args, **kwargs)
            finally:
                # Always decrement counter, even if function raises an exception
                __current_exes -= 1
            return result

        return wait_func  # type: ignore

    return decorator


class LLMServiceNoResponseError(Exception):
    """Custom exception raised when the LLM service returns an invalid or empty response."""
    pass


@dataclass
class VertexAIEmbeddingService(BaseEmbeddingService):
    """VertexAI implementation for text embeddings using Google's Gecko embedding model.

    This service converts text to dense vector embeddings using VertexAI's text embedding API.
    The Gecko model produces 768-dimensional vectors suitable for semantic search and similarity.

    Attributes:
        embedding_dim: Dimensionality of output embeddings (Gecko outputs 768)
        max_elements_per_request: Batch size for API requests (32, matching OpenAI's behavior)
        model: Model identifier (default: textembedding-gecko@latest)
        client: Client type identifier (always "vertex" for this implementation)
    """
    # Configuration for Vertex AI's Gecko embedding model
    embedding_dim: int = field(default=768)  # Output dimension of Gecko model
    max_elements_per_request: int = field(default=32)  # Batch size to match OpenAI's API
    model: Optional[str] = field(default="textembedding-gecko@latest")  # Latest Gecko model version
    client: Literal["vertex"] = field(default="vertex")  # Client type identifier

    def __post_init__(self):
        """Initialize the VertexAI Gecko embedding model.

        Called automatically after dataclass initialization. Sets up the connection
        to VertexAI's embedding service.
        """
        # Load the pre-trained Gecko embedding model from VertexAI
        self._embedding_model = TextEmbeddingModel.from_pretrained(self.model)
        logger.debug("Initialized VertexAIEmbeddingService with Vertex AI client.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Generate embeddings for a list of texts using batched processing.

        Breaks the input texts into batches to match VertexAI's API constraints,
        processes them concurrently, and returns consolidated results.

        Args:
            texts: List of strings to convert to embeddings
            model: Optional model override (currently unused)

        Returns:
            np.ndarray: Array of shape (len(texts), embedding_dim) containing the embeddings
        """
        # Split texts into batches to respect API limits
        # Each batch contains at most max_elements_per_request texts
        batched_texts = [
            texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
            for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
        ]

        # Process all batches concurrently to improve throughput
        response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])

        # Combine all batch results into a single numpy array
        embeddings = np.vstack([batch_embeddings for batch_embeddings in response])
        logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

        return embeddings

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times on failure
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4-10 seconds
        retry=retry_if_exception_type((TimeoutError, Exception)),  # Retry on these exceptions
    )
    async def _embedding_request(self, input_texts: List[str], model: str) -> np.ndarray:
        """Get embeddings for a single batch of texts with automatic retry.

        This method is decorated with @retry to automatically handle transient failures
        with exponential backoff.

        Args:
            input_texts: Batch of texts to embed (max 32 per call)
            model: Model name (unused, for API compatibility)

        Returns:
            np.ndarray: Array of embeddings with shape (len(input_texts), embedding_dim)

        Raises:
            Exception: If all retry attempts fail
        """
        try:
            # Call the VertexAI API to get embeddings asynchronously
            embeddings = await self._embedding_model.get_embeddings_async(input_texts)

            # Convert VertexAI embedding objects to numpy array with float32 dtype
            # This maintains compatibility with OpenAI's embedding format
            return np.array([emb.values for emb in embeddings], dtype=np.float32)

        except Exception as e:
            # Log the error and re-raise for retry logic to handle
            logger.error(f"Error in embedding request: {str(e)}")
            raise


@dataclass
class VertexAILLMService(BaseLLMService):
    """VertexAI implementation for LLM services using Google's Gemini models.

    This service handles all interactions with Gemini models through VertexAI's APIs.
    It provides structured output support, comprehensive error handling, and rate limiting.

    Attributes:
        model: Gemini model identifier (e.g., "gemini-1.5-pro-002")
        base_url: Optional custom endpoint URL
        api_key: Optional API key override
        max_retries: Maximum retry attempts for transient errors (default: 3)
        retry_delay: Base delay between retries in seconds (default: 1.0)
        rate_limit_max_retries: Maximum retries for rate limit (429) errors (default: 5)
        mode: Response format mode (default: instructor.Mode.JSON)
        temperature: Sampling temperature (0.0=deterministic, 1.0=creative) (default: 0.6)
        kwargs: Additional parameters passed to the model
        llm_calls_count: Tracks total number of LLM API calls made
    """
    # Model configuration
    model: Optional[str] = field(default="gemini-1.0-pro")  # Default Gemini model
    base_url: Optional[str] = field(default=None)  # Optional custom API endpoint
    api_key: Optional[str] = field(default=None)  # Optional API key override

    # Retry and error handling configuration
    max_retries: int = field(default=3)  # Max attempts for transient errors (5xx)
    retry_delay: float = field(default=1.0)  # Base delay between retries (seconds)
    rate_limit_max_retries: int = field(default=5)  # Max attempts for rate limit errors (429)
    mode: instructor.Mode = field(default=instructor.Mode.JSON)  # Response format mode
    temperature: float = field(default=0.6)  # Balance between creativity and determinism
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional model parameters

    # Internal tracking (not initialized in constructor)
    llm_calls_count: int = field(default=0, init=False)  # Counter for API calls
    _vertex_model: Any = field(default=None, init=False)  # Cached Gemini model instance

    def __post_init__(self):
        """Initialize after dataclass initialization.

        Validates the model name and initializes the VertexAI Gemini model instance.

        Raises:
            ValueError: If model name is not provided
        """
        if self.model is None:
            raise ValueError("Model name must be provided.")

        # Initialize the Vertex AI GenerativeModel (Gemini instance)
        self._vertex_model = GenerativeModel(self.model)

    def _extract_retry_time(self, error_message: str) -> float:
        """Extract retry-after delay from VertexAI error messages.

        Parses error messages to find recommended retry delays.
        VertexAI includes "Retry the request after X sec" in rate limit errors.

        Args:
            error_message: Error message from the API

        Returns:
            float: Recommended retry delay in seconds (default: 2.0 if not found)
        """
        # Use regex to extract the retry time from the error message
        match = re.search(r'Retry the request after (\d+) sec', str(error_message))
        if match:
            return float(match.group(1))
        # Default to 2 seconds if retry time isn't specified in the error
        return 2.0

    def _count_tokens(self, messages: List[dict[str, str]]) -> int:
        """Estimate token count for a list of messages.

        Uses a rough approximation: word count * 1.3
        This is a simple heuristic and not perfectly accurate for all models.

        Args:
            messages: List of message dictionaries with "content" field

        Returns:
            int: Estimated token count
        """
        return sum(len(msg["content"].split()) * 1.3 for msg in messages)
    

    @throttle_async_func_call(
        max_concurrent=50,  # Limit to 50 concurrent requests to avoid overwhelming the API
        stagger_time=0.1,  # Optional stagger between requests (currently disabled)
        waiting_time=0.001  # Sleep duration when at max capacity
    )
    async def send_message(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict[str, str]]] = None,
        response_model: Optional[Type[T_model]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[T_model, List[dict[str, str]]]:
        """Send a message to the VertexAI Gemini model and handle the response.

        This is the main method for LLM interaction. It handles:
        - System prompts and conversation history
        - Schema-based structured output (JSON)
        - Comprehensive error handling with retries for rate limits and transient errors
        - Token counting for monitoring

        Args:
            prompt: Main user input text to process
            model: Optional model override (uses self.model if not provided)
            system_prompt: Optional system instructions to guide the model behavior
            history_messages: Previous conversation messages for context
            response_model: Expected response structure (for structured outputs like JSON)
            temperature: Optional temperature override for this request
            **kwargs: Additional parameters passed to the model

        Returns:
            Tuple of (parsed_response, message_history):
                - parsed_response: The model's response, parsed according to response_model if provided
                - message_history: Updated conversation history including the new exchange

        Raises:
            LLMServiceNoResponseError: If the model returns an invalid or empty response
            ValueError: If model name is missing
            Exception: If all retry attempts fail (rate limits, server errors, etc.)
        """
        def convert_to_vertex_schema(schema_to_convert):
            """Convert JSON Schema to VertexAI's expected schema format.

            VertexAI uses a simplified schema format compared to JSON Schema.
            This function recursively converts JSON Schema definitions to VertexAI's format.

            Args:
                schema_to_convert: JSON Schema dictionary to convert

            Returns:
                Dict containing VertexAI-compatible schema
            """
            if "$ref" in schema_to_convert:
                # Handle schema references by following the path to the actual schema
                ref_path = schema_to_convert["$ref"].split("/")
                ref_schema = schema
                for path in ref_path[1:]:
                    ref_schema = ref_schema[path]
                # Recursively convert the referenced schema
                return convert_to_vertex_schema(ref_schema)

            if schema_to_convert["type"] == "array":
                # Handle array types by recursively converting items schema
                return {
                    "type": "array",
                    "items": convert_to_vertex_schema(schema_to_convert["items"])
                }
            elif schema_to_convert["type"] == "object":
                # Handle object types by recursively converting all properties
                props = {}
                for prop_name, prop_schema in schema_to_convert.get("properties", {}).items():
                    props[prop_name] = convert_to_vertex_schema(prop_schema)
                return {
                    "type": "object",
                    "properties": props
                }
            else:
                # Handle primitive types (string, number, boolean, etc.)
                return {
                    "type": schema_to_convert["type"],
                    "description": schema_to_convert.get("description", "")
                }

        # Initialize retry counters and temperature
        temperature = self.temperature
        retries = 0  # Counter for transient errors (5xx)
        rate_limit_retries = 0  # Counter for rate limit errors (429)

        logger.debug(f"Sending message with prompt: {prompt}")

        # Use provided model or fall back to instance default
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")

        # Build the message list: system prompt + history + current user message
        messages: List[dict[str, str]] = []
        if system_prompt:
            # Add system prompt if provided (guides model behavior)
            messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"Added system prompt: {system_prompt}")

        if history_messages:
            # Add previous messages for conversation context
            messages.extend(history_messages)
            logger.debug(f"Added history messages: {history_messages}")

        # Add the current user message
        messages.append({"role": "user", "content": prompt})

        # Prepare schema for structured output if a response model is specified
        if response_model:
            # Extract the actual model class (handles BaseModelAlias wrappers)
            model_class = (response_model.Model
                        if issubclass(response_model, BaseModelAlias)
                        else response_model)
            # Get JSON schema for the response model
            schema = model_class.model_json_schema()

            # Initialize VertexAI schema with object type
            vertex_schema = {
                "type": "object",
                "properties": {}
            }

            # Convert each top-level property from JSON Schema to VertexAI format
            for prop_name, prop_schema in schema["properties"].items():
                vertex_schema["properties"][prop_name] = convert_to_vertex_schema(prop_schema)


        # Custom schema instruction for ensuring consistent JSON responses
        # NOTE: This is a workaround until the Instructor library fully supports VertexAI
        # The instruction format helps guide the model to produce well-structured JSON outputs
        schema_instruction = (
            "IMPORTANT: Your response must be a valid JSON object containing all the following fields "
            "(use empty arrays [] for fields with no values):\n"
            "- entities\n"
            "- relationships\n"
            "- other_relationships\n\n"
            "Each entity MUST contain ALL of these required fields:\n"
            "- name: The unique identifier of the entity\n"
            "- type: The category or classification of the entity\n"
            "- desc: A detailed description of the entity (REQUIRED, never omit this field)\n\n"
            "Example of a valid entity:\n"
            "{\n"
            '    "name": "Scrooge",\n'
            '    "type": "person",\n'
            '    "desc": "A miserly businessman who undergoes a transformation"\n'
            "}\n\n"
            "Example of a complete valid response format:\n"
            "{\n"
            '    "entities": [\n'
            '        {"name": "Scrooge", "type": "person", "desc": "A miserly businessman"},\n'
            '        {"name": "London", "type": "location", "desc": "The city where the story takes place"}\n'
            '    ],\n'
            '    "relationships": [\n'
            '        {"source": "Scrooge", "target": "London", "desc": "Scrooge lives and works in London"}\n'
            '    ],\n'
            '    "other_relationships": []\n'
            "}"
        )

        # Insert the schema instruction at the beginning of messages
        # This ensures it's prominent and guides the model's response generation
        messages.insert(0, {
            "role": "system",
            "content": schema_instruction
        })

        # Combine all messages into a single prompt string
        # Format: "role: content" for each message
        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Main request loop with retry logic for transient errors and rate limits
        while True:
            try:
                # Configure generation parameters for the VertexAI API
                generation_config = GenerationConfig(
                    temperature=temperature,  # Creativity level (0.0=deterministic, 1.0=creative)
                    candidate_count=1,  # Only request one response candidate
                    response_mime_type="application/json",  # Force JSON response format
                    response_schema=vertex_schema  # Use the schema if response_model is provided
                )

                # Call the VertexAI Gemini model asynchronously
                vertex_response = await self._vertex_model.generate_content_async(
                    combined_prompt,
                    generation_config=generation_config,
                    stream=False  # Don't stream responses
                )

                # Validate that we received a response with content
                if not vertex_response or not vertex_response.text:
                    logger.error("Empty response from Vertex AI")
                    raise LLMServiceNoResponseError("Empty response from Vertex AI")

                # Extract the text content from the response
                response_text = vertex_response.text

                # Parse the response according to the response_model structure if provided
                try:
                    if response_model:
                        # Parse JSON response into the specified model
                        if issubclass(response_model, BaseModelAlias):
                            # Handle BaseModelAlias wrapper types
                            llm_response = response_model.Model.model_validate_json(response_text)
                        else:
                            # Handle standard Pydantic models
                            llm_response = response_model.model_validate_json(response_text)
                    else:
                        # If no response model specified, return raw text
                        llm_response = response_text
                except ValidationError as e:
                    # JSON validation failed - raise informative error
                    logger.error(f"JSON validation error: {str(e)}")
                    raise LLMServiceNoResponseError(f"Invalid JSON response: {str(e)}") from e

                # Increment the API call counter
                self.llm_calls_count += 1

                # Keep a copy of the original response for history
                original_llm_response = llm_response

                # Final check: ensure we actually got a response
                if not llm_response:
                    logger.error("No response received from the language model.")
                    raise LLMServiceNoResponseError("No response received from the language model.")

                # Add the assistant's response to the message history
                messages.append({
                    "role": "assistant",
                    "content": (llm_response.model_dump_json()  # Convert models to JSON for history
                              if isinstance(llm_response, BaseModel)
                              else str(llm_response)),
                })
                logger.debug(f"Received response: {llm_response}")

                # Convert BaseModelAlias back to dataclass format if needed
                if response_model and issubclass(response_model, BaseModelAlias):
                    llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

                # Return the parsed response and updated message history
                return llm_response, messages

            except Exception as e:
                # Extract status code from exception (default to 500 for unspecified errors)
                status_code = getattr(e, 'status_code', 500)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
                token_count = round(self._count_tokens(messages))

                if status_code == 500:
                    # Handle transient server errors with exponential backoff retry
                    if retries >= self.max_retries:
                        # Max retries exceeded - give up and raise error
                        error_log = (
                            f"{timestamp}|500|{model}|{token_count}|"
                            f"Max retries reached ({self.max_retries})|{str(e)}\n"
                        )
                        print(error_log)
                        err = f"LLM API failed with 500 error after {self.max_retries} retries: {e}"
                        logger.error(err)
                        raise Exception(err) from e

                    # Increment retry counter and wait before retrying
                    retries += 1
                    wait_time = self.retry_delay * (retries)  # Exponential backoff
                    error_log = f"{timestamp}|500|{model}|{token_count}|Attempt {retries}|{str(e)}\n"
                    print(error_log)

                    # Sleep before retrying
                    await asyncio.sleep(wait_time)
                    continue  # Retry the request

                if status_code == 429:
                    # Handle rate limit (too many requests) errors
                    if rate_limit_retries >= self.rate_limit_max_retries:
                        # Max rate limit retries exceeded - give up
                        error_log = (
                            f"{timestamp}|429|{model}|{token_count}|"
                            f"Rate limit max retries reached ({self.rate_limit_max_retries})|{str(e)}\n"
                        )
                        print(error_log)
                        raise Exception(f"Rate limit exceeded after {self.rate_limit_max_retries} retries: {e}") from e

                    # Extract recommended retry time from API error message
                    retry_time = self._extract_retry_time(str(e))
                    rate_limit_retries += 1
                    error_log = f"{timestamp}|429|{model}|{token_count}|Attempt {rate_limit_retries}|{str(e)}\n"
                    print(error_log)
                    err = (f"Rate limit hit (attempt {rate_limit_retries}/{self.rate_limit_max_retries}). "
                          f"Waiting {retry_time} seconds before retry...")

                    logger.warning(err)
                    # Wait the recommended time before retrying
                    await asyncio.sleep(retry_time)
                    continue  # Retry the request

                # For other status codes, re-raise immediately
                raise

            except Exception as e:
                # Handle any other unexpected errors
                err = f"Unexpected error: {str(e)}"
                logger.error(err)
                raise LLMServiceNoResponseError(err) from e


if __name__ == "__main__":
    # Example usage of GraphRAG with VertexAI services
    from fast_graphrag import GraphRAG

    # === Configuration for the example ===
    # Domain description for context-aware entity extraction
    DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

    # Example queries that guide entity extraction optimization
    EXAMPLE_QUERIES = [
        "What is the significance of Christmas Eve in A Christmas Carol?",
        "How does the setting of Victorian London contribute to the story's themes?",
        "Describe the chain of events that leads to Scrooge's transformation.",
        "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
        "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
    ]

    # Entity types to extract from the story
    ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

    # === Initialize VertexAI ===
    # NOTE: You can optionally specify project_id and location here:
    #   vertexai.init(project="your-project-id", location="us-central1")
    # If not specified, VertexAI will use the default project from your credentials
    vertexai.init(
        # project="<your-project-id>",  # Uncomment and set your Google Cloud project ID
        # location="<your-region>"  # Uncomment and set your region (e.g., "us-central1")
    )

    # Initialize the embedding service for semantic search
    # Uses VertexAI's Gecko model (768-dimensional embeddings)
    embedding_service = VertexAIEmbeddingService()

    # Initialize the LLM service for language understanding and entity extraction
    # Uses Gemini model with retry logic and rate limiting
    llm_service = VertexAILLMService(
        model="gemini-1.5-pro-002",  # Powerful Gemini model for complex reasoning
        max_retries=2,  # Retry transient errors up to 2 times
        retry_delay=1.0,  # Base delay between retries (seconds)
        rate_limit_max_retries=2,  # Retry rate limit errors up to 2 times
        temperature=0.6  # Balanced between creativity and determinism
    )

    # === Initialize GraphRAG ===
    # Create the knowledge graph system with VertexAI services
    grag = GraphRAG(
        working_dir="/tmp/book_example",  # Directory to store graph index and data
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
    )

    # === Process input document ===
    # Read the input text file
    # NOTE: Update the path to point to your actual input file
    with open("../mock_data.txt") as f:
        # Insert the document into GraphRAG
        # This triggers entity extraction and knowledge graph construction
        grag.insert(f.read())

    # === Query the knowledge graph ===
    # Run an example query to test the system
    user_query = "Who are three main characters in the story?"
    print(f"User query: {user_query}")
    # Query and print the response
    print(grag.query(user_query).response)

