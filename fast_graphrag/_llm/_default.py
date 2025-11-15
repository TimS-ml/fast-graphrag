"""Default LLM and Embedding Service implementations.

This module provides default implementations for Language Model (LLM) and Embedding
services used throughout the fast-graphrag system. These services act as convenient
aliases that point to the OpenAI-based implementations, providing a standardized
interface for LLM operations and text embeddings.

The default services can be used directly or subclassed to create customized
implementations. When no specific service is configured, these defaults are used
to handle all language model interactions and embedding generation tasks.

Exports:
    DefaultLLMService: Default implementation for Language Model operations.
    DefaultEmbeddingService: Default implementation for text embedding generation.
"""

__all__ = ['DefaultLLMService', 'DefaultEmbeddingService']

from ._llm_openai import OpenAIEmbeddingService, OpenAILLMService


class DefaultLLMService(OpenAILLMService):
    """Default Language Model service implementation.

    This service provides the default LLM interface for the fast-graphrag system.
    It inherits from OpenAILLMService and serves as the standard implementation
    for all LLM operations including text generation, prompt completion, and
    conversational interactions.

    The service handles:
        - Model selection and configuration
        - API communication with LLM providers
        - Request/response processing
        - Token counting and management
        - Error handling and retry logic

    This class is intentionally minimal as it uses the full implementation from
    OpenAILLMService. Applications can subclass this to add custom behavior or
    override specific methods for specialized use cases.

    Example:
        >>> service = DefaultLLMService()
        >>> response = service.generate(prompt="Hello, world!")
    """
    pass


class DefaultEmbeddingService(OpenAIEmbeddingService):
    """Default text embedding service implementation.

    This service provides the default interface for generating text embeddings
    throughout the fast-graphrag system. It inherits from OpenAIEmbeddingService
    and serves as the standard implementation for converting text into numerical
    vector representations.

    The service handles:
        - Text tokenization and preprocessing
        - Vector generation using embedding models
        - Batch processing of multiple texts
        - Caching and performance optimization
        - Dimension management and normalization

    Text embeddings are essential for:
        - Semantic similarity calculations
        - Vector-based search and retrieval
        - Machine learning model inputs
        - Clustering and classification tasks

    This class is intentionally minimal as it uses the full implementation from
    OpenAIEmbeddingService. Applications can subclass this to add custom behavior
    or override specific methods for specialized embedding requirements.

    Example:
        >>> service = DefaultEmbeddingService()
        >>> embedding = service.embed("sample text")
        >>> # Returns a numerical vector representation
    """
    pass
