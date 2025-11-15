"""LLM (Large Language Model) and Embedding Services Module.

This module provides a unified interface for interacting with various Large Language
Model (LLM) and embedding service providers. It abstracts the complexities of different
API implementations while maintaining a consistent, easy-to-use interface.

The module includes:
    - Abstract base classes for LLM and embedding services that define standard interfaces
    - Default implementations for quick prototyping and testing
    - Provider-specific implementations for OpenAI, Google Gemini, and VoyageAI
    - Utility functions for prompt formatting and LLM interaction

Key Components:
    Base Classes:
        - BaseLLMService: Abstract base class defining the interface for LLM services
        - BaseEmbeddingService: Abstract base class for text embedding services

    Default Implementations:
        - DefaultLLMService: Default LLM service implementation
        - DefaultEmbeddingService: Default embedding service implementation

    Provider Implementations:
        - OpenAILLMService: Integration with OpenAI's API (GPT models)
        - OpenAIEmbeddingService: Integration with OpenAI's embedding models
        - GeminiLLMService: Integration with Google's Gemini models
        - GeminiEmbeddingService: Integration with Google's embedding services
        - VoyageAIEmbeddingService: Integration with VoyageAI's embedding models

Typical Usage:
    from fast_graphrag._llm import OpenAILLMService, OpenAIEmbeddingService

    # Initialize services
    llm_service = OpenAILLMService(api_key="your-api-key")
    embedding_service = OpenAIEmbeddingService(api_key="your-api-key")

    # Use services for text generation and embedding operations
    response = llm_service.generate(prompt="Your prompt here")
    embeddings = embedding_service.embed(text="Text to embed")

Module Organization:
    This module follows a plugin-like architecture where new LLM/embedding providers
    can be added by creating new service classes that inherit from the base classes.
"""

# Public API exports for the LLM module
__all__ = [
    # Base classes defining standard interfaces for LLM and embedding services
    "BaseLLMService",
    "BaseEmbeddingService",

    # Default implementations for basic usage and testing
    "DefaultEmbeddingService",
    "DefaultLLMService",

    # Utility function for formatting and sending prompts to LLM services
    "format_and_send_prompt",

    # OpenAI provider implementations
    "OpenAIEmbeddingService",
    "OpenAILLMService",

    # Google Gemini provider implementations
    "GeminiLLMService",
    "GeminiEmbeddingService",

    # VoyageAI provider implementations
    "VoyageAIEmbeddingService"
]

# Import base classes and utility functions
from ._base import BaseEmbeddingService, BaseLLMService, format_and_send_prompt

# Import default implementations
from ._default import DefaultEmbeddingService, DefaultLLMService

# Import Google Gemini provider implementations
from ._llm_genai import GeminiEmbeddingService, GeminiLLMService

# Import OpenAI provider implementations
from ._llm_openai import OpenAIEmbeddingService, OpenAILLMService

# Import VoyageAI provider implementations
from ._llm_voyage import VoyageAIEmbeddingService
