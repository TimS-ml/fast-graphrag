"""Unit tests for OpenAI LLM and embedding services.

This module tests the OpenAI-specific implementations of:
- LLM service for text generation with retry logic
- Embedding service for vector generation
- Error handling for rate limits and API connection issues
"""
# type: ignore
import os
import unittest
from unittest.mock import AsyncMock, MagicMock

import instructor
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import RetryError

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._llm._llm_openai import OpenAIEmbeddingService, OpenAILLMService

# Set empty API key for testing
os.environ["OPENAI_API_KEY"] = ""

# Create a mock RateLimitError for testing retry logic
RateLimitError429 = RateLimitError(message="Rate limit exceeded", response=MagicMock(), body=None)


class TestOpenAILLMService(unittest.IsolatedAsyncioTestCase):
    """Test suite for OpenAI LLM service."""
    async def test_send_message_success(self):
        """Test successful message sending to OpenAI LLM.

        Verifies that the LLM response and message history are returned correctly.
        """
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response, messages = await service.send_message(prompt="Hello")

        self.assertEqual(response, mock_response)
        # Verify assistant message is added to history
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_no_response(self):
        """Test handling of empty LLM response.

        Verifies that LLMServiceNoResponseError is raised when the LLM
        returns None instead of a valid response.
        """
        service = OpenAILLMService(api_key="test")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create.return_value = None

        with self.assertRaises(LLMServiceNoResponseError):
            await service.send_message(prompt="Hello")

    async def test_send_message_rate_limit_error(self):
        """Test automatic retry on rate limit error.

        Verifies that the service retries the request after a rate limit
        error and succeeds on the second attempt.
        """
        service = OpenAILLMService()
        mock_response = str("Hi!")
        async_open_ai = AsyncOpenAI(api_key="test")
        # First call raises RateLimitError, second succeeds
        async_open_ai.chat.completions.create = AsyncMock(
            side_effect=(RateLimitError429, mock_response)
        )
        service.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(
            async_open_ai
        )

        response, messages = await service.send_message(prompt="Hello", response_model=None)

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_api_connection_error(self):
        """Test automatic retry on API connection error.

        Verifies that the service retries the request after an API
        connection error and succeeds on the second attempt.
        """
        service = OpenAILLMService()
        mock_response = str("Hi!")
        async_open_ai = AsyncOpenAI(api_key="test")
        # First call raises APIConnectionError, second succeeds
        async_open_ai.chat.completions.create = AsyncMock(
            side_effect=(APIConnectionError(request=MagicMock()), mock_response)
        )
        service.llm_async_client: instructor.AsyncInstructor = instructor.from_openai(
            async_open_ai
        )

        response, messages = await service.send_message(prompt="Hello")

        self.assertEqual(response, mock_response)
        self.assertEqual(messages[-1]["role"], "assistant")

    async def test_send_message_with_system_prompt(self):
        """Test sending message with a system prompt.

        Verifies that the system prompt is correctly added as the first
        message in the conversation history.
        """
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        response, messages = await service.send_message(
            prompt="Hello", system_prompt="System prompt"
        )

        self.assertEqual(response, mock_response)
        # Verify system prompt is first message
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")

    async def test_send_message_with_history(self):
        """Test sending message with conversation history.

        Verifies that existing conversation history is preserved
        and prepended to the new message.
        """
        service = OpenAILLMService(api_key="test")
        mock_response = str("Hi!")
        service.llm_async_client = AsyncMock()
        service.llm_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        history = [{"role": "user", "content": "Previous message"}]
        response, messages = await service.send_message(prompt="Hello", history_messages=history)

        self.assertEqual(response, mock_response)
        # Verify history is preserved
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Previous message")


class TestOpenAIEmbeddingService(unittest.IsolatedAsyncioTestCase):
    """Test suite for OpenAI embedding service."""
    async def test_get_embedding_success(self):
        """Test successful embedding generation.

        Verifies that embeddings are returned in the correct shape
        and with the expected values.
        """
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        service.embedding_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        # Verify shape is (num_texts, embedding_dim)
        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_rate_limit_error(self):
        """Test automatic retry on rate limit error for embeddings.

        Verifies that the service retries embedding generation after
        a rate limit error and succeeds on the second attempt.
        """
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        # First call raises RateLimitError, second succeeds
        service.embedding_async_client.embeddings.create = AsyncMock(side_effect=(RateLimitError429, mock_response))

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_api_connection_error(self):
        """Test automatic retry on API connection error for embeddings.

        Verifies that the service retries embedding generation after
        an API connection error and succeeds on the second attempt.
        """
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
        # First call raises APIConnectionError, second succeeds
        service.embedding_async_client.embeddings.create = AsyncMock(
            side_effect=(APIConnectionError(request=MagicMock()), mock_response)
        )
        embeddings = await service.encode(texts=["test"], model="text-embedding-3-small")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.1)

    async def test_get_embedding_retry_failure(self):
        """Test that retry eventually fails after max attempts.

        Verifies that RetryError is raised when all retry attempts
        are exhausted due to persistent rate limit errors.
        """
        service = OpenAIEmbeddingService(api_key="test")
        # Always raise RateLimitError
        service.embedding_async_client.embeddings.create = AsyncMock(
            side_effect=RateLimitError429
        )

        with self.assertRaises(RetryError):
            await service.encode(texts=["test"], model="text-embedding-3-small")

    async def test_get_embedding_with_different_model(self):
        """Test embedding generation with a different model.

        Verifies that the service works correctly with different
        embedding models (e.g., text-embedding-3-large).
        """
        service = OpenAIEmbeddingService(api_key="test")
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.4, 0.5, 0.6])]
        service.embedding_async_client.embeddings.create = AsyncMock(return_value=mock_response)

        embeddings = await service.encode(texts=["test"], model="text-embedding-3-large")

        self.assertEqual(embeddings.shape, (1, 3))
        self.assertEqual(embeddings[0][0], 0.4)


if __name__ == "__main__":
    unittest.main()
