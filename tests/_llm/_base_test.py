"""Unit tests for base LLM service functionality.

This module tests the core LLM service interface, including:
- Prompt formatting and sending
- Response model validation
- LLM service integration
"""
# type: ignore
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from fast_graphrag._llm._base import BaseLLMService, format_and_send_prompt

# Mock prompts for testing
PROMPTS = {
    "example_prompt": "Hello, {name}!"
}

class TestModel(BaseModel):
    """Test Pydantic model for response validation."""
    answer: str

class TestFormatAndSendPrompt(unittest.IsolatedAsyncioTestCase):
    """Test suite for format_and_send_prompt utility function."""

    @patch("fast_graphrag._llm._base.PROMPTS", PROMPTS)
    async def test_format_and_send_prompt(self):
        """Test basic prompt formatting and sending.

        Verifies that:
        - Prompt template is formatted with provided kwargs
        - LLM service receives the formatted prompt
        - Response model is passed through correctly
        - Response is returned unchanged
        """
        mock_llm = AsyncMock(spec=BaseLLMService(model=""))
        answer = TestModel(answer="TEST")
        mock_response = (answer, [{"key": "value"}])
        mock_llm.send_message = AsyncMock(return_value=mock_response)

        # Format and send prompt with name parameter
        result = await format_and_send_prompt(
            prompt_key="example_prompt",
            llm=mock_llm,
            format_kwargs={"name": "World"},
            response_model=TestModel
        )

        # Verify LLM received formatted prompt
        mock_llm.send_message.assert_called_once_with(
            prompt="Hello, World!",
            response_model=TestModel
        )
        self.assertEqual(result, mock_response)

    @patch("fast_graphrag._llm._base.PROMPTS", PROMPTS)
    async def test_format_and_send_prompt_with_additional_args(self):
        """Test prompt formatting with additional LLM parameters.

        Verifies that extra kwargs (model, max_tokens) are passed through
        to the LLM service send_message method.
        """
        mock_llm = AsyncMock(spec=BaseLLMService(model=""))
        answer = TestModel(answer="TEST")
        mock_response = (answer, [{"key": "value"}])
        mock_llm.send_message = AsyncMock(return_value=mock_response)

        # Format and send prompt with additional LLM parameters
        result = await format_and_send_prompt(
            prompt_key="example_prompt",
            llm=mock_llm,
            format_kwargs={"name": "World"},
            response_model=TestModel,
            model="test_model",
            max_tokens=100
        )

        # Verify all parameters were passed to LLM service
        mock_llm.send_message.assert_called_once_with(
            prompt="Hello, World!",
            response_model=TestModel,
            model="test_model",
            max_tokens=100
        )
        self.assertEqual(result, mock_response)

if __name__ == "__main__":
    unittest.main()
