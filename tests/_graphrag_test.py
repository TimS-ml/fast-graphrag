"""Unit tests for the BaseGraphRAG core functionality.

This module tests the main GraphRAG operations including document insertion
and querying capabilities of the base GraphRAG implementation.
"""
# type: ignore
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from fast_graphrag._graphrag import BaseGraphRAG
from fast_graphrag._models import TAnswer
from fast_graphrag._types import TContext, TQueryResponse


class TestBaseGraphRAG(unittest.IsolatedAsyncioTestCase):
    """Test suite for BaseGraphRAG class.

    Tests the core GraphRAG functionality including:
    - Document insertion and chunking
    - Information extraction
    - Query processing and response generation
    """
    def setUp(self):
        """Set up test fixtures before each test method.

        Creates mock instances of all required services and initializes
        a BaseGraphRAG instance without embedding validation for testing.
        """
        # Mock the core services used by GraphRAG
        self.llm_service = AsyncMock()
        self.chunking_service = AsyncMock()
        self.information_extraction_service = MagicMock()
        self.information_extraction_service.extract_entities_from_query = AsyncMock()
        self.state_manager = AsyncMock()
        # Set embedding dimensions to 1 for simplified testing
        self.state_manager.embedding_service.embedding_dim = self.state_manager.entity_storage.embedding_dim = 1

        @dataclass
        class BaseGraphRAGNoEmbeddingValidation(BaseGraphRAG):
            """Test subclass that skips embedding validation."""
            def __post_init__(self):
                pass

        # Initialize the GraphRAG instance with test configuration
        self.graph_rag = BaseGraphRAGNoEmbeddingValidation(
            working_dir="test_dir",
            domain="test_domain",
            example_queries="test_query",
            entity_types=["type1", "type2"],
        )
        # Inject mock services into the GraphRAG instance
        self.graph_rag.llm_service = self.llm_service
        self.graph_rag.chunking_service = self.chunking_service
        self.graph_rag.information_extraction_service = self.information_extraction_service
        self.graph_rag.state_manager = self.state_manager

    async def test_async_insert(self):
        """Test the async_insert method for document ingestion.

        Verifies that:
        1. Content is properly chunked
        2. New chunks are filtered
        3. Entities and relationships are extracted
        4. Graph is updated with extracted information
        """
        # Set up mock return values for the insertion pipeline
        self.chunking_service.extract = AsyncMock(return_value=["chunked_data"])
        self.state_manager.filter_new_chunks = AsyncMock(return_value=["new_chunks"])
        self.information_extraction_service.extract = MagicMock(return_value=["subgraph"])
        self.state_manager.upsert = AsyncMock()

        # Execute the insert operation
        await self.graph_rag.async_insert("test_content", {"meta": "data"})

        # Verify all pipeline stages were called
        self.chunking_service.extract.assert_called_once()
        self.state_manager.filter_new_chunks.assert_called_once()
        self.information_extraction_service.extract.assert_called_once()
        self.state_manager.upsert.assert_called_once()

    @patch("fast_graphrag._graphrag.format_and_send_prompt", new_callable=AsyncMock)
    async def test_async_query(self, format_and_send_prompt):
        """Test the async_query method for retrieving information.

        Verifies that:
        1. Entities are extracted from the query
        2. Relevant context is retrieved from the graph
        3. An answer is generated using the LLM
        4. Response is properly formatted
        """
        # Set up mock return values for the query pipeline
        self.information_extraction_service.extract_entities_from_query = AsyncMock(return_value=["entities"])
        self.state_manager.get_context = AsyncMock(return_value=TContext([], [], []))
        format_and_send_prompt.return_value=(TAnswer(answer="response"), None)

        # Execute the query operation
        response = await self.graph_rag.async_query("test_query")

        # Verify all query stages were called
        self.information_extraction_service.extract_entities_from_query.assert_called_once()
        self.state_manager.get_context.assert_called_once()
        format_and_send_prompt.assert_called_once()
        # Ensure response is properly typed
        self.assertIsInstance(response, TQueryResponse)


if __name__ == "__main__":
    unittest.main()
