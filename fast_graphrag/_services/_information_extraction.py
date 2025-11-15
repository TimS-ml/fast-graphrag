"""LLM-powered entity and relationship extraction service.

This module implements the second stage of the GraphRAG pipeline: converting unstructured
text chunks into structured knowledge graph elements (entities and relationships).

The extraction process leverages Large Language Models with specialized prompts to:
1. Identify entities (people, places, concepts, etc.) in text chunks
2. Identify relationships between entities
3. Iteratively refine extraction through "gleaning" (asking the LLM if anything was missed)
4. Merge entities and relationships from multiple chunks into a coherent graph
5. Validate entity types against a predefined schema

Key Features:
- **Parallel Processing**: Processes chunks concurrently for maximum throughput
- **Gleaning**: Iterative refinement to extract entities the LLM initially missed
- **Type Validation**: Ensures extracted entities match expected types
- **Graph Merging**: Combines per-chunk graphs while resolving duplicates
- **Query Entity Extraction**: Identifies entities in user queries for retrieval
- **Error Resilience**: Gracefully handles extraction failures per document

The quality of extraction directly impacts downstream retrieval and answer quality,
making prompt engineering and type validation critical.
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from fast_graphrag._llm import BaseLLMService, format_and_send_prompt
from fast_graphrag._models import TQueryEntities
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._storage._gdb_igraph import IGraphStorage, IGraphStorageConfig
from fast_graphrag._types import GTId, TChunk, TEntity, TGraph, TRelation
from fast_graphrag._utils import logger

from ._base import BaseInformationExtractionService


class TGleaningStatus(BaseModel):
    """Response model for the LLM's gleaning completion status.

    During the gleaning process, we ask the LLM if it has extracted all entities
    and relationships, or if it needs to continue searching.

    Attributes:
        status: Either "done" (extraction is complete) or "continue" (more entities
            may be present and should be extracted in the next gleaning iteration).
    """
    status: Literal["done", "continue"] = Field(
        description="done if all entities and relationship have been extracted, continue otherwise"
    )


@dataclass
class DefaultInformationExtractionService(BaseInformationExtractionService[TChunk, TEntity, TRelation, GTId]):
    """Production implementation of LLM-powered entity and relationship extraction.

    This service orchestrates the complete extraction pipeline:

    **Per-Document Processing**:
    1. Extract entities/relationships from each chunk independently (parallel)
    2. Optionally perform gleaning iterations to refine extraction
    3. Validate entity types against the provided schema
    4. Merge chunk-level graphs into a single document graph
    5. Return the merged graph or None if extraction failed

    **Query Processing**:
    - Extract entities from user queries for targeted retrieval
    - Categorize entities as "named" (specific) or "generic" (conceptual)

    The service uses structured output from the LLM (via Pydantic models) to ensure
    reliable parsing of extracted entities and relationships.
    """

    def extract(
        self,
        llm: BaseLLMService,
        documents: Iterable[Iterable[TChunk]],
        prompt_kwargs: Dict[str, str],
        entity_types: List[str],
    ) -> List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, GTId]]]]:
        """Extract entities and relationships from documents in parallel.

        This method creates an async task for each document, allowing multiple documents
        to be processed concurrently. Each task will extract from all chunks in that
        document and return a merged graph.

        Args:
            llm: The language model service to use for extraction.
            documents: Nested iterable where each inner iterable contains chunks
                from a single document.
            prompt_kwargs: Template variables for extraction prompts.
            entity_types: Valid entity types to extract (others marked as "UNKNOWN").

        Returns:
            List of futures that will resolve to graph storage objects (or None on failure).
        """
        return [
            asyncio.create_task(self._extract(llm, document, prompt_kwargs, entity_types)) for document in documents
        ]

    async def extract_entities_from_query(
        self, llm: BaseLLMService, query: str, prompt_kwargs: Dict[str, str]
    ) -> Dict[str, List[str]]:
        """Extract entities from a user query for targeted retrieval.

        This method identifies entities mentioned in the query, which are then used
        to find relevant nodes in the knowledge graph. It categorizes entities to
        enable different retrieval strategies:
        - Named entities: matched exactly (e.g., "Albert Einstein")
        - Generic entities: matched semantically (e.g., "physicist")

        Args:
            llm: The language model service to use.
            query: The user's search query string.
            prompt_kwargs: Template variables for the query entity extraction prompt.

        Returns:
            Dictionary with "named" and "generic" keys, each containing a list of
            extracted entity strings.
        """
        prompt_kwargs["query"] = query
        entities, _ = await format_and_send_prompt(
            prompt_key="entity_extraction_query",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TQueryEntities,
        )

        return {
            "named": entities.named,
            "generic": entities.generic
        }

    async def _extract(
        self, llm: BaseLLMService, chunks: Iterable[TChunk], prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> Optional[BaseGraphStorage[TEntity, TRelation, GTId]]:
        """Extract entities and relationships from all chunks in a document.

        This is the main extraction method for a single document. It:
        1. Processes all chunks in parallel
        2. Extracts a graph from each chunk
        3. Merges chunk graphs into a single document graph
        4. Returns None if extraction fails (logged as error)

        Args:
            llm: The language model service to use.
            chunks: All chunks from a single document.
            prompt_kwargs: Template variables for extraction prompts.
            entity_types: Valid entity types to extract.

        Returns:
            A graph storage containing all extracted entities and relationships,
            or None if extraction failed.
        """
        # Extract entities and relationships from each chunk in parallel
        try:
            chunk_graphs = await asyncio.gather(
                *[self._extract_from_chunk(llm, chunk, prompt_kwargs, entity_types) for chunk in chunks]
            )
            if len(chunk_graphs) == 0:
                return None

            # Combine chunk-level graphs into a single document-level graph
            return await self._merge(llm, chunk_graphs)
        except Exception as e:
            logger.error(f"Error during information extraction from document: {e}")
            return None

    async def _gleaning(
        self, llm: BaseLLMService, initial_graph: TGraph, history: list[dict[str, str]]
    ) -> Optional[TGraph]:
        """Perform iterative refinement to extract entities the LLM initially missed.

        "Gleaning" is the process of asking the LLM to review its extraction and find
        any entities or relationships it may have overlooked. This is done iteratively:

        1. Ask the LLM to continue extracting (with conversation history as context)
        2. Merge newly found entities/relationships with the existing graph
        3. Ask the LLM if it's done or should continue
        4. Repeat until done or max_gleaning_steps is reached

        This technique can significantly improve extraction recall, especially for
        dense or complex text.

        Args:
            llm: The language model service to use.
            initial_graph: The graph extracted in the first pass.
            history: Conversation history from the initial extraction.

        Returns:
            The enriched graph with gleaned entities/relationships, or None if
            gleaning failed.
        """
        current_graph = initial_graph

        try:
            for gleaning_count in range(self.max_gleaning_steps):
                # Ask the LLM to continue extraction (it has the previous conversation as context)
                gleaning_result, history = await format_and_send_prompt(
                    prompt_key="entity_relationship_continue_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGraph,
                    history_messages=history,
                )

                # Accumulate newly extracted entities and relationships
                current_graph.entities.extend(gleaning_result.entities)
                current_graph.relationships.extend(gleaning_result.relationships)

                # Stop if we've reached the maximum number of gleaning iterations
                if gleaning_count == self.max_gleaning_steps - 1:
                    break

                # Ask the LLM if extraction is complete or should continue
                gleaning_status, _ = await format_and_send_prompt(
                    prompt_key="entity_relationship_gleaning_done_extraction",
                    llm=llm,
                    format_kwargs={},
                    response_model=TGleaningStatus,
                    history_messages=history,
                )

                # If the LLM says it's done, stop gleaning early
                if gleaning_status.status == Literal["done"]:
                    break
        except Exception as e:
            logger.error(f"Error during gleaning: {e}")
            return None

        return current_graph

    async def _extract_from_chunk(
        self, llm: BaseLLMService, chunk: TChunk, prompt_kwargs: Dict[str, str], entity_types: List[str]
    ) -> TGraph:
        """Extract entities and relationships from a single chunk.

        This method handles the complete extraction pipeline for one chunk:

        1. **Initial Extraction**: Send chunk to LLM with extraction prompt
        2. **Gleaning**: Optionally perform iterative refinement
        3. **Type Validation**: Mark entities with invalid types as "UNKNOWN"
        4. **Chunk Association**: Tag relationships with their source chunk ID

        Args:
            llm: The language model service to use.
            chunk: The chunk to extract from.
            prompt_kwargs: Template variables for extraction prompts.
            entity_types: Valid entity types (normalized by removing spaces/underscores).

        Returns:
            A TGraph containing extracted entities and relationships.
        """
        # Insert the chunk content into the prompt template
        prompt_kwargs["input_text"] = chunk.content

        # Initial extraction: get entities and relationships from the LLM
        chunk_graph, history = await format_and_send_prompt(
            prompt_key="entity_relationship_extraction",
            llm=llm,
            format_kwargs=prompt_kwargs,
            response_model=TGraph,
        )

        # Perform gleaning if configured (max_gleaning_steps > 0)
        chunk_graph_with_gleaning = await self._gleaning(llm, chunk_graph, history)
        if chunk_graph_with_gleaning:
            chunk_graph = chunk_graph_with_gleaning

        # Validate entity types: normalize both extracted and expected types by
        # removing spaces/underscores and converting to uppercase
        _clean_entity_types = [re.sub("[ _]", "", entity_type).upper() for entity_type in entity_types]
        for entity in chunk_graph.entities:
            if re.sub("[ _]", "", entity.type).upper() not in _clean_entity_types:
                entity.type = "UNKNOWN"

        # Associate each relationship with its source chunk ID for provenance tracking
        for relationship in chunk_graph.relationships:
            relationship.chunks = [chunk.id]

        return chunk_graph

    async def _merge(self, llm: BaseLLMService, graphs: List[TGraph]) -> BaseGraphStorage[TEntity, TRelation, GTId]:
        """Merge multiple chunk-level graphs into a single document-level graph.

        This method combines graphs from all chunks in a document while:
        - Resolving duplicate entities via the graph_upsert policy
        - Resolving duplicate relationships via the graph_upsert policy
        - Preserving all chunk associations for relationships

        The graph_upsert policy determines how conflicts are resolved (e.g., merging
        descriptions, combining attributes).

        Args:
            llm: The language model service (used by upsert policy for resolution).
            graphs: List of chunk-level graphs to merge.

        Returns:
            A graph storage containing the merged, deduplicated entities and relationships.
        """
        # Create a new graph storage for the merged result
        graph_storage = IGraphStorage[TEntity, TRelation, GTId](config=IGraphStorageConfig(TEntity, TRelation))

        await graph_storage.insert_start()

        try:
            # Sequentially merge each chunk graph into the storage
            # This must be sequential (not parallel) because the upsert policy
            # needs to resolve conflicts against the current state
            for graph in graphs:
                await self.graph_upsert(llm, graph_storage, graph.entities, graph.relationships)
        finally:
            await graph_storage.insert_done()

        return graph_storage
