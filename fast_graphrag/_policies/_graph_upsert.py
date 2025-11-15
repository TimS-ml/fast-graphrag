"""Graph upsert policy implementations.

This module provides concrete implementations of graph upsert policies that control
how nodes (entities) and edges (relationships) are added or updated in the knowledge graph.

The policies in this module implement various strategies for:
- Merging duplicate entities by combining their descriptions
- Summarizing verbose entity descriptions using LLMs
- Validating edge existence (checking that source/target nodes exist)
- Detecting and merging similar relationships between the same entities
- Maintaining data provenance through chunk tracking

Key Features:
    - Default policies: Simple, efficient upsert with basic deduplication
    - Summarization policies: Use LLMs to condense lengthy entity descriptions
    - Validation policies: Ensure referential integrity by validating node existence
    - Similarity-based merging: Leverage LLMs to detect and merge semantically similar edges

The module uses async/await extensively to enable parallel processing of large batches
of nodes and edges, significantly improving throughput when building graphs from documents.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Counter, Dict, Iterable, List, Optional, Set, Tuple, Union

from fast_graphrag._llm._base import format_and_send_prompt
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._models import TEditRelationList, TEntityDescription
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import GTEdge, GTId, GTNode, TEntity, THash, TId, TIndex, TRelation
from fast_graphrag._utils import logger

from ._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy


async def summarize_entity_description(
    prompt: str, description: str, llm: BaseLLMService, max_tokens: Optional[int] = None
) -> str:
    """Summarize the given entity description using an LLM.

    This function uses a language model to condense lengthy entity descriptions
    while preserving key information. It supports both single-prompt and
    system/user prompt pair formats.

    Args:
        prompt: Key to lookup the summarization prompt template from PROMPTS.
        description: The entity description text to summarize.
        llm: Language model service to perform the summarization.
        max_tokens: Maximum tokens in the summary. Currently not implemented.

    Returns:
        The summarized description text.

    Raises:
        NotImplementedError: If max_tokens is specified (not yet supported).
    """
    if max_tokens is not None:
        raise NotImplementedError("Summarization with max tokens is not yet supported.")

    # Check if the prompt uses a system/user message pair or a single prompt
    system_key = prompt + '_system'

    if system_key in PROMPTS:
        # Use separate system and user prompts for better LLM instruction following
        entity_description_summarization_system = PROMPTS[system_key]
        entity_description_summarization_prompt = PROMPTS[prompt + '_prompt']

        # Format both prompts with the entity description
        formatted_system = entity_description_summarization_system.format(description=description)
        formatted_prompt = entity_description_summarization_prompt.format(description=description)

        # Request structured output using Pydantic model for reliable parsing
        new_description, _ = await llm.send_message(
            system_prompt=formatted_system,
            prompt=formatted_prompt,
            response_model=TEntityDescription,
            max_tokens=max_tokens,
        )
    else:
        # Fallback to single prompt format for simpler use cases
        entity_description_summarization_prompt = PROMPTS[prompt]
        formatted_entity_description_summarization_prompt = entity_description_summarization_prompt.format(
            description=description
        )
        new_description, _ = await llm.send_message(
            prompt=formatted_entity_description_summarization_prompt,
            response_model=TEntityDescription,
            max_tokens=max_tokens,
        )

    return new_description.description


####################################################################################################
# DEFAULT GRAPH UPSERT POLICIES
####################################################################################################
# These are simple, baseline implementations that provide basic upsert functionality
# without advanced features like summarization or similarity-based merging.


@dataclass
class DefaultNodeUpsertPolicy(BaseNodeUpsertPolicy[GTNode, GTId]):
    """Default node upsert policy with basic duplicate handling.

    This policy provides straightforward node upsert behavior:
    - If a node with the same identifier exists, update it
    - If the node doesn't exist, insert it as new
    - No merging, summarization, or complex conflict resolution

    This is suitable for simple use cases where nodes are already deduplicated
    or when you want minimal processing overhead.
    """
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_nodes: Iterable[GTNode]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTNode]]]:
        """Upsert nodes with simple update-or-insert logic.

        Args:
            llm: Language model service (unused in this simple policy).
            target: The graph storage to update.
            source_nodes: Nodes to be upserted.

        Returns:
            Tuple of (updated storage, iterator of (index, node) pairs).
        """
        upserted: Dict[TIndex, GTNode] = {}
        for node in source_nodes:
            # Check if node already exists in the graph
            _, index = await target.get_node(node)
            if index is not None:
                # Update existing node
                await target.upsert_node(node=node, node_index=index)
            else:
                # Insert new node
                index = await target.upsert_node(node=node, node_index=None)
            upserted[index] = node

        return target, upserted.items()


@dataclass
class DefaultEdgeUpsertPolicy(BaseEdgeUpsertPolicy[GTEdge, GTId]):
    """Default edge upsert policy with simple insertion.

    This policy provides basic edge insertion without validation or merging:
    - All edges are inserted as provided
    - No validation that source/target nodes exist
    - No deduplication or similarity-based merging

    Use this when you've already validated edges or want maximum insertion speed.
    """
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_edges: Iterable[GTEdge]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTEdge]]]:
        """Insert all edges without validation.

        Args:
            llm: Language model service (unused in this simple policy).
            target: The graph storage to update.
            source_edges: Edges to be inserted.

        Returns:
            Tuple of (updated storage, iterator of (index, edge) pairs).
        """
        # Bulk insert all edges at once for efficiency
        indices = await target.insert_edges(source_edges)

        # Pair indices with their corresponding edges
        r: Iterable[Tuple[TIndex, GTEdge]]
        if len(indices):
            r = zip(indices, source_edges)
        else:
            r = []
        return target, r


@dataclass
class DefaultGraphUpsertPolicy(BaseGraphUpsertPolicy[GTNode, GTEdge, GTId]):  # noqa: N801
    """Default graph upsert policy combining node and edge policies.

    This policy orchestrates the default node and edge upsert operations:
    1. First, upsert all nodes using the node policy
    2. Then, insert all edges using the edge policy

    This sequential approach ensures nodes exist before edges reference them,
    though the default edge policy doesn't enforce this validation.
    """
    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, GTEdge, GTId],
        source_nodes: Iterable[GTNode],
        source_edges: Iterable[GTEdge],
    ) -> Tuple[
        BaseGraphStorage[GTNode, GTEdge, GTId],
        Iterable[Tuple[TIndex, GTNode]],
        Iterable[Tuple[TIndex, GTEdge]],
    ]:
        """Execute node and edge upsert operations sequentially.

        Args:
            llm: Language model service (passed to sub-policies).
            target: The graph storage to update.
            source_nodes: Nodes to be upserted.
            source_edges: Edges to be inserted.

        Returns:
            Tuple of (updated storage, node pairs, edge pairs).
        """
        # Upsert nodes first to ensure they exist
        target, upserted_nodes = await self._nodes_upsert(llm, target, source_nodes)
        # Then insert edges
        target, upserted_edges = await self._edges_upsert(llm, target, source_edges)

        return target, upserted_nodes, upserted_edges


####################################################################################################
# NODE UPSERT POLICIES
####################################################################################################
# Advanced node upsert policies with features like description summarization,
# type resolution, and intelligent merging of duplicate entities.


@dataclass
class NodeUpsertPolicy_SummarizeDescription(BaseNodeUpsertPolicy[TEntity, TId]):  # noqa: N801
    """Node upsert policy with LLM-based description summarization.

    This policy intelligently merges nodes with the same identifier while managing
    their descriptions:
    - Groups nodes by name/ID
    - Combines descriptions from all instances
    - Uses LLM to summarize if description exceeds size limit
    - Resolves entity type by selecting the most frequent type

    This is ideal for building knowledge graphs from multiple documents where the
    same entity appears multiple times with different contextual descriptions.

    Attributes:
        config: Configuration for summarization behavior.
    """
    @dataclass
    class Config:
        """Configuration for description summarization.

        Attributes:
            max_node_description_size: Maximum character length for descriptions
                before triggering summarization.
            node_summarization_ratio: Target compression ratio for summarization
                (currently unused, may be used for token-based limits).
            node_summarization_prompt: Prompt key for LLM-based summarization.
            is_async: Whether to process nodes in parallel (True) or sequentially (False).
        """
        max_node_description_size: int = field(default=512)
        node_summarization_ratio: float = field(default=0.5)
        node_summarization_prompt: str = field(default="summarize_entity_descriptions")
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[TEntity, GTEdge, TId], source_nodes: Iterable[TEntity]
    ) -> Tuple[BaseGraphStorage[TEntity, GTEdge, TId], Iterable[Tuple[TIndex, TEntity]]]:
        """Upsert nodes with description merging and summarization.

        Process flow:
        1. Group source nodes by name/ID
        2. For each group, merge with existing node if present
        3. Combine all descriptions with newlines
        4. Summarize using LLM if combined description is too long
        5. Resolve type conflicts by selecting most frequent type
        6. Upsert the final merged node

        Args:
            llm: Language model service for summarization.
            target: The graph storage to update.
            source_nodes: Nodes to be upserted.

        Returns:
            Tuple of (updated storage, list of (index, node) pairs).
        """
        upserted: List[Tuple[TIndex, TEntity]] = []

        async def _upsert_node(node_id: TId, nodes: List[TEntity]) -> Optional[Tuple[TIndex, TEntity]]:
            """Process and upsert a single node group.

            Args:
                node_id: The identifier for this entity.
                nodes: All instances of this entity from source data.

            Returns:
                Tuple of (index, merged node) or None.
            """
            # Retrieve existing node if present
            existing_node, index = await target.get_node(node_id)
            if existing_node:
                # Include existing description in the merge
                nodes.append(existing_node)

            # Combine all descriptions with newline separators
            node_description = "\n".join((node.description for node in nodes))

            # Summarize if description is too long
            if len(node_description) > self.config.max_node_description_size:
                node_description = await summarize_entity_description(
                    self.config.node_summarization_prompt,
                    node_description,
                    llm,
                    # Token-based limiting (currently disabled):
                    # int(
                    #     self.config.max_node_description_size
                    #     * self.config.node_summarization_ratio
                    #     / TOKEN_TO_CHAR_RATIO
                    # ),
                )

            # Resolve type conflicts by majority vote
            node_type = Counter((node.type for node in nodes)).most_common(1)[0][0]

            # Create the final merged entity
            node = TEntity(name=node_id, description=node_description, type=node_type)
            index = await target.upsert_node(node=node, node_index=index)

            upserted.append((index, node))

        # Group source nodes by their identifier
        grouped_nodes: Dict[TId, List[TEntity]] = defaultdict(lambda: [])
        for node in source_nodes:
            grouped_nodes[node.name].append(node)

        # Process groups in parallel or sequentially based on configuration
        if self.config.is_async:
            node_upsert_tasks = (_upsert_node(node_id, nodes) for node_id, nodes in grouped_nodes.items())
            await asyncio.gather(*node_upsert_tasks)
        else:
            for node_id, nodes in grouped_nodes.items():
                await _upsert_node(node_id, nodes)

        return target, upserted


####################################################################################################
# EDGE UPSERT POLICIES
####################################################################################################
# Advanced edge upsert policies with validation, similarity detection,
# and intelligent merging of redundant relationships.


@dataclass
class EdgeUpsertPolicy_UpsertIfValidNodes(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    """Edge upsert policy with node existence validation.

    This policy ensures referential integrity by only inserting edges where
    both the source and target nodes exist in the graph. This prevents dangling
    edges that reference non-existent entities.

    Process:
    - For each edge, verify source and target nodes exist
    - Only insert edges with valid node references
    - Discard edges with missing nodes

    This is the recommended policy when building graphs incrementally or when
    edge extraction might produce references to entities not yet in the graph.

    Attributes:
        config: Configuration for async processing.
    """
    @dataclass
    class Config:
        """Configuration for edge validation.

        Attributes:
            is_async: Whether to validate edges in parallel (True) or sequentially (False).
        """
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        """Validate and insert edges with valid node references.

        Args:
            llm: Language model service (unused in this policy).
            target: The graph storage to update.
            source_edges: Edges to be validated and inserted.

        Returns:
            Tuple of (updated storage, iterator of (index, edge) pairs for valid edges).
        """
        new_edges: List[TRelation] = []

        async def _upsert_edge(edge: TRelation) -> Optional[Tuple[TIndex, TRelation]]:
            """Validate a single edge and add to new_edges if valid.

            Args:
                edge: The edge to validate.

            Returns:
                None (accumulates valid edges in new_edges list).
            """
            # Check if both source and target nodes exist
            source_node, _ = await target.get_node(edge.source)
            target_node, _ = await target.get_node(edge.target)

            # Only include edge if both nodes are present
            if source_node and target_node:
                new_edges.append(edge)

        # Validate edges in parallel or sequentially
        if self.config.is_async:
            edge_upsert_tasks = (_upsert_edge(edge) for edge in source_edges)
            await asyncio.gather(*edge_upsert_tasks)
        else:
            for edge in source_edges:
                await _upsert_edge(edge)

        # Bulk insert all validated edges
        indices = await target.insert_edges(new_edges)

        # Pair indices with edges
        r: Iterable[Tuple[TIndex, TRelation]]
        if len(indices):
            r = zip(indices, new_edges)
        else:
            r = []

        return target, r


@dataclass
class EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    """Advanced edge upsert policy with LLM-based similarity merging.

    This policy combines validation with intelligent deduplication:
    - Validates that source and target nodes exist
    - Groups edges by source/target node pair
    - Uses LLM to detect semantically similar edge descriptions
    - Merges similar edges to reduce redundancy
    - Tracks chunk provenance across merged edges

    The similarity detection is particularly powerful because:
    - It goes beyond exact string matching
    - It understands semantic equivalence (e.g., "works at" vs "employed by")
    - It consolidates redundant information from multiple sources

    This policy is ideal for high-quality knowledge graphs where reducing noise
    and redundancy is more important than raw insertion speed.

    Attributes:
        config: Configuration for merging behavior.
    """
    @dataclass
    class Config:
        """Configuration for LLM-based edge merging.

        Attributes:
            edge_merge_threshold: Minimum number of edges between a node pair
                before triggering merge analysis. Lower values run merging more
                frequently but increase LLM calls.
            is_async: Whether to process node pairs in parallel (True) or sequentially (False).
        """
        edge_merge_threshold: int = field(default=5)
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def _upsert_edge(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        edges: List[TRelation],
        source_entity: TId,
        target_entity: TId,
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        """Process edges between a specific node pair, merging if needed.

        This method decides whether to trigger LLM-based merging based on the
        total number of edges (existing + new) between the node pair.

        Args:
            llm: Language model service for similarity detection.
            target: The graph storage to query and update.
            edges: New edges to insert between this node pair.
            source_entity: Source node identifier.
            target_entity: Target node identifier.

        Returns:
            A tuple containing:
            - List of (index, edge) pairs for updated existing edges
            - List of new edges to insert
            - List of indices of edges to delete
        """
        # Retrieve all existing edges between this node pair
        existing_edges = list(await target.get_edges(source_entity, target_entity))

        # Trigger merge analysis if total edges exceed threshold
        if (len(existing_edges) + len(edges)) > self.config.edge_merge_threshold:
            upserted_eges, new_edges, to_delete_edges = await self._merge_similar_edges(
                llm, target, existing_edges, edges
            )
        else:
            # Below threshold: just insert new edges without merging
            upserted_eges = []
            new_edges = edges
            to_delete_edges = []

        return upserted_eges, new_edges, to_delete_edges

    async def _merge_similar_edges(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        existing_edges: List[Tuple[TRelation, TIndex]],
        edges: List[TRelation],
    ) -> Tuple[List[Tuple[TIndex, TRelation]], List[TRelation], List[TIndex]]:
        """Merge similar edges between the same pair of nodes using LLM-based similarity detection.

        This is the core similarity-based merging algorithm:
        1. Create a unified index mapping for both existing and new edges
        2. Send all edge descriptions to the LLM for grouping by similarity
        3. For each group of similar edges:
           - Keep the first edge as the canonical version
           - Merge descriptions and chunk provenance
           - Mark remaining edges for deletion
        4. Edges not grouped are considered unique and inserted as-is

        Args:
            llm: The language model that determines similarity between edges.
            target: The graph storage to upsert the edges to.
            existing_edges: List of (edge, index) tuples for edges already in the graph.
            edges: List of new edges to be upserted.

        Returns:
            A tuple containing:
            - List of (index, edge) pairs for updated existing edges
            - List of new edges that were not merged
            - List of indices of edges to be deleted
        """
        updated_edges: List[Tuple[TIndex, TRelation]] = []
        new_edges: List[TRelation] = []

        # Create a unified mapping: existing edges get indices 0..N-1, new edges get N..M
        # This allows the LLM to reference edges by a simple integer ID
        map_incremental_to_edge: Dict[int, Tuple[TRelation, Union[TIndex, None]]] = {
            **dict(enumerate(existing_edges)),  # Existing edges with their storage indices
            **{idx + len(existing_edges): (edge, None) for idx, edge in enumerate(edges)},  # New edges (index=None)
        }

        # Ask the LLM to group similar edge descriptions
        # The prompt sends a numbered list of edge descriptions and gets back groups of similar ones
        edge_grouping, _ = await format_and_send_prompt(
            prompt_key="edges_group_similar",
            llm=llm,
            format_kwargs={
                "edge_list": "\n".join(
                    (f"{idx}, {edge.description}" for idx, (edge, _) in map_incremental_to_edge.items())
                )
            },
            response_model=TEditRelationList,  # Structured output: list of edge groups
        )

        # Track which edges we've processed and whether they're marked for deletion
        # Key: incremental index, Value: storage index (None = keep, non-None = delete)
        visited_edges: Dict[TIndex, Union[TIndex, None]] = {}

        # Process each group of similar edges returned by the LLM
        for edges_group in edge_grouping.groups:
            # Filter to only valid edge indices (LLM might hallucinate invalid ones)
            relation_indices = [
                index
                for index in edges_group.ids
                if index < len(existing_edges) + len(edges)
            ]

            # A group needs at least 2 edges to be meaningful
            if len(relation_indices) < 2:
                logger.info("LLM returned invalid index for edge maintenance, ignoring.")
                continue

            # Collect all chunk hashes from edges being merged (for provenance tracking)
            chunks: Set[THash] = set()

            # Process all edges after the first (these will be deleted/merged)
            for second in relation_indices[1:]:
                edge, index = map_incremental_to_edge[second]

                # Mark edge for deletion on first visit
                # If we see it again in another group, we keep the first marking
                if second not in visited_edges:
                    visited_edges[second] = index  # Non-None index = mark for deletion
                if edge.chunks:
                    chunks.update(edge.chunks)

            # The first edge in the group becomes the canonical version
            first_index = relation_indices[0]
            edge, index = map_incremental_to_edge[first_index]

            # Use the LLM-generated merged description
            edge.description = edges_group.description

            # Mark as visited but NOT for deletion
            visited_edges[first_index] = None

            # Collect chunks from the canonical edge too
            if edge.chunks:
                chunks.update(edge.chunks)

            # Update the edge with all merged chunks
            edge.chunks = list(chunks)

            # Update existing edge or prepare new edge for insertion
            if index is not None:
                # This is an existing edge - update it in place
                updated_edges.append((await target.upsert_edge(edge, index), edge))
            else:
                # This is a new edge - add to insertion list
                new_edges.append(edge)

        # Handle new edges that weren't grouped (they're unique)
        for idx, edge in enumerate(edges):
            incremental_idx = idx + len(existing_edges)
            if incremental_idx not in visited_edges:
                # Edge was not grouped, so insert it as-is
                new_edges.append(edge)

        # Return: updated edges, new edges to insert, and indices to delete
        # Only existing edges marked for deletion have non-None values
        return updated_edges, new_edges, [v for v in visited_edges.values() if v is not None]

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        """Validate, merge similar edges, and insert into the graph.

        Process flow:
        1. Group source edges by (source, target) node pair
        2. For each node pair, process edges (with merging if above threshold)
        3. Delete redundant edges that were merged
        4. Insert new edges (including merged ones)
        5. Return all upserted edges

        Args:
            llm: Language model service for similarity detection.
            target: The graph storage to update.
            source_edges: Edges to be validated and inserted.

        Returns:
            Tuple of (updated storage, iterator of all (index, edge) pairs).
        """
        # Group edges by their source and target node pair
        grouped_edges: Dict[Tuple[TId, TId], List[TRelation]] = defaultdict(lambda: [])
        upserted_edges: Tuple[List[Tuple[TIndex, TRelation]], ...] = ()
        new_edges: Tuple[List[TRelation], ...] = ()
        to_delete_edges: Tuple[List[TIndex], ...] = ()

        for edge in source_edges:
            grouped_edges[(edge.source, edge.target)].append(edge)

        # Process each node pair in parallel or sequentially
        if self.config.is_async:
            edge_upsert_tasks = (
                self._upsert_edge(llm, target, edges, source_entity, target_entity)
                for (source_entity, target_entity), edges in grouped_edges.items()
            )
            tasks = await asyncio.gather(*edge_upsert_tasks)
            if len(tasks):
                # Unpack results from all node pairs
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)
        else:
            tasks = [
                await self._upsert_edge(llm, target, edges, source_entity, target_entity)
                for (source_entity, target_entity), edges in grouped_edges.items()
            ]
            if len(tasks):
                upserted_edges, new_edges, to_delete_edges = zip(*tasks)

        # Delete all edges that were merged away
        await target.delete_edges_by_index(tuple(chain(*to_delete_edges)))

        # Insert all new edges (including canonical versions of merged groups)
        new_indices = await target.insert_edges(tuple(chain(*new_edges)))

        # Return combined list of updated edges and newly inserted edges
        return target, chain(*upserted_edges, zip(new_indices, chain(*new_edges)))
