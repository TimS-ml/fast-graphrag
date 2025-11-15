"""Base policy classes for graph operations.

This module defines the foundational policy interfaces used throughout the fast-graphrag system.
Policies are pluggable strategies that control how graph operations are performed, such as:
- How nodes and edges are upserted into the graph
- How duplicate or similar entities are merged
- How search results are ranked and filtered

Policies enable customization of core graph behaviors without modifying the underlying storage
or retrieval logic. They act as a strategy pattern, allowing different implementations to be
swapped in based on use case requirements.

Key Concepts:
    - Policies separate configuration from logic, making the system more flexible and testable
    - Each policy type (upsert, ranking, etc.) has a base class defining the interface
    - Concrete implementations can override behavior while maintaining consistent APIs
    - Policies receive LLM services as parameters, enabling AI-powered decision making
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Tuple, Type

from scipy.sparse import csr_matrix

from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import GTEdge, GTId, GTNode, TIndex


@dataclass
class BasePolicy:
    """Base class for all policy implementations.

    This abstract base provides a common interface for policy objects. All policies
    require a configuration object that specifies their behavior parameters.

    Attributes:
        config: Configuration object containing policy-specific parameters.
            The structure and content of this config varies by policy type.
    """
    config: Any = field()


####################################################################################################
# GRAPH UPSERT POLICIES
####################################################################################################
# These policies control how nodes and edges are inserted or updated (upserted) in the graph.
# They handle deduplication, merging, and conflict resolution when adding new information.


@dataclass
class BaseNodeUpsertPolicy(BasePolicy, Generic[GTNode, GTId]):
    """Base policy for node upsert operations.

    Node upsert policies determine how nodes (entities) are added or updated in the graph.
    This includes handling duplicates, merging descriptions, and resolving conflicts.

    Common strategies include:
    - Simple insert: Add new nodes, update existing ones by ID
    - Similarity-based merging: Combine nodes with similar names/descriptions
    - Description summarization: Condense long descriptions to save space
    - Type resolution: Determine final node type when conflicts arise

    Type Parameters:
        GTNode: The node type (e.g., TEntity) stored in the graph
        GTId: The identifier type for nodes (e.g., str, int)
    """
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_nodes: Iterable[GTNode]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTNode]]]:
        """Execute the node upsert policy.

        Args:
            llm: Language model service for AI-powered operations (e.g., summarization).
            target: The graph storage where nodes will be upserted.
            source_nodes: Iterator of nodes to be added or updated in the graph.

        Returns:
            A tuple containing:
            - The updated graph storage
            - Iterator of (index, node) pairs for all upserted nodes
        """
        raise NotImplementedError


@dataclass
class BaseEdgeUpsertPolicy(BasePolicy, Generic[GTEdge, GTId]):
    """Base policy for edge upsert operations.

    Edge upsert policies control how relationships between nodes are added or updated.
    They handle validation, deduplication, and merging of similar relationships.

    Common strategies include:
    - Validation: Only insert edges if both source and target nodes exist
    - Merging: Combine similar relationships between the same node pair
    - Deduplication: Remove or consolidate redundant edges
    - Chunk tracking: Maintain provenance by tracking which document chunks mention each edge

    Type Parameters:
        GTEdge: The edge type (e.g., TRelation) stored in the graph
        GTId: The identifier type for node references in edges
    """
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_edges: Iterable[GTEdge]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTEdge]]]:
        """Execute the edge upsert policy.

        Args:
            llm: Language model service for AI-powered operations (e.g., similarity detection).
            target: The graph storage where edges will be upserted.
            source_edges: Iterator of edges to be added or updated in the graph.

        Returns:
            A tuple containing:
            - The updated graph storage
            - Iterator of (index, edge) pairs for all upserted edges
        """
        raise NotImplementedError


@dataclass
class BaseGraphUpsertPolicy(BasePolicy, Generic[GTNode, GTEdge, GTId]):
    """Composite policy for upserting both nodes and edges.

    This policy combines separate node and edge upsert policies to provide a unified
    interface for adding complete graph structures (nodes + edges) in a single operation.

    The policy uses composition to delegate node and edge operations to specialized
    sub-policies, ensuring consistent behavior across the graph upsert process.

    Type Parameters:
        GTNode: The node type stored in the graph
        GTEdge: The edge type stored in the graph
        GTId: The identifier type for nodes and edges

    Attributes:
        nodes_upsert_cls: Class of the node upsert policy to instantiate.
        edges_upsert_cls: Class of the edge upsert policy to instantiate.
        _nodes_upsert: Internal instance of the node upsert policy.
        _edges_upsert: Internal instance of the edge upsert policy.
    """
    nodes_upsert_cls: Type[BaseNodeUpsertPolicy[GTNode, GTId]] = field()
    edges_upsert_cls: Type[BaseEdgeUpsertPolicy[GTEdge, GTId]] = field()
    _nodes_upsert: BaseNodeUpsertPolicy[GTNode, GTId] = field(init=False)
    _edges_upsert: BaseEdgeUpsertPolicy[GTEdge, GTId] = field(init=False)

    def __post_init__(self):
        """Initialize the node and edge upsert policy instances.

        Called automatically after dataclass initialization to instantiate
        the sub-policies with the shared configuration.
        """
        self._nodes_upsert = self.nodes_upsert_cls(self.config)
        self._edges_upsert = self.edges_upsert_cls(self.config)

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
        """Execute both node and edge upsert operations.

        Args:
            llm: Language model service for AI-powered operations.
            target: The graph storage to be updated.
            source_nodes: Nodes to be upserted.
            source_edges: Edges to be upserted.

        Returns:
            A tuple containing:
            - The updated graph storage
            - Iterator of (index, node) pairs for all upserted nodes
            - Iterator of (index, edge) pairs for all upserted edges
        """
        raise NotImplementedError


####################################################################################################
# RANKING POLICIES
####################################################################################################
# These policies filter and rank search results based on relevance scores.
# They determine which entities/relationships are most relevant to a query.


class BaseRankingPolicy(BasePolicy):
    """Base policy for ranking and filtering search results.

    Ranking policies process sparse score matrices to select the most relevant results
    from a search operation. They apply various filtering strategies to reduce noise
    and focus on high-quality matches.

    Common strategies include:
    - Threshold filtering: Keep only results above a minimum score
    - Top-K selection: Return only the K highest-scoring results
    - Elbow method: Use score distribution to automatically determine cutoff
    - Confidence-based: Filter based on statistical confidence measures

    The scores are represented as CSR (Compressed Sparse Row) matrices for efficient
    storage and computation when dealing with large graphs.
    """
    def __call__(self, scores: csr_matrix) -> csr_matrix:
        """Apply ranking/filtering to search result scores.

        Args:
            scores: A sparse matrix of shape (1, N) where N is the number of entities.
                Each non-zero entry represents a relevance score for that entity.

        Returns:
            A filtered sparse matrix with the same shape, containing only the
            selected/ranked results. Filtered-out entries are set to zero.

        Raises:
            AssertionError: If scores has batch size != 1 (first dimension != 1).
        """
        assert scores.shape[0] == 1, "Ranking policies only supports batch size of 1"
        return scores
