"""Graph database implementation using the igraph library.

This module provides a graph database storage backend that uses the igraph library
for efficient graph operations. The knowledge graph is stored as an undirected igraph.Graph
where:
- Vertices (nodes) represent entities with arbitrary attributes stored as vertex properties
- Edges represent relationships between entities with attributes stored as edge properties
- The graph structure is persisted to disk using igraph's picklez format (compressed pickle)

The implementation supports:
- Entity CRUD operations (create, read, update, delete)
- Relationship management and querying
- Graph algorithms including Personalized PageRank for node scoring
- Efficient subgraph extraction and neighbor queries
- Sparse matrix representations for entity-relationship mappings
"""

import gzip
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import igraph as ig  # type: ignore
import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEdge, GTId, GTNode, TIndex
from fast_graphrag._utils import csr_from_indices_list, logger

from ._base import BaseGraphStorage


@dataclass
class IGraphStorageConfig(Generic[GTNode, GTEdge]):
    """Configuration for the igraph-based graph storage.

    This configuration class specifies the types and parameters for graph storage operations.

    Attributes:
        node_cls: The class type used to represent graph nodes/entities. Must match the
            generic type GTNode. Node attributes are stored as igraph vertex properties.
        edge_cls: The class type used to represent graph edges/relationships. Must match
            the generic type GTEdge. Edge attributes are stored as igraph edge properties.
        ppr_damping: Damping factor for Personalized PageRank algorithm (default: 0.85).
            This controls the probability of continuing to follow edges during random walks.
            Higher values (closer to 1.0) give more weight to the graph structure, while
            lower values give more weight to the initial node probabilities.
    """
    node_cls: Type[GTNode] = field()
    edge_cls: Type[GTEdge] = field()
    ppr_damping: float = field(default=0.85)


@dataclass
class IGraphStorage(BaseGraphStorage[GTNode, GTEdge, GTId]):
    """Graph database storage implementation using igraph library.

    This class provides a complete graph database implementation for storing and querying
    knowledge graphs. The underlying storage uses an undirected igraph.Graph object where:

    Graph Storage Structure:
        - Nodes (vertices): Represent entities in the knowledge graph
            * Each node has a unique 'name' attribute used as the primary identifier
            * Additional attributes are stored as vertex properties (arbitrary key-value pairs)
            * Internal index (0 to n-1) is used for efficient access
        - Edges: Represent relationships between entities
            * Each edge connects two vertices (undirected graph)
            * Attributes are stored as edge properties
            * Multiple edges can exist between the same pair of vertices
        - Persistence: Graph is serialized to disk using igraph's picklez format
            (compressed pickle) for fast loading/saving

    Key Operations:
        - Entity operations: Create, read, update nodes by name or index
        - Relationship operations: Add, query, delete edges between entities
        - Graph algorithms: Personalized PageRank for node importance scoring
        - Subgraph extraction: Retrieve entity-relationship mappings as sparse matrices
        - Neighbor queries: Check connectivity between entities

    Attributes:
        RESOURCE_NAME: Filename used for persisting the graph ('igraph_data.pklz')
        config: Configuration specifying node/edge types and PageRank parameters
        _graph: The underlying igraph.Graph instance (None until initialized)
    """
    RESOURCE_NAME = "igraph_data.pklz"
    config: IGraphStorageConfig[GTNode, GTEdge] = field()
    _graph: Optional[ig.Graph] = field(init=False, default=None)  # type: ignore

    async def save_graphml(self, path: str) -> None:
        """Export the graph to GraphML format (uncompressed).

        GraphML is an XML-based file format for graphs. This method exports the current
        graph structure and all node/edge attributes to a GraphML file for external analysis
        or visualization in tools like Gephi, Cytoscape, or yEd.

        The export process:
        1. Writes graph to compressed GraphML (.gz format)
        2. Decompresses the file to create uncompressed GraphML
        3. Removes the temporary compressed file

        Args:
            path: File path where the GraphML file should be saved (without extension)

        Note:
            If the graph is not initialized, this method does nothing.
        """
        if self._graph is not None:  # type: ignore
            # First save as compressed GraphML
            ig.Graph.write_graphmlz(self._graph, path + ".gz")  # type: ignore

            # Decompress to create uncompressed GraphML file
            with gzip.open(path + ".gz", 'rb') as f:
                file_content = f.read()
            with open(path, 'wb') as f:
                f.write(file_content)
            # Clean up temporary compressed file
            os.remove(path + ".gz")

    async def node_count(self) -> int:
        """Get the total number of nodes (vertices) in the graph.

        Returns:
            The number of nodes/entities currently stored in the graph.
        """
        return self._graph.vcount()  # type: ignore

    async def edge_count(self) -> int:
        """Get the total number of edges (relationships) in the graph.

        Returns:
            The number of edges/relationships currently stored in the graph.
        """
        return self._graph.ecount()  # type: ignore

    async def get_node(self, node: Union[GTNode, GTId]) -> Union[Tuple[GTNode, TIndex], Tuple[None, None]]:
        """Retrieve a node from the graph by its identifier or instance.

        This method performs node lookup using the 'name' attribute as the primary key.
        Nodes in igraph are identified by their 'name' vertex property, which must be unique.

        Args:
            node: Either a node instance (GTNode) or a node identifier (GTId).
                If a node instance is provided, its 'name' attribute is used for lookup.

        Returns:
            A tuple of (node_instance, index) if the node exists, or (None, None) if not found.
            The index is the internal igraph vertex index (0 to n-1).
        """
        # Extract node identifier from node instance or use the ID directly
        if isinstance(node, self.config.node_cls):
            node_id = node.name
        else:
            node_id = node

        # Search for vertex by name attribute
        try:
            vertex = self._graph.vs.find(name=node_id)  # type: ignore
        except ValueError:
            # Vertex not found
            vertex = None

        # Convert igraph vertex to node instance with its attributes
        return (self.config.node_cls(**vertex.attributes()), vertex.index) if vertex else (None, None)  # type: ignore

    async def get_edges(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[Tuple[GTEdge, TIndex]]:
        """Retrieve all edges between two nodes.

        Since the graph allows multiple edges between the same pair of vertices,
        this method returns all matching edges with their indices.

        Args:
            source_node: The source node identifier (name) or internal index
            target_node: The target node identifier (name) or internal index

        Returns:
            An iterable of tuples containing (edge_instance, edge_index) for each
            edge found between the two nodes. Returns empty list if no edges exist.
        """
        # Get all edge indices between the two nodes
        indices = await self._get_edge_indices(source_node, target_node)
        edges: List[Tuple[GTEdge, TIndex]] = []
        # Retrieve each edge by its index and construct edge instances
        for index in indices:
            edge = await self.get_edge_by_index(index)
            if edge:
                edges.append((edge, index))
        return edges

    async def _get_edge_indices(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[TIndex]:
        """Internal method to get edge indices between two nodes.

        Converts node identifiers to internal indices if needed, then queries
        for all edges connecting the two vertices.

        Args:
            source_node: Source node identifier or index
            target_node: Target node identifier or index

        Returns:
            Generator of edge indices connecting the two nodes
        """
        # Convert node names to internal vertex indices if needed
        if type(source_node) is TIndex:
            source_node = self._graph.vs.find(name=source_node).index  # type: ignore
        if type(target_node) is TIndex:
            target_node = self._graph.vs.find(name=target_node).index  # type: ignore
        # Select all edges between the two vertices
        edges = self._graph.es.select(_source=source_node, _target=target_node)  # type: ignore

        return (edge.index for edge in edges)  # type: ignore

    async def get_node_by_index(self, index: TIndex) -> Union[GTNode, None]:
        """Retrieve a node by its internal vertex index.

        Args:
            index: The internal igraph vertex index (0 to vcount-1)

        Returns:
            Node instance with all attributes if index is valid, None otherwise
        """
        node = self._graph.vs[index] if index < self._graph.vcount() else None  # type: ignore
        return self.config.node_cls(**node.attributes()) if index < self._graph.vcount() else None  # type: ignore

    async def get_edge_by_index(self, index: TIndex) -> Union[GTEdge, None]:
        """Retrieve an edge by its internal edge index.

        This method constructs an edge instance by combining:
        - Source and target node names (from the connected vertices)
        - All edge attributes stored on the igraph edge

        Args:
            index: The internal igraph edge index (0 to ecount-1)

        Returns:
            Edge instance with source/target names and all attributes if index is valid,
            None otherwise
        """
        edge = self._graph.es[index] if index < self._graph.ecount() else None  # type: ignore
        return (
            self.config.edge_cls(
                source=self._graph.vs[edge.source]["name"],  # type: ignore
                target=self._graph.vs[edge.target]["name"],  # type: ignore
                **edge.attributes(),  # type: ignore
            )
            if edge
            else None
        )

    async def upsert_node(self, node: GTNode, node_index: Union[TIndex, None]) -> TIndex:
        """Insert a new node or update an existing one.

        This method supports both insert and update operations:
        - If node_index is None: Creates a new vertex with the node's attributes
        - If node_index is provided: Updates the existing vertex's attributes

        All node attributes are converted to a dictionary and stored as igraph vertex
        properties. The 'name' attribute serves as the unique identifier.

        Args:
            node: The node instance containing all attributes to store
            node_index: The internal vertex index for updates, or None for inserts

        Returns:
            The internal vertex index of the inserted or updated node

        Raises:
            ValueError: If the provided node_index is out of bounds
        """
        if node_index is not None:
            # Update existing node
            if node_index >= self._graph.vcount():  # type: ignore
                logger.error(
                    f"Trying to update node with index {node_index} but graph has only {self._graph.vcount()} nodes."  # type: ignore
                )
                raise ValueError(f"Index {node_index} is out of bounds")
            already_node = self._graph.vs[node_index]  # type: ignore
            # Update all vertex attributes from node dataclass
            already_node.update_attributes(**asdict(node))  # type: ignore

            return already_node.index  # type: ignore
        else:
            # Insert new node - convert node dataclass to dictionary of attributes
            return self._graph.add_vertex(**asdict(node)).index  # type: ignore

    async def upsert_edge(self, edge: GTEdge, edge_index: Union[TIndex, None]) -> TIndex:
        """Insert a new edge or update an existing one.

        This method supports both insert and update operations:
        - If edge_index is None: Creates a new edge between source and target nodes
        - If edge_index is provided: Updates the existing edge's attributes

        Edge attributes are stored as igraph edge properties. The edge connects two
        vertices identified by the source and target node names.

        Args:
            edge: The edge instance containing source, target, and all attributes
            edge_index: The internal edge index for updates, or None for inserts

        Returns:
            The internal edge index of the inserted or updated edge

        Raises:
            ValueError: If the provided edge_index is out of bounds
        """
        if edge_index is not None:
            # Update existing edge
            if edge_index >= self._graph.ecount():  # type: ignore
                logger.error(
                    f"Trying to update edge with index {edge_index} but graph has only {self._graph.ecount()} edges."  # type: ignore
                )
                raise ValueError(f"Index {edge_index} is out of bounds")
            already_edge = self._graph.es[edge_index]  # type: ignore
            # Update edge attributes (excluding source/target which are immutable)
            already_edge.update_attributes(**edge.to_attrs(edge=edge))  # type: ignore

            return already_edge.index  # type: ignore
        else:
            # Insert new edge - includes source, target, and all other attributes
            return self._graph.add_edge(  # type: ignore
                **asdict(edge)
            ).index  # type: ignore

    async def insert_edges(
        self,
        edges: Optional[Iterable[GTEdge]] = None,
        indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None,
        attrs: Optional[Mapping[str, Sequence[Any]]] = None,
    ) -> List[TIndex]:
        """Bulk insert multiple edges efficiently.

        This method provides two modes of bulk edge insertion:
        1. Insert by indices: Provide vertex index pairs and optional attributes
        2. Insert by edge instances: Provide edge objects with all attributes

        Bulk insertion is significantly faster than inserting edges one-by-one as it
        minimizes overhead from igraph's internal data structure updates.

        Args:
            edges: Optional iterable of edge instances to insert. Each edge must have
                source/target node names and attributes.
            indices: Optional iterable of (source_index, target_index) tuples specifying
                edges by internal vertex indices.
            attrs: Optional mapping of attribute names to sequences of values, used when
                inserting by indices. Each sequence must have length equal to number of edges.

        Returns:
            List of internal edge indices for the newly inserted edges, in insertion order

        Note:
            Only one of (edges) or (indices, attrs) should be provided, not both.
            Returns empty list if no edges are provided.
        """
        if indices is not None:
            # Mode 1: Insert edges by vertex index pairs
            assert edges is None, "Cannot provide both indices and edges."

            indices = list(indices)
            if len(indices) == 0:
                return []
            # Bulk add edges with attributes
            self._graph.add_edges(  # type: ignore
                indices,
                attributes=attrs,
            )
            # Return indices of newly added edges (assumes sequential addition)
            # TODO: not sure if this is the best way to get the indices of the new edges
            return list(range(self._graph.ecount() - len(indices), self._graph.ecount()))  # type: ignore
        elif edges is not None:
            # Mode 2: Insert edges from edge instances
            assert indices is None and attrs is None, "Cannot provide both indices and edges."
            edges = list(edges)
            if len(edges) == 0:
                return []
            # Extract (source, target) pairs and convert edge attributes to columnar format
            self._graph.add_edges(  # type: ignore
                ((edge.source, edge.target) for edge in edges),
                attributes=type(edges[0]).to_attrs(edges=edges),
            )
            # Return indices of newly added edges (assumes sequential addition)
            # TODO: not sure if this is the best way to get the indices of the new edges
            return list(range(self._graph.ecount() - len(edges), self._graph.ecount()))  # type: ignore
        else:
            return []

    async def are_neighbours(self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]) -> bool:
        """Check if two nodes are directly connected by an edge.

        This performs a fast connectivity check without retrieving edge details.
        Since the graph is undirected, the order of nodes doesn't matter.

        Args:
            source_node: First node identifier or index
            target_node: Second node identifier or index

        Returns:
            True if at least one edge exists between the nodes, False otherwise
        """
        # get_eid returns -1 if no edge exists, otherwise returns edge ID
        return self._graph.get_eid(source_node, target_node, directed=False, error=False) != -1  # type: ignore

    async def delete_edges_by_index(self, indices: Iterable[TIndex]) -> None:
        """Delete multiple edges by their internal indices.

        This is a bulk deletion operation that removes edges from the graph.
        After deletion, remaining edges may be renumbered.

        Args:
            indices: Iterable of edge indices to delete

        Note:
            Edge indices may change after deletion as igraph renumbers edges to maintain
            a contiguous index space.
        """
        self._graph.delete_edges(indices)  # type: ignore

    async def score_nodes(self, initial_weights: Optional[csr_matrix]) -> csr_matrix:
        """Calculate node importance scores using Personalized PageRank algorithm.

        Personalized PageRank (PPR) is a graph algorithm that measures the importance
        of nodes based on the graph structure and initial node probabilities. It works
        by simulating random walks on the graph:

        Algorithm:
        1. Start at nodes according to initial_weights distribution
        2. At each step, with probability (1-damping): teleport back to initial distribution
        3. With probability damping: follow a random edge to a neighbor
        4. After convergence, return stationary probabilities as node scores

        This is useful for:
        - Ranking entities by relevance to a query (encoded in initial_weights)
        - Finding semantically related entities through graph connectivity
        - Extracting relevant subgraphs for query answering

        Args:
            initial_weights: Optional sparse matrix (1 x n_nodes) specifying the initial
                probability distribution over nodes. If None, uses uniform distribution.
                Higher weights indicate starting points of interest (e.g., query-relevant entities).

        Returns:
            Sparse matrix (1 x n_nodes) containing PageRank scores for each node.
            Higher scores indicate more important/relevant nodes.
            Returns empty matrix if graph is empty.

        Note:
            The damping factor (config.ppr_damping, default 0.85) controls the balance
            between graph structure and initial weights. Higher damping = more emphasis
            on graph structure.
        """
        if self._graph.vcount() == 0:  # type: ignore
            logger.info("Trying to score nodes in an empty graph.")
            return csr_matrix((1, 0))

        # Convert sparse matrix to dense array for igraph, or use None for uniform distribution
        reset_prob = initial_weights.toarray().flatten() if initial_weights is not None else None

        # Run Personalized PageRank algorithm
        # - damping: probability of following an edge (vs. teleporting to reset distribution)
        # - directed: False because this is an undirected graph
        # - reset: personalization vector (initial/restart distribution)
        ppr_scores = self._graph.personalized_pagerank(  # type: ignore
            damping=self.config.ppr_damping, directed=False, reset=reset_prob
        )
        ppr_scores = np.array(ppr_scores, dtype=np.float32)  # type: ignore

        # Return as sparse matrix for efficient storage and computation
        return csr_matrix(
            ppr_scores.reshape(1, -1)  # type: ignore
        )

    async def get_entities_to_relationships_map(self) -> csr_matrix:
        """Extract entity-to-relationship mapping as a sparse matrix.

        This method creates a sparse matrix representation of the graph's incidence
        structure, mapping each entity (node) to its incident relationships (edges).

        Matrix Structure:
        - Shape: (n_nodes, n_edges)
        - Entry [i, j] = 1 if edge j is incident to node i, 0 otherwise
        - Each row represents a node and contains 1s for all its incident edges
        - For undirected graphs, each edge appears in exactly 2 rows (its endpoints)

        This representation is useful for:
        - Subgraph extraction: Identifying which edges connect to specific entities
        - Relationship filtering: Finding edges associated with high-scoring nodes
        - Efficient set operations: Using sparse matrix operations for graph queries

        Returns:
            Sparse CSR matrix (n_nodes × n_edges) encoding the incidence structure.
            Returns empty (0×0) matrix if graph has no nodes.

        Example:
            For a graph with 3 nodes and 2 edges:
            - Edge 0 connects nodes 0-1
            - Edge 1 connects nodes 1-2
            Result matrix:
            [[1, 0],   # Node 0 has edge 0
             [1, 1],   # Node 1 has edges 0 and 1
             [0, 1]]   # Node 2 has edge 1
        """
        if len(self._graph.vs) == 0:  # type: ignore
            return csr_matrix((0, 0))

        # For each vertex, get indices of all incident edges
        # vertex.incident() returns edges connected to that vertex
        return csr_from_indices_list(
            [
                [edge.index for edge in vertex.incident()]  # type: ignore
                for vertex in self._graph.vs  # type: ignore
            ],
            shape=(await self.node_count(), await self.edge_count()),
        )

    async def get_relationships_attrs(self, key: str) -> List[List[Any]]:
        """Retrieve a specific attribute from all edges.

        This method extracts a single attribute across all edges in the graph,
        useful for batch processing of edge properties.

        Args:
            key: The name of the edge attribute to retrieve

        Returns:
            List of lists containing the attribute values for each edge.
            Returns empty list if graph has no edges.

        Note:
            The attribute values are expected to be iterable (e.g., lists).
            Each value is converted to a list in the result.
        """
        if len(self._graph.es) == 0:  # type: ignore
            return []

        lists_of_attrs: List[List[TIndex]] = []
        # Extract the specified attribute from all edges
        for attr in self._graph.es[key]:  # type: ignore
            lists_of_attrs.append(list(attr))  # type: ignore

        return lists_of_attrs

    async def _insert_start(self):
        """Initialize graph for insert/update operations.

        This lifecycle method is called before insert operations begin. It handles:
        1. Loading existing graph from disk if namespace is configured
        2. Creating new empty graph if no persisted data exists
        3. Creating volatile (in-memory only) graph if no namespace configured

        The graph is loaded from a compressed pickle file (picklez format) which
        preserves all vertices, edges, and their attributes.

        Raises:
            InvalidStorageError: If graph file exists but cannot be loaded
        """
        if self.namespace:
            # Namespace configured - attempt to load persisted graph
            graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)

            if graph_file_name:
                # Existing graph file found - load it
                try:
                    self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                    logger.debug(f"Loaded graph storage '{graph_file_name}'.")
                except Exception as e:
                    t = f"Error loading graph from {graph_file_name}: {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                # No existing graph file - create new empty graph
                logger.info(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
                self._graph = ig.Graph(directed=False)
        else:
            # No namespace - create volatile in-memory graph
            self._graph = ig.Graph(directed=False)
            logger.debug("Creating new volatile graphdb storage.")

    async def _insert_done(self):
        """Finalize insert/update operations and persist graph to disk.

        This lifecycle method is called after insert operations complete. If a namespace
        is configured, it saves the current graph state to disk using compressed pickle
        format (picklez). This preserves all vertices, edges, and attributes for future
        sessions.

        For volatile (non-namespaced) storage, this method does nothing as the graph
        exists only in memory.

        Raises:
            InvalidStorageError: If graph cannot be saved to disk
        """
        if self.namespace:
            # Save graph to compressed pickle file
            graph_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
            try:
                ig.Graph.write_picklez(self._graph, graph_file_name)  # type: ignore
            except Exception as e:
                t = f"Error saving graph to {graph_file_name}: {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e

    async def _query_start(self):
        """Initialize graph for read-only query operations.

        This lifecycle method is called before query operations begin. It loads the
        persisted graph from disk. Unlike _insert_start, this method requires a
        namespace to be configured as queries must operate on persisted data.

        The graph is loaded from a compressed pickle file (picklez format) which
        restores all vertices, edges, and their attributes.

        Raises:
            AssertionError: If no namespace is configured
            InvalidStorageError: If graph file exists but cannot be loaded

        Note:
            If no graph file exists, an empty graph is created with a warning.
            This allows queries to proceed without errors, returning empty results.
        """
        assert self.namespace, "Loading a graph requires a namespace."
        graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
        if graph_file_name:
            # Load existing graph from disk
            try:
                self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                logger.debug(f"Loaded graph storage '{graph_file_name}'.")
            except Exception as e:
                t = f"Error loading graph from '{graph_file_name}': {e}"
                logger.error(t)
                raise InvalidStorageError(t) from e
        else:
            # No graph file found - create empty graph to avoid query errors
            logger.warning(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
            self._graph = ig.Graph(directed=False)

    async def _query_done(self):
        """Finalize read-only query operations.

        This lifecycle method is called after query operations complete. For read-only
        queries, no persistence is needed, so this method does nothing.
        """
        pass
