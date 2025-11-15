"""State management and orchestration service for the knowledge graph.

This module implements the central coordinator of the GraphRAG system, managing the
complete lifecycle of knowledge graph operations from insertion to retrieval.

**Insertion Pipeline** (Building the Knowledge Graph):
1. Filter new chunks to avoid redundant processing
2. Upsert entities and relationships from extracted subgraphs
3. Generate embeddings for semantic search
4. Perform entity deduplication via embedding similarity
5. Create "identity" edges linking duplicate entities
6. Store all chunks and build mapping matrices

**Query Pipeline** (Retrieving Relevant Context):
1. Extract entities from user queries
2. Retrieve entities via vector similarity search
3. Rank entities using graph structure (e.g., PageRank)
4. Rank relationships connected to high-scoring entities
5. Rank chunks associated with high-scoring relationships
6. Return scored context (entities, relationships, chunks)

**Mapping Matrices** (Efficient Retrieval):
- Entities → Relationships: Sparse matrix mapping entity indices to relationship indices
- Relationships → Chunks: Sparse matrix mapping relationship indices to chunk indices

These matrices enable efficient propagation of relevance scores through the graph
structure using sparse matrix operations.

**Storage Coordination**:
The state manager coordinates three primary storage backends:
- Graph Storage: Entities and relationships with graph structure
- Vector Storage: Entity embeddings for semantic similarity search
- Key-Value Storage: Text chunks indexed by content hash

All operations are workspace-isolated to support multi-tenancy.
"""

import asyncio
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Awaitable, Dict, Iterable, List, Optional, Tuple, Type, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, vstack
from tqdm import tqdm

from fast_graphrag._llm import BaseLLMService
from fast_graphrag._storage._base import (
    BaseBlobStorage,
    BaseGraphStorage,
    BaseStorage,
)
from fast_graphrag._storage._blob_pickle import PickleBlobStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._types import (
    TChunk,
    TContext,
    TEmbedding,
    TEntity,
    THash,
    TId,
    TIndex,
    TRelation,
    TScore,
)
from fast_graphrag._utils import csr_from_indices_list, extract_sorted_scores, logger

from ._base import BaseStateManagerService


@dataclass
class DefaultStateManagerService(BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]):
    """Production implementation of state management and orchestration.

    This service coordinates all storage operations and implements both the insertion
    and query pipelines for the GraphRAG system.

    **Key Responsibilities**:
    1. Chunk deduplication to avoid redundant processing
    2. Entity/relationship upsert with conflict resolution
    3. Embedding generation and storage for semantic search
    4. Entity deduplication via similarity matching
    5. Mapping matrix construction for efficient retrieval
    6. Multi-stage retrieval and ranking pipeline

    **Storage Architecture**:
    - Primary storages: graph, entity vectors, chunks
    - Auxiliary blob storage: mapping matrices (entities→relationships, relationships→chunks)
    - All storages isolated within a workspace for multi-tenancy

    Attributes:
        blob_storage_cls: Storage class for persisting sparse matrices.
        insert_similarity_score_threshold: Similarity threshold (0-1) for identifying
            duplicate entities during insertion. Entities with similarity >= threshold
            are linked with "identity" edges. Default 0.9 is quite strict.
        query_similarity_score_threshold: Similarity threshold for entity retrieval
            during queries. Lower values retrieve more entities. Default 0.7 balances
            precision and recall.
    """
    blob_storage_cls: Type[BaseBlobStorage[csr_matrix]] = field(default=PickleBlobStorage)
    insert_similarity_score_threshold: float = field(default=0.9)
    query_similarity_score_threshold: Optional[float] = field(default=0.7)

    def __post_init__(self):
        """Initialize storage backends with workspace-isolated namespaces.

        This method:
        1. Validates that a workspace is provided
        2. Assigns namespaced identities to primary storages
        3. Creates blob storages for mapping matrices
        """
        assert self.workspace is not None, "Workspace must be provided."

        # Assign workspace namespaces to primary storages
        self.graph_storage.namespace = self.workspace.make_for("graph")
        self.entity_storage.namespace = self.workspace.make_for("entities")
        self.chunk_storage.namespace = self.workspace.make_for("chunks")

        # Create blob storages for mapping matrices
        # These map entity indices to relationship indices and relationship indices to chunk indices
        self._entities_to_relationships: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self.workspace.make_for("map_e2r"), config=None
        )
        self._relationships_to_chunks: BaseBlobStorage[csr_matrix] = self.blob_storage_cls(
            namespace=self.workspace.make_for("map_r2c"), config=None
        )

    async def get_num_entities(self) -> int:
        """Get the total number of entities in the knowledge graph."""
        return await self.graph_storage.node_count()

    async def get_num_relations(self) -> int:
        """Get the total number of relationships in the knowledge graph."""
        return await self.graph_storage.edge_count()

    async def get_num_chunks(self) -> int:
        """Get the total number of chunks stored."""
        return await self.chunk_storage.size()

    async def filter_new_chunks(self, chunks_per_data: Iterable[Iterable[TChunk]]) -> List[List[TChunk]]:
        """Filter out chunks that already exist in storage to avoid redundant processing.

        This method efficiently identifies which chunks are new by:
        1. Flattening all chunks into a single list
        2. Checking chunk IDs against storage in a single batch operation
        3. Reconstructing the original nested structure with only new chunks

        The document structure is preserved (nested lists) which is important for
        tracking which chunks belong to which documents.

        Args:
            chunks_per_data: Nested iterable where each inner iterable contains
                chunks from a single document.

        Returns:
            Nested list with the same structure, containing only new chunks.
        """
        # Flatten to a single list for efficient batch checking
        flattened_chunks = [chunk for chunks in chunks_per_data for chunk in chunks]
        if len(flattened_chunks) == 0:
            return []

        # Batch check: get a boolean mask indicating which chunks are new
        new_chunks_mask = await self.chunk_storage.mask_new(keys=[c.id for c in flattened_chunks])

        # Reconstruct the nested structure, filtering by the mask
        i = iter(new_chunks_mask)
        new_chunks = [[chunk for chunk in chunks if next(i)] for chunks in chunks_per_data]

        return new_chunks

    async def upsert(
        self,
        llm: BaseLLMService,
        subgraphs: List[asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]]],
        documents: Iterable[Iterable[TChunk]],
        show_progress: bool = True,
    ) -> None:
        """Insert or update entities, relationships, and chunks in the knowledge graph.

        This is the core insertion method that orchestrates the entire upsert pipeline:

        **Pipeline Stages**:
        1. Extract subgraphs from futures (with progress tracking)
        2. Upsert entities and relationships using configured policies
        3. Generate embeddings for all entities
        4. Store embeddings in vector database
        5. Identify duplicate entities via embedding similarity
        6. Create "identity" edges linking duplicate entities
        7. Store all chunks in key-value storage

        **Entity Deduplication**:
        The method identifies duplicate entities by:
        - Computing embeddings for all newly inserted entities
        - Finding similar entities using k-NN search (k=3)
        - Filtering by similarity threshold (insert_similarity_score_threshold)
        - Only linking entities in one direction (to avoid symmetric duplicates)
        - Creating special "is" relationships to link duplicates

        Args:
            llm: Language model service (used by upsert policies).
            subgraphs: Futures resolving to extracted graphs, one per document.
            documents: Original chunks grouped by document.
            show_progress: If True, display progress bars.
        """
        nodes: Iterable[List[TEntity]]
        edges: Iterable[List[TRelation]]

        # STEP 1: Extract subgraphs from futures
        async def _get_graphs(
            fgraph: asyncio.Future[Optional[BaseGraphStorage[TEntity, TRelation, TId]]],
        ) -> Optional[Tuple[List[TEntity], List[TRelation]]]:
            """Extract nodes and edges from a graph storage future."""
            graph = await fgraph
            if graph is None:
                return None

            # Collect all nodes from the graph
            nodes = [t for i in range(await graph.node_count()) if (t := await graph.get_node_by_index(i)) is not None]
            # Collect all edges from the graph
            edges = [t for i in range(await graph.edge_count()) if (t := await graph.get_edge_by_index(i)) is not None]

            return (nodes, edges)

        # Await all extraction futures with progress tracking
        graphs = [
            r
            for graph in tqdm(
                asyncio.as_completed([_get_graphs(fgraph) for fgraph in subgraphs]),
                total=len(subgraphs),
                desc="Extracting data",
                disable=not show_progress,
            )
            if (r := await graph) is not None
        ]

        # Nothing to do if no graphs were successfully extracted
        if len(graphs) == 0:
            return

        # Initialize progress bar for the remaining pipeline stages
        progress_bar = tqdm(total=7, disable=not show_progress, desc="Building...")

        # STEP 2: Upsert nodes and edges into graph storage
        nodes, edges = zip(*graphs)
        progress_bar.set_description("Building... [upserting graphs]")

        # Upsert entities using the configured policy (handles merging duplicates)
        _, upserted_nodes = await self.node_upsert_policy(llm, self.graph_storage, chain(*nodes))
        progress_bar.update(1)
        # Upsert relationships using the configured policy
        _, _ = await self.edge_upsert_policy(llm, self.graph_storage, chain(*edges))
        progress_bar.update(1)

        # STEP 3: Generate embeddings for all entities
        progress_bar.set_description("Building... [computing embeddings]")
        # Convert entities to text representations and encode them
        embeddings = await self.embedding_service.encode(texts=[d.to_str() for _, d in upserted_nodes])
        progress_bar.update(1)

        # STEP 4: Store embeddings in vector database
        await self.entity_storage.upsert(ids=(i for i, _ in upserted_nodes), embeddings=embeddings)
        progress_bar.update(1)

        # STEP 5: Entity deduplication via similarity matching
        # Find similar entities to identify potential duplicates
        # Note: get_knn will likely return the entity itself as the most similar,
        # so we filter it out based on index ordering
        progress_bar.set_description("Building... [entity deduplication]")
        upserted_indices = np.array([i for i, _ in upserted_nodes]).reshape(-1, 1)
        similar_indices, scores = await self.entity_storage.get_knn(embeddings, top_k=3)
        similar_indices = np.array(similar_indices)
        scores = np.array(scores)

        # Filter similar entities based on:
        # 1. Similarity score must be >= threshold
        # 2. Target index must be > source index (avoid symmetric duplicates)
        # This creates a matrix where each row contains indices of similar entities
        similar_indices[
            (scores < self.insert_similarity_score_threshold)
            | (similar_indices <= upserted_indices)  # Only link in one direction
        ] = 0  # 0 is used as a sentinel value for "no similar entity"
        progress_bar.update(1)

        # Example result matrix:
        # | entity_index  | similar_indices[]      |
        # |---------------|------------------------|
        # | 1             | 0, 7, 12, 0, 9         |
        # This means entity 1 is similar to entities 7, 12, and 9

        # STEP 6: Create "identity" edges linking duplicate entities
        progress_bar.set_description("Building... [identity edges]")

        async def _insert_identiy_edges(
            source_index: TIndex, target_indices: npt.NDArray[np.int32]
        ) -> Iterable[Tuple[TIndex, TIndex]]:
            """Create edge pairs for entities that are similar but not already connected."""
            return [
                (source_index, idx)
                for idx in target_indices
                if idx != 0 and not await self.graph_storage.are_neighbours(source_index, idx)
            ]

        # Gather all new identity edges in parallel
        new_edge_indices = list(
            chain(
                *await asyncio.gather(*[_insert_identiy_edges(i, indices) for i, indices in enumerate(similar_indices)])
            )
        )
        # Create attributes for identity edges
        new_edges_attrs: Dict[str, Any] = {
            "description": ["is"] * len(new_edge_indices),  # "is" relationship type
            "chunks": [[]] * len(new_edge_indices),  # No source chunks for identity edges
        }
        await self.graph_storage.insert_edges(indices=new_edge_indices, attrs=new_edges_attrs)
        progress_bar.update(1)

        # STEP 7: Store all chunks in key-value storage
        progress_bar.set_description("Building... [saving chunks]")
        flattened_chunks = [chunk for chunks in documents for chunk in chunks]
        await self.chunk_storage.upsert(keys=[chunk.id for chunk in flattened_chunks], values=flattened_chunks)
        progress_bar.update(1)
        progress_bar.set_description("Building [done]")

    async def get_context(
        self, query: str, entities: Dict[str, List[str]]
    ) -> Optional[TContext[TEntity, TRelation, THash, TChunk]]:
        """Retrieve and rank relevant context from the knowledge graph for a query.

        This method implements the complete query pipeline with multi-stage retrieval and ranking:

        **Stage 1: Entity Retrieval (Vector Similarity)**
        - Generate embeddings for query entities (named and generic) and the query itself
        - Perform k-NN search to find similar entities in the graph
        - Named entities: exact match (top-k=1)
        - Generic entities + query: broader match (top-k=20)
        - Combine scores using max (take highest score per entity)

        **Stage 2: Entity Ranking (Graph Structure)**
        - Apply graph-based scoring (e.g., PageRank, eigenvector centrality)
        - Propagate scores through the graph structure
        - Apply entity ranking policy for final filtering/re-ranking

        **Stage 3: Relationship Ranking (Entity Connections)**
        - Score relationships based on their connected entities
        - Use mapping matrix: entity_scores × entities_to_relationships
        - Apply relationship ranking policy for filtering/re-ranking

        **Stage 4: Chunk Ranking (Relationship Provenance)**
        - Score chunks based on their associated relationships
        - Use mapping matrix: relationship_scores × relationships_to_chunks
        - Apply chunk ranking policy for filtering/re-ranking

        Args:
            query: The user's search query string.
            entities: Entities extracted from query, with "named" and "generic" keys.

        Returns:
            TContext containing scored lists of entities, relationships, and chunks.
            Returns None if no entities in graph or no relevant entities found.
        """
        # Early exit if graph is empty
        if self.entity_storage.size == 0:
            return None

        # STAGE 1: Entity Retrieval via Vector Similarity
        try:
            # Generate embeddings for all query components
            # Format: [named entities] + [generic entities with "[NONE]" prefix] + [query text]
            query_embeddings = await self.embedding_service.encode(
                [f"{n}" for n in entities["named"]] + [f"[NONE] {n}" for n in entities["generic"]] + [query]
            )
            entity_scores: List[csr_matrix] = []

            # Score entities using named entities (strict matching, top-k=1)
            if len(entities["named"]) > 0:
                vdb_entity_scores_by_named_entity = await self._score_entities_by_vectordb(
                    query_embeddings=query_embeddings[: len(entities["named"])],
                    top_k=1,  # Exact match for named entities
                    threshold=self.query_similarity_score_threshold,
                )
                entity_scores.append(vdb_entity_scores_by_named_entity)

            # Score entities using generic entities and query (broader matching, top-k=20)
            vdb_entity_scores_by_generic_entity_and_query = await self._score_entities_by_vectordb(
                query_embeddings=query_embeddings[len(entities["named"]) :],  # Generic + query
                top_k=20,  # Broader search for generic concepts
                threshold=0.5,  # Lower threshold for recall
            )
            entity_scores.append(vdb_entity_scores_by_generic_entity_and_query)

            # Combine scores: take max score per entity across all query embeddings
            vdb_entity_scores = vstack(entity_scores).max(axis=0)

            # Early exit if no entities match
            if isinstance(vdb_entity_scores, int) or vdb_entity_scores.nnz == 0:
                return None
        except Exception as e:
            logger.error(f"Error during information extraction and scoring for query entities {entities}.\n{e}")
            raise e

        # STAGE 2: Entity Ranking via Graph Structure
        try:
            # Apply graph-based scoring (e.g., PageRank) and ranking policy
            graph_entity_scores = self.entity_ranking_policy(
                await self._score_entities_by_graph(entity_scores=vdb_entity_scores)
            )
        except Exception as e:
            logger.error(f"Error during graph scoring for entities. Non-zero elements: {vdb_entity_scores.nnz}.\n{e}")
            raise e

        try:
            # Convert sparse scores to sorted (entity, score) pairs
            indices, scores = extract_sorted_scores(graph_entity_scores)
            relevant_entities: List[Tuple[TEntity, TScore]] = []
            for i, s in zip(indices, scores):
                entity = await self.graph_storage.get_node_by_index(i)
                if entity is not None:
                    relevant_entities.append((entity, s))

            # STAGE 3: Relationship Ranking via Connected Entities
            # Score relationships based on scores of their connected entities
            relation_scores = self.relation_ranking_policy(
                await self._score_relationships_by_entities(entity_scores=graph_entity_scores)
            )

            indices, scores = extract_sorted_scores(relation_scores)
            relevant_relationships: List[Tuple[TRelation, TScore]] = []
            for i, s in zip(indices, scores):
                relationship = await self.graph_storage.get_edge_by_index(i)
                if relationship is not None:
                    relevant_relationships.append((relationship, s))

            # STAGE 4: Chunk Ranking via Associated Relationships
            # Score chunks based on scores of relationships that reference them
            chunk_scores = self.chunk_ranking_policy(
                await self._score_chunks_by_relations(relationships_score=relation_scores)
            )
            indices, scores = extract_sorted_scores(chunk_scores)
            relevant_chunks: List[Tuple[TChunk, TScore]] = []
            for chunk, s in zip(await self.chunk_storage.get_by_index(indices), scores):
                if chunk is not None:
                    relevant_chunks.append((chunk, s))

            return TContext(entities=relevant_entities, relations=relevant_relationships, chunks=relevant_chunks)
        except Exception as e:
            logger.error(f"Error during scoring of chunks and relationships.\n{e}")
            raise e

    async def _get_entities_to_num_docs(self) -> Any:
        """Get the document frequency for each entity (for IDF-like weighting)."""
        raise NotImplementedError

    async def _score_entities_by_vectordb(
        self, query_embeddings: Iterable[TEmbedding], top_k: int = 1, threshold: Optional[float] = None
    ) -> csr_matrix:
        """Score entities using vector similarity search.

        This method performs k-NN search to find entities similar to the query embeddings.
        It returns a sparse matrix where each row corresponds to a query embedding and
        contains similarity scores for the most similar entities.

        The scores are normalized so that each row sums to 1, then the max is taken
        across rows to get the final entity scores.

        Args:
            query_embeddings: Embeddings for query entities/text.
            top_k: Number of nearest neighbors to retrieve per query embedding.
            threshold: Minimum similarity score to include an entity.

        Returns:
            Sparse matrix (1, #entities) with max-normalized similarity scores.
        """
        # TODO: Validate top_k > 1 behavior
        # if top_k != 1:
        #     logger.warning(f"Top-k > 1 is not tested yet. Using top_k={top_k}.")
        if self.node_specificity:
            raise NotImplementedError("Node specificity is not supported yet.")

        # Perform k-NN search: returns sparse matrix (#query_embeddings, #all_entities)
        all_entity_probs_by_query_entity = await self.entity_storage.score_all(
            np.array(query_embeddings), top_k=top_k, threshold=threshold
        )  # (#query_entities, #all_entities)

        # TODO: if top_k > 1, we need to aggregate the scores here
        if all_entity_probs_by_query_entity.shape[1] == 0:
            return all_entity_probs_by_query_entity

        # Normalize scores: each row sums to 1 (probability distribution over entities)
        all_entity_probs_by_query_entity /= all_entity_probs_by_query_entity.sum(axis=1) + 1e-8

        # Take max score across all query embeddings for each entity
        all_entity_weights: csr_matrix = all_entity_probs_by_query_entity.max(axis=0)  # (1, #all_entities)

        # Apply IDF-like weighting based on document frequency (if enabled)
        if self.node_specificity:
            all_entity_weights = all_entity_weights.multiply(1.0 / await self._get_entities_to_num_docs())

        return all_entity_weights

    async def _score_entities_by_graph(self, entity_scores: Optional[csr_matrix]) -> csr_matrix:
        """Score entities using graph structure (e.g., PageRank, eigenvector centrality).

        This method propagates initial entity scores through the graph structure,
        allowing highly connected or central entities to receive boosted scores.

        Args:
            entity_scores: Initial entity scores from vector similarity.

        Returns:
            Sparse matrix (1, #entities) with graph-weighted scores.
        """
        # Delegate to graph storage's scoring algorithm (e.g., PageRank)
        graph_weighted_scores = await self.graph_storage.score_nodes(entity_scores)
        node_scores = csr_matrix(graph_weighted_scores)  # (1, #entities)
        return node_scores

    async def _score_relationships_by_entities(self, entity_scores: csr_matrix) -> csr_matrix:
        """Score relationships based on their connected entities.

        This method propagates entity scores to relationships using the
        entities→relationships mapping matrix. A relationship scores high if
        its connected entities score high.

        Matrix operation:
        relationship_score[i] = sum(entity_score[j] for all entities j connected to relationship i)

        Args:
            entity_scores: Sparse matrix (1, #entities) with entity scores.

        Returns:
            Sparse matrix (1, #relationships) with relationship scores.
        """
        e2r = await self._entities_to_relationships.get()
        if e2r is None:
            logger.warning("No entities to relationships map was loaded.")
            return csr_matrix((1, await self.graph_storage.edge_count()))

        # Sparse matrix multiplication: (1, #entities) × (#entities, #relationships) => (1, #relationships)
        return entity_scores.dot(e2r)

    async def _score_chunks_by_relations(self, relationships_score: csr_matrix) -> csr_matrix:
        """Score chunks based on their associated relationships.

        This method propagates relationship scores to chunks using the
        relationships→chunks mapping matrix. A chunk scores high if it's
        referenced by high-scoring relationships.

        Matrix operation:
        chunk_score[i] = sum(relationship_score[j] for all relationships j that reference chunk i)

        Args:
            relationships_score: Sparse matrix (1, #relationships) with relationship scores.

        Returns:
            Sparse matrix (1, #chunks) with chunk scores.
        """
        c2r = await self._relationships_to_chunks.get()
        if c2r is None:
            logger.warning("No relationships to chunks map was loaded.")
            return csr_matrix((1, await self.chunk_storage.size()))

        # Sparse matrix multiplication: (1, #relationships) × (#relationships, #chunks) => (1, #chunks)
        return relationships_score.dot(c2r)

    ####################################################################################################
    # I/O Management Methods
    #
    # These methods manage the lifecycle of storage operations, ensuring data consistency
    # and enabling checkpoint/restore functionality. They must be called in pairs:
    # - query_start() / query_done() for retrieval operations
    # - insert_start() / insert_done() for insertion operations
    ####################################################################################################

    async def query_start(self):
        """Initialize all storage backends for query/retrieval operations.

        This method:
        1. Restores data from checkpoints if available
        2. Loads indices and mapping matrices into memory
        3. Sets all storages to "in progress" state to prevent concurrent modifications

        Must be called before any get_context() operations.
        """
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        # Define initialization function for checkpoint restoration
        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.query_start())
            return asyncio.gather(*tasks)

        # Execute with checkpoint support (restores from checkpoint if available)
        await cast(Workspace, self.workspace).with_checkpoints(_fn)

        # Mark all storages as in-progress to prevent concurrent writes
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def query_done(self):
        """Finalize and cleanup after query/retrieval operations.

        This method:
        1. Commits any read transactions
        2. Releases cached data from memory
        3. Closes connections
        4. Marks storages as not in-progress

        Should be called after all get_context() operations are complete.
        """
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        # Finalize all storages in parallel
        for storage_inst in storages:
            tasks.append(storage_inst.query_done())
        await asyncio.gather(*tasks)

        # Mark all storages as not in-progress
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def insert_start(self):
        """Initialize all storage backends for insertion operations.

        This method:
        1. Opens database connections or file handles
        2. Loads existing indices for deduplication
        3. Sets up transactional contexts
        4. Creates checkpoints for rollback capability
        5. Sets all storages to "in progress" state

        Must be called before any upsert() operations.
        """
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]

        # Define initialization function for checkpoint support
        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.insert_start())
            return asyncio.gather(*tasks)

        # Execute with checkpoint support (creates checkpoint for rollback)
        await cast(Workspace, self.workspace).with_checkpoints(_fn)

        # Mark all storages as in-progress to prevent concurrent operations
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def insert_done(self):
        """Finalize and commit all insertion operations.

        This critical method:
        1. Builds the entities→relationships mapping matrix from the graph
        2. Builds the relationships→chunks mapping matrix from relationship attributes
        3. Persists both mapping matrices to blob storage
        4. Commits all storage transactions
        5. Flushes in-memory indices to disk
        6. Marks storages as not in-progress

        The mapping matrices are essential for the query pipeline's score propagation.

        Must be called after all upsert() operations are complete.
        """
        # Build and persist the entities→relationships mapping matrix
        await self._entities_to_relationships.set(await self.graph_storage.get_entities_to_relationships_map())

        # Build the relationships→chunks mapping matrix
        # Get chunk IDs from relationship attributes
        raw_relationships_to_chunks = await self.graph_storage.get_relationships_attrs(key="chunks")

        # Convert chunk IDs to chunk indices for efficient matrix operations
        raw_relationships_to_chunks = [
            [i for i in await self.chunk_storage.get_index(chunk_ids) if i is not None]
            for chunk_ids in raw_relationships_to_chunks
        ]

        # Create sparse matrix: rows=relationships, cols=chunks, values=1 if associated
        await self._relationships_to_chunks.set(
            csr_from_indices_list(
                raw_relationships_to_chunks, shape=(len(raw_relationships_to_chunks), await self.chunk_storage.size())
            )
        )

        # Finalize all storages in parallel
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [
            self.graph_storage,
            self.entity_storage,
            self.chunk_storage,
            self._relationships_to_chunks,
            self._entities_to_relationships,
        ]
        for storage_inst in storages:
            tasks.append(storage_inst.insert_done())
        await asyncio.gather(*tasks)

        # Mark all storages as not in-progress
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def save_graphml(self, output_path: str) -> None:
        """Export the knowledge graph to GraphML format for visualization.

        GraphML is an XML-based format supported by tools like Gephi, Cytoscape,
        and NetworkX. This enables visual exploration of the knowledge graph.

        Args:
            output_path: Filesystem path where the GraphML file should be written.
        """
        await self.graph_storage.save_graphml(output_path)
        logger.info(f"Graph saved to '{output_path}'.")
