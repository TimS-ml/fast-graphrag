# TODO: implement a non HNSW based vector storage

import pickle
import heapq
import os
import enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import hnswlib
import numpy as np
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTEmbedding, GTId, TScore
from fast_graphrag._utils import logger
from fast_graphrag._storage._base import BaseVectorStorage
from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig


class SimilarityMode(str, enum.Enum):
    """Modes for similarity/distance."""
    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def similarity(
    embedding1: List[float],
    embedding2: List[float],
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity based on different distance metrics."""
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        return -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        return float(np.dot(embedding1, embedding2))
    else:  # Default: cosine similarity
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        return float(product / norm) if norm != 0 else 0.0


def get_top_k_embeddings(
    query_embedding: List[float],
    embeddings: List[float],
    similarity_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List[int]] = None,
    similarity_cutoff: Optional[float] = None,
    similarity_mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> Tuple[List[float], List[int]]:
    """Get top nodes by similarity to the query."""
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    
    # Use provided similarity function or default to similarity with specified mode
    similarity_fn = similarity_fn or (lambda x, y: similarity(x, y, mode=similarity_mode))

    similarity_heap: List[Tuple[float, Any]] = []
    for i, emb in enumerate(embeddings):
        sim_score = similarity_fn(query_embedding, emb)
        if similarity_cutoff is None or sim_score > similarity_cutoff:
            heapq.heappush(similarity_heap, (sim_score, embedding_ids[i]))
            if similarity_top_k and len(similarity_heap) > similarity_top_k:
                heapq.heappop(similarity_heap)
    
    # Sort results by similarity (highest first)
    result_tups = sorted(similarity_heap, key=lambda x: x[0], reverse=True)
    
    result_similarities = [s for s, _ in result_tups]
    result_ids = [n for _, n in result_tups]
    
    return result_similarities, result_ids


def mean_agg(embeddings: List[List[float]]) -> List[float]:
    """Mean aggregation for embeddings."""
    return np.array(embeddings).mean(axis=0)


def get_top_k_mmr_embeddings(
    query_embedding: List[float],
    embeddings: List[float],
    similarity_top_k: Optional[int] = None,
    embedding_ids: Optional[List[int]] = None,
    mmr_threshold: Optional[float] = None,
    similarity_fn: Optional[Callable[[List[float], List[float]], float]] = None,
    lambda_mult: float = 0.5,
    similarity_mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> Tuple[List[float], List[int]]:
    """Get nodes by maximal marginal relevance to the query."""
    if embedding_ids is None:
        embedding_ids = list(range(len(embeddings)))
    
    # Use provided similarity function or default to similarity with specified mode
    similarity_fn = similarity_fn or (lambda x, y: similarity(x, y, mode=similarity_mode))

    # Calculate similarity for all embeddings
    similarity_list = [
        (similarity_fn(query_embedding, emb), id_)
        for emb, id_ in zip(embeddings, embedding_ids)
    ]
    
    if mmr_threshold is not None:
        similarity_list = [(sim, id_) for sim, id_ in similarity_list if sim > mmr_threshold]
    
    similarity_list.sort(reverse=True)
    
    if similarity_top_k is not None:
        similarity_list = similarity_list[:similarity_top_k]
    
    if not similarity_list:
        return [], []
    
    # Initialize result sets
    result_similarities = []
    result_ids = []
    
    # Add the most similar embedding first
    best_similarity, best_id = similarity_list[0]
    result_similarities.append(best_similarity)
    result_ids.append(best_id)
    
    # Remove the first element from candidates
    candidates = similarity_list[1:]
    candidate_embeddings = [
        embeddings[embedding_ids.index(id_)] for _, id_ in candidates
    ]
    
    # Calculate MMR iteratively
    while candidates and (similarity_top_k is None or len(result_ids) < similarity_top_k):
        best_score = -1.0
        best_idx = -1
        
        # Find embedding with highest MMR score
        for i, (similarity, _) in enumerate(candidates):
            # Calculate redundancy (maximum similarity to any element already in the result)
            redundancy = 0.0
            if result_ids:
                redundancy_list = [
                    similarity_fn(candidate_embeddings[i], embeddings[embedding_ids.index(idx)])
                    for idx in result_ids
                ]
                redundancy = max(redundancy_list)
            
            # Calculate MMR score
            mmr_score = lambda_mult * similarity - (1.0 - lambda_mult) * redundancy
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx == -1:
            break
        
        # Add the best candidate to results
        result_similarities.append(candidates[best_idx][0])
        result_ids.append(candidates[best_idx][1])
        
        # Remove the selected candidate
        candidates.pop(best_idx)
        candidate_embeddings.pop(best_idx)
    
    return result_similarities, result_ids


@dataclass
class ImprovedHNSWVectorStorageConfig(HNSWVectorStorageConfig):
    default_query_mode: str = field(default="default")  # 'default', 'mmr'
    lambda_mult: float = field(default=0.5)  # Lambda multiplier for MMR
    mmr_threshold: Optional[float] = field(default=None)
    similarity_mode: SimilarityMode = field(default=SimilarityMode.DEFAULT)


@dataclass
class ImprovedHNSWVectorStorage(HNSWVectorStorage):
    """Improved HNSW Vector Storage with MMR and other advanced retrieval options."""
    
    config: ImprovedHNSWVectorStorageConfig = field(default_factory=ImprovedHNSWVectorStorageConfig)
    
    async def get_knn(
        self, embeddings: Iterable[GTEmbedding], top_k: int, mode: Optional[str] = None 
    ) -> Tuple[Iterable[Iterable[GTId]], List[float]]:
        """Get K nearest neighbors with support for different retrieval methods."""
        if self.size == 0:
            empty_list: List[List[GTId]] = []
            logger.info("Querying knns in empty index.")
            return empty_list, np.array([], dtype=TScore)

        top_k = min(top_k, self.size)
        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)
            
        mode = mode or self.config.default_query_mode
        
        # Convert embeddings to numpy array if it's not already
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
            
        # Retrieve all embeddings for advanced methods
        if mode == 'mmr':
            # For MMR we need to get more candidates first, then post-process
            # Get 2x or at least 50 more candidates than requested
            candidate_count = max(top_k * 2, top_k + 50)
            candidate_count = min(candidate_count, self.size)
            
            # First get candidate embeddings using standard HNSW search
            ids, distances = self._index.knn_query(
                data=embeddings_array, 
                k=candidate_count, 
                num_threads=self.config.num_threads
            )
            
            all_result_similarities = []
            all_result_ids = []
            
            for i, query_emb in enumerate(embeddings_array):
                # Get the candidate embeddings for this query
                candidate_indices = ids[i]
                candidate_embs = np.array([
                    self._index.get_items([idx])[0] for idx in candidate_indices
                ])
                
                # Apply MMR to refine results
                result_similarities, result_indices = get_top_k_mmr_embeddings(
                    query_embedding=query_emb,
                    embeddings=candidate_embs,
                    similarity_top_k=top_k,
                    embedding_ids=candidate_indices.tolist(),
                    mmr_threshold=self.config.mmr_threshold,
                    lambda_mult=self.config.lambda_mult,
                    similarity_mode=self.config.similarity_mode
                )
                
                all_result_similarities.append(result_similarities)
                all_result_ids.append(result_indices)
                
            return all_result_ids, np.array(all_result_similarities, dtype=TScore)
        else:
            # Default HNSW retrieval
            ids, distances = self._index.knn_query(
                data=embeddings_array, 
                k=top_k, 
                num_threads=self.config.num_threads
            )
            return ids, 1.0 - np.array(distances, dtype=TScore) * 0.5
            
    async def score_all(
        self, embeddings: Iterable[GTEmbedding], top_k: int = 1, threshold: Optional[float] = None, 
        mode: Optional[str] = None
    ) -> csr_matrix:
        """Score all embeddings with support for different retrieval methods."""
        if not isinstance(embeddings, List[float]):
            embeddings = np.array(list(embeddings), dtype=np.float32)

        if embeddings.size == 0 or self.size == 0:
            logger.warning(f"No provided embeddings ({embeddings.size}) or empty index ({self.size}).")
            return csr_matrix((0, self.size))

        top_k = min(top_k, self.size)
        if top_k > self.config.ef_search:
            self._index.set_ef(top_k)

        mode = mode or self.config.default_query_mode
        
        if mode == 'mmr':
            # For MMR we need to get more candidates first, then post-process
            # Get 2x or at least 50 more candidates than requested
            candidate_count = max(top_k * 2, top_k + 50)
            candidate_count = min(candidate_count, self.size)
            
            # First get candidate ids and distances using standard HNSW search
            ids, distances = self._index.knn_query(
                data=embeddings, 
                k=candidate_count, 
                num_threads=self.config.num_threads
            )
            
            # Create sparse matrix to hold results
            num_queries = len(embeddings)
            result_matrix = csr_matrix((num_queries, self.size), dtype=np.float32)
            
            for i, query_emb in enumerate(embeddings):
                # Get candidate embeddings for this query
                candidate_indices = ids[i]
                candidate_embs = np.array([
                    self._index.get_items([idx])[0] for idx in candidate_indices
                ])
                
                # Apply MMR to refine results
                similarities, result_indices = get_top_k_mmr_embeddings(
                    query_embedding=query_emb,
                    embeddings=candidate_embs,
                    similarity_top_k=top_k,
                    embedding_ids=candidate_indices.tolist(),
                    mmr_threshold=self.config.mmr_threshold,
                    lambda_mult=self.config.lambda_mult,
                    similarity_mode=self.config.similarity_mode
                )
                
                # Update the sparse matrix
                for j, (similarity, idx) in enumerate(zip(similarities, result_indices)):
                    result_matrix[i, idx] = similarity
            
            return result_matrix
        else:
            # Use custom similarity calculation instead of HNSW's default
            all_embeddings = np.array([
                self._index.get_items([i])[0] for i in range(self.size)
            ])
            
            # Create sparse matrix to hold results
            num_queries = len(embeddings)
            result_matrix = csr_matrix((num_queries, self.size), dtype=np.float32)
            
            for i, query_emb in enumerate(embeddings):
                # Use our custom get_top_k_embeddings function
                similarities, result_indices = get_top_k_embeddings(
                    query_embedding=query_emb,
                    embeddings=all_embeddings,
                    similarity_top_k=top_k,
                    similarity_cutoff=threshold,
                    similarity_mode=self.config.similarity_mode
                )
                
                # Update the sparse matrix
                for similarity, idx in zip(similarities, result_indices):
                    result_matrix[i, idx] = similarity
            
            return result_matrix