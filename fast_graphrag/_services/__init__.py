"""Services module for the fast-graphrag document processing and retrieval pipeline.

This module provides the core services that power the GraphRAG system, implementing
a complete pipeline from raw documents to queryable knowledge graphs.

**Architecture Overview**:

The services module is organized around three primary service types, each responsible
for a distinct phase of the pipeline:

1. **Chunking Services** (_chunk_extraction.py):
   - Split documents into manageable chunks for LLM processing
   - Handle overlap to prevent information loss at boundaries
   - Deduplicate chunks using content-based hashing
   - Preserve document metadata throughout

2. **Information Extraction Services** (_information_extraction.py):
   - Use LLMs to extract entities and relationships from chunks
   - Perform iterative refinement through "gleaning"
   - Validate entity types against schema
   - Merge chunk-level graphs into document graphs
   - Extract entities from user queries for retrieval

3. **State Manager Services** (_state_manager.py):
   - Orchestrate all storage operations (graph, vector, key-value)
   - Implement the insertion pipeline:
     * Chunk deduplication
     * Entity/relationship upsert with conflict resolution
     * Embedding generation and storage
     * Entity deduplication via similarity
     * Mapping matrix construction
   - Implement the query pipeline:
     * Multi-stage retrieval (entities → relationships → chunks)
     * Score propagation through graph structure
     * Ranking and filtering via policies

**Pipeline Flow**:

Documents → Chunking → Information Extraction → State Manager (Insertion)
                                                        ↓
User Query → Entity Extraction → State Manager (Query) → Context Retrieval

**Key Design Principles**:

- **Separation of Concerns**: Each service has a focused responsibility
- **Async/Await**: All services use async operations for maximum throughput
- **Composability**: Base classes define contracts, concrete classes implement
- **Extensibility**: Users can implement custom chunking, extraction, or ranking
- **Error Resilience**: Services handle failures gracefully and log errors

**Exports**:

Base Classes (Abstract):
- BaseChunkingService
- BaseInformationExtractionService
- BaseStateManagerService

Concrete Implementations:
- DefaultChunkingService
- DefaultInformationExtractionService
- DefaultStateManagerService
"""

__all__ = [
    'BaseChunkingService',
    'BaseInformationExtractionService',
    'BaseStateManagerService',
    'DefaultChunkingService',
    'DefaultInformationExtractionService',
    'DefaultStateManagerService'
]

from ._base import BaseChunkingService, BaseInformationExtractionService, BaseStateManagerService
from ._chunk_extraction import DefaultChunkingService
from ._information_extraction import DefaultInformationExtractionService
from ._state_manager import DefaultStateManagerService
