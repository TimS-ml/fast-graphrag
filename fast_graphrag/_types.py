"""Type definitions for the Fast GraphRAG system.

This module contains all core type definitions used throughout the GraphRAG (Graph-based
Retrieval-Augmented Generation) system. It provides:

1. Generic base classes for serializable types (nodes, edges, chunks)
2. Type variables for generic programming across different data structures
3. Concrete type implementations for documents, entities, relations, and graphs
4. Query response types that combine retrieved context with generated answers

The type system is designed to be flexible and extensible, using generic type variables
to allow different storage backends and data representations while maintaining type safety.

Key Components:
    - TSerializable: Base class for types that can be serialized to dictionaries
    - BTNode, BTEdge, BTChunk: Base types for graph components
    - TEntity, TRelation, TGraph: Concrete types for knowledge graph elements
    - TContext, TQueryResponse: Types for query processing and responses
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, List, Optional, Tuple, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from ._models import BaseModelAlias, dump_to_csv, dump_to_reference_list

####################################################################################################
# GENERICS
####################################################################################################


@dataclass
class TSerializable:
  """Base class for types that can be serialized to dictionaries.

  This class provides common serialization functionality for all GraphRAG data types.
  Subclasses can specify which fields should be included in context via F_TO_CONTEXT.

  Attributes:
      F_TO_CONTEXT: Class variable listing field names to include when serializing
          for context generation (e.g., for LLM prompts). Empty list means no fields
          are specifically designated for context.
  """

  F_TO_CONTEXT: ClassVar[List[str]] = []

  @classmethod
  def to_dict(
    cls,
    obj: Optional["TSerializable"] = None,
    objs: Optional[Iterable["TSerializable"]] = None,
    include_fields: Optional[List[str]] = None,
  ) -> Dict[str, Any]:
    """Convert one or more serializable objects to a dictionary representation.

    This method can serialize either a single object or multiple objects. When serializing
    multiple objects, the result groups values by field name rather than by object.

    Args:
        obj: A single object to serialize. Mutually exclusive with objs.
        objs: An iterable of objects to serialize. Mutually exclusive with obj.
        include_fields: List of field names to include. If None, includes all dataclass fields.

    Returns:
        Dictionary mapping field names to values (single object) or lists of values
        (multiple objects). Returns empty dict if neither obj nor objs is provided.

    Raises:
        AssertionError: If both obj and objs are provided.

    Examples:
        Single object: {"name": "Alice", "age": 30}
        Multiple objects: {"name": ["Alice", "Bob"], "age": [30, 25]}
    """
    # Compute the fields to include
    if include_fields is None:
      include_fields = [f.name for f in fields(cls)]
    if obj is not None:
      assert objs is None, "Either edge or edges should be provided, not both"
      return {f: getattr(obj, f) for f in include_fields}
    elif objs is not None:
      return {f: [getattr(o, f) for o in objs] for f in include_fields}
    return {}


# Generic type variables for blob storage systems
# GTBlob: Generic type for binary large objects (blobs) stored in blob storage backends
GTBlob = TypeVar("GTBlob")

# Generic type variables for key-value storage systems
# GTKey: Generic type for keys in key-value stores (e.g., strings, integers)
GTKey = TypeVar("GTKey")
# GTValue: Generic type for values in key-value stores (can be any serializable type)
GTValue = TypeVar("GTValue")

# Generic type variables for vector database systems
# GTEmbedding: Generic type for vector embeddings (typically numpy arrays of floats)
GTEmbedding = TypeVar("GTEmbedding")
# GTHash: Generic type for hash values used to identify and deduplicate content
GTHash = TypeVar("GTHash")

# Generic type variables for graph storage systems
# GTId: Generic type for node/entity identifiers in the knowledge graph
GTId = TypeVar("GTId")


@dataclass
class BTNode(TSerializable):
  """Base type for graph nodes (entities).

  Represents a node in the knowledge graph with a unique name/identifier.
  All concrete node types must inherit from this class.

  Attributes:
      name: The unique identifier or name of the node. Type is flexible (Any)
          to support different identifier schemes.
  """

  name: Any


# GTNode: Generic type variable for node types, must be BTNode or a subclass
# This allows functions and classes to be generic over different node implementations
# while maintaining type safety (e.g., TEntity is a valid GTNode type)
GTNode = TypeVar("GTNode", bound=BTNode)


@dataclass
class BTEdge(TSerializable):
  """Base type for graph edges (relationships).

  Represents an edge in the knowledge graph connecting two nodes.
  All concrete edge types must inherit from this class.

  Attributes:
      source: The identifier of the source node. Type is flexible (Any) to support
          different identifier schemes.
      target: The identifier of the target node. Type is flexible (Any) to support
          different identifier schemes.
  """

  source: Any
  target: Any

  @staticmethod
  def to_attrs(edge: Optional[Any] = None, edges: Optional[Iterable[Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Convert edge(s) to a dictionary of attributes.

    This method must be implemented by subclasses to define how edge attributes
    are serialized for storage or transmission.

    Args:
        edge: A single edge to convert. Mutually exclusive with edges.
        edges: An iterable of edges to convert. Mutually exclusive with edge.
        **kwargs: Additional keyword arguments for customizing serialization.

    Returns:
        Dictionary mapping attribute names to values (or lists of values).

    Raises:
        NotImplementedError: This base implementation must be overridden.
    """
    raise NotImplementedError


# GTEdge: Generic type variable for edge types, must be BTEdge or a subclass
# This allows functions and classes to be generic over different edge implementations
# while maintaining type safety (e.g., TRelation is a valid GTEdge type)
GTEdge = TypeVar("GTEdge", bound=BTEdge)


@dataclass
class BTChunk(TSerializable):
  """Base type for text chunks.

  Represents a chunk of text extracted from a document. Chunks are the atomic
  units of text that are embedded and stored in the vector database.

  Attributes:
      id: The unique identifier for this chunk. Type is flexible (Any) to support
          different identifier schemes (typically hash values).
  """

  id: Any


# GTChunk: Generic type variable for chunk types, must be BTChunk or a subclass
# This allows functions and classes to be generic over different chunk implementations
# while maintaining type safety (e.g., TChunk is a valid GTChunk type)
GTChunk = TypeVar("GTChunk", bound=BTChunk)


####################################################################################################
# TYPES
####################################################################################################

# Concrete type aliases for the Fast GraphRAG implementation
# These type aliases define the specific types used throughout the system

# TEmbeddingType: The numeric type for individual embedding components (32-bit float)
# Using float32 balances precision with memory efficiency for vector embeddings
TEmbeddingType: TypeAlias = np.float32

# TEmbedding: Type for complete embedding vectors (numpy arrays of float32 values)
# Vector embeddings represent semantic meaning of text chunks for similarity search
TEmbedding: TypeAlias = npt.NDArray[TEmbeddingType]

# THash: Type for hash values used to identify and deduplicate chunks (64-bit signed int)
# Large integer type ensures low collision probability when hashing content
THash: TypeAlias = np.int64

# TScore: Type for similarity/relevance scores (32-bit float)
# Used for ranking search results and measuring entity/relation relevance
TScore: TypeAlias = np.float32

# TIndex: Type for array/list indices (standard Python int)
# Used for indexing into arrays and maintaining ordering
TIndex: TypeAlias = int

# TId: Type for entity/node identifiers (string)
# String identifiers allow human-readable entity names
TId: TypeAlias = str


@dataclass
class TDocument:
  """Represents a source document to be processed by the GraphRAG system.

  A document is the top-level container for input data. It contains the raw text
  content and associated metadata. Documents are split into chunks for processing.

  Attributes:
      data: The raw text content of the document.
      metadata: Optional metadata dictionary containing document-level information
          such as source, author, date, etc. Defaults to empty dict.
  """

  data: str = field()
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TChunk(BTChunk):
  """Represents a chunk of text extracted from a document.

  Chunks are the atomic units of text processing in GraphRAG. Each chunk is:
  - Assigned a unique hash-based identifier
  - Embedded as a vector for semantic search
  - Used as evidence for extracted entities and relationships

  The chunk inherits from BTChunk and implements the concrete type with specific
  field types (THash for id, str for content).

  Attributes:
      F_TO_CONTEXT: Class variable specifying that 'content' and 'metadata' fields
          should be included when this chunk is serialized for LLM context.
      id: Unique hash identifier for this chunk (64-bit integer).
      content: The actual text content of the chunk.
      metadata: Metadata inherited from the parent document (e.g., source, page number).
          Defaults to empty dict.
  """

  F_TO_CONTEXT = ["content", "metadata"]

  id: THash = field()
  content: str = field()
  metadata: Dict[str, Any] = field(default_factory=dict)

  def __str__(self) -> str:
    """Return the string representation of the chunk (its content).

    Returns:
        The text content of the chunk.
    """
    return self.content


# Graph types
@dataclass
class TEntity(BaseModelAlias, BTNode):
  """Represents an entity (node) in the knowledge graph.

  An entity is a named object or concept extracted from text. Entities form the nodes
  of the knowledge graph and are connected by relationships. Each entity has:
  - A unique name (normalized to uppercase)
  - A type/category (e.g., PERSON, ORGANIZATION, LOCATION)
  - A description of its properties or role

  This class inherits from both BaseModelAlias (for Pydantic model integration) and
  BTNode (marking it as a graph node type).

  Attributes:
      F_TO_CONTEXT: Class variable specifying that 'name' and 'description' fields
          should be included when serializing for LLM context.
      name: The unique identifier/name of the entity (normalized to uppercase).
      type: The category or type of entity (normalized to uppercase).
      description: Natural language description of the entity's properties or significance.
  """

  F_TO_CONTEXT = ["name", "description"]

  name: str = field()
  type: str = field()
  description: str = field()

  def to_str(self) -> str:
    """Convert entity to a human-readable string representation.

    Returns:
        Formatted string with entity type, name, and optionally description.
        Format: "[TYPE] NAME\n[DESCRIPTION] description" (if description exists)
                "[TYPE] NAME" (if no description)
    """
    s = f"[{self.type}] {self.name}"
    if len(self.description):
      s += f"\n[DESCRIPTION] {self.description}"
    return s

  class Model(BaseModelAlias.Model, alias="Entity"):
    """Pydantic model for entity extraction and structured output.

    This nested class defines the schema used when extracting entities from text
    using language models with structured output. It enforces validation and
    provides conversion to the TEntity dataclass.

    The model uses Pydantic's Field with descriptions that guide the LLM during
    entity extraction.

    Attributes:
        name: Name of the entity (auto-normalized to uppercase).
        type: Type/category of the entity (auto-normalized to uppercase).
        desc: Description of the entity (maps to 'description' in dataclass).
    """

    name: str = Field(..., description="Name of the entity", json_schema_extra={"example": ""})
    type: str = Field(..., description="Type of the entity", json_schema_extra={"example": ""})
    desc: str = Field(..., description="Description of the entity", json_schema_extra={"example": ""})

    @staticmethod
    def to_dataclass(pydantic: "TEntity.Model") -> "TEntity":
      """Convert Pydantic model instance to TEntity dataclass.

      Args:
          pydantic: The Pydantic model instance to convert.

      Returns:
          TEntity dataclass with fields populated from the Pydantic model.
      """
      return TEntity(name=pydantic.name, type=pydantic.type, description=pydantic.desc)

    @field_validator("name", mode="before")
    @classmethod
    def uppercase_name(cls, value: str):
      """Normalize entity name to uppercase for consistency.

      Args:
          value: The entity name to normalize.

      Returns:
          Uppercased entity name, or original value if empty/None.
      """
      return value.upper() if value else value

    @field_validator("type", mode="before")
    @classmethod
    def uppercase_type(cls, value: str):
      """Normalize entity type to uppercase for consistency.

      Args:
          value: The entity type to normalize.

      Returns:
          Uppercased entity type, or original value if empty/None.
      """
      return value.upper() if value else value


@dataclass
class TRelation(BaseModelAlias, BTEdge):
  """Represents a relationship (edge) in the knowledge graph.

  A relation connects two entities and describes how they are related. Relations form
  the edges of the knowledge graph. Each relation has:
  - A source entity (normalized to uppercase)
  - A target entity (normalized to uppercase)
  - A description of the relationship
  - Optional references to the chunks that support this relation

  This class inherits from both BaseModelAlias (for Pydantic model integration) and
  BTEdge (marking it as a graph edge type).

  Attributes:
      F_TO_CONTEXT: Class variable specifying that 'source', 'target', and 'description'
          fields should be included when serializing for LLM context.
      source: Name of the source entity (normalized to uppercase).
      target: Name of the target entity (normalized to uppercase).
      description: Natural language description of how source and target are related.
      chunks: Optional list of chunk hash IDs that provide evidence for this relation.
          Used for provenance tracking.
  """

  F_TO_CONTEXT = ["source", "target", "description"]

  source: str = field()
  target: str = field()
  description: str = field()
  chunks: List[THash] | None = field(default=None)

  @staticmethod
  def to_attrs(
    edge: Optional["TRelation"] = None,
    edges: Optional[Iterable["TRelation"]] = None,
    include_source_target: bool = False,
    **_,
  ) -> Dict[str, Any]:
    """Convert relation(s) to a dictionary of attributes.

    This method serializes relation attributes for storage or transmission. It can
    optionally include or exclude source/target fields (useful when these are
    already represented in the graph structure).

    Args:
        edge: A single relation to convert. Mutually exclusive with edges.
        edges: An iterable of relations to convert. Mutually exclusive with edge.
        include_source_target: If True, include source and target fields in output.
            If False (default), only include description and chunks.
        **_: Additional keyword arguments (ignored).

    Returns:
        Dictionary mapping attribute names to values (single edge) or lists of
        values (multiple edges). Returns empty dict if neither edge nor edges provided.

    Raises:
        AssertionError: If both edge and edges are provided.
    """
    if edge is not None:
      assert edges is None, "Either edge or edges should be provided, not both"
      return {
        "description": edge.description,
        "chunks": edge.chunks,
        **(
          {
            "source": edge.source,
            "target": edge.target,
          }
          if include_source_target
          else {}
        ),
      }
    elif edges is not None:
      return {
        "description": [e.description for e in edges],
        "chunks": [e.chunks for e in edges],
        **(
          {
            "source": [e.source for e in edges],
            "target": [e.target for e in edges],
          }
          if include_source_target
          else {}
        ),
      }
    else:
      return {}

  class Model(BaseModelAlias.Model, alias="Relationship"):
    """Pydantic model for relationship extraction and structured output.

    This nested class defines the schema used when extracting relationships from text
    using language models with structured output. It enforces validation and provides
    conversion to the TRelation dataclass.

    The model uses Pydantic's Field with descriptions that guide the LLM during
    relationship extraction.

    Attributes:
        source: Name of the source entity (auto-normalized to uppercase).
        target: Name of the target entity (auto-normalized to uppercase).
        desc: Description of the relationship (maps to 'description' in dataclass).
            Describes how the source and target entities are related.
    """

    source: str = Field(..., description="Name of the source entity", json_schema_extra={"example": ""})
    target: str = Field(..., description="Name of the target entity", json_schema_extra={"example": ""})
    # Alternative description: "Explanation of why the source entity and the target entity are related to each other"
    desc: str = Field(
      ...,
      description="Description of the relationship between the source and target entity",
      json_schema_extra={"example": ""},
    )

    @staticmethod
    def to_dataclass(pydantic: "TRelation.Model") -> "TRelation":
      """Convert Pydantic model instance to TRelation dataclass.

      Args:
          pydantic: The Pydantic model instance to convert.

      Returns:
          TRelation dataclass with fields populated from the Pydantic model.
          Note: chunks field is set to None as it's populated later from provenance.
      """
      return TRelation(source=pydantic.source, target=pydantic.target, description=pydantic.desc)

    @field_validator("source", mode="before")
    @classmethod
    def uppercase_source(cls, value: str):
      """Normalize source entity name to uppercase for consistency.

      Args:
          value: The source entity name to normalize.

      Returns:
          Uppercased source entity name, or original value if empty/None.
      """
      return value.upper() if value else value

    @field_validator("target", mode="before")
    @classmethod
    def uppercase_target(cls, value: str):
      """Normalize target entity name to uppercase for consistency.

      Args:
          value: The target entity name to normalize.

      Returns:
          Uppercased target entity name, or original value if empty/None.
      """
      return value.upper() if value else value


@dataclass
class TGraph(BaseModelAlias):
  """Represents a complete knowledge graph extracted from text.

  A graph is a collection of entities and the relationships between them. This type
  encapsulates the complete output of the entity/relationship extraction process
  for a chunk or document.

  Attributes:
      entities: List of all entities (nodes) extracted from the text.
      relationships: List of all relationships (edges) connecting the entities.
  """

  entities: List[TEntity] = field()
  relationships: List[TRelation] = field()

  class Model(BaseModelAlias.Model, alias="Graph"):
    """Pydantic model for knowledge graph extraction with structured output.

    This nested class defines the schema used when extracting complete knowledge graphs
    from text using language models. The model prompts for entities, their direct
    relationships, and any additional relationships that may have been missed initially.

    The two-phase relationship extraction (relationships + other_relationships) helps
    ensure completeness by explicitly prompting the LLM to reconsider connections.

    Attributes:
        entities: List of entity models extracted from the text.
        relationships: Primary list of relationship models between entities.
        other_relationships: Additional relationships that may have been missed in
            the first pass, particularly those involving minor or generic entities.
    """

    entities: List[TEntity.Model] = Field(description="List of extracted entities", json_schema_extra={"example": []})
    relationships: List[TRelation.Model] = Field(
      description="Relationships between the entities", json_schema_extra={"example": []}
    )
    other_relationships: List[TRelation.Model] = Field(
      description=(
        "Other relationships between the extracted entities previously missed"
        "(likely involving minor/generic entities)"
      ),
      json_schema_extra={"example": []},
    )

    @staticmethod
    def to_dataclass(pydantic: "TGraph.Model") -> "TGraph":
      """Convert Pydantic model instance to TGraph dataclass.

      This method combines both 'relationships' and 'other_relationships' into a
      single relationships list in the output dataclass.

      Args:
          pydantic: The Pydantic model instance to convert.

      Returns:
          TGraph dataclass with entities and combined relationships from both sources.
      """
      return TGraph(
        entities=[p.to_dataclass(p) for p in pydantic.entities],
        relationships=[p.to_dataclass(p) for p in pydantic.relationships]
        + [p.to_dataclass(p) for p in pydantic.other_relationships],
      )


@dataclass
class TContext(Generic[GTNode, GTEdge, GTHash, GTChunk]):
  """Represents the retrieved context used to generate a query response.

  A context combines three types of information retrieved from the knowledge graph:
  1. Entities: Relevant nodes with their descriptions
  2. Relations: Relevant edges showing how entities are connected
  3. Chunks: Original text chunks providing evidence

  Each item is paired with a relevance score indicating how closely it matches the query.

  This is a generic class parameterized by the specific types used for nodes, edges,
  hashes, and chunks, allowing it to work with different implementations.

  Type Parameters:
      GTNode: The type of graph nodes (typically TEntity).
      GTEdge: The type of graph edges (typically TRelation).
      GTHash: The type of hash identifiers (typically np.int64).
      GTChunk: The type of text chunks (typically TChunk).

  Attributes:
      entities: List of (entity, score) tuples, ordered by relevance.
      relations: List of (relation, score) tuples, ordered by relevance.
      chunks: List of (chunk, score) tuples, ordered by relevance.
  """

  entities: List[Tuple[GTNode, TScore]] = field()
  relations: List[Tuple[GTEdge, TScore]] = field()
  chunks: List[Tuple[GTChunk, TScore]] = field()

  def truncate(self, max_chars: Dict[str, int], output_context_str: bool = False) -> str:
    """Generate a tabular representation of the context, truncated to fit character limits.

    This method converts the context to CSV tables and truncates them to respect
    character budgets. It uses a smart allocation algorithm that:
    1. Assigns each table (entities/relations/chunks) a character budget
    2. Includes as many rows as possible within each budget
    3. Redistributes unused characters from one table to others

    The truncation modifies this TContext instance in-place, removing items that
    don't fit within the limits.

    Args:
        max_chars: Dictionary mapping table names to character limits.
            Keys: 'entities', 'relations', 'chunks'
            Values: Maximum characters for each table. Use -1 for unlimited.
        output_context_str: If True, generate and return a formatted markdown string
            with the context. If False, return empty string.

    Returns:
        If output_context_str is True, returns a markdown-formatted string with
        CSV tables for entities and relations, and a numbered list for chunks.
        If output_context_str is False, returns empty string.

    Side Effects:
        Truncates the entities, relations, and chunks lists in-place based on
        the character limits.
    """
    # Convert context items to CSV format for compact representation
    csv_tables: Dict[str, List[str]] = {
      "entities": dump_to_csv([e for e, _ in self.entities], ["name", "description"], with_header=True),
      "relations": dump_to_csv([r for r, _ in self.relations], ["source", "target", "description"], with_header=True),
      "chunks": dump_to_reference_list([str(c) for c, _ in self.chunks]),
    }
    # Pre-compute the length of each row for efficient allocation
    csv_tables_row_length = {k: [len(row) for row in table] for k, table in csv_tables.items()}

    # Truncate each CSV table to fit within the character budget
    # Track how many rows from each table are included
    included_up_to = {key: 0 for key in ["entities", "relations", "chunks"]}
    chars_remainder = 0  # Pool of unused characters that can be redistributed
    while True:
      last_char_remainder = chars_remainder
      # Iteratively try to include more rows, using remainder characters when available
      for table in csv_tables:
        for i in range(included_up_to[table], len(csv_tables_row_length[table])):
          length = csv_tables_row_length[table][i] + 1  # +1 for the newline character
          # Try to use remainder characters first (but not for unlimited tables marked with -1)
          if (length <= chars_remainder) and (max_chars[table] >= 0):
            included_up_to[table] += 1
            chars_remainder -= length
            break
          # Otherwise use the table's assigned character budget
          elif length <= max_chars[table]:
            included_up_to[table] += 1
            max_chars[table] -= length
          else:
            # Row doesn't fit, stop adding rows from this table
            break

        # Add any leftover budget from this table to the remainder pool for others to use
        if max_chars[table] >= 0:  # Don't pool unlimited budgets (marked with -1)
          chars_remainder += max_chars[table]
          max_chars[table] = 0

      # Stop when no more rows can be added (remainder unchanged)
      if chars_remainder == last_char_remainder:
        break

    # Apply truncation to the actual context lists (in-place modification)
    self.entities = self.entities[: included_up_to["entities"]]
    self.relations = self.relations[: included_up_to["relations"]]
    self.chunks = self.chunks[: included_up_to["chunks"]]

    # Generate the formatted context string for LLM consumption (if requested)
    context: List[str] = []
    if output_context_str:
      # Format entities as CSV table in markdown code block
      if len(self.entities):
        context.extend(
          [
            "\n## Entities",
            "```csv",
            *csv_tables["entities"][: included_up_to["entities"]],
            "```",
          ]
        )
      else:
        context.append("\n## Entities: Not provided\n")

      # Format relationships as CSV table in markdown code block
      if len(self.relations):
        context.extend(
          [
            "\n## Relationships",
            "```csv",
            *csv_tables["relations"][: included_up_to["relations"]],
            "```",
          ]
        )
      else:
        context.append("\n## Relationships: Not provided\n")

      # Format chunks as numbered list of text references
      if len(self.chunks):
        context.extend(["\n## Sources\n", *csv_tables["chunks"][: included_up_to["chunks"]], ""])
      else:
        context.append("\n## Sources: Not provided\n")
    return "\n".join(context)


@dataclass
class TQueryResponse(Generic[GTNode, GTEdge, GTHash, GTChunk]):
  """Represents a complete response to a user query.

  A query response combines the generated answer (response text) with the supporting
  context retrieved from the knowledge graph. This allows users to verify the answer
  against the source material.

  This is a generic class parameterized by the specific types used for nodes, edges,
  hashes, and chunks, matching the generic parameters of TContext.

  Type Parameters:
      GTNode: The type of graph nodes (typically TEntity).
      GTEdge: The type of graph edges (typically TRelation).
      GTHash: The type of hash identifiers (typically np.int64).
      GTChunk: The type of text chunks (typically TChunk).

  Attributes:
      response: The generated answer text from the LLM.
      context: The retrieved context (entities, relations, chunks) used to generate
          the response, with relevance scores.
  """

  response: str
  context: TContext[GTNode, GTEdge, GTHash, GTChunk]

  def to_dict(self) -> Dict[str, Any]:
    """Convert the query response to a dictionary representation.

    Serializes both the response text and the supporting context. Each context item
    (entity/relation/chunk) is converted to a dictionary containing only the fields
    designated for context (via F_TO_CONTEXT), paired with its score.

    Returns:
        Dictionary with 'response' (str) and 'context' (dict) keys. The context
        contains 'entities', 'relations', and 'chunks' lists, where each item is
        a tuple of (field_dict, score).
    """
    return {
      "response": self.response,
      "context": {
        "entities": [(e.to_dict(e, include_fields=e.F_TO_CONTEXT), float(s)) for e, s in self.context.entities],
        "relations": [(r.to_dict(r, include_fields=r.F_TO_CONTEXT), float(s)) for r, s in self.context.relations],
        "chunks": [(c.to_dict(c, include_fields=c.F_TO_CONTEXT), float(s)) for c, s in self.context.chunks],
      },
    }

  # All the machinery to format references
  ####################################################################################################

  @dataclass
  class _Chunk:
    """Internal helper for tracking chunks within documents for reference formatting.

    Attributes:
        id: Unique identifier for this chunk (hash value).
        content: The text content of the chunk.
        index: Optional display index assigned when chunk is referenced (1-based).
            None until the chunk is first referenced.
    """

    id: int = field()
    content: str = field()
    index: Optional[int] = field(init=False, default=None)

  @dataclass
  class _Document:
    """Internal helper for grouping chunks by document for reference formatting.

    Chunks with the same metadata are grouped into the same document. This allows
    for hierarchical citation (document -> chunk within document).

    Attributes:
        metadata: Document-level metadata (source, author, etc.).
        chunks: Dictionary mapping chunk IDs to _Chunk objects.
        index: Optional display index assigned when document is first referenced (1-based).
            None until the document is first referenced.
        _last_chunk_index: Counter for assigning sequential indices to chunks within
            this document.
    """

    metadata: Dict[str, Any] = field(init=False, default_factory=dict)
    chunks: Dict[int, "TQueryResponse._Chunk"] = field(init=False, default_factory=dict)
    index: Optional[int] = field(init=False, default=None)
    _last_chunk_index: int = field(init=False, default=0)

    def get_chunk(self, id: int) -> Tuple[int, "TQueryResponse._Chunk"]:
      """Get a chunk and assign it an index if not already assigned.

      Args:
          id: The chunk ID to retrieve.

      Returns:
          Tuple of (chunk_index, chunk_object).
      """
      chunk = self.chunks[id]
      if chunk.index is None:
        self._last_chunk_index += 1
        chunk.index = self._last_chunk_index
      return chunk.index, chunk

    def to_dict(self) -> Dict[str, Any]:
      """Convert document to dictionary with metadata and referenced chunks.

      Returns:
          Dictionary with 'meta' (metadata dict) and 'chunks' (dict mapping
          chunk indices to (content, id) tuples) for chunks that were referenced.
      """
      return {
        "meta": self.metadata,
        "chunks": {chunk.index: (chunk.content, chunk.id) for chunk in self.chunks.values() if chunk.index is not None},
      }

  @dataclass
  class _ReferenceList:
    """Internal helper for managing the complete list of referenced documents.

    Maintains a collection of documents and assigns display indices to them as they
    are referenced in the response text.

    Attributes:
        documents: Dictionary mapping document IDs (hash of metadata) to _Document
            objects. Uses defaultdict to create documents on-demand.
        _last_document_index: Counter for assigning sequential indices to documents.
    """

    documents: Dict[int, "TQueryResponse._Document"] = field(
      default_factory=lambda: defaultdict(lambda: TQueryResponse._Document())
    )
    _last_document_index: int = field(init=False, default=0)

    def get_doc(self, id: int) -> Tuple[int, "TQueryResponse._Document"]:
      """Get a document and assign it an index if not already assigned.

      Args:
          id: The document ID (hash of metadata).

      Returns:
          Tuple of (document_index, document_object).
      """
      doc = self.documents[id]
      if doc.index is None:
        self._last_document_index += 1
        doc.index = self._last_document_index
      return doc.index, doc

    def to_dict(self):
      """Convert reference list to dictionary of referenced documents.

      Returns:
          Dictionary mapping document indices to document dictionaries,
          for documents that were actually referenced in the response.
      """
      return {doc.index: doc.to_dict() for doc in self.documents.values() if doc.index is not None}

  def format_references(self, format_fn: Callable[[int, List[int], Any], str] = lambda i, _, __: f"[{i}]"):
    """Format citation references in the response text with customizable formatting.

    This method processes citation markers (e.g., [1], [1][2], [1 2 3]) in the response
    text and replaces them with formatted citations. It groups chunks by their parent
    documents and assigns hierarchical indices.

    The method:
    1. Groups chunks by document (based on metadata)
    2. Assigns sequential indices to documents and chunks as they're referenced
    3. Finds citation patterns in the response text
    4. Replaces them with formatted citations using the provided format function

    Args:
        format_fn: A function to format each citation. Takes three arguments:
            - doc_index (int): The document's display index (1-based)
            - chunk_indices (List[int]): List of chunk indices within the document (1-based)
            - metadata (Any): The document's metadata dictionary
            Returns: A formatted citation string.
            Default: lambda i, _, __: f"[{i}]" (simple document number)

    Returns:
        Tuple of (formatted_response, reference_dict) where:
        - formatted_response (str): The response text with citations replaced
        - reference_dict (dict): Dictionary mapping document indices to their details
          (metadata and referenced chunks)

    Example:
        Original response: "Paris is the capital [1] and largest city [2][3]."
        With default format_fn: "Paris is the capital [1] and largest city [2]."
        (assuming chunks 2 and 3 are from the same document)
    """
    # Build the reference list by grouping chunks into documents
    reference_list = self._ReferenceList()
    # Map from chunk reference numbers (as strings) to (doc_id, chunk_id) tuples
    ref2data: Dict[str, Tuple[int, int]] = {}

    # Process each chunk in the context and organize into documents
    for i, (chunk, _) in enumerate(self.context.chunks):
      metadata: Dict[str, Any] = getattr(chunk, "metadata", {})
      chunk_id = int(chunk.id)
      # Group chunks by metadata: chunks with same metadata belong to same document
      if metadata == {}:
        # No metadata: treat each chunk as its own document
        doc_id = chunk_id
      else:
        # Hash the metadata to create a document ID
        doc_id = hash(frozenset(metadata.items()))
      reference_list.documents[doc_id].metadata = metadata
      reference_list.documents[doc_id].chunks[chunk_id] = TQueryResponse._Chunk(chunk_id, str(chunk))
      # Map the chunk's reference number (1-based) to its doc and chunk IDs
      ref2data[str(i + 1)] = (doc_id, chunk_id)

    def _replace_fn(match: str | re.Match[str]) -> str:
      """Replace a citation pattern with formatted citation text.

      Args:
          match: Either a string or regex match object containing the citation pattern.

      Returns:
          Formatted citation string replacing the original pattern.
      """
      text = match if isinstance(match, str) else match.group()
      # Extract all reference numbers from the pattern (e.g., "[1][2][3]" -> ["1", "2", "3"])
      references = re.findall(r"(\d+)", text)
      # Track which chunks from each document are referenced
      seen_docs: Dict[int, List[int]] = defaultdict(list)

      # First pass: collect all chunk IDs grouped by document
      for reference in references:
        d = ref2data.get(reference, None)
        if d is None:
          continue
        seen_docs[d[0]].append(d[1])

      # Second pass: generate formatted citation for each unique document
      r = ""
      for reference in references:
        d = ref2data.get(reference, None)
        if d is None:
          continue

        doc_id = d[0]
        chunk_ids = seen_docs.get(doc_id, None)
        if chunk_ids is None:
          # Already processed this document, skip
          continue
        # Remove from seen_docs so we only format each document once
        seen_docs.pop(doc_id)

        # Get or assign document index, and get chunk indices within document
        doc_index, doc = reference_list.get_doc(doc_id)
        r += format_fn(doc_index, [doc.get_chunk(id)[0] for id in chunk_ids], doc.metadata)
      return r

    # Find and replace all citation patterns in the response text
    # Pattern matches: [1], [1][2], [1 2 3], etc.
    return re.sub(r"\[\d[\s\d\]\[]*\]", _replace_fn, self.response), reference_list.to_dict()

  ####################################################################################################
