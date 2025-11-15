"""Unit tests for GraphRAG type definitions and data structures.

This module tests the core data types used throughout the GraphRAG system,
including documents, chunks, entities, relationships, graphs, and query contexts.
Tests verify both data structure integrity and serialization/conversion functionality.
"""
# type: ignore
import re
import unittest
from dataclasses import asdict

from fast_graphrag._types import (
    TChunk,
    TContext,
    TDocument,
    TEntity,
    TGraph,
    TQueryResponse,
    TRelation,
    TScore,
)


class TestTypes(unittest.TestCase):
    """Test suite for GraphRAG type classes and their conversions."""
    def test_tdocument(self):
        """Test TDocument dataclass creation and attribute access."""
        doc = TDocument(data="Sample data", metadata={"key": "value"})
        self.assertEqual(doc.data, "Sample data")
        self.assertEqual(doc.metadata, {"key": "value"})

    def test_tchunk(self):
        """Test TChunk dataclass for text chunk representation."""
        chunk = TChunk(id=123, content="Sample content", metadata={"key": "value"})
        self.assertEqual(chunk.id, 123)
        self.assertEqual(chunk.content, "Sample content")
        self.assertEqual(chunk.metadata, {"key": "value"})

    def test_tentity(self):
        """Test TEntity dataclass and conversion from Pydantic model.

        Verifies that:
        - Entity attributes are properly stored
        - Pydantic model can be converted to dataclass
        - Name and type are uppercased during conversion
        """
        entity = TEntity(name="Entity1", type="Type1", description="Description1")
        self.assertEqual(entity.name, "Entity1")
        self.assertEqual(entity.type, "Type1")
        self.assertEqual(entity.description, "Description1")

        # Test conversion from Pydantic model to dataclass
        pydantic_entity = TEntity.Model(name="Entity1", type="Type1", desc="Description1")
        entity.name = entity.name.upper()
        entity.type = entity.type.upper()
        self.assertEqual(asdict(entity), asdict(pydantic_entity.to_dataclass(pydantic_entity)))

    def test_trelation(self):
        """Test TRelation dataclass and conversion from Pydantic model.

        Verifies that:
        - Relationship attributes (source, target, description) are properly stored
        - Pydantic model can be converted to dataclass
        - Source and target are uppercased during conversion
        """
        relation = TRelation(source="Entity1", target="Entity2", description="Relation description")
        self.assertEqual(relation.source, "Entity1")
        self.assertEqual(relation.target, "Entity2")
        self.assertEqual(relation.description, "Relation description")

        # Test conversion from Pydantic model to dataclass
        pydantic_relation = TRelation.Model(source="Entity1", target="Entity2", desc="Relation description")

        relation.source = relation.source.upper()
        relation.target = relation.target.upper()
        self.assertEqual(asdict(relation), asdict(pydantic_relation.to_dataclass(pydantic_relation)))

    def test_tgraph(self):
        """Test TGraph dataclass and conversion from Pydantic model.

        Verifies that:
        - Graph can store entities and relationships
        - Pydantic model can be converted to dataclass
        - Entity names/types and relationship sources/targets are uppercased
        """
        entity = TEntity(name="Entity1", type="Type1", description="Description1")
        relation = TRelation(source="Entity1", target="Entity2", description="Relation description")
        graph = TGraph(entities=[entity], relationships=[relation])
        self.assertEqual(graph.entities, [entity])
        self.assertEqual(graph.relationships, [relation])

        # Test conversion from Pydantic model to dataclass
        pydantic_graph = TGraph.Model(
            entities=[TEntity.Model(name="Entity1", type="Type1", desc="Description1")],
            relationships=[TRelation.Model(source="Entity1", target="Entity2", desc="Relation description")],
            other_relationships=[],
        )

        # Uppercase entity and relation fields for comparison
        for entity in graph.entities:
            entity.name = entity.name.upper()
            entity.type = entity.type.upper()
        for relation in graph.relationships:
            relation.source = relation.source.upper()
            relation.target = relation.target.upper()
        self.assertEqual(asdict(graph), asdict(pydantic_graph.to_dataclass(pydantic_graph)))

    def test_tcontext(self):
        """Test TContext dataclass and its truncation functionality.

        Verifies that:
        - Context can store entities, relationships, and chunks with scores
        - Truncation respects character limits for each section
        - Output format includes CSV sections for entities, relationships, and sources
        - Total output stays within specified character limits
        """
        # Create test data with duplicates to test truncation
        entities = [TEntity(name="Entity1", type="Type1", description="Sample description 1")] * 8 + [
            TEntity(name="Entity2", type="Type2", description="Sample description 2")
        ] * 8
        relationships = [
            TRelation(source="Entity1", target="Entity2", description="Relation description 12")
        ] * 8 + [
            TRelation(source="Entity2", target="Entity1", description="Relation description 21")
        ] * 8
        chunks = [
            TChunk(id=i, content=f"Long and repeated chunk content {i}" * 4, metadata={"key": f"value {i}"})
            for i in range(16)
        ]

        # Link relationships to chunks
        for r, c in zip(relationships, chunks):
            r.chunks = [c.id]

        # Create context with scored items
        context = TContext(
            entities=[(e, TScore(0.9)) for e in entities],
            relations=[(r, TScore(0.8)) for r in relationships],
            chunks=[(c, TScore(0.7)) for c in chunks],
        )

        # Test truncation with character limits
        max_chars = {"entities": 128, "relations": 128, "chunks": 512}
        csv = context.truncate(max_chars.copy(), True)

        # Extract sections from the formatted output
        csv_entities = re.findall(r"## Entities\n```csv\n(.*?)\n```", csv, re.DOTALL)
        csv_relationships = re.findall(r"## Relationships\n```csv\n(.*?)\n```", csv, re.DOTALL)
        csv_chunks = re.findall(r"## Sources\n.*=====", csv, re.DOTALL)

        # Verify all sections are present
        self.assertEqual(len(csv_entities), 1)
        self.assertEqual(len(csv_relationships), 1)
        self.assertEqual(len(csv_chunks), 1)

        # Verify truncation respects character limits (with small buffer for formatting)
        self.assertGreaterEqual(
            sum(max_chars.values()) + 16, len(csv_entities[0]) + len(csv_relationships[0]) + len(csv_chunks[0])
        )

    def test_tqueryresponse(self):
        """Test TQueryResponse dataclass for query results.

        Verifies that query responses properly store both the answer
        and the context used to generate it.
        """
        context = TContext(
            entities=[("Entity1", TScore(0.9))],
            relations=[("Relation1", TScore(0.8))],
            chunks=[("Chunk1", TScore(0.7))],
        )
        query_response = TQueryResponse(response="Sample response", context=context)
        self.assertEqual(query_response.response, "Sample response")
        self.assertEqual(query_response.context, context)


if __name__ == "__main__":
    unittest.main()
