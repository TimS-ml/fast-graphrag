"""Unit tests for GraphRAG data models and utility functions.

This module tests the various Pydantic models used in the GraphRAG system,
including entity and relationship models, as well as formatting utilities
for CSV and reference list generation.
"""
# type: ignore
import unittest

from pydantic import ValidationError

from fast_graphrag._models import (
    TEditRelation,
    TEditRelationList,
    TQueryEntities,
    dump_to_csv,
    dump_to_reference_list,
)
from fast_graphrag._types import TEntity


class TestModels(unittest.TestCase):
    """Test suite for GraphRAG model classes."""
    def test_tqueryentities(self):
        """Test TQueryEntities model behavior.

        Verifies that:
        - Named entities are automatically uppercased
        - Generic entities retain original casing
        - Invalid parameters raise ValidationError
        """
        query_entities = TQueryEntities(named=["Entity1", "Entity2"], generic=["Generic1", "Generic2"])
        # Named entities should be converted to uppercase
        self.assertEqual(query_entities.named, ["ENTITY1", "ENTITY2"])
        # Generic entities should retain original casing
        self.assertEqual(query_entities.generic, ["Generic1", "Generic2"])

        # Test that invalid parameters raise ValidationError
        with self.assertRaises(ValidationError):
            TQueryEntities(entities=["Entity1", "Entity2"], n="two")

    def test_teditrelationship(self):
        """Test TEditRelation model for relationship editing.

        Verifies that relationship IDs and descriptions are properly stored.
        """
        edit_relationship = TEditRelation(ids=[1, 2], description="Combined relationship description")
        self.assertEqual(edit_relationship.ids, [1, 2])
        self.assertEqual(edit_relationship.description, "Combined relationship description")

    def test_teditrelationshiplist(self):
        """Test TEditRelationList model for grouped relationships.

        Verifies that relationship groups are properly accessed via the groups property.
        """
        edit_relationship = TEditRelation(ids=[1, 2], description="Combined relationship description")
        edit_relationship_list = TEditRelationList(grouped_facts=[edit_relationship])
        self.assertEqual(edit_relationship_list.groups, [edit_relationship])

    def test_dump_to_csv(self):
        """Test dump_to_csv utility with entity data.

        Verifies that entities can be converted to CSV format with:
        - Selected fields
        - Optional headers
        - Additional column values
        """
        data = [TEntity(name="Sample name", type="SAMPLE TYPE", description="Sample description")]
        fields = ["name", "type"]
        values = {"score": [0.9]}
        csv_output = dump_to_csv(data, fields, with_header=True, **values)
        expected_output = ["name\ttype\tscore", "Sample name\tSAMPLE TYPE\t0.9"]
        self.assertEqual(csv_output, expected_output)


class TestDumpToReferenceList(unittest.TestCase):
    """Test suite for dump_to_reference_list utility function."""

    def test_empty_list(self):
        """Test that an empty list returns an empty result."""
        self.assertEqual(dump_to_reference_list([]), [])

    def test_single_element(self):
        """Test formatting a single element with numbered reference."""
        self.assertEqual(dump_to_reference_list(["item"]), ["[1]  item\n=====\n\n"])

    def test_multiple_elements(self):
        """Test formatting multiple elements with sequential numbering."""
        data = ["item1", "item2", "item3"]
        expected = [
            "[1]  item1\n=====\n\n",
            "[2]  item2\n=====\n\n",
            "[3]  item3\n=====\n\n"
        ]
        self.assertEqual(dump_to_reference_list(data), expected)

    def test_custom_separator(self):
        """Test using a custom separator instead of default formatting."""
        data = ["item1", "item2"]
        separator = " | "
        expected = [
            "[1]  item1 | ",
            "[2]  item2 | "
        ]
        self.assertEqual(dump_to_reference_list(data, separator), expected)


class TestDumpToCsv(unittest.TestCase):
    """Test suite for dump_to_csv utility function with various scenarios."""
    def test_empty_data(self):
        """Test that empty data returns an empty list."""
        self.assertEqual(dump_to_csv([], ["field1", "field2"]), [])

    def test_single_element(self):
        """Test CSV conversion with a single data element."""
        class Data:
            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2

        data = [Data("value1", "value2")]
        expected = ["value1\tvalue2"]
        self.assertEqual(dump_to_csv(data, ["field1", "field2"]), expected)

    def test_multiple_elements(self):
        """Test CSV conversion with multiple data elements."""
        class Data:
            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2

        data = [Data("value1", "value2"), Data("value3", "value4")]
        expected = ["value1\tvalue2", "value3\tvalue4"]
        self.assertEqual(dump_to_csv(data, ["field1", "field2"]), expected)

    def test_with_header(self):
        """Test CSV conversion with header row included."""
        class Data:
            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2

        data = [Data("value1", "value2")]
        expected = ["field1\tfield2", "value1\tvalue2"]
        self.assertEqual(dump_to_csv(data, ["field1", "field2"], with_header=True), expected)

    def test_custom_separator(self):
        """Test CSV conversion with a custom separator instead of tabs."""
        class Data:
            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2

        data = [Data("value1", "value2")]
        expected = ["value1 | value2"]
        self.assertEqual(dump_to_csv(data, ["field1", "field2"], separator=" | "), expected)

    def test_additional_values(self):
        """Test CSV conversion with additional column values via kwargs."""
        class Data:
            def __init__(self, field1, field2):
                self.field1 = field1
                self.field2 = field2

        data = [Data("value1", "value2")]
        expected = ["value1\tvalue2\tvalue3"]
        self.assertEqual(dump_to_csv(data, ["field1", "field2"], value3=["value3"]), expected)


if __name__ == "__main__":
    unittest.main()
