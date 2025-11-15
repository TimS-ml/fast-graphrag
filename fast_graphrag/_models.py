"""Pydantic data models and serialization utilities for LLM interactions.

This module provides:
- Base model classes with custom JSON schema generation for LLM structured outputs
- Response models for various LLM tasks (entity extraction, relation editing, etc.)
- Serialization utilities for converting data to CSV and reference list formats
- Custom metaclass for model aliasing to improve LLM prompt clarity

These models define the structured output format that LLMs should follow
when extracting entities, relationships, and generating responses.
"""

from itertools import chain
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic._internal import _model_construction

####################################################################################################
# LLM Models
####################################################################################################


def _json_schema_slim(schema: dict[str, Any]) -> None:
    """Remove unnecessary fields from JSON schema to create a slimmer version for LLM prompts.

    This function strips the 'required' field and 'title' from properties to reduce
    token usage in LLM prompts while maintaining essential schema information.

    Args:
        schema (dict): The JSON schema dictionary to modify in-place
    """
    schema.pop("required")
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


class _BaseModelAliasMeta(_model_construction.ModelMetaclass):
    """Custom metaclass for Pydantic models that supports aliasing and slim JSON schemas.

    This metaclass allows models to have alternative names in their JSON schema,
    which can make LLM prompts clearer and more concise. It also applies the
    slim JSON schema transformation to reduce token usage.
    """

    def __new__(
        cls, name: str, bases: tuple[type[Any], ...], dct: Dict[str, Any], alias: Optional[str] = None, **kwargs: Any
    ) -> type:
        """Create a new model class with optional aliasing.

        Args:
            name (str): The class name
            bases (tuple): Base classes
            dct (Dict): Class attributes dictionary
            alias (Optional[str]): Alternative name for the model in JSON schema
            **kwargs: Additional metaclass arguments

        Returns:
            type: The newly created model class
        """
        if alias:
            # Use the alias as the qualified name in the schema
            dct["__qualname__"] = alias
            name = alias
        return super().__new__(cls, name, bases, dct, json_schema_extra=_json_schema_slim, **kwargs)


class BaseModelAlias:
    """Base class for models that support aliasing and conversion operations.

    This class provides a foundation for creating Pydantic models with:
    - Custom name aliasing for clearer LLM prompts
    - Conversion to dataclass format
    - String serialization
    """

    class Model(BaseModel, metaclass=_BaseModelAliasMeta):
        """Inner Model class using the custom metaclass."""

        @staticmethod
        def to_dataclass(pydantic: Any) -> Any:
            """Convert a Pydantic model instance to a dataclass.

            Args:
                pydantic (Any): The Pydantic model instance

            Raises:
                NotImplementedError: This method must be implemented by subclasses
            """
            raise NotImplementedError

    def to_str(self) -> str:
        """Convert the model to a string representation.

        Returns:
            str: String representation of the model

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


####################################################################################################
# LLM Dumping to strings
####################################################################################################


def dump_to_csv(
    data: Iterable[object],
    fields: List[str],
    separator: str = "\t",
    with_header: bool = False,
    **values: Dict[str, List[Any]],
) -> List[str]:
    """Serialize data objects to CSV format for LLM consumption.

    This function converts structured data into tab-separated (or custom-separated)
    CSV rows, suitable for including in LLM prompts. It sanitizes newlines and tabs
    to prevent formatting issues.

    Args:
        data (Iterable[object]): Objects to serialize (must have attributes matching 'fields')
        fields (List[str]): Attribute names to extract from each object
        separator (str): Field separator. Defaults to tab character.
        with_header (bool): Whether to include a header row. Defaults to False.
        **values (Dict[str, List[Any]]): Additional columns to append, where keys are
            column names and values are lists of values (same length as data)

    Returns:
        List[str]: List of CSV rows as strings

    Example:
        >>> entities = [Entity(name="Alice", type="Person"), Entity(name="Bob", type="Person")]
        >>> dump_to_csv(entities, ["name", "type"], with_header=True)
        ['name\ttype', 'Alice\tPerson', 'Bob\tPerson']
    """
    rows = list(
        chain(
            # Optional header row
            (separator.join(chain(fields, values.keys())),) if with_header else (),
            # Data rows
            chain(
                separator.join(
                    chain(
                        # Extract fields from data object and sanitize
                        (str(getattr(d, field)).replace("\n", "  ").replace("\t", " ") for field in fields),
                        # Add additional values and sanitize
                        (str(v).replace("\n", "  ").replace("\t", " ") for v in vs),
                    )
                )
                for d, *vs in zip(data, *values.values())
            ),
        )
    )
    return rows


def dump_to_reference_list(data: Iterable[object], separator: str = "\n=====\n\n"):
    """Convert data objects to a numbered reference list format for LLM prompts.

    This format is useful for providing context to LLMs where each item needs
    a reference number (e.g., for citation or selection tasks).

    Args:
        data (Iterable[object]): Objects to format as references
        separator (str): Separator string between items. Defaults to "\n=====\n\n".

    Returns:
        List[str]: List of formatted reference strings with numbering

    Example:
        >>> chunks = ["First chunk content", "Second chunk content"]
        >>> dump_to_reference_list(chunks)
        ['[1]  First chunk content\n=====\n\n', '[2]  Second chunk content\n=====\n\n']
    """
    return [f"[{i + 1}]  {d}{separator}" for i, d in enumerate(data)]


####################################################################################################
# Response Models
####################################################################################################


class TAnswer(BaseModel):
    """Model for LLM-generated answers to user queries.

    Attributes:
        answer (str): The generated answer text
    """

    answer: str


class TEditRelation(BaseModel):
    """Model for grouping and summarizing multiple related facts into one.

    Used during graph construction to merge similar relationships or facts
    that refer to the same concept.

    Attributes:
        ids (List[int]): Indices of the facts being combined
        description (str): Comprehensive summarized description of all combined facts
    """

    ids: List[int] = Field(..., description="Ids of the facts that you are combining into one")
    description: str = Field(
        ..., description="Summarized description of the combined facts, in detail and comprehensive"
    )


class TEditRelationList(BaseModel):
    """Model for a collection of fact groups after relation editing.

    This model represents the output of relation summarization, where the LLM
    identifies groups of facts that should be merged.

    Attributes:
        groups (List[TEditRelation]): List of fact groups to be combined.
            Only includes groups with more than one fact.
    """

    groups: List[TEditRelation] = Field(
        ...,
        description="List of new fact groups; include only groups of more than one fact",
        alias="grouped_facts",
    )


class TEntityDescription(BaseModel):
    """Model for entity descriptions generated by LLM.

    Attributes:
        description (str): Textual description of an entity
    """

    description: str


class TQueryEntities(BaseModel):
    """Model for entities extracted from a user query.

    This model separates entities into two categories:
    - Named entities: Specific named items (e.g., "Alice", "Google", "Paris")
    - Generic entities: General concepts or types (e.g., "person", "company", "city")

    Attributes:
        named (List[str]): Named entities from the query (automatically uppercased)
        generic (List[str]): Generic entity types from the query
    """

    named: List[str] = Field(
        ...,
        description=("List of named entities extracted from the query"),
    )
    generic: List[str] = Field(
        ...,
        description=("List of generic entities extracted from the query"),
    )

    @field_validator("named", mode="before")
    @classmethod
    def uppercase_named(cls, value: List[str]):
        """Convert named entities to uppercase for consistency.

        Args:
            value (List[str]): List of named entity strings

        Returns:
            List[str]: Uppercased entity names, or original value if None
        """
        return [e.upper() for e in value] if value else value

    # Note: Generic entities are kept as-is (not uppercased)
    # Uncomment below to also uppercase generic entities:
    # @field_validator("generic", mode="before")
    # @classmethod
    # def uppercase_generic(cls, value: List[str]):
    #     return [e.upper() for e in value] if value else value
