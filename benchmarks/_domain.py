"""Domain configuration and benchmark dataset definitions.

This module defines benchmark domain prompts, test queries, and entity types for
evaluating GraphRAG performance on multi-hop question answering tasks. It serves
as a configuration module for benchmark scripts testing retrieval and reasoning
capabilities across different datasets.

Key components:
    - DOMAIN: System prompts tailored for each dataset to guide entity extraction
    - QUERIES: Example queries used for few-shot prompting in graph extraction
    - ENTITY_TYPES: Entity categories to extract and classify in knowledge graphs

Supported datasets:
    - 2wikimultihopqa: Multi-hop QA over Wikipedia articles with family relations
    - hotpotqa: Complex multi-hop reasoning questions requiring evidence synthesis
"""

from typing import Dict, List

# Domain-specific prompts for guiding knowledge extraction from text
# These prompts instruct the LLM on what entities and relationships to extract
DOMAIN: Dict[str, str] = {
    "2wikimultihopqa": """Analyse the following passage and identify the people, creative works, and places mentioned in it. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 IMPORTANT: among other entities and relationships you find, make sure to extract as separate entities (to be connected with the main one) a person's
 role as a family member (such as 'son', 'uncle', 'wife', ...), their profession (such as 'director'), and the location
 where they live or work. Pay attention to the spelling of the names.""",  # noqa: E501
    "hotpotqa": """Analyse the following passage and identify all the entities mentioned in it and their relationships. Your goal is to create an RDF (Resource Description Framework) graph from the given text.
 Pay attention to the spelling of the entity names."""
}

# Example queries for few-shot learning in entity and relationship extraction.
# These demonstrate the types of multi-hop questions the system should handle.
# Used to guide the LLM during knowledge graph construction.
QUERIES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "When did Prince Arthur's mother die?",
        "What is the place of birth of Elizabeth II's husband?",
        "Which film has the director died later, Interstellar or Harry Potter I?",
        "Where does the singer who wrote the song Blank Space work at?",
    ],
    "hotpotqa": [
        "Are Christopher Nolan and Sathish Kalathil both film directors?",
        "What language were books being translated into during the era of Haymo of Faversham?",
        "Who directed the film that was shot in or around Leland, North Carolina in 1986?",
        "Who wrote a song after attending a luau in the Koolauloa District on the island of Oahu in Honolulu County?"
    ]
}

# Entity type taxonomies for each dataset.
# These define the categories used to classify nodes in the knowledge graph.
# Different datasets have different emphasis on entity types based on their focus.
ENTITY_TYPES: Dict[str, List[str]] = {
    "2wikimultihopqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
    ],
    "hotpotqa": [
        "person",
        "familiy_role",
        "location",
        "organization",
        "creative_work",
        "profession",
        "event",
        "year"
    ],
}
