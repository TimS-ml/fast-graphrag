"""Benchmarking script for GraphRAG knowledge graph storage performance.

This module evaluates the performance of FastGraphRAG using its graph-based
knowledge storage backend. It implements a full pipeline for:
    1. Loading benchmark datasets (2wikimultihopqa, hotpotqa)
    2. Building knowledge graphs from document corpus
    3. Executing multi-hop queries against the graph
    4. Computing retrieval metrics (perfect retrieval percentage)

The benchmark measures how effectively the graph structure captures relationships
needed for multi-hop question answering compared to ground truth evidence documents.

Key operations:
    - Dataset loading from JSON files with deduplication via xxhash
    - Knowledge graph construction with domain-specific entity extraction
    - Asynchronous query execution with progress tracking
    - Retrieval accuracy evaluation on full dataset and multi-hop subsets

Usage:
    python graph_benchmark.py -d <dataset> -n <subset_size> [-c] [-b] [-s]

    Options:
        -c, --create: Build knowledge graph from corpus
        -b, --benchmark: Execute queries and measure retrieval
        -s, --score: Compute and display retrieval metrics
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import xxhash
from _domain import DOMAIN, ENTITY_TYPES, QUERIES
from dotenv import load_dotenv
from tqdm import tqdm

from fast_graphrag import GraphRAG, QueryParam
from fast_graphrag._utils import get_event_loop


@dataclass
class Query:
    """Data structure holding a benchmark query with ground truth evidence.

    Attributes:
        question: The multi-hop question to answer.
        answer: Ground truth answer string.
        evidence: List of (document_title, passage_index) tuples indicating
                 supporting passages in the corpus.
    """

    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()


def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """Load benchmark dataset from JSON file.

    Args:
        dataset_name: Name of the dataset ('2wikimultihopqa' or 'hotpotqa').
        subset: Number of examples to use (0 means all). Useful for testing
               with smaller corpus sizes.

    Returns:
        List of dataset examples, each containing context passages and
        a question with supporting facts.

    Raises:
        FileNotFoundError: If dataset file not found in ./datasets/ directory.
    """
    with open(f"./datasets/{dataset_name}.json", "r") as f:
        dataset = json.load(f)

    if subset:
        return dataset[:subset]
    else:
        return dataset


def get_corpus(dataset: Any, dataset_name: str) -> Dict[int, Tuple[int | str, str]]:
    """Extract unique passages from dataset and deduplicate them.

    Processes all context passages from dataset examples, using xxhash3 to
    identify and remove duplicate passages. This creates a unique corpus
    for knowledge graph building.

    Args:
        dataset: Loaded dataset containing datapoints with 'context' field.
        dataset_name: Name of dataset to validate format ('2wikimultihopqa'
                     or 'hotpotqa').

    Returns:
        Dictionary mapping hash value to (title, text) tuples of unique passages.

    Raises:
        NotImplementedError: If dataset_name is not recognized.

    Note:
        Deduplication is essential to avoid redundant graph node creation
        and ensures accurate passage-level retrieval metrics.
    """
    if dataset_name == "2wikimultihopqa" or dataset_name == "hotpotqa":
        passages: Dict[int, Tuple[int | str, str]] = {}

        for datapoint in dataset:
            context = datapoint["context"]

            for passage in context:
                title, text = passage
                title = title.encode("utf-8").decode()
                text = "\n".join(text).encode("utf-8").decode()
                # Use xxhash3 for fast, 64-bit content-based hashing
                hash_t = xxhash.xxh3_64_intdigest(text)
                if hash_t not in passages:
                    passages[hash_t] = (title, text)

        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def get_queries(dataset: Any) -> List[Query]:
    """Convert dataset examples into Query objects with ground truth evidence.

    Extracts questions and their supporting facts from the raw dataset format.
    Supporting facts are document title and passage index pairs indicating
    which passages contain evidence for the question's answer.

    Args:
        dataset: Loaded dataset containing datapoints with 'question',
                'answer', and 'supporting_facts' fields.

    Returns:
        List of Query objects with questions and ground truth evidence references.
    """
    queries: List[Query] = []

    for datapoint in dataset:
        queries.append(
            Query(
                question=datapoint["question"].encode("utf-8").decode(),
                answer=datapoint["answer"],
                evidence=list(datapoint["supporting_facts"]),
            )
        )

    return queries


if __name__ == "__main__":
    load_dotenv()  # Load environment variables for API keys and configuration

    # Parse command-line arguments for benchmark execution modes
    parser = argparse.ArgumentParser(description="GraphRAG CLI")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use (0 = all).")
    parser.add_argument("-c", "--create", action="store_true", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", help="Report scores after benchmarking.")
    args = parser.parse_args()

    # Load dataset and prepare corpus for benchmarking
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/graph/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)

    # Phase 1: Build knowledge graph from corpus passages
    # This phase extracts entities and relationships from raw text using the LLM
    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))
        grag = GraphRAG(
            working_dir=working_dir,
            domain=DOMAIN[args.dataset],
            example_queries="\n".join(QUERIES),
            entity_types=ENTITY_TYPES[args.dataset],
        )
        # Insert all passages with their document titles as metadata
        grag.insert(
            [f"{title}: {corpus}" for _, (title, corpus) in tuple(corpus.items())],
            metadata=[{"id": title} for title in tuple(corpus.keys())],
        )

    # Phase 2: Query the graph and evaluate retrieval performance
    # This measures how well the knowledge graph answers multi-hop questions
    if args.benchmark:
        queries = get_queries(dataset)
        print("Dataset loaded. Queries:", len(queries))
        grag = GraphRAG(
            working_dir=working_dir,
            domain=DOMAIN[args.dataset],
            example_queries="\n".join(QUERIES),
            entity_types=ENTITY_TYPES[args.dataset],
        )

        async def _query_task(query: Query) -> Dict[str, Any]:
            """Execute a single query and extract retrieved evidence documents.

            Args:
                query: Query object containing question and ground truth evidence.

            Returns:
                Dictionary with question, answer, predicted evidence, and ground truth.
            """
            # Query only for context retrieval, no answer generation needed for metrics
            answer = await grag.async_query(query.question, QueryParam(only_context=True))
            return {
                "question": query.question,
                "answer": answer.response,
                # Extract document titles from returned chunks
                "evidence": [
                    corpus[chunk.metadata["id"]][0]
                        if isinstance(chunk.metadata["id"], int)
                        else chunk.metadata["id"]
                    for chunk, _ in answer.context.chunks
                ],
                # Ground truth titles from dataset supporting facts
                "ground_truth": [e[0] for e in query.evidence],
            }

        async def _run():
            """Execute all queries asynchronously with progress tracking."""
            await grag.state_manager.query_start()
            # Run queries in parallel with progress bar
            answers = [
                await a
                for a in tqdm(asyncio.as_completed([_query_task(query) for query in queries]), total=len(queries))
            ]
            await grag.state_manager.query_done()
            return answers

        answers = get_event_loop().run_until_complete(_run())

        # Save results for later evaluation
        with open(f"./results/graph/{args.dataset}_{args.n}.json", "w") as f:
            json.dump(answers, f, indent=4)

    # Phase 3: Compute and report retrieval metrics
    # Evaluates how many queries had perfect retrieval (all evidence documents found)
    if args.benchmark or args.score:
        with open(f"./results/graph/{args.dataset}_{args.n}.json", "r") as f:
            answers = json.load(f)

        # Try to load multihop questions subset for detailed analysis
        try:
            with open(f"./questions/{args.dataset}_{args.n}.json", "r") as f:
                questions_multihop = json.load(f)
        except FileNotFoundError:
            questions_multihop = []

        # Compute retrieval metrics: proportion of ground truth documents retrieved
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []

        for answer in answers:
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]

            # Compute recall: fraction of ground truth documents in retrieved results
            p_retrieved: float = len(set(ground_truth).intersection(set(predicted_evidence))) / len(set(ground_truth))
            retrieval_scores.append(p_retrieved)

            # Track multihop questions separately for comparison
            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)

        # Report perfect retrieval metric (percentage with score = 1.0)
        print(
            f"Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores])}"
        )
        if len(retrieval_scores_multihop):
            print(
                f"[multihop] Percentage of queries with perfect retrieval: {
                    np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop])
                }"
            )
