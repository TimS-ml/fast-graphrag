"""Benchmarking script for Nano-GraphRAG knowledge graph implementation.

This module evaluates Nano-GraphRAG, a lightweight graph-based RAG system,
against the same benchmark datasets used for FastGraphRAG. It provides insights
into lightweight vs feature-rich knowledge graph implementations.

Nano-GraphRAG is designed for efficiency with minimal dependencies, making it
useful for deployment in resource-constrained environments.

Pipeline:
    1. Load benchmark datasets (2wikimultihopqa, hotpotqa)
    2. Build knowledge graph using Nano-GraphRAG
    3. Execute multi-hop queries against the graph
    4. Compute retrieval metrics and compare with ground truth

This comparison helps evaluate:
    - Trade-offs between feature completeness and efficiency
    - Lightweight implementation quality
    - Scalability for resource-constrained settings
    - Retrieval quality with minimal overhead
    - Performance on multi-hop reasoning

Key operations:
    - Dataset loading and deduplication
    - Nano-GraphRAG graph construction using GPT-4o-mini
    - Asynchronous query execution in different modes
    - Retrieval metric computation
    - Per-dataset analysis of performance

Usage:
    python nano_benchmark.py -d <dataset> -n <subset_size> [--mode <mode>] [-c] [-b] [-s]

    Modes: local, global, hybrid (default: local)
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import xxhash
from dotenv import load_dotenv
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._llm import gpt_4o_mini_complete
from nano_graphrag._utils import always_get_an_event_loop, logging
from tqdm import tqdm

# Suppress verbose logging from dependencies
logging.getLogger("nano-graphrag").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)

@dataclass
class Query:
    """Data structure for benchmark queries with ground truth evidence.

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

    Uses xxhash3 for fast content-based deduplication to avoid redundant
    nodes in the knowledge graph.

    Args:
        dataset: Loaded dataset containing datapoints with 'context' field.
        dataset_name: Name of dataset to validate format.

    Returns:
        Dictionary mapping hash value to (title, text) tuples of unique passages.

    Raises:
        NotImplementedError: If dataset_name is not recognized.
    """
    if dataset_name == "2wikimultihopqa" or dataset_name == "hotpotqa":
        passages: Dict[int, Tuple[int | str, str]] = {}

        for datapoint in dataset:
            context = datapoint["context"]

            for passage in context:
                title, text = passage
                title = title.encode("utf-8").decode()
                text = "\n".join(text).encode("utf-8").decode()
                # Use xxhash3 for fast content-based deduplication
                hash_t = xxhash.xxh3_64_intdigest(text)
                if hash_t not in passages:
                    passages[hash_t] = (title, text)

        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def get_queries(dataset: Any) -> List[Query]:
    """Convert dataset examples into Query objects with ground truth evidence.

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
    load_dotenv()  # Load environment variables for API keys

    # Parse command-line arguments for benchmark modes
    parser = argparse.ArgumentParser(description="Nano-GraphRAG Comparison Benchmark")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use (0 = all).")
    parser.add_argument("-c", "--create", action="store_true", help="Build Nano-GraphRAG graph from corpus.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Execute queries and measure retrieval.")
    parser.add_argument("-s", "--score", action="store_true", help="Compute and display retrieval metrics.")
    parser.add_argument("--mode", default="local", help="Query mode (local, global, hybrid).")
    args = parser.parse_args()

    # Load dataset and prepare corpus
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/nano/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)

    # Create working directory if needed
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # Phase 1: Build Nano-GraphRAG knowledge graph from corpus passages
    # Uses a lightweight approach optimized for efficiency
    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))
        grag = GraphRAG(
            working_dir=working_dir,
            best_model_func=gpt_4o_mini_complete
        )
        # Insert passages for lightweight knowledge graph construction
        grag.insert([f"{title}: {corpus}" for _, (title, corpus) in tuple(corpus.items())])

    # Phase 2: Query the Nano-GraphRAG graph and evaluate retrieval performance
    # Tests different query modes for the lightweight implementation
    if args.benchmark:
        queries = get_queries(dataset)
        print("Dataset loaded. Queries:", len(queries))
        grag = GraphRAG(
            working_dir=working_dir,
            best_model_func=gpt_4o_mini_complete
        )

        async def _query_task(query: Query, mode: str) -> Dict[str, Any]:
            """Execute a single query using Nano-GraphRAG.

            Args:
                query: Query object containing question and ground truth evidence.
                mode: Query mode (local, global, or hybrid).

            Returns:
                Dictionary with question, answer, predicted evidence, and ground truth.

            Note:
                Parses CSV-formatted source citations from Nano-GraphRAG output to extract
                document titles. Uses tab-separated values in CSV format. Limits to top-8
                results for consistent comparison.
            """
            # Query Nano-GraphRAG in specified mode without full answer generation
            answer = await grag.aquery(
                query.question, QueryParam(mode=mode, only_need_context=True, local_max_token_for_text_unit=9000)
            )
            # Parse Nano-GraphRAG's CSV-formatted sources to extract document titles
            chunks = []
            for c in re.findall(r"\n-----Sources-----\n```csv\n(.*?)\n```", answer, re.DOTALL)[0].split("\n")[
                1:-1
            ]:
                try:
                    # Nano uses tab-separated format in CSV
                    chunks.append(c.split(",\t")[1].split(":")[0].lstrip('"'))
                except IndexError:
                    pass
            return {
                "question": query.question,
                "answer": "",
                "evidence": chunks[:8],  # Top-8 results for comparison
                "ground_truth": [e[0] for e in query.evidence],
            }

        async def _run(mode: str):
            """Execute all queries asynchronously with progress tracking."""
            answers = [
                await a
                for a in tqdm(
                    asyncio.as_completed([_query_task(query, mode=mode) for query in queries]), total=len(queries)
                )
            ]
            return answers

        # Run queries in specified mode
        answers = always_get_an_event_loop().run_until_complete(_run(mode=args.mode))

        # Save results with mode suffix for comparison across different modes
        with open(f"./results/nano/{args.dataset}_{args.n}_{args.mode}.json", "w") as f:
            json.dump(answers, f, indent=4)

    # Phase 3: Compute and report retrieval metrics
    # Evaluates how well Nano-GraphRAG's lightweight graph retrieves ground truth evidence
    if args.benchmark or args.score:
        with open(f"./results/nano/{args.dataset}_{args.n}_{args.mode}.json", "r") as f:
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
