"""Benchmarking script for vector database (HNSW) retrieval performance.

This module evaluates FastGraphRAG using a pure semantic search backend with
HNSW (Hierarchical Navigable Small Worlds) approximate nearest neighbor search.
It measures dense retrieval performance without the knowledge graph structure.

Pipeline:
    1. Load benchmark datasets (2wikimultihopqa, hotpotqa)
    2. Embed all passages using OpenAI embeddings
    3. Build HNSW index for fast approximate nearest neighbor search
    4. Execute queries using semantic similarity
    5. Compute retrieval metrics on embedding-based ranking

This baseline helps understand the contribution of knowledge graph structure
vs. pure semantic similarity for multi-hop reasoning tasks.

Key components:
    - VectorStorage: Manages HNSW index and passage embeddings
    - LLMService: Optional answer generation from retrieved context
    - Query execution and retrieval evaluation

Usage:
    python vdb_benchmark.py -d <dataset> -n <subset_size> [-c] [-b] [-s]
"""

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Dict, Iterable, List, Tuple

import numpy as np
import xxhash
from dotenv import load_dotenv
from tqdm import tqdm

from fast_graphrag._llm._llm_openai import OpenAIEmbeddingService, OpenAILLMService
from fast_graphrag._models import TAnswer
from fast_graphrag._storage._base import BaseStorage
from fast_graphrag._storage._ikv_pickle import PickleIndexedKeyValueStorage
from fast_graphrag._storage._namespace import Workspace
from fast_graphrag._storage._vdb_hnswlib import HNSWVectorStorage, HNSWVectorStorageConfig
from fast_graphrag._utils import get_event_loop


async def format_and_send_prompt(
    prompt: str,
    llm: OpenAILLMService,
    format_kwargs: dict[str, Any],
    **args: Any,
) -> Tuple[str, list[dict[str, str]]]:
    """Format a prompt template and send it to the LLM for completion.

    Args:
        prompt: Template string with placeholders for format_kwargs.
        llm: OpenAI LLM service instance for calling the model.
        format_kwargs: Dictionary of values to substitute in the prompt template.
        **args: Additional keyword arguments to pass to the LLM service.

    Returns:
        Tuple of (answer_text, rest) where answer_text is the model's response
        and rest is metadata about the response.
    """
    # Format the prompt with the supplied arguments
    formatted_prompt = prompt.format(**format_kwargs)

    # Send the formatted prompt to the LLM
    response, rest = await llm.send_message(prompt=formatted_prompt, response_model=TAnswer, **args)
    return response.answer, rest


def dump_to_reference_list(data: Iterable[object], separator: str = "\n=====\n\n") -> str:
    """Convert list of chunks to a formatted reference string.

    Args:
        data: Iterable of objects to be converted to reference items.
        separator: String separator between items (default: newlines with =====).

    Returns:
        Formatted string with numbered reference items.
    """
    return separator.join([f"[{i + 1}]  {d}" for i, d in enumerate(data)])


class VectorStorage:
    """Manages dense vector retrieval using HNSW approximate nearest neighbor search.

    This class combines two storage backends:
    - HNSWVectorStorage: Fast approximate nearest neighbor search on embeddings
    - PickleIndexedKeyValueStorage: Stores actual passage text by ID

    HNSW parameters (ef_construction=96, ef_search=48) balance between:
        - ef_construction: Higher = more accurate but slower indexing
        - ef_search: Higher = more accurate search but slower queries
    """

    def __init__(self, workspace: Workspace):
        """Initialize vector storage with HNSW index and embedding service.

        Args:
            workspace: Workspace instance for managing storage locations.
        """
        self.workspace = workspace
        # Configure HNSW with parameters optimized for efficient search
        self.vdb = HNSWVectorStorage[int, Any](
            config=HNSWVectorStorageConfig(ef_construction=96, ef_search=48),
            namespace=workspace.make_for("vdb"),
            embedding_dim=1536,  # OpenAI embedding dimension
        )
        self.embedder = OpenAIEmbeddingService()
        # Key-value store for passage text lookup by hash ID
        self.ikv = PickleIndexedKeyValueStorage[int, Any](config=None, namespace=workspace.make_for("ikv"))

    async def upsert(self, ids: Iterable[int], data: Iterable[Tuple[str, str]]) -> None:
        """Add or update passages and their embeddings in the vector database.

        Args:
            ids: Iterable of integer hash IDs for passages.
            data: Iterable of (title, text) tuples for passages.

        Note:
            Concatenates title and text with newlines before embedding to
            capture context for semantic search.
        """
        data = list(data)
        ids = list(ids)
        # Embed passages as "title\n\ntext" to preserve document context
        embeddings = await self.embedder.encode([f"{t}\n\n{c}" for t, c in data])
        # Store both raw passages and their dense embeddings
        await self.ikv.upsert([int(i) for i in ids], data)
        await self.vdb.upsert(ids, embeddings)

    async def get_context(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """Retrieve top-k most similar passages to a query using HNSW search.

        Args:
            query: Query text to search for.
            top_k: Number of top results to return.

        Returns:
            List of (title, text) tuples for the most similar passages.
        """
        # Embed the query using the same model as passages
        embedding = await self.embedder.encode([query])
        # Find nearest neighbor IDs and similarity scores
        ids, _ = await self.vdb.get_knn(embedding, top_k)

        # Retrieve passage text for the top-k IDs
        return [c for c in await self.ikv.get([int(i) for i in np.array(ids).flatten()]) if c is not None]

    async def query_start(self):
        """Prepare storage for querying by loading indices into memory.

        Called before executing batch queries to optimize I/O.
        """
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            """Load all storages asynchronously."""
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.query_start())
            return asyncio.gather(*tasks)

        await self.workspace.with_checkpoints(_fn)

        # Mark storages as in use to prevent cleanup
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def query_done(self):
        """Finalize querying and release storage resources.

        Called after all queries complete to allow cleanup and flushing.
        """
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        # Flush and close storages
        for storage_inst in storages:
            tasks.append(storage_inst.query_done())
        await asyncio.gather(*tasks)

        # Mark storages as no longer in use
        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def insert_start(self):
        """Prepare storage for inserting new passages.

        Called before batch inserts to optimize I/O patterns.
        """
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            """Load all storages asynchronously."""
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.insert_start())
            return asyncio.gather(*tasks)

        await self.workspace.with_checkpoints(_fn)

        # Mark storages as in use
        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def insert_done(self):
        """Finalize insertion and flush embeddings to disk.

        Called after all passages are inserted to persist indices.
        """
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        # Finalize and persist changes
        for storage_inst in storages:
            tasks.append(storage_inst.insert_done())
        await asyncio.gather(*tasks)

        # Mark storages as no longer in use
        for storage_inst in storages:
            storage_inst.set_in_progress(False)


class LLMService:
    """Service to interact with OpenAI LLM for answer generation from retrieved context.

    This service is optional in retrieval benchmarking and used only for full
    end-to-end evaluation (not for retrieval-only metrics).
    """

    # System prompt for extracting answers from retrieved context passages
    PROMPT = """You are a helpful assistant analyzing the given input data to provide an helpful response to the user query.

    # INPUT DATA
    {context}

    # USER QUERY
    {query}

    # INSTRUCTIONS
    Your goal is to provide a response to the user query using the relevant information in the input data.
    The "Sources" list contains raw text sources to help answer the query. It may contain noisy data, so pay attention when analyzing it.

    Follow these steps:
    1. Read and understand the user query.
    2. Carefully analyze all the "Sources" to get detailed information. Information could be scattered across several sources.
    4. Write the response to the user query based on the information you have gathered. Be very concise and answer the user query directly. If the response cannot be inferred from the input data, just say no relevant information was found. Do not make anything up or add unrelevant information.

    Answer:
    """ # noqa: E501

    def __init__(self):
        """Initialize LLM service with OpenAI API client."""
        self.llm = OpenAILLMService()

    async def ask_query(self, context: str, query: str) -> str:
        """Generate an answer by sending context and query to the LLM.

        Args:
            context: Formatted text of retrieved context passages.
            query: User's question to be answered.

        Returns:
            LLM-generated answer string based on the context.
        """
        return (
            await format_and_send_prompt(
                prompt=self.PROMPT, llm=self.llm, format_kwargs={"context": context, "query": query}
            )
        )[0]


async def upsert_to_vdb(data: List[Tuple[str, str]], working_dir: str = "./"):
    """Build HNSW index and store all passages with their embeddings.

    Args:
        data: List of (hash, (title, text)) tuples for passages.
        working_dir: Directory path for persistent storage of indices.
    """
    workspace = Workspace(working_dir)
    storage = VectorStorage(workspace)
    await storage.insert_start()
    # Generate consistent hash IDs from passage content for lookups
    await storage.upsert(
        [xxhash.xxh64(corpus).intdigest() for _, (title, corpus) in data],
        [(title, corpus) for _, (title, corpus) in data]
    )
    await storage.insert_done()


async def query_from_vdb(query: str, top_k: int, working_dir: str = "./", only_context: bool = True) -> str:
    """Retrieve passages from vector database and optionally generate answer.

    Args:
        query: Query text to search for.
        top_k: Number of top passages to retrieve.
        working_dir: Directory containing persistent storage indices.
        only_context: If True, return only context without LLM answer generation.

    Returns:
        String with answer (if generated) and context delimited by backticks.
        Format: "<answer>`````<context_string>"
    """
    workspace = Workspace(working_dir)
    storage = VectorStorage(workspace)
    await storage.query_start()
    # Retrieve most similar passages using HNSW
    chunks = await storage.get_context(query, top_k)
    await storage.query_done()

    # Optionally generate answer from context using LLM
    if only_context:
        answer = ""
    else:
        llm = LLMService()
        answer = await llm.ask_query(dump_to_reference_list([content for _, content in chunks]), query)
    # Format context with title separators for parsing
    context = "=====".join([title + ":=" + content for title, content in chunks])

    return answer + "`````" + context


@dataclass
class Query:
    """Data structure for benchmark queries with ground truth evidence.

    Attributes:
        question: The multi-hop question to answer.
        answer: Ground truth answer string.
        evidence: List of (document_title, passage_index) tuples for evidence.
    """

    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()


def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """Load a dataset from the datasets folder."""
    with open(f"./datasets/{dataset_name}.json", "r") as f:
        dataset = json.load(f)

    if subset:
        return dataset[:subset]
    else:
        return dataset


def get_corpus(dataset: Any, dataset_name: str) -> Dict[int, Tuple[int | str, str]]:
    """Get the corpus from the dataset."""
    if dataset_name == "2wikimultihopqa" or dataset_name == "hotpotqa":
        passages: Dict[int, Tuple[int | str, str]] = {}

        for datapoint in dataset:
            context = datapoint["context"]

            for passage in context:
                title, text = passage
                title = title.encode("utf-8").decode()
                text = "\n".join(text).encode("utf-8").decode()
                hash_t = xxhash.xxh3_64_intdigest(text)
                if hash_t not in passages:
                    passages[hash_t] = (title, text)

        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def get_queries(dataset: Any):
    """Get the queries from the dataset."""
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
    parser = argparse.ArgumentParser(description="Vector Database Benchmark")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use (0 = all).")
    parser.add_argument("-c", "--create", action="store_true", help="Build HNSW index from corpus.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Execute queries and measure retrieval.")
    parser.add_argument("-s", "--score", action="store_true", help="Compute and display retrieval metrics.")
    args = parser.parse_args()

    # Load dataset and prepare corpus
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/vdb/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)

    # Phase 1: Build HNSW vector index from corpus passages
    # This embeds all passages and builds the approximate nearest neighbor index
    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))

        async def _run_create():
            await upsert_to_vdb(list(corpus.items()), working_dir)

        get_event_loop().run_until_complete(_run_create())

    # Phase 2: Execute queries against the HNSW index and evaluate retrieval
    # This measures pure semantic similarity without knowledge graph structure
    if args.benchmark:
        queries = get_queries(dataset)

        async def _query_task(query: Query) -> Tuple[Query, str]:
            """Execute single query and return results from vector search."""
            return query, await query_from_vdb(query.question, 8, working_dir)

        async def _run_benchmark():
            """Run all queries asynchronously with progress tracking."""
            answers = [await _query_task(query) for query in tqdm(queries)]
            return answers

        answers = get_event_loop().run_until_complete(_run_benchmark())
        response: List[Dict[str, Any]] = []

        # Parse results and extract retrieved passage titles for metrics
        with open(f"./results/vdb/{args.dataset}_{args.n}.json", "w") as f:
            for r in answers:
                question, answer = r
                # Split answer from context using delimiter
                a, c = answer.split("`````")
                # Parse context string to extract document titles
                chunks = c.split("=====")
                chunks = dict([chunk.split(":=") for chunk in chunks])
                response.append(
                    {
                        "answer": a,
                        "evidence": tuple(chunks.keys()),  # Retrieved document titles
                        "question": question.question,
                        "ground_truth": [e[0] for e in question.evidence],  # Ground truth titles
                    }
                )
            json.dump(response, f, indent=4)

    # Phase 3: Compute and report retrieval metrics
    # Calculates what percentage of queries had all ground truth documents retrieved
    if args.benchmark or args.score:
        with open(f"./results/vdb/{args.dataset}_{args.n}.json", "r") as f:
            answers = json.load(f)

        # Load optional multihop subset for detailed analysis
        try:
            with open(f"./questions/{args.dataset}_{args.n}.json", "r") as f:
                questions_multihop = json.load(f)
        except FileNotFoundError:
            questions_multihop = []

        # Compute retrieval recall: fraction of ground truth documents retrieved
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []

        for answer in answers:
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]

            # Compute recall: intersection of predicted and ground truth divided by ground truth size
            p_retrieved: float = len(set(ground_truth).intersection(set(predicted_evidence))) / len(set(ground_truth))
            retrieval_scores.append(p_retrieved)

            # Track multihop questions separately for comparison
            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)

        # Report perfect retrieval percentage (perfect = 1.0 score)
        print(
            f"Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores])}"
        )
        if len(retrieval_scores_multihop):
            print(
                f"[multihop] Percentage of queries with perfect retrieval: {np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop])}"
            )
