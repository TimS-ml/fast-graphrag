import argparse
import asyncio
import json
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

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG


async def format_and_send_prompt(
    prompt: str,
    llm: OpenAILLMService,
    format_kwargs: dict[str, Any],
    **args: Any,
) -> Tuple[str, list[dict[str, str]]]:
    """Get a prompt, format it with the supplied args, and send it to the LLM."""
    # Format the prompt with the supplied arguments
    formatted_prompt = prompt.format(**format_kwargs)
    cprint(formatted_prompt)

    # Send the formatted prompt to the LLM
    response, rest = await llm.send_message(prompt=formatted_prompt, response_model=TAnswer, **args)
    cprint(response)
    cprint(rest)
    return response.answer, rest


def dump_to_reference_list(data: Iterable[object], separator: str = "\n=====\n\n"):
    """Convert list of chunks to a string."""
    result = separator.join([f"[{i + 1}]  {d}" for i, d in enumerate(data)])
    cprint(result)
    return result


class VectorStorage:
    """Vector storage with HNSW."""

    def __init__(self, workspace: Workspace):
        """Create vector storage with HNSW."""
        self.workspace = workspace
        self.vdb = HNSWVectorStorage[int, Any](
            config=HNSWVectorStorageConfig(
                # ef_construction=96, 
                # ef_search=48
                ef_construction=1000, 
                M=160,
                ef_search=1000,
                num_threads=-1
            ),
            namespace=workspace.make_for("vdb"),
            embedding_dim=1536,
        )
        self.embedder = OpenAIEmbeddingService()
        self.ikv = PickleIndexedKeyValueStorage[int, Any](config=None, namespace=workspace.make_for("ikv"))
        if DEBUG > 2:
            tprint("Init Vector storage with HNSW.", sep="-")
            cprint(self.workspace)
            cprint(self.vdb)
            cprint(self.embedder)
            cprint(self.ikv)

    async def upsert(self, ids: Iterable[int], data: Iterable[Tuple[str, str]]) -> None:
        """Add or update embeddings in the storage."""
        tprint("Add or update embeddings in the storage.", sep="-")

        data = list(data)
        ids = list(ids)
        texts = [f"{t}\n\n{c}" for t, c in data]
        embeddings = await self.embedder.encode(texts)
        cprint(embeddings.shape)
        
        await self.ikv.upsert([int(i) for i in ids], data)
        await self.vdb.upsert(ids, embeddings)

    async def get_context(self, query: str, top_k: int) -> List[Tuple[str, str]]:
        """Get the most similar embeddings to the query."""
        tprint("Get the most similar embeddings to the query.", sep="-")
        
        embedding = await self.embedder.encode([query])
        ids, scores = await self.vdb.get_knn(embedding, top_k)  # NOTE: this is important
        cprint(ids, scores, new_line=True)

        cprint('converting ids to int...', c='gray')
        results = [c for c in await self.ikv.get([int(i) for i in np.array(ids).flatten()]) if c is not None]
        cprint(results, new_line=True)
        cprint(len(results))
        return results

    async def query_start(self):
        """Load the storage for querying."""
        tprint("Load the storage for querying.", sep="-")
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.query_start())
            return asyncio.gather(*tasks)

        await self.workspace.with_checkpoints(_fn)

        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def query_done(self):
        """Finish querying the storage."""
        tprint("Finish querying the storage.", sep="-")
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        for storage_inst in storages:
            tasks.append(storage_inst.query_done())
        await asyncio.gather(*tasks)

        for storage_inst in storages:
            storage_inst.set_in_progress(False)

    async def insert_start(self):
        """Prepare the storage for inserting."""
        tprint("Starting insert", sep="-")
        storages: List[BaseStorage] = [self.ikv, self.vdb]

        def _fn():
            tasks: List[Awaitable[Any]] = []
            for storage_inst in storages:
                tasks.append(storage_inst.insert_start())
            return asyncio.gather(*tasks)

        await self.workspace.with_checkpoints(_fn)

        for storage_inst in storages:
            storage_inst.set_in_progress(True)

    async def insert_done(self):
        """Finish inserting into the storage."""
        tprint("Finishing insert", sep="-")
        tasks: List[Awaitable[Any]] = []
        storages: List[BaseStorage] = [self.ikv, self.vdb]
        for storage_inst in storages:
            tasks.append(storage_inst.insert_done())
        await asyncio.gather(*tasks)

        for storage_inst in storages:
            storage_inst.set_in_progress(False)


class LLMService:
    """Service to interact with the LLM."""

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
        """Create the LLM service."""
        self.llm = OpenAILLMService()
        cprint(self.llm)

    async def ask_query(self, context: str, query: str) -> str:
        """Ask a query to the LLM."""
        cprint(context)
        cprint(query)
        
        response, history = await format_and_send_prompt(
            prompt=self.PROMPT, 
            llm=self.llm, 
            format_kwargs={"context": context, "query": query}
        )
        cprint(response)
        cprint(history)
        return response


async def upsert_to_vdb(data: List[Tuple[str, str]], working_dir: str = "./"):
    """Upsert data to the vector storage."""
    tprint("Upsert data to the vector storage.")
    cprint(len(data))
    
    workspace = Workspace(working_dir)
    storage = VectorStorage(workspace)
    
    await storage.insert_start()
    
    ids = [xxhash.xxh64(corpus).intdigest() for _, (title, corpus) in data]
    cprint(ids[:3], new_line=True)
    
    pair_data = [(title, corpus) for _, (title, corpus) in data]
    cprint(pair_data[:2], new_line=True)
    
    await storage.upsert(ids, pair_data)
    await storage.insert_done()


async def query_from_vdb(query: str, top_k: int, working_dir: str = "./", only_context: bool = True) -> str:
    """Query the vector storage."""
    tprint("Query the vector storage.", sep="-")
    cprint(query, top_k, only_context, c='red')
    
    workspace = Workspace(working_dir)
    storage = VectorStorage(workspace)
    
    await storage.query_start()
    chunks = await storage.get_context(query, top_k)
    await storage.query_done()

    if only_context:
        llm_answer = ""
    else:
        llm = LLMService()
        cprint(llm)
        chunk_text = dump_to_reference_list([content for _, content in chunks])
        cprint(chunk_text)
        llm_answer = await llm.ask_query(chunk_text, query)
        cprint(llm_answer)
    
    context = "=====".join([title + ":=" + content for title, content in chunks])
    result = llm_answer + "`````" + context
    cprint(result, new_line=True)
    return result


@dataclass
class Query:
    """Query dataclass."""

    question: str = field()
    answer: str = field()
    evidence: List[Tuple[str, int]] = field()


def load_dataset(dataset_name: str, subset: int = 0) -> Any:
    """Load a dataset from the datasets folder."""
    cprint(dataset_name, subset)
    
    with open(f"./datasets/{dataset_name}.json", "r") as f:
        dataset = json.load(f)
    # cprint(len(dataset))

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

        cprint(len(passages))
        return passages
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")


def get_queries(dataset: Any):
    """Get the queries from the dataset."""
    queries: List[Query] = []

    for datapoint in dataset:
        query = Query(
            question=datapoint["question"].encode("utf-8").decode(),
            answer=datapoint["answer"],
            evidence=list(datapoint["supporting_facts"]),
        )
        queries.append(query)

    cprint(len(queries))
    return queries


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="GraphRAG CLI")
    parser.add_argument("-d", "--dataset", default="2wikimultihopqa", help="Dataset to use.")
    parser.add_argument("-n", type=int, default=0, help="Subset of corpus to use.")
    parser.add_argument("-c", "--create", action="store_true", help="Create the graph for the given dataset.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Benchmark the graph for the given dataset.")
    parser.add_argument("-s", "--score", action="store_true", help="Report scores after benchmarking.")
    args = parser.parse_args()
    cprint(args)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset, subset=args.n)
    working_dir = f"./db/vdb/{args.dataset}_{args.n}"
    corpus = get_corpus(dataset, args.dataset)
    cprint(working_dir)
    cprint(len(corpus))
    if DEBUG > 2: cprint(corpus)

    if args.create:
        print("Dataset loaded. Corpus:", len(corpus))

        async def _run_create():
            await upsert_to_vdb(list(corpus.items()), working_dir)

        get_event_loop().run_until_complete(_run_create())

    if args.benchmark:
        queries = get_queries(dataset)

        async def _query_task(query: Query) -> Tuple[Query, str]:
            # NOTE: include the ground truth in the query
            cprint(query, c='blue', new_line=True)
            result = await query_from_vdb(query.question, 10, working_dir)
            return query, result

        async def _run_benchmark():
            answers = [await _query_task(query) for query in tqdm(queries)]
            return answers

        answers = get_event_loop().run_until_complete(_run_benchmark())
        response: List[Dict[str, Any]] = []

        with open(f"./results/vdb/{args.dataset}_{args.n}.json", "w") as f:
            for r in answers:
                question, answer = r
                a, c = answer.split("`````")
                cprint(c, new_line=True, c='red')
                chunks = c.split("=====")
                chunks = dict([chunk.split(":=") for chunk in chunks])
                
                result = {
                    "answer": a,
                    "evidence": tuple(chunks.keys()),
                    "question": question.question,
                    "ground_truth": [e[0] for e in question.evidence],
                }
                cprint(result)
                response.append(result)
            
            cprint(response)
            json.dump(response, f, indent=4)

    if args.benchmark or args.score:
        with open(f"./results/vdb/{args.dataset}_{args.n}.json", "r") as f:
            answers = json.load(f)
        cprint(len(answers))

        try:
            with open(f"./questions/{args.dataset}_{args.n}.json", "r") as f:
                questions_multihop = json.load(f)
            cprint(len(questions_multihop))
        except FileNotFoundError:
            questions_multihop = []
            print("No multihop questions file found")

        # Compute retrieval metrics
        retrieval_scores: List[float] = []
        retrieval_scores_multihop: List[float] = []

        for answer in answers:
            ground_truth = answer["ground_truth"]
            predicted_evidence = answer["evidence"]
            cprint(ground_truth)
            cprint(predicted_evidence, new_line=True)

            p_retrieved: float = len(set(ground_truth).intersection(set(predicted_evidence))) / len(set(ground_truth))
            cprint(p_retrieved, c='red')
            retrieval_scores.append(p_retrieved)

            if answer["question"] in questions_multihop:
                retrieval_scores_multihop.append(p_retrieved)

        perfect_retrieval = np.mean([1 if s == 1.0 else 0 for s in retrieval_scores])
        cprint(f"Percentage of queries with perfect retrieval: {perfect_retrieval}", c='red')
        
        if len(retrieval_scores_multihop):
            multihop_perfect_retrieval = np.mean([1 if s == 1.0 else 0 for s in retrieval_scores_multihop])
            cprint(f"[multihop] Percentage of queries with perfect retrieval: {multihop_perfect_retrieval}", c='red')