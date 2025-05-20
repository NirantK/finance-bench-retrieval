import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

import cohere
import turbopuffer as tpuf
from dotenv import load_dotenv
from loguru import logger

from src.embed import query_fastembedding

load_dotenv()
co = cohere.ClientV2()
tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"


def cohere_rerank(query: str, documents: List[str], top_k: int) -> str:
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_k,
    )
    indices = [result.index for result in response.results]
    return [documents[i] for i in indices]


def reciprocal_rank_fusion(result_lists, k=60):  # simple way to fuse results based on position
    scores, all_results = {}, {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.id] = scores.get(item.id, 0) + 1.0 / (k + rank)
            all_results[item.id] = item
    return [
        setattr(all_results[doc_id], "dist", score) or all_results[doc_id]
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# @tool
def hybrid_search(query: str, top_k: int) -> str:
    """
    Search the database for the most relevant documents using a hybrid approach.
    """
    namespace = os.getenv("TURBOPUFFER_NAMESPACE")
    ns = tpuf.Namespace(namespace)
    query_vector = query_fastembedding(query)
    with ThreadPoolExecutor() as executor:  # concurrent, could add more
        fts_future = executor.submit(
            ns.query,
            rank_by=["text", "BM25", query],
            include_attributes=["text", "doc_name", "doc_period"],
            top_k=top_k,
        )
        vector_future = executor.submit(
            ns.query,
            rank_by=["vector", "ANN", query_vector],
            include_attributes=["text", "doc_name", "doc_period"],
            top_k=top_k,
        )
        fts_result, vector_result = fts_future.result(), vector_future.result()
        fts_text, vector_text = (
            [row.attributes["text"] for row in fts_result.rows],
            [row.attributes["text"] for row in vector_result.rows],
        )
        logger.info(f"# FTS results: {len(fts_text)}, # Vector results: {len(vector_text)}")
        results = cohere_rerank(query, documents=list(set(fts_text + vector_text)), top_k=top_k)
        return results


# @tool
def vector_search(query: str, top_k: int) -> str:
    """
    Search the database for the most relevant documents.
    """
    namespace = os.getenv("TURBOPUFFER_NAMESPACE")
    ns = tpuf.Namespace(namespace)
    query_vector = query_fastembedding(query)
    query_result = ns.query(
        rank_by=["vector", "ANN", query_vector],
        top_k=top_k,
        include_attributes=["text", "doc_name", "doc_period"],
    ).rows
    results = []
    for row in query_result:
        results.append(row.attributes["text"])
    return "\n\n".join(results)


# @tool("bm25_search", parse_docstring=True, return_direct=False)
# @tool
def bm25_search(query: str, top_k: int, rank_by: str = "text") -> str:
    """
    Search the database for the most relevant documents using BM25.

    Args:
        query (str): The query to search for.
        top_k (int, optional): The number of results to return. Defaults to 10.
    """
    namespace = os.getenv("TURBOPUFFER_NAMESPACE")
    ns = tpuf.Namespace(namespace)
    query_result = ns.query(
        rank_by=[rank_by, "BM25", query],
        top_k=top_k,
        include_attributes=["text", "doc_name", "doc_period"],
    ).rows
    logger.info(query_result)
    results = []
    # logger.info(query_result)
    for row in query_result:
        results.append(row.attributes["text"])
    return "\n\n".join(results)
