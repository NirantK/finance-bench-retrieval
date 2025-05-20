import os

import turbopuffer as tpuf
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"


# @tool("bm25_search", parse_docstring=True, return_direct=False)
@tool
def bm25_search(query: str, top_k: int = 10, **kwargs) -> str:
    """
    Search the database for the most relevant documents.

    Args:
        query (str): The query to search for.
        top_k (int, optional): The number of results to return. Defaults to 10.
    """
    namespace = os.getenv("TURBOPUFFER_NAMESPACE")
    ns = tpuf.Namespace(namespace)
    query_result = ns.query(
        rank_by=["text", "BM25", query],
        top_k=top_k,
        include_attributes=["text", "doc_name", "doc_period"],
    ).rows
    results = []
    # logger.info(query_result)
    for row in query_result:
        results.append(row.attributes["text"])
    return "\n\n".join(results)
    # logger.info(text_results)
