import os
from itertools import batched
from typing import Callable, List

import pandas as pd
import turbopuffer as tpuf
from dotenv import load_dotenv
from fastembed import TextEmbedding
from loguru import logger
from openai import OpenAI

load_dotenv()
client = OpenAI()

tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"


def openai_embedding(texts: List[str], model: str = "text-embedding-3-large"):
    texts = [text.replace("\n", " ") for text in texts]
    embeddings = []
    for batch in batched(texts, 2):
        responses = client.embeddings.create(input=batch, model=model)
        embeddings.extend([response.embedding for response in responses.data])
    return embeddings


def fastembedding(
    texts: List[str], model: str = "mixedbread-ai/mxbai-embed-large-v1"
) -> List[List[float]]:
    fst = TextEmbedding(model=model)
    return list(fst.embed(texts))


def write_to_turbopuffer(
    embedding_function: Callable[[List[str]], List[List[float]]],
    annotated_chunks_filepath: str,
    namespace: str,
):
    ns = tpuf.Namespace(namespace)
    annotated_chunks = pd.read_json(annotated_chunks_filepath, lines=True)

    vectors = embedding_function(annotated_chunks["text"].tolist())
    logger.info(f"Writing {len(vectors)} vectors to Turbopuffer")
    ids = [
        f"{row['doc_name']}_{row['chunk_index']}"
        for row in annotated_chunks.itertuples()
    ]

    # Take all the columns from annotated_chunks and add them to the upsert_columns
    columns = annotated_chunks.to_dict(orient="records")
    ns.write(
        upsert_columns={
            "id": ids,
            "vector": vectors,
            **columns,
        },
        distance_metric="cosine_distance",
    )
