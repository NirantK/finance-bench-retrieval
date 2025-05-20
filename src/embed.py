from itertools import batched
from pathlib import Path
from typing import Callable, List

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI()


def openai_embedding(texts: List[str], model: str = "text-embedding-3-large"):
    texts = [text.replace("\n", " ") for text in texts]
    embeddings = []
    for batch in batched(texts, 2):
        responses = client.embeddings.create(input=batch, model=model)
        embeddings.extend([response.embedding for response in responses.data])
    return embeddings


def fastembedding(
    texts: List[str], model: str = "snowflake/snowflake-arctic-embed-xs"
) -> List[List[float]]:
    fst = TextEmbedding(model=model)
    vectors = []
    batch_size = 512
    for index, batch in tqdm(
        enumerate(batched(texts, batch_size)),
        desc=f"Embedding texts with {model}",
        total=len(texts) // batch_size,
        unit="batch",
    ):
        vectors.extend(list(fst.embed(batch)))
        # Make sure the directory exists
        vectors_path = Path("data/vectors")
        vectors_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectors, vectors_path / f"{index}.pkl")
    return vectors


def get_vectors(
    annotated_chunks_filepath: str,
    embedding_function: Callable[[List[str]], List[List[float]]],
) -> tuple[pd.DataFrame, List[List[float]]]:
    annotated_chunks = pd.read_json(annotated_chunks_filepath, lines=True)
    vectors = embedding_function(annotated_chunks["text"].tolist())
    logger.info(f"Got {len(vectors)} vectors")

    return annotated_chunks, vectors
