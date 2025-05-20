import os

import joblib
import pandas as pd
import turbopuffer as tpuf
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"


def chunk_dataframe(df, chunk_size):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]


def to_turbopuffer(
    annotated_chunks_filepath: str,
    namespace: str,
    vectors_filepath: str,
):
    ns = tpuf.Namespace(namespace)
    annotated_chunks = pd.read_json(annotated_chunks_filepath, lines=True)
    annotated_chunks["id"] = [str(i) for i in range(len(annotated_chunks))]
    annotated_chunks["vector"] = joblib.load(vectors_filepath)
    # Take all the columns from annotated_chunks and add them to the upsert_columns
    if len(annotated_chunks) != len(annotated_chunks["vector"]) or len(annotated_chunks) != len(annotated_chunks["id"]):
        logger.error(
            f"Vectors: {len(annotated_chunks['vector'])}, ids: {len(annotated_chunks['id'])}, annotated_chunks: {len(annotated_chunks)}"
        )
        raise ValueError("Vectors, ids, and annotated_chunks must have the same length")
    logger.debug(f"Vector type: {type(annotated_chunks['vector'][0])}, shape: {annotated_chunks['vector'][0].shape}")
    logger.info(f"Columns: {annotated_chunks.columns}")
    # logger.info(f"Found {len(chunks_columns)} columns")
    # # Upsert the data with vectors
    chunks = list(chunk_dataframe(annotated_chunks, 50_000))  # Creates chunks of 100 rows each
    for chunk in chunks:
        # logger.info(f"Chunk: {chunk}")
        columns_batch = chunk.to_dict(orient="list")
        logger.info(f"Upserting {type(columns_batch)}, {len(columns_batch)}")
        ns.write(
            upsert_columns=columns_batch,
            schema={
                "text": {
                    "type": "string",
                    "full_text_search": True,
                },
                "doc_name": {
                    "type": "string",
                    "full_text_search": True,
                },
            },
            distance_metric="cosine_distance",
        )
    # # Get and log the schema
    logger.info(ns.schema())
