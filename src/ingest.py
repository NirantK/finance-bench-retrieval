import os
import uuid
from itertools import batched

import joblib
import pandas as pd
import turbopuffer as tpuf
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"


def to_turbopuffer(
    annotated_chunks_filepath: str,
    namespace: str,
    vectors_filepath: str,
):
    ns = tpuf.Namespace(namespace)
    annotated_chunks = pd.read_json(annotated_chunks_filepath, lines=True)
    # Take all the columns from annotated_chunks and add them to the upsert_columns
    chunks_columns = {
        col: annotated_chunks[col].tolist() for col in annotated_chunks.columns
    }
    ids = [
        str(uuid.uuid4()) for _ in range(len(annotated_chunks))
    ]  # Convert UUIDs to strings
    vectors = joblib.load(vectors_filepath)
    if len(vectors) != len(ids) or len(vectors) != len(annotated_chunks):
        logger.error(
            f"Vectors: {len(vectors)}, ids: {len(ids)}, annotated_chunks: {len(annotated_chunks)}"
        )
        raise ValueError("Vectors, ids, and annotated_chunks must have the same length")
    logger.debug(f"Vector type: {type(vectors)}, shape: {vectors[0].shape}")
    # Define schema for the data
    schema = {}
    logger.info(f"Columns: {annotated_chunks.columns}")
    # annotated_chunks.drop(columns=["id"], inplace=True)
    # Add schema for text fields to enable full-text search
    for col in annotated_chunks.columns:
        if annotated_chunks[col].dtype == "object":  # String/text columns
            schema[col] = {
                "type": "string",
                "full_text_search": True,  # Enable full-text search for text fields
            }
        elif annotated_chunks[col].dtype in ["int64", "float64"]:  # Numeric columns
            schema[col] = {"type": "int", "filterable": True}

    # Validate schema against data structure
    def validate_schema(data, schema):
        for col, col_data in data.items():
            if col not in schema:
                raise ValueError(f"Column '{col}' exists in data but not in schema")
            if schema[col]["type"] == "string" and not all(
                isinstance(x, str) for x in col_data
            ):
                raise ValueError(
                    f"Column '{col}' is defined as string but contains non-string values"
                )
            if schema[col]["type"] == "int" and not all(
                isinstance(x, (int, float)) for x in col_data
            ):
                raise ValueError(
                    f"Column '{col}' is defined as int but contains non-numeric values"
                )

    validate_schema(annotated_chunks, schema)
    logger.info("Schema validation passed")

    # Upsert the data with vectors
    for batch in batched(zip(ids, chunks_columns, vectors), 50_000):
        ns.write(
            upsert_columns={
                "id": [id for id, _, _ in batch],
                **{k: [v for _, v, _ in batch] for k in chunks_columns},
                "vector": [vector for _, _, vector in batch],
            },
            # schema=schema,
            distance_metric="cosine_distance",
        )
    # Get and log the schema
    try:
        current_schema = ns.schema()
        logger.info(f"Current namespace schema: {current_schema}")
    except Exception as e:
        logger.warning(f"Could not retrieve schema: {str(e)}")

    logger.info(
        f"Successfully ingested {len(vectors)} records into namespace '{namespace}'"
    )
