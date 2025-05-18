import os
import uuid
from typing import List

import turbopuffer as tpuf
from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv()
client = OpenAI()


tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"
ns = tpuf.Namespace(f"namespace-a-py-{uuid.uuid4()}")


def openai_embedding(texts: List[str], model: str = "text-embedding-3-large"):
    texts = [text.replace("\n", " ") for text in texts]
    responses = client.embeddings.create(input=texts, model=model)
    return [response.embedding for response in responses.data]


def fastembeddings(
    texts: List[str], model: str = "mixedbread-ai/mxbai-embed-large-v1"
) -> List[List[float]]:
    fst = TextEmbedding(model=model)
    return list(fst.embed(texts))


# Upsert documents with vectors and attributes
ns.write(
    upsert_columns={
        "id": [1, 2],
        "vector": [
            openai_embedding("walrus narwhal"),
            openai_embedding("elephant walrus rhino"),
        ],
        "name": ["foo", "foo"],
        "public": [1, 0],
        "text": ["walrus narwhal", "elephant walrus rhino"],
    },
    distance_metric="cosine_distance",
    schema={
        "text": {  # Configure FTS/BM25, other attribtues have inferred types (name: str, public: int)
            "type": "string",
            # More schema & FTS options https://turbopuffer.com/docs/schema
            "full_text_search": True,
        }
    },
)
