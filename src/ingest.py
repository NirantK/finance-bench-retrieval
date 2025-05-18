import os
import uuid

import turbopuffer as tpuf

tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"
ns = tpuf.Namespace(f"namespace-a-py-{uuid.uuid4()}")

def openai_or_rand_vector(text: str) -> list[float]:
    if not os.getenv("COHERE_API_KEY"):
        print("COHERE_API_KEY not set, using random vectors")
        return [__import__("random").random()] * 2
    try:
        return (
            __import__("cohere")
            .embed(model="embed-english-v3.0", input=text)
            .data[0]
            .embedding
        )
    except ImportError:
        return [__import__("random").random()] * 2


# Upsert documents with vectors and attributes
ns.write(
    upsert_columns={
        "id": [1, 2],
        "vector": [
            openai_or_rand_vector("walrus narwhal"),
            openai_or_rand_vector("elephant walrus rhino"),
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

# Based on data/sample.json, the schema is:
schema = {
    "financebench_id": {
        "type": "string",
        "full_text_search": True,
    },
    "company_x": {
        "type": "string",
    },
    "question_type": {
        "type": "string",
    },
    "evidence_text": {
        "type": "string",
        "full_text_search": True,
        "vector_search": True,
    },
    "evidence_doc_name": {
        "type": "string",
    },
    "evidence_page_num": {
        "type": "int",
    },
    "evidence_text_full_page": {
        "type": "string",
        "full_text_search": True,
        "vector_search": True,
    },
    "gics_sector": {
        "type": "string",
        "full_text_search": True,
        "vector_search": True,
    },
    "doc_period": {
        "type": "int",
    },
    "doc_type": {
        "type": "string",
    },
    "doc_link": {
        "type": "string",
    },
}


# # Query nearest neighbors with filter
# print(
#     ns.query(
#         rank_by=["vector", "ANN", openai_or_rand_vector("walrus narwhal")],
#         top_k=10,
#         filters=["And", [["name", "Eq", "foo"], ["public", "Eq", 1]]],
#         include_attributes=["name"],
#     )
# )
# # [VectorRow(id=1, vector=None, attributes={'name': 'foo'}, dist=0.009067952632904053)]

# # Full-text search on an attribute
# # If you want to combine FTS and vector search, see 
# print(
#     ns.query(
#         top_k=10,
#         filters=["name", "Eq", "foo"],
#         rank_by=["text", "BM25", "quick walrus"],
#     )
# )
# # [VectorRow(id=1, vector=None, attributes={'name': 'foo'}, dist=0.19)]
# # [VectorRow(id=2, vector=None, attributes={'name': 'foo'}, dist=0.168)]

# # Vectors can be updated by passing new data for an existing ID
# ns.write(
#     upsert_columns={
#         "id": [1, 2, 3],
#         "vector": [
#             openai_or_rand_vector("foo"),
#             openai_or_rand_vector("foo"),
#             openai_or_rand_vector("foo"),
#         ],
#         "name": ["foo", "foo", "foo"],
#         "public": [1, 1, 1],
#     },
#     distance_metric="cosine_distance",
# )
# # Vectors are deleted by ID. Under the hood,
# # this upserts with the `vector` set to `null`
# ns.write(deletes=[1, 3])


# print(df_full.head())
