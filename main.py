import os

from dotenv import load_dotenv

from src.ingest import to_turbopuffer

load_dotenv()


def main():
    namespace = os.getenv("TURBOPUFFER_NAMESPACE")
    # prepare(
    #     pdf_path="pdfs",
    #     markdown_path="data/markdown",
    #     images_path="data/images",
    #     chunks_path="data/chunks",
    #     document_info_path="data/ground_truth/financebench_document_information.jsonl",
    # )
    to_turbopuffer(
        annotated_chunks_filepath="data/processed/annotated_chunks.jsonl",
        namespace=namespace,
        vectors_filepath="data/vectors/214.pkl",
    )
    # results = bm25_search(
    #     query="What were 3M revenue and net income in 2024?",
    #     top_k=10,
    #     include_attributes=["text", "doc_name"],
    # )
    # logger.info(results)


if __name__ == "__main__":
    main()
