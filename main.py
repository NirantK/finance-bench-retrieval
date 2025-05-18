import uuid
from datetime import datetime

from src.ingest import fastembedding, write_to_turbopuffer
from src.prepare import prepare


def main():
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    hour = now.hour
    prepare(
        pdf_path="pdfs",
        markdown_path="data/markdown",
        images_path="data/images",
        chunks_path="data/chunks",
        document_info_path="data/ground_truth/financebench_document_information.jsonl",
    )
    write_to_turbopuffer(
        embedding_function=fastembedding,
        annotated_chunks_filepath="data/processed/annotated_chunks.jsonl",
        namespace=f"nirantk-fastembedding-{date}-{hour}-{uuid.uuid4()}",
    )


if __name__ == "__main__":
    main()
