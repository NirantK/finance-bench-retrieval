import os

import click
from dotenv import load_dotenv
from loguru import logger

from src.ingest import to_turbopuffer
from src.search_tools import hybrid_search

load_dotenv()


@click.group()
def cli():
    """CLI for document search and ingestion"""
    pass


@cli.command()
@click.option("--chunks-file", default="data/processed/annotated_chunks.jsonl", help="Path to annotated chunks file")
@click.option("--vectors-file", default="data/vectors/214.pkl", help="Path to vectors file")
def ingest(chunks_file, vectors_file):
    """Ingest documents into Turbopuffer"""
    to_turbopuffer(
        annotated_chunks_filepath=chunks_file,
        namespace=os.getenv("TURBOPUFFER_NAMESPACE"),
        vectors_filepath=vectors_file,
    )


@cli.command()
@click.option("--query", prompt="Enter your search query", help="Search query")
@click.option("--top-k", default=3, help="Number of results to return")
def search(query, top_k):
    """Search documents using vector search"""
    results = hybrid_search(
        query=query,
        top_k=top_k,
    )
    logger.info(results)


def main():
    cli()


if __name__ == "__main__":
    main()
