import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from src.agentic_rag import rag_dag
from src.evaluate import evaluate_rag
from src.ingest import to_turbopuffer
from src.search_tools import hybrid_search
from src.simple_rag import rag_chain

load_dotenv()

tqdm.pandas()


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
@click.option("--top-k", default=10, help="Number of results to return")
def search(query, top_k):
    """Search documents using vector search"""
    results = hybrid_search(
        query=query,
        top_k=top_k,
    )
    logger.info(results)


@cli.command()
@click.option("--query", prompt="Enter your question", help="RAG query")
def rag(query):
    """Run a simple RAG workflow"""
    response = rag_chain(query)
    logger.info(response)


@cli.command()
@click.option("--query", prompt="Enter your question", help="Agentic RAG query")
def agentic_rag(query):
    """Run a simple RAG workflow"""
    response = rag_dag(query)
    logger.info(response)


@cli.command()
@click.option(
    "--data-filepath", default="data/ground_truth/financebench_open_source.jsonl", help="Path to ground truth data"
)
@click.option("--output-datapath", default="data/evals/", help="Path to output data")
@click.option("--eval-setup", default="simple", help="Evaluation setup")
@click.option("--head_n", default=None, help="Number of results to evaluate")
@click.option("--recursion_limit", default=25, help="Recursion limit for agentic RAG")
def evaluate(data_filepath, output_datapath, eval_setup, head_n, recursion_limit):
    """Evaluate the RAG workflow"""
    df = pd.read_json(data_filepath, lines=True)
    if head_n:
        df = df.head(head_n)
    def process_query(row):
        query = row["question"]
        if eval_setup == "simple":
            generated_answer = rag_chain(query)
        elif eval_setup == "agentic":
            generated_answer = rag_dag(query, recursion_limit=recursion_limit)
        else:
            raise ValueError(f"Invalid evaluation setup: {eval_setup}")
        ground_truth = row["answer"]
        score, reasoning = evaluate_rag(query, generated_answer, ground_truth)
        return pd.Series([generated_answer, score, reasoning])

    results = df.progress_apply(process_query, axis=1)
    results.columns = ["generated_answer", "score", "judge_reasoning"]
    results = pd.concat([df, results], axis=1)
    output_datapath = Path(output_datapath) / f"{eval_setup}_results.jsonl"
    output_datapath.parent.mkdir(parents=True, exist_ok=True)
    results.to_json(output_datapath, orient="records", lines=True)


def main():
    cli()


if __name__ == "__main__":
    main()
