import json
from pathlib import Path

import pandas as pd
import pymupdf
from chonkie import RecursiveChunker
from dotenv import load_dotenv
from loguru import logger
from markitdown import MarkItDown
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4.1")

chunker = RecursiveChunker(
    chunk_size=512,
    min_characters_per_chunk=100,
    return_type="texts",
).from_recipe("markdown", lang="en")


def prepare_markdown(file: Path, markdown_path: Path):
    output_path = markdown_path / f"{file.stem}.md"
    if not output_path.exists():
        try:
            result = md.convert(file)
        except Exception as e:
            logger.error(f"Error converting {file}: {e}. Skipping!")
            return
        with open(output_path, "w") as f:
            f.write(result.text_content)


def prepare_chunks(file: Path, chunks_path: Path, markdown_path: Path):
    output_path = chunks_path / f"{file.stem}.json"
    if not output_path.exists():
        with open(output_path, "w") as f:
            markdown_file_path = Path(markdown_path / f"{file.stem}.md")
            text = open(markdown_file_path).read()
            chunk_texts = [chunk.text for chunk in chunker(text)]
            f.write(json.dumps(chunk_texts, indent=4))


def prepare_images(file: Path, images_path: Path, dpi: int = 300):
    """
    Split a PDF into images—one PNG per page—using PyMuPDF.
    Pages are rendered at the specified DPI (default 300).
    """
    output_path = Path(images_path / f"{file.stem}")

    # Skip work if images already exist
    if output_path.exists() and any(output_path.iterdir()):
        # logger.debug(f"Images already exist for {file}, skipping.")
        return

    # logger.info(f"Converting {file} to images and saving to {output_path}")
    zoom = dpi / 72  # 72 DPI is the native resolution for PDF points
    mat = pymupdf.Matrix(zoom, zoom)

    with pymupdf.open(file) as doc:
        output_path.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(output_path / f"{i}.png")


def annotate_chunks(chunks_path: Path, document_info_path: Path, output_path: Path):
    document_info = pd.read_json(document_info_path, lines=True)
    if output_path.exists():
        logger.info(f"Skipping {output_path} because it already exists")
        return
    chunks = []
    for file in chunks_path.rglob("*.json"):
        with open(file) as f:
            try:
                texts = json.loads(f.read())
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing {file}: {e}. Skipping!")
                continue
            chunks.extend(
                [
                    {
                        "doc_name": file.stem,
                        "text": text,
                        "chunk_index": index,
                    }
                    for index, text in enumerate(texts)
                ]
            )

    # logger.debug(type(chunks[0]))
    df_chunks = pd.DataFrame(chunks)
    # logger.debug(df_chunks.columns)
    # logger.debug(df_chunks.head())
    # logger.debug(document_info.head())
    df_chunks_merged = pd.merge(df_chunks, document_info, on="doc_name")
    # logger.debug(df_chunks_merged.head())
    df_chunks_merged.to_json(output_path, orient="records", lines=True)


def prepare(
    pdf_path: Path,
    markdown_path: Path,
    images_path: Path,
    chunks_path: Path,
    document_info_path: Path,
):
    pdf_path, markdown_path, images_path, chunks_path, document_info_path = (
        Path(pdf_path),
        Path(markdown_path),
        Path(images_path),
        Path(chunks_path),
        Path(document_info_path),
    )
    markdown_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    chunks_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    with tqdm(pdf_files, desc="Processing PDFs", unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix(file=file.name, refresh=False)
            prepare_markdown(file, markdown_path)
            prepare_images(file, images_path)
            prepare_chunks(file, chunks_path, markdown_path)
        # logger.info("Annotating chunks...")
        annotate_chunks(
            chunks_path=chunks_path,
            document_info_path=document_info_path,
            output_path=Path("data/processed/annotated_chunks.jsonl"),
        )
