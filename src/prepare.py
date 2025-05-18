from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from loguru import logger
from markitdown import MarkItDown
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4.1")


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


def prepare(pdf_path: Path, markdown_path: Path, images_path: Path):
    pdf_path, markdown_path, images_path = (
        Path(pdf_path),
        Path(markdown_path),
        Path(images_path),
    )
    markdown_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    with tqdm(pdf_files, desc="Processing PDFs", unit="file") as pbar:
        for file in pbar:
            pbar.set_postfix(file=file.name, refresh=False)
            prepare_markdown(file, markdown_path)
            prepare_images(file, images_path)
