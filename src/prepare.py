from pathlib import Path

from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI
from tqdm.auto import tqdm

load_dotenv()

client = OpenAI()
md = MarkItDown(llm_client=client, llm_model="gpt-4.1")

def prepare(pdf_path: Path, markdown_path: Path):
    for file in tqdm(pdf_path.glob("*.pdf"), desc="Converting PDFs to Markdown"):
        result = md.convert(file)
        with open(f"{markdown_path}/{file.stem}.md", "w") as f:
            f.write(result.text_content)