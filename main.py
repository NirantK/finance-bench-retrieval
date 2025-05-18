from src.prepare import prepare


def main():
    prepare(
        pdf_path="pdfs",
        markdown_path="data/markdown",
        images_path="data/images",
        chunks_path="data/chunks",
    )


if __name__ == "__main__":
    main()
