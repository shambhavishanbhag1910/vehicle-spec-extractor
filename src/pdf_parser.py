import pymupdf
from pathlib import Path


def extract_pdf_pages(pdf_path: str | Path):
    """
    Extract text from each PDF page while retaining page metadata.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1

            text = page.get_text("text")

            pages.append(
                {
                    "page_number": page_number,
                    "text": text
                }
            )

    return pages