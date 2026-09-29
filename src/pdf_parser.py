import fitz
from pathlib import Path


def extract_pdf_pages(pdf_path: str):
    """
    Extract text from each PDF page while retaining page metadata.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append(
            {
                "page_number": page_number,
                "text": text
            }
        )

    document.close()

    return pages