import os
import pymupdf
from pathlib import Path
from typing import cast


def _env_bool(name, default):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _is_text_poor(text: str, minimum_characters: int) -> bool:
    return len(text.strip()) < minimum_characters


def extract_pdf_pages(
    pdf_path: str | Path,
    *,
    ocr_enabled: bool | None = None,
    ocr_language: str | None = None,
    ocr_dpi: int | None = None,
    min_text_chars: int | None = None
):
    """
    Extract text from each PDF page while retaining page metadata.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if ocr_enabled is None:
        ocr_enabled = _env_bool("OCR_ENABLED", False)

    ocr_language = ocr_language or os.getenv("OCR_LANGUAGE", "eng")
    ocr_dpi = ocr_dpi or int(os.getenv("OCR_DPI", "300"))
    min_text_chars = (
        min_text_chars
        if min_text_chars is not None
        else int(os.getenv("OCR_MIN_TEXT_CHARS", "40"))
    )

    if ocr_dpi <= 0:
        raise ValueError("ocr_dpi must be greater than zero.")

    if min_text_chars < 0:
        raise ValueError("min_text_chars cannot be negative.")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1

            text = cast(str, page.get_text("text"))
            ocr_used = False

            if ocr_enabled and _is_text_poor(text, min_text_chars):
                try:
                    text_page = page.get_textpage_ocr(
                        language=ocr_language,
                        dpi=ocr_dpi,
                        full=True
                    )
                    ocr_text = cast(
                        str,
                        page.get_text(
                            "text",
                            textpage=text_page
                        )
                    )
                except Exception as error:
                    raise RuntimeError(
                        "OCR was requested but failed. Install Tesseract OCR, "
                        "ensure its executable is available to PyMuPDF, and "
                        f"verify the language data for {ocr_language!r}."
                    ) from error

                if not _is_text_poor(ocr_text, 1):
                    text = ocr_text
                    ocr_used = True

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "ocr_used": ocr_used
                }
            )

    return pages