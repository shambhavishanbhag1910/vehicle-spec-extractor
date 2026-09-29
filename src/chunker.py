import re


def detect_section(text: str):

    match = re.search(
        r"SECTION\s+([0-9A-Z\-]+):\s*([^\n]+)",
        text,
        re.IGNORECASE
    )

    if match:
        return {
            "section_id": match.group(1).strip(),
            "section_title": match.group(2).strip()
        }

    return {
        "section_id": None,
        "section_title": None
    }


def chunk_pages(
    pages,
    chunk_size=1200,
    overlap=200
):

    chunks = []

    chunk_id = 0

    for page in pages:

        text = page["text"]

        section_info = detect_section(text)

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = text[start:end]

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page["page_number"],
                    "section_id":
                        section_info["section_id"],
                    "section_title":
                        section_info["section_title"],
                    "text": chunk_text
                }
            )

            chunk_id += 1

            if end == len(text):
                break

            start = end - overlap

    return chunks