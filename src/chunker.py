import re


SECTION_PATTERN = re.compile(
    r"SECTION\s+([0-9A-Z\-]+):\s*([^\n]+)",
    re.IGNORECASE
)


def detect_section(text: str):

    match = SECTION_PATTERN.search(text)

    if match:
        return {
            "section_id": match.group(1).strip(),
            "section_title": match.group(2).strip()
        }

    return {
        "section_id": None,
        "section_title": None
    }


def _split_page_sections(text, section_info):
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        yield text, section_info
        return

    start = 0

    for match in matches:
        if match.start() > start:
            yield text[start:match.start()], section_info

        section_info = {
            "section_id": match.group(1).strip(),
            "section_title": match.group(2).strip()
        }
        start = match.start()

    if start < len(text):
        yield text[start:], section_info


def _find_chunk_end(text, start, chunk_size):
    limit = min(start + chunk_size, len(text))

    if limit == len(text):
        return limit

    minimum = start + int(chunk_size * 0.6)

    for delimiter in ("\n\n", "\n", ". ", "; "):
        boundary = text.rfind(delimiter, minimum, limit)

        if boundary >= minimum:
            return boundary + len(delimiter)

    return limit


def chunk_pages(
    pages,
    chunk_size=1200,
    overlap=200
):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size.")

    chunks = []

    chunk_id = 0
    current_section = {
        "section_id": None,
        "section_title": None
    }

    for page in pages:

        text = page["text"]

        page_sections = list(
            _split_page_sections(text, current_section)
        )

        if page_sections:
            current_section = page_sections[-1][1]

        for section_text, section_info in page_sections:
            start = 0

            while start < len(section_text):

                end = _find_chunk_end(
                    section_text,
                    start,
                    chunk_size
                )

                chunk_text = section_text[start:end]

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

                if end == len(section_text):
                    break

                start = max(end - overlap, start + 1)

    return chunks