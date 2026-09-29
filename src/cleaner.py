import re


def clean_text(text: str) -> str:

    # Remove local file URI
    text = re.sub(r"file:///C:/[^\n]+", " ", text)

    # Normalize bullet artifacts
    text = text.replace("z ", "• ")

    # Normalize broken whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(pages):

    cleaned = []

    for page in pages:

        cleaned.append(
            {
                **page,
                "text": clean_text(page["text"])
            }
        )

    return cleaned