import os

from groq import Groq
from pydantic import ValidationError

from src.schemas import ExtractionResponse


SYSTEM_PROMPT = """
You are a technical specification extraction system.

Extract information only from the supplied
automotive service manual context.

Rules:

1. Do not use outside knowledge.
2. Do not invent missing specifications.
3. Preserve exact numeric values.
4. Preserve exact measurement units.
5. Preserve vehicle configuration when given.
6. If multiple valid specifications exist,
   return all of them.
    Return only specifications directly requested by the query. Do not include
    nearby values for the same component from a different procedure or context
    unless the query asks for alternatives or a comparison.
7. If no answer exists in the context,
   return status = "not_found".
8. Include the supporting source text.
    Evidence must be a verbatim quote from the supplied context.
9. Return valid JSON only.

Return one JSON object matching this schema. Use status "not_found" and an empty
results list when the supplied context does not support an answer. For part-number
questions, populate part_number and do not place a part number in a measurement value.
Do not add Markdown fences or fields outside this schema.

{
    "status": "found",
    "results": [
        {
            "component": "",
            "spec_type": "",
            "value": "",
            "unit": "",
            "part_number": null,
            "alternate_value": null,
            "alternate_unit": null,
            "vehicle": "2014 F-150",
            "configuration": null,
            "section": null,
            "page": null,
            "evidence": ""
        }
    ]
}
"""


class GroqExtractor:

    def __init__(
        self,
        model=None
    ):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )


    def extract(
        self,
        query,
        retrieved_chunks
    ):

        context_parts = []

        for chunk in retrieved_chunks:

            context_parts.append(
                f"""
PAGE: {chunk['page_number']}
SECTION: {chunk.get('section_id')}

CONTENT:
{chunk['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        user_prompt = f"""
USER QUERY:

{query}

SERVICE MANUAL CONTEXT:

{context}
"""

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
        )

        output = (
            response.choices[0]
            .message.content
        )

        if output is None:
            raise ValueError(
                "Groq returned an empty response."
            )

        try:
            result = ExtractionResponse.model_validate_json(output)
        except ValidationError as error:
            raise ValueError(
                "Groq returned JSON that does not match the extraction schema."
            ) from error

        return result.model_dump()