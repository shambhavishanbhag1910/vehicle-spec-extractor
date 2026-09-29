import json
from openai import OpenAI


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
7. If no answer exists in the context,
   return status = "not_found".
8. Include the supporting source text.
9. Return valid JSON only.

Expected JSON format:

{
    "status": "found",
    "results": [
        {
            "component": "",
            "spec_type": "",
            "value": "",
            "unit": "",
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


class OpenAIExtractor:

    def __init__(
        self,
        model="gpt-5-mini"
    ):

        self.client = OpenAI()

        self.model = model


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

        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=user_prompt
        )

        output = response.output_text

        return json.loads(output)