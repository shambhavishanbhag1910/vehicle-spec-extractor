import os
import re
from groq import Groq
from pydantic import ValidationError

from src.schemas import ExtractionResponse


SYSTEM_PROMPT = """
You are a precise automotive service manual specification extraction system.

Use ONLY the supplied SERVICE MANUAL CONTEXT.
Never use outside knowledge.
Never guess or infer a value that is not directly supported.

IMPORTANT: The retrieved context may contain the same component more than once
with different part numbers or specifications because the component appears in
different tables, illustrations, procedures, or applicability blocks.

You MUST first determine the exact source context requested by the user,
and only then extract the value.

SOURCE SELECTION RULES

1. Treat source wording in the user's query as a HARD CONSTRAINT.

Examples include:
- parts illustration
- illustrated parts table
- illustration
- All Vehicles
- removal and installation parts list
- specifications table
- torque specifications
- a specific procedure
- a specific vehicle configuration

2. If the query asks for a "parts illustration",
"illustrated parts table", or "illustration":

   - Use ONLY the parts table associated with that illustration.
   - If the retrieved context also contains a later "All Vehicles" heading,
     DO NOT use any value appearing under the "All Vehicles" heading.
   - A part number from an All Vehicles procedure is NOT a valid answer
     to a parts illustration question.
   - Stop considering other occurrences once the matching illustrated
     parts table has been identified.

3. If the query explicitly asks for a part number from
"All Vehicles":

   - Use ONLY the part number appearing in the content governed by the
     "All Vehicles" heading.
   - Do NOT use a part number from a preceding illustration or illustrated
     parts table.
   - A value appearing before the All Vehicles heading is not a valid answer
     unless the query explicitly asks for the illustration.

4. Do not combine values across source boundaries.

The following are separate source contexts:
- illustrated parts table
- All Vehicles parts list
- removal procedure
- installation procedure
- specification table
- different vehicle configuration
- different page or subsection when the context clearly changes

5. When the same component appears more than once:

   FIRST identify which occurrence belongs to the source requested
   by the user.

   THEN ignore all occurrences belonging to other source contexts.

   Do NOT choose a value merely because:
   - it appears later,
   - it has more surrounding text,
   - it appears on an adjacent page,
   - or it has the same component name.

6. For a source-specific query, return exactly ONE result unless
the user explicitly requests multiple results or a comparison.

7. If you cannot determine which occurrence belongs to the exact
source requested by the user, return:

{
    "status": "not_found",
    "results": []
}

Never fall back to a value from a different source context.

VALUE RULES

8. Preserve exact component names, numeric values, units, and part numbers
from the selected source.

9. For part-number questions:
   - populate "part_number"
   - set "value" to null
   - set "unit" to null
   - do not put a part number in "value"

10. Evidence MUST be a verbatim quote from the SAME selected source
that supports the returned result.

11. The output page number MUST be the page containing the exact
source used for the answer.

12. Preserve vehicle configuration when explicitly stated.

13. If the requested answer does not exist in the requested source,
return "not_found". Do not substitute another occurrence.

14. When SOURCE_CONTEXT labels are supplied, treat those labels as
structural metadata for source selection.

15. For a parts illustration or illustrated parts table query, use only
PARTS_TABLE_WITHOUT_ALL_VEHICLES_HEADING.

16. For an All Vehicles query, use only ALL_VEHICLES_CONTEXT.
OUTPUT

Return valid JSON only.

Successful response:

{
    "status": "found",
    "results": [
        {
            "component": "",
            "spec_type": "",
            "value": null,
            "unit": null,
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

Unsupported response:

{
    "status": "not_found",
    "results": []
}

Do not output Markdown.
Do not provide explanations outside the JSON.
"""

def identify_source_context(text: str) -> str:

    normalized = re.sub(
        r"\s+",
        " ",
        text
    ).casefold()

    has_parts_table = (
        "item part number description"
        in normalized
    )

    has_all_vehicles = (
        "all vehicles"
        in normalized
    )

    if has_all_vehicles:
        return "ALL_VEHICLES_CONTEXT"

    if has_parts_table:
        return "PARTS_TABLE_WITHOUT_ALL_VEHICLES_HEADING"

    return "GENERAL_CONTEXT"

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

            source_context = identify_source_context(
                chunk["text"]
            )

            context_parts.append(
                f"""
        ======== SOURCE BLOCK ========

        PAGE: {chunk['page_number']}
        SECTION: {chunk.get('section_id')}
        SOURCE_CONTEXT: {source_context}

        CONTENT:
        {chunk['text']}

        ======== END SOURCE BLOCK ========

"""
            )

        context = "\n\n".join(
            context_parts
        )

        user_prompt = f"""
USER QUERY:

{query}

SOURCE SELECTION RULE:

The retrieved material is divided into SOURCE BLOCKS.

Each block contains a SOURCE_CONTEXT label.

If the user asks for:

"parts illustration",
"illustrated parts table",
or "illustration"

then use:

SOURCE_CONTEXT:
PARTS_TABLE_WITHOUT_ALL_VEHICLES_HEADING

and reject:

SOURCE_CONTEXT:
ALL_VEHICLES_CONTEXT


If the user asks specifically for:

"All Vehicles"

then use:

SOURCE_CONTEXT:
ALL_VEHICLES_CONTEXT

and reject values from:

PARTS_TABLE_WITHOUT_ALL_VEHICLES_HEADING


The same component may appear in both contexts with
different part numbers.

Do not merge them.
Do not substitute one for the other.

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

        for specification in result.results:
            if specification.vehicle is None:
                specification.vehicle = "2014 F-150"

        return result.model_dump()