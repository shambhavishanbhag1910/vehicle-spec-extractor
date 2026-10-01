# Vehicle Specification Extractor

A compact RAG based application for extracting structured vehicle specifications from the supplied **2014 Ford F-150 Workshop Manual**.

It supports queries for torque values, fluid capacities, part numbers, ride height procedures, and other service specifications, and returns structured JSON with page, section, and evidence.

## Architecture

```text
PDF
 -> PyMuPDF text extraction
 -> Cleaning
 -> Section aware chunking
 -> Sentence Transformers embeddings
 -> FAISS semantic search
 +  BM25 lexical search
 -> Weighted rank fusion
 -> Relevant context
 -> Groq LLM
 -> Pydantic validation
 -> Structured JSON
```

### Main design choices

- **Chunking:** section aware, up to 1200 characters with 200 character overlap
- **Embeddings:** `all-MiniLM-L6-v2`
- **Vector store:** FAISS `IndexFlatIP`
- **Lexical retrieval:** BM25
- **Retrieval:** hybrid FAISS + BM25 using weighted Reciprocal Rank Fusion
- **LLM:** Groq hosted model
- **Validation:** Pydantic
- **Interfaces:** CLI and Streamlit
- **Optional OCR:** Tesseract fallback for text poor pages

## Project Structure

```text
vehicle-spec-extractor/
|-- data/
|   `-- sample-service-manual.pdf
|-- src/
|   |-- pdf_parser.py
|   |-- cleaner.py
|   |-- chunker.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- retriever.py
|   |-- extractor.py
|   |-- schemas.py
|   `-- pipeline.py
|-- evaluation/
|   |-- cases.json
|   |-- extraction_cases.json
|   |-- run_retrieval_eval.py
|   |-- run_extraction_eval.py
|   `-- extraction_report_final.json
|-- tests/
|   `-- test_components.py
|-- main.py
|-- app.py
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your Groq key in `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
OCR_ENABLED=false
```

## Run

### CLI

```powershell
python main.py
```

### Streamlit

```powershell
streamlit run app.py
```

## Example Queries

```text
What is the torque specification for the front brake caliper anchor plate bolts? Include both listed units.
```

```text
What is the specified fill capacity for High Performance DOT 3 Motor Vehicle Brake Fluid in the front disc brake section? Include the alternate unit if listed.
```

```text
In the rear suspension shock absorber parts illustration, what part number is listed for the shock absorber? Use only the illustrated parts table.
```

```text
What is the towing capacity for a 2024 Toyota Camry?
```

Unsupported questions return:

```json
{
  "status": "not_found",
  "results": []
}
```

## Testing

### Unit tests

```powershell
python -m unittest discover -s tests -v
```

Result:

```text
20 / 20 PASS
```

### Retrieval evaluation

```powershell
python -m evaluation.run_retrieval_eval --top-k 5
```

Result:

```text
Recall@5: 7/7 (100.0%)
```

### End to end extraction evaluation

```powershell
python -m evaluation.run_extraction_eval --call-llm --top-k 5 --output evaluation/extraction_report_final.json
```

Final result:

```text
Exact case accuracy: 5/5 (100.0%)

PASS front-brake-caliper-anchor-bolt-torque
PASS front-brake-fluid-fill-capacity
PASS rear-shock-parts-illustration-number
PASS rear-shock-all-vehicles-procedure-number
PASS unsupported-vehicle-not-found
```

Final metrics:

```text
Unit tests:               20/20 PASS
Retrieval Recall@5:       100%
End to end cases:         5/5 PASS
Exact case accuracy:      100%
Evidence grounding rate:  100%
Relevant page recall:     100%
```

## Notes

- The full manual is not sent to the LLM. Only retrieved chunks are passed to Groq.
- Page and section metadata are preserved for traceability.
- Source context handling distinguishes nearby tables or procedures when the same component appears more than once.
- The FAISS index is currently rebuilt on startup.
- Chunking is text based rather than fully table aware.
- OCR is optional and requires Tesseract.
- The evaluation set is intentionally small and is intended for assignment level validation, not production benchmarking.

## Security

Keep the real Groq API key only in `.env`.

`.env` is excluded by `.gitignore`.
