# Vehicle Specification Extractor

A retrieval-augmented application for finding vehicle specifications in the supplied 2014 F-150 Workshop Manual. The system extracts PDF text, retrieves relevant passages, and asks a Groq-hosted language model to return cited structured JSON. Optional OCR can recover text from scanned pages; diagrams and visual interpretation remain out of scope.

## Requirements

- Python 3.10 or newer
- A Groq API key
- The provided manual at `data/sample-service-manual.pdf`
- Optional: Tesseract OCR for scanned/text-poor PDF pages

## Setup

In PowerShell from the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` to a valid key. `GROQ_MODEL` is optional; it defaults to `openai/gpt-oss-120b`. Keep credentials in `.env`, not in source control or `.env.example`.

OCR is disabled by default. To enable OCR fallback for pages with little embedded text, install Tesseract OCR separately and make sure its executable and language data are available to PyMuPDF. On Windows, install Tesseract (including the English language data) and add its installation directory to `PATH`, then set `OCR_ENABLED=true` in `.env`. `OCR_LANGUAGE`, `OCR_DPI`, and `OCR_MIN_TEXT_CHARS` configure the OCR language, rendering resolution, and threshold for deciding a page needs OCR. OCR can slow ingestion substantially. PyMuPDF's OCR interface is included with the Python package, but the Tesseract engine itself is an external system dependency.

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face. The embedding model and the manual's FAISS index are held in memory; the index is rebuilt on each application start.

## Run

Start the Streamlit UI:

```powershell
streamlit run app.py
```

Or use the interactive terminal application:

```powershell
python main.py
```

Ask a question such as “How is rear ride height measured on F-150 models other than the SVT Raptor?” The result is a JSON object with `status` and `results`. Each specification includes its component, type, value and unit when applicable, optional part number/configuration, source page, and supporting evidence.

## Design

1. `src/pdf_parser.py` extracts text page by page with PyMuPDF and retains PDF page numbers. When enabled, it OCRs pages whose embedded text is below the configured threshold and records whether OCR was used.
2. `src/cleaner.py` normalizes whitespace and known line-leading PDF bullet artifacts.
3. `src/chunker.py` creates overlapping chunks, prefers paragraph/line/sentence boundaries, and carries section metadata across pages.
4. `src/embeddings.py` creates normalized document and query embeddings with Sentence Transformers.
5. `src/vector_store.py` indexes vectors in FAISS using inner product, equivalent to cosine similarity for normalized vectors.
6. `src/retriever.py` combines FAISS semantic results with BM25 lexical results using BM25-weighted reciprocal-rank fusion. It adds bounded adjacent-page context from the same section so procedures spanning page breaks remain available to the extractor.
7. `src/extractor.py` sends only the retrieved context and question to Groq in JSON mode, then validates the response with the Pydantic models in `src/schemas.py`.

`app.py` provides the Streamlit interface and caches its in-memory pipeline. `main.py` provides the CLI. The main orchestration lives in `src/pipeline.py`.

## Tests and retrieval evaluation

Run offline unit tests and verify that the evaluation facts still exist in the supplied PDF:

```powershell
python -m unittest discover -s tests -v
```

Run the hybrid retrieval benchmark:

```powershell
python -m evaluation.run_retrieval_eval --top-k 5
```

The benchmark combines `evaluation/cases.json` and the positive cases in `evaluation/extraction_cases.json`, using their expected source pages to report Recall@k across ride-height, torque, fluid-capacity, and part-number questions. The first run may download the embedding model and embed the full manual. This benchmark does not call Groq.

Run the end-to-end gold-answer evaluation with explicit permission to make live Groq requests:

```powershell
python -m evaluation.run_extraction_eval --call-llm --top-k 5 --output evaluation/extraction_report.json
```

This command reads the five cases in `evaluation/extraction_cases.json`, retrieves context, calls Groq once per case, validates each answer, compares expected fields, checks that evidence is quoted from retrieved context, and reports exact-case accuracy plus status, field, evidence, and relevant-page metrics. The part-number cases distinguish the illustration's `18125` from the separate “All Vehicles” procedure list's `18080`; the not-found case tests refusal for a different vehicle. Live evaluation sends API requests and may incur charges; it requires a working `GROQ_API_KEY` in `.env`. The report contains model answers and evidence but never API credentials.

## Limitations and next improvements

- Both benchmarks are intentionally small: retrieval cases focus on ride-height procedures, while four extraction cases cover front brake torque, brake-fluid capacity, a rear shock absorber part number, and not-found behavior. Expand the gold set across more components, configurations, and specification types before drawing broad accuracy conclusions.
- The evaluator checks exact expected fields and whether evidence is quoted from retrieved context, but neither Pydantic nor substring checks can fully prove semantic entailment. Human review remains important for safety-critical specifications.
- Chunk boundaries are improved but remain text-based; complex multi-column tables may not preserve row/column relationships.
- OCR is optional and requires a separately installed Tesseract engine; it is disabled by default and may produce recognition errors on low-quality scans or complex tables.
- Hybrid retrieval uses a fixed BM25-weighted reciprocal-rank fusion rule and has not been tuned beyond the included small benchmark; evaluate it on a larger question set before changing the weighting or adding reranking.
- The index is rebuilt at startup and is not persisted to disk.
