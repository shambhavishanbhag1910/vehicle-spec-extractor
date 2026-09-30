import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_pages
from src.cleaner import clean_pages
from src.embeddings import EmbeddingModel
from src.extractor import GroqExtractor
from src.pdf_parser import extract_pdf_pages
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def result_matches(expected, actual):
    for field, expected_value in expected.items():
        actual_value = actual.get(field)

        if isinstance(expected_value, str):
            if normalize_text(actual_value) != normalize_text(expected_value):
                return False
        elif actual_value != expected_value:
            return False

    return True


def score_case(case, response, retrieved_chunks):
    expected_status = case["expected_status"]
    results = response.get("results", [])
    status_correct = response.get("status") == expected_status
    expected_results = case["expected_results"]

    if expected_status == "not_found":
        fields_correct = not results and not expected_results
        evidence_grounded = not results
    else:
        unmatched_results = list(results)
        fields_correct = len(unmatched_results) == len(expected_results)

        for expected in expected_results:
            matching_index = next(
                (
                    index
                    for index, actual in enumerate(unmatched_results)
                    if result_matches(expected, actual)
                ),
                None
            )

            if matching_index is None:
                fields_correct = False
                break

            unmatched_results.pop(matching_index)

        context = normalize_text(" ".join(
            chunk["text"]
            for chunk in retrieved_chunks
        ))
        evidence_grounded = bool(results) and all(
            normalize_text(result.get("evidence", ""))
            and normalize_text(result["evidence"]) in context
            for result in results
        )

    retrieved_pages = sorted({
        chunk["page_number"]
        for chunk in retrieved_chunks
    })
    relevant_pages = case.get("relevant_pages", [])
    relevant_page_retrieved = (
        any(page in retrieved_pages for page in relevant_pages)
        if relevant_pages
        else None
    )
    passed = status_correct and fields_correct and evidence_grounded

    return {
        "id": case["id"],
        "passed": passed,
        "status_correct": status_correct,
        "expected_fields_correct": fields_correct,
        "evidence_grounded": evidence_grounded,
        "relevant_page_retrieved": relevant_page_retrieved,
        "retrieved_pages": retrieved_pages,
        "response": response
    }


def run_evaluation(pdf_path, top_k, cases, extractor):
    pages = clean_pages(extract_pdf_pages(pdf_path))
    chunks = chunk_pages(pages)
    texts = [chunk["text"] for chunk in chunks]

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(texts)
    vector_store = FaissVectorStore()
    vector_store.build(embeddings, chunks)
    retriever = Retriever(embedding_model, vector_store)

    case_results = []

    for case in cases:
        retrieved_chunks = retriever.retrieve(case["query"], top_k=top_k)

        try:
            response = extractor.extract(case["query"], retrieved_chunks)
            case_results.append(
                score_case(case, response, retrieved_chunks)
            )
        except Exception as error:
            case_results.append({
                "id": case["id"],
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "retrieved_pages": sorted({
                    chunk["page_number"]
                    for chunk in retrieved_chunks
                })
            })

    passed_count = sum(result["passed"] for result in case_results)
    total_cases = len(case_results)
    status_results = [
        result["status_correct"]
        for result in case_results
        if "status_correct" in result
    ]
    field_results = [
        result["expected_fields_correct"]
        for result in case_results
        if "expected_fields_correct" in result
    ]
    evidence_results = [
        result["evidence_grounded"]
        for result in case_results
        if "evidence_grounded" in result
    ]
    retrieval_results = [
        result["relevant_page_retrieved"]
        for result in case_results
        if result.get("relevant_page_retrieved") is not None
    ]

    def accuracy(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "pdf": str(pdf_path),
        "model": extractor.model,
        "top_k": top_k,
        "cases_passed": passed_count,
        "cases_total": total_cases,
        "exact_case_accuracy": (
            passed_count / total_cases if total_cases else 0.0
        ),
        "status_accuracy": accuracy(status_results),
        "expected_fields_accuracy": accuracy(field_results),
        "evidence_grounding_rate": accuracy(evidence_results),
        "relevant_page_recall": accuracy(retrieval_results),
        "cases": case_results
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run gold-answer extraction checks. This sends one Groq request "
            "per case and may incur API charges."
        )
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "data" / "sample-service-manual.pdf"
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "evaluation" / "extraction_cases.json"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON evaluation report."
    )
    parser.add_argument(
        "--call-llm",
        action="store_true",
        help="Explicitly enable live Groq calls for each evaluation case."
    )
    args = parser.parse_args()

    if not args.call_llm:
        parser.error(
            "Live API evaluation is opt-in; pass --call-llm to proceed."
        )

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    load_dotenv(ROOT / ".env")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    extractor = GroqExtractor()
    report = run_evaluation(
        args.pdf,
        args.top_k,
        cases,
        extractor
    )
    report_text = json.dumps(report, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text, encoding="utf-8")
        print(f"Wrote report to {args.output}")

    print(
        f"Exact case accuracy: {report['cases_passed']}/"
        f"{report['cases_total']} ({report['exact_case_accuracy']:.1%})"
    )

    for result in report["cases"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['id']}")

    if not args.output:
        print(report_text)

    return 0 if report["cases_passed"] == report["cases_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())