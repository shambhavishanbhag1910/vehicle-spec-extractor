import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunker import chunk_pages
from src.cleaner import clean_pages
from src.embeddings import EmbeddingModel
from src.pdf_parser import extract_pdf_pages
from src.retriever import Retriever
from src.vector_store import FaissVectorStore


def run_evaluation(pdf_path, top_k):
    cases_path = ROOT / "evaluation" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    extraction_cases_path = ROOT / "evaluation" / "extraction_cases.json"
    extraction_cases = json.loads(
        extraction_cases_path.read_text(encoding="utf-8")
    )
    cases.extend(
        case
        for case in extraction_cases
        if case["relevant_pages"]
    )

    pages = clean_pages(extract_pdf_pages(pdf_path))
    chunks = chunk_pages(pages)
    texts = [chunk["text"] for chunk in chunks]

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.encode_documents(texts)
    vector_store = FaissVectorStore()
    vector_store.build(embeddings, chunks)
    retriever = Retriever(embedding_model, vector_store)

    passed = 0

    for case in cases:
        results = retriever.retrieve(case["query"], top_k=top_k)
        retrieved_pages = [result["page_number"] for result in results]
        hit = any(
            page_number in case["relevant_pages"]
            for page_number in retrieved_pages
        )
        passed += int(hit)
        status = "PASS" if hit else "MISS"
        print(
            f"{status} {case['id']}: "
            f"relevant pages={case['relevant_pages']}; "
            f"retrieved pages={retrieved_pages}"
        )

    hit_rate = passed / len(cases) if cases else 0.0
    print(f"Recall@{top_k}: {passed}/{len(cases)} ({hit_rate:.1%})")
    return 0 if passed == len(cases) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether semantic retrieval finds gold manual pages."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "data" / "sample-service-manual.pdf"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    return run_evaluation(args.pdf, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())