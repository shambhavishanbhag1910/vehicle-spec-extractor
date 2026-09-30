import json
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import ValidationError

from src.chunker import chunk_pages
from src.cleaner import clean_text
from src.extractor import GroqExtractor
from src.pdf_parser import extract_pdf_pages
from src.retriever import Retriever
from src.schemas import ExtractionResponse
from src.vector_store import FaissVectorStore
from evaluation.run_extraction_eval import score_case


ROOT = Path(__file__).resolve().parents[1]


class ChunkerTests(unittest.TestCase):

    def test_sections_carry_across_pages_and_switch_on_heading(self):
        pages = [
            {
                "page_number": 1,
                "text": (
                    "SECTION 204-00: Suspension\n"
                    "Front measurements."
                )
            },
            {
                "page_number": 2,
                "text": "Continued suspension procedure."
            },
            {
                "page_number": 3,
                "text": (
                    "SECTION 204-01: Alignment\n"
                    "Alignment procedure."
                )
            }
        ]

        chunks = chunk_pages(pages, chunk_size=200, overlap=20)

        self.assertEqual(chunks[0]["section_id"], "204-00")
        self.assertEqual(chunks[1]["section_id"], "204-00")
        self.assertEqual(chunks[2]["section_id"], "204-01")

    def test_chunks_respect_size_and_reject_invalid_overlap(self):
        pages = [{
            "page_number": 1,
            "text": ("Paragraph sentence.\n\n" * 30)
        }]

        chunks = chunk_pages(pages, chunk_size=80, overlap=15)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= 80 for chunk in chunks))
        with self.assertRaises(ValueError):
            chunk_pages(pages, chunk_size=10, overlap=10)


class CleanerTests(unittest.TestCase):

    def test_bullet_artifact_only_changes_line_prefix(self):
        self.assertEqual(
            clean_text("z First item\nMazda z engine"),
            "• First item\nMazda z engine"
        )


class PdfParserOcrTests(unittest.TestCase):

    class FakePage:
        def __init__(self, text, ocr_text=""):
            self.text = text
            self.ocr_text = ocr_text
            self.ocr_calls = []

        def get_text(self, mode, textpage=None):
            if textpage is not None:
                return self.ocr_text
            return self.text

        def get_textpage_ocr(self, **kwargs):
            self.ocr_calls.append(kwargs)
            return object()

    class FakeDocument:
        def __init__(self, page):
            self.page_count = 1
            self.page = page

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def load_page(self, index):
            return self.page

    def setUp(self):
        self.pdf_path = ROOT / "data" / "sample-service-manual.pdf"

    def test_ocr_runs_only_on_text_poor_pages_when_enabled(self):
        page = self.FakePage("", "Recognized torque specification")

        with patch("src.pdf_parser.pymupdf.open", return_value=self.FakeDocument(page)):
            pages = extract_pdf_pages(
                self.pdf_path,
                ocr_enabled=True,
                ocr_language="eng",
                ocr_dpi=250,
                min_text_chars=20
            )

        self.assertEqual(pages[0]["text"], "Recognized torque specification")
        self.assertTrue(pages[0]["ocr_used"])
        self.assertEqual(page.ocr_calls, [{
            "language": "eng",
            "dpi": 250,
            "full": True
        }])

    def test_ocr_is_not_used_for_native_text_or_when_disabled(self):
        page = self.FakePage("Existing searchable manual text")

        with patch("src.pdf_parser.pymupdf.open", return_value=self.FakeDocument(page)):
            pages = extract_pdf_pages(
                self.pdf_path,
                ocr_enabled=True,
                min_text_chars=10
            )

        self.assertEqual(pages[0]["text"], "Existing searchable manual text")
        self.assertFalse(pages[0]["ocr_used"])
        self.assertEqual(page.ocr_calls, [])

    def test_ocr_failure_has_actionable_message(self):
        page = self.FakePage("")

        def fail_ocr(**kwargs):
            raise RuntimeError("Tesseract executable not found")

        page.get_textpage_ocr = fail_ocr

        with patch("src.pdf_parser.pymupdf.open", return_value=self.FakeDocument(page)):
            with self.assertRaisesRegex(RuntimeError, "Install Tesseract OCR"):
                extract_pdf_pages(self.pdf_path, ocr_enabled=True)


class VectorStoreTests(unittest.TestCase):

    def test_search_clamps_top_k_to_index_size(self):
        store = FaissVectorStore()
        store.build(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
            [{"text": "first"}, {"text": "second"}]
        )

        results = store.search(
            np.array([[1.0, 0.0]], dtype="float32"),
            top_k=5
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["text"], "first")

    def test_search_requires_a_built_index(self):
        with self.assertRaises(RuntimeError):
            FaissVectorStore().search(np.array([[1.0]], dtype="float32"))


class RetrieverTests(unittest.TestCase):

    def test_tokenizer_discards_common_question_words(self):
        self.assertEqual(
            Retriever._tokenize(
                "What is the torque for caliper bolts?"
            ),
            ["torque", "caliper", "bolts"]
        )

    def test_adjacent_same_section_page_is_included_within_top_k(self):
        class FakeEmbeddingModel:
            def encode_query(self, query):
                return np.array([[1.0, 0.0]], dtype="float32")

        class FakeVectorStore:
            chunks = [
                {
                    "chunk_id": 0,
                    "page_number": 18,
                    "section_id": "204-00",
                    "text": "Front procedure continuation."
                },
                {
                    "chunk_id": 1,
                    "page_number": 19,
                    "section_id": "204-00",
                    "text": "Raptor rear ride height procedure starts."
                },
                {
                    "chunk_id": 2,
                    "page_number": 20,
                    "section_id": "204-00",
                    "text": "Raptor measurement points."
                }
            ]

            def search(self, query_embedding, top_k):
                return [{**self.chunks[1], "score": 0.9}]

        retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())
        results = retriever.retrieve("Raptor ride height", top_k=3)

        self.assertEqual(len(results), 3)
        self.assertIn(20, [result["page_number"] for result in results])
        adjacent = next(result for result in results if result["page_number"] == 20)
        self.assertTrue(adjacent["adjacent_context"])
        self.assertIsNone(adjacent["score"])

    def test_bm25_finds_exact_identifier_without_dense_matches(self):
        class FakeEmbeddingModel:
            def encode_query(self, query):
                return np.array([[1.0, 0.0]], dtype="float32")

        class FakeVectorStore:
            chunks = [
                {
                    "chunk_id": 0,
                    "page_number": 1,
                    "section_id": "206-00",
                    "text": "General brake service procedure."
                },
                {
                    "chunk_id": 1,
                    "page_number": 2,
                    "section_id": "206-00",
                    "text": "Caliper bolt part number AB-123."
                }
            ]

            def search(self, query_embedding, top_k):
                return []

        retriever = Retriever(FakeEmbeddingModel(), FakeVectorStore())
        results = retriever.retrieve("AB-123", top_k=1)

        self.assertEqual(results[0]["page_number"], 2)
        self.assertIsNone(results[0]["dense_score"])
        self.assertIsNotNone(results[0]["bm25_score"])


class SchemaTests(unittest.TestCase):

    def test_part_number_response_validates(self):
        response = ExtractionResponse.model_validate({
            "status": "found",
            "results": [{
                "component": "brake caliper bolt",
                "spec_type": "Part Number",
                "part_number": "ABC123",
                "evidence": "Part number ABC123 is specified."
            }]
        })

        self.assertEqual(response.results[0].part_number, "ABC123")

    def test_status_must_match_results(self):
        with self.assertRaises(ValidationError):
            ExtractionResponse.model_validate({
                "status": "found",
                "results": []
            })

        response = ExtractionResponse.model_validate({
            "status": "not_found",
            "results": []
        })
        self.assertEqual(response.status, "not_found")


class ExtractorTests(unittest.TestCase):

    def test_extractor_validates_and_returns_structured_json(self):
        payload = {
            "status": "found",
            "results": [{
                "component": "brake caliper bolt",
                "spec_type": "Part Number",
                "part_number": "ABC123",
                "evidence": "The manual lists part number ABC123."
            }]
        }
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload))
            )]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("src.extractor.Groq", return_value=client):
                extractor = GroqExtractor()

        result = extractor.extract("Part number?", [{
            "page_number": 1,
            "section_id": "205-00",
            "text": "The manual lists part number ABC123."
        }])

        self.assertEqual(result["results"][0]["part_number"], "ABC123")
        self.assertEqual(result["status"], "found")
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["response_format"],
            {"type": "json_object"}
        )

    def test_extractor_rejects_invalid_model_output(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"status":"found","results":[]}')
            )]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("src.extractor.Groq", return_value=client):
                extractor = GroqExtractor()

        with self.assertRaisesRegex(ValueError, "does not match"):
            extractor.extract("Question", [{
                "page_number": 1,
                "section_id": None,
                "text": "Context"
            }])


class ExtractionEvaluationTests(unittest.TestCase):

    def setUp(self):
        self.case = {
            "id": "brake-caliper-torque",
            "expected_status": "found",
            "relevant_pages": [636],
            "expected_results": [{
                "component": "Brake caliper anchor plate bolts",
                "spec_type": "Torque",
                "value": "250",
                "unit": "Nm",
                "page": 636
            }]
        }
        self.context = [{
            "page_number": 636,
            "text": "Brake caliper anchor plate bolts 250 184 Nm lb-ft"
        }]

    def test_gold_fields_and_verbatim_evidence_pass(self):
        response = {
            "status": "found",
            "results": [{
                "component": "brake caliper anchor plate bolts",
                "spec_type": "torque",
                "value": "250",
                "unit": "Nm",
                "page": 636,
                "evidence": "Brake caliper anchor plate bolts 250 184"
            }]
        }

        result = score_case(self.case, response, self.context)

        self.assertTrue(result["passed"])
        self.assertTrue(result["evidence_grounded"])

    def test_hallucinated_evidence_fails(self):
        response = {
            "status": "found",
            "results": [{
                "component": "Brake caliper anchor plate bolts",
                "spec_type": "Torque",
                "value": "250",
                "unit": "Nm",
                "page": 636,
                "evidence": "The manual says 999 Nm."
            }]
        }

        result = score_case(self.case, response, self.context)

        self.assertFalse(result["passed"])
        self.assertFalse(result["evidence_grounded"])

    def test_not_found_requires_empty_results(self):
        case = {
            "id": "unsupported",
            "expected_status": "not_found",
            "relevant_pages": [],
            "expected_results": []
        }

        correct = score_case(
            case,
            {"status": "not_found", "results": []},
            self.context
        )
        incorrect = score_case(
            case,
            {"status": "found", "results": [{"evidence": "unsourced"}]},
            self.context
        )

        self.assertTrue(correct["passed"])
        self.assertFalse(incorrect["passed"])

    def test_distinct_part_numbers_are_scored_in_their_source_contexts(self):
        cases = json.loads(
            (ROOT / "evaluation" / "extraction_cases.json").read_text(
                encoding="utf-8"
            )
        )
        part_number_cases = [
            case for case in cases if "shock-" in case["id"]
        ]

        self.assertEqual(len(part_number_cases), 2)

        for case in part_number_cases:
            expected = case["expected_results"][0]
            result = score_case(
                case,
                {
                    "status": "found",
                    "results": [{
                        **expected,
                        "evidence": case["source_facts"][0]["text"]
                    }]
                },
                [{
                    "page_number": case["source_facts"][0]["page"],
                    "text": case["source_facts"][0]["text"]
                }]
            )

            with self.subTest(case=case["id"]):
                self.assertTrue(result["passed"])


class ManualEvaluationDataTests(unittest.TestCase):

    def test_expected_source_facts_exist_in_manual(self):
        cases_path = ROOT / "evaluation" / "cases.json"
        extraction_cases_path = ROOT / "evaluation" / "extraction_cases.json"
        manual_path = ROOT / "data" / "sample-service-manual.pdf"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        cases.extend(
            json.loads(extraction_cases_path.read_text(encoding="utf-8"))
        )
        pages = extract_pdf_pages(manual_path)
        texts_by_page = {
            page["page_number"]: re.sub(r"\s+", " ", page["text"])
            for page in pages
        }

        for case in cases:
            for source in case["source_facts"]:
                with self.subTest(case=case["id"], page=source["page"]):
                    expected_text = re.sub(r"\s+", " ", source["text"])
                    self.assertIn(
                        expected_text,
                        texts_by_page[source["page"]]
                    )


if __name__ == "__main__":
    unittest.main()