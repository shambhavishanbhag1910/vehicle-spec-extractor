from src.pdf_parser import extract_pdf_pages
from src.cleaner import clean_pages
from src.chunker import chunk_pages
from src.embeddings import EmbeddingModel
from src.vector_store import FaissVectorStore
from src.retriever import Retriever
from src.extractor import GroqExtractor


class VehicleSpecificationPipeline:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.vector_store = FaissVectorStore()

        self.retriever = None

        self.extractor = GroqExtractor()


    def ingest(self, pdf_path):

        print("Extracting PDF...")

        pages = extract_pdf_pages(
            pdf_path
        )

        print(
            f"Extracted {len(pages)} pages"
        )

        pages = clean_pages(pages)

        print("Creating chunks...")

        chunks = chunk_pages(pages)

        print(
            f"Created {len(chunks)} chunks"
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        print("Creating embeddings...")

        embeddings = (
            self.embedding_model
            .encode_documents(texts)
        )

        print("Building FAISS index...")

        self.vector_store.build(
            embeddings,
            chunks
        )

        self.retriever = Retriever(
            self.embedding_model,
            self.vector_store
        )


    def ask(
        self,
        query,
        top_k=5
    ):

        if self.retriever is None:
            raise RuntimeError(
                "Ingest a service manual before asking questions."
            )

        chunks = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        result = self.extractor.extract(
            query,
            chunks
        )

        return result