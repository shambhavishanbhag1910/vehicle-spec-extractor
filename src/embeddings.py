from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(
            model_name
        )


    def encode_documents(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )


    def encode_query(self, query):

        return self.model.encode(
            [query],
            normalize_embeddings=True
        )