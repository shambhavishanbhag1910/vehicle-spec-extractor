import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self):
        self.index = None
        self.chunks = None


    def build(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("At least one two-dimensional embedding is required.")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Each embedding must have a corresponding chunk.")

        dimension = embeddings.shape[1]

        # Cosine similarity because vectors
        # are normalized
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        self.chunks = chunks


    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index is None or self.chunks is None:
            raise RuntimeError("Build the vector index before searching.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        result_count = min(top_k, self.index.ntotal)

        if result_count == 0:
            return []

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            result_count
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            result = dict(
                self.chunks[idx]
            )

            result["score"] = float(score)

            results.append(result)

        return results