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

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            result = dict(
                self.chunks[idx]
            )

            result["score"] = float(score)

            results.append(result)

        return results