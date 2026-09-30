import re

from rank_bm25 import BM25Okapi


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "both", "by",
    "do", "for", "from", "how", "i", "in", "include", "is", "it",
    "of", "on", "or", "the", "this", "to", "what", "with"
}
BM25_RRF_WEIGHT = 2.0


class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunks = vector_store.chunks or []
        self.tokenized_chunks = [
            self._tokenize(chunk["text"])
            for chunk in self.chunks
        ]
        self.bm25 = (
            BM25Okapi(self.tokenized_chunks)
            if any(self.tokenized_chunks)
            else None
        )

    @staticmethod
    def _tokenize(text):
        tokens = re.findall(
            r"[a-z0-9]+(?:[.-][a-z0-9]+)*",
            text.casefold()
        )
        return [token for token in tokens if token not in STOP_WORDS]


    def retrieve(
        self,
        query,
        top_k=5
    ):

        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        dense_matches = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        if top_k <= 0:
            return []

        candidate_scores = {}

        def add_ranked_match(chunk, rank, source, source_score):
            chunk_id = chunk["chunk_id"]
            candidate = candidate_scores.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "fused_score": 0.0,
                    "dense_score": None,
                    "bm25_score": None
                }
            )
            weight = BM25_RRF_WEIGHT if source == "bm25" else 1.0
            candidate["fused_score"] += weight / (60 + rank)
            candidate[source + "_score"] = float(source_score)

        for rank, match in enumerate(dense_matches, start=1):
            add_ranked_match(match, rank, "dense", match["score"])

        query_tokens = self._tokenize(query)

        if self.bm25 is not None and query_tokens:
            lexical_scores = self.bm25.get_scores(query_tokens)
            lexical_ranking = sorted(
                enumerate(lexical_scores),
                key=lambda item: item[1],
                reverse=True
            )

            lexical_rank = 0

            for chunk_index, lexical_score in lexical_ranking:
                if not any(
                    token in self.tokenized_chunks[chunk_index]
                    for token in query_tokens
                ):
                    continue

                lexical_rank += 1
                add_ranked_match(
                    self.chunks[chunk_index],
                    lexical_rank,
                    "bm25",
                    lexical_score
                )

                if lexical_rank >= top_k:
                    break

        matches = []

        for candidate in sorted(
            candidate_scores.values(),
            key=lambda item: item["fused_score"],
            reverse=True
        )[:top_k]:
            match = dict(candidate["chunk"])
            match["score"] = candidate["fused_score"]
            match["dense_score"] = candidate["dense_score"]
            match["bm25_score"] = candidate["bm25_score"]
            matches.append(match)

        if not matches:
            return []

        indexed_chunks = self.chunks
        chunks_by_page_and_section = {}

        for chunk in indexed_chunks:
            section_id = chunk.get("section_id")

            if section_id is not None:
                key = (chunk["page_number"], section_id)
                chunks_by_page_and_section.setdefault(key, []).append(chunk)

        results = []
        seen_chunk_ids = set()

        def append_result(result):
            chunk_id = result["chunk_id"]

            if chunk_id in seen_chunk_ids or len(results) >= top_k:
                return

            seen_chunk_ids.add(chunk_id)
            results.append(result)

        for match in matches:
            append_result(match)

            if len(results) >= top_k:
                break

            section_id = match.get("section_id")

            if section_id is None:
                continue

            for neighboring_page in (
                match["page_number"] - 1,
                match["page_number"] + 1
            ):
                neighboring_chunks = chunks_by_page_and_section.get(
                    (neighboring_page, section_id),
                    []
                )

                if not neighboring_chunks:
                    continue

                nearest_chunk = min(
                    neighboring_chunks,
                    key=lambda chunk: abs(
                        chunk["chunk_id"] - match["chunk_id"]
                    )
                )
                adjacent_result = dict(nearest_chunk)
                adjacent_result["score"] = None
                adjacent_result["adjacent_context"] = True
                append_result(adjacent_result)

                if len(results) >= top_k:
                    break

            if len(results) >= top_k:
                break

        return results