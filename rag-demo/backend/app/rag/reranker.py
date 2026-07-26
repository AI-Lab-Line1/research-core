from __future__ import annotations

from .index import tokenize
from .models import RetrievalHit


def rerank(question: str, hits: list[RetrievalHit], method: str, top_k: int) -> list[RetrievalHit]:
    if method == "none":
        selected = hits[:top_k]
        for rank, hit in enumerate(selected, start=1):
            hit.rank = rank
            hit.selected_for_context = True
        return selected

    if method != "term_coverage":
        raise ValueError(f"重排方法 {method!r} 尚未实现")

    query_terms = set(tokenize(question))
    max_score = max((hit.score for hit in hits), default=1.0)
    scored: list[tuple[float, RetrievalHit]] = []
    for hit in hits:
        chunk_terms = set(tokenize(hit.chunk.text))
        coverage = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
        normalized_retrieval = hit.score / max_score if max_score > 0 else 0
        rerank_score = normalized_retrieval * 0.65 + coverage * 0.35
        hit.rerank_score = round(rerank_score, 4)
        hit.selected_for_context = False
        scored.append((rerank_score, hit))

    scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
    selected = [hit for _, hit in scored[:top_k]]
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
        hit.selected_for_context = True
    return selected
