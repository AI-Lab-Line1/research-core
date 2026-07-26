from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .models import Chunk, RetrievalHit, VectorPreview, VectorTerm


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
STOPWORDS = {
    "的", "是", "有", "在", "和", "与", "什么", "哪些", "请", "如何", "可以", "吗", "了", "为", "这", "个", "中", "从", "到", "一下",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in jieba.lcut(text)
        if TOKEN_PATTERN.fullmatch(token) and token.lower() not in STOPWORDS
    ]


class SearchIndex(ABC):
    id: str
    display_name: str
    representation_name: str
    score_name: str
    chunks: list[Chunk] | None

    @property
    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def search(self, question: str, top_k: int) -> tuple[list[RetrievalHit], int, list[str]]:
        raise NotImplementedError


@dataclass
class TfidfIndex(SearchIndex):
    id = "tfidf"
    display_name = "TF-IDF 稀疏向量检索"
    representation_name = "TF-IDF 稀疏向量"
    score_name = "cosine_similarity"

    vectorizer: TfidfVectorizer | None = None
    matrix: csr_matrix | None = None
    chunks: list[Chunk] | None = None

    @property
    def ready(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None and bool(self.chunks)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix is not None else 0

    def build(self, chunks: list[Chunk]) -> list[Chunk]:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks]).tocsr()
        feature_names = vectorizer.get_feature_names_out()

        for row_index, chunk in enumerate(chunks):
            row = matrix.getrow(row_index)
            top_indices = row.indices[np.argsort(row.data)[::-1][:6]] if row.nnz else []
            weights = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
            chunk.vector = VectorPreview(
                dimension=matrix.shape[1],
                nonzero_count=row.nnz,
                top_terms=[
                    VectorTerm(term=str(feature_names[index]), weight=round(float(weights[index]), 4))
                    for index in top_indices
                ],
            )

        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks
        return chunks

    def search(self, question: str, top_k: int) -> tuple[list[RetrievalHit], int, list[str]]:
        if not self.ready or self.vectorizer is None or self.matrix is None or self.chunks is None:
            raise RuntimeError("索引尚未构建")

        query_vector = self.vectorizer.transform([question])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1]
        ranking = [index for index in ranked_indices if scores[index] > 0][: min(top_k, len(self.chunks))]
        query_terms = set(tokenize(question))
        hits = [
            self._hit(rank, int(chunk_index), float(scores[chunk_index]), query_terms)
            for rank, chunk_index in enumerate(ranking, start=1)
        ]
        query_preview_terms = [
            str(self.vectorizer.get_feature_names_out()[index])
            for index in query_vector.indices[np.argsort(query_vector.data)[::-1][:8]]
        ]
        return hits, int(query_vector.nnz), query_preview_terms

    def _hit(self, rank: int, chunk_index: int, score: float, query_terms: set[str]) -> RetrievalHit:
        assert self.chunks is not None
        chunk = self.chunks[chunk_index]
        matched = sorted(query_terms.intersection(tokenize(chunk.text)), key=len, reverse=True)
        return RetrievalHit(
            rank=rank,
            retrieval_rank=rank,
            score=round(score, 4),
            score_label="余弦相似度",
            score_components={"tfidf": round(score, 4)},
            selected_for_context=True,
            matched_terms=matched[:8],
            chunk=chunk,
        )


@dataclass
class Bm25Index(SearchIndex):
    id = "bm25"
    display_name = "BM25 关键词检索"
    representation_name = "BM25 词项统计"
    score_name = "bm25_score"

    engine: BM25Okapi | None = None
    tokenized_chunks: list[list[str]] | None = None
    vocabulary: set[str] | None = None
    chunks: list[Chunk] | None = None

    @property
    def ready(self) -> bool:
        return self.engine is not None and self.tokenized_chunks is not None and bool(self.chunks)

    @property
    def dimension(self) -> int:
        return len(self.vocabulary or set())

    def build(self, chunks: list[Chunk]) -> list[Chunk]:
        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        engine = BM25Okapi(tokenized_chunks)
        vocabulary = {term for tokens in tokenized_chunks for term in tokens}

        for chunk, tokens in zip(chunks, tokenized_chunks, strict=True):
            counts: dict[str, int] = {}
            for term in tokens:
                counts[term] = counts.get(term, 0) + 1
            weighted_terms = sorted(
                ((term, count * max(float(engine.idf.get(term, 0)), 0)) for term, count in counts.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:6]
            chunk.vector = VectorPreview(
                dimension=len(vocabulary),
                nonzero_count=len(counts),
                top_terms=[VectorTerm(term=term, weight=round(weight, 4)) for term, weight in weighted_terms],
            )

        self.engine = engine
        self.tokenized_chunks = tokenized_chunks
        self.vocabulary = vocabulary
        self.chunks = chunks
        return chunks

    def search(self, question: str, top_k: int) -> tuple[list[RetrievalHit], int, list[str]]:
        if not self.ready or self.engine is None or self.chunks is None:
            raise RuntimeError("索引尚未构建")

        query_tokens = tokenize(question)
        scores = self.engine.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        ranking = [index for index in ranked_indices if scores[index] > 0][: min(top_k, len(self.chunks))]
        query_terms = set(query_tokens)
        hits: list[RetrievalHit] = []
        for rank, chunk_index in enumerate(ranking, start=1):
            chunk = self.chunks[int(chunk_index)]
            matched = sorted(query_terms.intersection(tokenize(chunk.text)), key=len, reverse=True)
            score = round(float(scores[chunk_index]), 4)
            hits.append(RetrievalHit(
                rank=rank,
                retrieval_rank=rank,
                score=score,
                score_label="BM25 相关度",
                score_components={"bm25": score},
                selected_for_context=True,
                matched_terms=matched[:8],
                chunk=chunk,
            ))
        known_terms = [term for term in dict.fromkeys(query_tokens) if term in (self.vocabulary or set())]
        return hits, len(known_terms), known_terms[:8]


class HybridIndex(SearchIndex):
    id = "hybrid"
    display_name = "TF-IDF + BM25 混合检索"
    representation_name = "TF-IDF 向量与 BM25 词项双索引"
    score_name = "weighted_score_fusion"

    def __init__(self, tfidf_weight: float = 0.55):
        self.tfidf_weight = tfidf_weight
        self.tfidf = TfidfIndex()
        self.bm25 = Bm25Index()
        self.chunks: list[Chunk] | None = None

    @property
    def ready(self) -> bool:
        return self.tfidf.ready and self.bm25.ready and bool(self.chunks)

    @property
    def dimension(self) -> int:
        return self.tfidf.dimension

    def build(self, chunks: list[Chunk]) -> list[Chunk]:
        self.bm25.build(chunks)
        self.tfidf.build(chunks)
        self.chunks = chunks
        return chunks

    def search(self, question: str, top_k: int) -> tuple[list[RetrievalHit], int, list[str]]:
        if not self.ready or self.chunks is None:
            raise RuntimeError("索引尚未构建")

        tfidf_hits, query_nonzero, query_terms = self.tfidf.search(question, len(self.chunks))
        bm25_hits, bm25_nonzero, bm25_terms = self.bm25.search(question, len(self.chunks))
        tfidf_scores = {hit.chunk.id: hit.score for hit in tfidf_hits}
        bm25_scores = {hit.chunk.id: hit.score for hit in bm25_hits}
        max_tfidf = max(tfidf_scores.values(), default=1.0)
        max_bm25 = max(bm25_scores.values(), default=1.0)
        bm25_weight = 1 - self.tfidf_weight
        query_token_set = set(tokenize(question))
        candidates: list[tuple[float, Chunk, float, float]] = []

        for chunk in self.chunks:
            tfidf_score = tfidf_scores.get(chunk.id, 0) / max_tfidf if max_tfidf > 0 else 0
            bm25_score = bm25_scores.get(chunk.id, 0) / max_bm25 if max_bm25 > 0 else 0
            fused_score = self.tfidf_weight * tfidf_score + bm25_weight * bm25_score
            if fused_score > 0:
                candidates.append((fused_score, chunk, tfidf_score, bm25_score))

        candidates.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for rank, (score, chunk, tfidf_score, bm25_score) in enumerate(candidates[:top_k], start=1):
            matched = sorted(query_token_set.intersection(tokenize(chunk.text)), key=len, reverse=True)
            hits.append(RetrievalHit(
                rank=rank,
                retrieval_rank=rank,
                score=round(score, 4),
                score_label="融合分数",
                score_components={"tfidf": round(tfidf_score, 4), "bm25": round(bm25_score, 4)},
                selected_for_context=True,
                matched_terms=matched[:8],
                chunk=chunk,
            ))

        merged_terms = list(dict.fromkeys([*query_terms, *bm25_terms]))[:8]
        return hits, max(query_nonzero, bm25_nonzero), merged_terms


def get_index(method: str) -> SearchIndex:
    indexes: dict[str, type[SearchIndex]] = {
        "tfidf": TfidfIndex,
        "bm25": Bm25Index,
        "hybrid": HybridIndex,
    }
    try:
        return indexes[method]()
    except KeyError as exc:
        raise ValueError(f"检索方法 {method!r} 尚未实现") from exc
