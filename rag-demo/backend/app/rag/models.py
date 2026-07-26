from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MethodOption(BaseModel):
    id: str
    name: str
    category: Literal["chunking", "retrieval", "reranking", "generation"]
    status: Literal["available", "planned"]
    description: str
    advantages: list[str]
    limitations: list[str]


class SourceDocument(BaseModel):
    name: str
    path: str
    content: str
    character_count: int
    paragraph_count: int
    section_count: int
    sections: list[str]


class VectorTerm(BaseModel):
    term: str
    weight: float


class VectorPreview(BaseModel):
    dimension: int
    nonzero_count: int
    top_terms: list[VectorTerm]


class Chunk(BaseModel):
    id: str
    order: int
    section: str
    text: str
    character_count: int
    start_char: int
    end_char: int
    overlap_chars: int = 0
    boundary_similarity: float | None = None
    split_reason: str | None = None
    semantic_unit_count: int = 1
    vector: VectorPreview | None = None


class PipelineStep(BaseModel):
    id: str
    name: str
    status: Literal["completed", "skipped", "failed"] = "completed"
    duration_ms: float
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class IndexStatus(BaseModel):
    ready: bool
    chunking_method: str
    retrieval_method: str
    chunk_count: int
    vector_dimension: int
    vocabulary_size: int
    chunk_size: int
    chunk_overlap: int
    semantic_threshold: float
    semantic_max_chars: int
    built_at: str | None
    source_name: str


class RuntimeStatus(BaseModel):
    api_version: str
    source_name: str
    source_ready: bool
    longcat_configured: bool
    longcat_model: str | None
    available_methods: list[str]
    planned_methods: list[str]


class IndexBuildRequest(BaseModel):
    chunking_method: str = "structure"
    retrieval_method: str = "tfidf"
    chunk_size: int = Field(default=260, ge=80, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=300)
    semantic_threshold: float = Field(default=0.05, ge=0, le=1)
    semantic_max_chars: int = Field(default=620, ge=160, le=1600)


class IndexBuildResponse(BaseModel):
    status: IndexStatus
    chunks: list[Chunk]
    trace: list[PipelineStep]


class RetrievalHit(BaseModel):
    rank: int
    retrieval_rank: int
    score: float
    score_label: str
    score_components: dict[str, float] = Field(default_factory=dict)
    rerank_score: float | None = None
    selected_for_context: bool
    matched_terms: list[str]
    chunk: Chunk


class ContextBlock(BaseModel):
    chunk_id: str
    citation: str
    section: str
    text: str


class AnswerPoint(BaseModel):
    text: str
    citation: str
    chunk_id: str
    selection_reason: str


class GenerationMetadata(BaseModel):
    requested_method: str
    effective_method: str
    provider: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    fallback_used: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)
    chunking_method: str = "structure"
    retrieval_method: str = "tfidf"
    reranking_method: str = "none"
    generation_method: str = "extractive"
    chunk_size: int = Field(default=260, ge=80, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=300)
    semantic_threshold: float = Field(default=0.05, ge=0, le=1)
    semantic_max_chars: int = Field(default=620, ge=160, le=1600)


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_mode: str
    answer_intent: Literal["list", "process", "fact", "general", "fallback"]
    answer_points: list[AnswerPoint]
    generation_metadata: GenerationMetadata
    generation_warning: str | None = None
    citations: list[str]
    retrieval_hits: list[RetrievalHit]
    context: list[ContextBlock]
    prompt_preview: str
    trace: list[PipelineStep]
    total_duration_ms: float
    index_status: IndexStatus


class ComparisonConfig(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    chunking_method: str = "structure"
    retrieval_method: str = "tfidf"
    reranking_method: str = "none"
    generation_method: str = "extractive"
    chunk_size: int = Field(default=260, ge=80, le=1000)
    chunk_overlap: int = Field(default=40, ge=0, le=300)
    semantic_threshold: float = Field(default=0.05, ge=0, le=1)
    semantic_max_chars: int = Field(default=620, ge=160, le=1600)


class ComparisonRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)
    configs: list[ComparisonConfig] = Field(min_length=2, max_length=6)


class ComparisonRun(BaseModel):
    config: ComparisonConfig
    result: QueryResponse


class ComparisonResponse(BaseModel):
    question: str
    runs: list[ComparisonRun]
