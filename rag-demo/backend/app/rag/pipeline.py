from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from time import perf_counter

from .chunkers import get_chunker
from .generator import ExtractiveTeachingGenerator, GenerationResult
from .index import SearchIndex, get_index
from .loader import MarkdownLoader
from .llm import LongCatConfigurationError, LongCatGenerator, LongCatRequestError
from .models import ContextBlock, IndexBuildResponse, IndexStatus, PipelineStep, QueryRequest, QueryResponse, RetrievalHit, SourceDocument
from .reranker import rerank


class RagPipeline:
    def __init__(self, source_path: Path):
        self.loader = MarkdownLoader(source_path)
        self.index: SearchIndex = get_index("tfidf")
        self.generator = ExtractiveTeachingGenerator()
        self.longcat_generator: LongCatGenerator | None = None
        self.document: SourceDocument | None = None
        self.chunking_method = "structure"
        self.retrieval_method = "tfidf"
        self.chunk_size = 260
        self.chunk_overlap = 40
        self.semantic_threshold = 0.05
        self.semantic_max_chars = 620
        self.built_at: str | None = None
        self.last_build_trace: list[PipelineStep] = []
        self._lock = RLock()

    def get_document(self) -> SourceDocument:
        if self.document is None:
            self.document = self.loader.load()
        return self.document

    def status(self) -> IndexStatus:
        chunks = self.index.chunks or []
        return IndexStatus(
            ready=self.index.ready,
            chunking_method=self.chunking_method,
            retrieval_method=self.retrieval_method,
            chunk_count=len(chunks),
            vector_dimension=self.index.dimension,
            vocabulary_size=self.index.dimension,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            semantic_threshold=self.semantic_threshold,
            semantic_max_chars=self.semantic_max_chars,
            built_at=self.built_at,
            source_name=self.loader.source_path.name,
        )

    def build_index(
        self,
        chunking_method: str = "structure",
        retrieval_method: str = "tfidf",
        chunk_size: int = 260,
        chunk_overlap: int = 40,
        semantic_threshold: float = 0.05,
        semantic_max_chars: int = 620,
    ) -> IndexBuildResponse:
        with self._lock:
            trace: list[PipelineStep] = []
            started = perf_counter()
            document = self.loader.load()
            self.document = document
            trace.append(PipelineStep(
                id="load", name="加载知识库", duration_ms=self._elapsed(started),
                summary=f"读取 {document.name}，共 {document.character_count} 个字符",
                detail={"paragraphs": document.paragraph_count, "sections": document.section_count},
            ))

            started = perf_counter()
            chunker = get_chunker(
                chunking_method,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                semantic_threshold=semantic_threshold,
                semantic_max_chars=semantic_max_chars,
            )
            chunks = chunker.split(document)
            if chunking_method == "structure":
                chunking_label = "结构切分"
            elif chunking_method == "fixed_length":
                chunking_label = f"固定长度切分（{chunk_size} / overlap {chunk_overlap}）"
            else:
                chunking_label = f"TF-IDF 语义切分（阈值 {semantic_threshold:.2f} / 最大 {semantic_max_chars} 字符）"
            trace.append(PipelineStep(
                id="chunk", name="文本切分", duration_ms=self._elapsed(started),
                summary=f"使用{chunking_label}生成 {len(chunks)} 个 chunk",
                detail={
                    "method": chunking_method,
                    "chunk_size": chunk_size if chunking_method == "fixed_length" else None,
                    "chunk_overlap": chunk_overlap if chunking_method == "fixed_length" else None,
                    "semantic_threshold": semantic_threshold if chunking_method == "semantic" else None,
                    "semantic_max_chars": semantic_max_chars if chunking_method == "semantic" else None,
                    "semantic_boundaries": sum(chunk.split_reason == "semantic_drop" for chunk in chunks),
                    "length_boundaries": sum(chunk.split_reason == "max_chars" for chunk in chunks),
                    "merged_units": sum(chunk.semantic_unit_count for chunk in chunks),
                    "average_chars": round(sum(c.character_count for c in chunks) / max(len(chunks), 1), 1),
                },
            ))

            started = perf_counter()
            self.index = get_index(retrieval_method)
            chunks = self.index.build(chunks)
            trace.append(PipelineStep(
                id="index", name="表示生成与写入索引", duration_ms=self._elapsed(started),
                summary=f"生成 {len(chunks)} 条{self.index.representation_name}，词表维度 {self.index.dimension}",
                detail={
                    "method": retrieval_method,
                    "representation": self.index.representation_name,
                    "records_written": len(chunks),
                    "dimension": self.index.dimension,
                },
            ))

            self.chunking_method = chunking_method
            self.retrieval_method = retrieval_method
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.semantic_threshold = semantic_threshold
            self.semantic_max_chars = semantic_max_chars
            self.built_at = datetime.now(timezone.utc).isoformat()
            self.last_build_trace = trace
            return IndexBuildResponse(status=self.status(), chunks=chunks, trace=trace)

    def query(self, request: QueryRequest) -> QueryResponse:
        total_started = perf_counter()
        trace: list[PipelineStep] = []
        with self._lock:
            config_changed = (
                request.chunking_method != self.chunking_method
                or request.retrieval_method != self.retrieval_method
                or (request.chunking_method == "fixed_length" and (
                    request.chunk_size != self.chunk_size or request.chunk_overlap != self.chunk_overlap
                ))
                or (request.chunking_method == "semantic" and (
                    request.semantic_threshold != self.semantic_threshold
                    or request.semantic_max_chars != self.semantic_max_chars
                ))
            )
            if not self.index.ready or config_changed:
                build_result = self.build_index(
                    request.chunking_method,
                    request.retrieval_method,
                    request.chunk_size,
                    request.chunk_overlap,
                    request.semantic_threshold,
                    request.semantic_max_chars,
                )
                trace.extend(build_result.trace)
            else:
                trace.extend(self.last_build_trace)

            started = perf_counter()
            candidate_k = request.top_k if request.reranking_method == "none" else request.top_k * 3
            hits, query_nonzero, query_terms = self.index.search(request.question.strip(), candidate_k)
            trace.append(PipelineStep(
                id="retrieve", name="问题向量化与检索", duration_ms=self._elapsed(started),
                summary=f"问题表示命中 {query_nonzero} 个索引词项，召回 {len(hits)} 个 chunk",
                detail={
                    "candidate_k": candidate_k,
                    "query_top_terms": query_terms,
                    "method": request.retrieval_method,
                    "score": self.index.score_name,
                    "initial_ranking": [hit.chunk.id for hit in hits],
                },
            ))

            started = perf_counter()
            initial_ranking = [hit.chunk.id for hit in hits]
            hits = rerank(request.question.strip(), hits, request.reranking_method, request.top_k)
            rerank_status = "skipped" if request.reranking_method == "none" else "completed"
            trace.append(PipelineStep(
                id="rerank", name="候选重排", status=rerank_status, duration_ms=self._elapsed(started),
                summary=(
                    "未启用重排，保留初次检索顺序"
                    if request.reranking_method == "none"
                    else f"按查询词覆盖率将 {len(initial_ranking)} 个候选重排并保留 {len(hits)} 个"
                ),
                detail={
                    "method": request.reranking_method,
                    "before": initial_ranking,
                    "after": [hit.chunk.id for hit in hits],
                    "formula": "0.65 * normalized_retrieval + 0.35 * term_coverage" if request.reranking_method != "none" else None,
                },
            ))

            started = perf_counter()
            context = self.generator.build_context(hits)
            trace.append(PipelineStep(
                id="context", name="构造上下文", duration_ms=self._elapsed(started),
                summary=f"将 {len(context)} 个证据片段按检索顺序拼装",
                detail={"context_chars": sum(len(item.text) for item in context)},
            ))

            started = perf_counter()
            prompt_preview, generation = self._generate_answer(request, hits, context)
            trace.append(PipelineStep(
                id="generate", name="生成教学回答", duration_ms=self._elapsed(started),
                summary=(
                    f"使用 {generation.metadata['effective_method']} 生成 {len(generation.points)} 个可追踪答案点"
                    + ("，已触发回退" if generation.metadata.get("fallback_used") else "")
                ),
                detail={
                    "mode": generation.metadata["effective_method"],
                    "llm_called": generation.metadata["effective_method"] == "longcat",
                    "evidence_sufficient": bool(generation.citation_chunk_ids),
                    "output_intent": generation.intent,
                    "generation_metadata": generation.metadata,
                    "warning": generation.warning,
                    **generation.detail,
                },
            ))

            citations = []
            for chunk_id in generation.citation_chunk_ids:
                block = next((item for item in context if item.chunk_id == chunk_id), None)
                if block:
                    citations.append(f"{block.citation} {block.section} / {block.chunk_id}")

            return QueryResponse(
                question=request.question.strip(), answer=generation.answer,
                answer_mode=(
                    "LongCat-2.0 证据生成"
                    if generation.metadata["effective_method"] == "longcat"
                    else "结构化教学抽取回答（未调用外部 LLM）"
                ),
                answer_intent=generation.intent, answer_points=generation.points, citations=citations,
                generation_metadata=generation.metadata, generation_warning=generation.warning,
                retrieval_hits=hits, context=context, prompt_preview=prompt_preview,
                trace=trace, total_duration_ms=self._elapsed(total_started), index_status=self.status(),
            )

    def _generate_answer(
        self,
        request: QueryRequest,
        hits: list[RetrievalHit],
        context: list[ContextBlock],
    ) -> tuple[str, GenerationResult]:
        question = request.question.strip()
        if request.generation_method == "extractive":
            return self.generator.prompt_preview(question, context), self.generator.generate(question, hits)
        if request.generation_method != "longcat":
            raise ValueError(f"生成方法 {request.generation_method!r} 尚未实现")

        try:
            longcat = self._get_longcat_generator()
            prompt_preview = longcat.prompt_preview(question, context)
            return prompt_preview, longcat.generate(question, context)
        except (LongCatConfigurationError, LongCatRequestError) as exc:
            generation = self.generator.generate(question, hits)
            generation.metadata = {
                "requested_method": "longcat",
                "effective_method": "extractive",
                "provider": "LongCat",
                "model": self.longcat_generator.settings.model if self.longcat_generator else None,
                "fallback_used": True,
            }
            generation.warning = f"{exc}，已回退到本地结构化抽取回答"
            prompt_preview = (
                self.longcat_generator.prompt_preview(question, context)
                if self.longcat_generator
                else self.generator.prompt_preview(question, context)
            )
            return prompt_preview, generation

    def _get_longcat_generator(self) -> LongCatGenerator:
        if self.longcat_generator is None:
            project_root = self.loader.source_path.parents[2]
            self.longcat_generator = LongCatGenerator.from_env_file(project_root / ".env")
        return self.longcat_generator

    @staticmethod
    def _elapsed(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)
