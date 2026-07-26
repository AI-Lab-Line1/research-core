from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .rag.methods import METHODS
from .rag.llm import longcat_configuration_summary
from .rag.models import (
    Chunk,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonRun,
    IndexBuildRequest,
    IndexBuildResponse,
    IndexStatus,
    MethodOption,
    QueryRequest,
    QueryResponse,
    RuntimeStatus,
    SourceDocument,
)
from .rag.pipeline import RagPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_VERSION = "0.3.0"
pipeline = RagPipeline(PROJECT_ROOT / "data" / "source" / "知识库.md")


@asynccontextmanager
async def lifespan(_: FastAPI):
    pipeline.build_index()
    yield


app = FastAPI(title="RAG Learning Demo API", version=APP_VERSION, description="返回完整中间过程的教学型 RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/runtime", response_model=RuntimeStatus)
def get_runtime_status() -> RuntimeStatus:
    longcat_configured, longcat_model = longcat_configuration_summary(PROJECT_ROOT / ".env")
    return RuntimeStatus(
        api_version=APP_VERSION,
        source_name=pipeline.loader.source_path.name,
        source_ready=pipeline.loader.source_path.exists(),
        longcat_configured=longcat_configured,
        longcat_model=longcat_model,
        available_methods=[method.id for method in METHODS if method.status == "available"],
        planned_methods=[method.id for method in METHODS if method.status == "planned"],
    )


@app.get("/api/methods", response_model=list[MethodOption])
def get_methods() -> list[MethodOption]:
    return METHODS


@app.get("/api/knowledge-base", response_model=SourceDocument)
def get_knowledge_base() -> SourceDocument:
    return pipeline.get_document()


@app.get("/api/index/status", response_model=IndexStatus)
def get_index_status() -> IndexStatus:
    return pipeline.status()


@app.post("/api/index/build", response_model=IndexBuildResponse)
def build_index(request: IndexBuildRequest) -> IndexBuildResponse:
    try:
        return pipeline.build_index(
            request.chunking_method,
            request.retrieval_method,
            request.chunk_size,
            request.chunk_overlap,
            request.semantic_threshold,
            request.semantic_max_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/chunks", response_model=list[Chunk])
def get_chunks(offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200)) -> list[Chunk]:
    if not pipeline.index.ready:
        pipeline.build_index()
    chunks = pipeline.index.chunks or []
    return chunks[offset : offset + limit]


@app.post("/api/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    try:
        return pipeline.query(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/compare", response_model=ComparisonResponse)
def compare_methods(request: ComparisonRequest) -> ComparisonResponse:
    runs: list[ComparisonRun] = []
    try:
        for config in request.configs:
            comparison_pipeline = RagPipeline(pipeline.loader.source_path)
            result = comparison_pipeline.query(QueryRequest(
                question=request.question,
                top_k=request.top_k,
                chunking_method=config.chunking_method,
                retrieval_method=config.retrieval_method,
                reranking_method=config.reranking_method,
                generation_method=config.generation_method,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                semantic_threshold=config.semantic_threshold,
                semantic_max_chars=config.semantic_max_chars,
            ))
            runs.append(ComparisonRun(config=config, result=result))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ComparisonResponse(question=request.question, runs=runs)
