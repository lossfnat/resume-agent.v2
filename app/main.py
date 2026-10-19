from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agent import analyze_resume
from app.config import settings
from app.ingest import build_knowledge_base
from app.metrics import evaluate_recall_at_k
from app.retriever import JDRetriever
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
    VectorStoreInfoResponse,
)
from app.vectorstore import describe_vector_store

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Optimization Agent",
    description="上传 JD + 简历，输出结构化优化建议（LlamaIndex + RAG + Milvus/pgvector）",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index = None
_documents = None
_retriever: Optional[JDRetriever] = None
_retriever_mode: str = "keyword"
_llm = None

# 向量库探测结果做短 TTL 缓存，避免每次 /health 都重新连库
_info_cache: Optional[VectorStoreInfoResponse] = None
_info_cache_ts: float = 0.0
_INFO_TTL = 5.0


def _vector_store_info(force_refresh: bool = False) -> VectorStoreInfoResponse:
    global _info_cache, _info_cache_ts
    now = time.time()
    if force_refresh or _info_cache is None or now - _info_cache_ts > _INFO_TTL:
        _info_cache = VectorStoreInfoResponse(**describe_vector_store().as_dict())
        _info_cache_ts = now
    return _info_cache


def _rebuild(force: bool, reset_store: bool = False) -> None:
    global _index, _documents, _retriever, _retriever_mode
    _index, _documents, _retriever_mode = build_knowledge_base(
        force=force, reset_store=reset_store
    )
    _retriever = JDRetriever(vector_index=_index, documents=_documents, mode=_retriever_mode)
    _vector_store_info(force_refresh=True)


@app.on_event("startup")
async def _startup() -> None:
    global _llm
    _rebuild(force=False)
    if settings.use_openai_llm:
        from llama_index.llms.openai import OpenAI

        _llm = OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            api_base=settings.openai_base_url,
        )
    logger.info(
        "启动完成：vector_store=%s retriever_mode=%s", settings.vector_store, _retriever_mode
    )


@app.get("/")
async def root() -> dict:
    return {
        "name": "Resume Optimization Agent",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "vectors": "/vectors",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_mode="openai" if settings.use_openai_llm else "mock",
        vector_store=settings.vector_store,
        retriever_mode=_retriever_mode,
        vector_store_info=_vector_store_info(),
    )


@app.get("/vectors", response_model=VectorStoreInfoResponse)
async def vectors(refresh: bool = False) -> VectorStoreInfoResponse:
    """查看当前向量库后端状态：连接、集合/表、维度、条目数、度量方式。"""
    return _vector_store_info(force_refresh=refresh)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if _retriever is None:
        raise HTTPException(503, "知识库尚未就绪，请先 POST /ingest")
    return await analyze_resume(req, _retriever, _llm)


def _decode_upload(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("gbk", errors="ignore")


@app.post("/upload", response_model=AnalyzeResponse)
async def upload(
    jd_file: UploadFile = File(...),
    resume_file: UploadFile = File(...),
    top_k: int = Form(5),
) -> AnalyzeResponse:
    if _retriever is None:
        raise HTTPException(503, "知识库尚未就绪，请先 POST /ingest")
    jd_text = _decode_upload(await jd_file.read(), jd_file.filename)
    resume_text = _decode_upload(await resume_file.read(), resume_file.filename)
    req = AnalyzeRequest(jd_text=jd_text, resume_text=resume_text, top_k=top_k)
    return await analyze_resume(req, _retriever, _llm)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(force: bool = True, reset: bool = False) -> IngestResponse:
    """重建知识库。reset=True 时先清空向量库历史数据（Milvus 集合 / pgvector 表 / simple 文件）。"""
    _rebuild(force=force, reset_store=reset)
    info = _vector_store_info(force_refresh=True)
    return IngestResponse(
        status="ok",
        vector_store=settings.vector_store,
        retriever_mode=_retriever_mode,
        retriever_ready=_retriever is not None,
        reset=reset,
        documents=len(_documents or []),
        vector_store_info=info,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    if _retriever is None:
        raise HTTPException(503, "知识库尚未就绪，请先 POST /ingest")
    eval_data = _load_eval_data()
    if not eval_data:
        raise HTTPException(404, f"评估集为空或不存在: {settings.eval_data_path}")
    result = evaluate_recall_at_k(
        eval_data,
        retrieve_fn=lambda q: _retriever.retrieve_ids(q),
        k=settings.top_k,
    )
    return MetricsResponse(
        metric=result["metric"],
        value=result["value"],
        target=settings.recall_target,
        passed=result["value"] >= settings.recall_target,
        num_queries=result["num_queries"],
        details=result["per_query"],
    )


def _load_eval_data() -> List[dict]:
    if not settings.eval_data_path.exists():
        return []
    items: List[dict] = []
    with open(settings.eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
