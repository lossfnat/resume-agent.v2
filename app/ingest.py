"""构建知识库：真正接入 simple / milvus / pgvector 三种向量库后端。

流程（向量模式）：
  1. 加载 embedding 模型，失败 → 降级关键词
  2. 自动探测向量维度（可用 VECTOR_DIM 覆盖）
  3. force=False 时优先复用后端已有索引；否则按后端重建
  4. 任一环节不可用 → 降级到 TF-IDF 关键词检索（零网络依赖）

milvus-lite（MILVUS_MODE=local）为本地文件模式，无需 Docker 即可跑通 Milvus 检索。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import settings
from app.retriever import DocumentRecord
from app.vectorstore import (
    create_vector_store,
    describe_vector_store,
    embed_dim_of,
    reset_vector_store,
    simple_persist_dir,
)

logger = logging.getLogger(__name__)

IndexType = Optional[object]


def _load_documents() -> List[DocumentRecord]:
    if not settings.jd_samples_dir.exists():
        return []
    docs: List[DocumentRecord] = []
    for path in sorted(settings.jd_samples_dir.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(DocumentRecord(id=path.name, title=path.name, text=text))
    return docs


def build_knowledge_base(
    force: bool = True, reset_store: bool = False
) -> Tuple[IndexType, List[DocumentRecord], str]:
    documents = _load_documents()
    if not documents:
        logger.warning("JD 样本目录为空：%s", settings.jd_samples_dir)
        return None, [], "keyword"

    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    # 强制关键词模式：完全跳过向量构建
    if settings.retriever_mode == "keyword":
        logger.info("RETRIEVER_MODE=keyword，跳过向量构建，直接使用关键词检索")
        return None, documents, "keyword"

    embed_model = _load_embed_model()
    if embed_model is None:
        logger.warning("向量索引不可用，降级到关键词检索模式（零网络依赖）")
        return None, documents, "keyword"

    dim = embed_dim_of(embed_model)
    logger.info("使用向量库后端=%s，向量维度=%d", settings.vector_store, dim)

    # 复用已有索引（仅在非强制重建时）
    if not force and not reset_store:
        index = _try_load_existing(embed_model, dim)
        if index is not None:
            logger.info("复用已有向量索引（backend=%s）", settings.vector_store)
            return index, documents, "vector"

    # 强制重建：清空后端历史数据（milvus 由 overwrite 内建删除集合）
    if reset_store or force:
        if settings.vector_store != "milvus":
            reset_vector_store()

    store = create_vector_store(dim, overwrite=force)
    if store is None:
        logger.warning("向量库后端 %s 不可用，降级到关键词检索", settings.vector_store)
        return None, documents, "keyword"

    index = _build_index_from_documents(documents, store, embed_model)
    if index is not None:
        logger.info("知识库构建成功：向量模式（backend=%s）", settings.vector_store)
        return index, documents, "vector"

    logger.warning("向量索引构建失败，降级到关键词检索模式（零网络依赖）")
    return None, documents, "keyword"


def _build_index_from_documents(documents, store, embed_model) -> IndexType:
    from llama_index.core import Document, StorageContext, VectorStoreIndex

    llama_docs = [
        Document(text=d.text, metadata={"source_file": d.id}) for d in documents
    ]
    try:
        storage_context = StorageContext.from_defaults(vector_store=store)
        index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=False,
        )
        persist_dir = simple_persist_dir()
        if persist_dir:
            index.storage_context.persist(persist_dir=persist_dir)
        return index
    except Exception as e:  # noqa: BLE001
        logger.error("向量索引构建失败：%s", e)
        return None


def _try_load_existing(embed_model, dim) -> IndexType:
    """后端已有数据时复用；否则返回 None（触发重建）。"""
    info = describe_vector_store()
    if not info.ready:
        logger.info("后端 %s 尚无数据，将构建新索引", settings.vector_store)
        return None
    try:
        if settings.vector_store == "simple":
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(persist_dir=str(settings.storage_dir))
            return load_index_from_storage(storage_context, embed_model=embed_model)

        from llama_index.core import VectorStoreIndex

        store = create_vector_store(dim, overwrite=False)
        if store is None:
            return None
        return VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
    except Exception as e:  # noqa: BLE001
        logger.warning("加载已有索引失败，将重建：%s", e)
        return None


def _load_embed_model() -> Optional[object]:
    if settings.use_openai_embed:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(
                model=settings.embed_model,
                api_key=settings.openai_api_key,
                api_base=settings.openai_base_url,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("OpenAI embedding 加载失败（已降级关键词）：%s", e)
            return None

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=settings.embed_model)
    except Exception as e:  # noqa: BLE001
        logger.error("Embedding 模型加载失败（可能是网络问题）：%s", e)
        return None
