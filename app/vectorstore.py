"""向量库统一接入层：SimpleVectorStore / MilvusVectorStore / PGVectorStore。

提供三个能力，供 ingest / main 复用：
  - create_vector_store(dim, overwrite)：按 settings.vector_store 构建向量库实例
  - describe_vector_store(dim)：探测连接状态、集合/表、维度、条目数、度量方式（免重建索引）
  - reset_vector_store()：删除 Milvus 集合 / pgvector 表 / simple 持久化文件，用于重建

设计原则：任何后端不可用（依赖缺失 / 连接失败）时返回 None 或 available=False，
由调用方决定是否降级到关键词检索，绝不抛异常打断主流程。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("simple", "milvus", "pgvector")
_MILVUS_METRICS = {"IP", "COSINE", "L2"}
_DEFAULT_DIM = 384  # all-MiniLM-L6-v2 维度，仅作探测失败时的兜底


@dataclass
class VectorStoreInfo:
    backend: str
    available: bool = False  # 后端是否可连接
    ready: bool = False  # 集合 / 表是否已就绪
    target: str = ""  # 集合名 / 表名 / 持久化目录
    dimension: Optional[int] = None
    count: Optional[int] = None  # 已入库向量数
    metric: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "ready": self.ready,
            "target": self.target,
            "dimension": self.dimension,
            "count": self.count,
            "metric": self.metric,
            "error": self.error,
        }


def embed_dim_of(embed_model: Any, fallback: int = _DEFAULT_DIM) -> int:
    """从 embedding 模型自动探测向量维度；settings.vector_dim>0 时以显式配置为准。"""
    if settings.vector_dim > 0:
        return settings.vector_dim
    try:
        model = getattr(embed_model, "_model", None)
        if model is not None:
            # sentence-transformers 新老版本方法名不同，依次尝试
            for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    dim = fn()
                    if dim:
                        return int(dim)
    except Exception:  # noqa: BLE001 - 探测失败不应影响主流程
        pass
    try:
        return len(embed_model.get_text_embedding("dimension probe"))
    except Exception as e:  # noqa: BLE001
        logger.warning("无法探测 embedding 维度，回退默认值 %d：%s", fallback, e)
        return fallback


def create_vector_store(dim: int, overwrite: bool = False) -> Optional[Any]:
    """按 settings.vector_store 构建向量库实例，失败返回 None。"""
    backend = settings.vector_store
    try:
        if backend == "simple":
            return _create_simple()
        if backend == "milvus":
            return _create_milvus(dim, overwrite=overwrite)
        if backend == "pgvector":
            return _create_pgvector(dim)
    except Exception as e:  # noqa: BLE001
        logger.error("构建 %s 向量库失败：%s", backend, e)
        return None
    logger.error("未知的 VECTOR_STORE=%s，可选：%s", backend, SUPPORTED_BACKENDS)
    return None


def _create_simple() -> Any:
    from llama_index.core.vector_stores import SimpleVectorStore

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return SimpleVectorStore()


def _milvus_metric() -> str:
    return settings.milvus_metric if settings.milvus_metric in _MILVUS_METRICS else "COSINE"


def _create_milvus(dim: int, overwrite: bool) -> Any:
    from llama_index.vector_stores.milvus import IndexManagement, MilvusVectorStore

    if settings.milvus_mode != "remote":
        settings.milvus_lite_path.parent.mkdir(parents=True, exist_ok=True)

    uri = settings.effective_milvus_uri
    metric = _milvus_metric()
    logger.info(
        "接入 Milvus：mode=%s uri=%s collection=%s dim=%s metric=%s overwrite=%s",
        settings.milvus_mode, uri, settings.milvus_collection, dim, metric, overwrite,
    )
    store = MilvusVectorStore(
        uri=uri,
        token=settings.milvus_token,
        collection_name=settings.milvus_collection,
        dim=dim,
        overwrite=overwrite,
        similarity_metric=metric,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    )
    _ensure_milvus_loaded(store)
    return store


def _ensure_milvus_loaded(store: Any) -> None:
    """复用已存在的集合时必须 load()：Milvus 进程重启后集合处于 released 状态，否则检索报错。"""
    try:
        if settings.milvus_collection in list(store.client.list_collections()):
            store.client.load_collection(settings.milvus_collection)
    except Exception as e:  # noqa: BLE001
        logger.warning("加载 Milvus 集合失败：%s", e)


def _create_pgvector(dim: int) -> Any:
    from llama_index.vector_stores.postgres import PGVectorStore

    hnsw_kwargs = None
    if settings.pg_hnsw:
        hnsw_kwargs = {
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        }
    logger.info(
        "接入 pgvector：%s table=%s.%s dim=%s hnsw=%s",
        settings.pg_connection_string, settings.pg_schema, settings.pg_table, dim, settings.pg_hnsw,
    )
    return PGVectorStore.from_params(
        connection_string=settings.pg_connection_string,
        table_name=settings.pg_table,
        schema_name=settings.pg_schema,
        embed_dim=dim,
        use_jsonb=settings.pg_use_jsonb,
        hnsw_kwargs=hnsw_kwargs,
    )


def describe_vector_store() -> VectorStoreInfo:
    """探测当前向量库状态，供 /health、/vectors 使用。全流程 try/except 兜底。"""
    backend = settings.vector_store
    info = VectorStoreInfo(backend=backend)
    try:
        if backend == "simple":
            _describe_simple(info)
        elif backend == "milvus":
            _describe_milvus(info)
        elif backend == "pgvector":
            _describe_pgvector(info)
        else:
            info.error = f"未知 VECTOR_STORE={backend}，可选：{SUPPORTED_BACKENDS}"
    except Exception as e:  # noqa: BLE001
        info.error = str(e)
        logger.warning("探测 %s 向量库异常：%s", backend, e)
    return info


def _describe_simple(info: VectorStoreInfo) -> None:
    info.target = str(settings.storage_dir)
    info.metric = "cosine"
    info.available = settings.storage_dir.exists()
    vec_file = settings.storage_dir / "default__vector_store.json"
    if not vec_file.exists():
        return
    raw = json.loads(vec_file.read_text(encoding="utf-8"))
    embedding_dict = raw.get("embedding_dict", {}) or {}
    info.count = len(embedding_dict)
    info.ready = info.count > 0
    first = next(iter(embedding_dict.values()), None)
    if isinstance(first, list):
        info.dimension = len(first)


def _describe_milvus(info: VectorStoreInfo) -> None:
    from pymilvus import DataType, MilvusClient

    info.target = settings.milvus_collection
    info.metric = _milvus_metric()
    if settings.milvus_mode != "remote":
        settings.milvus_lite_path.parent.mkdir(parents=True, exist_ok=True)

    client = MilvusClient(uri=settings.effective_milvus_uri, token=settings.milvus_token)
    try:
        info.available = True
        if settings.milvus_collection not in list(client.list_collections()):
            return
        info.ready = True
        try:
            stats = client.get_collection_stats(settings.milvus_collection)
            info.count = int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001 - 统计失败不影响其他信息
            pass
        try:
            desc = client.describe_collection(settings.milvus_collection)
            for f in desc.get("fields", []):
                if f.get("type") == DataType.FLOAT_VECTOR:
                    dim = (f.get("params") or {}).get("dim")
                    if dim is not None:
                        info.dimension = int(dim)
        except Exception:  # noqa: BLE001
            pass
    finally:
        client.close()


def _describe_pgvector(info: VectorStoreInfo) -> None:
    from sqlalchemy import create_engine, text

    info.target = f"{settings.pg_schema}.{settings.pg_table}"
    info.metric = "cosine"
    qualified = f"{settings.pg_schema}.{settings.pg_table}"

    engine = create_engine(settings.pg_connection_string, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            info.available = True
            exists = conn.execute(
                text("SELECT to_regclass(:tbl) IS NOT NULL"), {"tbl": qualified}
            ).scalar()
            if not exists:
                return
            info.ready = True
            info.count = int(
                conn.execute(
                    text(f'SELECT count(*) FROM "{settings.pg_schema}"."{settings.pg_table}"')
                ).scalar()
                or 0
            )
            dim = conn.execute(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = :tbl::regclass AND attname = 'embedding'"
                ),
                {"tbl": qualified},
            ).scalar()
            if dim and int(dim) > 0:
                info.dimension = int(dim)
    finally:
        engine.dispose()


def reset_vector_store() -> bool:
    """删除当前后端的数据（集合 / 表 / 持久化文件），用于强制重建。"""
    backend = settings.vector_store
    try:
        if backend == "simple":
            for f in settings.storage_dir.glob("*.json"):
                f.unlink()
            return True
        if backend == "milvus":
            from pymilvus import MilvusClient

            client = MilvusClient(uri=settings.effective_milvus_uri, token=settings.milvus_token)
            try:
                if settings.milvus_collection in list(client.list_collections()):
                    client.drop_collection(settings.milvus_collection)
            finally:
                client.close()
            return True
        if backend == "pgvector":
            from sqlalchemy import create_engine, text

            engine = create_engine(settings.pg_connection_string)
            try:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            f'DROP TABLE IF EXISTS "{settings.pg_schema}"."{settings.pg_table}"'
                        )
                    )
            finally:
                engine.dispose()
            return True
    except Exception as e:  # noqa: BLE001
        logger.error("重置 %s 向量库失败：%s", backend, e)
    return False


def simple_persist_dir() -> Optional[str]:
    """simple 后端需要把索引持久化到磁盘，其余后端由数据库自身持久化。"""
    if settings.vector_store == "simple":
        return str(settings.storage_dir)
    return None
