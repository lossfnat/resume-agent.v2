from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.vectorstore import (  # noqa: E402
    _milvus_metric,
    create_vector_store,
    describe_vector_store,
    embed_dim_of,
    reset_vector_store,
)


def test_effective_milvus_uri_local(monkeypatch):
    monkeypatch.setattr(settings, "milvus_mode", "local")
    monkeypatch.setattr(settings, "milvus_lite_path", Path("/data/milvus.db"))
    assert settings.effective_milvus_uri.endswith("milvus.db")


def test_effective_milvus_uri_remote(monkeypatch):
    monkeypatch.setattr(settings, "milvus_mode", "remote")
    monkeypatch.setattr(settings, "milvus_uri", "http://milvus:19530")
    assert settings.effective_milvus_uri == "http://milvus:19530"


def test_pg_connection_string(monkeypatch):
    monkeypatch.setattr(settings, "pg_user", "postgres")
    monkeypatch.setattr(settings, "pg_password", "postgres")
    monkeypatch.setattr(settings, "pg_host", "localhost")
    monkeypatch.setattr(settings, "pg_port", "5432")
    monkeypatch.setattr(settings, "pg_database", "resume_agent")
    assert settings.pg_connection_string == (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/resume_agent"
    )


def test_default_embed_dim(monkeypatch):
    monkeypatch.setattr(settings, "vector_dim", 0)
    assert embed_dim_of(None) == 384


def test_embed_dim_override(monkeypatch):
    monkeypatch.setattr(settings, "vector_dim", 768)
    assert embed_dim_of(None) == 768


def test_create_simple_store(monkeypatch):
    from llama_index.core.vector_stores import SimpleVectorStore

    monkeypatch.setattr(settings, "vector_store", "simple")
    store = create_vector_store(384)
    assert isinstance(store, SimpleVectorStore)


def test_create_unknown_backend_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "vector_store", "not-a-backend")
    assert create_vector_store(384) is None


def test_milvus_metric_normalization(monkeypatch):
    monkeypatch.setattr(settings, "milvus_metric", "cosine")
    assert _milvus_metric() == "COSINE"
    monkeypatch.setattr(settings, "milvus_metric", "NOT_A_METRIC")
    assert _milvus_metric() == "COSINE"


def test_describe_simple_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "vector_store", "simple")
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    info = describe_vector_store()
    assert info.backend == "simple"
    assert info.available is True
    assert info.ready is False
    assert info.count is None


def test_describe_simple_with_vectors(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "vector_store", "simple")
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    payload = {"embedding_dict": {"n1": [0.1, 0.2, 0.3], "n2": [0.4, 0.5, 0.6]}}
    (tmp_path / "default__vector_store.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    info = describe_vector_store()
    assert info.ready is True
    assert info.count == 2
    assert info.dimension == 3


def test_describe_pgvector_unavailable_no_raise(monkeypatch):
    monkeypatch.setattr(settings, "vector_store", "pgvector")
    monkeypatch.setattr(settings, "pg_host", "127.0.0.1")
    monkeypatch.setattr(settings, "pg_port", "1")  # 必然不可达
    info = describe_vector_store()
    assert info.backend == "pgvector"
    assert info.available is False
    assert info.error
    assert info.target == f"{settings.pg_schema}.{settings.pg_table}"


def test_reset_simple_removes_persist_files(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "vector_store", "simple")
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    (tmp_path / "default__vector_store.json").write_text("{}", encoding="utf-8")
    assert reset_vector_store() is True
    assert list(tmp_path.glob("*.json")) == []
