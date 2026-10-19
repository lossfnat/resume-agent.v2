from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    llm_mode: str = field(default_factory=lambda: os.getenv("LLM_MODE", "mock").lower())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    embed_mode: str = field(default_factory=lambda: os.getenv("EMBED_MODE", "local").lower())
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))

    hf_endpoint: str = field(default_factory=lambda: os.getenv("HF_ENDPOINT", ""))

    # ===== 向量库后端：simple | milvus | pgvector =====
    vector_store: str = field(default_factory=lambda: os.getenv("VECTOR_STORE", "simple").lower())

    # ----- Milvus -----
    # MILVUS_MODE: local（milvus-lite，本地文件，免 Docker）/ remote（连接 Milvus 服务）
    milvus_mode: str = field(default_factory=lambda: os.getenv("MILVUS_MODE", "local").lower())
    milvus_uri: str = field(default_factory=lambda: os.getenv("MILVUS_URI", "http://localhost:19530"))
    milvus_token: str = field(default_factory=lambda: os.getenv("MILVUS_TOKEN", ""))
    milvus_collection: str = field(
        default_factory=lambda: os.getenv("MILVUS_COLLECTION", "resume_agent_kb")
    )
    milvus_metric: str = field(
        default_factory=lambda: os.getenv("MILVUS_METRIC", "COSINE").upper()
    )
    milvus_lite_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MILVUS_LITE_PATH", str(PROJECT_ROOT / "storage" / "milvus.db"))
        )
    )

    # ----- pgvector -----
    pg_host: str = field(default_factory=lambda: os.getenv("PG_HOST", "localhost"))
    pg_port: str = field(default_factory=lambda: os.getenv("PG_PORT", "5432"))
    pg_database: str = field(default_factory=lambda: os.getenv("PG_DATABASE", "resume_agent"))
    pg_user: str = field(default_factory=lambda: os.getenv("PG_USER", "postgres"))
    pg_password: str = field(default_factory=lambda: os.getenv("PG_PASSWORD", "postgres"))
    pg_table: str = field(default_factory=lambda: os.getenv("PG_TABLE", "resume_agent_kb"))
    pg_schema: str = field(default_factory=lambda: os.getenv("PG_SCHEMA", "public"))
    pg_use_jsonb: bool = field(default_factory=lambda: _env_bool("PG_USE_JSONB", True))
    pg_hnsw: bool = field(default_factory=lambda: _env_bool("PG_HNSW", True))

    # ===== 通用 =====
    # VECTOR_DIM=0 表示从 embedding 模型自动探测维度
    vector_dim: int = field(default_factory=lambda: int(os.getenv("VECTOR_DIM", "0")))

    retriever_mode: str = field(default_factory=lambda: os.getenv("RETRIEVER_MODE", "auto").lower())
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    recall_target: float = field(default_factory=lambda: float(os.getenv("RECALL_TARGET", "0.85")))

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    jd_samples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "jd_samples")
    resume_samples_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "resume_samples"
    )
    storage_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage")
    eval_data_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "evaluation" / "eval_data.jsonl"
    )

    @property
    def use_openai_llm(self) -> bool:
        return self.llm_mode == "openai" and bool(self.openai_api_key)

    @property
    def use_openai_embed(self) -> bool:
        return self.embed_mode == "openai" and bool(self.openai_api_key)

    @property
    def effective_milvus_uri(self) -> str:
        """local 模式走 milvus-lite 本地文件；remote 模式走 Milvus 服务地址。"""
        if self.milvus_mode == "remote":
            return self.milvus_uri
        return str(self.milvus_lite_path)

    @property
    def pg_connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


settings = Settings()

if settings.hf_endpoint:
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
