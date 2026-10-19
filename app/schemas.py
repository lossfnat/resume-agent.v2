from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    jd_text: str = Field(..., description="目标岗位 JD 全文")
    resume_text: str = Field(..., description="待优化的简历全文")
    top_k: int = Field(default=5, description="检索召回的 JD 样本数")


class SuggestionItem(BaseModel):
    category: str = Field(..., description="建议类别: skills / experience / projects / structure")
    severity: str = Field(..., description="严重程度: high / medium / low")
    issue: str = Field(..., description="发现的问题")
    suggestion: str = Field(..., description="具体的修改建议")


class AnalyzeResponse(BaseModel):
    jd_skills: List[str] = Field(default_factory=list, description="JD 中提取的核心技能")
    matched_skills: List[str] = Field(default_factory=list, description="简历中已匹配的技能")
    missing_skills: List[str] = Field(default_factory=list, description="简历中缺失的关键技能")
    suggestions: List[SuggestionItem] = Field(default_factory=list)
    retrieved_context: List[str] = Field(default_factory=list)
    llm_mode: str = Field(..., description="mock / openai")


class MetricsResponse(BaseModel):
    metric: str
    value: float
    target: float
    passed: bool
    num_queries: int
    details: List[dict] = Field(default_factory=list)


class VectorStoreInfoResponse(BaseModel):
    backend: str = Field(..., description="simple / milvus / pgvector")
    available: bool = Field(..., description="向量库后端是否可连接")
    ready: bool = Field(..., description="集合 / 表是否已就绪")
    target: str = Field("", description="集合名 / 表名 / 持久化目录")
    dimension: Optional[int] = Field(None, description="向量维度")
    count: Optional[int] = Field(None, description="已入库向量数")
    metric: Optional[str] = Field(None, description="相似度度量方式")
    error: Optional[str] = Field(None, description="探测异常信息")


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_mode: str
    vector_store: str
    retriever_mode: str = Field("vector", description="vector / keyword (fallback)")
    vector_store_info: VectorStoreInfoResponse


class IngestResponse(BaseModel):
    status: str
    vector_store: str
    retriever_mode: str
    retriever_ready: bool
    reset: bool = Field(False, description="本次是否清空了向量库历史数据")
    documents: int = Field(0, description="本次入库的 JD 文档数")
    vector_store_info: VectorStoreInfoResponse
