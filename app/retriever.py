"""检索器：向量检索 + TF-IDF 关键词检索 fallback。

JDRetriever 根据初始化时传入的模式自动选择检索策略：
  - "vector"：LlamaIndex VectorStoreIndex 相似度检索
  - "keyword"：sklearn TfidfVectorizer 关键词匹配（零网络依赖）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    id: str
    title: str
    text: str


class JDRetriever:
    def __init__(
        self,
        vector_index: Optional[object] = None,
        documents: Optional[List[DocumentRecord]] = None,
        mode: str = "vector",
    ):
        self.mode = mode
        self._documents = documents or []
        self._vector_index = vector_index

        if mode == "vector" and vector_index is not None:
            self._llama_retriever = vector_index.as_retriever(similarity_top_k=settings.top_k)
        elif mode == "keyword" and self._documents:
            self._init_keyword_index()
        else:
            self._llama_retriever = None
            self._tfidf = None
            self._doc_vectors = None

    def _init_keyword_index(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            texts = [d.text for d in self._documents]
            self._doc_vectors = self._tfidf.fit_transform(texts)
            logger.info("TF-IDF 关键词索引构建完成，文档数=%d", len(self._documents))
        except ImportError:
            logger.error("scikit-learn 未安装，关键词检索不可用")
            self._tfidf = None
            self._doc_vectors = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        k = top_k or settings.top_k
        if self.mode == "vector" and self._llama_retriever is not None:
            return self._retrieve_vector(query, k)
        return self._retrieve_keyword(query, k)

    def retrieve_ids(self, query: str, top_k: Optional[int] = None) -> List[str]:
        return [r["id"] for r in self.retrieve(query, top_k)]

    def _retrieve_vector(self, query: str, k: int) -> List[dict]:
        self._llama_retriever.similarity_top_k = k
        nodes = self._llama_retriever.retrieve(query)
        results: List[dict] = []
        for n in nodes:
            meta = getattr(n.node, "metadata", {}) or {}
            doc_id = meta.get("source_file", n.node.id_)
            results.append(
                {
                    "id": doc_id,
                    "title": meta.get("source_file", "unknown"),
                    "text": n.node.text,
                    "score": float(n.score) if n.score else 0.0,
                }
            )
        return results

    def _retrieve_keyword(self, query: str, k: int) -> List[dict]:
        if not self._documents:
            return []
        if self._tfidf is None or self._doc_vectors is None:
            return self._retrieve_simple(query, k)

        import numpy as np

        q_vec = self._tfidf.transform([query])
        scores = (self._doc_vectors * q_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            d = self._documents[int(idx)]
            results.append(
                {
                    "id": d.id,
                    "title": d.title,
                    "text": d.text,
                    "score": float(scores[idx]),
                }
            )
        return results

    def _retrieve_simple(self, query: str, k: int) -> List[dict]:
        query_lower = query.lower()
        scored = []
        for d in self._documents:
            text_lower = d.text.lower()
            count = sum(1 for w in query_lower.split() if w in text_lower and len(w) > 1)
            if count > 0:
                scored.append((count, d))
        scored.sort(key=lambda x: -x[0])
        results: List[dict] = []
        for score, d in scored[:k]:
            results.append({"id": d.id, "title": d.title, "text": d.text, "score": float(score)})
        return results
