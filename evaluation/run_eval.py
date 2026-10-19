from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.ingest import build_knowledge_base  # noqa: E402
from app.metrics import evaluate_recall_at_k  # noqa: E402
from app.retriever import JDRetriever  # noqa: E402
from app.vectorstore import describe_vector_store  # noqa: E402


def load_eval_data() -> list[dict]:
    items: list[dict] = []
    with open(settings.eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> None:
    print(f"[INFO] vector_store    = {settings.vector_store}")
    print(f"[INFO] embed_mode      = {settings.embed_mode}")
    print(f"[INFO] hf_endpoint     = {settings.hf_endpoint or '(未设)'}")
    print(f"[INFO] top_k           = {settings.top_k}")
    print(f"[INFO] recall_target   = {settings.recall_target}")
    print("[INFO] 构建知识库 ...")

    index, documents, mode = build_knowledge_base(force=True)
    print(f"[INFO] 检索模式 = {mode} (vector/keyword)")

    info = describe_vector_store()
    print(
        f"[INFO] 向量库后端 = {info.backend} available={info.available} "
        f"ready={info.ready} count={info.count} dim={info.dimension} metric={info.metric}"
    )

    retriever = JDRetriever(vector_index=index, documents=documents, mode=mode)
    eval_data = load_eval_data()
    print(f"[INFO] 评估集大小 = {len(eval_data)}")

    result = evaluate_recall_at_k(
        eval_data,
        retrieve_fn=lambda q: retriever.retrieve_ids(q),
        k=settings.top_k,
    )

    passed = result["value"] >= settings.recall_target
    print(f"\n===== {result['metric']} =====")
    print(f"值:      {result['value']:.4f}")
    print(f"目标:    {settings.recall_target}")
    print(f"是否达标: {'是' if passed else '否'}")
    print("\n每个 query 的明细：")
    for item in result["per_query"]:
        print(f"  - recall={item['recall']:.2f} | query={item['query'][:50]}")
        print(f"      retrieved = {item['retrieved']}")
        print(f"      relevant  = {item['relevant']}")

    out_dir = ROOT / "eval_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "metrics.json"
    payload = {
        "metric": result["metric"],
        "value": result["value"],
        "target": settings.recall_target,
        "passed": passed,
        "retriever_mode": mode,
        "vector_store": settings.vector_store,
        "vector_store_info": info.as_dict(),
        "num_queries": result["num_queries"],
        "per_query": result["per_query"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 评估结果已写入: {out_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
