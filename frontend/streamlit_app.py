from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_API = "http://localhost:8000"

SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}


def call_health(api_url: str):
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_analyze(api_url: str, jd_text: str, resume_text: str, top_k: int):
    resp = requests.post(
        f"{api_url}/analyze",
        json={"jd_text": jd_text, "resume_text": resume_text, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def call_metrics(api_url: str):
    try:
        resp = requests.get(f"{api_url}/metrics", timeout=120)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"评估失败：{e}")
        return None


def render_skills(skills: list[str]) -> str:
    if not skills:
        return "—"
    return "  ".join(f"`{s}`" for s in skills)


def main() -> None:
    st.set_page_config(page_title="Resume Optimization Agent", page_icon="📝", layout="wide")
    st.title("📝 简历优化 Agent")
    st.caption("基于 LlamaIndex + RAG 的简历优化建议系统 ｜ FastAPI 后端 + Streamlit 前端")

    with st.sidebar:
        st.header("⚙️ 配置")
        api_url = st.text_input("FastAPI 后端地址", value=DEFAULT_API)
        health = call_health(api_url)
        if health:
            st.success(
                f"后端就绪 ｜ LLM={health.get('llm_mode')} ｜ 向量库={health.get('vector_store')}"
                f" ｜ 检索={health.get('retriever_mode')}"
            )
            info = health.get("vector_store_info") or {}
            st.caption(
                f"向量库后端：`{info.get('backend')}` ｜ 可连接={info.get('available')} "
                f"就绪={info.get('ready')} ｜ 条目={info.get('count')} "
                f"维度={info.get('dimension')} 度量={info.get('metric')}"
            )
        else:
            st.warning("后端未启动，请先运行 `uvicorn app.main:app --port 8000`")

        st.divider()
        st.header("📊 量化指标")
        if st.button("跑 recall@k 评估", use_container_width=True):
            with st.spinner("评估中（首次会构建索引）..."):
                m = call_metrics(api_url)
            if m:
                passed = m["passed"]
                recall_pct = m["value"] * 100
                target_pct = m["target"] * 100
                st.metric(
                    label=m["metric"],
                    value=f"{recall_pct:.1f}%",
                    delta=f"目标 {target_pct:.0f}%",
                    delta_color="normal" if passed else "inverse",
                )
                if passed:
                    st.success(f"已达标 🎯 ({recall_pct:.1f}% >= {target_pct:.0f}%)")
                else:
                    st.warning(f"未达标 ⚠️ ({recall_pct:.1f}% < {target_pct:.0f}%)")
                with st.expander("每个 query 的明细"):
                    for d in m["details"]:
                        st.write(f"- recall={d['recall']*100:.0f}% | {d['query']}")
                        st.caption(f"  retrieved: {d.get('retrieved', [])}")

    st.subheader("输入")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎯 目标 JD**")
        jd_mode = st.radio("输入方式", ["文本", "上传文件"], horizontal=True, key="jd_mode")
        jd_text = ""
        if jd_mode == "文本":
            jd_text = st.text_area(
                "粘贴 JD",
                height=300,
                key="jd_text",
                placeholder="在此粘贴目标岗位的 JD 全文 ...",
                label_visibility="collapsed",
            )
        else:
            jd_file = st.file_uploader("上传 JD (.txt/.md)", type=["txt", "md"])
            if jd_file is not None:
                jd_text = jd_file.read().decode("utf-8", errors="ignore")
    with col2:
        st.markdown("**📄 我的简历**")
        resume_mode = st.radio("输入方式", ["文本", "上传文件"], horizontal=True, key="resume_mode")
        resume_text = ""
        if resume_mode == "文本":
            resume_text = st.text_area(
                "粘贴简历",
                height=300,
                key="resume_text",
                placeholder="在此粘贴简历全文 ...",
                label_visibility="collapsed",
            )
        else:
            resume_file = st.file_uploader("上传简历 (.txt/.md)", type=["txt", "md"])
            if resume_file is not None:
                resume_text = resume_file.read().decode("utf-8", errors="ignore")
        if st.button("📥 载入示例简历"):
            sample_path = ROOT / "data" / "resume_samples" / "sample_resume.txt"
            if sample_path.exists():
                st.session_state["resume_text"] = sample_path.read_text(encoding="utf-8")
                st.rerun()

    top_k = st.slider("召回 JD 样本数 (top_k)", 1, 8, 5)

    analyze_disabled = not (jd_text.strip() and resume_text.strip())
    if st.button("🚀 开始分析", type="primary", disabled=analyze_disabled, use_container_width=False):
        with st.spinner("分析中 ..."):
            try:
                result = call_analyze(api_url, jd_text, resume_text, top_k)
            except Exception as e:
                st.error(f"分析失败：{e}")
                return

        st.divider()
        st.header("🎯 分析结果")

        st.subheader("1️⃣ 从 JD 提取的核心技能")
        st.markdown(render_skills(result["jd_skills"]))

        st.subheader("2️⃣ 简历匹配情况")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("##### ✅ 已匹配")
            st.markdown(render_skills(result["matched_skills"]))
        with cm2:
            st.markdown("##### ❌ 缺失")
            st.markdown(render_skills(result["missing_skills"]))

        st.subheader("3️⃣ 检索召回的相似 JD 样本（可解释性）")
        if result["retrieved_context"]:
            for t in result["retrieved_context"]:
                st.markdown(f"- `{t}`")
        else:
            st.write("—")

        st.subheader("4️⃣ 修改建议")
        for i, sug in enumerate(result["suggestions"], 1):
            sev = str(sug.get("severity", "")).lower()
            icon = SEVERITY_ICON.get(sev, "⚪")
            with st.container(border=True):
                st.markdown(
                    f"**{i}. {icon} [{sug['category']}] {sug['issue']}**  ·  严重程度: `{sev}`"
                )
                st.markdown(f"💡 建议：{sug['suggestion']}")

        st.caption(f"LLM 模式：`{result['llm_mode']}`")


if __name__ == "__main__":
    main()
