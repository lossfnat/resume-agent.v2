from __future__ import annotations

import json
import re
from typing import List

from app.config import settings
from app.retriever import JDRetriever
from app.schemas import AnalyzeRequest, AnalyzeResponse, SuggestionItem

SKILL_PATTERNS = [
    "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#",
    "react", "vue", "angular", "node", "next.js", "nuxt",
    "fastapi", "django", "flask", "spring", "springboot", "gin",
    "sql", "mysql", "postgresql", "mongodb", "redis", "kafka", "rabbitmq",
    "elasticsearch", "clickhouse",
    "pytorch", "tensorflow", "pandas", "numpy", "spark", "flink", "hive", "etl",
    "llm", "rag", "nlp",
    "docker", "kubernetes", "k8s", "aws", "gcp", "azure", "ci/cd", "git", "linux",
    "streamlit", "llamaindex", "langchain",
]

SKILL_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in SKILL_PATTERNS if s.isascii()) + r")\b",
    re.IGNORECASE,
)

CN_SKILL_KEYWORDS = [
    "机器学习", "深度学习", "自然语言处理", "推荐系统", "数据挖掘",
    "微服务", "分布式", "高并发", "高可用", "数据中台",
]


def extract_skills_from_jd(jd_text: str) -> List[str]:
    found = set()
    for m in SKILL_RE.finditer(jd_text):
        found.add(m.group(1).lower())
    for kw in CN_SKILL_KEYWORDS:
        if kw in jd_text:
            found.add(kw)
    return sorted(found, key=lambda s: (-len(s), s))


def _compare_skills(jd_skills: List[str], resume_text: str):
    resume_lower = resume_text.lower()
    matched, missing = [], []
    for s in jd_skills:
        if s.lower() in resume_lower:
            matched.append(s)
        else:
            missing.append(s)
    return matched, missing


def _mock_suggestions(jd_skills, matched, missing, retrieved) -> List[SuggestionItem]:
    items: List[SuggestionItem] = []
    if missing:
        items.append(
            SuggestionItem(
                category="skills",
                severity="high",
                issue=f"简历未体现 JD 中要求的关键技能：{', '.join(missing)}",
                suggestion=(
                    "在『专业技能』或『项目经历』中补充相关技能描述，"
                    "并尽量结合具体项目给出量化成果。"
                ),
            )
        )
    if retrieved:
        items.append(
            SuggestionItem(
                category="experience",
                severity="medium",
                issue="项目经历描述与目标 JD 强调的能力匹配度不足",
                suggestion=(
                    f"已召回 {len(retrieved)} 个相似 JD 样本作为参考，"
                    "在项目描述中显式体现 JD 强调的能力（端到端交付 / 跨团队协作 / 高并发场景）。"
                ),
            )
        )
    items.append(
        SuggestionItem(
            category="structure",
            severity="low",
            issue="简历缺少可量化的成果指标",
            suggestion="项目经历使用 STAR 法则重写，并加入数字（QPS、用户数、提升百分比）。",
        )
    )
    if not missing:
        items.append(
            SuggestionItem(
                category="skills",
                severity="low",
                issue="JD 关键技能已基本覆盖",
                suggestion="可在简历顶部增加一个『技能匹配度』小结，方便快速识别。",
            )
        )
    return items


async def _llm_suggestions(req: AnalyzeRequest, jd_skills, matched, missing, retrieved, llm):
    context = "\n\n".join(
        f"[相似JD样本 {i+1}] {r['title']}\n{r['text'][:600]}"
        for i, r in enumerate(retrieved[:3])
    )
    prompt = (
        "你是一名资深技术招聘官，请基于以下信息给出简历优化建议。\n\n"
        f"【目标 JD】\n{req.jd_text}\n\n"
        f"【JD 核心技能】\n{', '.join(jd_skills) or '（无）'}\n\n"
        f"【当前简历】\n{req.resume_text}\n\n"
        f"【已匹配技能】{', '.join(matched) or '（无）'}\n"
        f"【缺失技能】{', '.join(missing) or '（无）'}\n\n"
        f"【相似 JD 参考】\n{context or '（无）'}\n\n"
        "请输出严格的 JSON 数组，每项包含：category, severity(high/medium/low), "
        "issue, suggestion。仅输出 JSON，不要解释。"
    )
    try:
        resp = await llm.acomplete(prompt)
        text = resp.text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        arr = json.loads(text)
        return [SuggestionItem(**item) for item in arr]
    except Exception:
        return _mock_suggestions(jd_skills, matched, missing, retrieved)


async def analyze_resume(req: AnalyzeRequest, retriever: JDRetriever, llm=None) -> AnalyzeResponse:
    jd_skills = extract_skills_from_jd(req.jd_text)
    retrieved = retriever.retrieve(req.jd_text, top_k=req.top_k)
    retrieved_titles = [r["title"] for r in retrieved]
    matched, missing = _compare_skills(jd_skills, req.resume_text)

    if settings.use_openai_llm and llm is not None:
        suggestions = await _llm_suggestions(req, jd_skills, matched, missing, retrieved, llm)
    else:
        suggestions = _mock_suggestions(jd_skills, matched, missing, retrieved)

    return AnalyzeResponse(
        jd_skills=jd_skills,
        matched_skills=matched,
        missing_skills=missing,
        suggestions=suggestions,
        retrieved_context=retrieved_titles,
        llm_mode="openai" if settings.use_openai_llm else "mock",
    )
