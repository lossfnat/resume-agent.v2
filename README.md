# 📝 Resume Optimization Agent v2

> 上传 JD 和简历，输出结构化的简历修改建议 — 基于 **FastAPI + LlamaIndex + RAG** 的 AI 应用。
>
> v2（v0.2.0）重点：**Milvus / pgvector 深度整合** —— `VECTOR_STORE` 不再只是配置占位，而是真正可切换、可探测、可重建的向量库后端。

---

## ✨ 功能

- 🎯 **JD 技能抽取**：从目标岗位 JD 中自动提取核心技能要求
- 🔍 **RAG 语义召回**：检索相似 JD 样本，提供可解释的参考上下文
- 📊 **简历匹配分析**：比对简历与 JD 技能，识别缺失项
- 💡 **结构化修改建议**：按类别（skills / experience / structure）给出优先级建议
- 🔄 **双模式**：mock 规则模式（零 API Key）+ OpenAI LLM 模式
- 🗄️ **三向量库后端**：`simple` / `milvus` / `pgvector` 一键切换，真实接入
- 🧭 **向量库自省**：`/vectors` 查看连接状态、集合/表、维度、条目数、度量方式
- 🛡️ **自动降级**：embedding 或向量库不可用时，自动降级到 TF-IDF 关键词检索（零网络依赖）

---

## 🗄️ v2 新增：Milvus / pgvector 深度整合

| 后端 | 依赖 | 适用场景 |
|---|---|---|
| `simple` | 无（内存 + 本地 JSON） | 默认、零依赖，开箱即跑 |
| `milvus` | `MILVUS_MODE=local` 用 milvus-lite（本地文件，**免 Docker**）；`remote` 连 Milvus 服务 | 本地快速体验 / 生产级向量库 |
| `pgvector` | PostgreSQL + vector 扩展（`docker compose up -d pgvector`） | 与业务库同源，SQL 生态 |

整合点：

- **统一工厂** `app/vectorstore.py`：按 `VECTOR_STORE` 构建对应后端，失败返回 `None` 触发降级
- **生命周期管理**：`/ingest?force=true&reset=true` 会删除 Milvus 集合 / pgvector 表 / simple 持久化文件后重建
- **索引复用**：`force=false` 时优先复用后端已有索引，避免重复入库（Milvus 会自动 `load()` 集合）
- **维度自动探测**：从 embedding 模型读取维度（`VECTOR_DIM=0` 即自动），支持 `VECTOR_DIM` 显式覆盖
- **健康自省**：`/health` 与 `/vectors` 返回 `available / ready / target / dimension / count / metric`

### 只用 Milvus-lite（无需 Docker，推荐本地体验）

```bash
# .env
VECTOR_STORE=milvus
MILVUS_MODE=local
MILVUS_LITE_PATH=storage/milvus.db
MILVUS_COLLECTION=resume_agent_kb
```

### 用真实 Milvus 服务

```bash
docker compose up -d etcd minio milvus   # 起 Milvus standalone
# .env
VECTOR_STORE=milvus
MILVUS_MODE=remote
MILVUS_URI=http://localhost:19530
```

### 用 pgvector

```bash
docker compose up -d pgvector            # PostgreSQL 16 + vector 扩展
# .env
VECTOR_STORE=pgvector
PG_HOST=localhost
PG_TABLE=resume_agent_kb
```

> pgvector 后端默认使用 cosine 距离（`PGVectorStore` 内置），并默认创建 HNSW 索引（`PG_HNSW=true`）。

---

## 📊 量化指标

### 指标定义：`recall@5`

对于评估集中每条 query：

```
recall@k = |检索 top-k ∩ ground-truth 相关| / |ground-truth 相关|
```

整体指标 = 所有 query 的 recall@k 平均值。

- **评估集**：`evaluation/eval_data.jsonl`（10 条 query，覆盖 8 类岗位 JD）
- **目标**：`recall@5 ≥ 0.85`（在 `.env` 中通过 `RECALL_TARGET` 配置）

### 跑评估

```bash
python -m evaluation.run_eval
```

输出示例（Milvus-lite 后端）：

```
[INFO] 向量库后端 = milvus available=True ready=True count=8 dim=384 metric=COSINE
===== recall@5 =====
值:      1.0000
目标:    0.85
是否达标: 是
[INFO] 评估结果已写入: eval_output/metrics.json
```

各后端实测（本机）：

| 后端 / 模式 | recall@5 | 说明 |
|---|---|---|
| keyword（TF-IDF） | 0.95 | 零网络依赖 |
| simple + 向量 | 1.00 | sentence-transformers + 本地持久化 |
| milvus（lite） + 向量 | 1.00 | 免 Docker 真实 Milvus 检索 |

---

## 🏗️ 技术架构

```
用户输入 JD + 简历
       │
       ▼
┌─────────────────────────────┐
│  FastAPI 后端 (app/main.py) │
│  - /analyze  文本分析       │
│  - /upload   文件上传       │
│  - /ingest   重建知识库     │
│  - /vectors  向量库自省     │
│  - /metrics  跑量化评估     │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Agent 核心 (app/agent.py)  │
│  1. 从 JD 抽取技能          │
│  2. 检索相似 JD 样本        │
│  3. 比对简历                │
│  4. 生成修改建议            │
└──────┬──────────┬───────────┘
       │          │
       ▼          ▼
┌─────────────┐  ┌──────────────────────┐
│ 向量检索     │  │ 关键词检索 fallback   │
│ LlamaIndex  │  │ TF-IDF (sklearn)      │
│ +sentence-  │  │ 零网络依赖            │
│ transformers│  └──────────────────────┘
└──────┬──────┘
       │  向量库工厂 (app/vectorstore.py)
       ▼
┌──────────────────────────────────────┐
│ simple        │ milvus (lite/remote)  │
│ SimpleVector  │ MilvusVectorStore     │
│ Store         │                       │
├───────────────┼───────────────────────┤
│ pgvector      │ 失败 → 自动降级关键词 │
│ PGVectorStore │                       │
└──────────────────────────────────────┘
```

---

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + Pydantic |
| RAG 框架 | LlamaIndex |
| Embedding | sentence-transformers / OpenAI |
| 向量存储 | SimpleVectorStore / **Milvus** / **pgvector** |
| LLM（可选） | OpenAI GPT |
| 降级检索 | sklearn TfidfVectorizer |
| 前端 | Streamlit |

---

## 🚀 快速开始

### 0. 前置条件

- Python 3.10+
- pip
- （可选）Docker，用于 pgvector / Milvus 服务

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动后端

```bash
# .env 已内置 HF_ENDPOINT=https://hf-mirror.com（国内镜像）
uvicorn app.main:app --reload --port 8000
```

首次启动会按 `VECTOR_STORE` 构建知识库。向量模式下会自动下载 embedding 模型（~90MB，走国内镜像几秒完成）；若模型或向量库不可用，自动降级到 TF-IDF 关键词检索。

后端启动成功后：
- 健康检查（含向量库状态）：http://localhost:8000/health
- 向量库自省：http://localhost:8000/vectors
- API 文档：http://localhost:8000/docs

### 3. 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

浏览器自动打开 http://localhost:8501

### 4. 跑量化指标

```bash
python -m evaluation.run_eval
```

或在 Streamlit 侧边栏点 **"跑 recall@k 评估"** 按钮。

### 5. 跑单元测试

```bash
pytest tests -v
```

---

## 📁 项目结构

```
resume-agent.v2/
├── app/                        # FastAPI 后端
│   ├── __init__.py             # 版本号 0.2.0
│   ├── config.py               # 环境变量配置（含 Milvus/pgvector 参数）
│   ├── schemas.py              # Pydantic 请求/响应模型
│   ├── metrics.py              # recall@k 量化指标实现
│   ├── vectorstore.py          # 🆕 向量库工厂 + 自省 + 生命周期
│   ├── ingest.py               # 知识库构建（三后端 + 关键词 fallback）
│   ├── retriever.py            # JDRetriever（双模式检索）
│   ├── agent.py                # 简历分析核心逻辑
│   └── main.py                 # FastAPI 路由（含 /vectors）
├── data/
│   ├── jd_samples/             # 8 个岗位 JD（知识库文档）
│   └── resume_samples/        # 示例简历
├── evaluation/
│   ├── eval_data.jsonl         # 评估集（10 条 query + ground truth）
│   └── run_eval.py             # 评估脚本（打印向量库状态）
├── frontend/
│   └── streamlit_app.py        # Streamlit 前端（展示向量库状态）
├── tests/
│   ├── test_metrics.py         # 指标单元测试
│   └── test_vectorstore.py     # 🆕 向量库工厂/自省/降级测试
├── storage/                    # simple 持久化 + milvus-lite 数据文件
├── docker-compose.yml          # pgvector + Milvus standalone 一键启动
├── requirements.txt
├── .env.example                # 环境变量模板（含 Milvus/pgvector）
├── .gitignore
└── README.md
```

---

## ⚙️ 配置说明

所有配置通过 `.env` 文件或环境变量设置（环境变量优先级高于 `.env`）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODE` | mock | `mock`（规则生成）/ `openai`（LLM 生成） |
| `OPENAI_API_KEY` | 空 | OpenAI API Key（LLM_MODE=openai 时必填） |
| `OPENAI_BASE_URL` | https://api.openai.com/v1 | 可设代理地址 |
| `LLM_MODEL` | gpt-4o-mini | LLM 模型 |
| `EMBED_MODE` | local | `local`（sentence-transformers）/ `openai` |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Embedding 模型 |
| `HF_ENDPOINT` | 空 | HuggingFace 镜像（国内设 `https://hf-mirror.com`） |
| `VECTOR_STORE` | simple | `simple` / `milvus` / `pgvector` |
| `VECTOR_DIM` | 0 | 向量维度；`0` 表示从模型自动探测 |
| `MILVUS_MODE` | local | `local`（milvus-lite 本地文件）/ `remote`（连服务） |
| `MILVUS_URI` | http://localhost:19530 | remote 模式的服务地址 |
| `MILVUS_LITE_PATH` | storage/milvus.db | local 模式的本地数据文件 |
| `MILVUS_COLLECTION` | resume_agent_kb | Milvus 集合名 |
| `MILVUS_METRIC` | COSINE | Milvus 相似度度量（IP / COSINE / L2） |
| `MILVUS_TOKEN` | 空 | Milvus RBAC token（如 `user:password`） |
| `PG_HOST` / `PG_PORT` | localhost / 5432 | PostgreSQL 连接 |
| `PG_DATABASE` / `PG_USER` / `PG_PASSWORD` | resume_agent / postgres / postgres | PostgreSQL 凭据 |
| `PG_TABLE` | resume_agent_kb | pgvector 表名 |
| `PG_SCHEMA` | public | pgvector schema |
| `PG_USE_JSONB` | true | metadata 是否用 JSONB |
| `PG_HNSW` | true | 是否创建 HNSW 索引 |
| `RETRIEVER_MODE` | auto | `auto`（尝试向量，失败降级）/ `keyword` / `vector` |
| `TOP_K` | 5 | 检索召回数量 |
| `RECALL_TARGET` | 0.85 | 量化指标目标值 |

---

## 🔌 API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（含检索模式 + 向量库状态） |
| GET | `/vectors` | 向量库自省：后端 / 连接 / 集合表 / 维度 / 条目数 / 度量 |
| POST | `/analyze` | 文本分析：传 JD + 简历 |
| POST | `/upload` | 文件上传：上传 JD + 简历文件 |
| POST | `/ingest` | 重建知识库；`?force=true&reset=true` 先清空后端历史数据 |
| GET | `/metrics` | 跑评估集，返回 recall@k |

`/vectors` 返回示例：

```json
{
  "backend": "milvus",
  "available": true,
  "ready": true,
  "target": "resume_agent_kb",
  "dimension": 384,
  "count": 8,
  "metric": "COSINE",
  "error": null
}
```

---

## 🛤️ 路线图

- [x] v0.2.0 — Milvus / pgvector 深度整合
- [ ] v0.3.0 — 支持 PDF / Word 简历解析
- [ ] v0.4.0 — 多 Agent 协作（JD 解析 Agent + 简历改写 Agent）
- [ ] v1.0.0 — 生产可用

---

## 🙏 参考开源

- [datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents) — Agent 学习框架
- [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) — RAG 全链路实践

---

## 📄 License

MIT License
