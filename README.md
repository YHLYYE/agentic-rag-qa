# AgenticRAG-QA

自纠错 + 路由的 Agentic RAG 问答系统，基于 FinanceBench 财报语料，用 LangGraph 编排，带可复现的三组对照评估。

核心：**确定性检索 + 引用硬闸门，解决 AI 幻觉**（答案引用的 chunk 必须真实存在，否则打回重查）。

---

## 快速开始

### 1. 环境准备（独立 conda 环境，别污染主环境）

```bash
conda create -n raggpu python=3.11 -y
conda install -n raggpu numpy=1.26.4 scipy -y
```

**关键版本（已踩坑，务必照抄，否则版本冲突）：**

```bash
pip install "C:/path/to/torch-2.5.1+cu124-cp311-cp311-win_amd64.whl" \
            "C:/path/to/torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl"
pip install "transformers==4.47.1" "sentence-transformers==4.1.0"
pip install faiss-cpu rank-bm25 pydantic PyYAML PyMuPDF openai langgraph datasets python-dotenv pytest
```

> ⚠️ **版本红线**：transformers 必须 4.47.1（5.x 要 torch 2.6+，torch 2.5.1 会崩）；sentence-transformers 必须 4.1.0（6.x 要 transformers 5.x）。

### 2. 网络配置（国内必须）

huggingface.co 被墙，一律走国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> ⚠️ 如果开着 v2rayN 等代理软件，关掉后系统代理(127.0.0.1:7890)会死，导致连国内 DeepSeek 也失败。需同时关系统代理：Windows 设置 → 代理 → 关掉「使用代理服务器」。

### 3. 下载数据（FinanceBench 财报 PDF）

```bash
python src/ingest/download_pdfs.py data/raw
```

> 63 份年报 PDF 从公司投资者关系站 + SEC 归档下载（需能访问外网或代理）。约 21 份链接失效可忽略。

### 4. 下载模型

- **bge-m3**（embedding，~2.2GB）：首次建索引时自动下载（走 hf-mirror）。
- **bge-reranker-base**（重排，~1.1GB）：huggingface-hub 的 HEAD 请求被 hf-mirror 拒，需手动下载到 `data/models/bge-reranker-base/`（含 config.json + model.safetensors + tokenizer 文件）。

### 5. 建索引

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python -m ingest.build_financebench_index data/raw data/index
```

产出：`data/index/chunks.pkl`（切块）+ `data/index/faiss.index`（bge-m3 向量）。

### 6. 跑评估

```bash
# 检索侧（MRR / HitRate@5）
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python -m eval.run_eval

# 生成侧（RAGAS：faithfulness / context_precision / context_recall，DeepSeek 当裁判）
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python -m eval.ragas_self data/index 15

# 重排对比（无重排 vs bge-reranker）
PYTHONPATH=src python -m eval.run_rerank_compare data/index 15
```

> DeepSeek API key 放 `.env`（`DEEPSEEK_API_KEY=...`，已 gitignore）。

---

## 架构

- **离线**：PDF → 解析（PyMuPDF 文本 + find_tables 结构化表格）→ 切块 → bge-m3 向量化 → Faiss + BM25
- **在线（LangGraph）**：route → retrieve → CRAG critique → generate → verify-then-answer（有界重试环）
- **评估**：检索质量（MRR/HitRate）与生成质量（RAGAS）分开测，互不掩盖

## 目录结构

```
src/
  ingest/   下载、解析（含 find_tables 表格抽取）、切块、建索引
  rag/      embedding、dense/BM25/hybrid(RRF)、reranker、citation
  graph/    LangGraph state、5 节点、图组装
  eval/     MRR/HitRate、RAGAS（自实现拆句法）、三组对照
data/
  raw/      财报 PDF（gitignore）
  index/    chunks + faiss 索引（gitignore）
  models/   reranker 模型（gitignore）
  reports/  评估报告
```

## 评估结果摘要

| 指标 | 值 |
|------|-----|
| MRR（naive dense） | 0.577 |
| HitRate@5（naive dense） | 0.733 |
| faithfulness | ~0.7-0.83 |
| context_precision / recall | 偏低（根因见下） |

## 已知问题与下一步

**根因（已定位）**：`get_text()` 会把财报财务表格抽乱（数字连在一起），导致 LLM 识别不了答案。已改用 `find_tables()` 结构化抽表（`src/ingest/parse.py`），正在重建索引验证。

详细评估数据与根因诊断见 `data/reports/evaluation_findings.md`。
