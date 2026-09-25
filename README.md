# AgenticRAG-QA

自纠错 + 路由的 Agentic RAG 问答系统，用 LangGraph 编排，带可复现的评估。

**核心**：LLM 意图路由（factoid/comparison/multi-hop 三分流）+ 引用硬闸门（答案引用的 chunk 必须真实存在，否则打回重查），解决 AI 幻觉。

---

## 快速开始

### 1. 环境准备（独立 conda 环境）

```bash
conda create -n raggpu python=3.11 -y
conda install -n raggpu numpy=1.26.4 scipy -y
pip install "torch==2.5.1+cu124" --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==4.47.1" "sentence-transformers==4.1.0"
pip install faiss-cpu rank-bm25 pydantic PyYAML PyMuPDF openai langgraph datasets python-dotenv pytest streamlit pdfplumber
```

> ⚠️ **版本红线**：transformers 必须 4.47.1（5.x 要 torch 2.6+）；sentence-transformers 必须 4.1.0（6.x 要 transformers 5.x）。

### 2. 网络（国内）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

> 关掉 v2rayN 等代理后，系统代理(127.0.0.1:7890)会死，导致连国内 DeepSeek 也失败。需同时关系统代理。

### 3. 下载语料（HotpotQA + TriviaQA）

```bash
PYTHONPATH=src python -m ingest.download_qa
```

> 用 httpx 直接下 parquet（绕开 huggingface_hub 的 HEAD 请求问题）。数据集 URL 要带 `/datasets/` 前缀。

### 4. 建索引

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python -m ingest.build_qa_index
```

产出：`data/qa/index/`（chunks + faiss 向量）+ `data/qa/eval_set.pkl`（300 题评估集）。

### 5. 跑评估

```bash
# 路由准确率
PYTHONPATH=src python -m eval.route_accuracy

# RAGAS（faithfulness / precision / recall，拆句法）
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src python -m eval.run_qa_eval 300
```

### 6. 跑 Streamlit 演示

```bash
streamlit run src/ui/app.py
```

---

## 架构

- **路由（route_node）**：LLM 意图分类 → factoid（查事实→BM25）/ comparison（对比→dense）/ multi-hop（多跳→混合）
- **检索**：dense（bge-m3）+ BM25 + hybrid（RRF 融合）
- **自纠错**：CRAG 检索评估 + verify-then-answer 引用硬闸门，有界重试环
- **评估**：路由准确率 + RAGAS（拆句法 faithfulness / precision / recall）

## 评估结果

> ⚠️ **下方这组 n=300 的数字目前没有原始记录**（早期跑完只打印到终端，未落盘），引用前请看 [`data/reports/评估存档说明.md`](data/reports/评估存档说明.md)。
> 有存档支撑的是 2026-09-26 实跑的 **n=30 分层抽样**：路由准确率 **0.833（25/30）**、faithfulness 0.856 / precision 0.207 / recall 0.833，逐题明细在 `data/reports/runs/`。

| 指标 | 值 |
|------|-----|
| 路由准确率 | 84.3%（300 题，三类分流） |
| recall（整体） | 0.780 |
| faithfulness | 0.785 |
| precision | 0.232 |

分题型：factoid recall 0.840 / comparison 0.690 / multi-hop 0.810。

复现：`python -m eval.route_accuracy --per-type 10`、`python -m eval.run_qa_eval --per-type 10`（每次都落盘到 `data/reports/runs/`）。

## 目录结构

```
src/
  ingest/   下载、解析、切块、建索引（download_qa / build_qa_index）
  rag/      embedding、dense/BM25/hybrid(RRF)、reranker、citation
  graph/    LangGraph state、节点（route/retrieve/critique/generate/verify）
  eval/     路由准确率、RAGAS（拆句法）、三组对照
  ui/       Streamlit 演示
data/
  raw_qa/   下载的 parquet（gitignore）
  qa/       索引 + 评估集（gitignore）
  reports/  评估报告
```

## 背景

早期用 FinanceBench（财报 PDF）时，recall 卡在 0.11——根因是财报表格的多行表头无法被 get_text/find_tables/pdfplumber 正确解析（连官方基准自己的抽取都是乱的）。换用纯文本 QA 语料（HotpotQA + TriviaQA）后，recall 提升到 0.78。详见 `data/reports/evaluation_findings.md`。
