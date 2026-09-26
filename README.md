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

非技术演示（一键启动）：**双击仓库根目录的 `启动页面.bat`**（自动 cd 到项目 + 用 raggpu 环境起服务）。

演示问题示例（界面目前加载英文索引 `data/qa/index`）：

```
Which facility was founded in Missouri, Discovery Zone or Valentino's?     → Discovery Zone
Which opera has more acts, La jolie fille de Perth or Mitridate, re di Ponto?  → 四幕 vs 三幕
Who invented the telephone?   → 应回答「查不到」（不在语料里），可演示引用硬闸门与拒答
```

> 界面目前只接了英文轨道；要演示中文轨道需改用 `data/zh/T2Ranking/` 的语料（见 `data/reports/中文检索轨道-T2Ranking.md`）。

---

## 架构

- **路由（route_node）**：LLM 意图分类 → factoid（查事实→BM25）/ comparison（对比→dense）/ multi-hop（多跳→混合）
- **检索**：dense（bge-m3）+ BM25 + hybrid（RRF 融合）
- **自纠错**：CRAG 检索评估 + verify-then-answer 引用硬闸门，有界重试环
- **评估**：路由准确率 + RAGAS（拆句法 faithfulness / precision / recall）

## 评估结果

### 主结果：中文检索（T2Ranking，真实人工相关性标注）

评测设定：**1000 条真实中文搜索查询 / 30,365 个段落 / 16,433 条 4 级人工相关性标注**。
指标为 nDCG@10 / Recall@100 / MRR@10，逐题明细在 `data/reports/runs/`（前缀 `zh_`）。

| 检索器 | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| BM25（字符 bigram 中文分词） | **0.4749** | **0.8390** | **0.7116** |
| BM25（英文分词器，作对照） | 0.0498 | 0.0941 | 0.0763 |

**实现正确性验证**：与官方 T2Ranking BM25 基线 run 做**同候选池**对照 —— 官方 0.4638 / 本项目 0.4749（差 2.4%，Top-10 重合率 0.62）。
详见 [`data/reports/中文检索轨道-T2Ranking.md`](data/reports/中文检索轨道-T2Ranking.md)。

### 对照：英文 benchmark 轮（HotpotQA / TriviaQA）

> ⚠️ 历史那组 n=300 的数字**没有原始记录**（早期只打印到终端），引用前请看 [`data/reports/评估存档说明.md`](data/reports/评估存档说明.md)。
> 有存档支撑的是 2026-09-26 实跑的 **n=30 分层抽样**：路由准确率 **0.833（25/30）**、faithfulness 0.856 / precision 0.207 / recall 0.833。

| 指标 | 值 |
|------|-----|
| 路由准确率 | 84.3%（300 题，三类分流） |
| recall（整体） | 0.780 |
| faithfulness | 0.785 |
| precision | 0.232 |

分题型：factoid recall 0.840 / comparison 0.690 / multi-hop 0.810。

**两轮语料不可混用**：中文轮用真实 qrels、英文轮用「答案字符串匹配」代理指标，且语料难度不同。

复现：
```bash
# 中文轨道（主）
python -m eval.qrels_retrieval --tokenizer zh          # 确定性、零 LLM 成本
python -m eval.qrels_dense_eval                        # dense / 混合对比（CPU 约 80 分钟）
# 英文对照轮
python -m eval.route_accuracy --per-type 10
python -m eval.run_qa_eval --per-type 10
```

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

早期用 FinanceBench（财报 PDF）时，recall 卡在 0.11——根因是财报表格的多行表头无法被 get_text/find_tables/pdfplumber 正确解析（连官方基准自己的抽取都是乱的）。换用纯文本 QA 语料（HotpotQA + TriviaQA）后，recall 提升到 **0.833**（n=30 分层抽样，逐题明细见 `data/reports/runs/`）。详见 `data/reports/evaluation_findings.md`。

> 两轮语料的定位不同：**FinanceBench 轮**（63 份真实 10-K 年报）贴近「私有知识库」场景，产出表格解析根因证据链；**HotpotQA/TriviaQA 轮**是可复现的量化评测主线。**两组数字不可混用。**
