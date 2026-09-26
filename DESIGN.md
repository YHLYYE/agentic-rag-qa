# 「自纠错 + 路由」Agentic RAG 知识库问答系统 — 设计说明

> 版本：v0.2（裁剪冗余后的最终版）
> 日期：2026-09-21
> 性质：AI Agent 应用开发作品集项目（与 MiniCode 互补）
> 定位：RAG + LangGraph 当主角，确定性可评估，公开 benchmark 语料可复现

---

## 1. 项目定位

**可量化、可复现的 Agentic RAG 问答系统**：用 LangGraph 编排「路由 → 多路检索 → 检索自纠错 → 生成 → 引用校验 → 定稿」全流程。
重点**不在模型能力，而在「评估体系 + 工程可靠性」**——确定性指标、逐题存档可复算、可控重试、引用硬闸门。

项目分两个阶段推进，各自承担一个侧面（此前文档把它们混在一起讲，导致「私有知识库」这个说法没有落脚点）：

| 阶段 | 语料 | 承担的定位 | 产出 |
|------|------|-----------|------|
| ① 真实企业文档 | **FinanceBench**（63 份 10-K 年报 PDF，24,563 chunks） | 贴近「私有知识库」场景 | 用证据链定位「财报表格多行表头无法解析」这一本质难题（连官方基准自己的抽取都是乱的） |
| ② 公开 benchmark | **HotpotQA + TriviaQA**（英文维基问答） | **可复现的量化评测** | 路由 / 检索 / 自纠错全链路指标 + 逐题存档，任何汇总数字可追溯到原始记录 |

> 口径说明：**评测用公开 benchmark，是为了可复现；「私有知识库」的场景证据来自阶段 ①**。
> 两个阶段的数字不可混用（语料难度与指标口径都不同）。

**与 MiniCode 的互补关系**：

| 维度 | MiniCode | 本项目（AgenticRAG-QA） |
|------|----------|------------------------|
| 范式 | 自研 while-true Agent Loop（无框架） | LangGraph 图编排（有框架） |
| 检索 | 自研 n-gram 伪向量（讲工程权衡） | 真向量 RAG + 混合检索 + 自纠错 |
| 领域 | Coding Agent | 知识库问答 / Agentic RAG |
| 主角 | 手搓底层 | 框架工程化 + 评估 |
| 验证 | 基准脚本（Token 压缩比）+ 单测/对抗审计 | MRR/HitRate@k + RAGAS + 引用命中率 |

**一句话**：MiniCode 证明「懂 Agent 底层原理」，本项目证明「懂 RAG 工程化与可评估」。

---

## 2. 设计原则

1. **路由 + 自纠错是控制流，RAG 是能力**：路由/重查/校验用 LangGraph 编排，检索/向量/切块用 RAG 组件，二者边界清晰。
2. **强制溯源**：所有回答必须携带可回溯的引用（chunk_id / 文档坐标），引用校验是**硬闸门**，不是软检查。
3. **检索与生成分离**：检索质量（MRR/HitRate）与生成质量（RAGAS）分别评估，互不掩盖。
4. **可复现评估**：语料用公开 benchmark（含 ground truth），三组对照量化「路由 / 自纠错」各自贡献。
5. **单轨够用**：一套检索栈（bge-m3 + Faiss + BM25 + bge-reranker），不背双轨/容器等重基础设施。

---

## 3. 架构总览

```
┌─ 数据接入 & 索引层（离线，RAG 组件）───────────────────┐
│ benchmark 文档 → 解析 → 切块 → embedding → 向量库+倒排 │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌─ LangGraph 编排层（在线，主角，5 节点）────────────────┐
│ route_node（意图路由）                                 │
│   ├─ 语义类  → dense 向量检索                          │
│   ├─ 关键词/数值类 → BM25 检索                         │
│   └─ 混合类  → 多路并行（Send API 扇出）               │
│ retrieve_node（多路检索 + 合并去重 + rerank）           │
│ critique_node（CRAG 检索自纠错：correct/ambiguous/incorrect）│
│ generate_node（LLM 生成候选答案 + 引用）               │
│ verify_node（引用是否支撑结论 → 定稿 or 重查/降级）     │
└──────────────────────┬───────────────────────────────┘
                       ▼
         带引用回答 + 评估指标（MRR / RAGAS / 引用命中率）
```

**数据流方向**：用户问题 → 路由 → 检索 → 自纠错（可能回退重查）→ 生成 → 引用校验（可能重查/降级）→ 定稿回答。

---

## 4. LangGraph 编排层（核心）

### 4.1 State Schema

```python
class AgenticRAGState(TypedDict):
    question: str               # 用户问题
    intent: str                 # 路由意图：semantic / keyword / hybrid
    route_decision: dict        # 路由明细（原因 + 置信度）
    retrieved_chunks: list      # 检索结果（chunk_id / 文本 / score / 坐标）
    retrieval_verdict: str      # CRAG 评估：correct / ambiguous / incorrect
    candidate_answer: str       # 候选答案
    citations: list             # 引用列表（chunk_id）
    grounding_verdict: str      # verify-then-answer：supported / unsupported
    final_answer: str           # 定稿回答
    retry_count: int            # 有界重试计数（两条回退环共用，默认上限 2）
```

### 4.2 节点与边（5 节点）

| 节点 | 职责 | 类型 |
|------|------|------|
| `route_node` | 意图分类 + 路由决策（LLM 分类 + 关键词规则兜底） | LLM |
| `retrieve_node` | 执行多路检索，合并去重 + rerank | 普通函数（非 LLM） |
| `critique_node` | CRAG：给检索结果打相关性分 | LLM |
| `generate_node` | 生成候选答案 + 引用 | LLM |
| `verify_node` | 逐条校验引用是否支撑结论；unsupported 时重查或降级「请查阅原文」 | LLM |

**条件边（控制流）**：

```
route → retrieve                    （按意图选单路 or 扇出多路）
retrieve → critique                 （检索评估）
critique → correct    → generate    （检索过关，进生成）
critique → incorrect  → retrieve    （重查，retry_count 有界）
critique → ambiguous  → retrieve    （合并重查，retry_count 有界）
generate → verify                   （引用校验）
verify → supported    → END         （定稿）
verify → unsupported  → retrieve    （重查，有界）或 降级「请查阅原文」
```

> 降级「请查阅原文」是 `verify_node` 的 `unsupported` 分支里的一个返回值，不单设节点。

**LangGraph 能力覆盖**：

| LangGraph 能力 | 落点 |
|----------------|------|
| StateGraph + TypedDict | `AgenticRAGState` |
| 条件边（路由） | `route_node` 多分支 |
| 循环（有界重试） | `critique/verify → retrieve` 回退环 |
| Send API 并行扇出 | 混合类问题多路检索并行 |
| interrupt() 人机协作（stretch） | 歧义问题多轮澄清 + checkpoint 恢复 |

### 4.3 路由（route_node）

| 意图 | 判定 | 检索策略 |
|------|------|---------|
| 语义类（「X 和 Y 有什么关系」） | LLM 分类 | dense 向量 |
| 关键词/数值类（「某术语定义」「某公司某季度营收」） | 关键词规则兜底 | BM25（实体+字段做关键词匹配，不单建结构化查询引擎） |
| 混合类（「比较两家公司的 X」） | 多实体 | 多路并行 + rerank 合并 |

### 4.4 自纠错（双层）

**第一层：CRAG 检索自纠错（检索后、生成前）**
- 评估器对每篇检索结果打分：`correct`（直接用）/ `ambiguous`（合并补充检索）/ `incorrect`（丢弃，触发关键词回退重查）。
- 作用：纠正「检索没找对」，避免无关 chunk 污染生成。

**第二层：verify-then-answer 引用硬闸门（生成后、定稿前）**
- 生成器先产出候选答案 + 引用，校验器逐条判断「该引用是否真的支撑结论」。
- `supported` → 定稿；`unsupported` → 重查（有界）或降级「请查阅原文」。
- 作用：纠正「答案没据」，压制 LLM 幻觉。

**重试有界**：两条回退环共用 `retry_count`（默认上限 2），超限直接降级，杜绝死循环。

---

## 5. RAG 检索层

### 5.1 语料（两阶段，均已落地）

| 阶段 | 语料 | 用途与现状 |
|------|------|-----------|
| ① | **FinanceBench**（63 份 10-K 年报 PDF，专家标注问答，文档级 ground truth） | 已完成。用于验证真实企业文档场景，产出 `data/reports/evaluation_findings.md`（含表格解析根因证据链） |
| ② | **HotpotQA + TriviaQA**（英文维基段落问答，3 类题各 100） | 当前主线。用于可复现的量化评测，产出 `data/reports/runs/` 逐题存档与 `优化进展-检索准确率.md` |

### 5.1.1 中文检索轨道（**当前主评测**：T2Ranking）

选择理由（实测）：`THUIR/T2Ranking` 是唯一同时具备**原生中文真实搜索查询**、**4 级人工相关性标注**、
**官方 BM25/DPR 基线 run** 的可获取基准（DuReader-retrieval / Multi-CPR / T2Ranking(mteb) 均实测 401 不可得）。

评测设定与结果见 [`data/reports/中文检索轨道-T2Ranking.md`](data/reports/中文检索轨道-T2Ranking.md)：

| 项 | 值 |
|---|---|
| 评测集 | 1000 条真实中文查询 / 30,365 段落（16,365 判定 + 14,000 干扰）/ 16,433 条 qrels |
| BM25（字符 bigram） | nDCG@10 **0.4749** / Recall@100 **0.8390** / MRR@10 **0.7116** |
| 对照（英文分词器） | nDCG@10 0.0498（差 9.5 倍，再次证明 BM25 的瓶颈是分词） |
| 官方基线对齐 | 同候选池下官方 0.4638 / 本项目 0.4749（差 2.4%）→ 实现正确 |

> ⚠️ 该评测在 **pooled 判定池**上进行（标准 IR 做法），比在 300 万段落全量里检索**容易**；
> 因此本项目数字**不可与官方全量基线直接比较**（这与「下采样使任务变简单」是同一件事）。

已排除的备选：`hfl/cmrc2018`（单段落抽取，检索难度过低，仅作跨语料验证）、`mMARCO-zh`（MS MARCO 翻译版，有翻译腔，非原生中文）。

### 5.2 切块

- 财报/长文档：结构化切块（按章节/表格边界），块大小 512~1024 token，重叠 10%。
- 保留元数据：`chunk_id`、`source_doc`、`section`、`page/坐标`，供溯源。

### 5.3 检索栈（单轨）

| 组件 | 选型 |
|------|------|
| embedding | bge-m3（多语言，一个模型够用） |
| 向量库 | Faiss（本地，演示够用；要生产再换 Qdrant） |
| 关键词 | BM25（rank_bm25） |
| 重排 | bge-reranker |
| LLM | DeepSeek（兼容 OpenAI 协议） |

> 语料小到「全文塞 prompt 都能答」时，检索价值在「结构化切块 + 自纠错」，不在 embedding 尺寸——README 中坦诚说明这一判断。

### 5.4 溯源

回答强制携带 `[source_doc §section]` 或 `[chunk_id]`；引用校验（verify_node）只接受「引用 ∈ 本次检索结果集合」。

---

## 6. 端到端数据流（示例）

```
问：「A 公司 2024 年营收增速和 B 公司相比如何？」
   │
route_node ── 混合类（多实体）──→ Send API 扇出
   │                              ├─ dense 检索「营收增速」
   │                              └─ BM25 检索「A 公司」「B 公司」
   ▼
retrieve_node 合并去重 + rerank → Top-K chunks
   ▼
critique_node（CRAG）→ 1 条 incorrect 被丢弃 → 回退重查（retry_count=1）
   ▼
generate_node → 候选答案 + [A年报§营收] [B年报§营收]
   ▼
verify_node → 引用支撑结论 → supported
   ▼
定稿回答 + 引用
```

---

## 7. 评估层

### 7.1 指标

| 指标 | 测什么 | 计算 |
|------|--------|------|
| MRR / HitRate@k | 检索质量 | 检索结果 vs benchmark ground truth |
| RAGAS faithfulness | 答案是否忠于检索到的证据 | LLM-as-judge |
| RAGAS context precision/recall | 上下文是否精准/完整 | LLM-as-judge |
| 引用命中率 | 引用是否命中真实 chunk | 硬闸门统计 |

### 7.2 三组对照（核心卖点）

| 组 | 配置 | 目的 |
|----|------|------|
| ① naive | 单路 dense，无路由无自纠错 | 基线 |
| ② + 路由 | 多路路由 + 混合检索 | 量化「路由」贡献 |
| ③ + 路由 + 自纠错 | 完整系统 | 量化「自纠错」贡献 |

**目标产出**：「路由使 MRR 从 X 提到 Y，自纠错使 faithfulness 从 Y 提到 Z、低置信查询回收了 W%」——三组数字写进简历。

---

## 8. 技术栈（单轨）

| 层 | 选型 |
|----|------|
| 语言 | Python 3.11+ |
| 编排 | LangGraph（StateGraph + 条件边 + 循环 + Send API） |
| RAG | bge-m3、Faiss、BM25、bge-reranker |
| LLM | DeepSeek（兼容 OpenAI 协议） |
| 评估 | RAGAS、MRR/HitRate@k（自实现）、引用命中率 |
| 界面 | Streamlit（可选，Phase 5 的 stretch goal） |

> 不引入 Docker / Qdrant / 双轨——演示不背重基础设施，README 说明「生产化要换什么」。

---

## 9. 目录结构

```
agentic-rag-qa/
├── DESIGN.md                  # 本文档
├── README.md
├── src/
│   ├── ingest/                # benchmark 文档下载、解析、切块、入库
│   ├── rag/                   # 检索器（dense/BM25）、rerank、溯源
│   ├── graph/                 # LangGraph 节点、状态、条件边
│   └── eval/                  # MRR / RAGAS / 引用命中率 三组对照
├── tests/
│   ├── unit/                  # 各节点单测
│   └── e2e/                   # 端到端问答链路
├── configs/                   # config.yaml（语料路径、模型、检索参数）
└── data/
    ├── raw/                   # benchmark 原始文件（gitignore）
    ├── index/                 # 切块 + 向量索引产物
    └── eval_set/              # 三组对照评估配置
```

---

## 10. 实施路线

| 阶段 | 内容 | 产出 |
|------|------|------|
| 1 | 语料落地：下载 FinanceBench，解析切块入库 | 可检索索引 + 切块统计 |
| 2 | RAG 检索层：dense/BM25 两路 + 混合 + rerank | MRR/HitRate 基线（组①） |
| 3 | LangGraph 编排：路由 + 生成 + 引用硬闸门 | 端到端问答 + 组② |
| 4 | 自纠错：CRAG + verify-then-answer 双层回退环 | 组③ + 完整三组对照 |
| 5 | （可选）Streamlit 界面 + README + 评估报告 | 可演示 + 可发布 |
| 6 | （stretch）interrupt() 多轮澄清 + checkpoint 恢复 | LangGraph 人机协作能力补齐 |

---

## 11. 风险与应对

| 风险 | 应对 |
|------|------|
| benchmark 语料 license 限制再分发 | 文档原文只做本机检索，仓库只存索引/统计/评估结果，不重发布语料 |
| 三组对照指标波动 | 固定随机 seed、固定 LLM 温度、多次采样取均值 |
| 自纠错回退环死循环 | retry_count 硬上限 + 降级兜底 |
| 「又是 RAG demo」的印象 | 靠三组对照硬数字 + 领域垂直语料 + 完整溯源打破 |
| bge-m3 模型较大/下载慢 | CPU 版够用；或换 OpenAI embedding 兜底 |

---

## 12. 验收标准

- ✅ 端到端链路跑通：问 → 路由 → 检索 → 自纠错 → 引用校验 → 带引用回答
- ✅ 三组对照报告产出（MRR / HitRate@k / RAGAS / 引用命中率）
- ✅ 自纠错层有可量化收益（faithfulness 或低置信回收率提升）
- ✅ 引用硬闸门生效：无引用或引用不命中时正确降级
- ✅ pytest 全绿；README 完整；demo 只用公开 benchmark 数据
- ✅ （stretch）interrupt() 多轮澄清可演示：歧义问题能暂停追问并带上下文恢复

---

## 13. 待确认决策

1. 项目正式命名（暂定 AgenticRAG-QA）。
2. LLM 接入方式（DeepSeek API key 或本地模型）。

---

*本文档为设计说明（v0.2 裁剪版），不含实现代码；确认 §13 后进入阶段 1 实施。*
