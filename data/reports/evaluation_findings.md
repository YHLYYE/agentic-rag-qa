# AgenticRAG-QA 评估发现与修复方案存档

> 日期：2026-09-22
> 状态：② 三组对照评估完成，根因已定位，find_tables 修复待做

---

## 1. 项目现状

AgenticRAG-QA：自纠错 + 路由的 Agentic RAG 问答系统，基于 FinanceBench 财报语料。

**已完成：**
- ✅ 项目骨架 + LangGraph 编排（route → retrieve → CRAG critique → generate → verify-then-answer）
- ✅ FinanceBench 语料：63 份 PDF，256 词切块 24563 chunks，bge-m3 向量化 + faiss/BM25 索引
- ✅ 检索侧评估（MRR / HitRate@5）
- ✅ 生成侧评估（RAGAS 三指标，自实现 + DeepSeek 当裁判）
- ✅ 三项优化对照（切块 / top_k / 重排）

---

## 2. 评估结果

### 2.1 检索侧（doc 级，确定性指标）

| 组 | MRR | HitRate@5 |
|----|-----|-----------|
| naive dense | 0.577 | 0.733 |
| hybrid（RRF） | 0.512 | 0.713 |
| routed | 0.533 | 0.673 |

### 2.2 生成侧（chunk 级，自实现 RAGAS，n=15）

| 指标 | 512词 | 256词 | 无重排 | bge-reranker |
|------|-------|-------|--------|--------------|
| faithfulness | 0.827 | 0.760 | 0.667 | 0.733 |
| context_precision | 0.093 | 0.113 | 0.093 | 0.120 |
| context_recall | 0.133 | 0.113 | 0.100 | 0.040 |

### 2.3 三项优化对照结论

| 优化 | 结果 |
|------|------|
| 切块 512→256 词 | 改善不明显 |
| top_k 5→20 | 反而更差 |
| bge-reranker 重排 | faithfulness/precision 微升，recall 降 |

**三个维度都不是关键杠杆。**

---

## 3. 根因定位（核心结论）

**证据链（逐条已验证）：**

1. 答案数字「1577」（资本支出）**没有丢** —— 在 `get_text()` 抽取的第 39/46/49/60 页（现金流量表）。
2. **检索没有问题** —— 查「capital expenditure」，含 1577 的 chunk 排**第 1 名**。
3. 但数字是**乱表格形态** —— `...157713731420...` = 三年数据(1577/1373/1420)连在一起，无分隔符。
4. 所以 LLM 无法从乱文本里识别「资本支出 = $1577」→ context_recall 指标才 ~0.1。

**根因：`get_text()` 把财务表格抽乱了（单元格内容无分隔拼接），导致 LLM 识别不了答案。**

**推翻之前的两个错误判断：**
- ❌「数字抽丢了」→ 数字在，只是表格结构被打乱
- ❌「检索是瓶颈」→ 检索排第 1 名，没问题

**附加教训：** 自实现的简版 RAGAS 指标（每指标一次 LLM 粗判 0~1）噪声大，曾误导优化方向。要更可靠需用 RAGAS 官方的「拆句逐句判断」法。

---

## 4. 修复方案（待做）

**用 `find_tables()` 替代 `get_text()` 结构化抽表格。**

步骤：
1. 重写 `src/ingest/parse.py`：`extract_pages` → `extract_tables`（`page.find_tables()` 抽取行列单元格，保留「字段名 + 值」结构）
2. 重新抽取 63 份 PDF → 切块 → bge-m3 建索引（约 30-50 分钟）
3. 重新跑 RAGAS 评估，对比修复前后 context_precision/recall

---

## 5. 环境记录（避免重复踩坑）

- **独立环境**：`raggpu`（conda env，`D:\anaconda3\envs\raggpu`），不要污染 Anaconda 主环境
- **关键版本**：
  - torch 2.5.1+cu124（本地 wheel，CUDA 可用）
  - transformers 4.47.1（不能升 5.x，5.x 要 torch 2.6+）
  - sentence-transformers 4.1.0（不能升 6.x，6.x 要 transformers 5.x）
  - langgraph 1.2.11、langchain-core 1.6.3
  - ragas 0.4.3（有 vertexai import bug，需 `eval/ragas_patch.py` 打补丁；最终评估改用自实现）
- **模型缓存**：
  - bge-m3（embedding）已缓存，走 `HF_ENDPOINT=https://hf-mirror.com`
  - bge-reranker-base 手动下载到 `data/models/bge-reranker-base/`（huggingface-hub 的 HEAD 请求被 hf-mirror 拒，改用 httpx GET 手动下）
- **网络**：huggingface.co 被墙，一律走 `hf-mirror.com`；v2rayN 关掉后系统代理(7890)会死，需关系统代理(ProxyEnable=0)否则连国内 DeepSeek 也会卡
- **运行环境变量**：所有 Python 脚本加 `HF_ENDPOINT=https://hf-mirror.com`（或 `HF_HUB_OFFLINE=1`），否则 SentenceTransformer 加载会去连 huggingface.co 卡住

---

## 6. 教训（执行纪律）

1. **数据先行**：写解析器前先 dump 真实数据看格式（section 检测、get_text 都栽在这个上）
2. **环境隔离**：独立 venv/conda env，别往共享主环境塞依赖
3. **评估用正式指标**：简版指标要明说噪声，别拿噪声指标指导优化方向
4. **根因验证**：断言前先验证（「表格解析是瓶颈」这个断言，靠打印实际文本才证实）
