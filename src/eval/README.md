# src/eval 脚本索引：每个脚本对应哪个已归档的结论

> 这个目录看起来"脚本很多、互相重叠"，但每个脚本其实对应一个**已经写进报告的具体结论**。
> 这份索引的作用：面试官问"这些都要吗"时，你能一条条对上「脚本 → 结论 → 存档」。
> 2026-09-26 清理：删掉了 3 个无产出的死脚本（见文末）。

## 一、脚本 → 归档结论对照

| 脚本 | 干什么 | 对应哪个已归档的结论 |
|---|---|---|
| `route_accuracy.py` | LLM 意图路由 vs 真实标签，含混淆矩阵 | `data/reports/runs/*_route_accuracy.*`（n=30 实测 0.833）；README 的 84.3%（n=300，**无原始记录**） |
| `run_qa_eval.py` | QA 语料 RAGAS 三指标，分题型 | `data/reports/runs/*_qa_ragas.*`（n=30：0.856 / 0.207 / 0.833） |
| `ragas_self.py` | 拆句法指标实现（被 `run_qa_eval` 复用） | `evaluation_findings.md` §4「可靠的最终指标（拆句法）」 |
| `run_topk_sweep_fast.py` | top-K 扫描（单次粗判的简版指标） | `evaluation_findings.md` §2.4 top-K 表（top-5 最优） |
| `run_topk_sweep.py` | top-K 扫描（拆句法版） | 与简版对照，佐证「简版指标噪声大」 |
| `run_eval.py` | FinanceBench 三组检索对照（MRR / HitRate@5） | `evaluation_findings.md` §2.1（naive dense / hybrid / routed 三行） |
| `run_rerank_compare.py` | 无重排 vs bge-reranker（FinanceBench） | `evaluation_findings.md` §2.2 的两个重排列 |
| `run_qa_rerank.py` | 无重排 vs bge-reranker（QA 语料） | 简历「reranker 提升 precision」（**原始记录缺失**，见 评估存档说明） |
| `ragas_patch.py` | 官方 ragas 0.4.3 的 vertexai import monkey patch | `evaluation_findings.md` §5 环境记录（解释为什么改用自实现） |
| `artifacts.py` | 逐题明细 + 汇总落盘（自校验、带 git 版本） | `data/reports/runs/` 全部存档 |
| `sampling.py` | 分层抽样，避免「取前缀 = 单题型」假证据 | 见 `data/reports/评估存档说明.md` 第五节 |
| `metrics.py` | MRR / HitRate@k（被 `run_eval.py` 使用） | `evaluation_findings.md` §2.1 |

## 二、为什么 FinanceBench 那批脚本要留着

`run_eval.py`、`run_rerank_compare.py`、`ragas_self.py` 的 FinanceBench 路径、以及 `src/ingest/` 里的 PDF 解析脚本，都对应 `evaluation_findings.md` 里**带完整证据链的结论**（表格解析是本质难题）。

它们是**历史轮次的实现**，不是冗余：删掉之后就查不到"那个结论是怎么算出来的"了。
代价是它们和当前 QA 语料的代码混在一起、看起来杂乱——这份索引就是用来消除这个困惑的。

对应关系：

- FinanceBench 轮次（历史）：`ingest/parse.py`、`ingest/download_pdfs.py`、`ingest/build_financebench_index.py`、`eval/run_eval.py`、`eval/run_rerank_compare.py`、`eval/ragas_self.py` 的 `main()`
- QA 轮次（当前主线）：`ingest/download_qa.py`、`ingest/build_qa_index.py`、`eval/run_qa_eval.py`、`eval/route_accuracy.py`、`run_graph.py`、`ui/app.py`

## 三、入口在哪（哪些是"能跑的"，哪些是被 import 的）

- **CLI 入口**（`python -m eval.xxx`）：`route_accuracy`、`run_qa_eval`、`run_topk_sweep`、`run_topk_sweep_fast`、`run_qa_rerank`、`run_rerank_compare`、`run_eval`
- **被 import 的库**：`ragas_self`（三个指标函数）、`artifacts`、`sampling`、`metrics`、`ragas_patch`

所以 `rg "import"` 找不到 `run_qa_eval` 的导入方是正常的——它是入口，不是库。

## 四、已知的重复（待清理，未做）

以下重复是真实存在的，属于"可以合并但还没合并"：

1. `LLM` 包装类在 `run_qa_eval.py` / `run_topk_sweep.py` / `run_topk_sweep_fast.py` / `ui/app.py` 各写了一遍（`run_graph.py` 里是 `DeepSeekLLM`），可抽成 `eval/llm.py`。
2. `_retrieve` 在 `run_topk_sweep*.py` 各一份；`_ask` 在 `ragas_self.py` 与其他脚本各一份。

3. `configs/config.yaml` + `src/config.py` **只被 `tests/unit/test_config.py` 使用**，运行时全是硬编码——要么接进 `run_graph`，要么删掉。

4. **CLI 参数解析不统一**：`route_accuracy.py` / `run_qa_eval.py` / `run_graph.py` 用 argparse（有 `--help`），而 `run_topk_sweep.py` / `run_topk_sweep_fast.py` / `run_qa_rerank.py` / `run_rerank_compare.py` / `run_eval.py` 直接用 `sys.argv`，所以 `--help` 会抛 `ValueError`。
   不影响正常调用（如 `python -m eval.run_topk_sweep 60`），属于待统一项。

## 五、2026-09-26 删除的脚本

| 文件 | 原因 |
|---|---|
| `run_comparison.py` | 库函数形式、无 CLI、**无任何调用方**，是设计稿里的三组对照驱动器，已被 `run_eval.py` + 落盘存档取代 |
| `ragas_eval.py` | 官方 ragas 的薄封装，其 docstring 自己写着「not yet installed」，已被 `ragas_self.py` 取代 |
| `run_ragas.py` | 官方 RAGAS 路径（要先 `ragas_patch()`），因版本冲突废弃；相关经历保存在 `ragas_patch.py` + `evaluation_findings.md` §5 |
| `metrics.citation_hit_rate` | 唯一调用方是上面的 `run_comparison.py`；引用是否命中现在由 `verify_node` 在请求链路里强制，不再作为聚合指标 |
