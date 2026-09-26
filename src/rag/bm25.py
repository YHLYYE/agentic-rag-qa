import re

from rank_bm25 import BM25Okapi

from models import Chunk, RetrievedChunk

# 常用英文停用词（不引第三方依赖）
_STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been being it its "
    "this that these those with by from as which who whom whose what when "
    "where why how do does did s".split()
)


def simple_tokenize(text: str | None) -> list[str]:
    """小写 + 只保留字母数字。

    修复的问题：原实现是 `text.split()`，**大小写敏感且不剥离标点**，
    导致 query 的 "Who"/"telephone?" 与语料里的 "who"/"telephone" 是两个
    不同 token —— 实测交集只剩 {'invented', 'the'}，最有信息量的词全废。
    """
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def tokenize_no_stopwords(text: str | None) -> list[str]:
    """在 simple_tokenize 基础上再去掉常见停用词（实测优于不去：BM25 +1.3pt、hybrid +0.7pt）。

    边界情况：整句都是停用词时（如 "who is it?"）不能返回空列表——
    空 token 会让 BM25 打分退化成全 0、排序变成随机。此时回退到不去停用词的结果。
    """
    tokens = [t for t in simple_tokenize(text) if t not in _STOPWORDS]
    return tokens or simple_tokenize(text)


def zh_tokenize(text: str | None) -> list[str]:
    """中文分词：字符二元组（bigram）。

    为什么必须单独做：`simple_tokenize` 用的是 `[a-z0-9]+`，中文基本全被丢掉
    （实测「蜂巢取快递验证码摁错怎么办」只切出 1 个 token），中文 BM25 命中率只有 0.08。

    为什么用 bigram 而不是 jieba：无新依赖，且是 Lucene CJKAnalyzer 的经典做法；
    实测在 T2Ranking 上把 BM25 从 0.08 拉到 0.9967（见 data/reports/中文语料可行性实验.md）。
    """
    chars = "".join(ch for ch in str(text or "") if ch.isalnum())
    if len(chars) < 2:
        return [chars] if chars else []
    return [chars[i:i + 2] for i in range(len(chars) - 1)]


class BM25Retriever:
    """默认使用 tokenize_no_stopwords；语料与 query 必须走同一个 tokenizer，
    否则索引和查询的 token 空间对不上（这是分词类 bug 最常见的来源）。"""

    def __init__(self, chunks: list[Chunk], tokenize=tokenize_no_stopwords):
        self.chunks = chunks
        self.tokenize = tokenize
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(self.tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [RetrievedChunk(self.chunks[i], float(scores[i])) for i in ranked]
