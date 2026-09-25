"""BM25 分词：修复「大小写敏感 + 标点不剥离」导致关键词白白失配。"""
from models import Chunk
from rag.bm25 import BM25Retriever, simple_tokenize, tokenize_no_stopwords


def test_lowercases_and_strips_punctuation():
    assert simple_tokenize("Who invented the telephone?") == \
        ["who", "invented", "the", "telephone"]


def test_keeps_digits():
    assert simple_tokenize("In 1898, 3.5% growth") == \
        ["in", "1898", "3", "5", "growth"]


def test_empty_and_none_are_safe():
    assert simple_tokenize("") == []
    assert simple_tokenize(None) == []


def test_query_and_corpus_tokens_actually_overlap():
    q = set(simple_tokenize("Who invented the telephone?"))
    c = set(simple_tokenize("who invented the telephone"))
    assert q == c      # 修复前交集只有 {invented, the}


def test_no_stopwords_variant():
    assert tokenize_no_stopwords("the quick brown fox") == ["quick", "brown", "fox"]


def test_retriever_matches_despite_case_and_punctuation():
    """这是回归测试：老实现下 'WHO'/'TELEPHONE?' 匹配不上任何东西。"""
    chunks = [Chunk("a", "The Telephone was invented by Alexander Graham Bell", "d", "s", 1),
              Chunk("b", "methane reacts with oxygen to give carbon dioxide", "d", "s", 2)]
    out = BM25Retriever(chunks).retrieve("WHO invented the TELEPHONE?", top_k=1)
    assert out[0].chunk.chunk_id == "a"
