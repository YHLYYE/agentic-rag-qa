"""BM25 分词：修复「大小写敏感 + 标点不剥离」导致关键词白白失配。"""
from models import Chunk
from rag.bm25 import (BM25Retriever, simple_tokenize, tokenize_no_stopwords,
                      zh_tokenize)


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


def test_stopword_only_query_does_not_become_empty():
    """全是停用词时不能让 token 列表变空，否则 BM25 打分退化成全 0、排序随机。"""
    assert tokenize_no_stopwords("Who is it?") == ["who", "is", "it"]


def test_default_tokenizer_is_the_no_stopwords_variant():
    chunks = [Chunk("a", "text", "d", "s", 1)]
    assert BM25Retriever(chunks).tokenize is tokenize_no_stopwords


def test_retriever_matches_despite_case_and_punctuation():
    """这是回归测试：老实现下 'WHO'/'TELEPHONE?' 匹配不上任何东西。"""
    chunks = [Chunk("a", "The Telephone was invented by Alexander Graham Bell", "d", "s", 1),
              Chunk("b", "methane reacts with oxygen to give carbon dioxide", "d", "s", 2)]
    out = BM25Retriever(chunks).retrieve("WHO invented the TELEPHONE?", top_k=1)
    assert out[0].chunk.chunk_id == "a"


# --- 中文分词：simple_tokenize 会把中文全丢掉（只剩数字），必须用 bigram ---

def test_zh_tokenize_makes_character_bigrams():
    assert zh_tokenize("蜂巢取快递") == ["蜂巢", "巢取", "取快", "快递"]


def test_zh_tokenize_handles_short_and_empty_input():
    assert zh_tokenize("") == []
    assert zh_tokenize(None) == []
    assert zh_tokenize("好") == ["好"]
    assert zh_tokenize("3") == ["3"]


def test_zh_tokenize_query_and_corpus_overlap():
    q = set(zh_tokenize("蜂巢取快递验证码摁错怎么办"))
    c = set(zh_tokenize("蜂巢快递柜取件时验证码错误如何处理"))
    assert q & c          # 必须有真实重叠，否则 BM25 无从匹配


def test_zh_bm25_actually_retrieves_chinese():
    chunks = [Chunk("a", "蜂巢快递柜取件验证码错误可以直接联系客服重置", "d", "s", 1),
              Chunk("b", "产后恢复期腹直肌分离会导致肚子仍然隆起", "d", "s", 2)]
    out = BM25Retriever(chunks, tokenize=zh_tokenize).retrieve("蜂巢取快递验证码摁错怎么办", top_k=1)
    assert out[0].chunk.chunk_id == "a"


def test_simple_tokenize_would_fail_on_the_same_chinese_query():
    """留证据：默认英文分词器在中文上几乎无 token，这就是必须单独做中文分词的原因。"""
    tokens = simple_tokenize("蜂巢取快递验证码摁错怎么办")
    assert len(tokens) <= 1
