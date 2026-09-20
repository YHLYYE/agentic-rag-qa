from pathlib import Path
import yaml
from pydantic import BaseModel


class CorpusConfig(BaseModel):
    name: str
    raw_dir: str
    index_dir: str
    eval_set_dir: str


class EmbeddingConfig(BaseModel):
    model: str
    dim: int


class RetrievalConfig(BaseModel):
    top_k: int
    rerank_top_k: int


class LLMConfig(BaseModel):
    base_url: str
    model: str


class GraphConfig(BaseModel):
    max_retry: int


class Config(BaseModel):
    corpus: CorpusConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    graph: GraphConfig


def load_config(path: str) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(**data)
