from config import load_config, Config


def test_load_config_defaults():
    cfg = load_config("configs/config.yaml")
    assert cfg.embedding.model == "BAAI/bge-m3"
    assert cfg.retrieval.top_k == 8
    assert cfg.graph.max_retry == 2


def test_config_is_pydantic_model():
    cfg = load_config("configs/config.yaml")
    assert isinstance(cfg, Config)
