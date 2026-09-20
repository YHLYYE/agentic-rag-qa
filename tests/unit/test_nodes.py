from graph.nodes import route_node, verify_node


def test_route_node_semantic():
    state = {"question": "How does revenue relate to profit?"}
    out = route_node(state)
    assert out["intent"] == "semantic"


def test_route_node_keyword_rule_fallback():
    state = {"question": "What is the definition of gross margin?"}
    out = route_node(state)
    assert out["intent"] == "keyword"


def test_verify_node_supported():
    state = {
        "candidate_answer": "revenue grew {{abc123}}",
        "citations": ["abc123"],
        "retrieved_chunks": [{"chunk_id": "abc123"}],
    }
    out = verify_node(state)
    assert out["grounding_verdict"] == "supported"


def test_verify_node_unsupported():
    state = {
        "candidate_answer": "revenue grew {{deadbeef}}",
        "citations": ["deadbeef"],
        "retrieved_chunks": [{"chunk_id": "abc123"}],
    }
    out = verify_node(state)
    assert out["grounding_verdict"] == "unsupported"
