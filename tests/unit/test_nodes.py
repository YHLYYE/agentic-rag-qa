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


from models import Chunk, RetrievedChunk
from graph.nodes import retrieve_node, generate_node, critique_node


class _FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Yes, revenue grew by 20% {{a}}"


def test_retrieve_node_uses_retrievers():
    chunks = [Chunk("a", "revenue grew", "d", "s", 1)]
    retrievers = {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
                  "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}
    state = {"question": "revenue", "intent": "hybrid", "retrieved_chunks": []}
    out = retrieve_node(state, retrievers=retrievers)
    assert len(out["retrieved_chunks"]) == 1


def test_critique_node_returns_verdict():
    state = {"question": "did revenue grow?",
             "retrieved_chunks": [{"chunk_id": "a", "text": "revenue grew 20%"}]}
    out = critique_node(state)
    assert out["retrieval_verdict"] in {"correct", "ambiguous", "incorrect"}


def test_generate_node_produces_citations():
    state = {"question": "did revenue grow?",
             "retrieved_chunks": [{"chunk_id": "a", "text": "revenue grew 20%"}]}
    out = generate_node(state, llm=_FakeLLM())
    assert "{{a}}" in out["candidate_answer"]
    assert out["citations"] == ["a"]


class _FakeRouteLLM:
    def __init__(self, intent: str):
        self.intent = intent

    def complete(self, prompt: str) -> str:
        return self.intent


def test_route_node_llm_factoid_maps_to_keyword():
    out = route_node({"question": "who invented the telephone"}, llm=_FakeRouteLLM("factoid"))
    assert out["intent"] == "keyword"
    assert out["route_decision"]["semantic_intent"] == "factoid"
    assert out["needs_clarify"] is False


def test_route_node_llm_comparison_maps_to_semantic():
    out = route_node({"question": "compare A and B"}, llm=_FakeRouteLLM("comparison"))
    assert out["intent"] == "semantic"


def test_route_node_llm_multihop_maps_to_hybrid():
    out = route_node({"question": "who directed the movie A starred in"}, llm=_FakeRouteLLM("multi-hop"))
    assert out["intent"] == "hybrid"


def test_route_node_ambiguous_short_question():
    out = route_node({"question": "who?"})
    assert out["needs_clarify"] is True
