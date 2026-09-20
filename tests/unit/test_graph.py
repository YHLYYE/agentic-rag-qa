from graph.build import build_graph


def test_graph_builds_and_has_nodes():
    graph = build_graph(retrievers={"dense": lambda q, k: [], "bm25": lambda q, k: []},
                        llm=None, max_retry=2)
    names = [n for n in graph.nodes]
    for expected in ["route", "retrieve", "critique", "generate", "verify"]:
        assert expected in names


def test_graph_invokes_end_to_end_happy_path():
    from models import Chunk, RetrievedChunk
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1)]
    retrievers = {"dense": lambda q, k: [RetrievedChunk(chunks[0], 0.9)],
                  "bm25": lambda q, k: [RetrievedChunk(chunks[0], 0.5)]}
    graph = build_graph(retrievers=retrievers, llm=_LLM(), max_retry=2)
    result = graph.invoke({"question": "did revenue grow?",
                           "intent": "hybrid", "retry_count": 0,
                           "retrieved_chunks": [], "citations": [],
                           "retrieval_verdict": "", "grounding_verdict": "",
                           "candidate_answer": "", "final_answer": "",
                           "route_decision": {}})
    assert result["grounding_verdict"] == "supported"


class _LLM:
    def complete(self, prompt: str) -> str:
        return "Yes, revenue grew 20% {{a}}"
