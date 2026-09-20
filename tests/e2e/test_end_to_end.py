from models import Chunk
from rag.bm25 import BM25Retriever
from graph.build import build_graph


def test_full_pipeline_smoke():
    chunks = [Chunk("a", "revenue grew 20%", "d", "s", 1),
              Chunk("b", "operating income stable", "d", "s2", 2)]
    rets = {"dense": BM25Retriever(chunks).retrieve,
            "bm25": BM25Retriever(chunks).retrieve}

    class LLM:
        def complete(self, p):
            return "revenue grew 20% {{a}}"

    g = build_graph(rets, LLM(), max_retry=2)
    out = g.invoke({"question": "did revenue grow?", "intent": "hybrid",
                    "retry_count": 0, "retrieved_chunks": [], "citations": [],
                    "retrieval_verdict": "", "grounding_verdict": "",
                    "candidate_answer": "", "final_answer": "", "route_decision": {}})
    assert out["grounding_verdict"] == "supported"
    assert "{{a}}" in out["candidate_answer"]
