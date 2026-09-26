"""引用编号：答案里的 {{chunk_id}} 要能对上「来源 [n]」，才叫可溯源。"""
import run_graph


def _chunks():
    return [
        {"chunk_id": "aaa111", "text": "t1", "score": 1.0,
         "source_doc": "opera.pdf", "page": 3, "section": "Acts"},
        {"chunk_id": "bbb222", "text": "t2", "score": 0.9,
         "source_doc": "opera.pdf", "page": 5, "section": "Notes"},
    ]


def test_number_citations_rewrites_markers_to_bracket_numbers():
    rendered, sources = run_graph.number_citations(
        "four acts {{aaa111}}, three acts {{bbb222}}", _chunks())
    assert rendered == "four acts [1], three acts [2]"
    assert [s["n"] for s in sources] == [1, 2]


def test_sources_carry_traceable_metadata():
    _, sources = run_graph.number_citations("x {{aaa111}}", _chunks())
    assert sources[0]["source_doc"] == "opera.pdf"
    assert sources[0]["page"] == 3
    assert sources[0]["section"] == "Acts"


def test_repeated_citation_keeps_one_number():
    rendered, sources = run_graph.number_citations(
        "a {{aaa111}} b {{aaa111}}", _chunks())
    assert rendered == "a [1] b [1]"
    assert len(sources) == 1


def test_unknown_marker_is_marked_not_silently_dropped():
    """引用校验兜底：对不上的 id 要显式标出来，不能悄悄消失。"""
    rendered, _ = run_graph.number_citations("bad {{deadbeef}}", _chunks())
    assert "[?]" in rendered
