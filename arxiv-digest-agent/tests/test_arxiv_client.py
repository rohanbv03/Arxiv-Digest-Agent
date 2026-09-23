from arxiv_agent import arxiv_client


def test_extract_id_from_bare_id():
    assert arxiv_client.extract_arxiv_id("2401.12345") == "2401.12345"


def test_extract_id_strips_version():
    assert arxiv_client.extract_arxiv_id("2401.12345v2") == "2401.12345"


def test_extract_id_from_abs_url():
    assert arxiv_client.extract_arxiv_id("https://arxiv.org/abs/2401.12345") == "2401.12345"


def test_extract_id_from_pdf_url():
    assert arxiv_client.extract_arxiv_id("https://arxiv.org/pdf/2401.12345") == "2401.12345"


def test_extract_id_returns_none_for_topic():
    assert arxiv_client.extract_arxiv_id("recent work on KV-cache compression") is None


def test_clean_topic_query_strips_filler():
    cleaned = arxiv_client.clean_topic_query("recent work on KV-cache compression for LLMs")
    assert "recent work on" not in cleaned.lower()
    assert "kv-cache compression" in cleaned.lower()


def test_clean_topic_query_falls_back_to_original_if_emptied():
    # A query that's ONLY filler words shouldn't collapse to an empty string.
    cleaned = arxiv_client.clean_topic_query("what is the latest")
    assert cleaned.strip() != ""
