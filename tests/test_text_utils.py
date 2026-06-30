from paper_agent.code_finder import extract_urls, normalize_url
from paper_agent.live_agent import fallback_queries, is_arxiv_transient_error, parse_json_string_list
from paper_agent.pdf import chunk_text
from paper_agent.source_manager import is_cloneable_repo_url


def test_extract_urls_trims_punctuation():
    assert extract_urls("Code: https://github.com/org/repo.") == ["https://github.com/org/repo"]


def test_normalize_url_removes_git_suffix():
    assert normalize_url("https://github.com/org/repo.git") == "https://github.com/org/repo"


def test_chunk_text_keeps_short_text():
    chunks = chunk_text("short abstract", chunk_size=100, chunk_overlap=10, min_chunk_chars=50)
    assert chunks == ["short abstract"]


def test_papers_with_code_link_is_not_cloneable():
    assert not is_cloneable_repo_url("https://paperswithcode.com/paper/example")
    assert is_cloneable_repo_url("https://github.com/org/repo")


def test_parse_json_string_list_from_fenced_block():
    text = '```json\n["retrieval augmented generation", "agent planning"]\n```'
    assert parse_json_string_list(text) == ["retrieval augmented generation", "agent planning"]


def test_fallback_queries_extracts_english_keyword_from_chinese_goal():
    assert fallback_queries("我想学习 meshMAE的相关知识") == ["meshMAE"]


def test_arxiv_rate_limit_is_transient():
    assert is_arxiv_transient_error(Exception("Page request resulted in HTTP 429"))
