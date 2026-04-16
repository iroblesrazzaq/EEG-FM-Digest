from eegfm_digest.arxiv import build_keyword_query
from eegfm_digest.config import load_topic
from eegfm_digest.keywords import (
    ARXIV_CATEGORIES,
    KEYWORD_ANCHOR_TERMS,
    KEYWORD_QUERY_A_TERMS,
    KEYWORD_QUERY_B_TERMS,
    QUERY_A,
    QUERY_B,
)


def test_eeg_fm_topic_matches_legacy_keywords() -> None:
    topic = load_topic("eeg-fm")

    assert topic.slug == "eeg-fm"
    assert set(topic.arxiv_categories) == ARXIV_CATEGORIES
    assert topic.keyword_anchor_terms == KEYWORD_ANCHOR_TERMS
    assert topic.keyword_query_a_terms == KEYWORD_QUERY_A_TERMS
    assert topic.keyword_query_b_terms == KEYWORD_QUERY_B_TERMS
    assert build_keyword_query(topic.keyword_anchor_terms, topic.keyword_query_a_terms) == QUERY_A
    assert build_keyword_query(topic.keyword_anchor_terms, topic.keyword_query_b_terms) == QUERY_B
    assert str(topic.triage_prompt_path) == "prompts/triage.md"
    assert str(topic.summarize_prompt_path) == "prompts/summarize.md"
