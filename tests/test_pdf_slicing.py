from eegfm_digest.pdf import slice_paper_text


def test_slice_paper_text_is_deterministic_with_expected_keys():
    text = """
Abstract
This paper studies EEG pretraining.

1 Introduction
We introduce a model for transfer.

Methods
We pretrain with masked prediction.

Results
The model improves on benchmarks.

Conclusion
The method transfers across tasks.
""".strip()

    first = slice_paper_text(text, excerpt_chars=120, tail_chars=30)
    second = slice_paper_text(text, excerpt_chars=120, tail_chars=30)

    expected_keys = {
        "abstract",
        "introduction",
        "methods",
        "results",
        "conclusion",
        "excerpt",
    }
    assert set(first.keys()) == expected_keys
    assert first == second
    assert first["introduction"]
    assert first["excerpt"]


def test_slice_paper_text_returns_empty_sections_when_headings_missing():
    text = "No formal headings exist in this extracted text. " * 20
    out = slice_paper_text(text, excerpt_chars=80)

    assert out["abstract"] == ""
    assert out["introduction"] == ""
    assert out["methods"] == ""
    assert out["results"] == ""
    assert out["conclusion"] == ""
    assert out["excerpt"] == text[:80].strip()
