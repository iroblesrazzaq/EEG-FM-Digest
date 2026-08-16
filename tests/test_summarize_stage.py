from eegfm_digest.summarize_stage import PdfTextResult, summary_inputs_from_pdf_result


def _paper() -> dict:
    return {
        "arxiv_id_base": "2501.00001",
        "summary": "Title abstract from arXiv.",
    }


def test_summary_inputs_uses_fulltext_when_extract_succeeds():
    result = PdfTextResult(
        raw_text="Abstract\nEEG abstract\n\nIntroduction\nIntro",
        pdf_state={},
        notes='{"tool": "pymupdf"}',
    )
    inputs = summary_inputs_from_pdf_result(
        _paper(),
        result,
        head_chars=8000,
        excerpt_chars=4000,
        tail_chars=1000,
    )
    assert inputs is not None
    assert inputs.used_fulltext is True
    assert "EEG abstract" in inputs.raw_text
    assert inputs.notes == '{"tool": "pymupdf"}'


def test_summary_inputs_falls_back_to_abstract_on_pdf_failure():
    result = PdfTextResult(
        raw_text="",
        pdf_state={},
        notes="summary_skipped:pdf_failed:HTTPStatusError",
    )
    inputs = summary_inputs_from_pdf_result(
        _paper(),
        result,
        head_chars=8000,
        excerpt_chars=4000,
        tail_chars=1000,
    )
    assert inputs is not None
    assert inputs.used_fulltext is False
    assert inputs.raw_text == "Title abstract from arXiv."
    assert inputs.slices["abstract"] == "Title abstract from arXiv."
    assert inputs.slices["excerpt"] == "Title abstract from arXiv."
    assert inputs.notes.endswith("abstract_only_fallback")


def test_summary_inputs_returns_none_for_no_pdf_mode():
    result = PdfTextResult(
        raw_text="",
        pdf_state={},
        notes="summary_skipped:no_pdf_mode",
    )
    assert (
        summary_inputs_from_pdf_result(
            _paper(),
            result,
            head_chars=8000,
            excerpt_chars=4000,
            tail_chars=1000,
        )
        is None
    )
