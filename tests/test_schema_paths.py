import json
from pathlib import Path

from eegfm_digest.summarize import summarize_paper
from eegfm_digest.triage import load_schema, triage_paper


class FakeLLM:
    def __init__(self, outputs, token_count: int = 100):
        self.outputs = outputs
        self.i = 0
        self.prompts: list[str] = []
        self.token_count = token_count

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        out = self.outputs[self.i]
        self.i += 1
        return out

    def count_tokens(self, content: str) -> int:
        return self.token_count


def _summary_slices() -> dict[str, str]:
    return {
        "abstract": "",
        "introduction": "",
        "methods": "",
        "results": "",
        "conclusion": "",
        "excerpt": "fallback excerpt",
    }


def test_triage_prompt_payload_is_title_and_abstract_only():
    triage_schema = load_schema(Path("schemas/triage.json"))
    paper = {
        "arxiv_id_base": "2501.99999",
        "title": "Title only paper",
        "authors": ["A"],
        "categories": ["cs.LG"],
        "published": "2025-01-10T00:00:00Z",
        "summary": "Abstract only",
        "links": {"abs": "x", "pdf": "y"},
    }
    model_out = {
        "decision": "accept",
        "confidence": 0.7,
        "reasons": ["eeg is central", "pretraining is explicit"],
    }
    llm = FakeLLM([json.dumps(model_out)])
    out = triage_paper(
        paper,
        llm,
        "Title: {{TITLE}}\n\nAbstract: {{ABSTRACT}}",
        Path("prompts/repair_json.md").read_text(),
        triage_schema,
    )
    prompt = llm.prompts[0]
    assert "Title: Title only paper" in prompt
    assert "Abstract: Abstract only" in prompt
    assert "authors" not in prompt
    assert "categories" not in prompt
    assert "published_date" not in prompt
    assert "links" not in prompt
    assert "arxiv_id_base" not in prompt
    assert out == {
        "arxiv_id_base": "2501.99999",
        "decision": "accept",
        "confidence": 0.7,
        "reasons": ["eeg is central", "pretraining is explicit"],
    }


def test_triage_repair_path():
    triage_schema = load_schema(Path("schemas/triage.json"))
    paper = {
        "arxiv_id_base": "2501.12345",
        "title": "EEG paper",
        "authors": ["A"],
        "categories": ["cs.LG"],
        "published": "2025-01-10T00:00:00Z",
        "summary": "abstract",
        "links": {"abs": "x", "pdf": "y"},
    }
    repaired = {
        "decision": "accept",
        "confidence": 0.8,
        "reasons": ["eeg is central modality", "uses pretrained transferable representation"],
    }
    llm = FakeLLM(["not json", json.dumps(repaired)])
    out = triage_paper(
        paper,
        llm,
        Path("prompts/triage.md").read_text(),
        Path("prompts/repair_json.md").read_text(),
        triage_schema,
    )
    assert out["decision"] == "accept"
    assert out["arxiv_id_base"] == "2501.12345"


def test_summary_fallback_path():
    schema = load_schema(Path("schemas/summary.json"))
    paper = {
        "arxiv_id_base": "2501.12345",
        "title": "EEG FM",
        "categories": ["cs.LG"],
        "published": "2025-01-10T00:00:00Z",
        "summary": "abs",
    }
    triage = {"decision": "accept", "confidence": 0.9, "reasons": ["a", "b"]}
    llm = FakeLLM(["{bad", "also bad"])
    out = summarize_paper(
        paper=paper,
        triage=triage,
        raw_fulltext="txt",
        fulltext_slices=_summary_slices(),
        used_fulltext=False,
        notes="abstract_only",
        llm=llm,
        prompt_template=Path("prompts/summarize.md").read_text(),
        repair_template=Path("prompts/repair_json.md").read_text(),
        schema=schema,
        max_input_tokens=1,
    )
    assert out["arxiv_id_base"] == "2501.12345"
    assert out["used_fulltext"] is False
    assert "summary_json_error" in out["notes"]
