import json
from pathlib import Path

from eegfm_digest.summarize import summarize_paper
from eegfm_digest.triage import load_schema, triage_paper


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = outputs
        self.i = 0

    def generate(self, prompt, schema=None):
        out = self.outputs[self.i]
        self.i += 1
        return out


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
        "arxiv_id_base": "2501.12345",
        "is_eeg_related": True,
        "is_foundation_model_related": True,
        "borderline": False,
        "paper_type": "new_model",
        "confidence": 0.8,
        "reasons": ["reason"],
        "suggested_digest_tags": ["tag"],
        "decision": "accept",
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


def test_summary_fallback_path():
    schema = load_schema(Path("schemas/summary.json"))
    paper = {
        "arxiv_id_base": "2501.12345",
        "title": "EEG FM",
        "categories": ["cs.LG"],
        "published": "2025-01-10T00:00:00Z",
        "summary": "abs",
    }
    triage = {"paper_type": "method", "suggested_digest_tags": ["eeg", "fm", "ssl"]}
    llm = FakeLLM(["{bad", "also bad"])
    out = summarize_paper(
        paper,
        triage,
        "txt",
        False,
        "abstract_only",
        llm,
        Path("prompts/summarize.md").read_text(),
        Path("prompts/repair_json.md").read_text(),
        schema,
    )
    assert out["arxiv_id_base"] == "2501.12345"
    assert out["used_fulltext"] is False
    assert "summary_json_error" in out["notes"]
