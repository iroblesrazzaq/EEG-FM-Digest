import json
from pathlib import Path

from eegfm_digest.summarize import summarize_paper
from eegfm_digest.triage import load_schema


class CaptureLLM:
    def __init__(self, token_result):
        self.token_result = token_result
        self.prompts: list[str] = []

    def count_tokens(self, content: str) -> int:
        if isinstance(self.token_result, Exception):
            raise self.token_result
        return int(self.token_result)

    def generate(self, prompt, schema=None):  # noqa: ANN001
        self.prompts.append(prompt)
        return json.dumps(
            {
                "arxiv_id_base": "2501.10000",
                "title": "EEG FM",
                "published_date": "2025-01-10",
                "categories": ["cs.LG"],
                "paper_type": "method",
                "one_liner": "A concise summary line for the digest.",
                "detailed_summary": (
                    "This paper proposes a deterministic EEG representation method that combines "
                    "self-supervised pretraining and lightweight finetuning for transfer. "
                    "The core novelty is its stable objective and architecture choices designed "
                    "for reproducible downstream performance across benchmarks."
                ),
                "unique_contribution": "A deterministic contribution sentence.",
                "key_points": ["k1 point", "k2 point", "k3 point"],
                "data_scale": {
                    "datasets": ["Dataset-A"],
                    "subjects": 10,
                    "eeg_hours": 2.5,
                    "channels": 64,
                },
                "method": {
                    "architecture": "Transformer",
                    "objective": "Masked prediction",
                    "pretraining": "Self-supervised pretraining",
                    "finetuning": "Linear probe",
                },
                "evaluation": {
                    "tasks": ["classification"],
                    "benchmarks": ["Benchmark-A"],
                    "headline_results": ["Improved AUROC"],
                },
                "open_source": {
                    "code_url": None,
                    "weights_url": None,
                    "license": None,
                },
                "tags": {
                    "paper_type": ["eeg-fm"],
                    "backbone": ["transformer"],
                    "objective": ["masked-reconstruction"],
                    "tokenization": ["time-patch"],
                    "topology": ["fixed-montage"],
                },
                "limitations": ["limited cohorts", "needs broader evaluation"],
                "used_fulltext": True,
                "notes": "placeholder",
            }
        )


PAPER = {
    "arxiv_id_base": "2501.10000",
    "title": "EEG FM",
    "categories": ["cs.LG"],
    "published": "2025-01-10T00:00:00Z",
    "summary": "abstract text",
}
TRIAGE = {
    "arxiv_id_base": "2501.10000",
    "decision": "accept",
    "confidence": 0.85,
    "reasons": ["eeg central", "fm framing"],
}
SLICES = {
    "abstract": "a",
    "introduction": "b",
    "methods": "c",
    "results": "d",
    "conclusion": "e",
    "excerpt": "f",
}


def _payload_from_prompt(prompt: str) -> dict:
    marker = "PAYLOAD:\n"
    return json.loads(prompt.split(marker, 1)[1])


def test_summarize_uses_fulltext_when_under_token_limit():
    schema = load_schema(Path("schemas/summary.json"))
    llm = CaptureLLM(token_result=50)
    summarize_paper(
        paper=PAPER,
        triage=TRIAGE,
        raw_fulltext="long full text",
        fulltext_slices=SLICES,
        used_fulltext=True,
        notes="meta",
        llm=llm,
        prompt_template="PAYLOAD:\n{{INPUT_JSON}}",
        repair_template="schema={{SCHEMA_JSON}} bad={{BAD_OUTPUT}}",
        schema=schema,
        max_input_tokens=100,
    )
    payload = _payload_from_prompt(llm.prompts[0])
    assert "fulltext" in payload
    assert "fulltext_slices" not in payload


def test_summarize_uses_slices_when_over_token_limit():
    schema = load_schema(Path("schemas/summary.json"))
    llm = CaptureLLM(token_result=500)
    summarize_paper(
        paper=PAPER,
        triage=TRIAGE,
        raw_fulltext="long full text",
        fulltext_slices=SLICES,
        used_fulltext=True,
        notes="meta",
        llm=llm,
        prompt_template="PAYLOAD:\n{{INPUT_JSON}}",
        repair_template="schema={{SCHEMA_JSON}} bad={{BAD_OUTPUT}}",
        schema=schema,
        max_input_tokens=100,
    )
    payload = _payload_from_prompt(llm.prompts[0])
    assert "fulltext" not in payload
    assert payload["fulltext_slices"]["methods"] == "c"


def test_summarize_uses_slices_when_count_tokens_fails():
    schema = load_schema(Path("schemas/summary.json"))
    llm = CaptureLLM(token_result=RuntimeError("count failed"))
    summarize_paper(
        paper=PAPER,
        triage=TRIAGE,
        raw_fulltext="long full text",
        fulltext_slices=SLICES,
        used_fulltext=True,
        notes="meta",
        llm=llm,
        prompt_template="PAYLOAD:\n{{INPUT_JSON}}",
        repair_template="schema={{SCHEMA_JSON}} bad={{BAD_OUTPUT}}",
        schema=schema,
        max_input_tokens=100,
    )
    payload = _payload_from_prompt(llm.prompts[0])
    assert "fulltext" not in payload
    assert "fulltext_slices" in payload


def test_summarize_normalizes_paper_type_and_numeric_fields():
    schema = load_schema(Path("schemas/summary.json"))

    class ListPaperTypeLLM(CaptureLLM):
        def generate(self, prompt, schema=None):  # noqa: ANN001
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "paper_type": ["eeg-fm"],
                    "one_liner": "Compact EEG FM summary.",
                    "detailed_summary": (
                        "This paper proposes a compact EEG foundation model architecture with "
                        "efficient attention for long sequences and broad transfer. "
                        "It demonstrates improved efficiency and competitive results across tasks."
                    ),
                    "unique_contribution": "Efficient alternating attention for EEG FM scaling.",
                    "key_points": ["Single point only"],
                    "data_scale": {
                        "datasets": ["Dataset-A"],
                        "subjects": "10k+",
                        "eeg_hours": "20000+",
                        "channels": "64",
                    },
                    "method": {
                        "architecture": "Transformer",
                        "objective": "Masked prediction",
                        "pretraining": "Self-supervised",
                        "finetuning": "Linear probe",
                    },
                    "evaluation": {
                        "tasks": ["classification"],
                        "benchmarks": ["Benchmark-A"],
                        "headline_results": ["Improved AUROC"],
                    },
                    "open_source": {"code_url": None, "weights_url": None, "license": None},
                    "tags": {
                        "paper_type": ["eeg-fm"],
                        "backbone": ["transformer"],
                        "objective": ["masked-reconstruction"],
                        "tokenization": ["time-patch"],
                        "topology": ["channel-flexible"],
                    },
                    "limitations": ["limited cohorts", "single data source"],
                }
            )

    llm = ListPaperTypeLLM(token_result=50)
    out = summarize_paper(
        paper=PAPER,
        triage=TRIAGE,
        raw_fulltext="long full text",
        fulltext_slices=SLICES,
        used_fulltext=True,
        notes="meta",
        llm=llm,
        prompt_template="PAYLOAD:\n{{INPUT_JSON}}",
        repair_template="schema={{SCHEMA_JSON}} bad={{BAD_OUTPUT}}",
        schema=schema,
        max_input_tokens=100,
    )
    assert out["paper_type"] == "new_model"
    assert out["data_scale"]["subjects"] == 10000.0
    assert out["data_scale"]["eeg_hours"] == 20000.0
    assert out["data_scale"]["channels"] == 64.0
    assert len(out["key_points"]) >= 2
