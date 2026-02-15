from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    gemini_model_triage: str
    gemini_model_summary: str
    arxiv_rate_limit_seconds: float = 2.0
    pdf_rate_limit_seconds: float = 5.0
    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")
    docs_dir: Path = Path("docs")
    max_candidates: int = 500
    max_accepted: int = 80
    include_borderline: bool = False
    max_borderline_pdfs: int = 20
    text_head_chars: int = 80_000
    text_tail_chars: int = 20_000
    llm_temperature_triage: float = 0.2
    llm_temperature_summary: float = 0.2
    llm_max_output_tokens_triage: int = 1024
    llm_max_output_tokens_summary: int = 2048


def load_config() -> Config:
    return Config(
        gemini_model_triage=os.environ.get("GEMINI_MODEL_TRIAGE", "gemini-2.0-flash"),
        gemini_model_summary=os.environ.get("GEMINI_MODEL_SUMMARY", "gemini-2.0-pro"),
        arxiv_rate_limit_seconds=float(os.environ.get("ARXIV_RATE_LIMIT_SECONDS", "2")),
        pdf_rate_limit_seconds=float(os.environ.get("PDF_RATE_LIMIT_SECONDS", "5")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        llm_temperature_triage=float(os.environ.get("LLM_TEMPERATURE_TRIAGE", "0.2")),
        llm_temperature_summary=float(os.environ.get("LLM_TEMPERATURE_SUMMARY", "0.2")),
        llm_max_output_tokens_triage=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS_TRIAGE", "1024")),
        llm_max_output_tokens_summary=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS_SUMMARY", "2048")),
    )
