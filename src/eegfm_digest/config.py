from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

DEFAULT_OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"


@dataclass(frozen=True)
class Config:
    llm_model_triage: str
    llm_model_summary: str
    arxiv_rate_limit_seconds: float = 2.0
    arxiv_connect_timeout_seconds: float = 10.0
    arxiv_read_timeout_seconds: float = 60.0
    arxiv_retries: int = 2
    arxiv_retry_backoff_seconds: float = 2.0
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
    summary_max_input_tokens: int = 120_000
    llm_temperature_triage: float = 0.2
    llm_temperature_summary: float = 0.2
    llm_max_output_tokens_triage: int = 1024
    llm_max_output_tokens_summary: int = 2048


def load_config() -> Config:
    return Config(
        llm_model_triage=(
            os.environ.get("OPENROUTER_MODEL_TRIAGE")
            or os.environ.get("LLM_MODEL_TRIAGE")
            or os.environ.get("GEMINI_MODEL_TRIAGE")
            or DEFAULT_OPENROUTER_MODEL
        ),
        llm_model_summary=(
            os.environ.get("OPENROUTER_MODEL_SUMMARY")
            or os.environ.get("LLM_MODEL_SUMMARY")
            or os.environ.get("GEMINI_MODEL_SUMMARY")
            or DEFAULT_OPENROUTER_MODEL
        ),
        arxiv_rate_limit_seconds=float(os.environ.get("ARXIV_RATE_LIMIT_SECONDS", "2")),
        arxiv_connect_timeout_seconds=float(os.environ.get("ARXIV_CONNECT_TIMEOUT_SECONDS", "10")),
        arxiv_read_timeout_seconds=float(os.environ.get("ARXIV_READ_TIMEOUT_SECONDS", "60")),
        arxiv_retries=int(os.environ.get("ARXIV_RETRIES", "2")),
        arxiv_retry_backoff_seconds=float(os.environ.get("ARXIV_RETRY_BACKOFF_SECONDS", "2")),
        pdf_rate_limit_seconds=float(os.environ.get("PDF_RATE_LIMIT_SECONDS", "5")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        docs_dir=Path(os.environ.get("DOCS_DIR", "docs")),
        max_candidates=int(os.environ.get("MAX_CANDIDATES", "500")),
        max_accepted=int(os.environ.get("MAX_ACCEPTED", "80")),
        include_borderline=os.environ.get("INCLUDE_BORDERLINE", "false").lower() in {"1", "true", "yes"},
        max_borderline_pdfs=int(os.environ.get("MAX_BORDERLINE_PDFS", "20")),
        text_head_chars=int(os.environ.get("TEXT_HEAD_CHARS", "80000")),
        text_tail_chars=int(os.environ.get("TEXT_TAIL_CHARS", "20000")),
        summary_max_input_tokens=int(os.environ.get("SUMMARY_MAX_INPUT_TOKENS", "120000")),
        llm_temperature_triage=float(os.environ.get("LLM_TEMPERATURE_TRIAGE", "0.2")),
        llm_temperature_summary=float(os.environ.get("LLM_TEMPERATURE_SUMMARY", "0.2")),
        llm_max_output_tokens_triage=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS_TRIAGE", "1024")),
        llm_max_output_tokens_summary=int(os.environ.get("LLM_MAX_OUTPUT_TOKENS_SUMMARY", "2048")),
    )
