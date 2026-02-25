from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    # Deprecated in BYOM template branch; profile controls model settings.
    gemini_model_triage: str = ""
    gemini_model_summary: str = ""

    profiles_dir: Path = Path("profiles")
    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")
    docs_dir: Path = Path("docs")
    max_candidates: int | None = None
    max_accepted: int | None = None
    include_borderline: bool | None = None
    max_borderline_pdfs: int | None = None
    text_head_chars: int | None = None
    text_tail_chars: int | None = None
    summary_max_input_tokens: int | None = None
    llm_temperature_triage: float | None = None
    llm_temperature_summary: float | None = None
    llm_max_output_tokens_triage: int | None = None
    llm_max_output_tokens_summary: int | None = None


def load_config() -> Config:
    return Config(
        gemini_model_triage=os.environ.get("GEMINI_MODEL_TRIAGE", ""),
        gemini_model_summary=os.environ.get("GEMINI_MODEL_SUMMARY", ""),
        profiles_dir=Path(os.environ.get("PROFILES_DIR", "profiles")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs")),
        data_dir=Path(os.environ.get("DATA_DIR", "data")),
        docs_dir=Path(os.environ.get("DOCS_DIR", "docs")),
        max_candidates=int(os.environ["MAX_CANDIDATES"]) if "MAX_CANDIDATES" in os.environ else None,
        max_accepted=int(os.environ["MAX_ACCEPTED"]) if "MAX_ACCEPTED" in os.environ else None,
        include_borderline=(
            os.environ.get("INCLUDE_BORDERLINE", "").lower() in {"1", "true", "yes"}
            if "INCLUDE_BORDERLINE" in os.environ
            else None
        ),
        max_borderline_pdfs=(
            int(os.environ["MAX_BORDERLINE_PDFS"]) if "MAX_BORDERLINE_PDFS" in os.environ else None
        ),
        text_head_chars=int(os.environ["TEXT_HEAD_CHARS"]) if "TEXT_HEAD_CHARS" in os.environ else None,
        text_tail_chars=int(os.environ["TEXT_TAIL_CHARS"]) if "TEXT_TAIL_CHARS" in os.environ else None,
        summary_max_input_tokens=(
            int(os.environ["SUMMARY_MAX_INPUT_TOKENS"]) if "SUMMARY_MAX_INPUT_TOKENS" in os.environ else None
        ),
        llm_temperature_triage=(
            float(os.environ["LLM_TEMPERATURE_TRIAGE"]) if "LLM_TEMPERATURE_TRIAGE" in os.environ else None
        ),
        llm_temperature_summary=(
            float(os.environ["LLM_TEMPERATURE_SUMMARY"]) if "LLM_TEMPERATURE_SUMMARY" in os.environ else None
        ),
        llm_max_output_tokens_triage=(
            int(os.environ["LLM_MAX_OUTPUT_TOKENS_TRIAGE"])
            if "LLM_MAX_OUTPUT_TOKENS_TRIAGE" in os.environ
            else None
        ),
        llm_max_output_tokens_summary=(
            int(os.environ["LLM_MAX_OUTPUT_TOKENS_SUMMARY"])
            if "LLM_MAX_OUTPUT_TOKENS_SUMMARY" in os.environ
            else None
        ),
    )
