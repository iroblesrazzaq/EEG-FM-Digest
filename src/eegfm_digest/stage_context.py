"""Shared schema, prompt, and cache-descriptor setup for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cache_meta import (
    SUMMARY_STAGE_LOGIC_VERSION,
    TRIAGE_STAGE_LOGIC_VERSION,
    build_stage_descriptor,
)
from .llm import LLMCallConfig
from .resources import prompt_path, schema_path
from .triage import load_schema


@dataclass(frozen=True)
class TriageStageContext:
    schema: dict[str, Any]
    triage_prompt: str
    summarize_prompt: str
    repair_prompt: str
    descriptor: dict[str, Any]


@dataclass(frozen=True)
class SummaryStageContext:
    schema: dict[str, Any]
    summarize_prompt: str
    repair_prompt: str
    descriptor: dict[str, Any]


def read_repo_text(path) -> str:
    from . import pipeline

    return pipeline._read(path)


def load_triage_stage_context(llm_config: LLMCallConfig) -> TriageStageContext:
    triage_schema = load_schema(schema_path("triage.json"))
    triage_prompt = read_repo_text(prompt_path("triage.md"))
    summarize_prompt = read_repo_text(prompt_path("summarize.md"))
    repair_prompt = read_repo_text(prompt_path("repair_json.md"))
    descriptor = build_stage_descriptor(
        stage="triage",
        provider=llm_config.provider,
        model=llm_config.model,
        prompt_template=triage_prompt,
        repair_template=repair_prompt,
        schema=triage_schema,
        stage_logic_version=TRIAGE_STAGE_LOGIC_VERSION,
    )
    return TriageStageContext(
        schema=triage_schema,
        triage_prompt=triage_prompt,
        summarize_prompt=summarize_prompt,
        repair_prompt=repair_prompt,
        descriptor=descriptor,
    )


def load_summary_stage_context(llm_config: LLMCallConfig) -> SummaryStageContext:
    summary_schema = load_schema(schema_path("summary.json"))
    summarize_prompt = read_repo_text(prompt_path("summarize.md"))
    repair_prompt = read_repo_text(prompt_path("repair_json.md"))
    descriptor = build_stage_descriptor(
        stage="summary",
        provider=llm_config.provider,
        model=llm_config.model,
        prompt_template=summarize_prompt,
        repair_template=repair_prompt,
        schema=summary_schema,
        stage_logic_version=SUMMARY_STAGE_LOGIC_VERSION,
    )
    return SummaryStageContext(
        schema=summary_schema,
        summarize_prompt=summarize_prompt,
        repair_prompt=repair_prompt,
        descriptor=descriptor,
    )
