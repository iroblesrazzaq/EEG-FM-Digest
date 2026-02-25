from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .arxiv import fetch_month_candidates
from .config import Config
from .db import DigestDB
from .llm_openai_compat import OpenAICompatClient, OpenAICompatConfig, RateLimitError, load_api_key
from .pdf import download_pdf, extract_text, slice_paper_text
from .profile import DigestProfile
from .render import build_digest, write_json, write_jsonl
from .site import update_home, write_month_site
from .summarize import summarize_paper
from .triage import load_schema, triage_paper


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _empty_pdf_state() -> dict[str, object | None]:
    return {
        "downloaded": False,
        "pdf_path": None,
        "text_path": None,
        "extract_meta": None,
    }


def _triage_view(triage: dict[str, object] | None) -> dict[str, object]:
    triage = triage or {}
    reasons = triage.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": triage.get("decision", "reject"),
        "confidence": float(triage.get("confidence", 0.0)),
        "reasons": reasons,
    }


def _choose_int(override: int | None, cfg_value: int | None, profile_value: int) -> int:
    if override is not None:
        return int(override)
    if cfg_value is not None:
        return int(cfg_value)
    return int(profile_value)


def _choose_bool(override: bool | None, cfg_value: bool | None, profile_value: bool) -> bool:
    if override is not None:
        return bool(override)
    if cfg_value is not None:
        return bool(cfg_value)
    return bool(profile_value)


def _month_output_dir(cfg: Config, profile: DigestProfile, month: str) -> Path:
    return cfg.output_dir / profile.id / month


def _db_path(cfg: Config, profile: DigestProfile) -> Path:
    return cfg.data_dir / f"{profile.id}.sqlite"


def run_month(
    cfg: Config,
    profile: DigestProfile,
    month: str,
    no_pdf: bool = False,
    no_site: bool = False,
    force: bool = False,
    force_triage: bool = False,
    force_summary: bool = False,
    triage_sleep_seconds: float = 0.0,
    summary_sleep_seconds: float = 0.0,
    stop_on_rate_limit: bool = False,
    max_candidates_override: int | None = None,
    max_accepted_override: int | None = None,
    include_borderline_override: bool | None = None,
) -> None:
    force_triage = bool(force or force_triage)
    force_summary = bool(force or force_summary)

    max_candidates = _choose_int(max_candidates_override, cfg.max_candidates, profile.pipeline.max_candidates)
    max_accepted = _choose_int(max_accepted_override, cfg.max_accepted, profile.pipeline.max_accepted)
    include_borderline = _choose_bool(
        include_borderline_override,
        cfg.include_borderline,
        profile.pipeline.include_borderline,
    )
    max_borderline_pdfs = _choose_int(None, cfg.max_borderline_pdfs, profile.pipeline.max_borderline_pdfs)
    text_tail_chars = _choose_int(None, cfg.text_tail_chars, profile.pipeline.text_tail_chars)
    summary_max_input_tokens = _choose_int(
        None,
        cfg.summary_max_input_tokens,
        profile.pipeline.summary_max_input_tokens,
    )

    month_out = _month_output_dir(cfg, profile, month)
    month_out.mkdir(parents=True, exist_ok=True)
    db = DigestDB(_db_path(cfg, profile))

    triage_schema = load_schema(profile.paths.triage_schema)
    summary_schema = load_schema(profile.paths.summary_schema)
    triage_prompt = _read(profile.paths.triage_prompt)
    summarize_prompt = _read(profile.paths.summary_prompt)
    repair_prompt = _read(profile.paths.repair_prompt)

    # Stage 1: fetch
    candidates = fetch_month_candidates(
        max_candidates,
        month,
        profile.retrieval.rate_limit_seconds,
        queries=profile.retrieval.queries,
        categories=profile.retrieval.categories,
        connect_timeout_seconds=profile.retrieval.connect_timeout_seconds,
        read_timeout_seconds=profile.retrieval.read_timeout_seconds,
        retries=profile.retrieval.retries,
        retry_backoff_seconds=profile.retrieval.retry_backoff_seconds,
    )
    write_json(month_out / "arxiv_raw.json", candidates)
    for candidate in candidates:
        db.upsert_paper(month, candidate)

    # Stage 2: triage
    triage_llm = OpenAICompatClient(
        OpenAICompatConfig(
            api_key=load_api_key(profile.llm.api_key_env),
            base_url=profile.llm.base_url,
            model=profile.llm.triage_model,
            temperature=(
                cfg.llm_temperature_triage
                if cfg.llm_temperature_triage is not None
                else profile.llm.temperature_triage
            ),
            max_output_tokens=(
                cfg.llm_max_output_tokens_triage
                if cfg.llm_max_output_tokens_triage is not None
                else profile.llm.max_output_tokens_triage
            ),
            request_timeout_seconds=profile.llm.request_timeout_seconds,
            token_chars_per_token=profile.llm.token_chars_per_token,
        )
    )
    triage_rows: list[dict[str, Any]] = []
    try:
        for paper in candidates:
            aid = paper["arxiv_id_base"]
            cached = None if force_triage else db.get_triage(aid)
            if cached:
                triage_rows.append(
                    {
                        "arxiv_id_base": aid,
                        "decision": cached.get("decision", "reject"),
                        "confidence": float(cached.get("confidence", 0.0)),
                        "reasons": list(cached.get("reasons", [])),
                    }
                )
                continue
            try:
                result = triage_paper(
                    paper=paper,
                    llm=triage_llm,
                    prompt_template=triage_prompt,
                    repair_template=repair_prompt,
                    schema=triage_schema,
                )
            except Exception as exc:
                if stop_on_rate_limit and isinstance(exc, RateLimitError):
                    raise
                result = {
                    "arxiv_id_base": aid,
                    "decision": "reject",
                    "confidence": 0.0,
                    "reasons": [f"triage_exception:{type(exc).__name__}", "automatic_reject_fallback"],
                }
            triage_rows.append(result)
            db.upsert_triage(month, result)
            if triage_sleep_seconds > 0:
                time.sleep(triage_sleep_seconds)
    finally:
        triage_llm.close()
    triage_rows = sorted(triage_rows, key=lambda x: x["arxiv_id_base"])
    write_jsonl(month_out / "triage.jsonl", triage_rows)

    # Stage 3: summarize
    summary_llm = OpenAICompatClient(
        OpenAICompatConfig(
            api_key=load_api_key(profile.llm.api_key_env),
            base_url=profile.llm.base_url,
            model=profile.llm.summary_model,
            temperature=(
                cfg.llm_temperature_summary
                if cfg.llm_temperature_summary is not None
                else profile.llm.temperature_summary
            ),
            max_output_tokens=(
                cfg.llm_max_output_tokens_summary
                if cfg.llm_max_output_tokens_summary is not None
                else profile.llm.max_output_tokens_summary
            ),
            request_timeout_seconds=profile.llm.request_timeout_seconds,
            token_chars_per_token=profile.llm.token_chars_per_token,
        )
    )
    triage_map = {row["arxiv_id_base"]: row for row in triage_rows}
    accepted = [p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "accept"]
    if include_borderline:
        borderline = [
            p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "borderline"
        ][:max_borderline_pdfs]
        accepted.extend(borderline)
    accepted = sorted(accepted, key=lambda x: (x["published"], x["arxiv_id_base"]))[:max_accepted]

    summaries: list[dict[str, Any]] = []
    summary_map: dict[str, dict[str, Any]] = {}
    pdf_map: dict[str, dict[str, object | None]] = {}
    try:
        for paper in accepted:
            aid = paper["arxiv_id_base"]
            cached_summary = None if force_summary else db.get_summary(aid)
            if cached_summary:
                summaries.append(cached_summary)
                summary_map[aid] = cached_summary
                pdf_map[aid] = _empty_pdf_state()
                continue

            pdf_state: dict[str, object | None] = _empty_pdf_state()
            raw_text = ""
            notes = "summary_not_attempted"

            if no_pdf:
                notes = "summary_skipped:no_pdf_mode"
                pdf_state = {
                    "downloaded": False,
                    "pdf_path": None,
                    "text_path": None,
                    "extract_meta": {"error": "no_pdf_mode"},
                }
            elif not paper.get("links", {}).get("pdf"):
                notes = "summary_skipped:missing_pdf_link"
                pdf_state = {
                    "downloaded": False,
                    "pdf_path": None,
                    "text_path": None,
                    "extract_meta": {"error": "missing_pdf_link"},
                }
            else:
                pdf_path = month_out / "pdfs" / f"{aid}.pdf"
                txt_path = month_out / "text" / f"{aid}.txt"
                try:
                    download_pdf(paper["links"]["pdf"], pdf_path, profile.pipeline.pdf_rate_limit_seconds)
                    meta = extract_text(pdf_path, txt_path)
                    raw_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
                    pdf_state = {
                        "downloaded": True,
                        "pdf_path": str(pdf_path),
                        "text_path": str(txt_path),
                        "extract_meta": meta,
                    }
                    notes = json.dumps(meta, sort_keys=True)
                except Exception as exc:
                    notes = f"summary_skipped:pdf_failed:{type(exc).__name__}"
                    pdf_state = {
                        "downloaded": False,
                        "pdf_path": str(pdf_path),
                        "text_path": str(txt_path),
                        "extract_meta": {"error": f"download_or_extract_failed:{type(exc).__name__}"},
                    }

            if raw_text.strip():
                summary = summarize_paper(
                    paper=paper,
                    triage=triage_map[aid],
                    raw_fulltext=raw_text,
                    fulltext_slices=slice_paper_text(
                        raw_text,
                        excerpt_chars=18_000,
                        tail_chars=text_tail_chars,
                    ),
                    used_fulltext=True,
                    notes=notes,
                    llm=summary_llm,
                    prompt_template=summarize_prompt,
                    repair_template=repair_prompt,
                    schema=summary_schema,
                    max_input_tokens=summary_max_input_tokens,
                    allowed_tags=profile.summary.allowed_tags,
                )
                summaries.append(summary)
                summary_map[aid] = summary
                db.upsert_summary(month, summary)
                if summary_sleep_seconds > 0:
                    time.sleep(summary_sleep_seconds)
            pdf_map[aid] = pdf_state
    finally:
        summary_llm.close()

    summaries = sorted(summaries, key=lambda x: (x["published_date"], x["arxiv_id_base"]))
    write_jsonl(month_out / "papers.jsonl", summaries)

    backend_rows: list[dict[str, Any]] = []
    for paper in sorted(candidates, key=lambda x: (x["published"], x["arxiv_id_base"])):
        aid = paper["arxiv_id_base"]
        backend_rows.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "arxiv_id_base": aid,
                "version": paper["version"],
                "title": paper["title"],
                "summary": paper["summary"],
                "authors": paper["authors"],
                "categories": paper["categories"],
                "published": paper["published"],
                "updated": paper["updated"],
                "links": paper["links"],
                "triage": _triage_view(triage_map.get(aid)),
                "paper_summary": summary_map.get(aid),
                "pdf": pdf_map.get(aid, _empty_pdf_state()),
            }
        )
    write_jsonl(month_out / "backend_rows.jsonl", backend_rows)

    digest = build_digest(month, candidates, triage_rows, summaries)
    write_json(month_out / "digest.json", digest)

    if not no_site:
        docs_root = cfg.docs_dir / profile.id
        metadata_map = {candidate["arxiv_id_base"]: candidate for candidate in candidates}
        write_month_site(
            docs_root,
            month,
            summaries,
            metadata_map,
            digest,
            backend_rows=backend_rows,
            site_content=profile.site,
        )
        update_home(
            docs_root,
            site_content=profile.site,
            triage_prompt_path=profile.paths.triage_prompt,
            summary_prompt_path=profile.paths.summary_prompt,
        )
    db.upsert_run(month, digest["stats"])
    db.close()
