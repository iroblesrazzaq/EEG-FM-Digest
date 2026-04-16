from __future__ import annotations

import json
import shutil
from pathlib import Path

from .arxiv import fetch_month_candidates
from .cache_meta import (
    SUMMARY_STAGE_LOGIC_VERSION,
    TRIAGE_STAGE_LOGIC_VERSION,
    build_stage_descriptor,
    build_stage_metadata,
    is_cache_current,
)
from .config import DEFAULT_TOPIC, Config, load_topic
from .db import DigestDB
from .llm import LLMCallConfig, LLMRateLimitError, build_llm_call, load_api_key, provider_base_url
from .pdf import download_pdf, extract_text, slice_paper_text
from .render import build_digest, write_json, write_jsonl
from .site import update_home, write_month_site
from .summarize import summarize_paper, summarize_paper_with_meta
from .triage import load_schema, triage_paper, triage_paper_with_meta

_ORIGINAL_TRIAGE_PAPER = triage_paper
_ORIGINAL_SUMMARIZE_PAPER = summarize_paper


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _empty_pdf_state() -> dict[str, object | None]:
    return {
        "downloaded": False,
        "pdf_path": None,
        "text_path": None,
        "extract_meta": None,
    }


def _topic_cache_key(topic_slug: str, arxiv_id_base: str) -> str:
    if topic_slug == DEFAULT_TOPIC:
        return arxiv_id_base
    return f"{topic_slug}:{arxiv_id_base}"


def _topic_run_key(topic_slug: str, month: str) -> str:
    if topic_slug == DEFAULT_TOPIC:
        return month
    return f"{topic_slug}:{month}"


def _month_output_dir(base_output_dir: Path, topic_slug: str, month: str) -> Path:
    return base_output_dir / topic_slug / month


def _sync_legacy_output_dir(source_dir: Path, legacy_dir: Path) -> None:
    if legacy_dir.is_symlink():
        try:
            if legacy_dir.resolve() == source_dir.resolve():
                return
        except FileNotFoundError:
            pass
        legacy_dir.unlink()

    if not legacy_dir.exists():
        legacy_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
            return
        except OSError:
            legacy_dir.mkdir(parents=True, exist_ok=True)

    legacy_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.rglob("*"):
        target_path = legacy_dir / source_path.relative_to(source_dir)
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


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


def _run_triage_call_with_meta(*args, **kwargs) -> tuple[dict[str, object], dict[str, object]]:
    if triage_paper is not _ORIGINAL_TRIAGE_PAPER:
        return triage_paper(*args, **kwargs), {"repair_used": False}
    return triage_paper_with_meta(*args, **kwargs)


def _run_summary_call_with_meta(*args, **kwargs) -> tuple[dict[str, object], dict[str, object]]:
    if summarize_paper is not _ORIGINAL_SUMMARIZE_PAPER:
        return summarize_paper(*args, **kwargs), {"repair_used": False}
    return summarize_paper_with_meta(*args, **kwargs)


def run_month(
    cfg: Config,
    month: str,
    no_pdf: bool = False,
    no_site: bool = False,
    force: bool = False,
    feature_paper: str | None = None,
) -> None:
    topic_config = load_topic(topic)
    month_out = _month_output_dir(cfg.output_dir, topic_config.slug, month)
    month_out.mkdir(parents=True, exist_ok=True)
    db = DigestDB(cfg.data_dir / "digest.sqlite")

    triage_schema = load_schema(Path("schemas/triage.json"))
    summary_schema = load_schema(Path("schemas/summary.json"))

    # Stage 1: fetch
    candidates = fetch_month_candidates(
        cfg.max_candidates,
        month,
        cfg.arxiv_rate_limit_seconds,
        connect_timeout_seconds=cfg.arxiv_connect_timeout_seconds,
        read_timeout_seconds=cfg.arxiv_read_timeout_seconds,
        retries=cfg.arxiv_retries,
        retry_backoff_seconds=cfg.arxiv_retry_backoff_seconds,
        topic=topic_config,
    )
    write_json(month_out / "arxiv_raw.json", candidates)
    for c in candidates:
        db.upsert_paper(month, c, cache_key=_topic_cache_key(topic_config.slug, c["arxiv_id_base"]))

    api_key = load_api_key(cfg.llm_provider)
    triage_llm_config = LLMCallConfig(
        provider=cfg.llm_provider,
        api_key=api_key,
        model=cfg.llm_model_triage,
        temperature=cfg.llm_temperature_triage,
        max_output_tokens=cfg.llm_max_output_tokens_triage,
        base_url=provider_base_url(cfg.llm_provider),
    )
    summary_llm_config = LLMCallConfig(
        provider=cfg.llm_provider,
        api_key=api_key,
        model=cfg.llm_model_summary,
        temperature=cfg.llm_temperature_summary,
        max_output_tokens=cfg.llm_max_output_tokens_summary,
        base_url=provider_base_url(cfg.llm_provider),
    )
    triage_llm = build_llm_call(triage_llm_config)
    summary_llm = build_llm_call(summary_llm_config)

    try:
        triage_prompt = _read(str(topic_config.triage_prompt_path))
        summarize_prompt = _read(str(topic_config.summarize_prompt_path))
        repair_prompt = _read("prompts/repair_json.md")
        triage_descriptor = build_stage_descriptor(
            stage="triage",
            provider=triage_llm_config.provider,
            model=triage_llm_config.model,
            prompt_template=triage_prompt,
            repair_template=repair_prompt,
            schema=triage_schema,
            stage_logic_version=TRIAGE_STAGE_LOGIC_VERSION,
        )
        summary_descriptor = build_stage_descriptor(
            stage="summary",
            provider=summary_llm_config.provider,
            model=summary_llm_config.model,
            prompt_template=summarize_prompt,
            repair_template=repair_prompt,
            schema=summary_schema,
            stage_logic_version=SUMMARY_STAGE_LOGIC_VERSION,
        )

        # Stage 2: triage
        triage_rows: list[dict] = []
        for paper in candidates:
            cache_key = _topic_cache_key(topic_config.slug, paper["arxiv_id_base"])
            try:
                cached = None if force else db.get_triage_with_meta(paper["arxiv_id_base"], cache_key=cache_key)
                if cached and is_cache_current(cached.get("meta"), triage_descriptor["cache_version"]):
                    result_raw = cached["data"]
                else:
                    result_raw, triage_call_meta = _run_triage_call_with_meta(
                        paper, triage_llm, triage_prompt, repair_prompt, triage_schema
                    )
                    db.upsert_triage(
                        month,
                        result_raw,
                        meta=build_stage_metadata(
                            triage_descriptor,
                            repair_used=bool(triage_call_meta.get("repair_used", False)),
                            updated_at_source=str(paper.get("updated", "")).strip() or None,
                        ),
                        cache_key=cache_key,
                    )
                reasons_raw = result_raw.get("reasons", [])
                if not isinstance(reasons_raw, list):
                    reasons_raw = [str(reasons_raw)]
                result = {
                    "arxiv_id_base": paper["arxiv_id_base"],
                    "decision": result_raw.get("decision", "reject"),
                    "confidence": float(result_raw.get("confidence", 0.0)),
                    "reasons": reasons_raw,
                }
                triage_rows.append(result)
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    raise
                fallback = {
                    "arxiv_id_base": paper["arxiv_id_base"],
                    "decision": "reject",
                    "confidence": 0.0,
                    "reasons": [
                        f"triage_exception:{type(exc).__name__}",
                        "automatic_reject_fallback",
                    ],
                }
                triage_rows.append(fallback)
                db.upsert_triage(
                    month,
                    fallback,
                    meta=build_stage_metadata(
                        triage_descriptor,
                        repair_used=False,
                        updated_at_source=str(paper.get("updated", "")).strip() or None,
                    ),
                    cache_key=cache_key,
                )

        write_jsonl(month_out / "triage.jsonl", sorted(triage_rows, key=lambda x: x["arxiv_id_base"]))

        # Stage 3: summarize
        triage_map = {t["arxiv_id_base"]: t for t in triage_rows}
        for arxiv_id_base, triage_row in triage_map.items():
            if triage_row.get("decision") == "reject":
                db.delete_summary(arxiv_id_base, cache_key=_topic_cache_key(topic_config.slug, arxiv_id_base))

        accepted = [p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "accept"]
        if cfg.include_borderline:
            borderline = [
                p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "borderline"
            ][: cfg.max_borderline_pdfs]
            accepted.extend(borderline)
        accepted = sorted(accepted, key=lambda x: (x["published"], x["arxiv_id_base"]))[: cfg.max_accepted]

        summaries: list[dict] = []
        summary_map: dict[str, dict] = {}
        pdf_map: dict[str, dict[str, object | None]] = {}
        for paper in accepted:
            arxiv_id_base = paper["arxiv_id_base"]
            cache_key = _topic_cache_key(topic_config.slug, arxiv_id_base)
            pdf_state: dict[str, object | None] = _empty_pdf_state()
            try:
                cached_summary = None if force else db.get_summary_with_meta(arxiv_id_base, cache_key=cache_key)
                if cached_summary and is_cache_current(cached_summary.get("meta"), summary_descriptor["cache_version"]):
                    summaries.append(cached_summary["data"])
                    summary_map[arxiv_id_base] = cached_summary["data"]
                    pdf_map[arxiv_id_base] = pdf_state
                    continue

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
                    pdf_path = month_out / "pdfs" / f"{arxiv_id_base}.pdf"
                    txt_path = month_out / "text" / f"{arxiv_id_base}.txt"
                    try:
                        download_pdf(paper["links"]["pdf"], pdf_path, cfg.pdf_rate_limit_seconds)
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
                    summary, summary_call_meta = _run_summary_call_with_meta(
                        paper=paper,
                        triage=triage_map[arxiv_id_base],
                        raw_fulltext=raw_text,
                        fulltext_slices=slice_paper_text(
                            raw_text,
                            excerpt_chars=18_000,
                            tail_chars=cfg.text_tail_chars,
                        ),
                        used_fulltext=True,
                        notes=notes,
                        llm=summary_llm,
                        prompt_template=summarize_prompt,
                        repair_template=repair_prompt,
                        schema=summary_schema,
                        max_input_tokens=cfg.summary_max_input_tokens,
                    )
                    summaries.append(summary)
                    summary_map[arxiv_id_base] = summary
                    db.upsert_summary(
                        month,
                        summary,
                        meta=build_stage_metadata(
                            summary_descriptor,
                            repair_used=bool(summary_call_meta.get("repair_used", False)),
                            updated_at_source=str(paper.get("updated", "")).strip() or None,
                        ),
                        cache_key=cache_key,
                    )
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    raise
            pdf_map[arxiv_id_base] = pdf_state

        summaries = sorted(summaries, key=lambda x: (x["published_date"], x["arxiv_id_base"]))
        write_jsonl(month_out / "papers.jsonl", summaries)

        backend_rows: list[dict] = []
        for paper in sorted(candidates, key=lambda x: (x["published"], x["arxiv_id_base"])):
            arxiv_id_base = paper["arxiv_id_base"]
            backend_rows.append(
                {
                    "arxiv_id": paper["arxiv_id"],
                    "arxiv_id_base": arxiv_id_base,
                    "version": paper["version"],
                    "title": paper["title"],
                    "summary": paper["summary"],
                    "authors": paper["authors"],
                    "categories": paper["categories"],
                    "published": paper["published"],
                    "updated": paper["updated"],
                    "links": paper["links"],
                    "triage": _triage_view(triage_map.get(arxiv_id_base)),
                    "paper_summary": summary_map.get(arxiv_id_base),
                    "pdf": pdf_map.get(arxiv_id_base, _empty_pdf_state()),
                }
            )
        write_jsonl(month_out / "backend_rows.jsonl", backend_rows)

        digest = build_digest(month, candidates, triage_rows, summaries, featured_paper=feature_paper)
        write_json(month_out / "digest.json", digest)
        if topic_config.slug == DEFAULT_TOPIC:
            _sync_legacy_output_dir(month_out, cfg.output_dir / month)
        if not no_site:
            metadata_map = {c["arxiv_id_base"]: c for c in candidates}
            write_month_site(
                cfg.docs_dir,
                month,
                summaries,
                metadata_map,
                digest,
                backend_rows=backend_rows,
            )
            update_home(cfg.docs_dir)
        db.upsert_run(month, digest["stats"], run_key=_topic_run_key(topic_config.slug, month))
    finally:
        triage_llm.close()
        summary_llm.close()
        db.close()
