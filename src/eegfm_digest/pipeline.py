from __future__ import annotations

import json
from pathlib import Path

from .arxiv import fetch_month_candidates
from .config import Config
from .db import DigestDB
from .llm_gemini import GeminiClient, LLMConfig, load_api_key
from .pdf import download_pdf, extract_text, slice_paper_text
from .render import build_digest, write_json, write_jsonl
from .site import update_home, write_month_site
from .summarize import summarize_paper
from .triage import load_schema, triage_paper


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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


def run_month(
    cfg: Config,
    month: str,
    no_pdf: bool = False,
    no_site: bool = False,
    force: bool = False,
) -> None:
    month_out = cfg.output_dir / month
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
    )
    write_json(month_out / "arxiv_raw.json", candidates)
    for c in candidates:
        db.upsert_paper(month, c)

    triage_llm = GeminiClient(
        LLMConfig(
            api_key=load_api_key(),
            model=cfg.gemini_model_triage,
            temperature=cfg.llm_temperature_triage,
            max_output_tokens=cfg.llm_max_output_tokens_triage,
        )
    )

    triage_prompt = _read("prompts/triage.md")
    summarize_prompt = _read("prompts/summarize.md")
    repair_prompt = _read("prompts/repair_json.md")

    # Stage 2: triage
    triage_rows: list[dict] = []
    for paper in candidates:
        try:
            cached = None if force else db.get_triage(paper["arxiv_id_base"])
            result_raw = cached or triage_paper(
                paper, triage_llm, triage_prompt, repair_prompt, triage_schema
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
            db.upsert_triage(month, result)
        except Exception as exc:
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
            db.upsert_triage(month, fallback)

    write_jsonl(month_out / "triage.jsonl", sorted(triage_rows, key=lambda x: x["arxiv_id_base"]))

    # Stage 3: summarize
    summary_llm = GeminiClient(
        LLMConfig(
            api_key=load_api_key(),
            model=cfg.gemini_model_summary,
            temperature=cfg.llm_temperature_summary,
            max_output_tokens=cfg.llm_max_output_tokens_summary,
        )
    )
    triage_map = {t["arxiv_id_base"]: t for t in triage_rows}
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
        pdf_state: dict[str, object | None] = _empty_pdf_state()
        try:
            cached_summary = None if force else db.get_summary(arxiv_id_base)
            if cached_summary:
                summaries.append(cached_summary)
                summary_map[arxiv_id_base] = cached_summary
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
                summary = summarize_paper(
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
                db.upsert_summary(month, summary)
        except Exception:
            pass
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

    # Stage 4: digest + site
    digest = build_digest(month, candidates, triage_rows, summaries)
    write_json(month_out / "digest.json", digest)
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
    db.upsert_run(month, digest["stats"])
    db.close()
