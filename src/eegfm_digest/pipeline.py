from __future__ import annotations

import json
from pathlib import Path

from .arxiv import fetch_month_candidates
from .config import Config
from .db import DigestDB
from .llm_gemini import GeminiClient, LLMConfig, load_api_key
from .pdf import bounded_text, download_pdf, extract_text
from .render import build_digest, write_json, write_jsonl
from .site import update_home, write_month_site
from .summarize import summarize_paper
from .triage import load_schema, triage_paper


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def run_month(cfg: Config, month: str, no_pdf: bool = False, force: bool = False) -> None:
    month_out = cfg.output_dir / month
    month_out.mkdir(parents=True, exist_ok=True)
    db = DigestDB(cfg.data_dir / "digest.sqlite")

    triage_schema = load_schema(Path("schemas/triage.json"))
    summary_schema = load_schema(Path("schemas/summary.json"))

    # Stage 1: fetch
    candidates = fetch_month_candidates(cfg.max_candidates, month, cfg.arxiv_rate_limit_seconds)
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
            result = cached or triage_paper(
                paper, triage_llm, triage_prompt, repair_prompt, triage_schema
            )
            triage_rows.append(result)
            db.upsert_triage(month, result)
        except Exception as exc:
            fallback = {
                "arxiv_id_base": paper["arxiv_id_base"],
                "is_eeg_related": False,
                "is_foundation_model_related": False,
                "borderline": False,
                "paper_type": "other",
                "confidence": 0.0,
                "reasons": [f"triage_exception:{type(exc).__name__}"],
                "suggested_digest_tags": [],
                "decision": "reject",
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
    for paper in accepted:
        try:
            cached_summary = None if force else db.get_summary(paper["arxiv_id_base"])
            if cached_summary:
                summaries.append(cached_summary)
                continue
            text_window = paper["summary"]
            used_fulltext = False
            notes = "abstract_only"
            if not no_pdf and paper.get("links", {}).get("pdf"):
                pdf_path = month_out / "pdfs" / f"{paper['arxiv_id_base']}.pdf"
                txt_path = month_out / "text" / f"{paper['arxiv_id_base']}.txt"
                try:
                    download_pdf(paper["links"]["pdf"], pdf_path, cfg.pdf_rate_limit_seconds)
                    meta = extract_text(pdf_path, txt_path)
                    raw_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
                    if raw_text.strip():
                        text_window = bounded_text(raw_text, cfg.text_head_chars, cfg.text_tail_chars)
                        used_fulltext = True
                        notes = json.dumps(meta, sort_keys=True)
                    else:
                        notes = f"pdf_empty_text:{meta}"
                except Exception as exc:
                    notes = f"pdf_failed:{type(exc).__name__}"
            summary = summarize_paper(
                paper,
                triage_map[paper["arxiv_id_base"]],
                text_window,
                used_fulltext,
                notes,
                summary_llm,
                summarize_prompt,
                repair_prompt,
                summary_schema,
            )
            summaries.append(summary)
            db.upsert_summary(month, summary)
        except Exception:
            continue

    summaries = sorted(summaries, key=lambda x: (x["published_date"], x["arxiv_id_base"]))
    write_jsonl(month_out / "papers.jsonl", summaries)

    # Stage 4: digest + site
    digest = build_digest(month, candidates, triage_rows, summaries)
    write_json(month_out / "digest.json", digest)
    metadata_map = {c["arxiv_id_base"]: c for c in candidates}
    write_month_site(cfg.docs_dir, month, summaries, metadata_map, digest)
    update_home(cfg.docs_dir)
    db.upsert_run(month, digest["stats"])
    db.close()
