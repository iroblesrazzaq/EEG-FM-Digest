from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from .arxiv import fetch_month_candidates
from .config import Config, load_config
from .db import DigestDB
from .llm_gemini import GeminiClient, LLMConfig, load_api_key
from .pdf import download_pdf, extract_text, slice_paper_text
from .render import build_digest, write_json, write_jsonl
from .site import update_home, write_month_site
from .summarize import summarize_paper
from .triage import load_schema, triage_paper


class RateLimitStop(BaseException):
    """Raised to stop batch execution when provider quota/rate limits are hit."""


@dataclass(frozen=True)
class BatchRunConfig:
    months: list[str]
    months_from_outputs: bool = True
    no_site: bool = False
    triage_force: bool = False
    summary_force: bool = False
    include_borderline: bool = False
    triage_provider: str = "gemini"
    triage_model: str = ""
    summary_provider: str = "gemini"
    summary_model: str = ""
    triage_sleep_seconds: float = 0.0
    summary_sleep_seconds: float = 0.0
    stop_on_rate_limit: bool = True
    sync_cache_from_outputs: bool = True
    max_candidates: int | None = None
    max_accepted: int | None = None
    env_path: str = "~/2_cs_projects/env/.env"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _discover_months_from_outputs(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    months = []
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "arxiv_raw.json").exists():
            months.append(entry.name)
    return sorted(months)


def _parse_batch_config(path: Path) -> BatchRunConfig:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid batch config at {path}: expected object")

    months = raw.get("months", [])
    if isinstance(months, str):
        months = [months]
    if not isinstance(months, list):
        raise RuntimeError("`months` must be a list of YYYY-MM strings")
    months = [str(m) for m in months if str(m).strip()]

    return BatchRunConfig(
        months=months,
        months_from_outputs=bool(raw.get("months_from_outputs", True)),
        no_site=bool(raw.get("no_site", False)),
        triage_force=bool(raw.get("triage_force", False)),
        summary_force=bool(raw.get("summary_force", False)),
        include_borderline=bool(raw.get("include_borderline", False)),
        triage_provider=str(raw.get("triage_provider", raw.get("summary_provider", "gemini"))),
        triage_model=str(raw.get("triage_model", raw.get("summary_model", ""))),
        summary_provider=str(raw.get("summary_provider", "gemini")),
        summary_model=str(raw.get("summary_model", "")),
        triage_sleep_seconds=float(raw.get("triage_sleep_seconds", 0.0)),
        summary_sleep_seconds=float(raw.get("summary_sleep_seconds", 0.0)),
        stop_on_rate_limit=bool(raw.get("stop_on_rate_limit", True)),
        sync_cache_from_outputs=bool(raw.get("sync_cache_from_outputs", True)),
        max_candidates=int(raw["max_candidates"]) if raw.get("max_candidates") is not None else None,
        max_accepted=int(raw["max_accepted"]) if raw.get("max_accepted") is not None else None,
        env_path=str(raw.get("env_path", "~/2_cs_projects/env/.env")),
    )


def _effective_months(cfg: BatchRunConfig, base_cfg: Config) -> list[str]:
    months = list(cfg.months)
    if cfg.months_from_outputs:
        for m in _discover_months_from_outputs(base_cfg.output_dir):
            if m not in months:
                months.append(m)
    months = sorted(set(months))
    if not months:
        raise RuntimeError(
            "No months found. Provide `months` in config or enable `months_from_outputs` with existing outputs/*/arxiv_raw.json."
        )
    return months


def _triage_view(triage: dict[str, Any] | None) -> dict[str, Any]:
    triage = triage or {}
    reasons = triage.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "decision": triage.get("decision", "reject"),
        "confidence": float(triage.get("confidence", 0.0)),
        "reasons": reasons,
    }


def _empty_pdf_state() -> dict[str, Any]:
    return {
        "downloaded": False,
        "pdf_path": None,
        "text_path": None,
        "extract_meta": None,
    }


def _bootstrap_cache_from_outputs(db: DigestDB, month: str, month_out: Path) -> None:
    for row in _load_jsonl(month_out / "triage.jsonl"):
        db.upsert_triage(month, row)
    for row in _load_jsonl(month_out / "papers.jsonl"):
        db.upsert_summary(month, row)
    raw_path = month_out / "arxiv_raw.json"
    if raw_path.exists():
        for row in _load_json(raw_path):
            db.upsert_paper(month, row)


def _triage_client_error_ids(month_out: Path) -> set[str]:
    ids: set[str] = set()
    for row in _load_jsonl(month_out / "triage.jsonl"):
        reasons = row.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        if any("triage_exception:ClientError" in str(reason) for reason in reasons):
            aid = row.get("arxiv_id_base")
            if isinstance(aid, str) and aid:
                ids.add(aid)
    return ids


class OpenRouterClient:
    def __init__(self, api_key: str, model: str, temperature: float, max_output_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = httpx.Client(timeout=180)

    def close(self) -> None:
        self._client.close()

    def count_tokens(self, content: str) -> int:
        # Approximate token count for payload-routing heuristics.
        return max(1, len(content) // 4)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return ""

    def generate(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "reasoning": {"enabled": True},
        }
        if schema is not None:
            body["response_format"] = {"type": "json_object"}

        resp = self._client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        if resp.status_code in {402, 429}:
            raise RateLimitStop(
                f"openrouter_rate_limit_or_quota status={resp.status_code} body={resp.text[:220]}"
            )
        resp.raise_for_status()
        text = self._extract_text(resp.json())
        if not text:
            raise RuntimeError("OpenRouter returned empty content")
        return text


def _normalize_triage_row(arxiv_id_base: str, result: dict[str, Any]) -> dict[str, Any]:
    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return {
        "arxiv_id_base": arxiv_id_base,
        "decision": result.get("decision", "reject"),
        "confidence": float(result.get("confidence", 0.0)),
        "reasons": reasons,
    }


def _run_triage_phase_for_month(
    cfg: Config,
    run_cfg: BatchRunConfig,
    month: str,
    db: DigestDB,
    llm: Any,
) -> None:
    month_out = cfg.output_dir / month
    month_out.mkdir(parents=True, exist_ok=True)
    raw_path = month_out / "arxiv_raw.json"
    if raw_path.exists() and not run_cfg.triage_force:
        candidates = _load_json(raw_path)
    else:
        candidates = fetch_month_candidates(
            cfg.max_candidates,
            month,
            cfg.arxiv_rate_limit_seconds,
            connect_timeout_seconds=cfg.arxiv_connect_timeout_seconds,
            read_timeout_seconds=cfg.arxiv_read_timeout_seconds,
            retries=cfg.arxiv_retries,
            retry_backoff_seconds=cfg.arxiv_retry_backoff_seconds,
        )
        write_json(raw_path, candidates)
    for row in candidates:
        db.upsert_paper(month, row)

    triage_schema = load_schema(Path("schemas/triage.json"))
    triage_prompt = Path("prompts/triage.md").read_text(encoding="utf-8")
    repair_prompt = Path("prompts/repair_json.md").read_text(encoding="utf-8")

    triage_rows: list[dict[str, Any]] = []
    for paper in candidates:
        aid = paper["arxiv_id_base"]
        if run_cfg.triage_force:
            cached = None
        else:
            cached = db.get_triage(aid)
        if cached:
            triage_rows.append(_normalize_triage_row(aid, cached))
            continue
        try:
            result = triage_paper(
                paper=paper,
                llm=llm,
                prompt_template=triage_prompt,
                repair_template=repair_prompt,
                schema=triage_schema,
            )
            row = _normalize_triage_row(aid, result)
        except RateLimitStop:
            raise
        except Exception as exc:
            row = {
                "arxiv_id_base": aid,
                "decision": "reject",
                "confidence": 0.0,
                "reasons": [f"triage_exception:{type(exc).__name__}", "automatic_reject_fallback"],
            }
        triage_rows.append(row)
        db.upsert_triage(month, row)
        if run_cfg.triage_sleep_seconds > 0:
            time.sleep(run_cfg.triage_sleep_seconds)

    triage_rows = sorted(triage_rows, key=lambda x: x["arxiv_id_base"])
    write_jsonl(month_out / "triage.jsonl", triage_rows)
    print(f"[triage] {month}: done candidates={len(candidates)} triage_rows={len(triage_rows)}")


def _run_summary_phase_for_month(
    cfg: Config,
    run_cfg: BatchRunConfig,
    month: str,
    db: DigestDB,
    llm: Any,
) -> None:
    month_out = cfg.output_dir / month
    raw_path = month_out / "arxiv_raw.json"
    triage_path = month_out / "triage.jsonl"
    if not raw_path.exists() or not triage_path.exists():
        print(f"[summary] {month}: skipped (missing arxiv_raw.json or triage.jsonl)")
        return

    candidates = _load_json(raw_path)
    triage_rows = _load_jsonl(triage_path)
    triage_map = {row["arxiv_id_base"]: row for row in triage_rows}

    accepted = [p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "accept"]
    if run_cfg.include_borderline:
        accepted.extend(
            p for p in candidates if triage_map.get(p["arxiv_id_base"], {}).get("decision") == "borderline"
        )
    accepted = sorted(accepted, key=lambda x: (x["published"], x["arxiv_id_base"]))[: cfg.max_accepted]

    summary_schema = _load_json(Path("schemas/summary.json"))
    summarize_prompt = Path("prompts/summarize.md").read_text(encoding="utf-8")
    repair_prompt = Path("prompts/repair_json.md").read_text(encoding="utf-8")

    existing_summaries = [] if run_cfg.summary_force else _load_jsonl(month_out / "papers.jsonl")
    summary_map: dict[str, dict[str, Any]] = {s["arxiv_id_base"]: s for s in existing_summaries}

    existing_backend = _load_jsonl(month_out / "backend_rows.jsonl")
    pdf_map: dict[str, dict[str, Any]] = {
        row.get("arxiv_id_base", ""): row.get("pdf") or _empty_pdf_state() for row in existing_backend
    }

    print(f"[summary] {month}: accepted={len(accepted)} cached={len(summary_map)}")

    for paper in accepted:
        aid = paper["arxiv_id_base"]
        if aid in summary_map and not run_cfg.summary_force:
            continue

        pdf_state = _empty_pdf_state()
        raw_text = ""
        notes = "summary_not_attempted"

        if not paper.get("links", {}).get("pdf"):
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
                triage=triage_map.get(aid, {}),
                raw_fulltext=raw_text,
                fulltext_slices=slice_paper_text(
                    raw_text,
                    excerpt_chars=18_000,
                    tail_chars=cfg.text_tail_chars,
                ),
                used_fulltext=True,
                notes=notes,
                llm=llm,
                prompt_template=summarize_prompt,
                repair_template=repair_prompt,
                schema=summary_schema,
                max_input_tokens=cfg.summary_max_input_tokens,
            )
            summary_map[aid] = summary
            db.upsert_summary(month, summary)
            print(f"[summary] {month}: summarized {aid}")
            if run_cfg.summary_sleep_seconds > 0:
                time.sleep(run_cfg.summary_sleep_seconds)

        pdf_map[aid] = pdf_state

    summaries = sorted(summary_map.values(), key=lambda x: (x["published_date"], x["arxiv_id_base"]))
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

    if not run_cfg.no_site:
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
        local_dir = cfg.docs_dir / "local" / month
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg.docs_dir / "digest" / month / "index.html", local_dir / "index.html")

    db.upsert_run(month, digest["stats"])
    print(
        f"[summary] {month}: done summarized={len(summaries)} accepted={digest['stats']['accepted']} candidates={digest['stats']['candidates']}"
    )


def run_batch(config_path: Path) -> None:
    run_cfg = _parse_batch_config(config_path)
    cfg = load_config()
    if run_cfg.max_candidates is not None:
        cfg = replace(cfg, max_candidates=run_cfg.max_candidates)
    if run_cfg.max_accepted is not None:
        cfg = replace(cfg, max_accepted=run_cfg.max_accepted)
    if run_cfg.include_borderline:
        cfg = replace(cfg, include_borderline=True)

    months = _effective_months(run_cfg, cfg)
    print(f"[batch] months={months}")

    # Load API keys for providers from configured env file.
    load_dotenv(Path(run_cfg.env_path).expanduser())
    triage_provider = run_cfg.triage_provider.strip().lower()
    summary_provider = run_cfg.summary_provider.strip().lower()
    if triage_provider not in {"gemini", "openrouter"}:
        raise RuntimeError(
            f"Unsupported triage_provider={run_cfg.triage_provider}. Use 'gemini' or 'openrouter'."
        )
    if summary_provider not in {"gemini", "openrouter"}:
        raise RuntimeError(
            f"Unsupported summary_provider={run_cfg.summary_provider}. Use 'gemini' or 'openrouter'."
        )
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if (triage_provider == "openrouter" or summary_provider == "openrouter") and not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment or env file.")
    gemini_key: str | None = None
    if triage_provider == "gemini" or summary_provider == "gemini":
        gemini_key = load_api_key()

    db = DigestDB(cfg.data_dir / "digest.sqlite")
    try:
        if triage_provider == "gemini":
            triage_llm: Any = GeminiClient(
                LLMConfig(
                    api_key=gemini_key or load_api_key(),
                    model=run_cfg.triage_model or cfg.gemini_model_triage,
                    temperature=cfg.llm_temperature_triage,
                    max_output_tokens=cfg.llm_max_output_tokens_triage,
                )
            )
            triage_close = lambda: None
        else:
            triage_llm = OpenRouterClient(
                api_key=openrouter_key or "",
                model=run_cfg.triage_model or "arcee-ai/trinity-large-preview:free",
                temperature=cfg.llm_temperature_triage,
                max_output_tokens=cfg.llm_max_output_tokens_triage,
            )
            triage_close = triage_llm.close

        # Phase 1: triage all months first.
        try:
            for month in months:
                month_out = cfg.output_dir / month
                month_out.mkdir(parents=True, exist_ok=True)
                if run_cfg.sync_cache_from_outputs:
                    _bootstrap_cache_from_outputs(db, month, month_out)
                print(f"[triage] {month}: start")
                _run_triage_phase_for_month(cfg, run_cfg, month, db, triage_llm)
        finally:
            triage_close()

        # Phase 2: summarize accepted for all months.
        if summary_provider == "gemini":
            summary_llm: Any = GeminiClient(
                LLMConfig(
                    api_key=gemini_key or load_api_key(),
                    model=run_cfg.summary_model or cfg.gemini_model_summary,
                    temperature=cfg.llm_temperature_summary,
                    max_output_tokens=cfg.llm_max_output_tokens_summary,
                )
            )
            for month in months:
                _run_summary_phase_for_month(cfg, run_cfg, month, db, summary_llm)
        else:
            summary_llm = OpenRouterClient(
                api_key=openrouter_key or "",
                model=run_cfg.summary_model or "arcee-ai/trinity-large-preview:free",
                temperature=cfg.llm_temperature_summary,
                max_output_tokens=cfg.llm_max_output_tokens_summary,
            )
            try:
                for month in months:
                    _run_summary_phase_for_month(cfg, run_cfg, month, db, summary_llm)
            finally:
                summary_llm.close()
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner: triage all configured months, then summarize accepted papers."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file (see configs/batch_all_months.json or configs/batch_single_month.json).",
    )
    args = parser.parse_args()
    run_batch(Path(args.config))


if __name__ == "__main__":
    main()
