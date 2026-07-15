from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .arxiv import fetch_month_candidates, fetch_window_candidates
from .backend_rows import build_backend_rows
from .cache_meta import build_stage_metadata, is_cache_current
from .config import Config
from .db import DigestDB
from .llm import LLMCallConfig, LLMRateLimitError, build_llm_call, load_api_key, provider_base_url
from .llm_logging import log_stage_failure
from .pdf import download_pdf, extract_text, slice_paper_text
from .render import build_digest, write_json, write_jsonl
from .row_views import empty_pdf_state, normalize_triage_row
from .selection import select_papers_for_summary
from .site import update_home, write_month_site
from .stage_context import load_summary_stage_context, load_triage_stage_context
from .summarize import summarize_paper, summarize_paper_with_meta
from .summarize_stage import prepare_pdf_and_text
from .triage import triage_paper, triage_paper_with_meta


@dataclass(frozen=True)
class MonthRunStats:
    """Summary returned by :func:`run_month` for downstream callers."""

    month: str
    candidates: int
    accepted: int
    summarized: int
    triage_failures: int = 0
    summary_failures: int = 0
    failed_triage_ids: tuple[str, ...] = ()
    failed_summary_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowRunStats:
    """Aggregate summary returned by :func:`run_window`."""

    since: datetime
    until: datetime
    window_candidates: int
    affected_months: tuple[str, ...]
    per_month: tuple[MonthRunStats, ...]

    @property
    def total_accepted(self) -> int:
        return sum(m.accepted for m in self.per_month)

    @property
    def total_triage_failures(self) -> int:
        return sum(m.triage_failures for m in self.per_month)

    @property
    def total_summary_failures(self) -> int:
        return sum(m.summary_failures for m in self.per_month)

    @property
    def failed_triage_ids(self) -> tuple[str, ...]:
        return tuple(
            aid for month in self.per_month for aid in month.failed_triage_ids
        )

    @property
    def failed_summary_ids(self) -> tuple[str, ...]:
        return tuple(
            aid for month in self.per_month for aid in month.failed_summary_ids
        )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_ORIGINAL_TRIAGE_PAPER = triage_paper
_ORIGINAL_SUMMARIZE_PAPER = summarize_paper


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
) -> MonthRunStats:
    month_out = cfg.output_dir / month
    month_out.mkdir(parents=True, exist_ok=True)
    db = DigestDB(cfg.data_dir / "digest.sqlite")

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
        triage_ctx = load_triage_stage_context(triage_llm_config)
        summary_ctx = load_summary_stage_context(summary_llm_config)

        # Stage 2: triage
        triage_rows: list[dict] = []
        triage_failure_count = 0
        failed_triage_ids: list[str] = []
        for paper in candidates:
            arxiv_id_base = paper["arxiv_id_base"]
            try:
                cached = None if force else db.get_triage_with_meta(arxiv_id_base)
                if cached and is_cache_current(cached.get("meta"), triage_ctx.descriptor["cache_version"]):
                    result_raw = cached["data"]
                else:
                    result_raw, triage_call_meta = _run_triage_call_with_meta(
                        paper,
                        triage_llm,
                        triage_ctx.triage_prompt,
                        triage_ctx.repair_prompt,
                        triage_ctx.schema,
                    )
                    db.upsert_triage(
                        month,
                        result_raw,
                        meta=build_stage_metadata(
                            triage_ctx.descriptor,
                            repair_used=bool(triage_call_meta.get("repair_used", False)),
                            updated_at_source=str(paper.get("updated", "")).strip() or None,
                        ),
                    )
                triage_rows.append(normalize_triage_row(arxiv_id_base, result_raw))
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    raise
                triage_failure_count += 1
                failed_triage_ids.append(arxiv_id_base)
                log_stage_failure(
                    "pipeline.triage",
                    arxiv_id_base=arxiv_id_base,
                    provider=triage_llm_config.provider,
                    model=triage_llm_config.model,
                    exc=exc,
                )
                print(
                    f"[pipeline] WARNING: triage failed for {arxiv_id_base}: "
                    f"{type(exc).__name__}: {exc}; skipping (will retry next run)",
                    file=sys.stderr,
                )

        write_jsonl(month_out / "triage.jsonl", sorted(triage_rows, key=lambda x: x["arxiv_id_base"]))

        # Stage 3: summarize
        triage_map = {t["arxiv_id_base"]: t for t in triage_rows}
        # Summaries are preserved across triage flips: site rendering already
        # filters by current triage decision, so a previously-accepted paper
        # that now triages as reject is hidden but its summary work is kept.

        accepted = select_papers_for_summary(
            candidates,
            triage_map,
            include_borderline=cfg.include_borderline,
            max_borderline_pdfs=cfg.max_borderline_pdfs,
            max_accepted=cfg.max_accepted,
            borderline_policy="pipeline",
        )

        summaries: list[dict] = []
        summary_map: dict[str, dict] = {}
        pdf_map: dict[str, dict[str, object | None]] = {}
        summary_failure_count = 0
        failed_summary_ids: list[str] = []
        for paper in accepted:
            arxiv_id_base = paper["arxiv_id_base"]
            pdf_state: dict[str, object | None] = empty_pdf_state()
            try:
                cached_summary = None if force else db.get_summary_with_meta(arxiv_id_base)
                if cached_summary and is_cache_current(
                    cached_summary.get("meta"), summary_ctx.descriptor["cache_version"]
                ):
                    summaries.append(cached_summary["data"])
                    summary_map[arxiv_id_base] = cached_summary["data"]
                    pdf_map[arxiv_id_base] = pdf_state
                    continue

                pdf_result = prepare_pdf_and_text(paper, month_out, cfg, no_pdf=no_pdf)
                pdf_state = pdf_result.pdf_state
                if pdf_result.notes == "summary_skipped:missing_pdf_link":
                    summary_failure_count += 1
                    failed_summary_ids.append(arxiv_id_base)
                    print(
                        f"[pipeline] WARNING: pdf missing for {arxiv_id_base}; "
                        "skipping (will retry next run)",
                        file=sys.stderr,
                    )
                elif pdf_result.notes.startswith("summary_skipped:pdf_failed:"):
                    summary_failure_count += 1
                    failed_summary_ids.append(arxiv_id_base)
                    print(
                        f"[pipeline] WARNING: pdf download/extract failed for "
                        f"{arxiv_id_base}; skipping (will retry next run)",
                        file=sys.stderr,
                    )
                elif pdf_result.raw_text.strip():
                    summary, summary_call_meta = _run_summary_call_with_meta(
                        paper=paper,
                        triage=triage_map[arxiv_id_base],
                        raw_fulltext=pdf_result.raw_text,
                        fulltext_slices=slice_paper_text(
                            pdf_result.raw_text,
                            excerpt_chars=18_000,
                            tail_chars=cfg.text_tail_chars,
                        ),
                        used_fulltext=True,
                        notes=pdf_result.notes,
                        llm=summary_llm,
                        prompt_template=summary_ctx.summarize_prompt,
                        repair_template=summary_ctx.repair_prompt,
                        schema=summary_ctx.schema,
                        max_input_tokens=cfg.summary_max_input_tokens,
                    )
                    summaries.append(summary)
                    summary_map[arxiv_id_base] = summary
                    db.upsert_summary(
                        month,
                        summary,
                        meta=build_stage_metadata(
                            summary_ctx.descriptor,
                            repair_used=bool(summary_call_meta.get("repair_used", False)),
                            updated_at_source=str(paper.get("updated", "")).strip() or None,
                        ),
                    )
            except Exception as exc:
                if isinstance(exc, LLMRateLimitError):
                    raise
                summary_failure_count += 1
                failed_summary_ids.append(arxiv_id_base)
                log_stage_failure(
                    "pipeline.summary",
                    arxiv_id_base=arxiv_id_base,
                    provider=summary_llm_config.provider,
                    model=summary_llm_config.model,
                    exc=exc,
                )
                print(
                    f"[pipeline] WARNING: summary failed for {arxiv_id_base}: "
                    f"{type(exc).__name__}: {exc}; skipping (will retry next run)",
                    file=sys.stderr,
                )
            pdf_map[arxiv_id_base] = pdf_state

        summaries = sorted(summaries, key=lambda x: (x["published_date"], x["arxiv_id_base"]))
        write_jsonl(month_out / "papers.jsonl", summaries)

        backend_rows = build_backend_rows(candidates, triage_map, summary_map, pdf_map)
        write_jsonl(month_out / "backend_rows.jsonl", backend_rows)

        digest = build_digest(month, candidates, triage_rows, summaries, featured_paper=feature_paper)
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
        return MonthRunStats(
            month=month,
            candidates=len(candidates),
            accepted=sum(1 for t in triage_rows if t.get("decision") == "accept"),
            summarized=len(summaries),
            triage_failures=triage_failure_count,
            summary_failures=summary_failure_count,
            failed_triage_ids=tuple(failed_triage_ids),
            failed_summary_ids=tuple(failed_summary_ids),
        )
    finally:
        triage_llm.close()
        summary_llm.close()
        db.close()


def run_window(
    cfg: Config,
    since: datetime,
    until: datetime,
    no_pdf: bool = False,
    no_site: bool = False,
    force: bool = False,
) -> WindowRunStats:
    """Run the pipeline for all arXiv papers submitted in ``[since, until)``.

    Discovery is scoped to the window; rendering still happens at the
    per-month level because the static site is organized by month.  For
    each month that contains newly discovered papers, :func:`run_month`
    is invoked to refresh the full month view.  Previously triaged papers
    are cache hits, so the LLM cost scales with *new* papers, not with
    the size of each affected month.

    Raises:
        ArxivFetchError: arXiv API failed after retries.
        LLMRateLimitError: LLM provider quota exhausted.

    Neither exception is caught here — daily-mode callers rely on them
    to short-circuit advancement of ``last_successful_run.json``.
    """
    if until <= since:
        raise ValueError(f"until ({until!r}) must be strictly greater than since ({since!r})")

    window_candidates = fetch_window_candidates(
        since,
        until,
        max_candidates=cfg.max_candidates,
        rate_limit_seconds=cfg.arxiv_rate_limit_seconds,
        connect_timeout_seconds=cfg.arxiv_connect_timeout_seconds,
        read_timeout_seconds=cfg.arxiv_read_timeout_seconds,
        retries=cfg.arxiv_retries,
        retry_backoff_seconds=cfg.arxiv_retry_backoff_seconds,
    )

    affected_months: list[str] = sorted(
        {str(p.get("published", ""))[:7] for p in window_candidates if p.get("published")}
    )

    per_month: list[MonthRunStats] = []
    for month in affected_months:
        stats = run_month(
            cfg,
            month,
            no_pdf=no_pdf,
            no_site=no_site,
            force=force,
        )
        per_month.append(stats)

    return WindowRunStats(
        since=since.astimezone(timezone.utc),
        until=until.astimezone(timezone.utc),
        window_candidates=len(window_candidates),
        affected_months=tuple(affected_months),
        per_month=tuple(per_month),
    )
