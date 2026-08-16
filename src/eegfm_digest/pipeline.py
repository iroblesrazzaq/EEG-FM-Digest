from __future__ import annotations

import json
import sys
import time
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
from .pdf import download_pdf, extract_text
from .render import build_digest, write_json, write_jsonl
from .row_views import empty_pdf_state, normalize_triage_row
from .selection import select_papers_for_summary
from .site import update_home, write_month_site
from .stage_context import load_summary_stage_context, load_triage_stage_context
from .summarize import summarize_paper, summarize_paper_with_meta
from .summarize_stage import (
    prepare_pdf_and_text,
    summary_inputs_from_pdf_result,
    summary_used_fulltext,
)
from .triage import triage_paper, triage_paper_with_meta


def _maybe_sleep_after_summary_call(cfg: Config) -> None:
    if cfg.llm_call_sleep_seconds > 0:
        time.sleep(cfg.llm_call_sleep_seconds)


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
    straggler_attempted: int = 0
    straggler_succeeded: int = 0
    straggler_failures: int = 0
    failed_straggler_ids: tuple[str, ...] = ()

    @property
    def total_accepted(self) -> int:
        return sum(m.accepted for m in self.per_month)

    @property
    def total_triage_failures(self) -> int:
        return sum(m.triage_failures for m in self.per_month)

    @property
    def total_summary_failures(self) -> int:
        return sum(m.summary_failures for m in self.per_month) + self.straggler_failures

    @property
    def failed_triage_ids(self) -> tuple[str, ...]:
        return tuple(
            aid for month in self.per_month for aid in month.failed_triage_ids
        )

    @property
    def failed_summary_ids(self) -> tuple[str, ...]:
        return tuple(
            aid for month in self.per_month for aid in month.failed_summary_ids
        ) + self.failed_straggler_ids


@dataclass(frozen=True)
class StragglerStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    affected_months: tuple[str, ...] = ()
    failed_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OnePaperSummaryOutcome:
    arxiv_id_base: str
    summary: dict | None
    pdf_state: dict[str, object | None]
    failed: bool
    repair_used: bool = False


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


def _summarize_one_paper(
    *,
    paper: dict,
    triage: dict,
    month: str,
    month_out: Path,
    cfg: Config,
    db: DigestDB,
    summary_llm,
    summary_ctx,
    summary_llm_config: LLMCallConfig,
    no_pdf: bool,
    force: bool = False,
) -> OnePaperSummaryOutcome:
    """Download/extract/summarize one accepted paper (or use cache)."""
    arxiv_id_base = paper["arxiv_id_base"]
    pdf_state: dict[str, object | None] = empty_pdf_state()
    try:
        cached_summary = None if force else db.get_summary_with_meta(arxiv_id_base)
        cache_current = bool(
            cached_summary
            and is_cache_current(
                cached_summary.get("meta"), summary_ctx.descriptor["cache_version"]
            )
        )
        cached_data = cached_summary["data"] if cached_summary else None
        if cache_current and summary_used_fulltext(cached_data):
            return OnePaperSummaryOutcome(
                arxiv_id_base=arxiv_id_base,
                summary=cached_data,
                pdf_state=pdf_state,
                failed=False,
            )

        pdf_result = prepare_pdf_and_text(paper, month_out, cfg, no_pdf=no_pdf)
        pdf_state = pdf_result.pdf_state
        summary_inputs = summary_inputs_from_pdf_result(
            paper,
            pdf_result,
            head_chars=cfg.text_head_chars,
            excerpt_chars=cfg.summary_excerpt_chars,
            tail_chars=cfg.text_tail_chars,
        )
        if summary_inputs is None:
            print(
                f"[pipeline] WARNING: pdf skipped for {arxiv_id_base} (--no-pdf); "
                "skipping (will retry next run)",
                file=sys.stderr,
            )
            return OnePaperSummaryOutcome(
                arxiv_id_base=arxiv_id_base,
                summary=None,
                pdf_state=pdf_state,
                failed=True,
            )
        if not summary_inputs.used_fulltext:
            if cache_current and cached_data is not None:
                print(
                    f"[pipeline] WARNING: pdf still unavailable for {arxiv_id_base}; "
                    "keeping abstract-only summary",
                    file=sys.stderr,
                )
                return OnePaperSummaryOutcome(
                    arxiv_id_base=arxiv_id_base,
                    summary=cached_data,
                    pdf_state=pdf_state,
                    failed=False,
                )
            print(
                f"[pipeline] WARNING: pdf unavailable for {arxiv_id_base}; "
                "summarizing from abstract",
                file=sys.stderr,
            )

        summary, summary_call_meta = _run_summary_call_with_meta(
            paper=paper,
            triage=triage,
            raw_fulltext=summary_inputs.raw_text,
            fulltext_slices=summary_inputs.slices,
            used_fulltext=summary_inputs.used_fulltext,
            notes=summary_inputs.notes,
            llm=summary_llm,
            prompt_template=summary_ctx.summarize_prompt,
            repair_template=summary_ctx.repair_prompt,
            schema=summary_ctx.schema,
            max_input_tokens=cfg.summary_max_input_tokens,
        )
        repair_used = bool(summary_call_meta.get("repair_used", False))
        db.upsert_summary(
            month,
            summary,
            meta=build_stage_metadata(
                summary_ctx.descriptor,
                repair_used=repair_used,
                updated_at_source=str(paper.get("updated", "")).strip() or None,
            ),
        )
        _maybe_sleep_after_summary_call(cfg)
        return OnePaperSummaryOutcome(
            arxiv_id_base=arxiv_id_base,
            summary=summary,
            pdf_state=pdf_state,
            failed=False,
            repair_used=repair_used,
        )
    except Exception as exc:
        if isinstance(exc, LLMRateLimitError):
            raise
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
        return OnePaperSummaryOutcome(
            arxiv_id_base=arxiv_id_base,
            summary=None,
            pdf_state=pdf_state,
            failed=True,
        )


def _load_existing_pdf_map(month_out: Path) -> dict[str, dict]:
    """Preserve PDF extract metadata already written to backend_rows.jsonl."""
    path = month_out / "backend_rows.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        aid = str(row.get("arxiv_id_base", "")).strip()
        pdf = row.get("pdf")
        if aid and isinstance(pdf, dict):
            out[aid] = pdf
    return out


def _rerender_month_from_db(
    cfg: Config,
    db: DigestDB,
    month: str,
    *,
    pdf_overrides: dict[str, dict] | None = None,
) -> None:
    """Rebuild month outputs/site from SQLite after a straggler summary succeeds."""
    month_out = cfg.output_dir / month
    month_out.mkdir(parents=True, exist_ok=True)
    candidates = db.list_papers_for_month(month)
    triage_raw = db.list_triage_for_month(month)
    triage_rows = [
        normalize_triage_row(str(t.get("arxiv_id_base", "")), t) for t in triage_raw
    ]
    triage_map = {t["arxiv_id_base"]: t for t in triage_rows}
    summaries = sorted(
        db.list_summaries_for_month(month),
        key=lambda x: (x.get("published_date", ""), x.get("arxiv_id_base", "")),
    )
    summary_map = {s["arxiv_id_base"]: s for s in summaries}
    pdf_map = _load_existing_pdf_map(month_out)
    if pdf_overrides:
        pdf_map.update(pdf_overrides)
    for candidate in candidates:
        aid = candidate["arxiv_id_base"]
        if aid not in pdf_map:
            pdf_map[aid] = empty_pdf_state()

    write_jsonl(month_out / "papers.jsonl", summaries)
    backend_rows = build_backend_rows(candidates, triage_map, summary_map, pdf_map)
    write_jsonl(month_out / "backend_rows.jsonl", backend_rows)

    featured_paper = None
    digest_path = month_out / "digest.json"
    if digest_path.exists():
        try:
            existing = json.loads(digest_path.read_text(encoding="utf-8"))
            featured_paper = existing.get("featured_paper")
        except (json.JSONDecodeError, OSError):
            featured_paper = None

    digest = build_digest(
        month, candidates, triage_rows, summaries, featured_paper=featured_paper
    )
    write_json(month_out / "digest.json", digest)
    metadata_map = {c["arxiv_id_base"]: c for c in candidates}
    write_month_site(
        cfg.docs_dir,
        month,
        summaries,
        metadata_map,
        digest,
        backend_rows=backend_rows,
    )
    db.upsert_run(month, digest["stats"])


def resummarize_stragglers(
    cfg: Config,
    *,
    no_pdf: bool = False,
    no_site: bool = False,
    skip_ids: set[str] | None = None,
) -> StragglerStats:
    """Re-attempt summarization for accept'd papers that have no summary row.

    Opens the DB first; if there are no stragglers, returns without building an
    LLM client. On success, re-renders affected months (unless ``no_site``).

    ``skip_ids`` excludes papers that already failed summary earlier in the same
    ``run_window`` so we do not double-process them before the next daily run.
    """
    db = DigestDB(cfg.data_dir / "digest.sqlite")
    try:
        skip = skip_ids or set()
        stragglers = [
            (month, aid)
            for month, aid in db.get_accepted_without_summary()
            if aid not in skip
        ]
        if not stragglers:
            return StragglerStats()

        api_key = load_api_key(cfg.llm_provider)
        summary_llm_config = LLMCallConfig(
            provider=cfg.llm_provider,
            api_key=api_key,
            model=cfg.llm_model_summary,
            temperature=cfg.llm_temperature_summary,
            max_output_tokens=cfg.llm_max_output_tokens_summary,
            base_url=provider_base_url(cfg.llm_provider),
        )
        summary_llm = build_llm_call(summary_llm_config)
        try:
            summary_ctx = load_summary_stage_context(summary_llm_config)
            attempted = 0
            succeeded = 0
            failed = 0
            failed_ids: list[str] = []
            months_touched: set[str] = set()
            pdf_by_month: dict[str, dict[str, dict]] = {}

            for month, arxiv_id_base in stragglers:
                paper = db.get_paper(arxiv_id_base)
                triage = db.get_triage(arxiv_id_base)
                if paper is None or triage is None:
                    failed += 1
                    failed_ids.append(arxiv_id_base)
                    print(
                        f"[stragglers] WARNING: missing paper/triage for {arxiv_id_base}; skipping",
                        file=sys.stderr,
                    )
                    continue

                attempted += 1
                month_out = cfg.output_dir / month
                month_out.mkdir(parents=True, exist_ok=True)
                prior_summary = db.get_summary(arxiv_id_base)
                outcome = _summarize_one_paper(
                    paper=paper,
                    triage=normalize_triage_row(arxiv_id_base, triage),
                    month=month,
                    month_out=month_out,
                    cfg=cfg,
                    db=db,
                    summary_llm=summary_llm,
                    summary_ctx=summary_ctx,
                    summary_llm_config=summary_llm_config,
                    no_pdf=no_pdf,
                    force=False,
                )
                if outcome.failed or outcome.summary is None:
                    failed += 1
                    failed_ids.append(arxiv_id_base)
                else:
                    succeeded += 1
                    upgraded = summary_used_fulltext(outcome.summary) and not summary_used_fulltext(
                        prior_summary
                    )
                    if prior_summary is None or upgraded:
                        months_touched.add(month)
                        pdf_by_month.setdefault(month, {})[arxiv_id_base] = outcome.pdf_state

            if months_touched and not no_site:
                for month in sorted(months_touched):
                    _rerender_month_from_db(
                        cfg,
                        db,
                        month,
                        pdf_overrides=pdf_by_month.get(month),
                    )
                update_home(cfg.docs_dir)

            print(
                f"[stragglers] attempted={attempted} succeeded={succeeded} failed={failed}",
                flush=True,
            )
            return StragglerStats(
                attempted=attempted,
                succeeded=succeeded,
                failed=failed,
                affected_months=tuple(sorted(months_touched)),
                failed_ids=tuple(failed_ids),
            )
        finally:
            summary_llm.close()
    finally:
        db.close()


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
            outcome = _summarize_one_paper(
                paper=paper,
                triage=triage_map[arxiv_id_base],
                month=month,
                month_out=month_out,
                cfg=cfg,
                db=db,
                summary_llm=summary_llm,
                summary_ctx=summary_ctx,
                summary_llm_config=summary_llm_config,
                no_pdf=no_pdf,
                force=force,
            )
            pdf_map[arxiv_id_base] = outcome.pdf_state
            if outcome.failed or outcome.summary is None:
                summary_failure_count += 1
                failed_summary_ids.append(arxiv_id_base)
            else:
                summaries.append(outcome.summary)
                summary_map[arxiv_id_base] = outcome.summary

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

    # Same-run summary failures already logged a retry-next-run warning; do not
    # immediately re-attempt them in the straggler sweep (avoids double LLM spend
    # and keeps WindowRunStats.failed_summary_ids authoritative for this window).
    same_run_failures = {
        aid for month_stats in per_month for aid in month_stats.failed_summary_ids
    }
    straggler_stats = resummarize_stragglers(
        cfg,
        no_pdf=no_pdf,
        no_site=no_site,
        skip_ids=same_run_failures,
    )
    merged_months = tuple(
        sorted(set(affected_months) | set(straggler_stats.affected_months))
    )

    return WindowRunStats(
        since=since.astimezone(timezone.utc),
        until=until.astimezone(timezone.utc),
        window_candidates=len(window_candidates),
        affected_months=merged_months,
        per_month=tuple(per_month),
        straggler_attempted=straggler_stats.attempted,
        straggler_succeeded=straggler_stats.succeeded,
        straggler_failures=straggler_stats.failed,
        failed_straggler_ids=straggler_stats.failed_ids,
    )
