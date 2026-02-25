from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .config import Config, load_config
from .db import DigestDB
from .llm_openai_compat import RateLimitError
from .pipeline import run_month
from .profile import load_profile


@dataclass(frozen=True)
class BatchRunConfig:
    months: list[str]
    months_from_outputs: bool = True
    no_site: bool = False
    triage_force: bool = False
    summary_force: bool = False
    include_borderline: bool = False
    triage_sleep_seconds: float = 0.0
    summary_sleep_seconds: float = 0.0
    stop_on_rate_limit: bool = True
    sync_cache_from_outputs: bool = True
    max_candidates: int | None = None
    max_accepted: int | None = None
    env_path: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid batch config at {path}: expected object")
    return raw


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_batch_config(path: Path) -> BatchRunConfig:
    raw = _load_yaml(path)
    months = raw.get("months", [])
    if isinstance(months, str):
        months = [months]
    if not isinstance(months, list):
        raise RuntimeError("`months` must be a list of YYYY-MM strings")
    months = [str(month) for month in months if str(month).strip()]

    return BatchRunConfig(
        months=months,
        months_from_outputs=bool(raw.get("months_from_outputs", True)),
        no_site=bool(raw.get("no_site", False)),
        triage_force=bool(raw.get("triage_force", False)),
        summary_force=bool(raw.get("summary_force", False)),
        include_borderline=bool(raw.get("include_borderline", False)),
        triage_sleep_seconds=float(raw.get("triage_sleep_seconds", 0.0)),
        summary_sleep_seconds=float(raw.get("summary_sleep_seconds", 0.0)),
        stop_on_rate_limit=bool(raw.get("stop_on_rate_limit", True)),
        sync_cache_from_outputs=bool(raw.get("sync_cache_from_outputs", True)),
        max_candidates=int(raw["max_candidates"]) if raw.get("max_candidates") is not None else None,
        max_accepted=int(raw["max_accepted"]) if raw.get("max_accepted") is not None else None,
        env_path=str(raw.get("env_path")).strip() if raw.get("env_path") else None,
    )


def _discover_months_from_outputs(output_dir: Path, profile_id: str) -> list[str]:
    root = output_dir / profile_id
    if not root.exists():
        return []
    months: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "arxiv_raw.json").exists():
            months.append(entry.name)
    return sorted(months)


def _effective_months(run_cfg: BatchRunConfig, cfg: Config, profile_id: str) -> list[str]:
    months = list(run_cfg.months)
    if run_cfg.months_from_outputs:
        for month in _discover_months_from_outputs(cfg.output_dir, profile_id):
            if month not in months:
                months.append(month)
    months = sorted(set(months))
    if not months:
        raise RuntimeError(
            "No months found. Provide `months` in config or enable `months_from_outputs` with existing outputs."
        )
    return months


def _bootstrap_cache_from_outputs(db: DigestDB, month: str, month_out: Path) -> None:
    for row in _load_jsonl(month_out / "triage.jsonl"):
        db.upsert_triage(month, row)
    for row in _load_jsonl(month_out / "papers.jsonl"):
        db.upsert_summary(month, row)
    raw_path = month_out / "arxiv_raw.json"
    if raw_path.exists():
        for row in _load_json(raw_path):
            db.upsert_paper(month, row)


def run_batch(profile_id: str, config_path: Path, profiles_dir: Path | None = None) -> None:
    run_cfg = _parse_batch_config(config_path)
    cfg = load_config()
    profile = load_profile(profile_id, profiles_dir=profiles_dir or cfg.profiles_dir)

    if run_cfg.env_path:
        load_dotenv(Path(run_cfg.env_path).expanduser())

    if run_cfg.max_candidates is not None:
        cfg = replace(cfg, max_candidates=run_cfg.max_candidates)
    if run_cfg.max_accepted is not None:
        cfg = replace(cfg, max_accepted=run_cfg.max_accepted)
    if run_cfg.include_borderline:
        cfg = replace(cfg, include_borderline=True)

    months = _effective_months(run_cfg, cfg, profile.id)
    print(f"[batch] profile={profile.id} months={months}")

    db = DigestDB(cfg.data_dir / f"{profile.id}.sqlite")
    try:
        if run_cfg.sync_cache_from_outputs:
            for month in months:
                month_out = cfg.output_dir / profile.id / month
                month_out.mkdir(parents=True, exist_ok=True)
                _bootstrap_cache_from_outputs(db, month, month_out)
    finally:
        db.close()

    for month in months:
        try:
            run_month(
                cfg=cfg,
                profile=profile,
                month=month,
                no_pdf=False,
                no_site=run_cfg.no_site,
                force=False,
                force_triage=run_cfg.triage_force,
                force_summary=run_cfg.summary_force,
                triage_sleep_seconds=run_cfg.triage_sleep_seconds,
                summary_sleep_seconds=run_cfg.summary_sleep_seconds,
                stop_on_rate_limit=run_cfg.stop_on_rate_limit,
                max_candidates_override=run_cfg.max_candidates,
                max_accepted_override=run_cfg.max_accepted,
                include_borderline_override=run_cfg.include_borderline,
            )
        except RateLimitError:
            if run_cfg.stop_on_rate_limit:
                raise
            print(f"[batch] rate limit on month={month}, continuing by config.")
        time.sleep(0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for BYOM profile template.")
    parser.add_argument("--profile", required=True, help="Profile id (e.g. eeg_fm)")
    parser.add_argument("--config", required=True, help="YAML config path (e.g. configs/batch/all_months.yaml)")
    parser.add_argument("--profiles-dir", default=None, help="Optional override for profiles root")
    args = parser.parse_args()

    run_batch(
        profile_id=args.profile,
        config_path=Path(args.config),
        profiles_dir=Path(args.profiles_dir) if args.profiles_dir else None,
    )


if __name__ == "__main__":
    main()
