from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _expand_env_str(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        if env_name in os.environ:
            return os.environ[env_name]
        if default is not None:
            return default
        raise RuntimeError(f"Missing required environment variable: {env_name}")

    return _ENV_PATTERN.sub(repl, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_str(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _expand_env(v) for k, v in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid YAML object in {path}")
    return _expand_env(raw)


def _required_dict(root: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Missing object `{path}.{key}`")
    return value


def _required_list(root: dict[str, Any], key: str, path: str) -> list[Any]:
    value = root.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"Missing list `{path}.{key}`")
    return value


def _required_str(root: dict[str, Any], key: str, path: str) -> str:
    value = root.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing string `{path}.{key}`")
    return value.strip()


@dataclass(frozen=True)
class LLMProfileConfig:
    provider: str
    base_url: str
    api_key_env: str
    triage_model: str
    summary_model: str
    temperature_triage: float
    temperature_summary: float
    max_output_tokens_triage: int
    max_output_tokens_summary: int
    token_chars_per_token: int
    request_timeout_seconds: float


@dataclass(frozen=True)
class RetrievalConfig:
    categories: list[str]
    queries: list[str]
    rate_limit_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class PipelineProfileConfig:
    max_candidates: int
    max_accepted: int
    include_borderline: bool
    max_borderline_pdfs: int
    pdf_rate_limit_seconds: float
    text_head_chars: int
    text_tail_chars: int
    summary_max_input_tokens: int


@dataclass(frozen=True)
class SummaryProfileConfig:
    allowed_tags: dict[str, list[str]] | None


@dataclass(frozen=True)
class ProfilePaths:
    profile_yaml: Path
    triage_prompt: Path
    summary_prompt: Path
    repair_prompt: Path
    triage_schema: Path
    summary_schema: Path


@dataclass(frozen=True)
class DigestProfile:
    id: str
    display_name: str
    root: Path
    llm: LLMProfileConfig
    retrieval: RetrievalConfig
    pipeline: PipelineProfileConfig
    summary: SummaryProfileConfig
    site: dict[str, Any]
    paths: ProfilePaths


def _resolve_profile_root(profile_id: str, profiles_dir: Path) -> Path:
    candidate = Path(profile_id)
    if candidate.exists() and candidate.is_dir():
        return candidate
    return profiles_dir / profile_id


def load_profile(profile_id: str, profiles_dir: Path = Path("profiles")) -> DigestProfile:
    root = _resolve_profile_root(profile_id, profiles_dir)
    profile_yaml = root / "profile.yaml"
    if not profile_yaml.exists():
        raise RuntimeError(f"Profile not found: {profile_yaml}")

    raw = _load_yaml(profile_yaml)
    pid = _required_str(raw, "id", "profile")
    display_name = _required_str(raw, "display_name", "profile")

    llm_raw = _required_dict(raw, "llm", "profile")
    provider = _required_str(llm_raw, "provider", "profile.llm").lower()
    if provider != "openai_compatible":
        raise RuntimeError("Only `openai_compatible` provider is supported in this template branch.")

    retrieval_raw = _required_dict(raw, "retrieval", "profile")
    pipeline_raw = _required_dict(raw, "pipeline", "profile")
    summary_raw = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    site_raw = raw.get("site") if isinstance(raw.get("site"), dict) else {}

    paths = ProfilePaths(
        profile_yaml=profile_yaml,
        triage_prompt=root / "prompts" / "triage.md",
        summary_prompt=root / "prompts" / "summarize.md",
        repair_prompt=root / "prompts" / "repair_json.md",
        triage_schema=root / "schemas" / "triage.json",
        summary_schema=root / "schemas" / "summary.json",
    )
    for required_path in (
        paths.triage_prompt,
        paths.summary_prompt,
        paths.repair_prompt,
        paths.triage_schema,
        paths.summary_schema,
    ):
        if not required_path.exists():
            raise RuntimeError(f"Missing profile file: {required_path}")

    allowed_tags: dict[str, list[str]] | None = None
    raw_allowed_tags = summary_raw.get("allowed_tags")
    if isinstance(raw_allowed_tags, dict):
        allowed_tags = {}
        for k, v in raw_allowed_tags.items():
            if isinstance(v, list):
                allowed_tags[str(k)] = [str(item) for item in v if str(item).strip()]

    return DigestProfile(
        id=pid,
        display_name=display_name,
        root=root,
        llm=LLMProfileConfig(
            provider=provider,
            base_url=_required_str(llm_raw, "base_url", "profile.llm"),
            api_key_env=_required_str(llm_raw, "api_key_env", "profile.llm"),
            triage_model=_required_str(llm_raw, "triage_model", "profile.llm"),
            summary_model=_required_str(llm_raw, "summary_model", "profile.llm"),
            temperature_triage=float(llm_raw.get("temperature_triage", 0.2)),
            temperature_summary=float(llm_raw.get("temperature_summary", 0.2)),
            max_output_tokens_triage=int(llm_raw.get("max_output_tokens_triage", 1024)),
            max_output_tokens_summary=int(llm_raw.get("max_output_tokens_summary", 2048)),
            token_chars_per_token=int(llm_raw.get("token_chars_per_token", 4)),
            request_timeout_seconds=float(llm_raw.get("request_timeout_seconds", 180.0)),
        ),
        retrieval=RetrievalConfig(
            categories=[str(item) for item in _required_list(retrieval_raw, "categories", "profile.retrieval")],
            queries=[str(item) for item in _required_list(retrieval_raw, "queries", "profile.retrieval")],
            rate_limit_seconds=float(retrieval_raw.get("rate_limit_seconds", 2.0)),
            connect_timeout_seconds=float(retrieval_raw.get("connect_timeout_seconds", 10.0)),
            read_timeout_seconds=float(retrieval_raw.get("read_timeout_seconds", 60.0)),
            retries=int(retrieval_raw.get("retries", 2)),
            retry_backoff_seconds=float(retrieval_raw.get("retry_backoff_seconds", 2.0)),
        ),
        pipeline=PipelineProfileConfig(
            max_candidates=int(pipeline_raw.get("max_candidates", 500)),
            max_accepted=int(pipeline_raw.get("max_accepted", 80)),
            include_borderline=bool(pipeline_raw.get("include_borderline", False)),
            max_borderline_pdfs=int(pipeline_raw.get("max_borderline_pdfs", 20)),
            pdf_rate_limit_seconds=float(pipeline_raw.get("pdf_rate_limit_seconds", 5.0)),
            text_head_chars=int(pipeline_raw.get("text_head_chars", 80_000)),
            text_tail_chars=int(pipeline_raw.get("text_tail_chars", 20_000)),
            summary_max_input_tokens=int(pipeline_raw.get("summary_max_input_tokens", 120_000)),
        ),
        summary=SummaryProfileConfig(allowed_tags=allowed_tags),
        site=site_raw,
        paths=paths,
    )


def init_profile(
    new_profile_id: str,
    from_profile: str = "generic_demo",
    profiles_dir: Path = Path("profiles"),
) -> Path:
    if not re.match(r"^[a-zA-Z0-9_-]+$", new_profile_id):
        raise RuntimeError("Profile id must match [a-zA-Z0-9_-]+")

    src_root = _resolve_profile_root(from_profile, profiles_dir)
    if not src_root.exists():
        raise RuntimeError(f"Template profile not found: {src_root}")
    dst_root = profiles_dir / new_profile_id
    if dst_root.exists():
        raise RuntimeError(f"Destination profile already exists: {dst_root}")

    dst_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dst_root)

    profile_yaml = dst_root / "profile.yaml"
    raw = _load_yaml(profile_yaml)
    raw["id"] = new_profile_id
    if not str(raw.get("display_name", "")).strip():
        raw["display_name"] = new_profile_id.replace("_", " ").replace("-", " ").title()
    profile_yaml.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Validate scaffold output.
    load_profile(new_profile_id, profiles_dir=profiles_dir)
    return dst_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile utilities for BYOM template branch.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Scaffold a new profile from a template.")
    init_parser.add_argument("profile_id", help="New profile id (e.g. robotics_fm)")
    init_parser.add_argument("--from", dest="from_profile", default="generic_demo")
    init_parser.add_argument("--profiles-dir", default="profiles")

    args = parser.parse_args()
    if args.command == "init":
        dst = init_profile(
            new_profile_id=args.profile_id,
            from_profile=args.from_profile,
            profiles_dir=Path(args.profiles_dir),
        )
        print(f"Created profile: {dst}")


if __name__ == "__main__":
    main()
