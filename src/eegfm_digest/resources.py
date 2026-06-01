"""Repository-root paths for schemas, prompts, and static assets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def schema_path(name: str) -> Path:
    return repo_root() / "schemas" / name


def prompt_path(name: str) -> Path:
    return repo_root() / "prompts" / name


def docs_asset_path(name: str) -> Path:
    return repo_root() / "docs" / "assets" / name
