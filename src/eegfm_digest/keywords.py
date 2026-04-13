"""Topic-config-backed keyword and category helpers for arXiv retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

ARXIV_CATEGORIES = {
    "q-bio.NC",
    "cs.LG",
    "stat.ML",
    "eess.SP",
    "cs.AI",
    "cs.NE",
}

QUERY_A = (
    'all:(eeg OR electroencephalograph* OR brainwave*) AND '
    'all:("foundation model" OR pretrain OR pretrained OR "self-supervised" OR "self supervised")'
)

QUERY_B = (
    'all:(eeg OR electroencephalograph* OR brainwave*) AND '
    'all:("representation learning" OR masked OR transfer OR generaliz*)'
)

DEFAULT_TOPIC_CONFIG_PATH = Path("configs/topics/eeg_fm.json")


@dataclass(frozen=True)
class TopicConfig:
    slug: str
    title: str
    description: str
    categories: tuple[str, ...]
    queries: tuple[str, ...]


def default_topic_payload() -> dict[str, object]:
    return {
        "slug": "eeg_fm",
        "title": "EEG Foundation Model Digest",
        "description": "Monthly arXiv digest for EEG-FM papers",
        "categories": sorted(ARXIV_CATEGORIES),
        "queries": [QUERY_A, QUERY_B],
    }


def _topic_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return Path(os.environ.get("TOPIC_CONFIG_PATH", str(DEFAULT_TOPIC_CONFIG_PATH)))


def load_topic_config(path: Path | None = None) -> TopicConfig:
    payload = default_topic_payload()
    config_path = _topic_config_path(path)
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Invalid topic config at {config_path}: expected a JSON object.")
        payload.update(loaded)

    categories = tuple(
        sorted(
            {
                str(item).strip()
                for item in payload.get("categories", [])
                if str(item).strip()
            }
        )
    )
    queries = tuple(str(item).strip() for item in payload.get("queries", []) if str(item).strip())
    if not categories:
        raise RuntimeError(f"Invalid topic config at {config_path}: categories must be a non-empty list.")
    if not queries:
        raise RuntimeError(f"Invalid topic config at {config_path}: queries must be a non-empty list.")

    return TopicConfig(
        slug=str(payload.get("slug", "topic")).strip() or "topic",
        title=str(payload.get("title", "Research Digest")).strip() or "Research Digest",
        description=str(payload.get("description", "")).strip(),
        categories=categories,
        queries=queries,
    )


def active_arxiv_categories(path: Path | None = None) -> set[str]:
    return set(load_topic_config(path).categories)


def active_arxiv_queries(path: Path | None = None) -> list[str]:
    return list(load_topic_config(path).queries)
