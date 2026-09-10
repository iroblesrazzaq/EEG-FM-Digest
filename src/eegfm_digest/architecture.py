"""Hugging Face config.json fact sheets for new open-weight models.

Daily enrichment fetches public ``config.json`` only. It never downloads
weight files, and callers must treat failures as skips.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

HF_USER_AGENT = (
    "eegfm-digest/0.1 (https://github.com/iroblesrazzaq/EEG-FM-Digest; "
    "mailto:ismaelroblesrazzaq@gmail.com)"
)
HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"})
HF_RESERVED_OWNERS = frozenset(
    {
        "models",
        "datasets",
        "spaces",
        "organizations",
        "collections",
        "papers",
        "docs",
        "login",
        "settings",
        "blog",
        "learn",
        "tasks",
        "chat",
        "api-inference",
        "huggingface",
    }
)
HF_RESERVED_NAMES = frozenset({"tree", "blob", "resolve", "raw", "discussions", "commits"})
CONFIG_TIMEOUT_SECONDS = 8.0


def looks_like_hf_url(url: str | None) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if "huggingface.co" in lowered or "hf.co/" in lowered or lowered.startswith("hf.co/"):
        return True
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return parsed.netloc.lower() in HF_HOSTS


def parse_hf_repo_id(url: str | None) -> str | None:
    """Return ``owner/name`` for a Hugging Face *model* repo URL, else None."""
    raw = str(url or "").strip()
    if not raw:
        return None
    if "://" not in raw and "/" in raw and " " not in raw:
        parts = [p for p in raw.split("/") if p]
        if len(parts) >= 2 and parts[0].lower() not in HF_RESERVED_OWNERS:
            owner, name = parts[0], parts[1]
            if name.lower() not in HF_RESERVED_NAMES:
                return f"{owner}/{name}"
        return None
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host not in HF_HOSTS:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    if parts[0].lower() in HF_RESERVED_OWNERS:
        return None
    owner, name = parts[0], parts[1]
    if name.lower() in HF_RESERVED_NAMES:
        return None
    return f"{owner}/{name}"


def hfviewer_url_for_repo(repo_id: str) -> str:
    return f"https://hfviewer.com/{repo_id}"


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            raw = value.strip().replace(",", "")
            if raw.isdigit():
                return int(raw)
    return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None


def fact_sheet_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compact Raschka-like fields from a transformers-style config.json."""
    architectures = cfg.get("architectures")
    arch_name = None
    if isinstance(architectures, list):
        arch_name = _first_str(*architectures)
    return {
        "model_type": _first_str(cfg.get("model_type")),
        "architectures": arch_name,
        "hidden_size": _first_int(
            cfg.get("hidden_size"),
            cfg.get("n_embd"),
            cfg.get("d_model"),
            cfg.get("n_embed"),
            cfg.get("embed_dim"),
        ),
        "num_layers": _first_int(
            cfg.get("num_hidden_layers"),
            cfg.get("n_layer"),
            cfg.get("n_layers"),
            cfg.get("num_layers"),
            cfg.get("depth"),
        ),
        "num_attention_heads": _first_int(
            cfg.get("num_attention_heads"),
            cfg.get("n_head"),
            cfg.get("n_heads"),
            cfg.get("heads"),
        ),
        "num_key_value_heads": _first_int(
            cfg.get("num_key_value_heads"),
            cfg.get("num_kv_heads"),
        ),
        "context_length": _first_int(
            cfg.get("max_position_embeddings"),
            cfg.get("n_positions"),
            cfg.get("max_seq_len"),
            cfg.get("max_sequence_length"),
            cfg.get("seq_length"),
        ),
        "vocab_size": _first_int(cfg.get("vocab_size")),
        "hidden_act": _first_str(
            cfg.get("hidden_act"),
            cfg.get("hidden_activation"),
            cfg.get("activation"),
        ),
        "intermediate_size": _first_int(
            cfg.get("intermediate_size"),
            cfg.get("ffn_dim"),
            cfg.get("ffn_hidden_size"),
            cfg.get("n_inner"),
        ),
        "num_params": _first_int(
            cfg.get("num_params"),
            cfg.get("n_params"),
            cfg.get("num_parameters"),
        ),
    }


def fetch_hf_config(
    repo_id: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = CONFIG_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """GET ``config.json`` for a public HF repo. Raises on HTTP/parse errors."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    headers = {"User-Agent": HF_USER_AGENT}
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)
        close_client = True
    try:
        response = client.get(url, headers=headers, timeout=timeout)
        if response.status_code in {401, 403}:
            raise PermissionError(f"gated:{response.status_code}")
        if response.status_code == 404:
            raise FileNotFoundError("missing_config")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("invalid_config")
        return payload
    finally:
        if close_client:
            client.close()


def _skip_payload(
    *,
    reason: str,
    hf_repo: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "hf_repo": hf_repo,
        "hfviewer_url": hfviewer_url_for_repo(hf_repo) if hf_repo else None,
        "fact_sheet": None,
        "skip_reason": reason,
    }


def _ok_payload(repo_id: str, fact_sheet: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "hf_repo": repo_id,
        "hfviewer_url": hfviewer_url_for_repo(repo_id),
        "fact_sheet": fact_sheet,
        "skip_reason": None,
    }


def architecture_from_summary(
    summary: dict[str, Any] | None,
    existing: dict[str, Any] | None = None,
    *,
    force: bool = False,
    fetch_config: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Best-effort architecture payload. Never raises to the digest pipeline."""
    if not isinstance(summary, dict):
        return None
    if str(summary.get("paper_type") or "").strip() != "new_model":
        return None

    weights_url = None
    open_source = summary.get("open_source")
    if isinstance(open_source, dict):
        weights_url = open_source.get("weights_url")
    if not str(weights_url or "").strip():
        return None

    repo_id = parse_hf_repo_id(str(weights_url))
    if repo_id is None:
        reason = "not_a_model_repo" if looks_like_hf_url(str(weights_url)) else "not_huggingface"
        return _skip_payload(reason=reason)

    if (
        not force
        and isinstance(existing, dict)
        and existing.get("status") == "ok"
        and existing.get("hf_repo") == repo_id
    ):
        return existing

    fetcher = fetch_config or fetch_hf_config
    try:
        cfg = fetcher(repo_id)
        if not isinstance(cfg, dict):
            return _skip_payload(reason="invalid_config", hf_repo=repo_id)
        return _ok_payload(repo_id, fact_sheet_from_config(cfg))
    except PermissionError:
        return _skip_payload(reason="gated", hf_repo=repo_id)
    except FileNotFoundError:
        return _skip_payload(reason="missing_config", hf_repo=repo_id)
    except Exception:  # noqa: BLE001 — enrichment must never fail the digest
        return _skip_payload(reason="fetch_failed", hf_repo=repo_id)
