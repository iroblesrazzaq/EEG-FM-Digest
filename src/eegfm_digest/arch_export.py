"""Local CLI to export interactive architecture graphs into docs/.

Not hooked from ``run.py --daily``. Downloads ``config.json``, Hub metadata,
and safetensors *headers* only — never weight tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from .arch_graph import (
    fact_sheet_for_graph,
    graph_for_model,
    looks_like_reve,
    merge_reve_config,
    short_model_label,
)
from .arch_safetensors import RangeIgnoredError, SafetensorsHeaderError, fetch_safetensors_header
from .architecture import HF_USER_AGENT, parse_hf_repo_id
from .site import refresh_html_shells, update_home

DEFAULT_FALLBACK_REPO = "brain-bzh/reve-base"
GRAPH_REL_PREFIX = "data/arch"


def hf_token() -> str | None:
    raw = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if raw and raw.strip():
        return raw.strip()
    return None


def hub_headers(token: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": HF_USER_AGENT}
    tok = token if token is not None else hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def fetch_hub_model_info(
    repo_id: str,
    *,
    client: httpx.Client,
    token: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any] | None:
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        response = client.get(url, headers=hub_headers(token), timeout=timeout)
        if response.status_code in {401, 403, 404}:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 — local export should fall back
        return None


def fetch_hf_config_optional(
    repo_id: str,
    *,
    client: httpx.Client,
    token: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any] | None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    try:
        response = client.get(url, headers=hub_headers(token), timeout=timeout)
        if response.status_code in {401, 403, 404}:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 — local export should fall back
        return None


def safetensors_filenames(info: dict[str, Any] | None) -> list[str]:
    if not isinstance(info, dict):
        return []
    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        return []
    names: list[str] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("rfilename") or "").strip()
        if name.endswith(".safetensors") and not name.endswith(".index.json"):
            names.append(name)
    unsharded = [name for name in names if "-of-" not in name.split("/")[-1]]
    return unsharded or names


def hub_param_count(info: dict[str, Any] | None) -> int | None:
    if not isinstance(info, dict):
        return None
    safetensors = info.get("safetensors")
    if isinstance(safetensors, dict):
        total = safetensors.get("total")
        if isinstance(total, int):
            return total
    return None


def _try_safetensors_header(
    repo_id: str,
    filenames: list[str],
    *,
    client: httpx.Client,
    token: str | None,
) -> dict[str, dict[str, Any]] | None:
    candidates = filenames or ["model.safetensors"]
    for name in candidates[:3]:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{name}"
        try:
            return fetch_safetensors_header(url, client=client, token=token)
        except (PermissionError, FileNotFoundError, RangeIgnoredError, SafetensorsHeaderError, httpx.HTTPError):
            continue
    return None


def resolve_export_sources(
    repo_id: str,
    *,
    arxiv_id: str,
    fallback_repo: str | None,
    client: httpx.Client,
    token: str | None,
) -> tuple[str, dict[str, Any] | None, dict[str, dict[str, Any]] | None, str]:
    """Return repo used, config, tensors, and a source label."""
    repos = [repo_id]
    if fallback_repo and fallback_repo != repo_id:
        repos.append(fallback_repo)

    cfg: dict[str, Any] | None = None
    tensors: dict[str, dict[str, Any]] | None = None
    used_repo = repo_id
    saw_hub_metadata = False
    source_bits: list[str] = []

    for candidate in repos:
        info = fetch_hub_model_info(candidate, client=client, token=token)
        if info is not None:
            saw_hub_metadata = True
        candidate_cfg = fetch_hf_config_optional(candidate, client=client, token=token)
        filenames = safetensors_filenames(info)
        candidate_tensors = _try_safetensors_header(
            candidate, filenames, client=client, token=token
        )
        params = hub_param_count(info)
        if candidate_cfg is not None:
            cfg = dict(candidate_cfg)
            if params is not None:
                cfg["num_params"] = params
            used_repo = candidate
            source_bits.append("hf_config")
        elif params is not None:
            cfg = {"num_params": params}
            used_repo = candidate
        if candidate_tensors is not None:
            tensors = candidate_tensors
            used_repo = candidate
            source_bits.append("safetensors_header")
        if source_bits or (looks_like_reve(candidate, arxiv_id, candidate_cfg) and saw_hub_metadata):
            if looks_like_reve(candidate, arxiv_id, candidate_cfg) and candidate_cfg is None:
                used_repo = candidate
            if source_bits:
                break

    if looks_like_reve(used_repo, arxiv_id, cfg):
        cfg = merge_reve_config(cfg)
        bits: list[str] = []
        if "hf_config" in source_bits:
            bits.append("hf_config")
        elif saw_hub_metadata:
            bits.append("hub_metadata")
        bits.append("published_defaults")
        if "safetensors_header" in source_bits:
            bits.append("safetensors_header")
        return used_repo, cfg, tensors, "+".join(bits)

    if cfg is None and tensors is None:
        raise RuntimeError(
            f"Could not read config or safetensors header for {repo_id}"
            + (f" (fallback {fallback_repo})" if fallback_repo else "")
            + ". Set HF_TOKEN for gated repos."
        )
    source = "+".join(source_bits) if source_bits else "unknown"
    return used_repo, cfg, tensors, source


def build_export_payload(
    *,
    arxiv_id: str,
    month: str,
    title: str,
    repo_id: str,
    cfg: dict[str, Any] | None,
    tensor_names: list[str] | None,
    source: str,
    weights_url: str | None = None,
) -> dict[str, Any]:
    graph = graph_for_model(cfg=cfg, tensor_names=tensor_names, repo_id=repo_id, arxiv_id=arxiv_id)
    fact_sheet = fact_sheet_for_graph(cfg)
    graph_path = f"{GRAPH_REL_PREFIX}/{arxiv_id}.json"
    payload: dict[str, Any] = {
        "arxiv_id_base": arxiv_id,
        "title": title,
        "label": short_model_label(title, repo_id),
        "month": month,
        "hf_repo": repo_id,
        "weights_url": weights_url,
        "source": source,
        "graph_path": graph_path,
        "fact_sheet": fact_sheet,
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }
    diagram = graph.get("diagram")
    if isinstance(diagram, dict):
        payload["diagram"] = diagram
    return payload


def architecture_site_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "hf_repo": payload["hf_repo"],
        "graph_path": payload["graph_path"],
        "fact_sheet": payload.get("fact_sheet"),
        "skip_reason": None,
    }


def catalog_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id_base": payload["arxiv_id_base"],
        "title": payload["title"],
        "label": payload.get("label") or short_model_label(payload.get("title"), payload.get("hf_repo")),
        "month": payload["month"],
        "hf_repo": payload["hf_repo"],
        "graph_path": payload["graph_path"],
    }


def upsert_catalog(catalog_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    catalog: dict[str, Any] = {"models": []}
    if catalog_path.exists():
        try:
            loaded = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("models"), list):
                catalog = loaded
            elif isinstance(loaded, list):
                catalog = {"models": loaded}
        except json.JSONDecodeError:
            catalog = {"models": []}
    models = [row for row in catalog.get("models", []) if isinstance(row, dict)]
    models = [row for row in models if str(row.get("arxiv_id_base")) != entry["arxiv_id_base"]]
    models.append(entry)
    models.sort(key=lambda row: (str(row.get("month") or ""), str(row.get("arxiv_id_base") or "")))
    catalog = {"models": models}
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def patch_paper_architecture(papers_path: Path, arxiv_id: str, architecture: dict[str, Any]) -> None:
    if not papers_path.exists():
        raise FileNotFoundError(f"papers.json not found: {papers_path}")
    payload = json.loads(papers_path.read_text(encoding="utf-8"))
    papers = payload.get("papers") if isinstance(payload, dict) else payload
    if not isinstance(papers, list):
        raise TypeError("papers.json has no papers list")
    found = False
    for row in papers:
        if not isinstance(row, dict):
            continue
        if str(row.get("arxiv_id_base") or "").strip() != arxiv_id:
            continue
        found = True
        row["architecture"] = architecture
        break
    if not found:
        raise KeyError(f"{arxiv_id} not found in {papers_path}")
    papers_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def title_from_papers(papers_path: Path, arxiv_id: str) -> str | None:
    if not papers_path.exists():
        return None
    try:
        payload = json.loads(papers_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    papers = payload.get("papers") if isinstance(payload, dict) else payload
    if not isinstance(papers, list):
        return None
    for row in papers:
        if not isinstance(row, dict):
            continue
        if str(row.get("arxiv_id_base") or "").strip() != arxiv_id:
            continue
        title = str(row.get("title") or "").strip()
        summary = row.get("summary")
        if isinstance(summary, dict):
            title = str(summary.get("title") or title).strip() or title
        return title or None
    return None


def write_export_artifacts(
    docs_dir: Path,
    payload: dict[str, Any],
    *,
    patch_papers: bool = True,
    refresh_shells: bool = False,
) -> Path:
    docs_dir = Path(docs_dir)
    papers_path = docs_dir / "digest" / str(payload["month"]) / "papers.json"
    if patch_papers:
        # Validate the digest row before writing graph/catalog so a bad
        # --month/--arxiv cannot leave orphan artifacts.
        if not papers_path.exists():
            raise FileNotFoundError(f"papers.json not found: {papers_path}")
        existing = json.loads(papers_path.read_text(encoding="utf-8"))
        papers = existing.get("papers") if isinstance(existing, dict) else existing
        if not isinstance(papers, list):
            raise TypeError("papers.json has no papers list")
        arxiv_id = str(payload["arxiv_id_base"])
        if not any(
            isinstance(row, dict) and str(row.get("arxiv_id_base") or "").strip() == arxiv_id
            for row in papers
        ):
            raise KeyError(f"{arxiv_id} not found in {papers_path}")
    graph_rel = str(payload["graph_path"])
    graph_path = docs_dir / graph_rel
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upsert_catalog(docs_dir / "data" / "architectures.json", catalog_entry(payload))
    if patch_papers:
        patch_paper_architecture(
            papers_path,
            payload["arxiv_id_base"],
            architecture_site_payload(payload),
        )
    if refresh_shells:
        refresh_html_shells(docs_dir)
    else:
        update_home(docs_dir)
    return graph_path


def export_architecture(
    *,
    arxiv_id: str,
    repo: str,
    month: str,
    docs_dir: Path,
    title: str | None = None,
    fallback_repo: str | None = DEFAULT_FALLBACK_REPO,
    token: str | None = None,
    client: httpx.Client | None = None,
    patch_papers: bool = True,
    refresh_shells: bool = False,
) -> dict[str, Any]:
    repo_id = parse_hf_repo_id(repo) or repo
    token = token if token is not None else hf_token()
    papers_path = Path(docs_dir) / "digest" / month / "papers.json"
    resolved_title = title or title_from_papers(papers_path, arxiv_id) or arxiv_id

    close_client = False
    if client is None:
        client = httpx.Client(timeout=20.0, headers={"User-Agent": HF_USER_AGENT}, follow_redirects=True)
        close_client = True
    try:
        used_repo, cfg, tensors, source = resolve_export_sources(
            repo_id,
            arxiv_id=arxiv_id,
            fallback_repo=fallback_repo,
            client=client,
            token=token,
        )
    finally:
        if close_client:
            client.close()

    tensor_names = list(tensors.keys()) if tensors else None
    weights_url = f"https://huggingface.co/{repo_id}"
    payload = build_export_payload(
        arxiv_id=arxiv_id,
        month=month,
        title=resolved_title,
        repo_id=used_repo,
        cfg=cfg,
        tensor_names=tensor_names,
        source=source,
        weights_url=weights_url,
    )
    write_export_artifacts(
        docs_dir,
        payload,
        patch_papers=patch_papers,
        refresh_shells=refresh_shells,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a local architecture graph into docs/ (no weight download).")
    parser.add_argument("--arxiv", required=True, help="arXiv id base, e.g. 2510.21585")
    parser.add_argument("--repo", required=True, help="Hugging Face owner/name or model URL")
    parser.add_argument("--month", required=True, help="Digest month YYYY-MM")
    parser.add_argument("--title", default=None, help="Override paper title")
    parser.add_argument("--docs-dir", default="docs", help="Site docs directory")
    parser.add_argument(
        "--fallback-repo",
        default=DEFAULT_FALLBACK_REPO,
        help="Public Hub repo to try when --repo is gated",
    )
    parser.add_argument("--no-patch-papers", action="store_true", help="Do not edit digest papers.json")
    parser.add_argument(
        "--refresh-shells",
        action="store_true",
        help="Rewrite all month HTML shells plus home/explore/models nav",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = export_architecture(
        arxiv_id=args.arxiv,
        repo=args.repo,
        month=args.month,
        docs_dir=Path(args.docs_dir),
        title=args.title,
        fallback_repo=args.fallback_repo or None,
        patch_papers=not args.no_patch_papers,
        refresh_shells=args.refresh_shells,
    )
    print(
        json.dumps(
            {
                "arxiv_id_base": payload["arxiv_id_base"],
                "hf_repo": payload["hf_repo"],
                "graph_path": payload["graph_path"],
                "source": payload["source"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
