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
# Digest papers whose Hub owner/name is known even when papers.json has no weights_url.
KNOWN_DIGEST_HF_REPOS = {
    "2510.22257": "PulpBio/LUNA",
    "2505.18185": "OpenTSLab/BrainOmni",
    "2410.19779": "braindecode/eegpt-pretrained",
    "2502.06438": "PulpBio/FEMBA",
}


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
    path: str = "config.json",
) -> dict[str, Any] | None:
    rel = str(path or "config.json").lstrip("/")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{rel}"
    try:
        response = client.get(url, headers=hub_headers(token), timeout=timeout)
        if response.status_code in {401, 403, 404}:
            return None
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 — local export should fall back
        return None


def _config_rank(name: str) -> tuple[int, int, str]:
    low = name.lower()
    depth = name.count("/")
    if name == "config.json":
        return (0, 0, name)
    if "tokenizer" in low:
        return (8, depth, name)
    if "classifier" in low:
        return (9, depth, name)
    if name.startswith("base/") or "/base/" in name:
        return (1, depth, name)
    if name.endswith("config.json"):
        return (2, depth, name)
    if name.endswith("model_cfg.json"):
        return (3, depth, name)
    return (5, depth, name)


def config_filenames(info: dict[str, Any] | None) -> list[str]:
    names = ["config.json"]
    if isinstance(info, dict):
        siblings = info.get("siblings")
        if isinstance(siblings, list):
            for item in siblings:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("rfilename") or "").strip()
                if name.endswith(("config.json", "model_cfg.json")):
                    names.append(name)
    ranked = sorted(dict.fromkeys(names), key=_config_rank)
    return ranked


def _safetensors_rank(name: str) -> tuple[int, int, int, str]:
    low = name.lower()
    base = name.split("/")[-1]
    sharded = 1 if "-of-" in base else 0
    depth = name.count("/")
    if "base" in low:
        size = 0
    elif "tiny" in low:
        size = 1
    elif "small" in low:
        size = 2
    elif "large" in low:
        size = 3
    elif "huge" in low or "xl" in low:
        size = 4
    else:
        size = 0
    return (sharded, depth, size, name)


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
    names.sort(key=_safetensors_rank)
    return names


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
    for name in candidates[:6]:
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
        candidate_cfg = None
        for cfg_path in config_filenames(info):
            candidate_cfg = fetch_hf_config_optional(
                candidate, client=client, token=token, path=cfg_path
            )
            if candidate_cfg is not None:
                break
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
    tensors: dict[str, dict[str, Any]] | None = None,
    source: str,
    weights_url: str | None = None,
) -> dict[str, Any]:
    label = short_model_label(title, repo_id, arxiv_id)
    graph = graph_for_model(
        cfg=cfg,
        tensor_names=tensor_names,
        tensors=tensors,
        repo_id=repo_id,
        arxiv_id=arxiv_id,
        label=label,
    )
    fact_sheet = fact_sheet_for_graph(cfg)
    graph_path = f"{GRAPH_REL_PREFIX}/{arxiv_id}.json"
    payload: dict[str, Any] = {
        "arxiv_id_base": arxiv_id,
        "title": title,
        "label": label,
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
    if fallback_repo == DEFAULT_FALLBACK_REPO and not looks_like_reve(repo_id, arxiv_id, None):
        fallback_repo = None
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
        tensors=tensors,
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


def paper_hf_repo(row: dict[str, Any]) -> str | None:
    if not isinstance(row, dict):
        return None
    architecture = row.get("architecture")
    if isinstance(architecture, dict):
        parsed = parse_hf_repo_id(str(architecture.get("hf_repo") or ""))
        if parsed:
            return parsed
    for blob in (row.get("summary"), row):
        if not isinstance(blob, dict):
            continue
        open_source = blob.get("open_source")
        if isinstance(open_source, dict):
            parsed = parse_hf_repo_id(str(open_source.get("weights_url") or ""))
            if parsed:
                return parsed
    arxiv_id = str(row.get("arxiv_id_base") or "").strip()
    known = KNOWN_DIGEST_HF_REPOS.get(arxiv_id)
    if known:
        return parse_hf_repo_id(known) or known
    return None


def paper_title(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    summary = row.get("summary")
    if isinstance(summary, dict):
        title = str(summary.get("title") or title).strip() or title
    return title


def iter_digest_hf_papers(docs_dir: Path) -> list[dict[str, str]]:
    """Papers whose weights_url or architecture.hf_repo is an owner/name Hub repo."""
    digest = Path(docs_dir) / "digest"
    if not digest.is_dir():
        return []
    found: list[dict[str, str]] = []
    for month_dir in sorted(path for path in digest.iterdir() if path.is_dir()):
        if not month_dir.name[:4].isdigit():
            continue
        papers_path = month_dir / "papers.json"
        if not papers_path.exists():
            continue
        try:
            payload = json.loads(papers_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        papers = payload.get("papers") if isinstance(payload, dict) else payload
        if not isinstance(papers, list):
            continue
        for row in papers:
            if not isinstance(row, dict):
                continue
            arxiv_id = str(row.get("arxiv_id_base") or "").strip()
            repo = paper_hf_repo(row)
            if not arxiv_id or not repo:
                continue
            found.append(
                {
                    "month": month_dir.name,
                    "arxiv_id": arxiv_id,
                    "repo": repo,
                    "title": paper_title(row) or arxiv_id,
                }
            )
    return found


def export_all_digest(
    docs_dir: Path,
    *,
    token: str | None = None,
    client: httpx.Client | None = None,
    refresh_shells: bool = False,
    exporter: Any = None,
) -> dict[str, Any]:
    """Export gallery graphs for every digest paper with a public owner/name Hub repo.

    Not hooked from daily CI. Skips org pages, gated repos, and missing configs.
    """
    docs_dir = Path(docs_dir)
    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    run_one = exporter or export_architecture
    for item in iter_digest_hf_papers(docs_dir):
        fallback = (
            DEFAULT_FALLBACK_REPO if looks_like_reve(item["repo"], item["arxiv_id"], None) else None
        )
        try:
            payload = run_one(
                arxiv_id=item["arxiv_id"],
                repo=item["repo"],
                month=item["month"],
                docs_dir=docs_dir,
                title=item.get("title"),
                fallback_repo=fallback,
                token=token,
                client=client,
                patch_papers=True,
                refresh_shells=False,
            )
        except Exception as exc:  # noqa: BLE001 — batch continues on gated/missing
            skipped.append(
                {
                    "arxiv_id": item["arxiv_id"],
                    "month": item["month"],
                    "repo": item["repo"],
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        exported.append(
            {
                "arxiv_id": item["arxiv_id"],
                "month": item["month"],
                "hf_repo": payload.get("hf_repo"),
                "graph_path": payload.get("graph_path"),
                "source": payload.get("source"),
            }
        )
    if refresh_shells:
        refresh_html_shells(docs_dir)
    return {"exported": exported, "skipped": skipped}


def _print_skip_table(skipped: list[dict[str, Any]]) -> None:
    if not skipped:
        print("skipped: (none)", file=sys.stderr)
        return
    print("skipped:", file=sys.stderr)
    print(f"{'arxiv':<16} {'month':<8} {'repo':<36} reason", file=sys.stderr)
    for row in skipped:
        print(
            f"{row.get('arxiv_id', ''):<16} {row.get('month', ''):<8} "
            f"{row.get('repo', ''):<36} {row.get('reason', '')}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export architecture gallery graphs into docs/ (config + safetensors names, not daily CI)."
    )
    parser.add_argument(
        "--all-digest",
        action="store_true",
        help="Scan docs/digest/*/papers.json and export every public owner/name Hub repo",
    )
    parser.add_argument("--arxiv", default=None, help="arXiv id base")
    parser.add_argument("--repo", default=None, help="Hugging Face owner/name")
    parser.add_argument("--month", default=None, help="Digest month YYYY-MM")
    parser.add_argument("--title", default=None, help="Override paper title")
    parser.add_argument("--docs-dir", default="docs", help="Site docs directory")
    parser.add_argument(
        "--fallback-repo",
        default=DEFAULT_FALLBACK_REPO,
        help="Public Hub repo to try when --repo is gated (ignored for non-REVE unless overridden)",
    )
    parser.add_argument("--no-patch-papers", action="store_true", help="Do not edit digest papers.json")
    parser.add_argument(
        "--refresh-shells",
        action="store_true",
        help="Rewrite all month HTML shells plus home/explore/Model Gallery nav",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    docs_dir = Path(args.docs_dir)
    if args.all_digest:
        if args.arxiv or args.repo or args.month:
            parser.error("--all-digest cannot be combined with --arxiv, --repo, or --month")
        result = export_all_digest(docs_dir, refresh_shells=args.refresh_shells)
        _print_skip_table(result.get("skipped") or [])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not args.arxiv or not args.repo or not args.month:
        parser.error("--arxiv, --repo, and --month are required unless --all-digest")
    payload = export_architecture(
        arxiv_id=args.arxiv,
        repo=args.repo,
        month=args.month,
        docs_dir=docs_dir,
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
