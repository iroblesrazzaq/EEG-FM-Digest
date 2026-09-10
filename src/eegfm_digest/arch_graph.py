"""Build nested architecture graphs from config and/or safetensors names.

Encoder/block stacks collapse to a single ``×N`` repeat node whose children
are the modules of one layer. Dedicated REVE layout is used when the config
or repo looks like REVE; otherwise names are grouped generically.
"""

from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

from .architecture import _first_int, _first_str, fact_sheet_from_config

# Published REVE-base (braindecode / paper) used when Hub config is gated.
REVE_BASE_PUBLISHED: dict[str, Any] = {
    "model_type": "reve",
    "architectures": ["Reve"],
    "embed_dim": 512,
    "hidden_size": 512,
    "depth": 22,
    "num_hidden_layers": 22,
    "heads": 8,
    "num_attention_heads": 8,
    "head_dim": 64,
    "mlp_dim_ratio": 2.66,
    "use_geglu": True,
    "freqs": 4,
    "patch_size": 200,
    "patch_overlap": 20,
    "num_params": 69_189_632,
}

_LAYER_RE = re.compile(
    r"^(?:(?P<prefix>.+)\.)?(?:layers|layer|blocks|block)\.(?P<idx>\d+)\.(?P<rest>.+)$"
)


def looks_like_reve(
    repo_id: str | None = None,
    arxiv_id: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> bool:
    if str(arxiv_id or "").strip() == "2510.21585":
        return True
    rid = str(repo_id or "").lower()
    if "reve" in rid:
        return True
    if isinstance(cfg, dict):
        model_type = str(cfg.get("model_type") or "").lower()
        if model_type == "reve":
            return True
        architectures = cfg.get("architectures")
        if isinstance(architectures, list) and any(str(item).lower() == "reve" for item in architectures):
            return True
    return False


def merge_reve_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(REVE_BASE_PUBLISHED)
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if value is not None:
                merged[key] = value
    return merged


def _shape_label(shape: Any) -> str | None:
    if isinstance(shape, str) and shape.strip():
        return shape.strip()
    if isinstance(shape, list | tuple) and shape:
        parts = ["B" if dim is None else str(dim) for dim in shape]
        return "(" + ", ".join(parts) + ")"
    return None


def _module_kind(name: str) -> str:
    key = name.lower()
    if any(token in key for token in ("attn", "attention", "mha", "self_attn")):
        return "attn"
    if any(token in key for token in ("mlp", "ffn", "geglu", "feed_forward", "feedforward")):
        return "mlp"
    if any(token in key for token in ("norm", "ln_", "rmsnorm", "layernorm")):
        return "norm"
    if "residual" in key or key in {"res", "skip"}:
        return "residual"
    if any(token in key for token in ("embed", "patch", "stem", "pos", "pe")):
        return "stem"
    if any(token in key for token in ("head", "pool", "proj_out", "lm_head")):
        return "head"
    return "module"


def _pretty_module(name: str) -> str:
    parts = [p for p in re.split(r"[._]", name) if p and p not in {"weight", "bias"}]
    if not parts:
        return name
    label = " ".join(parts[:3])
    return label[:1].upper() + label[1:]


def collapse_repeated_layers(tensor_names: list[str]) -> dict[str, Any]:
    """Collapse ``*.layers.{i}.*`` stacks into a repeat node plus leftover names."""
    grouped: dict[str, dict[int, list[str]]] = {}
    leftover: list[str] = []
    for name in tensor_names:
        match = _LAYER_RE.match(name)
        if not match:
            leftover.append(name)
            continue
        prefix = match.group("prefix") or "encoder"
        idx = int(match.group("idx"))
        rest = match.group("rest")
        grouped.setdefault(prefix, {}).setdefault(idx, []).append(rest)

    repeats: list[dict[str, Any]] = []
    for prefix, by_idx in grouped.items():
        indices = sorted(by_idx)
        if len(indices) < 2:
            for idx in indices:
                leftover.extend(f"{prefix}.layers.{idx}.{rest}" for rest in by_idx[idx])
            continue
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            for idx in indices:
                leftover.extend(f"{prefix}.layers.{idx}.{rest}" for rest in by_idx[idx])
            continue
        first_rests = by_idx[indices[0]]
        child_ids: list[str] = []
        children: list[dict[str, Any]] = []
        for rest in first_rests:
            child_key = rest.split(".", 1)[0]
            child_id = f"{prefix}.{child_key}"
            if child_id in child_ids:
                continue
            child_ids.append(child_id)
            children.append(
                {
                    "id": child_id,
                    "label": _pretty_module(child_key),
                    "kind": _module_kind(child_key),
                }
            )
        repeats.append(
            {
                "id": prefix,
                "label": _pretty_module(prefix),
                "kind": "repeat",
                "repeat": len(indices),
                "children": children,
            }
        )
    return {"repeats": repeats, "leftover": leftover}


def _format_param_label(count: int | None) -> str | None:
    if not isinstance(count, int) or count <= 0:
        return None
    if count >= 1_000_000_000:
        value = count / 1_000_000_000
        text = f"{value:.0f}B" if value >= 10 else f"{value:.1f}B"
        return text.replace(".0B", "B")
    if count >= 1_000_000:
        value = count / 1_000_000
        text = f"{value:.0f}M" if value >= 10 else f"{value:.1f}M"
        return text.replace(".0M", "M")
    if count >= 1_000:
        value = count / 1_000
        text = f"{value:.0f}K" if value >= 10 else f"{value:.1f}K"
        return text.replace(".0K", "K")
    return str(count)


def _mlp_hidden_dim(embed: int, mlp_ratio: Any) -> int:
    try:
        ratio = float(mlp_ratio)
    except (TypeError, ValueError):
        ratio = 2.66
    return max(1, int(round(embed * ratio)))


def reve_diagram(
    *,
    embed: int,
    depth: int,
    heads: int,
    head_dim: int,
    mlp_ratio: Any,
    use_geglu: bool,
    freqs: int,
    patch_size: int,
    patch_overlap: int,
    num_params: int | None,
) -> dict[str, Any]:
    """Sebastian Raschka gallery layout: bottom-up chassis, ×N block, side callouts."""
    hidden = _mlp_hidden_dim(embed, mlp_ratio)
    activation = "GELU" if use_geglu else "GELU"
    ffn_title = "FeedForward (GeGLU) module" if use_geglu else "FeedForward module"
    param_label = _format_param_label(num_params)
    return {
        "title": "REVE",
        "param_label": param_label,
        "below": [{"id": "eeg", "label": "Sample EEG", "kind": "input"}],
        "stem": [{"id": "patch", "label": "Patch embedding layer", "kind": "embed"}],
        "repeat": {
            "id": "enc",
            "count": depth,
            "steps": [
                {"id": "enc.n1", "label": "RMSNorm 1", "kind": "norm"},
                {"id": "enc.attn", "label": "Multi-head attention", "kind": "attention"},
                {"id": "enc.n2", "label": "RMSNorm 2", "kind": "norm"},
                {"id": "enc.mlp", "label": "Feed forward", "kind": "ffn"},
                {"id": "enc.add", "label": "+", "kind": "add"},
            ],
        },
        "head": [
            {"id": "pool", "label": "Pooling", "kind": "pool"},
            {"id": "out", "label": "Linear output layer", "kind": "linear"},
        ],
        "callouts": [
            {
                "id": "ffn-mod",
                "kind": "ffn",
                "anchor": "enc.mlp",
                "title": ffn_title,
                "activation": activation,
                "hidden_dim": hidden,
            },
            {
                "id": "attn-heads",
                "kind": "heads",
                "anchor": "enc.attn",
                "label": f"{heads} heads",
                "detail": f"Head dim {head_dim}",
            },
        ],
        "annotations": {
            "embed_dim": embed,
            "left": [
                {"id": "pe", "label": "Fourier PE", "anchor": "enc.attn"},
                {
                    "id": "patch-meta",
                    "label": f"Patch size {patch_size},\noverlap {patch_overlap}",
                    "anchor": "patch",
                },
            ],
        },
        "notes": {"freqs": freqs, "head_dim": head_dim},
    }


def graph_from_tensors(
    tensor_names: list[str],
    *,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    """Generic input → collapsed encoder → head graph from tensor names."""
    collapsed = collapse_repeated_layers(tensor_names)
    if num_layers:
        for node in collapsed["repeats"]:
            node["repeat"] = num_layers
    nodes: list[dict[str, Any]] = [
        {"id": "input", "label": "Input", "kind": "input"},
    ]
    leftover = collapsed["leftover"]
    stem_names = [
        name
        for name in leftover
        if _module_kind(name) in {"stem"} or any(tok in name.lower() for tok in ("patch", "embed", "pos"))
    ]
    if stem_names:
        nodes.append({"id": "stem", "label": "Stem / embeddings", "kind": "stem"})
    nodes.extend(collapsed["repeats"])
    head_names = [
        name
        for name in leftover
        if name not in stem_names and _module_kind(name) == "head"
    ]
    if head_names or hidden_size is not None:
        out_node: dict[str, Any] = {"id": "output", "label": "Output", "kind": "output"}
        if hidden_size is not None:
            out_node["shape"] = [None, hidden_size]
        nodes.append(out_node)
    elif leftover:
        nodes.append({"id": "output", "label": "Output", "kind": "output"})

    edges: list[dict[str, Any]] = []
    for left, right in pairwise(nodes):
        edge: dict[str, Any] = {"from": left["id"], "to": right["id"]}
        shape = _shape_label(right.get("shape"))
        if shape:
            edge["shape"] = shape
        edges.append(edge)
    return {"nodes": nodes, "edges": edges}


def reve_graph(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """REVE node graph plus a Raschka-gallery ``diagram`` for the SVG renderer."""
    merged = merge_reve_config(cfg)
    embed = _first_int(merged.get("embed_dim"), merged.get("hidden_size"), merged.get("d_model")) or 512
    depth = _first_int(merged.get("depth"), merged.get("num_hidden_layers"), merged.get("n_layer")) or 22
    heads = _first_int(merged.get("heads"), merged.get("num_attention_heads"), merged.get("n_head")) or 8
    head_dim = _first_int(merged.get("head_dim")) or max(1, embed // heads)
    mlp_ratio = merged.get("mlp_dim_ratio", 2.66)
    use_geglu = bool(merged.get("use_geglu", True))
    freqs = _first_int(merged.get("freqs")) or 4
    patch_size = _first_int(merged.get("patch_size")) or 200
    patch_overlap = _first_int(merged.get("patch_overlap")) or 20
    mlp_label = "GeGLU MLP" if use_geglu else "MLP"
    num_params = _first_int(merged.get("num_params"))

    nodes: list[dict[str, Any]] = [
        {
            "id": "eeg",
            "label": "EEG",
            "kind": "input",
            "shape": [None, "C", "T"],
            "detail": "multichannel time series",
        },
        {
            "id": "patch",
            "label": "Temporal patches",
            "kind": "stem",
            "detail": f"size {patch_size}, overlap {patch_overlap}",
        },
        {
            "id": "pe",
            "label": "4D Fourier PE",
            "kind": "posenc",
            "detail": f"freqs = {freqs}",
        },
        {
            "id": "enc",
            "label": "Transformer encoder",
            "kind": "repeat",
            "repeat": depth,
            "children": [
                {"id": "enc.n1", "label": "RMSNorm", "kind": "norm"},
                {"id": "enc.attn", "label": f"MHA {heads}×{head_dim}", "kind": "attn"},
                {"id": "enc.res1", "label": "Residual add", "kind": "residual"},
                {"id": "enc.n2", "label": "RMSNorm", "kind": "norm"},
                {"id": "enc.mlp", "label": mlp_label, "kind": "mlp", "detail": f"ratio {mlp_ratio}"},
                {"id": "enc.res2", "label": "Residual add", "kind": "residual"},
            ],
        },
        {"id": "pool", "label": "Pooling", "kind": "pool"},
        {
            "id": "out",
            "label": "Embedding",
            "kind": "output",
            "shape": [None, embed],
        },
    ]
    edges = [
        {"from": "eeg", "to": "patch", "shape": "(B, C, T)"},
        {"from": "patch", "to": "pe"},
        {"from": "pe", "to": "enc", "shape": f"(B, N, {embed})"},
        {"from": "enc", "to": "pool"},
        {"from": "pool", "to": "out", "shape": f"(B, {embed})"},
    ]
    diagram = reve_diagram(
        embed=embed,
        depth=depth,
        heads=heads,
        head_dim=head_dim,
        mlp_ratio=mlp_ratio,
        use_geglu=use_geglu,
        freqs=freqs,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        num_params=num_params,
    )
    return {"nodes": nodes, "edges": edges, "diagram": diagram}


def graph_for_model(
    *,
    cfg: dict[str, Any] | None,
    tensor_names: list[str] | None = None,
    repo_id: str | None = None,
    arxiv_id: str | None = None,
) -> dict[str, Any]:
    if looks_like_reve(repo_id, arxiv_id, cfg):
        return reve_graph(cfg)
    hidden = None
    num_layers = None
    if isinstance(cfg, dict):
        hidden = _first_int(cfg.get("hidden_size"), cfg.get("n_embd"), cfg.get("d_model"), cfg.get("embed_dim"))
        num_layers = _first_int(
            cfg.get("num_hidden_layers"),
            cfg.get("n_layer"),
            cfg.get("n_layers"),
            cfg.get("num_layers"),
            cfg.get("depth"),
        )
    if tensor_names:
        return graph_from_tensors(tensor_names, hidden_size=hidden, num_layers=num_layers)
    return graph_from_tensors([], hidden_size=hidden, num_layers=num_layers)


def fact_sheet_for_graph(cfg: dict[str, Any] | None) -> dict[str, Any]:
    sheet = fact_sheet_from_config(cfg or {})
    if sheet.get("hidden_size") is None and isinstance(cfg, dict):
        sheet["hidden_size"] = _first_int(cfg.get("embed_dim"))
    if sheet.get("num_layers") is None and isinstance(cfg, dict):
        sheet["num_layers"] = _first_int(cfg.get("depth"))
    if sheet.get("num_attention_heads") is None and isinstance(cfg, dict):
        sheet["num_attention_heads"] = _first_int(cfg.get("heads"))
    if sheet.get("architectures") is None and isinstance(cfg, dict):
        sheet["architectures"] = _first_str(cfg.get("architectures"))
    return sheet


def short_model_label(title: str | None, repo_id: str | None = None) -> str:
    text = str(title or "").strip()
    if text:
        head = text.split(":", 1)[0].split("—", 1)[0].split("--", 1)[0].strip()
        if head:
            return head
    if repo_id and "/" in repo_id:
        return repo_id.split("/", 1)[1]
    return str(repo_id or "model")
