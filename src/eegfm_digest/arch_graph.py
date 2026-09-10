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

_RMS_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen",
        "gemma",
        "gemma2",
        "gemma3",
        "phi3",
        "olmo",
        "olmo2",
        "reve",
        "stablelm",
        "cohere",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
        "grok",
    }
)
_ROPE_TYPES = _RMS_TYPES | {"gpt_neox", "phi"}
_GATED_TYPES = _RMS_TYPES


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
    return max(1, round(embed * ratio))


def _name_blob(tensor_names: list[str] | None) -> str:
    return " ".join(tensor_names or []).lower()


def _model_type(cfg: dict[str, Any]) -> str:
    return str(cfg.get("model_type") or "").strip().lower()


def _is_eeg_like(cfg: dict[str, Any], label: str | None) -> bool:
    if _model_type(cfg) == "reve":
        return True
    if looks_like_reve(cfg=cfg):
        return True
    if _first_int(cfg.get("patch_size")):
        return True
    return "eeg" in str(label or "").lower()


def _is_causal(cfg: dict[str, Any], blob: str, eeg_like: bool) -> bool:
    if eeg_like:
        return False
    architectures = cfg.get("architectures")
    if isinstance(architectures, list) and any(
        "causallm" in str(item).lower() or "lmhead" in str(item).lower() for item in architectures
    ):
        return True
    if "lm_head" in blob:
        return True
    return _model_type(cfg) in _ROPE_TYPES | {"gpt2", "gpt_neox", "phi", "opt"}


def _uses_rms(cfg: dict[str, Any], blob: str) -> bool:
    if _model_type(cfg) in _RMS_TYPES:
        return True
    if "rmsnorm" in blob or "rms_norm" in blob or "rms" in blob:
        return True
    return bool(cfg.get("rms_norm_eps"))


def _uses_rope(cfg: dict[str, Any], blob: str) -> bool:
    if cfg.get("rope_theta") is not None or cfg.get("rope_scaling") is not None:
        return True
    if _model_type(cfg) in _ROPE_TYPES:
        return True
    return "rotary" in blob


def _is_gated_ffn(cfg: dict[str, Any], blob: str) -> bool:
    if bool(cfg.get("use_geglu")) or bool(cfg.get("use_swiglu")):
        return True
    if any(token in blob for token in ("gate_proj", "up_proj", ".w1.", ".w3.", "geglu")):
        return True
    if _model_type(cfg) in _GATED_TYPES:
        return True
    act = str(_first_str(cfg.get("hidden_act"), cfg.get("hidden_activation"), cfg.get("activation")) or "").lower()
    return "geglu" in act or "swiglu" in act


def _gallery_activation(cfg: dict[str, Any], gated: bool) -> str:
    raw = str(_first_str(cfg.get("hidden_act"), cfg.get("hidden_activation"), cfg.get("activation")) or "").lower()
    if bool(cfg.get("use_geglu")) or "geglu" in raw or "gelu" in raw:
        return "GELU"
    if bool(cfg.get("use_swiglu")) or "swiglu" in raw or "silu" in raw or "swish" in raw:
        return "SiLU"
    if "relu" in raw:
        return "ReLU"
    if gated and _model_type(cfg) in _GATED_TYPES:
        return "SiLU"
    return "GELU"


def _ffn_title(cfg: dict[str, Any], gated: bool, activation: str) -> str:
    if bool(cfg.get("use_geglu")) or (gated and activation == "GELU"):
        return "FeedForward (GeGLU) module"
    if bool(cfg.get("use_swiglu")) or (gated and activation == "SiLU"):
        return "FeedForward (SwiGLU) module"
    return "FeedForward module"


def _attention_label(*, heads: int, kv_heads: int | None, causal: bool) -> str:
    kv = kv_heads if kv_heads is not None else heads
    if kv <= 0:
        kv = heads
    if kv == 1 and heads > 1:
        core = "multi-query attention"
    elif kv < heads:
        core = "grouped-query attention"
    else:
        core = "multi-head attention"
    if causal:
        return "Masked " + core
    return core[:1].upper() + core[1:]


def _context_note(length: int) -> str:
    if length >= 10_000 and length % 1000 == 0:
        return f"Supported context length\nof {length // 1000}k tokens"
    return f"Supported context length\nof {length:,} tokens"


def _ffn_width(cfg: dict[str, Any], embed: int) -> int | None:
    direct = _first_int(
        cfg.get("intermediate_size"),
        cfg.get("ffn_dim"),
        cfg.get("ffn_hidden_size"),
        cfg.get("n_inner"),
    )
    if direct:
        return direct
    if cfg.get("mlp_dim_ratio") is not None and embed:
        return _mlp_hidden_dim(embed, cfg.get("mlp_dim_ratio"))
    return None


def diagram_from_hf(
    cfg: dict[str, Any] | None,
    tensor_names: list[str] | None = None,
    *,
    label: str | None = None,
    num_params: int | None = None,
) -> dict[str, Any]:
    """Compile a Raschka-gallery diagram from transformers-style config + tensor names."""
    cfg = cfg if isinstance(cfg, dict) else {}
    blob = _name_blob(tensor_names)
    title = (label or _first_str(cfg.get("model_type")) or "Model").strip() or "Model"
    eeg_like = _is_eeg_like(cfg, title)
    causal = _is_causal(cfg, blob, eeg_like)
    gated = _is_gated_ffn(cfg, blob)
    rms = _uses_rms(cfg, blob)
    embed = (
        _first_int(cfg.get("hidden_size"), cfg.get("n_embd"), cfg.get("d_model"), cfg.get("n_embed"), cfg.get("embed_dim"))
        or 0
    )
    depth = (
        _first_int(
            cfg.get("num_hidden_layers"),
            cfg.get("n_layer"),
            cfg.get("n_layers"),
            cfg.get("num_layers"),
            cfg.get("depth"),
        )
        or 1
    )
    heads = _first_int(cfg.get("num_attention_heads"), cfg.get("n_head"), cfg.get("n_heads"), cfg.get("heads")) or 0
    kv_heads = _first_int(cfg.get("num_key_value_heads"), cfg.get("num_kv_heads"))
    head_dim = _first_int(cfg.get("head_dim"))
    if head_dim is None and embed and heads:
        head_dim = max(1, embed // heads)
    freqs = _first_int(cfg.get("freqs"))
    patch_size = _first_int(cfg.get("patch_size"))
    patch_overlap = _first_int(cfg.get("patch_overlap"))
    vocab = _first_int(cfg.get("vocab_size"))
    context = _first_int(
        cfg.get("max_position_embeddings"),
        cfg.get("n_positions"),
        cfg.get("max_seq_len"),
        cfg.get("max_sequence_length"),
        cfg.get("seq_length"),
    )
    params = _first_int(num_params, cfg.get("num_params"), cfg.get("n_params"))
    activation = _gallery_activation(cfg, gated)
    hidden = _ffn_width(cfg, embed)
    norm_label = "RMSNorm" if rms else "LayerNorm"
    has_pool = eeg_like or any(token in blob for token in ("pooler", ".pool.", "pooling", "avg_pool", "mean_pool"))
    has_patch = bool(patch_size) or "patch_embed" in blob or "patch_embedding" in blob
    has_wpe = "wpe" in blob or "wpe.weight" in blob

    if eeg_like:
        below_id, stem_id, prefix, attn_id, mlp_id = "eeg", "patch", "enc", "enc.attn", "enc.mlp"
        below_label = "Sample EEG"
        stem_label = "Patch embedding layer"
    else:
        below_id, stem_id, prefix, attn_id, mlp_id = "input", "embed", "block", "block.attn", "block.mlp"
        below_label = "Sample input"
        stem_label = "Token embedding layer"

    steps = [
        {"id": f"{prefix}.n1", "label": f"{norm_label} 1", "kind": "norm"},
        {
            "id": attn_id,
            "label": _attention_label(heads=heads or 1, kv_heads=kv_heads, causal=causal),
            "kind": "attention",
        },
        {"id": f"{prefix}.n2", "label": f"{norm_label} 2", "kind": "norm"},
        {"id": mlp_id, "label": "Feed forward", "kind": "ffn"},
        {"id": f"{prefix}.add", "label": "+", "kind": "add"},
    ]
    head: list[dict[str, Any]] = []
    if has_pool:
        head.append({"id": "pool", "label": "Pooling", "kind": "pool"})
    elif not eeg_like:
        head.append({"id": "final_norm", "label": f"Final {norm_label}", "kind": "norm"})
    head.append({"id": "out", "label": "Linear output layer", "kind": "linear"})

    callouts: list[dict[str, Any]] = [
        {
            "id": "ffn-mod",
            "kind": "ffn",
            "anchor": mlp_id,
            "title": _ffn_title(cfg, gated, activation),
            "activation": activation,
            "hidden_dim": hidden,
            "gated": gated,
        }
    ]
    if heads:
        heads_callout: dict[str, Any] = {
            "id": "attn-heads",
            "kind": "heads",
            "anchor": attn_id,
            "label": f"{heads} heads",
        }
        if head_dim:
            heads_callout["detail"] = f"Head dim {head_dim}"
        callouts.append(heads_callout)

    left: list[dict[str, Any]] = []
    if freqs:
        left.append({"id": "pe", "label": "Fourier PE", "anchor": attn_id})
    elif _uses_rope(cfg, blob):
        left.append({"id": "pe", "label": "RoPE", "anchor": attn_id})
    elif has_wpe:
        left.append({"id": "pe", "label": "Absolute PE", "anchor": stem_id})
    if patch_size:
        overlap_bit = f",\noverlap {patch_overlap}" if patch_overlap is not None else ""
        left.append(
            {
                "id": "patch-meta",
                "label": f"Patch size {patch_size}{overlap_bit}",
                "anchor": stem_id,
            }
        )
    if context and not eeg_like:
        left.append({"id": "context", "label": _context_note(context), "anchor": stem_id})

    annotations: dict[str, Any] = {"left": left}
    if embed:
        annotations["embed_dim"] = embed
    if vocab:
        annotations["vocab_size"] = vocab

    notes: dict[str, Any] = {}
    if freqs:
        notes["freqs"] = freqs
    if head_dim:
        notes["head_dim"] = head_dim

    return {
        "title": title,
        "param_label": _format_param_label(params),
        "below": [{"id": below_id, "label": below_label, "kind": "input"}],
        "stem": [{"id": stem_id, "label": stem_label, "kind": "embed"}],
        "repeat": {"id": prefix, "count": depth, "steps": steps},
        "head": head,
        "callouts": callouts,
        "annotations": annotations,
        "notes": notes,
        "meta": {"eeg_like": eeg_like, "causal": causal, "gated": gated, "has_patch": has_patch},
    }


def reve_diagram(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """REVE gallery diagram via the shared HF compiler."""
    merged = merge_reve_config(cfg)
    return diagram_from_hf(merged, label="REVE", num_params=_first_int(merged.get("num_params")))


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
    diagram = reve_diagram(merged)
    return {"nodes": nodes, "edges": edges, "diagram": diagram}


def graph_for_model(
    *,
    cfg: dict[str, Any] | None,
    tensor_names: list[str] | None = None,
    repo_id: str | None = None,
    arxiv_id: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    if looks_like_reve(repo_id, arxiv_id, cfg):
        return reve_graph(cfg)
    hidden = None
    num_layers = None
    params = None
    if isinstance(cfg, dict):
        hidden = _first_int(cfg.get("hidden_size"), cfg.get("n_embd"), cfg.get("d_model"), cfg.get("embed_dim"))
        num_layers = _first_int(
            cfg.get("num_hidden_layers"),
            cfg.get("n_layer"),
            cfg.get("n_layers"),
            cfg.get("num_layers"),
            cfg.get("depth"),
        )
        params = _first_int(cfg.get("num_params"), cfg.get("n_params"))
    graph = graph_from_tensors(tensor_names or [], hidden_size=hidden, num_layers=num_layers)
    graph["diagram"] = diagram_from_hf(
        cfg,
        tensor_names,
        label=label or short_model_label(None, repo_id),
        num_params=params,
    )
    return graph


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
