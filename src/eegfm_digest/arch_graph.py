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
    r"^(?:(?P<prefix>.+)\.)?(?:layers|layer|mamba_blocks|blocks|block)\.(?P<idx>\d+)\.(?P<rest>.+)$"
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

GALLERY_LABELS = {
    "2405.18765": "LaBraM",
    "2410.19779": "BrainGPT",
    "2412.07236": "CBraMod",
    "2502.06438": "FEMBA",
    "2505.18185": "BrainOmni",
    "2510.21585": "REVE",
    "2510.22257": "LUNA",
    "2607.27308": "ZUNA1.1",
}
CAUSAL_ARXIV_IDS = frozenset({"2410.19779"})


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


def _flatten_hf_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Promote nested ``model`` / ``encoder`` blocks used by EEG Hub configs."""
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg)
    for key in ("model", "encoder", "backbone", "transformer", "text_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict):
            for nested_key, value in nested.items():
                out.setdefault(nested_key, value)
    if out.get("mlp_ratio") is not None and out.get("mlp_dim_ratio") is None:
        out["mlp_dim_ratio"] = out.get("mlp_ratio")
    if out.get("dim") is not None and out.get("hidden_size") is None:
        out["hidden_size"] = out.get("dim")
    if out.get("lm_dim") is not None:
        out["hidden_size"] = out.get("lm_dim")
    if out.get("lm_head") is not None:
        out["num_attention_heads"] = out.get("lm_head")
    if out.get("lm_depth") is not None:
        out["num_hidden_layers"] = out.get("lm_depth")
    return out


def _is_backbone_tensor(name: str) -> bool:
    """True for the main repeated trunk, not nested query/decoder stacks."""
    match = _LAYER_RE.match(name)
    if not match:
        return False
    prefix = (match.group("prefix") or "").strip(".").lower()
    if not prefix:
        return True
    if any(token in prefix for token in ("cross_attn", "query", "decoder_head", "classifier")):
        return False
    head = prefix.split(".")[-1]
    if head in {"encoder", "model", "transformer", "backbone", "decoder"}:
        return True
    return head.endswith("encoder") or head.endswith("decoder")


def _depth_from_names(tensor_names: list[str] | None) -> int | None:
    indices: set[int] = set()
    for name in tensor_names or []:
        if not _is_backbone_tensor(name):
            continue
        match = _LAYER_RE.match(name)
        if match:
            indices.add(int(match.group("idx")))
    if len(indices) >= 2:
        return max(indices) - min(indices) + 1
    return None


def _embed_from_tensors(tensors: dict[str, dict[str, Any]] | None) -> int | None:
    if not isinstance(tensors, dict):
        return None
    patch = None
    ffn_in = None
    attn = None
    for name, info in tensors.items():
        if not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            continue
        dims = [int(dim) for dim in shape if isinstance(dim, int) and dim > 1]
        if len(dims) < 2:
            continue
        key = name.lower()
        stacked = _is_backbone_tensor(name)
        if any(token in key for token in ("patch_embed", "patch_embedding", "embed_tokens", "wte")):
            patch = max(patch or 0, min(dims))
        if stacked and any(
            token in key for token in ("mlp.0.weight", "linear1.weight", "mlp.fc1.weight", "w1.weight")
        ):
            ffn_in = min(dims)
        if stacked and (
            "qkv.weight" in key
            or "qkv_proj.weight" in key
            or key.endswith(("wq.weight", "q_proj.weight"))
        ):
            attn = min(dims)
        if stacked and "mamba" in key and key.endswith(("in_proj.weight", "out_proj.weight")):
            attn = min(dims)
    return ffn_in or attn or patch


def _hidden_from_tensors(tensors: dict[str, dict[str, Any]] | None) -> int | None:
    if not isinstance(tensors, dict):
        return None
    for name, info in tensors.items():
        key = name.lower()
        shape = info.get("shape") if isinstance(info, dict) else None
        if not isinstance(shape, list) or not shape:
            continue
        if _is_backbone_tensor(name) and any(
            token in key for token in ("mlp.0.weight", "linear1.weight", "mlp.fc1.weight", "w1.weight")
        ):
            return int(shape[0])
    return None


def _attn_from_tensors(
    tensors: dict[str, dict[str, Any]] | None,
) -> tuple[int | None, int | None, int | None]:
    if not isinstance(tensors, dict):
        return None, None, None
    head_dim = None
    q_out = None
    for name, info in tensors.items():
        if not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        key = name.lower()
        if not _is_backbone_tensor(name):
            continue
        if "q_norm" in key or "k_norm" in key:
            head_dim = int(shape[-1])
        if key.endswith(("wq.weight", "q_proj.weight")):
            q_out = int(shape[0])
        elif "qkv.weight" in key or "qkv_proj.weight" in key:
            q_out = int(shape[0]) // 3
    heads = None
    if q_out and head_dim:
        heads = max(1, q_out // head_dim)
    return heads, head_dim, q_out


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
    if _first_int(cfg.get("max_chans"), cfg.get("n_chans"), cfg.get("n_neuro")):
        return True
    blob = f"{label or ''} {_model_type(cfg)} {cfg.get('tok_idx_type') or ''}".lower()
    return any(
        token in blob
        for token in (
            "eeg",
            "zuna",
            "labram",
            "cbramod",
            "braindecode",
            "luna",
            "brainomni",
            "csbrain",
            "eegpt",
            "braingpt",
            "femba",
        )
    )


def _is_causal(
    cfg: dict[str, Any],
    blob: str,
    eeg_like: bool,
    *,
    label: str | None = None,
    title: str | None = None,
    arxiv_id: str | None = None,
) -> bool:
    if str(arxiv_id or "").strip() in CAUSAL_ARXIV_IDS:
        return True
    text = f"{label or ''} {title or ''}".lower()
    if any(token in text for token in ("autoregressive", "next-token", "next_token", "braingpt")):
        return True
    if cfg.get("is_causal") is True:
        return True
    architectures = cfg.get("architectures")
    if isinstance(architectures, list) and any(
        "causallm" in str(item).lower() or "lmhead" in str(item).lower() for item in architectures
    ):
        return True
    if "lm_head" in blob:
        return True
    if eeg_like:
        return False
    return _model_type(cfg) in _ROPE_TYPES | {"gpt2", "gpt_neox", "phi", "opt"}


def _uses_rms(cfg: dict[str, Any], blob: str) -> bool:
    if _model_type(cfg) in _RMS_TYPES:
        return True
    if "rmsnorm" in blob or "rms_norm" in blob or "rms" in blob:
        return True
    if "attention_norm" in blob and any(
        token in blob for token in (".w1.", "feed_forward.w1", "w3.weight")
    ):
        return True
    return bool(cfg.get("rms_norm_eps"))


def _uses_rope(cfg: dict[str, Any], blob: str) -> bool:
    if cfg.get("rope_theta") is not None or cfg.get("rope_scaling") is not None or cfg.get("rope_dim"):
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


def _gallery_activation(cfg: dict[str, Any], gated: bool, blob: str = "") -> str:
    raw = str(_first_str(cfg.get("hidden_act"), cfg.get("hidden_activation"), cfg.get("activation")) or "").lower()
    if bool(cfg.get("use_geglu")) or "geglu" in raw or "gelu" in raw:
        return "GELU"
    if bool(cfg.get("use_swiglu")) or "swiglu" in raw or "silu" in raw or "swish" in raw:
        return "SiLU"
    if "relu" in raw:
        return "ReLU"
    if gated and any(token in blob for token in (".w1.", "w3.weight", "feed_forward.w1")):
        return "SiLU"
    if gated and _model_type(cfg) in _GATED_TYPES:
        return "SiLU"
    return "GELU"


def _ffn_title(cfg: dict[str, Any], gated: bool, activation: str) -> str:
    if bool(cfg.get("use_geglu")) or (gated and activation == "GELU"):
        return "FeedForward (GeGLU) module"
    if bool(cfg.get("use_swiglu")) or (gated and activation == "SiLU"):
        return "FeedForward (SwiGLU) module"
    if activation:
        return f"FeedForward ({activation} · 2 layers)"
    return "FeedForward (2-layer MLP)"


def _attention_label(*, heads: int, kv_heads: int | None, causal: bool, eeg_like: bool = False) -> str:
    kv = kv_heads if kv_heads is not None else heads
    if kv <= 0:
        kv = heads
    if kv == 1 and heads > 1:
        core = "multi-query attention"
    elif kv < heads:
        core = "grouped-query attention"
    else:
        core = "multi-head attention"
    if causal and eeg_like and core == "multi-head attention":
        return "Causal attention"
    if causal:
        return "Masked " + core
    return core[:1].upper() + core[1:]


def _family_label(
    *,
    mamba: bool,
    bidirectional_mamba: bool,
    criss_cross: bool,
    causal: bool,
    vq: bool,
    channel_unify: bool,
    sensor: bool,
    freqs: bool,
    title: str,
) -> str:
    if mamba:
        return "Bidirectional Mamba" if bidirectional_mamba else "Mamba state-space model"
    if criss_cross:
        return "Criss-cross transformer"
    if causal:
        return "Autoregressive transformer"
    if sensor:
        return "Sensor-encoder transformer"
    if vq:
        return "VQ-tokenized transformer"
    if channel_unify:
        return "Channel-query transformer"
    if freqs:
        return "Fourier-PE transformer"
    low = title.lower()
    if "denois" in low or "super-resolution" in low or "zuna" in low:
        return "Denoising transformer"
    return "Bidirectional transformer"


def _is_mamba(blob: str) -> bool:
    return "mamba" in blob


def _is_criss_cross(blob: str) -> bool:
    spatial = any(token in blob for token in ("self_attn_s", "spatial_attn", ".attn_s."))
    temporal = any(token in blob for token in ("self_attn_t", "temporal_attn", ".attn_t."))
    return spatial and temporal


_STACK_ROLE_ORDER = {"encoder": 0, "query": 1, "decoder": 2}
_STACK_LABELS = {"encoder": "Encoder", "decoder": "Decoder", "query": "Channel unifier"}
_GALLERY_STACK_IDS = {"encoder": "enc", "decoder": "dec", "query": "qry"}


def _is_denoise(label: str | None, title: str | None, arxiv_id: str | None = None) -> bool:
    if str(arxiv_id or "").strip() == "2607.27308":
        return True
    text = f"{label or ''} {title or ''}".lower()
    return any(token in text for token in ("denois", "super-resolution", "superresolution", "zuna"))


def _stack_role(prefix: str, children: list[dict[str, Any]] | None = None) -> str:
    p = (prefix or "").strip(".").lower()
    child_blob = " ".join(
        f"{item.get('id', '')} {item.get('label', '')}" for item in (children or [])
    ).lower()
    if any(token in p for token in ("query", "unifier")) and "decoder" not in p:
        return "query"
    if "cross_attn" in p and "decoder" not in p:
        return "query"
    if "decoder" in p:
        return "decoder"
    if any(token in p for token in ("encoder", "backbone", "mamba")):
        return "encoder"
    if "cross_attention" in child_blob or (".cross_attn" in child_blob and "query" not in p):
        return "decoder"
    return "encoder"


def _stacks_from_names(tensor_names: list[str] | None) -> list[dict[str, Any]]:
    """Group ``*.layers.{i}.*`` / ``*.blocks.{i}.*`` prefixes into encoder/query/decoder stacks."""
    grouped: dict[str, dict[int, list[str]]] = {}
    for name in tensor_names or []:
        match = _LAYER_RE.match(name)
        if not match:
            continue
        prefix = match.group("prefix") or "encoder"
        idx = int(match.group("idx"))
        rest = match.group("rest")
        grouped.setdefault(prefix, {}).setdefault(idx, []).append(rest)
    stacks: list[dict[str, Any]] = []
    for prefix, by_idx in grouped.items():
        indices = sorted(by_idx)
        if not indices:
            continue
        count = max(indices) - min(indices) + 1 if len(indices) >= 2 else 1
        if count < 2:
            continue
        first_rests = by_idx[indices[0]]
        children: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rest in first_rests:
            key = rest.split(".", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            children.append(
                {
                    "id": f"{prefix}.{key}",
                    "label": _pretty_module(key),
                    "kind": _module_kind(key),
                }
            )
        stacks.append(
            {
                "prefix": prefix,
                "count": count,
                "children": children,
                "role": _stack_role(prefix, children),
            }
        )
    stacks.sort(key=lambda item: (_STACK_ROLE_ORDER.get(item["role"], 9), item["prefix"]))
    return stacks


def _block_steps(
    prefix: str,
    *,
    role: str,
    children: list[dict[str, Any]] | None,
    norm_label: str,
    mixer_label: str,
    has_ffn: bool,
    mamba: bool,
) -> list[dict[str, Any]]:
    """One Pre-LN (or Mamba) cartoon; decoder stacks insert cross-attention."""
    has_cross = role == "decoder" or any(
        "cross_attention" in str(item.get("id") or "").lower()
        for item in (children or [])
    )
    if role == "query":
        has_cross = False
    steps: list[dict[str, Any]] = [{"id": f"{prefix}.n1", "label": f"{norm_label} 1", "kind": "norm"}]
    if mamba and role == "encoder":
        steps.append({"id": f"{prefix}.attn", "label": mixer_label, "kind": "attention"})
        steps.append({"id": f"{prefix}.add", "label": "+", "kind": "add"})
        return steps
    steps.append({"id": f"{prefix}.attn", "label": mixer_label, "kind": "attention"})
    next_norm = 2
    if has_cross:
        steps.extend(
            [
                {"id": f"{prefix}.n{next_norm}", "label": f"{norm_label} {next_norm}", "kind": "norm"},
                {"id": f"{prefix}.cross", "label": "Cross attention", "kind": "attention"},
            ]
        )
        next_norm += 1
    if has_ffn:
        steps.extend(
            [
                {"id": f"{prefix}.n{next_norm}", "label": f"{norm_label} {next_norm}", "kind": "norm"},
                {"id": f"{prefix}.mlp", "label": "Feed forward", "kind": "ffn"},
            ]
        )
    steps.append({"id": f"{prefix}.add", "label": "+", "kind": "add"})
    return steps


def _mixer_for_role(
    role: str,
    *,
    mamba: bool,
    bidirectional_mamba: bool,
    criss_cross: bool,
    causal: bool,
    eeg_like: bool,
    heads: int,
    kv_heads: int | None,
) -> str:
    if role == "query":
        return "Query attention"
    if mamba and role == "encoder":
        return "Bidirectional Mamba" if bidirectional_mamba else "Mamba"
    if criss_cross and role == "encoder":
        return "Criss-cross attention"
    return _attention_label(
        heads=heads or 1, kv_heads=kv_heads, causal=causal and role == "encoder", eeg_like=eeg_like
    )


def _uses_vq_codebook(label: str | None, blob: str, cfg: dict[str, Any] | None = None) -> bool:
    if isinstance(cfg, dict) and _first_int(cfg.get("codebook_size"), cfg.get("num_quantizers")):
        return True
    text = f"{label or ''} {blob}".lower()
    if any(token in text for token in ("codebook", "quantize", "vqvae", "vq_vae", "neural_tokenizer")):
        return True
    if "labram" in text:
        return True
    return "cls_token" in blob and "temporal_embedding" in blob


def _uses_qk_norm(blob: str) -> bool:
    return "q_norm" in blob and "k_norm" in blob


def _uses_acpe(blob: str) -> bool:
    return "positional_encoding" in blob and "patch_embedding" in blob


def _context_note(length: int) -> str:
    if length >= 10_000 and length % 1000 == 0:
        return f"Supported context length\nof {length // 1000}k tokens"
    return f"Supported context length\nof {length:,} tokens"


def _ffn_width(cfg: dict[str, Any], embed: int, tensors: dict[str, dict[str, Any]] | None = None) -> int | None:
    direct = _first_int(
        cfg.get("intermediate_size"),
        cfg.get("ffn_dim"),
        cfg.get("ffn_hidden_size"),
        cfg.get("n_inner"),
    )
    if direct:
        return direct
    from_tensors = _hidden_from_tensors(tensors)
    if from_tensors:
        return from_tensors
    if cfg.get("mlp_dim_ratio") is not None and embed:
        return _mlp_hidden_dim(embed, cfg.get("mlp_dim_ratio"))
    return None


def diagram_from_hf(
    cfg: dict[str, Any] | None,
    tensor_names: list[str] | None = None,
    *,
    tensors: dict[str, dict[str, Any]] | None = None,
    label: str | None = None,
    num_params: int | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Compile a Raschka-gallery diagram from transformers-style config + tensor names."""
    cfg = _flatten_hf_cfg(cfg)
    if tensor_names is None and isinstance(tensors, dict):
        tensor_names = list(tensors.keys())
    blob = _name_blob(tensor_names)
    display = (label or _first_str(cfg.get("model_type")) or "Model").strip() or "Model"
    paper_title = (title or "").strip()
    eeg_like = _is_eeg_like(cfg, f"{display} {paper_title}")
    causal = _is_causal(
        cfg, blob, eeg_like, label=display, title=paper_title, arxiv_id=arxiv_id
    )
    gated = _is_gated_ffn(cfg, blob)
    rms = _uses_rms(cfg, blob)
    embed = (
        _first_int(
            cfg.get("hidden_size"),
            cfg.get("n_embd"),
            cfg.get("d_model"),
            cfg.get("n_embed"),
            cfg.get("embed_dim"),
            cfg.get("dim"),
        )
        or _embed_from_tensors(tensors)
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
        or _depth_from_names(tensor_names)
        or 1
    )
    heads = _first_int(cfg.get("num_attention_heads"), cfg.get("n_head"), cfg.get("n_heads"), cfg.get("heads")) or 0
    kv_heads = _first_int(cfg.get("num_key_value_heads"), cfg.get("num_kv_heads"))
    head_dim = _first_int(cfg.get("head_dim"))
    inferred_heads, inferred_head_dim, q_out = _attn_from_tensors(tensors)
    if head_dim is None:
        head_dim = inferred_head_dim
    if heads == 0:
        heads = inferred_heads or 0
    if heads == 0 and q_out and head_dim is None and q_out % 64 == 0:
        head_dim = 64
    if heads == 0 and q_out and head_dim:
        heads = max(1, q_out // head_dim)
    if heads == 0 and embed and head_dim:
        heads = max(1, embed // head_dim)
    if head_dim is None and embed and heads:
        head_dim = max(1, embed // heads)
    freqs = _first_int(cfg.get("freqs"))
    rope_dim = _first_int(cfg.get("rope_dim"))
    patch_size = _first_int(cfg.get("patch_size"))
    patch_overlap = _first_int(cfg.get("patch_overlap"))
    vocab = _first_int(cfg.get("vocab_size"))
    context = _first_int(
        cfg.get("max_position_embeddings"),
        cfg.get("n_positions"),
        cfg.get("max_seq_len"),
        cfg.get("max_sequence_length"),
        cfg.get("seq_length"),
        cfg.get("max_seqlen"),
        cfg.get("n_times"),
    )
    params = _first_int(num_params, cfg.get("num_params"), cfg.get("n_params"))
    activation = _gallery_activation(cfg, gated, blob)
    hidden = _ffn_width(cfg, embed, tensors)
    norm_label = "RMSNorm" if rms else "LayerNorm"
    denoise = eeg_like and _is_denoise(display, paper_title, arxiv_id)
    has_pool = (not causal) and (not denoise) and (
        eeg_like or any(token in blob for token in ("pooler", ".pool.", "pooling", "avg_pool", "mean_pool"))
    )
    has_patch = bool(patch_size) or "patch_embed" in blob or "patch_embedding" in blob
    has_wpe = "wpe" in blob or "wpe.weight" in blob or "position_embedding" in blob

    if eeg_like:
        below_id, stem_id, prefix = "eeg", "patch", "enc"
        below_label = "Noisy EEG" if denoise else "Sample EEG"
        stem_label = "Sensor encoder" if _first_int(cfg.get("n_neuro")) else "Patch embedding layer"
    else:
        below_id, stem_id, prefix = "input", "embed", "block"
        below_label = "Sample input"
        stem_label = "Token embedding layer"

    criss_cross = _is_criss_cross(blob)
    mamba = _is_mamba(blob)
    has_ffn = (not mamba) or any(
        token in blob for token in ("mlp.", ".linear1.", ".fc1.", "feed_forward", "w1.weight")
    )
    vq_codebook = _uses_vq_codebook(display, blob, cfg)
    qk_norm = _uses_qk_norm(blob)
    acpe = _uses_acpe(blob)
    channel_unify = eeg_like and (
        "channel_location_embedder" in blob
        or "channel_emb" in blob
        or ("cross_attn" in blob and "channel" in blob)
    )
    electrode_wise = "chan_embed" in blob or "chans_id" in blob
    bidirectional_mamba = mamba and ("mamba_fwd" in blob and "mamba_rev" in blob)
    sensor = bool(_first_int(cfg.get("n_neuro")))
    detected = _stacks_from_names(tensor_names)
    if not detected:
        detected = [{"prefix": prefix, "count": depth, "children": [], "role": "encoder"}]
    multi = len(detected) > 1
    gallery_stacks: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for item in detected:
        role = str(item.get("role") or "encoder")
        gallery_id = _GALLERY_STACK_IDS.get(role, "enc") if eeg_like else "block"
        if gallery_id in used_ids:
            gallery_id = f"{gallery_id}{len(used_ids)}"
        used_ids.add(gallery_id)
        mixer_label = _mixer_for_role(
            role,
            mamba=mamba,
            bidirectional_mamba=bidirectional_mamba,
            criss_cross=criss_cross,
            causal=causal,
            eeg_like=eeg_like,
            heads=heads,
            kv_heads=kv_heads,
        )
        if mamba and role == "encoder":
            stack_has_ffn = False
        elif role == "query":
            child_blob = " ".join(str(child.get("id") or "") for child in item.get("children") or []).lower()
            stack_has_ffn = any(
                token in child_blob for token in ("mlp", "linear", "feed_forward", "fc1")
            )
        else:
            stack_has_ffn = has_ffn
        name_count = int(item.get("count") or 1)
        count = max(name_count, depth) if role in {"encoder", "decoder"} and depth else name_count
        stack: dict[str, Any] = {
            "id": gallery_id,
            "count": count,
            "role": role,
            "steps": _block_steps(
                gallery_id,
                role=role,
                children=list(item.get("children") or []),
                norm_label=norm_label,
                mixer_label=mixer_label,
                has_ffn=stack_has_ffn,
                mamba=mamba,
            ),
        }
        if multi:
            stack["label"] = _STACK_LABELS.get(role, role[:1].upper() + role[1:])
        gallery_stacks.append(stack)
    primary = next((item for item in gallery_stacks if item.get("role") == "encoder"), gallery_stacks[0])
    prefix = str(primary["id"])
    attn_id, mlp_id = f"{prefix}.attn", f"{prefix}.mlp"
    steps = list(primary["steps"])
    has_query_stack = any(item.get("role") == "query" for item in gallery_stacks)
    has_decoder_stack = any(item.get("role") == "decoder" for item in gallery_stacks)
    stem = [{"id": stem_id, "label": stem_label, "kind": "embed"}]
    if vq_codebook:
        stem.append({"id": "vq", "label": "VQ-VAE codebook", "kind": "embed"})
    if electrode_wise:
        stem.append({"id": "chan", "label": "Electrode embedding", "kind": "embed"})
    head: list[dict[str, Any]] = []
    if causal and eeg_like:
        head.append({"id": "out", "label": "Next-token head", "kind": "linear"})
    elif denoise:
        head.append({"id": "out", "label": "Reconstruction head", "kind": "linear"})
    else:
        if has_pool:
            head.append({"id": "pool", "label": "Pooling", "kind": "pool"})
        elif not eeg_like:
            head.append({"id": "final_norm", "label": f"Final {norm_label}", "kind": "norm"})
        head.append({"id": "out", "label": "Linear output layer", "kind": "linear"})

    callouts: list[dict[str, Any]] = []
    if any(step.get("kind") == "ffn" for step in steps):
        callouts.append(
            {
                "id": "ffn-mod",
                "kind": "ffn",
                "anchor": mlp_id,
                "title": _ffn_title(cfg, gated, activation),
                "activation": activation,
                "hidden_dim": hidden,
                "gated": gated,
                "layers": 1 if gated else 2,
            }
        )
    if heads and not mamba:
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
    elif rope_dim and rope_dim >= 4:
        left.append({"id": "pe", "label": f"{rope_dim}D RoPE", "anchor": attn_id})
    elif _uses_rope(cfg, blob):
        left.append({"id": "pe", "label": "RoPE", "anchor": attn_id})
    elif acpe:
        left.append({"id": "pe", "label": "Asymmetric PE", "anchor": stem_id})
    elif has_wpe:
        left.append({"id": "pe", "label": "Absolute PE", "anchor": stem_id})
    if qk_norm:
        left.append({"id": "qk-norm", "label": "QK-Norm", "anchor": attn_id})
    if criss_cross:
        left.append({"id": "cost-st", "label": "O(N²T) ∥ O(NT²)", "anchor": attn_id})
    if causal:
        left.append({"id": "causal-mask", "label": "Causal mask\nnext-token", "anchor": attn_id})
    if has_ffn and any(step.get("kind") == "ffn" for step in steps):
        if gated:
            glu = "GeGLU" if activation == "GELU" else "SwiGLU"
            left.append({"id": "ffn-kind", "label": f"{glu}\n1 layer", "anchor": mlp_id})
        else:
            left.append({"id": "ffn-kind", "label": f"{activation}\n2-layer MLP", "anchor": mlp_id})
    if bidirectional_mamba:
        left.append({"id": "mamba-dir", "label": "Forward ∥ Reverse", "anchor": attn_id})
    if vq_codebook:
        left.append({"id": "vq-meta", "label": "Frozen codebook", "anchor": "vq"})
    if electrode_wise:
        left.append({"id": "chan-meta", "label": "Electrode-wise", "anchor": "chan"})
    if sensor:
        left.append({"id": "sensor-meta", "label": "EEG + MEG", "anchor": stem_id})
    if channel_unify:
        unify_anchor = "qry.attn" if has_query_stack else "unify"
        if not has_query_stack:
            stem.append({"id": "unify", "label": "Channel unifier", "kind": "embed"})
        left.append({"id": "unify-meta", "label": "Learned queries", "anchor": unify_anchor})
    if has_decoder_stack:
        left.append({"id": "cross-meta", "label": "Attend encoder", "anchor": "dec.cross"})
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
    if criss_cross:
        notes["attn"] = "criss_cross"
    if mamba:
        notes["backbone"] = "mamba"
    if vq_codebook:
        notes["tokenizer"] = "vqvae"
    if causal:
        notes["objective"] = "autoregressive"
    family = _family_label(
        mamba=mamba,
        bidirectional_mamba=bidirectional_mamba,
        criss_cross=criss_cross,
        causal=causal,
        vq=vq_codebook,
        channel_unify=channel_unify,
        sensor=sensor,
        freqs=bool(freqs),
        title=f"{display} {paper_title}",
    )
    notes["family"] = family
    if denoise and not causal:
        notes["objective"] = "denoise"
    if multi:
        notes["stacks"] = [str(item.get("role") or "") for item in gallery_stacks]

    repeat: dict[str, Any] = {"id": prefix, "count": int(primary["count"]), "steps": steps}
    if primary.get("label"):
        repeat["label"] = primary["label"]
    if primary.get("role"):
        repeat["role"] = primary["role"]

    return {
        "title": display,
        "family": family,
        "param_label": _format_param_label(params),
        "below": [{"id": below_id, "label": below_label, "kind": "input"}],
        "stem": stem,
        "repeat": repeat,
        "stacks": gallery_stacks,
        "head": head,
        "callouts": callouts,
        "annotations": annotations,
        "notes": notes,
        "meta": {
            "eeg_like": eeg_like,
            "causal": causal,
            "gated": gated,
            "has_patch": has_patch,
            "denoise": denoise,
        },
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
    repeats = sorted(
        collapsed["repeats"],
        key=lambda node: (
            _STACK_ROLE_ORDER.get(_stack_role(str(node.get("id") or ""), list(node.get("children") or [])), 9),
            str(node.get("id") or ""),
        ),
    )
    stem_names = [
        name
        for name in leftover
        if _module_kind(name) in {"stem"} or any(tok in name.lower() for tok in ("patch", "embed", "pos"))
    ]
    if stem_names:
        nodes.append({"id": "stem", "label": "Stem / embeddings", "kind": "stem"})
    nodes.extend(repeats)
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
    tensors: dict[str, dict[str, Any]] | None = None,
    repo_id: str | None = None,
    arxiv_id: str | None = None,
    label: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if looks_like_reve(repo_id, arxiv_id, cfg):
        return reve_graph(cfg)
    if tensor_names is None and isinstance(tensors, dict):
        tensor_names = list(tensors.keys())
    flat = _flatten_hf_cfg(cfg)
    hidden = _first_int(
        flat.get("hidden_size"),
        flat.get("n_embd"),
        flat.get("d_model"),
        flat.get("embed_dim"),
        flat.get("dim"),
    ) or _embed_from_tensors(tensors)
    num_layers = _first_int(
        flat.get("num_hidden_layers"),
        flat.get("n_layer"),
        flat.get("n_layers"),
        flat.get("num_layers"),
        flat.get("depth"),
    ) or _depth_from_names(tensor_names)
    params = _first_int(flat.get("num_params"), flat.get("n_params"))
    graph = graph_from_tensors(tensor_names or [], hidden_size=hidden, num_layers=num_layers)
    graph["diagram"] = diagram_from_hf(
        cfg,
        tensor_names,
        tensors=tensors,
        label=label or short_model_label(title, repo_id, arxiv_id),
        num_params=params,
        arxiv_id=arxiv_id,
        title=title,
    )
    return graph


def fact_sheet_for_graph(cfg: dict[str, Any] | None) -> dict[str, Any]:
    sheet = fact_sheet_from_config(_flatten_hf_cfg(cfg))
    if sheet.get("hidden_size") is None and isinstance(cfg, dict):
        sheet["hidden_size"] = _first_int(cfg.get("embed_dim"), cfg.get("dim"))
    if sheet.get("num_layers") is None and isinstance(cfg, dict):
        sheet["num_layers"] = _first_int(cfg.get("depth"), cfg.get("n_layers"))
    if sheet.get("num_attention_heads") is None and isinstance(cfg, dict):
        sheet["num_attention_heads"] = _first_int(cfg.get("heads"))
    if sheet.get("architectures") is None and isinstance(cfg, dict):
        sheet["architectures"] = _first_str(cfg.get("architectures"))
    return sheet


def short_model_label(title: str | None, repo_id: str | None = None, arxiv_id: str | None = None) -> str:
    known = GALLERY_LABELS.get(str(arxiv_id or "").strip())
    if known:
        return known
    text = str(title or "").strip()
    if text:
        head = text.split(":", 1)[0].split("—", 1)[0].split("--", 1)[0].strip()
        if head:
            return head
    if repo_id and "/" in repo_id:
        return repo_id.split("/", 1)[1]
    return str(repo_id or "model")
