from __future__ import annotations

from eegfm_digest.arch_graph import (
    collapse_repeated_layers,
    graph_from_tensors,
    looks_like_reve,
    reve_graph,
    short_model_label,
)


def _reve_like_names(depth: int = 22) -> list[str]:
    names = ["patch_embed.proj.weight", "pos_embed.weight"]
    for idx in range(depth):
        names.extend(
            [
                f"encoder.layers.{idx}.norm1.weight",
                f"encoder.layers.{idx}.attn.qkv.weight",
                f"encoder.layers.{idx}.attn.proj.weight",
                f"encoder.layers.{idx}.norm2.weight",
                f"encoder.layers.{idx}.mlp.fc1.weight",
                f"encoder.layers.{idx}.mlp.fc2.weight",
            ]
        )
    names.append("head.weight")
    return names


def test_collapse_repeated_layers_uses_times_n():
    collapsed = collapse_repeated_layers(_reve_like_names(22))
    assert len(collapsed["repeats"]) == 1
    repeat = collapsed["repeats"][0]
    assert repeat["repeat"] == 22
    assert repeat["kind"] == "repeat"
    kinds = {child["id"]: child["kind"] for child in repeat["children"]}
    assert kinds["encoder.attn"] == "attn"
    assert kinds["encoder.mlp"] == "mlp"
    assert kinds["encoder.norm1"] == "norm"
    leftover = collapsed["leftover"]
    assert "patch_embed.proj.weight" in leftover
    assert not any("encoder.layers." in name for name in leftover)


def test_graph_from_tensors_collapses_encoder_stack():
    graph = graph_from_tensors(_reve_like_names(8), hidden_size=512)
    ids = [node["id"] for node in graph["nodes"]]
    assert ids[0] == "input"
    assert "encoder" in ids
    encoder = next(node for node in graph["nodes"] if node["id"] == "encoder")
    assert encoder["repeat"] == 8
    assert encoder["children"]
    assert graph["nodes"][-1]["shape"] == [None, 512]


def test_reve_graph_overview_and_expandable_encoder():
    graph = reve_graph(None)
    ids = [node["id"] for node in graph["nodes"]]
    assert ids == ["eeg", "patch", "pe", "enc", "pool", "out"]
    enc = next(node for node in graph["nodes"] if node["id"] == "enc")
    assert enc["repeat"] == 22
    assert enc["kind"] == "repeat"
    child_labels = [child["label"] for child in enc["children"]]
    assert "RMSNorm" in child_labels
    assert any("MHA" in label for label in child_labels)
    assert any("GeGLU" in label for label in child_labels)
    shapes = [edge.get("shape") for edge in graph["edges"]]
    assert "(B, C, T)" in shapes
    assert "(B, 512)" in shapes


def test_looks_like_reve_from_arxiv_and_repo():
    assert looks_like_reve("brain-bzh/reve-base", "2510.21585", None)
    assert looks_like_reve("reve-model/reve", None, None)
    assert looks_like_reve("org/other", None, {"model_type": "reve"})
    assert not looks_like_reve("org/other", "2501.00001", {"model_type": "llama"})


def test_short_model_label_uses_title_head():
    assert short_model_label(
        "REVE: A Foundation Model for EEG -- Adapting to Any Setup",
        "brain-bzh/reve-base",
    ) == "REVE"
