from __future__ import annotations

from eegfm_digest.arch_graph import (
    collapse_repeated_layers,
    diagram_from_hf,
    graph_for_model,
    graph_from_tensors,
    looks_like_reve,
    merge_reve_config,
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


def test_graph_from_tensors_prefers_config_layer_count():
    graph = graph_from_tensors(_reve_like_names(8), hidden_size=512, num_layers=32)
    encoder = next(node for node in graph["nodes"] if node["id"] == "encoder")
    assert encoder["repeat"] == 32


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


def test_reve_diagram_matches_raschka_gallery_structure():
    graph = reve_graph(None)
    diagram = graph["diagram"]
    assert diagram["title"] == "REVE"
    assert diagram["param_label"] == "69M"
    assert [item["id"] for item in diagram["below"]] == ["eeg"]
    assert [item["id"] for item in diagram["stem"]] == ["patch"]
    assert diagram["repeat"]["count"] == 22
    assert [step["kind"] for step in diagram["repeat"]["steps"]] == [
        "norm",
        "attention",
        "norm",
        "ffn",
        "add",
    ]
    assert [item["label"] for item in diagram["head"]] == ["Pooling", "Linear output layer"]
    kinds = {item["kind"] for item in diagram["callouts"]}
    assert kinds == {"ffn", "heads"}
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["activation"] == "GELU"
    assert ffn["hidden_dim"] == 1362
    assert "GeGLU" in ffn["title"]
    assert ffn.get("gated") is True
    assert diagram["annotations"]["embed_dim"] == 512
    left = {item["id"]: item["label"] for item in diagram["annotations"]["left"]}
    assert left["pe"] == "Fourier PE"
    assert diagram["family"] == "Fourier-PE transformer"


def _llama_cfg() -> dict:
    return {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "hidden_act": "silu",
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "num_params": 8_000_000_000,
    }


def test_diagram_from_hf_llama_gqa_swiglu_rope():
    diagram = diagram_from_hf(_llama_cfg(), label="Llama")
    attn = next(step for step in diagram["repeat"]["steps"] if step["kind"] == "attention")
    assert attn["label"] == "Masked grouped-query attention"
    assert diagram["family"] == "Autoregressive transformer"
    assert diagram["repeat"]["steps"][0]["label"].startswith("RMSNorm")
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["activation"] == "SiLU"
    assert ffn["gated"] is True
    assert ffn["hidden_dim"] == 14336
    assert "SwiGLU" in ffn["title"]
    assert diagram["annotations"]["vocab_size"] == 128256
    assert diagram["annotations"]["embed_dim"] == 4096
    left_labels = [item["label"] for item in diagram["annotations"]["left"]]
    assert "RoPE" in left_labels
    assert any("8,192" in label for label in left_labels)
    assert diagram["below"][0]["label"] == "Sample input"
    assert diagram["stem"][0]["label"] == "Token embedding layer"
    head_labels = [item["label"] for item in diagram["head"]]
    assert "Final RMSNorm" in head_labels
    assert "Linear output layer" in head_labels


def test_diagram_from_hf_gpt2_ungated_gelu_layernorm():
    cfg = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "n_inner": 3072,
        "hidden_act": "gelu",
        "vocab_size": 50257,
        "n_positions": 1024,
    }
    names = ["wte.weight", "wpe.weight", "h.0.mlp.c_fc.weight", "lm_head.weight"]
    diagram = diagram_from_hf(cfg, names, label="GPT-2")
    attn = next(step for step in diagram["repeat"]["steps"] if step["kind"] == "attention")
    assert attn["label"] == "Masked multi-head attention"
    assert diagram["repeat"]["steps"][0]["label"].startswith("LayerNorm")
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["activation"] == "GELU"
    assert ffn["gated"] is False
    assert ffn["hidden_dim"] == 3072
    assert "GeGLU" not in ffn["title"]
    assert "SwiGLU" not in ffn["title"]
    assert "2 layers" in ffn["title"]
    left_labels = [item["label"] for item in diagram["annotations"]["left"]]
    assert "Absolute PE" in left_labels
    assert "RoPE" not in left_labels
    assert not any("grouped-query" in step["label"] for step in diagram["repeat"]["steps"])


def test_reve_published_defaults_go_through_compiler():
    diagram = diagram_from_hf(merge_reve_config(None), label="REVE")
    assert diagram["repeat"]["count"] == 22
    assert diagram["title"] == "REVE"
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["hidden_dim"] == 1362
    assert "GeGLU" in ffn["title"]


def test_graph_for_model_llama_attaches_diagram_without_reve():
    cfg = _llama_cfg()
    graph = graph_for_model(
        cfg=cfg,
        tensor_names=["model.embed_tokens.weight", "lm_head.weight"],
        repo_id="meta-llama/Llama-3-8B",
        arxiv_id="2401.00001",
        label="Llama",
    )
    assert not looks_like_reve("meta-llama/Llama-3-8B", "2401.00001", cfg)
    assert "diagram" in graph
    assert graph["diagram"]["title"] == "Llama"
    assert graph["diagram"]["callouts"][0]["hidden_dim"] == 14336
    ids = [node["id"] for node in graph["nodes"]]
    assert "eeg" not in ids


def test_diagram_from_hf_zuna_nested_config():
    cfg = {
        "model": {
            "dim": 1024,
            "n_layers": 16,
            "head_dim": 64,
            "rope_dim": 4,
            "rope_theta": 10000.0,
            "max_chans": 512,
        },
        "num_params": 382_106_752,
    }
    diagram = diagram_from_hf(cfg, label="ZUNA")
    assert diagram["repeat"]["count"] == 16
    assert diagram["annotations"]["embed_dim"] == 1024
    attn = next(step for step in diagram["repeat"]["steps"] if step["kind"] == "attention")
    assert attn["label"] == "Multi-head attention"
    left = [item["label"] for item in diagram["annotations"]["left"]]
    assert "4D RoPE" in left
    heads = next(item for item in diagram["callouts"] if item["kind"] == "heads")
    assert heads["label"] == "16 heads"
    assert diagram["below"][0]["label"] == "Sample EEG"
    assert diagram["family"] == "Denoising transformer"


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
    assert short_model_label(
        "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI",
        "braindecode/labram-pretrained",
        "2405.18765",
    ) == "LaBraM"
    assert (
        short_model_label(
            "BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training",
            "braindecode/eegpt-pretrained",
            "2410.19779",
        )
        == "BrainGPT"
    )


def test_diagram_from_hf_braingpt_is_autoregressive_with_electrode_stem():
    cfg = {"n_chans": 62, "n_times": 1024, "input_window_seconds": 4}
    tensors = {
        "chans_id": {"shape": [1, 62]},
        "target_encoder.patch_embed.proj.weight": {"shape": [512, 1, 1, 64]},
        "target_encoder.chan_embed.weight": {"shape": [62, 512]},
        "target_encoder.blocks.0.attn.qkv.weight": {"shape": [1536, 512]},
        "target_encoder.blocks.0.mlp.fc1.weight": {"shape": [2048, 512]},
        "target_encoder.blocks.1.mlp.fc1.weight": {"shape": [2048, 512]},
        "target_encoder.blocks.7.mlp.fc1.weight": {"shape": [2048, 512]},
        "target_encoder.summary_token": {"shape": [1, 4, 512]},
    }
    diagram = diagram_from_hf(
        cfg,
        tensors=tensors,
        label="BrainGPT",
        arxiv_id="2410.19779",
        title="BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training",
    )
    assert diagram["title"] == "BrainGPT"
    assert diagram["family"] == "Autoregressive transformer"
    assert diagram["repeat"]["count"] == 8
    assert diagram["annotations"]["embed_dim"] == 512
    attn = next(step for step in diagram["repeat"]["steps"] if step["kind"] == "attention")
    assert attn["label"] == "Masked multi-head attention"
    assert [item["label"] for item in diagram["stem"]] == [
        "Patch embedding layer",
        "Electrode embedding",
    ]
    assert [item["label"] for item in diagram["head"]] == ["Next-token head"]
    left = {item["id"]: item["label"] for item in diagram["annotations"]["left"]}
    assert left["causal-mask"] == "Causal mask\nnext-token"
    assert left["chan-meta"] == "Electrode-wise"
    assert diagram["meta"]["causal"] is True
    assert diagram["notes"]["objective"] == "autoregressive"
    heads = next(item for item in diagram["callouts"] if item["kind"] == "heads")
    assert heads["label"] == "8 heads"


def test_diagram_from_hf_zuna_tensors_infer_swiglu_and_q_out_heads():
    cfg = {
        "model": {
            "dim": 1024,
            "n_layers": 16,
            "head_dim": 64,
            "rope_dim": 4,
            "rope_theta": 10000.0,
            "max_chans": 512,
        },
        "num_params": 382_106_752,
    }
    tensors = {
        "model.decoder.layers.0.attention.wq.weight": {"shape": [512, 1024]},
        "model.decoder.layers.0.attention_norm.weight": {"shape": [1024]},
        "model.decoder.layers.0.feed_forward.w1.weight": {"shape": [2816, 1024]},
        "model.decoder.layers.0.feed_forward.w2.weight": {"shape": [1024, 2816]},
        "model.decoder.layers.0.feed_forward.w3.weight": {"shape": [2816, 1024]},
        "model.decoder.layers.1.feed_forward.w1.weight": {"shape": [2816, 1024]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="ZUNA")
    heads = next(item for item in diagram["callouts"] if item["kind"] == "heads")
    assert heads["label"] == "8 heads"
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["activation"] == "SiLU"
    assert ffn["gated"] is True
    assert ffn["hidden_dim"] == 2816
    assert "SwiGLU" in ffn["title"]
    assert diagram["repeat"]["steps"][0]["label"].startswith("RMSNorm")
    assert diagram["annotations"]["embed_dim"] == 1024


def test_diagram_from_hf_labram_tensors_infer_heads_pe_and_mlp_width():
    cfg = {"n_chans": 19, "n_times": 800, "patch_size": 200}
    tensors = {
        "patch_embed.weight": {"shape": [200, 1, 200]},
        "position_embedding.weight": {"shape": [256, 200]},
        "cls_token": {"shape": [1, 1, 200]},
        "temporal_embedding": {"shape": [1, 16, 200]},
        "blocks.0.attn.qkv.weight": {"shape": [600, 200]},
        "blocks.0.attn.q_norm.weight": {"shape": [20]},
        "blocks.0.attn.k_norm.weight": {"shape": [20]},
        "blocks.0.mlp.0.weight": {"shape": [800, 200]},
        "blocks.1.mlp.0.weight": {"shape": [800, 200]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="LaBraM")
    heads = next(item for item in diagram["callouts"] if item["kind"] == "heads")
    assert heads["label"] == "10 heads"
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["gated"] is False
    assert ffn["activation"] == "GELU"
    assert ffn["hidden_dim"] == 800
    assert [item["label"] for item in diagram["stem"]] == ["Patch embedding layer", "VQ-VAE codebook"]
    left = [item["label"] for item in diagram["annotations"]["left"]]
    assert "Absolute PE" in left
    assert "Frozen codebook" in left
    assert "QK-Norm" in left
    assert diagram["annotations"]["embed_dim"] == 200
    assert diagram["notes"]["tokenizer"] == "vqvae"
    assert diagram["family"] == "VQ-tokenized transformer"


def test_diagram_from_hf_cbramod_prefers_ffn_width_over_spatial_attn():
    cfg = {"n_chans": 22, "n_times": 1000}
    tensors = {
        "patch_embedding.weight": {"shape": [100, 22, 16]},
        "encoder.layers.0.self_attn_s.in_proj_weight": {"shape": [300, 100]},
        "encoder.layers.0.linear1.weight": {"shape": [800, 200]},
        "encoder.layers.1.linear1.weight": {"shape": [800, 200]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="CBraMod")
    assert diagram["annotations"]["embed_dim"] == 200
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["hidden_dim"] == 800
    assert ffn["gated"] is False
    assert diagram["repeat"]["count"] == 2


def test_diagram_from_hf_cbramod_criss_cross_attention():
    cfg = {"n_chans": 22, "n_times": 1000}
    tensors = {
        "patch_embedding.positional_encoding.0.weight": {"shape": [200, 1, 19, 7]},
        "encoder.layers.0.self_attn_s.in_proj_weight": {"shape": [300, 100]},
        "encoder.layers.0.self_attn_t.in_proj_weight": {"shape": [300, 100]},
        "encoder.layers.0.linear1.weight": {"shape": [800, 200]},
        "encoder.layers.1.self_attn_s.in_proj_weight": {"shape": [300, 100]},
        "encoder.layers.1.self_attn_t.in_proj_weight": {"shape": [300, 100]},
        "encoder.layers.1.linear1.weight": {"shape": [800, 200]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="CBraMod")
    attn_labels = [step["label"] for step in diagram["repeat"]["steps"] if step["kind"] == "attention"]
    assert attn_labels == ["Criss-cross attention"]
    left = {item["id"]: item["label"] for item in diagram["annotations"]["left"]}
    assert left["cost-st"] == "O(N²T) ∥ O(NT²)"
    assert left["pe"] == "Asymmetric PE"
    assert left["ffn-kind"] == "GELU\n2-layer MLP"
    assert diagram["notes"]["attn"] == "criss_cross"
    assert "Spatial attention" not in [step["label"] for step in diagram["repeat"]["steps"]]
    assert "Temporal attention" not in [step["label"] for step in diagram["repeat"]["steps"]]
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["gated"] is False
    assert ffn["layers"] == 2
    assert "2 layers" in ffn["title"]
    assert diagram["family"] == "Criss-cross transformer"


def test_diagram_from_hf_luna_channel_unifier_and_rope():
    cfg = {"model_type": "luna"}
    tensors = {
        "patch_embed.proj_in.0.weight": {"shape": [16, 1, 1, 19]},
        "channel_emb": {"shape": [1, 64]},
        "channel_location_embedder.weight": {"shape": [64, 3]},
        "cross_attn.0.q_proj.weight": {"shape": [256, 64]},
        "freq_embed.frequency_to_embed.fc1.weight": {"shape": [168, 42]},
        "cross_attn.query_self_attn.layers.0.linear1.weight": {"shape": [256, 64]},
        "blocks.0.attn.qkv_proj.weight": {"shape": [768, 256]},
        "blocks.0.attn.rotary_emb.freqs": {"shape": [16]},
        "blocks.0.mlp.fc1.weight": {"shape": [1024, 256]},
        "blocks.1.mlp.fc1.weight": {"shape": [1024, 256]},
        "blocks.7.mlp.fc1.weight": {"shape": [1024, 256]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="LUNA")
    assert diagram["repeat"]["count"] == 8
    assert diagram["annotations"]["embed_dim"] == 256
    assert [item["label"] for item in diagram["stem"]] == [
        "Patch embedding layer",
        "Channel unifier",
    ]
    left = {item["id"]: item["label"] for item in diagram["annotations"]["left"]}
    assert left["pe"] == "RoPE"
    assert left["unify-meta"] == "Learned queries"
    assert diagram["family"] == "Channel-query transformer"
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["hidden_dim"] == 1024
    assert ffn["gated"] is False


def test_diagram_from_hf_brainomni_lm_aliases_and_codebook():
    cfg = {
        "n_neuro": 16,
        "n_dim": 256,
        "n_head": 4,
        "codebook_size": 512,
        "num_quantizers": 4,
        "lm_dim": 512,
        "lm_head": 16,
        "lm_depth": 12,
    }
    diagram = diagram_from_hf(cfg, label="BrainOmni")
    assert diagram["repeat"]["count"] == 12
    assert diagram["annotations"]["embed_dim"] == 512
    heads = next(item for item in diagram["callouts"] if item["kind"] == "heads")
    assert heads["label"] == "16 heads"
    assert [item["label"] for item in diagram["stem"]] == [
        "Sensor encoder",
        "VQ-VAE codebook",
    ]
    assert diagram["notes"]["tokenizer"] == "vqvae"
    left = {item["id"]: item["label"] for item in diagram["annotations"]["left"]}
    assert left["vq-meta"] == "Frozen codebook"
    assert diagram["family"] == "Sensor-encoder transformer"
    ffn = next(item for item in diagram["callouts"] if item["kind"] == "ffn")
    assert ffn["gated"] is False
    assert "2 layers" in ffn["title"]


def test_diagram_from_hf_femba_bidirectional_mamba():
    cfg = {"model_type": "femba"}
    tensors = {
        "patch_embed.proj.weight": {"shape": [385, 1, 1, 16]},
        "mamba_blocks.0.mamba_fwd.in_proj.weight": {"shape": [3080, 385]},
        "mamba_blocks.0.mamba_rev.out_proj.weight": {"shape": [385, 1540]},
        "mamba_blocks.1.mamba_fwd.in_proj.weight": {"shape": [3080, 385]},
        "mamba_blocks.9.mamba_fwd.in_proj.weight": {"shape": [3080, 385]},
    }
    diagram = diagram_from_hf(cfg, tensors=tensors, label="FEMBA")
    assert diagram["repeat"]["count"] == 10
    assert [step["kind"] for step in diagram["repeat"]["steps"]] == ["norm", "attention", "add"]
    assert diagram["repeat"]["steps"][1]["label"] == "Bidirectional Mamba"
    assert diagram["annotations"]["embed_dim"] == 385
    assert diagram["notes"]["backbone"] == "mamba"
    assert diagram["family"] == "Bidirectional Mamba"
    assert not any(item["kind"] == "ffn" for item in diagram["callouts"])
