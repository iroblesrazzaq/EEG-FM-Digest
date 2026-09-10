from __future__ import annotations

import json

import httpx
import pytest

from eegfm_digest.arch_safetensors import (
    MAX_HEADER_SIZE,
    RangeIgnoredError,
    SafetensorsHeaderError,
    fetch_safetensors_header,
    parse_safetensors_header,
)


def encode_header(tensors: dict[str, list[int]]) -> bytes:
    meta = {
        name: {"dtype": "F32", "shape": shape, "data_offsets": [0, 4]}
        for name, shape in tensors.items()
    }
    raw = json.dumps(meta).encode("utf-8")
    return len(raw).to_bytes(8, "little") + raw


def test_parse_safetensors_header_reads_shapes():
    blob = encode_header(
        {
            "encoder.layers.0.attn.qkv.weight": [768, 768],
            "head.weight": [512],
        }
    )
    tensors = parse_safetensors_header(blob)
    assert tensors["encoder.layers.0.attn.qkv.weight"]["shape"] == [768, 768]
    assert tensors["head.weight"]["dtype"] == "F32"


def test_parse_safetensors_header_rejects_huge_length():
    blob = (MAX_HEADER_SIZE + 1).to_bytes(8, "little")
    with pytest.raises(SafetensorsHeaderError, match="header_too_large"):
        parse_safetensors_header(blob)


def test_fetch_safetensors_header_uses_range(monkeypatch):
    blob = encode_header({"emb.weight": [4, 4]})
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Range", ""))
        rng = request.headers.get("Range", "")
        if rng == "bytes=0-7":
            return httpx.Response(206, content=blob[:8])
        if rng.startswith("bytes=8-"):
            return httpx.Response(206, content=blob[8:])
        return httpx.Response(500, content=b"nope")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        tensors = fetch_safetensors_header("https://huggingface.co/org/m/resolve/main/model.safetensors", client=client)
    assert tensors["emb.weight"]["shape"] == [4, 4]
    assert calls[0] == "bytes=0-7"
    assert calls[1].startswith("bytes=8-")


def test_fetch_safetensors_header_aborts_when_range_ignored():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 32,
            headers={"Content-Length": "80000000"},
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client, pytest.raises(RangeIgnoredError):
        fetch_safetensors_header(
            "https://huggingface.co/org/m/resolve/main/model.safetensors",
            client=client,
        )


def test_fetch_safetensors_header_caps_body_without_content_length():
    blob = encode_header({"emb.weight": [2, 2]}) + (b"w" * 5000)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=blob)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client, pytest.raises(RangeIgnoredError):
        fetch_safetensors_header(
            "https://huggingface.co/org/m/resolve/main/model.safetensors",
            client=client,
        )
