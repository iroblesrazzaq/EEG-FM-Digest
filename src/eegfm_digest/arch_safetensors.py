"""Parse safetensors JSON headers without downloading weight tensors.

A safetensors file starts with an 8-byte little-endian header length, then a
JSON object mapping tensor names to dtype/shape/offsets. Callers must Range-GET
only those prefix bytes. If a server ignores Range and returns a large body,
we abort rather than buffering the blob.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .architecture import HF_USER_AGENT

HEADER_LEN_BYTES = 8
MAX_HEADER_SIZE = 2_000_000
# First probe is 8 bytes. If Range is ignored, refuse anything larger than the
# header budget plus a small slack so we never hold a checkpoint in memory.
MAX_RANGE_BODY = MAX_HEADER_SIZE + HEADER_LEN_BYTES + 64


class SafetensorsHeaderError(ValueError):
    """Header is missing, truncated, or larger than the allowed budget."""


class RangeIgnoredError(RuntimeError):
    """HTTP server returned a full file (or a huge body) instead of a Range."""


def parse_safetensors_header(blob: bytes) -> dict[str, dict[str, Any]]:
    """Return ``{tensor_name: {shape, dtype}}`` from a header prefix.

    ``blob`` may be the 8-byte length plus JSON, or a slightly longer prefix.
    Raises ``SafetensorsHeaderError`` if the header is invalid or too large.
    """
    if len(blob) < HEADER_LEN_BYTES:
        raise SafetensorsHeaderError("truncated_header_len")
    header_len = int.from_bytes(blob[:HEADER_LEN_BYTES], "little", signed=False)
    if header_len <= 0 or header_len > MAX_HEADER_SIZE:
        raise SafetensorsHeaderError("header_too_large")
    end = HEADER_LEN_BYTES + header_len
    if len(blob) < end:
        raise SafetensorsHeaderError("truncated_header")
    try:
        meta = json.loads(blob[HEADER_LEN_BYTES:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsHeaderError("invalid_header_json") from exc
    if not isinstance(meta, dict):
        raise SafetensorsHeaderError("invalid_header_json")
    tensors: dict[str, dict[str, Any]] = {}
    for name, info in meta.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        shape = info.get("shape")
        if not isinstance(shape, list):
            continue
        tensors[str(name)] = {
            "shape": [int(dim) if isinstance(dim, int) else dim for dim in shape],
            "dtype": info.get("dtype"),
        }
    return tensors


def _auth_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": HF_USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _reject_if_range_ignored(response: httpx.Response, expected_max: int) -> None:
    content_len = response.headers.get("Content-Length")
    if content_len and content_len.isdigit() and int(content_len) > expected_max:
        raise RangeIgnoredError("range_ignored")
    if len(response.content) > expected_max:
        raise RangeIgnoredError("range_ignored")


def fetch_safetensors_header(
    url: str,
    *,
    client: httpx.Client | None = None,
    token: str | None = None,
    timeout: float = 20.0,
) -> dict[str, dict[str, Any]]:
    """Range-GET a safetensors JSON header. Never downloads the tensor blob."""
    headers = _auth_headers(token)
    close_client = False
    if client is None:
        client = httpx.Client(timeout=timeout, headers={"User-Agent": HF_USER_AGENT}, follow_redirects=True)
        close_client = True
    try:
        length_headers = {**headers, "Range": "bytes=0-7"}
        probe = client.get(url, headers=length_headers, timeout=timeout)
        if probe.status_code in {401, 403}:
            raise PermissionError(f"gated:{probe.status_code}")
        if probe.status_code == 404:
            raise FileNotFoundError("missing_safetensors")
        probe.raise_for_status()
        _reject_if_range_ignored(probe, MAX_RANGE_BODY)
        if len(probe.content) >= HEADER_LEN_BYTES + 2:
            # Server ignored Range but the body still fits the header budget.
            return parse_safetensors_header(probe.content)

        if len(probe.content) < HEADER_LEN_BYTES:
            raise SafetensorsHeaderError("truncated_header_len")
        header_len = int.from_bytes(probe.content[:HEADER_LEN_BYTES], "little", signed=False)
        if header_len <= 0 or header_len > MAX_HEADER_SIZE:
            raise SafetensorsHeaderError("header_too_large")
        json_headers = {
            **headers,
            "Range": f"bytes={HEADER_LEN_BYTES}-{HEADER_LEN_BYTES + header_len - 1}",
        }
        body = client.get(url, headers=json_headers, timeout=timeout)
        if body.status_code in {401, 403}:
            raise PermissionError(f"gated:{body.status_code}")
        body.raise_for_status()
        _reject_if_range_ignored(body, header_len + 64)
        prefix = probe.content[:HEADER_LEN_BYTES] + body.content[:header_len]
        return parse_safetensors_header(prefix)
    finally:
        if close_client:
            client.close()
