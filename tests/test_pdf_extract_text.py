import builtins
import types
from pathlib import Path

from eegfm_digest.pdf import extract_text


class _FakePyMuPage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakePyMuDoc:
    def __init__(self, texts: list[str]):
        self._pages = [_FakePyMuPage(text) for text in texts]
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def close(self) -> None:
        self.closed = True


class _FakePyPdfPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


def _patch_imports(monkeypatch, module_map: dict[str, object], blocked: set[str] | None = None) -> list[str]:
    blocked = blocked or set()
    original_import = builtins.__import__
    seen: list[str] = []

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        seen.append(name)
        if name in module_map:
            return module_map[name]
        if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return seen


def _make_pymupdf_module(texts: list[str] | None = None, error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("pymupdf")

    def open_doc(_path: str):  # noqa: ANN001
        if error is not None:
            raise error
        return _FakePyMuDoc(texts or [])

    module.open = open_doc  # type: ignore[attr-defined]
    return module


def _make_pypdf_module(texts: list[str] | None = None, error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("pypdf")

    class FakeReader:
        def __init__(self, _path: str):
            if error is not None:
                raise error
            self.pages = [_FakePyPdfPage(text) for text in (texts or [])]

    module.PdfReader = FakeReader  # type: ignore[attr-defined]
    return module


def _make_pdfminer_module(text: str | None = None, error: Exception | None = None) -> types.ModuleType:
    module = types.ModuleType("pdfminer.high_level")

    def extract_fn(_path: str) -> str:
        if error is not None:
            raise error
        return text or ""

    module.extract_text = extract_fn  # type: ignore[attr-defined]
    return module


def _pdf_path(tmp_path: Path, name: str = "paper.pdf") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4")
    return path


def test_extract_text_cached_short_circuit(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("cached content", encoding="utf-8")
    seen = _patch_imports(monkeypatch, module_map={}, blocked={"pymupdf", "pypdf", "pdfminer"})

    meta = extract_text(pdf_path, text_path)

    assert meta == {"tool": "cached", "pages": None, "chars": len("cached content"), "error": None}
    assert seen == []


def test_extract_text_pymupdf_success(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    _patch_imports(
        monkeypatch,
        module_map={"pymupdf": _make_pymupdf_module(["alpha", "beta"])},
        blocked={"pypdf", "pdfminer"},
    )

    meta = extract_text(pdf_path, text_path)

    assert text_path.read_text(encoding="utf-8") == "alpha\nbeta"
    assert meta == {"tool": "pymupdf", "pages": 2, "chars": len("alpha\nbeta"), "error": None}


def test_extract_text_pymupdf_import_missing_falls_back_to_pypdf(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    _patch_imports(
        monkeypatch,
        module_map={"pypdf": _make_pypdf_module(["from pypdf"])},
        blocked={"pymupdf", "pdfminer"},
    )

    meta = extract_text(pdf_path, text_path)

    assert meta["tool"] == "pypdf"
    assert meta["pages"] == 1
    assert meta["chars"] == len("from pypdf")
    assert "pymupdf_failed:" in (meta["error"] or "")


def test_extract_text_pymupdf_runtime_failure_falls_back_to_pypdf(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    _patch_imports(
        monkeypatch,
        module_map={
            "pymupdf": _make_pymupdf_module(error=RuntimeError("pymupdf boom")),
            "pypdf": _make_pypdf_module(["fallback pypdf"]),
        },
        blocked={"pdfminer"},
    )

    meta = extract_text(pdf_path, text_path)

    assert meta["tool"] == "pypdf"
    assert "pymupdf_failed:pymupdf boom" in (meta["error"] or "")
    assert text_path.read_text(encoding="utf-8") == "fallback pypdf"


def test_extract_text_falls_to_pdfminer_after_first_two_fail(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    _patch_imports(
        monkeypatch,
        module_map={
            "pymupdf": _make_pymupdf_module(error=RuntimeError("pymupdf boom")),
            "pypdf": _make_pypdf_module(error=RuntimeError("pypdf boom")),
            "pdfminer.high_level": _make_pdfminer_module(text="from pdfminer"),
        },
    )

    meta = extract_text(pdf_path, text_path)

    assert meta["tool"] == "pdfminer"
    assert meta["pages"] is None
    assert meta["chars"] == len("from pdfminer")
    assert "pymupdf_failed:pymupdf boom" in (meta["error"] or "")
    assert "pypdf_failed:pypdf boom" in (meta["error"] or "")
    assert text_path.read_text(encoding="utf-8") == "from pdfminer"


def test_extract_text_all_extractors_fail(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    _patch_imports(
        monkeypatch,
        module_map={
            "pymupdf": _make_pymupdf_module(error=RuntimeError("pymupdf boom")),
            "pypdf": _make_pypdf_module(error=RuntimeError("pypdf boom")),
            "pdfminer.high_level": _make_pdfminer_module(error=RuntimeError("pdfminer boom")),
        },
    )

    meta = extract_text(pdf_path, text_path)

    assert meta["tool"] == "none"
    assert meta["pages"] is None
    assert meta["chars"] == 0
    assert "pymupdf_failed:pymupdf boom" in (meta["error"] or "")
    assert "pypdf_failed:pypdf boom" in (meta["error"] or "")
    assert "pdfminer_failed:pdfminer boom" in (meta["error"] or "")
    assert text_path.read_text(encoding="utf-8") == ""


def test_extract_text_is_deterministic_for_identical_inputs(monkeypatch, tmp_path):
    _patch_imports(monkeypatch, module_map={"pymupdf": _make_pymupdf_module(["same", "text"])})
    pdf_a = _pdf_path(tmp_path, "a.pdf")
    pdf_b = _pdf_path(tmp_path, "b.pdf")
    text_a = tmp_path / "text" / "a.txt"
    text_b = tmp_path / "text" / "b.txt"

    first = extract_text(pdf_a, text_a)
    second = extract_text(pdf_b, text_b)

    assert first == second
    assert text_a.read_text(encoding="utf-8") == text_b.read_text(encoding="utf-8")


def test_extract_text_unicode_round_trip(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "text" / "paper.txt"
    unicode_text = "naïve café Δ"
    _patch_imports(monkeypatch, module_map={"pymupdf": _make_pymupdf_module([unicode_text])})

    meta = extract_text(pdf_path, text_path)

    assert text_path.read_text(encoding="utf-8") == unicode_text
    assert meta["chars"] == len(unicode_text)


def test_extract_text_creates_parent_directories(monkeypatch, tmp_path):
    pdf_path = _pdf_path(tmp_path)
    text_path = tmp_path / "nested" / "deep" / "paper.txt"
    _patch_imports(monkeypatch, module_map={"pymupdf": _make_pymupdf_module(["ok"])})

    meta = extract_text(pdf_path, text_path)

    assert text_path.exists()
    assert meta["tool"] == "pymupdf"


def test_extract_text_metadata_key_stability_for_all_branches(monkeypatch, tmp_path):
    expected = {"tool", "pages", "chars", "error"}

    pdf_cached = _pdf_path(tmp_path, "cached.pdf")
    txt_cached = tmp_path / "out" / "cached.txt"
    txt_cached.parent.mkdir(parents=True, exist_ok=True)
    txt_cached.write_text("cached", encoding="utf-8")
    meta_cached = extract_text(pdf_cached, txt_cached)
    assert set(meta_cached.keys()) == expected

    _patch_imports(monkeypatch, module_map={"pymupdf": _make_pymupdf_module(["ok"])})
    meta_pymu = extract_text(_pdf_path(tmp_path, "pymu.pdf"), tmp_path / "out" / "pymu.txt")
    assert set(meta_pymu.keys()) == expected

    _patch_imports(
        monkeypatch,
        module_map={"pypdf": _make_pypdf_module(["ok"])},
        blocked={"pymupdf"},
    )
    meta_pypdf = extract_text(_pdf_path(tmp_path, "pypdf.pdf"), tmp_path / "out" / "pypdf.txt")
    assert set(meta_pypdf.keys()) == expected

    _patch_imports(
        monkeypatch,
        module_map={
            "pymupdf": _make_pymupdf_module(error=RuntimeError("boom1")),
            "pypdf": _make_pypdf_module(error=RuntimeError("boom2")),
            "pdfminer.high_level": _make_pdfminer_module(text="ok"),
        },
    )
    meta_pdfminer = extract_text(_pdf_path(tmp_path, "pdfminer.pdf"), tmp_path / "out" / "pdfminer.txt")
    assert set(meta_pdfminer.keys()) == expected

    _patch_imports(
        monkeypatch,
        module_map={
            "pymupdf": _make_pymupdf_module(error=RuntimeError("boom1")),
            "pypdf": _make_pypdf_module(error=RuntimeError("boom2")),
            "pdfminer.high_level": _make_pdfminer_module(error=RuntimeError("boom3")),
        },
    )
    meta_none = extract_text(_pdf_path(tmp_path, "none.pdf"), tmp_path / "out" / "none.txt")
    assert set(meta_none.keys()) == expected
