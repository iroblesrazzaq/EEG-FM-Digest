import httpx

from eegfm_digest.pdf import ARXIV_USER_AGENT, download_pdf, unversioned_arxiv_pdf_url


class _FakeResponse:
    def __init__(self, content: bytes = b"%PDF-1.4", status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://arxiv.org/pdf/2309.12056v2")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)


class _FakeClient:
    def __init__(self, *args, **kwargs):  # noqa: ANN002
        self._responses = kwargs.pop("_responses", None)
        self.kwargs = kwargs
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):  # noqa: ANN002
        return False

    def get(self, url: str):
        self.urls.append(url)
        if self._responses is not None:
            return self._responses.pop(0)
        return _FakeResponse()


def test_unversioned_arxiv_pdf_url():
    assert (
        unversioned_arxiv_pdf_url("https://arxiv.org/pdf/2309.12056v2")
        == "https://arxiv.org/pdf/2309.12056"
    )
    assert unversioned_arxiv_pdf_url("https://example.com/file.pdf") is None


def test_download_pdf_sends_user_agent(monkeypatch, tmp_path):
    created: list[_FakeClient] = []

    def fake_client(*args, **kwargs):  # noqa: ANN002
        client = _FakeClient(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("eegfm_digest.pdf.httpx.Client", fake_client)
    out = tmp_path / "paper.pdf"
    download_pdf("https://arxiv.org/pdf/2309.12056", out, 0.0)
    assert created[0].kwargs["headers"]["User-Agent"] == ARXIV_USER_AGENT
    assert created[0].kwargs["follow_redirects"] is True
    assert out.read_bytes() == b"%PDF-1.4"


def test_download_pdf_retries_unversioned_url(monkeypatch, tmp_path):
    created: list[_FakeClient] = []

    def fake_client(*args, **kwargs):  # noqa: ANN002
        client = _FakeClient(
            *args,
            **kwargs,
            _responses=[_FakeResponse(status_code=403), _FakeResponse()],
        )
        created.append(client)
        return client

    monkeypatch.setattr("eegfm_digest.pdf.httpx.Client", fake_client)
    out = tmp_path / "paper.pdf"
    download_pdf("https://arxiv.org/pdf/2309.12056v2", out, 0.0)
    assert created[0].urls == [
        "https://arxiv.org/pdf/2309.12056v2",
        "https://arxiv.org/pdf/2309.12056",
    ]
    assert out.read_bytes() == b"%PDF-1.4"
