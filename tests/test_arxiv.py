import httpx
import pytest

from eegfm_digest.arxiv import dedupe_latest, fetch_query, in_month, parse_arxiv_id


FEED_ONE_ENTRY = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <updated>2025-01-01T00:00:00Z</updated>
    <published>2025-01-01T00:00:00Z</published>
    <title>Test EEG FM Paper</title>
    <summary>Abstract text.</summary>
    <author><name>Author One</name></author>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2501.00001v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2501.00001v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _RetryOnceClient:
    def __init__(self):
        self.calls = 0

    def get(self, _url, params=None):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            raise httpx.ReadTimeout("timed out")
        return _FakeResponse(FEED_ONE_ENTRY)


class _AlwaysTimeoutClient:
    def get(self, _url, params=None):  # noqa: ANN001
        raise httpx.ReadTimeout("timed out")


def test_parse_arxiv_id_version():
    base, ver = parse_arxiv_id("http://arxiv.org/abs/2501.01234v3")
    assert base == "2501.01234"
    assert ver == 3


def test_month_boundaries():
    assert in_month("2025-01-01T00:00:00Z", "2025-01")
    assert in_month("2025-01-31T23:59:59Z", "2025-01")
    assert not in_month("2024-12-31T23:59:59Z", "2025-01")
    assert not in_month("2025-02-01T00:00:00Z", "2025-01")


def test_dedupe_keeps_latest_version():
    rows = [
        {"arxiv_id_base": "2501.00001", "version": 1, "published": "2025-01-02T00:00:00Z"},
        {"arxiv_id_base": "2501.00001", "version": 3, "published": "2025-01-02T00:00:00Z"},
        {"arxiv_id_base": "2501.00002", "version": 1, "published": "2025-01-03T00:00:00Z"},
    ]
    out = dedupe_latest(rows)
    assert len(out) == 2
    keep = [r for r in out if r["arxiv_id_base"] == "2501.00001"][0]
    assert keep["version"] == 3


def test_fetch_query_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("eegfm_digest.arxiv.time.sleep", lambda *_args, **_kwargs: None)
    client = _RetryOnceClient()
    rows = fetch_query(
        "all:eeg",
        max_results=1,
        rate_limit_seconds=0,
        retries=2,
        retry_backoff_seconds=0,
        client=client,
    )
    assert client.calls == 2
    assert len(rows) == 1
    assert rows[0]["arxiv_id_base"] == "2501.00001"


def test_fetch_query_raises_after_retry_exhaustion(monkeypatch):
    monkeypatch.setattr("eegfm_digest.arxiv.time.sleep", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        fetch_query(
            "all:eeg",
            max_results=1,
            rate_limit_seconds=0,
            retries=2,
            retry_backoff_seconds=0,
            client=_AlwaysTimeoutClient(),
        )
