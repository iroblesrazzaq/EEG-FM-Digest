from datetime import datetime, timezone

import httpx
import pytest

from eegfm_digest.arxiv import (
    ArxivFetchError,
    fetch_query,
    fetch_window_candidates,
    format_arxiv_datetime,
    group_candidates_by_month,
)


FEED_WINDOW = """\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2604.00010v1</id>
    <updated>2026-04-22T11:00:00Z</updated>
    <published>2026-04-22T11:00:00Z</published>
    <title>Windowed EEG FM Paper</title>
    <summary>Abstract text mentioning pretrain and representation learning.</summary>
    <author><name>Author</name></author>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2604.00010v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2604.00010v1" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2604.00011v1</id>
    <updated>2026-04-22T20:00:00Z</updated>
    <published>2026-04-22T20:00:00Z</published>
    <title>Edge Paper After Window</title>
    <summary>Abstract.</summary>
    <author><name>Author</name></author>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2604.00011v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2604.00011v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _CapturingClient:
    """Records every request's query params so we can assert the query string."""

    def __init__(self, body: str = FEED_WINDOW):
        self.calls: list[dict] = []
        self.body = body

    def get(self, _url, params=None):  # noqa: ANN001
        self.calls.append(dict(params or {}))
        return _FakeResponse(self.body)


def test_format_arxiv_datetime_uses_minute_precision_utc():
    dt = datetime(2026, 4, 22, 14, 37, 59, tzinfo=timezone.utc)
    assert format_arxiv_datetime(dt) == "202604221437"


def test_format_arxiv_datetime_converts_to_utc():
    from datetime import timedelta
    eastern = timezone(timedelta(hours=-4))
    dt = datetime(2026, 4, 22, 10, 30, tzinfo=eastern)  # 14:30 UTC
    assert format_arxiv_datetime(dt) == "202604221430"


def test_fetch_window_candidates_adds_date_filter_to_query(monkeypatch):
    monkeypatch.setattr("eegfm_digest.arxiv.time.sleep", lambda *a, **k: None)
    client = _CapturingClient()
    since = datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc)
    until = datetime(2026, 4, 23, 0, 0, tzinfo=timezone.utc)

    fetch_window_candidates(
        since,
        until,
        max_candidates=10,
        rate_limit_seconds=0,
        retries=0,
        retry_backoff_seconds=0,
        client=client,
    )

    assert len(client.calls) == 2  # QUERY_A + QUERY_B
    for params in client.calls:
        q = params["search_query"]
        assert "submittedDate:[202604220000 TO 202604230000]" in q


def test_fetch_window_candidates_filters_papers_outside_window(monkeypatch):
    monkeypatch.setattr("eegfm_digest.arxiv.time.sleep", lambda *a, **k: None)
    client = _CapturingClient()
    since = datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc)
    # Until 14:00 UTC: catches 11:00 paper, drops 20:00 paper
    until = datetime(2026, 4, 22, 14, 0, tzinfo=timezone.utc)

    papers = fetch_window_candidates(
        since,
        until,
        max_candidates=10,
        rate_limit_seconds=0,
        retries=0,
        retry_backoff_seconds=0,
        client=client,
    )
    ids = {p["arxiv_id_base"] for p in papers}
    assert "2604.00010" in ids
    assert "2604.00011" not in ids


def test_fetch_window_candidates_requires_ordered_bounds():
    since = datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="must be strictly greater"):
        fetch_window_candidates(
            since,
            since,
            max_candidates=1,
            rate_limit_seconds=0,
        )


def test_arxiv_fetch_error_is_runtimeerror_subclass():
    """Preserves backward compatibility with existing except RuntimeError: handlers."""
    assert issubclass(ArxivFetchError, RuntimeError)


def test_fetch_query_raises_arxiv_fetch_error_after_exhaustion(monkeypatch):
    monkeypatch.setattr("eegfm_digest.arxiv.time.sleep", lambda *a, **k: None)

    class _Always503:
        def get(self, _url, params=None):  # noqa: ANN001
            resp = httpx.Response(503, request=httpx.Request("GET", "https://x"))
            raise httpx.HTTPStatusError("boom", request=resp.request, response=resp)

    with pytest.raises(ArxivFetchError):
        fetch_query(
            "all:eeg",
            max_results=1,
            rate_limit_seconds=0,
            retries=0,
            retry_backoff_seconds=0,
            client=_Always503(),
        )


def test_group_candidates_by_month():
    candidates = [
        {"arxiv_id_base": "2603.00001", "published": "2026-03-31T23:00:00Z"},
        {"arxiv_id_base": "2604.00001", "published": "2026-04-01T01:00:00Z"},
        {"arxiv_id_base": "2604.00002", "published": "2026-04-10T00:00:00Z"},
    ]
    grouped = group_candidates_by_month(candidates)
    assert set(grouped.keys()) == {"2026-03", "2026-04"}
    assert len(grouped["2026-03"]) == 1
    assert len(grouped["2026-04"]) == 2
    # Sorted within month
    assert grouped["2026-04"][0]["arxiv_id_base"] == "2604.00001"


def test_group_candidates_skips_malformed_published():
    candidates = [
        {"arxiv_id_base": "x", "published": ""},
        {"arxiv_id_base": "y", "published": "2026"},
        {"arxiv_id_base": "z", "published": "2026-04-10T00:00:00Z"},
    ]
    grouped = group_candidates_by_month(candidates)
    assert set(grouped.keys()) == {"2026-04"}
