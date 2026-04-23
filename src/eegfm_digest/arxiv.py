from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx

from .keywords import ARXIV_CATEGORIES, QUERY_A, QUERY_B

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivFetchError(RuntimeError):
    """Raised when the arXiv API fails after all retries.

    Distinct from other ``RuntimeError``s so daily-mode callers can decide
    not to advance ``last_successful_run.json`` while leaving partial DB
    state in place for the next run's overlap window to re-process.
    """


def month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = month.split("-")
    start = datetime(int(year), int(mon), 1, tzinfo=timezone.utc)
    if int(mon) == 12:
        end = datetime(int(year) + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(int(year), int(mon) + 1, 1, tzinfo=timezone.utc)
    return start, end


def parse_arxiv_id(raw_id: str) -> tuple[str, int]:
    tail = raw_id.rsplit("/", 1)[-1]
    m = re.match(r"^(?P<base>.+?)(?:v(?P<ver>\d+))?$", tail)
    if not m:
        return tail, 1
    base = m.group("base")
    ver = int(m.group("ver") or 1)
    return base, ver


def parse_entry(entry: ET.Element) -> dict[str, Any]:
    paper_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    arxiv_id_base, version = parse_arxiv_id(paper_id)
    categories = [el.attrib.get("term", "") for el in entry.findall("atom:category", ATOM_NS)]
    links = {el.attrib.get("title", ""): el.attrib.get("href", "") for el in entry.findall("atom:link", ATOM_NS)}
    pdf_link = ""
    for el in entry.findall("atom:link", ATOM_NS):
        if el.attrib.get("type") == "application/pdf":
            pdf_link = el.attrib.get("href", "")
            break
    if not pdf_link:
        pdf_link = links.get("pdf", "")

    return {
        "arxiv_id": paper_id.rsplit("/", 1)[-1],
        "arxiv_id_base": arxiv_id_base,
        "version": version,
        "title": " ".join((entry.findtext("atom:title", default="", namespaces=ATOM_NS)).split()),
        "summary": " ".join((entry.findtext("atom:summary", default="", namespaces=ATOM_NS)).split()),
        "authors": [a.findtext("atom:name", default="", namespaces=ATOM_NS) for a in entry.findall("atom:author", ATOM_NS)],
        "categories": categories,
        "published": entry.findtext("atom:published", default="", namespaces=ATOM_NS),
        "updated": entry.findtext("atom:updated", default="", namespaces=ATOM_NS),
        "links": {
            "abs": paper_id,
            "pdf": pdf_link,
        },
    }


def in_month(published: str, month: str) -> bool:
    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    start, end = month_bounds(month)
    return start <= dt < end


def category_match(categories: list[str]) -> bool:
    return any(cat in ARXIV_CATEGORIES for cat in categories)


def dedupe_latest(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_base: dict[str, dict[str, Any]] = {}
    for p in papers:
        existing = by_base.get(p["arxiv_id_base"])
        if existing is None or int(p.get("version", 1)) > int(existing.get("version", 1)):
            by_base[p["arxiv_id_base"]] = p
    return sorted(by_base.values(), key=lambda x: (x["published"], x["arxiv_id_base"]))


def fetch_query(
    query: str,
    max_results: int,
    rate_limit_seconds: float,
    page_size: int = 100,
    max_start: int = 5000,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 60.0,
    retries: int = 2,
    retry_backoff_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    created_client = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(connect=connect_timeout_seconds, read=read_timeout_seconds, write=30.0, pool=30.0)
    )
    results: list[dict[str, Any]] = []
    try:
        start = 0
        while start <= max_start and len(results) < max_results:
            chunk = min(page_size, max_results - len(results))
            params = {
                "search_query": query,
                "start": start,
                "max_results": chunk,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            attempt = 0
            while True:
                try:
                    resp = client.get(ARXIV_API_URL, params=params)
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    retryable = status == 429 or (status is not None and status >= 500)
                    attempt += 1
                    if (not retryable) or attempt > retries:
                        raise ArxivFetchError(
                            f"arXiv request failed after {attempt} attempts "
                            f"(status={status}, start={start}, max_results={chunk})"
                        ) from exc
                    time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                except (httpx.ReadTimeout, httpx.TransportError) as exc:
                    attempt += 1
                    if attempt > retries:
                        raise ArxivFetchError(
                            f"arXiv request failed after {attempt} attempts "
                            f"(start={start}, max_results={chunk})"
                        ) from exc
                    time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
            root = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", ATOM_NS)
            parsed = [parse_entry(e) for e in entries]
            results.extend(parsed)
            if len(entries) < chunk:
                break
            start += chunk
            time.sleep(rate_limit_seconds)
    finally:
        if created_client:
            client.close()
    return results


def fetch_month_candidates(
    max_candidates: int,
    month: str,
    rate_limit_seconds: float,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 60.0,
    retries: int = 2,
    retry_backoff_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    combined = fetch_query(
        QUERY_A,
        max_candidates,
        rate_limit_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    ) + fetch_query(
        QUERY_B,
        max_candidates,
        rate_limit_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    filtered = [p for p in combined if category_match(p["categories"]) and in_month(p["published"], month)]
    return dedupe_latest(filtered)


def format_arxiv_datetime(dt: datetime) -> str:
    """Render a datetime for arXiv's ``submittedDate:[...]`` query parameter.

    arXiv accepts ``YYYYMMDDHHMM`` in UTC with minute precision.  Seconds
    are truncated; callers should use inclusive start / inclusive end
    and rely on SQLite deduplication for boundary overlap.
    """
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%d%H%M")


def _in_window(published: str, since: datetime, until: datetime) -> bool:
    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
    dt = dt.astimezone(timezone.utc)
    return since <= dt < until


def fetch_window_candidates(
    since: datetime,
    until: datetime,
    max_candidates: int,
    rate_limit_seconds: float,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 60.0,
    retries: int = 2,
    retry_backoff_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Return candidates with ``submittedDate`` in ``[since, until)``.

    The ``submittedDate:[...]`` filter is appended to ``QUERY_A`` and
    ``QUERY_B`` to keep the daily window tight at the source instead of
    paging through a full month.  Client-side filtering still applies
    because arXiv's minute-precision bounds are coarser than our window.
    """
    if until <= since:
        raise ValueError(f"until ({until!r}) must be strictly greater than since ({since!r})")

    since_utc = since.astimezone(timezone.utc)
    until_utc = until.astimezone(timezone.utc)
    date_filter = (
        f" AND submittedDate:[{format_arxiv_datetime(since_utc)} "
        f"TO {format_arxiv_datetime(until_utc)}]"
    )

    combined = fetch_query(
        QUERY_A + date_filter,
        max_candidates,
        rate_limit_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        client=client,
    ) + fetch_query(
        QUERY_B + date_filter,
        max_candidates,
        rate_limit_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        client=client,
    )

    filtered = [
        p
        for p in combined
        if category_match(p["categories"]) and _in_window(p["published"], since_utc, until_utc)
    ]
    return dedupe_latest(filtered)


def group_candidates_by_month(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket candidates by their published ``YYYY-MM`` so windows that
    straddle a month boundary can be published per-month."""
    out: dict[str, list[dict[str, Any]]] = {}
    for p in candidates:
        published = str(p.get("published", ""))
        if len(published) < 7:
            continue
        key = published[:7]
        out.setdefault(key, []).append(p)
    for month in out:
        out[month] = sorted(out[month], key=lambda x: (x["published"], x["arxiv_id_base"]))
    return out
