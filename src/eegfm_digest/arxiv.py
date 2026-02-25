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


def category_match(categories: list[str], allowed_categories: set[str]) -> bool:
    return any(cat in allowed_categories for cat in categories)


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
                        raise RuntimeError(
                            f"arXiv request failed after {attempt} attempts "
                            f"(status={status}, start={start}, max_results={chunk})"
                        ) from exc
                    time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                except (httpx.ReadTimeout, httpx.TransportError) as exc:
                    attempt += 1
                    if attempt > retries:
                        raise RuntimeError(
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
    queries: list[str] | None = None,
    categories: list[str] | set[str] | None = None,
    connect_timeout_seconds: float = 10.0,
    read_timeout_seconds: float = 60.0,
    retries: int = 2,
    retry_backoff_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    query_list = queries or [QUERY_A, QUERY_B]
    allowed_categories = set(categories or ARXIV_CATEGORIES)
    combined: list[dict[str, Any]] = []
    for query in query_list:
        combined.extend(
            fetch_query(
                query,
                max_candidates,
                rate_limit_seconds,
                connect_timeout_seconds=connect_timeout_seconds,
                read_timeout_seconds=read_timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        )
    filtered = [
        p
        for p in combined
        if category_match(p["categories"], allowed_categories) and in_month(p["published"], month)
    ]
    return dedupe_latest(filtered)
