"""URL scraping utilities.

Keeps scraping deliberately small: fetch the HTML, strip chrome,
convert main content to Markdown so the chunker has clean input.
For heavier needs swap this for trafilatura or Firecrawl later.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown

from backend.config import settings

log = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    url: str
    title: str
    markdown: str


def _headers() -> dict:
    return {"User-Agent": settings.scraper_user_agent}


def urls_from_sitemap(sitemap_url: str, limit: Optional[int] = None) -> List[str]:
    """Expand a sitemap.xml (or sitemap index) into a flat list of URLs."""
    try:
        r = requests.get(sitemap_url, headers=_headers(), timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.warning("Could not fetch sitemap %s: %s", sitemap_url, e)
        return []

    urls: List[str] = []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        log.warning("Sitemap %s is not valid XML: %s", sitemap_url, e)
        return []

    # Sitemap index: recurse. Otherwise gather <loc> entries.
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    if root.tag.endswith("sitemapindex"):
        for loc in root.findall(".//sm:sitemap/sm:loc", ns):
            if loc.text:
                urls.extend(urls_from_sitemap(loc.text.strip()))
    else:
        for loc in root.findall(".//sm:url/sm:loc", ns):
            if loc.text:
                urls.append(loc.text.strip())

    if limit:
        urls = urls[:limit]
    return urls


def scrape_page(url: str) -> Optional[ScrapedPage]:
    """Fetch a URL and return its main content as Markdown."""
    try:
        r = requests.get(url, headers=_headers(), timeout=30)
        r.raise_for_status()
    except Exception as e:
        log.warning("Scrape failed for %s: %s", url, e)
        return None

    soup = BeautifulSoup(r.text, "lxml")

    # Drop chrome that pollutes chunks.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else url)

    # Prefer <main> / <article>, fall back to <body>.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    md = to_markdown(str(main), heading_style="ATX")

    # Collapse excessive blank lines.
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return ScrapedPage(url=url, title=title, markdown=md)


def resolve_target_urls(
    urls: List[str],
    sitemap_url: Optional[str],
    limit: int,
) -> List[str]:
    """Combine direct URLs and sitemap expansion, deduplicated, length-capped."""
    out: List[str] = []
    seen: set[str] = set()

    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)

    if sitemap_url:
        for u in urls_from_sitemap(sitemap_url, limit=limit):
            if u not in seen:
                out.append(u)
                seen.add(u)

    return out[:limit]
