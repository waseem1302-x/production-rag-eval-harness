"""Ingestion: fetch official pages, extract text, save raw + metadata.

One document per scholarship. Every document is saved twice:
- data/raw/<slug>.html  (exact bytes fetched, when fetched by HTTP)
- data/raw/<slug>.txt    (extracted plain text used downstream)
- data/raw/<slug>.json   (source_url, fetched_at, http_status, fetch_method)

Chevening returned HTTP 403 to direct requests, so its text was
fetched with a text fetcher and saved as <slug>.txt with
fetch_method "browser-text-fetch" and no .html. That is recorded
in data/sources.json, not hidden.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import RAW_DIR, SOURCES_FILE, USER_AGENT

SOURCES = json.loads(SOURCES_FILE.read_text()) if SOURCES_FILE.exists() else []


def extract_text(html: str) -> str:
    """Extract readable text from HTML, dropping nav/script/style."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def fetch_one(source: dict, raw_dir: Path = RAW_DIR) -> dict:
    """Fetch one source. Returns its metadata record."""
    slug = source["slug"]
    url = source["source_url"]
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        status = resp.status_code
        if status != 200:
            return {**source, "fetched_at": fetched_at, "http_status": status,
                    "fetch_method": "http", "error": f"HTTP {status}", "chars": 0}
        (raw_dir / f"{slug}.html").write_bytes(resp.content)
        text = extract_text(resp.text)
        (raw_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
        record = {**source, "fetched_at": fetched_at, "http_status": status,
                  "fetch_method": "http", "chars": len(text)}
    except Exception as exc:  # network failure: record it, do not crash the run
        record = {**source, "fetched_at": fetched_at, "http_status": None,
                  "fetch_method": "http", "error": str(exc), "chars": 0}
    (raw_dir / f"{slug}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_sources() -> list[dict]:
    return json.loads(SOURCES_FILE.read_text())


def load_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Load all extracted texts that exist on disk, with their metadata."""
    docs = []
    for source in load_sources():
        slug = source["slug"]
        txt_path = raw_dir / f"{slug}.txt"
        meta_path = raw_dir / f"{slug}.json"
        if not txt_path.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else source
        docs.append({
            "doc": slug,
            "title": source["title"],
            "source_url": source["source_url"],
            "text": txt_path.read_text(encoding="utf-8"),
            "fetched_at": meta.get("fetched_at"),
            "fetch_method": meta.get("fetch_method"),
        })
    return docs
