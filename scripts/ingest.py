"""Fetch the corpus. Usage: python -m scripts.ingest"""
from app.ingest import RAW_DIR, fetch_one, load_sources

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source in load_sources():
        # Chevening (HTTP 403) and DAAD (proxy error) texts were fetched
        # with a text fetcher; skip re-fetching them and keep the saved text.
        if source["slug"] in ("chevening", "daad") and (RAW_DIR / f"{source['slug']}.txt").exists():
            print(f"{source['slug']}: kept existing browser-text-fetch (direct HTTP fetch blocked)")
            continue
        record = fetch_one(source)
        print(f"{source['slug']}: http={record.get('http_status')} "
              f"chars={record.get('chars')} method={record.get('fetch_method')} "
              f"{record.get('error', '')}")
