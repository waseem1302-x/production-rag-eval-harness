"""Fetch the corpus. Usage: python -m scripts.ingest"""
from app.ingest import RAW_DIR, fetch_one, load_sources

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Pages whose text was fetched with a text fetcher (direct HTTP
    # fetch blocked or thin/JS-rendered): skip re-fetching them and
    # keep the saved text, as for Chevening and DAAD.
    BROWSER_TEXT_FETCH_SLUGS = (
        "chevening", "daad",
        "gates-cambridge", "rhodes", "knight-hennessy", "schwarzman",
        "commonwealth-shared", "rtp", "marshall",
        "erasmus-mundus-org", "si-pioneering-women",
    )
    for source in load_sources():
        if source["slug"] in BROWSER_TEXT_FETCH_SLUGS and (RAW_DIR / f"{source['slug']}.txt").exists():
            print(f"{source['slug']}: kept existing browser-text-fetch (direct HTTP fetch blocked or thin)")
            continue
        record = fetch_one(source)
        print(f"{source['slug']}: http={record.get('http_status')} "
              f"chars={record.get('chars')} method={record.get('fetch_method')} "
              f"{record.get('error', '')}")
