"""Data-quality checker: find the messy parts of the corpus and report them.

Run: python -m scripts.check_data_quality

The corpus is 10 real official pages, and real pages are messy:
a scholarship that has closed, a database that is not one award,
two different stipends on one page, figures that are approximate
or tied to one academic year, pages that carry no deadline or
stipend at all. A RAG system that hides that mess will invent
answers for it. This script finds it, cites the exact sentence
and chunk, and writes the report the documentation is built from.

Output:
- benchmarks/results/data_quality.json (machine-readable findings)
- benchmarks/results/data_quality.md (human-readable report)

Checks (each finding cites sentence + chunk_id + source URL):
1. closure: the page says the programme is closed, discontinued
   or replaced.
2. multiple_amounts: two or more different money amounts on one
   page for what looks like the same field (stipend). Flagged even
   when a resolution is known — the resolution is recorded, the
   flag stays, so the ambiguity is never silently forgotten.
3. database_page: the page is a search database / listing, not one
   award — so no single deadline or stipend exists to answer with.
4. approximate_figure: figures hedged with "approximately", "~",
   "around".
5. year_specific_figure: figures tied to one academic year
   (2026/27, 2027/28) — quoting them without the year is wrong.
6. missing_field: deadline / stipend not stated anywhere on the
   fetched page, per document. Missing is not an error in the page;
   it is a constraint on the system: those questions must abstain.
7. stale_page_marker: the page's own "last updated" marker, with
   its date, so staleness is measured, not assumed.

Resolutions are recorded in RESOLUTIONS below with their evidence.
A resolution never deletes the finding: the next run still flags
the item, marked "resolved", with the explanation attached.
"""
import json
import re
from pathlib import Path

from app.config import CHUNKS_FILE
from app.eligibility import RULES
from app.ingest import load_sources
from app.textnorm import segment_sentences

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"

CLOSURE_PATTERNS = [
    "no longer accepting", "no longer available", "discontinued",
    "closed for applications", "new harmonized program", "replaced by",
]
# Specific to the page itself being a database. A sentence that merely
# points the reader at another site's database ("look for other
# scholarships on Study in Sweden's scholarship database") is not
# this page being one — that false positive was caught during
# verification and the patterns were narrowed.
DATABASE_PATTERNS = [
    "scholarship database -", "welcome to our scholarship database",
    "options for your selection", "refine your selection",
]
APPROX_PATTERNS = ["approximately", "around ", "~"]
YEAR_RE = re.compile(r"\b20\d\d/\d\d\b")
MONEY_RE = re.compile(r"(€|\$|HK\$|US\$)\s?([\d,]+(?:\.\d+)?)")
DATE_RE = re.compile(
    r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)(\s+20\d\d)?\b")
DEADLINE_WORDS = ["deadline", "apply", "application", "admission", "portal closes",
                  "key dates"]
STALE_RE = re.compile(
    r"(Last updated|Mise à jour|Date modified)\s*[:.]?\s*"
    r"(\d{1,2}\s+\w+\s+20\d\d|\w+\s+20\d\d)")

# Known items with their resolution. Each resolution quotes the
# evidence that settles the ambiguity — a resolution without
# evidence is an opinion, not a finding.
RESOLUTIONS = {
    ("eiffel", "multiple_amounts"): {
        "resolution": "Not a contradiction. The page gives the amounts under "
                      "separate level headings: €1,200/month is the Master's "
                      "level allowance, €2,000/month is the Doctorate level "
                      "stipend, both from January 2026. Tuition fees are not "
                      "covered at either level.",
        "evidence": "Master's level. France Excellence Eiffel scholarship "
                    "holders receive a monthly allowance of €1,200 (from "
                    "January 2026). Doctorate level. France Excellence Eiffel "
                    "scholarship holders receive a monthly stipend of €2,000 "
                    "(from January 2026).",
    },
    ("vanier", "closure"): {
        "resolution": "The Vanier programme as described on this page is "
                      "closed: applications are no longer accepted and the "
                      "page points to the new harmonized Canada Graduate "
                      "Research Scholarship - Doctoral program. The system "
                      "answers the closure fact (golden q21) but the "
                      "eligibility rules do not yet factor closure into a "
                      "profile verdict — a known gap, see DATA_QUALITY.md.",
        "evidence": "We are no longer accepting applications for the Vanier "
                    "Canada Graduate Scholarships. Please refer to the Canada "
                    "Graduate Research Scholarship - Doctoral program web page "
                    "for information regarding the new harmonized program.",
    },
    ("hkpfs", "multiple_amounts"): {
        "resolution": "Not two stipends. HK$344,400 and approximately US$44,150 "
                      "are the same 2026/27 annual stipend expressed in two "
                      "currencies. Quote both figures with the year, and keep "
                      "the 'approximately' on the US dollar figure.",
        "evidence": "In the 2026/27 academic year, the Fellowship provides an "
                    "annual stipend of HK$344,400 (approximately US$44,150).",
    },
    ("daad", "database_page"): {
        "resolution": "The DAAD page is a scholarship database (147 options), "
                      "not one award. There is no single DAAD deadline or "
                      "stipend to answer with; the system must abstain on "
                      "those questions (golden o-questions cover this).",
        "evidence": "147 Scholarship options for your selection.",
    },
}


def _sentences(chunks: list[dict]):
    for chunk in chunks:
        for sent in segment_sentences(chunk["text"]):
            yield chunk, sent


def check(chunks: list[dict], sources: list[dict]) -> dict:
    urls = {s["slug"]: s["source_url"] for s in sources}
    findings: list[dict] = []

    def add(doc, kind, severity, sentence, chunk_id, note=""):
        key = (doc, kind)
        resolution = RESOLUTIONS.get(key)
        findings.append({
            "doc": doc, "kind": kind, "severity": severity,
            "status": "resolved" if resolution else "open",
            "sentence": sentence, "chunk_id": chunk_id,
            "source_url": urls.get(doc, ""), "note": note,
            "resolution": resolution["resolution"] if resolution else None,
            "resolution_evidence": resolution["evidence"] if resolution else None,
        })

    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc"], []).append(chunk)

    for chunk, sent in _sentences(chunks):
        low = sent.lower()
        doc = chunk["doc"]
        if any(p in low for p in CLOSURE_PATTERNS):
            add(doc, "closure", "high", sent, chunk["chunk_id"],
                "The programme as described may no longer exist.")
        if any(p in low for p in DATABASE_PATTERNS):
            add(doc, "database_page", "medium", sent, chunk["chunk_id"],
                "The page is a listing, not one award.")
        if any(p in low for p in APPROX_PATTERNS) and re.search(r"\d", sent):
            add(doc, "approximate_figure", "low", sent, chunk["chunk_id"],
                "Quote the figure with its hedge ('approximately').")
        if YEAR_RE.search(sent) and re.search(r"\d", sent):
            add(doc, "year_specific_figure", "low", sent, chunk["chunk_id"],
                "Quote the figure with its academic year.")

    # Stale markers span two normalised sentences ("Mise à jour." then
    # "October 2022."), so they are checked against the whole chunk
    # text, not per sentence.
    for chunk in chunks:
        stale = STALE_RE.search(chunk["text"])
        if stale:
            add(chunk["doc"], "stale_page_marker", "low",
                f"{stale.group(1)} {stale.group(2)}.", chunk["chunk_id"],
                f"Page's own last-updated marker: {stale.group(2)}.")

    # Multiple amounts per document (same field, different values).
    for doc, doc_chunks in sorted(by_doc.items()):
        amounts: dict[str, dict] = {}
        for chunk in doc_chunks:
            for sent in segment_sentences(chunk["text"]):
                for currency, value in MONEY_RE.findall(sent):
                    key = f"{currency}{value}"
                    amounts.setdefault(key, {"sentence": sent, "chunk_id": chunk["chunk_id"]})
        if len(amounts) >= 2:
            detail = "; ".join(f"{amt} ({info['chunk_id']})" for amt, info in sorted(amounts.items()))
            first = next(iter(amounts.values()))
            add(doc, "multiple_amounts", "medium", first["sentence"],
                first["chunk_id"], f"Distinct amounts on one page: {detail}.")

    # Missing fields per document: deadline and stipend.
    missing: dict[str, list[str]] = {}
    for doc, doc_chunks in sorted(by_doc.items()):
        doc_missing = []
        text = " ".join(c["text"] for c in doc_chunks)
        has_date = False
        for chunk in doc_chunks:
            for sent in segment_sentences(chunk["text"]):
                if DATE_RE.search(sent) and any(w in sent.lower() for w in DEADLINE_WORDS):
                    has_date = True
        if not has_date:
            doc_missing.append("deadline")
        if not MONEY_RE.search(text):
            doc_missing.append("stipend")
        if RULES.get(doc, {}).get("levels") is None:
            doc_missing.append("degree_levels")
        if RULES.get(doc, {}).get("citizenship") is None:
            doc_missing.append("citizenship_rule")
        if doc_missing:
            missing[doc] = doc_missing
            for field in doc_missing:
                add(doc, "missing_field", "info", "",
                    doc_chunks[0]["chunk_id"],
                    f"No {field} stated anywhere on the fetched page; "
                    "questions about it must abstain, not guess.")

    return {"findings": findings, "missing_fields": missing}


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]
    sources = load_sources()
    report = check(chunks, sources)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    lines = ["# Data quality report\n",
             "Generated by `python -m scripts.check_data_quality`. Every finding cites "
             "the exact sentence and chunk it came from.\n"]
    kinds: dict[str, int] = {}
    for f in report["findings"]:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    lines.append(f"Findings: {len(report['findings'])} "
                 f"({', '.join(f'{k}: {v}' for k, v in sorted(kinds.items()))})")
    lines.append("\n## Missing fields per document\n")
    for doc, fields in sorted(report["missing_fields"].items()):
        lines.append(f"- {doc}: {', '.join(fields)}")
    lines.append("\n## Findings\n")
    for f in report["findings"]:
        lines.append(f"### {f['doc']} — {f['kind']} [{f['severity']}] ({f['status']})")
        if f["sentence"]:
            lines.append(f"> {f['sentence']}")
            lines.append(f"Chunk: {f['chunk_id']} | {f['source_url']}")
        if f["note"]:
            lines.append(f"Note: {f['note']}")
        if f["resolution"]:
            lines.append(f"Resolution: {f['resolution']}")
        lines.append("")
    (RESULTS_DIR / "data_quality.md").write_text("\n".join(lines))
    print(f"{len(report['findings'])} findings "
          f"({sum(1 for f in report['findings'] if f['status'] == 'resolved')} resolved, "
          f"{sum(1 for f in report['findings'] if f['status'] == 'open')} open); "
          f"missing fields in {len(report['missing_fields'])} documents")
    for doc, fields in sorted(report["missing_fields"].items()):
        print(f"  {doc}: missing {', '.join(fields)}")


if __name__ == "__main__":
    main()
