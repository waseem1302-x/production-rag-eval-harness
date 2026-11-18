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
# Phase 2, Step 7 (re-verification): the money pattern now matches £,
# SEK and ¥ as well as €/$/HK$/US$, and amounts whose currency sign
# is detached from the digits — the RTP table writes "($) 35,550",
# i.e. the sign in parentheses before the number. Before this fix
# the checker flagged "stipend missing" for Rhodes (£21,060),
# Commonwealth Shared (£1,712/month), SI Pioneering Women
# (SEK 12,000/month), Sweden SI (SEK 12,000/month) and MEXT
# (¥117,000–¥242,000/month) although all five pages state amounts,
# and missed RTP entirely although it states six rates. The value
# group ends on a digit so "£4,000," in running text is recorded
# as £4,000, not £4,000,. The pattern still only proves that a
# money amount appears on the page, not that it is a stipend:
# Gates Cambridge's "US$210m" founding donation still matches,
# and is documented in its resolution.
MONEY_RE = re.compile(
    r"(€|\$|HK\$|US\$|£|SEK|¥)\s?([\d,]*\d(?:\.\d+)?)"
    r"|\(\$\)\s?([\d,]*\d(?:\.\d+)?)")
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
                      "Research Scholarship - Doctoral program. FIXED in "
                      "Phase 2, Step 4: the closure is now a flag in "
                      "data/closures.json, read and verbatim-validated by "
                      "the eligibility module, so a profile check for "
                      "Vanier returns verdict 'closed' with this evidence "
                      "and the replacement, whatever the profile (golden "
                      "e02, e05, e11).",
        "evidence": "We are no longer accepting applications for the Vanier "
                    "Canada Graduate Scholarships. Please refer to the Canada "
                    "Graduate Research Scholarship - Doctoral program web page "
                    "for information regarding the new harmonized program.",
    },
    ("hkpfs", "multiple_amounts"): {
        "resolution": "Two fields, each in two currencies — not two "
                      "stipends. HK$344,400 and approximately US$44,150 "
                      "are the same 2026/27 annual stipend expressed in "
                      "two currencies; HK$14,400 (approximately US$1,840) "
                      "is a separate conference and research-related "
                      "travel allowance per year. Quote both figures with "
                      "the year, and keep the 'approximately' on the US "
                      "dollar figures. (Step 7 correction: the original "
                      "resolution mentioned only the stipend; the travel "
                      "allowance is a second field on the same page.)",
        "evidence": "In the 2026/27 academic year, the Fellowship provides an "
                    "annual stipend of HK$344,400 (approximately US$44,150) "
                    "and a conference and research-related travel allowance "
                    "of HK$14,400 (approximately US$1,840) per year.",
    },
    ("daad", "database_page"): {
        "resolution": "The DAAD page is a scholarship database (147 options), "
                      "not one award. There is no single DAAD deadline or "
                      "stipend to answer with; the system must abstain on "
                      "those questions (golden o-questions cover this).",
        "evidence": "147 Scholarship options for your selection.",
    },
    ("erasmus-mundus-org", "multiple_amounts"): {
        "resolution": "Not a contradiction. The two amounts are different "
                      "fields: €5 million is the maximum grant per EMJM "
                      "project (to the consortium), €1,400 per month is the "
                      "EU scholarship paid to an individual student (max. "
                      "24 months). Quote each with its field.",
        "evidence": "Grant per project: up to €5 million. EU scholarship "
                    "for individuals: €1,400 per month (max. 24 months).",
    },
    # Phase 2, Step 7: these pages were flagged "multiple amounts"
    # only once the checker learned £/SEK/¥ and detached "($)".
    # Each resolution names the field each amount belongs to.
    ("gates-cambridge", "multiple_amounts"): {
        "resolution": "Different fields, plus a donation. £23,152 is the "
                      "maintenance allowance for a single student (2026-27 "
                      "rate, 12 months); £750–£4,000 is discretionary "
                      "academic development funding; £12,793 / £18,244 "
                      "per annum is the Dependent Children Allowance (one "
                      "child / two or more); US$210m is the Trust's "
                      "founding donation from 2000, not a stipend. The "
                      "checker counts 'a money amount appears', not 'a "
                      "stipend is stated' — the donation is why Gates was "
                      "never flagged missing even before the £ fix.",
        "evidence": "a maintenance allowance for a single student (£23,152 "
                    "for 12 months at the 2026-27 rate). The Gates Cambridge "
                    "Scholarship programme was established in October 2000 "
                    "by an historic donation of US$210m from the Gates "
                    "Foundation.",
    },
    ("rhodes", "multiple_amounts"): {
        "resolution": "Not three stipends. £21,060 is the annual stipend "
                      "for 2026-27; £1,755 is the same stipend paid in "
                      "monthly instalments (£21,060 / 12); £6,950 is not a "
                      "payment at all — it is 33% of the annual stipend, "
                      "the threshold above which a secondary award must be "
                      "declared to the Registrar.",
        "evidence": "For the 2026-27 academic year, the stipend is £21,060 "
                    "paid in monthly instalments of £1,755. If the value of "
                    "such an award is more than 33% of the annual stipend, "
                    "i.e. £6,950 pa, then you would need to declare the "
                    "award.",
    },
    ("commonwealth-shared", "multiple_amounts"): {
        "resolution": "Different fields. £1,712 per month (£2,000 in the "
                      "London metropolitan area) is the stipend (living "
                      "allowance); £622 / £154 per month are child "
                      "allowances for a widowed, divorced or single-parent "
                      "Scholar (first child / second and third child under "
                      "16). Quote the stipend with the London uplift, and "
                      "the child allowances only as child allowances.",
        "evidence": "Stipend (living allowance) at the rate of £1,712 per "
                    "month, or £2,000 per month for those at universities "
                    "in the London metropolitan area. Child allowance of "
                    "£622 per month for the first child, and £154 per month "
                    "for the second and third child under the age of 16.",
    },
    ("rtp", "multiple_amounts"): {
        "resolution": "One table, six rates: a base and a maximum rate for "
                      "each year. Full-time base RTP stipend rate ($) "
                      "35,550 (2027), 34,315 (2026), 33,511 (2025); "
                      "maximum 55,537 (2027), 53,608 (2026), 52,352 "
                      "(2025). The rates are indexed each 1 January and "
                      "the next year's are announced by 1 October. Golden "
                      "q29 quotes the 2026 base rate, $34,315 — quote any "
                      "rate with its year and base/maximum label.",
        "evidence": "Full-time base RTP stipend rate ($) 35,550 (2027). "
                    "Full-time base RTP stipend rate ($) 34,315 (2026). "
                    "The RTP base and maximum stipend rates are indexed "
                    "each year, effective from 1 January.",
    },
    ("sweden-si", "multiple_amounts"): {
        "resolution": "Different fields. SEK 12,000 per month is the "
                      "monthly living allowance; SEK 15,000 is a one-time "
                      "travel grant (SEK 10,000 for scholars from Armenia, "
                      "Azerbaijan, Belarus, Georgia, Moldova and Ukraine), "
                      "not a monthly payment. Both are stated on the page, "
                      "which is why the checker no longer flags Sweden SI "
                      "as missing a stipend after the Step 7 £/SEK fix.",
        "evidence": "A monthly allowance of SEK 12,000 to cover your living "
                    "expenses throughout the study period. A travel grant "
                    "of SEK 15,000.",
    },
    ("si-pioneering-women", "multiple_amounts"): {
        "resolution": "Different fields. SEK 12,000 per month is the "
                      "monthly living allowance; SEK 15,000 is a one-time "
                      "travel grant, not a monthly payment. Golden q32 "
                      "quotes the SEK 12,000 allowance.",
        "evidence": "A monthly allowance of SEK 12,000 to cover your living "
                    "expenses throughout the study period. A travel grant "
                    "of SEK 15,000.",
    },
    ("mext", "multiple_amounts"): {
        "resolution": "Three per-category monthly scholarship amounts "
                      "(¥117,000, ¥143,000–145,000 and ¥242,000 per month) "
                      "in a table whose category column headings are lost "
                      "in the extracted text, so no amount can be tied to "
                      "a category from the stored text alone. The page "
                      "states amounts, but there is no single 'MEXT "
                      "stipend in yen' to quote — golden o02 (monthly "
                      "stipend for the MEXT scholarship in yen) still "
                      "correctly abstains.",
        "evidence": "Â¥143,000ï½145,000 per month Â¥117,000 per month "
                    "ï¿¥242,000 per month (amounts in a table whose category "
                    "headings are lost in the extracted text; the yen sign "
                    "is mojibake in the stored file).",
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
                for currency, value, detached in MONEY_RE.findall(sent):
                    # Two alternatives: a currency-prefixed amount, or
                    # a detached "($) 35,550" amount (currency "$").
                    if detached:
                        currency, value = "$", detached
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
