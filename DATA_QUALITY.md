# Data Quality — Messy Real Scholarship Data

The corpus is 19 real official scholarship pages, fetched 18 Nov 2026
(the original 10 plus 9 more in Phase 2, Step 6) and **re-verified
18 Nov 2026** (Phase 2, Step 7: every page re-fetched and compared
with the stored text — see §11 and
`benchmarks/results/reverification_2026-11-18.json`). Real pages are
messy. This document records every mess found, the exact evidence
for it, what it means for the system, and how it is handled. Nothing
here is hidden: the checker that produced these findings is
`scripts/check_data_quality.py`, and its machine-readable
report is `benchmarks/results/data_quality.json` (73 findings:
16 resolved, 57 open). Re-run the checker after any re-ingest;
`DATA_QUALITY.md` must be updated from its output, not from memory.

## Summary of findings

| # | Issue | Documents | Status |
|---|---|---|---|
| 1 | Programme closed / replaced | Vanier | Resolved (documented; eligibility returns `closed` since Phase 2, Step 4) |
| 2 | Two stipend amounts, one page | Eiffel, HKPFS, Erasmus Mundus (organisations) | Resolved (documented) |
| 3 | Page is a database, not one award | DAAD | Resolved (documented) |
| 4 | Approximate figures | Fulbright, HKPFS, Gates Cambridge, Schwarzman | Open — quote with hedge |
| 5 | Year-specific figures | HKPFS, Commonwealth Shared | Open — quote with year |
| 6 | Deadline not stated on fetched page | 15 of 19 docs | Open — must abstain |
| 7 | Stipend not stated on fetched page | **8 of 19 docs** (was 14 before the Step 7 checker fix) | Open — must abstain |
| 8 | Degree levels / citizenship not stated | 11 docs missing degree levels, 12 missing a citizenship rule (all 9 new docs, by construction — no rules authored yet); Sweden SI now has no missing field at all | Open — verdict `unknown` |
| 9 | Stale page markers | Erasmus (26 June 2025), Eiffel (Oct 2022); EMJM organisations page footer "Last updated: 15 April 2026" newly observed in Step 7 | Open — re-fetch before launch (done 18 Nov 2026, §11) |
| 10 | Checker currency gap (£/SEK not matched) | **Fixed in Step 7** — Rhodes, Commonwealth Shared, SI Pioneering, Sweden SI, MEXT and RTP no longer falsely flagged; Gates now matches on its real stipend too. Remaining: the "US$210m" donation still matches, and MEXT's ¥ amounts cannot be tied to a category | Fixed (with named remainders) — see §10 |

## 1. Vanier is closed (high severity)

Evidence (chunk `vanier-0`, https://vanier.gc.ca/en/home-accueil.html):

> "We are no longer accepting applications for the Vanier Canada Graduate Scholarships."
> "Please refer to the Canada Graduate Research Scholarship - Doctoral program web page for information regarding the new harmonized program."

What it means: the programme as described on this page no longer
accepts applications and has been replaced by a harmonized doctoral
programme. Quoting the Vanier $50,000 stipend without the closure
would mislead an applicant.

How the system handles it: the closure is a golden question (q21:
"Is the Vanier Canada Graduate Scholarship still accepting
applications?"), and the system answers it correctly from the
corpus. **Fixed in Phase 2, Step 4:** the eligibility module now
reads the closure flag — `data/closures.json`, one entry for
Vanier with the two verbatim sentences above and the replacement
programme, validated against chunk `vanier-0` at load time — so a
profile check for Vanier returns verdict `closed` with this
evidence and the replacement named, whatever the profile (golden
e02, e05, e11). A doctoral applicant who meets every other Vanier
rule is told the programme is closed, not "doctoral only".

## 2. Two stipend amounts on one page

### Eiffel — resolved, not a contradiction

The page shows €1,200/month and €2,000/month. Reading the level
headings in chunk `eiffel-1` resolves it:

> "Master's level. France Excellence Eiffel scholarship holders receive a monthly allowance of €1,200 (from January 2026)…"
> "Doctorate level. France Excellence Eiffel scholarship holders receive a monthly stipend of €2,000 (from January 2026)…"

€1,200 is the Master's allowance; €2,000 is the Doctorate stipend.
Both from January 2026. Note the wording differs too ("allowance"
vs "stipend") and tuition fees are not covered at either level.
Golden questions q22 (master's → €1,200) and q23 (doctorate →
€2,000) verify the system picks the right amount for the right
level; both pass.

### Erasmus Mundus (organisations page) — resolved, different fields

The organisations-side EMJM page shows €5 million and €1,400/month.
Reading the field labels in chunk `erasmus-mundus-org-0` resolves it:

> "Grant per project: up to €5 million."
> "EU scholarship for individuals: €1,400 per month (max. 24 months)."

€5 million is the maximum grant per EMJM project (to the
consortium); €1,400/month is the EU scholarship paid to an
individual student. Quote each with its field. Golden question q31
(individual → €1,400/month) passes; the resolution is recorded in
`scripts/check_data_quality.py:RESOLUTIONS` and pinned by a test.

### HKPFS — resolved, two fields, each in two currencies

Chunk `hkpfs-0`:

> "In the 2026/27 academic year, the Fellowship provides an annual stipend of HK$344,400 (approximately US$44,150) and a conference and research-related travel allowance of HK$14,400 (approximately US$1,840) per year for each awardee for a period up to three years."

HK$344,400 and ~US$44,150 are the same stipend in two currencies,
not two stipends — and HK$14,400 (~US$1,840) is a separate
conference/travel allowance, also in two currencies. Quote both
figures with the year, and keep "approximately" on the US dollar
figures. (Step 7 correction: the earlier resolution mentioned only
the stipend; the travel allowance is a second field on the same
page.) The 400 fellowships figure is separate and year-specific:
"400 PhD Fellowships will be awarded in the 2027/28 academic year".

## 3. DAAD is a database, not one award

Evidence (chunk `daad-0`):

> "Scholarship Database - DAAD - Deutscher Akademischer Austauschdienst."
> "Welcome to our scholarship database!"
> "147 Scholarship options for your selection."

There is no single DAAD deadline or stipend to answer with — the
page is a search interface over 147 programmes, each with its own
terms. The system must abstain on DAAD deadline/stipend questions
rather than invent one; the golden out-of-corpus questions cover
this behaviour (a DAAD deadline question and a DAAD GPA question
both correctly abstain).

## 4. Approximate figures

- Fulbright (chunk `fulbright-0`): "Approximately 4,000 foreign
  students receive Fulbright scholarships each year." The "~4,000"
  is an approximation; the answer must keep "approximately".
- HKPFS (chunk `hkpfs-0`): "approximately US$44,150" (see §2).
- Gates Cambridge (chunk `gates-cambridge-0`): "around 70 full-cost
  scholarships" each year, and "approximately 250 Scholars studying
  in Cambridge at any one time". Both figures are hedged on the page.
- Schwarzman (chunk `schwarzman-8`): an interview timing of
  "approximately one minute" — a hedged figure inside the FAQ, not a
  scholarship amount.

## 5. Year-specific figures

- HKPFS stipend HK$344,400 is for the **2026/27** academic year.
- HKPFS "400 PhD Fellowships" is for the **2027/28** academic year.
  Quoting either figure without its year is wrong, and both will go
  stale: the checker flags any figure tied to an academic year so a
  re-ingest can catch it.
- Commonwealth Shared (chunk `commonwealth-shared-1`): "Applications
  for Commonwealth Shared Scholarships for the **2027/28** academic
  year are now open." The 2027/28 label is part of the fact; the
  closing date the system answers with (8 December 2026, golden q28)
  belongs to that intake.

## 6. Deadline not stated on the fetched page (15 of 19)

The checker counts a deadline as stated only when a sentence carries
both a date and a deadline word ("deadline", "apply", "application",
"admission", "portal closes", …). By that rule, 4 of the 19 pages
state one — SI Global Professionals, Commonwealth Shared (closing
date 8 December 2026, golden q28), Marshall, and SI Pioneering
Women — and 15 do not:

| Document | Deadline on fetched page? |
|---|---|
| sweden-si | Yes — 15 Jan 2027 (programmes), 24 Feb 2027 (scholarship) |
| commonwealth-shared | Yes — 8 December 2026 (2027/28 intake) |
| marshall | Yes (per the checker: a date-bearing sentence with a deadline word) |
| si-pioneering-women | Yes (per the checker: a date-bearing sentence with a deadline word) |
| chevening | No |
| fulbright | No — mentions "university deadline" but no date (country-specific) |
| daad | No — database, see §3 |
| erasmus-mundus | No — the only date is "Last updated: 26 June 2025", a page marker, not a deadline |
| erasmus-mundus-org | No |
| eiffel | No — "contact the international relations department" for deadlines |
| vanier | No — and the programme is closed, see §1 |
| manaaki | No |
| hkpfs | No |
| mext | No |
| gates-cambridge | No |
| rhodes | No |
| knight-hennessy | No |
| schwarzman | No |
| rtp | No |

The system must answer "not found in the corpus" for these, never
a guessed date. The golden set tests this: a Chevening deadline
question and a DAAD deadline question both abstain — and an
Australia Awards deadline question (o06) abstains too, since that
scholarship is not in the corpus at all.

## 7. Stipend not stated on the fetched page (8 of 19, by the checker)

The checker counts a stipend as stated when the text contains a
€/$/HK$/US$/£/SEK/¥ amount, or a detached "($) 35,550" table rate
(the £/SEK/¥ and detached patterns were added in Phase 2, Step 7 —
see §10). By that rule it flags 8 of 19 pages as missing a stipend,
and counts 11 as stating one. Before Step 7 the rule matched €/$
only and flagged 14 — the six pages it wrongly flagged are Rhodes,
Commonwealth Shared, SI Pioneering Women, Sweden SI, MEXT and RTP
(see §10). Read the rule honestly, in both directions:

- **Gates is still a partial false positive.** The amounts the
  checker matches include "US$210m" — the Trust's founding
  donation, not a stipend. But since Step 7 the page's real stipend
  (£23,152) also matches, so the page is flagged as stating a
  stipend for the right reason as well as the wrong one (see §10).
- **MEXT states amounts, but not one quotable stipend.** The page
  lists ¥117,000, ¥143,000–145,000 and ¥242,000 per month in a
  table whose category headings are lost in the extracted text, so
  no amount can be tied to a category from the stored text alone.
  The checker now correctly counts MEXT as stating an amount; the
  golden out-of-corpus question o02 ("monthly stipend for the MEXT
  scholarship in yen") still correctly abstains, because there is
  no single MEXT stipend to quote.
- The true stipend statements on the fetched pages: Vanier
  ($50,000/year, doctoral), Eiffel (€1,200 / €2,000 per month, see
  §2), HKPFS (HK$344,400/year plus a HK$14,400 travel allowance,
  see §2), Erasmus Mundus (organisations) (€1,400/month, see §2),
  Gates (£23,152), Rhodes (£21,060 in 2026-27, golden q25),
  Commonwealth Shared (£1,712 per month, £2,000 in London),
  SI Pioneering Women (SEK 12,000 per month, golden q32), Sweden SI
  (SEK 12,000 per month, plus a SEK 15,000 one-time travel grant),
  MEXT (¥ amounts per category, see above), and RTP (base $35,550/year
  for 2027; $34,315 for 2026, golden q29). Chevening ("fully-funded",
  no figure), Fulbright, DAAD, Erasmus Mundus (student page), Manaaki,
  Knight-Hennessy, Schwarzman and Marshall state no amount on the
  fetched page. The 8 pages the checker flags as missing a stipend
  are exactly those 8.

The 15-of-19 deadline count in §6 is unchanged by the re-verification
(§11): no re-fetched page gained or lost a deadline.

## 8. Degree levels / citizenship not stated

The eligibility module (`app/eligibility.py`) returns `unknown`
where the corpus states no rule, rather than guessing from general
knowledge:

- **Fulbright**: no degree-level and no citizenship sentence in the
  fetched text (golden e10 tests this: verdict `unknown`).
- **DAAD**: no degree-level rule (it is a database); citizenship is
  stated only as "international students and researchers".
- **MEXT**: degree levels are stated (undergraduate / master's /
  doctoral), but no citizenship rule is in the fetched text.
- **Chevening**: degree level is stated (one-year master's), but no
  citizenship list is in the fetched text — the page says only
  "from all over the world".
- **The 9 new documents (Phase 2, Step 6)**: no eligibility rules
  are authored for any of them yet, so the checker flags
  degree-levels and/or citizenship as missing for all nine by
  construction — 11 docs missing degree levels and 12 missing a
  citizenship rule in total. Several pages DO state them (e.g.
  SI Pioneering Women lists its 10 eligible countries; Rhodes and
  Schwarzman state degree levels), so these flags mean "no rule
  authored", not "page says nothing". Authoring rules is a separate
  step; until then a profile check for any of the nine returns
  `unknown`.

Correction to an earlier assumption: the SI page **does** list its
34 eligible countries in the corpus (chunk `sweden-si-2`), and
Pakistan is **not** among them. The earlier note that the list was
"not on the fetched page" was wrong; the checker and the
eligibility module both use the list from the chunk.

## 9. Stale page markers

- Erasmus Mundus (chunk `erasmus-mundus-1`): "Last updated:
  26 June 2025." — about 5 months before the fetch (18 Nov 2026).
  **Re-verified 18 Nov 2026 (§11): the marker is still 26 June 2025.**
- Eiffel (chunk `eiffel-1`): "Mise à jour. October 2022." for the
  main page content — about 4 years old — while the brochures were
  updated December 2025 (chunk `eiffel-2`). The amounts are marked
  "from January 2026", so they are current, but the page's own
  update marker is stale. **Re-verified 18 Nov 2026 (§11): both
  markers unchanged.**
- Erasmus Mundus (organisations page): the browser-text extraction
  carries no last-updated marker, but the direct-HTTP re-fetch in
  Step 7 shows a page footer the text fetcher strips: "Last
  updated: 15 April 2026", plus "Related calls: There are no calls
  open at the moment. The next round of funding will open towards
  the end of the year." The 18 Nov fetch also lacked the footer, so
  this marker is recorded as **newly observed, not a verified
  change** — it may have been there all along, below the extracted
  text. It does not change any fact on the page.

## 10. Checker currency gap: £ and SEK amounts are missed — fixed in Step 7

The checker's money pattern (`MONEY_RE` in
`scripts/check_data_quality.py`) matched €, $ and HK$/US$ only —
not £ or SEK — until Phase 2, Step 7 fixed it. The pattern now
matches €, $, HK$, US$, £, SEK and ¥, plus amounts with the
currency sign detached from the digits ("($) 35,550"), and a value
group that ends on a digit so "£4,000," in running text is recorded
as £4,000. What the fix changed, measured by re-running the
checker (72 → 73 findings, 9 → 16 resolved, 63 → 57 open):

- Six false "stipend missing" flags are gone: Rhodes (£21,060),
  Commonwealth Shared (£1,712/month), SI Pioneering Women
  (SEK 12,000/month), Sweden SI (SEK 12,000/month), MEXT
  (¥117,000–¥242,000/month) and RTP ("($) 35,550"). The
  "stipend not stated" count in §7 is now 8 of 19, not 14 of 19.
- Seven new `multiple_amounts` findings appear (Gates, Rhodes,
  Commonwealth Shared, RTP, Sweden SI, SI Pioneering Women, MEXT),
  all resolved with the field each amount belongs to (see §2 and
  `RESOLUTIONS` in the checker). The HKPFS resolution was also
  corrected in the same step: the page carries a travel allowance
  (HK$14,400) as well as the stipend, not only one figure in two
  currencies.
- Gates Cambridge is still not flagged, but now for the right
  reason as well as the wrong one: its real stipend (£23,152)
  matches, alongside the "US$210m" founding donation.

What the fix did **not** do (remaining gaps, pinned by
`tests/test_reverification.py`):

- The pattern still only proves that a money amount *appears* on
  the page, not that it is a stipend. Gates's "US$210m" donation
  still matches; the resolution documents it, the pattern does not
  distinguish it.
- MEXT's ¥ amounts cannot be tied to a category from the stored
  text (the table headings are lost in extraction), so the checker
  counts MEXT as stating an amount while no single MEXT stipend
  can be quoted — golden o02 still correctly abstains.
- Amounts written without any currency sign or code (a bare
  "35,550" with the "$" only in a table header, or a "kr" amount
  with no SEK prefix) would still be missed; no such amount is in
  this corpus.

## How the system uses this

- **Abstention**: questions about a missing field return "Not found
  in the corpus." with a recorded reason (`missing_specific_term`
  / `weak_overlap` / `unanswered_scholarship_subject`). Measured:
  abstention accuracy 1.0 (7/7) on the grown corpus — including
  o06 (Australia Awards deadline, a scholarship not in the corpus)
  and o07 (KHS acceptance rate, a fact not on the fetched page),
  both fixed in Phase 2, Step 6 after corpus growth first broke
  them (1.0 → 0.714 mid-step).
- **Eligibility**: `unknown` verdicts where the corpus states no
  rule (Fulbright e10; all 9 new documents until rules are
  authored); `closed` where the corpus says the programme
  is closed (Vanier e02, e05, e11 — Phase 2, Step 4). Measured:
  eligibility accuracy 1.0 (11/11).
- **Golden set**: q21 (Vanier closure), q22/q23 (Eiffel amounts by
  level) were added for this step and all pass; e11 (Vanier doctoral
  applicant, verdict `closed`) was added in Phase 2, Step 4 and passes.
  Step 6 added q24–q32 (one per new document); q24–q31 pass, q32 is
  a documented miss. The Phase 2 improvement split the dash-led
  bullet items into separate sentences (q03 fixed by the same
  change), but q32 still misses on scoring: the SEK 12,000 sentence
  shares only "monthly" and "allowance" with the question, while the
  page intro shares pioneering, women, stem and scholar.
- **Re-ingest rule**: after any re-fetch, run
  `python -m scripts.check_data_quality` and update this document
  from its report before claiming the numbers still hold.

## 11. Re-verification, 18 Nov 2026 (Phase 2, Step 7)

Every source page was re-fetched and compared with the stored
`.txt` before any launch claim: direct HTTP with the repo's own
extractor (`app.ingest.extract_text`) for all 19, plus a browser
text fetch for the 11 pages originally saved that way (Chevening,
DAAD, Gates Cambridge, Rhodes, Knight-Hennessy, Schwarzman,
Commonwealth Shared, RTP, Marshall, Erasmus Mundus organisations
page, SI Pioneering Women) and for RTP, whose direct fetch timed
out. No raw HTML was saved or committed; the secret scan over the
re-fetched text found 0 matches (counts only). Machine-readable
record: `benchmarks/results/reverification_2026-11-18.json`;
per-source date: `reverified_at` in `data/sources.json`.

**Result: all 19 pages live; 18 substantively unchanged, 1 changed
only in a sidebar teaser. No `data/raw/*.txt` or `*.json` was
rewritten** — a re-fetch that differs only by extraction method
(line-wrapping, punctuation spacing, nav duplication, table
flattening, a duplicated FAQ section) is not a page change, and
rewriting on that basis would churn the chunks and break the
pinned chunk ids for no fact change. The one real change (Eiffel)
is a recommended-article teaser title, not Eiffel content, so it
was recorded and the file left alone.

| # | Scholarship | Source URL | Status | Changed? | Re-verified |
|---|---|---|---|---|---|
| 1 | Chevening (UK) | https://www.chevening.org/scholarships/ | live | Unchanged — no deadline, no stipend figure; fully-funded, one-year master's | 2026-11-18 |
| 2 | Sweden SI Global Professionals | https://si.se/en/apply/scholarships/swedish-institute-scholarships-for-global-professionals/ | live | Unchanged — byte-identical | 2026-11-18 |
| 3 | Fulbright Foreign Student Program (USA) | https://foreign.fulbrightonline.org/about/foreign-student-program | live | Unchanged — byte-identical | 2026-11-18 |
| 4 | DAAD (Germany) | https://www.daad.de/en/study-and-research-in-germany/scholarships/ | live | Unchanged — 147 options; direct extraction differs only in line-wrapping | 2026-11-18 |
| 5 | Erasmus Mundus Joint Masters | https://erasmus-plus.ec.europa.eu/opportunities/opportunities-for-individuals/students/erasmus-mundus-joint-masters | live | Unchanged — byte-identical; marker still "Last updated: 26 June 2025" | 2026-11-18 |
| 6 | France Excellence Eiffel | https://www.campusfrance.org/en/the-france-excellence-eiffel-scholarship-program | live | **Changed (sidebar only)** — recommended-article teaser "MOPGA Research Themes" → "Co-financing of 12- to 24-month postdoctoral research contracts"; €1,200 / €2,000 and Oct 2022 / Dec 2025 markers unchanged | 2026-11-18 |
| 7 | Vanier Canada Graduate Scholarships | https://vanier.gc.ca/en/home-accueil.html | live | Unchanged — byte-identical; still closed | 2026-11-18 |
| 8 | Manaaki New Zealand Scholarships | https://www.nzscholarships.govt.nz/ | live | Unchanged — byte-identical | 2026-11-18 |
| 9 | Hong Kong PhD Fellowship Scheme (HKPFS) | https://cerg1.ugc.edu.hk/hkpfs/index.html | live | Unchanged — byte-identical | 2026-11-18 |
| 10 | MEXT (Japan) | https://www.studyinjapan.go.jp/en/planning/scholarships/mext-scholarships/ | live | Unchanged — byte-identical | 2026-11-18 |
| 11 | Gates Cambridge | https://www.gatescambridge.org/programme/the-scholarship/ | live | Unchanged — around 70/year, £23,152, US$210m | 2026-11-18 |
| 12 | Rhodes Scholarship (FAQs) | https://www.rhodeshouse.ox.ac.uk/scholarships/faqs/ | live | Unchanged — £21,060 for 2026-27; direct extraction is a single copy where the stored text has narrative+Q&A duplicated | 2026-11-18 |
| 13 | Knight-Hennessy Scholars — Funding (Stanford) | https://knight-hennessy.stanford.edu/program-overview/funding | live | Unchanged — up to three years | 2026-11-18 |
| 14 | Schwarzman Scholars (FAQs) | https://www.schwarzmanscholars.org/frequently-asked-questions/ | live | Unchanged — not yet 29 on 1 Aug | 2026-11-18 |
| 15 | Commonwealth Shared Scholarships | https://cscuk.fcdo.gov.uk/scholarships/commonwealth-shared-scholarships-applications/ | live | Unchanged — closing 8 December 2026, £1,712/month | 2026-11-18 |
| 16 | Research Training Program (RTP, Australia) | https://www.education.gov.au/research-block-grants/research-training-program | live | Unchanged — 2027 base $35,550, 2026 base $34,315 (direct fetch timed out; browser text fetch used) | 2026-11-18 |
| 17 | Marshall Scholarship | https://marshallscholarship.org/about/ | live | Unchanged — up to 50/year, closing 29 September 2026 | 2026-11-18 |
| 18 | Erasmus Mundus Joint Masters (organisations page) | https://erasmus-plus.ec.europa.eu/opportunities/organisations/cooperation-among-organisations-and-institutions/erasmus-mundus-joint-masters | live | Unchanged — €1,400/month, €5m/project; footer marker "Last updated: 15 April 2026" newly observed, not a verified change (§9) | 2026-11-18 |
| 19 | SI Scholarship Pioneering Women in STEM | https://si.se/en/apply/scholarships/pioneeringwomeninstem/ | live | Unchanged — SEK 12,000/month; key dates 8–24 Feb 2027 | 2026-11-18 |

What the re-verification did **not** change: no golden question
was updated (every expected answer still appears on its page —
verified per question during the re-fetch), no question was
dropped, the rebuilt index is byte-identical (101 chunks), and
the re-run eval is byte-identical to the before-run file
(`benchmarks/results/extractive_eval.json` ==
`extractive_eval_before_reverification.json`: answer-only 0.907,
recall@5 1.0, faithfulness 1.0, abstention 1.0, eligibility 1.0).
The next re-verification should re-run the same comparison; the
stale markers to watch are Erasmus (26 June 2025) and Eiffel
(October 2022), and the EMJM organisations footer (15 April 2026).
