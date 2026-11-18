# Data Quality — Messy Real Scholarship Data

The corpus is 19 real official scholarship pages, fetched 18 Nov 2026
(the original 10 plus 9 more in Phase 2, Step 6). Real pages are
messy. This document records every mess found, the exact evidence
for it, what it means for the system, and how it is handled. Nothing
here is hidden: the checker that produced these findings is
`scripts/check_data_quality.py`, and its machine-readable
report is `benchmarks/results/data_quality.json` (72 findings:
9 resolved, 63 open). Re-run the checker after any re-ingest;
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
| 7 | Stipend not stated on fetched page | 14 of 19 docs | Open — must abstain |
| 8 | Degree levels / citizenship not stated | 11 docs missing degree levels, 12 missing a citizenship rule (all 9 new docs, by construction — no rules authored yet) | Open — verdict `unknown` |
| 9 | Stale page markers | Erasmus, Eiffel | Open — re-fetch before launch |
| 10 | Checker currency gap (£/SEK not matched) | Rhodes, Commonwealth Shared, SI Pioneering Women flagged missing though they state amounts; Gates not flagged though its stipend is in £ (a US$210m donation matches instead) | Open — checker limitation, pinned by a test; the pages DO state amounts |

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

### HKPFS — resolved, one figure in two currencies

Chunk `hkpfs-0`:

> "In the 2026/27 academic year, the Fellowship provides an annual stipend of HK$344,400 (approximately US$44,150)…"

HK$344,400 and ~US$44,150 are the same stipend in two currencies,
not two stipends. Quote both with the year, and keep
"approximately" on the US dollar figure. The 400 fellowships figure
is separate and year-specific: "400 PhD Fellowships will be awarded
in the 2027/28 academic year".

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

## 7. Stipend not stated on the fetched page (14 of 19, by the checker)

The checker counts a stipend as stated when the text contains a
€/$/HK$/US$ amount. By that rule it flags 14 of 19 pages as missing
a stipend, and counts 5 as stating one: Eiffel, Erasmus Mundus
(organisations), Gates Cambridge, HKPFS and Vanier. Read the rule
honestly, in both directions:

- **Gates is a false positive.** The amount the checker matched is
  "US$210m" — the Trust's founding donation, not a stipend. The
  page's real stipend (£23,152) is in pounds, which the pattern
  does not match (see §10). So the checker's "states a stipend"
  really means "contains a €/$ amount somewhere".
- **RTP is a false negative.** The page states the rates, but
  written as "Full-time base RTP stipend rate ($) 35,550" — the
  dollar sign detached from the digits by a parenthesis — so the
  pattern finds nothing and the page is flagged missing.
- The true stipend statements on the fetched pages: Vanier
  ($50,000/year, doctoral), Eiffel (€1,200 / €2,000 per month, see
  §2), HKPFS (HK$344,400/year, see §2), Erasmus Mundus
  (organisations) (€1,400/month, see §2), Gates (£23,152), Rhodes
  (£21,060 in 2026-27, golden q25), Commonwealth Shared (£1,712
  per month, £2,000 in London), SI Pioneering Women (SEK 12,000
  per month, golden q32), and RTP (base $35,550/year for 2027;
  $34,315 for 2026, golden q29). Chevening ("fully-funded", no
  figure), Sweden SI, Fulbright, DAAD, Erasmus Mundus (student
  page), Manaaki, MEXT, Knight-Hennessy, Schwarzman and Marshall
  state no amount on the fetched page.

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
- Eiffel (chunk `eiffel-1`): "Mise à jour. October 2022." for the
  main page content — about 4 years old — while the brochures were
  updated December 2025 (chunk `eiffel-2`). The amounts are marked
  "from January 2026", so they are current, but the page's own
  update marker is stale. Re-fetch before launch and re-run the
  checker.

## 10. Checker currency gap: £ and SEK amounts are missed

The checker's money pattern (`MONEY_RE` in
`scripts/check_data_quality.py`) matches €, $ and HK$/US$ only —
not £ or SEK. So it flags "stipend missing" for three pages that DO
state an amount:

- Rhodes: £21,060 for 2026-27 (golden q25).
- Commonwealth Shared: £1,712 per month (£2,000 in London).
- SI Pioneering Women: SEK 12,000 per month (golden q32).

And it fails to flag a fourth for the wrong reason: Gates Cambridge
is NOT flagged, because the page's "US$210m" founding donation
matches the pattern, while its real stipend (£23,152) is in pounds.

The same pattern gap is why RTP is flagged (§7: "($) 35,550").
This is a checker limitation, not a corpus gap — pinned by
`tests/test_corpus_growth.py::test_stipend_flag_is_a_currency_gap`,
so a future checker fix (adding £/SEK/¥, and amounts with the
currency sign detached from the digits) shows up as a changed
count. Until then, the "14 of 19" stipend-missing count in §7 is
the checker's count, not the truth.

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
  a documented miss (bullet-list merging — the SEK 12,000 sentence
  is merged into one long list sentence by the splitter).
- **Re-ingest rule**: after any re-fetch, run
  `python -m scripts.check_data_quality` and update this document
  from its report before claiming the numbers still hold.
