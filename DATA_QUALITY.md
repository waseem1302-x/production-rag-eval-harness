# Data Quality — Messy Real Scholarship Data

The corpus is 10 real official scholarship pages, fetched 18 Nov 2026.
Real pages are messy. This document records every mess found, the
exact evidence for it, what it means for the system, and how it is
handled. Nothing here is hidden: the checker that produced these
findings is `scripts/check_data_quality.py`, and its machine-readable
report is `benchmarks/results/data_quality.json` (36 findings:
8 resolved, 28 open). Re-run the checker after any re-ingest;
`DATA_QUALITY.md` must be updated from its output, not from memory.

## Summary of findings

| # | Issue | Documents | Status |
|---|---|---|---|
| 1 | Programme closed / replaced | Vanier | Resolved (documented; eligibility returns `closed` since Phase 2, Step 4) |
| 2 | Two stipend amounts, one page | Eiffel, HKPFS | Resolved (documented) |
| 3 | Page is a database, not one award | DAAD | Resolved (documented) |
| 4 | Approximate figures | Fulbright, HKPFS | Open — quote with hedge |
| 5 | Year-specific figures | HKPFS | Open — quote with year |
| 6 | Deadline not stated on fetched page | 9 of 10 docs | Open — must abstain |
| 7 | Stipend not stated on fetched page | 7 of 10 docs | Open — must abstain |
| 8 | Degree levels / citizenship not stated | Fulbright, DAAD, MEXT, Chevening | Open — verdict `unknown` |
| 9 | Stale page markers | Erasmus, Eiffel | Open — re-fetch before launch |

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

## 5. Year-specific figures

- HKPFS stipend HK$344,400 is for the **2026/27** academic year.
- HKPFS "400 PhD Fellowships" is for the **2027/28** academic year.
  Quoting either figure without its year is wrong, and both will go
  stale: the checker flags any figure tied to an academic year so a
  re-ingest can catch it.

## 6. Deadline not stated on the fetched page (9 of 10)

Only the SI page carries specific dates (University Admissions
16 Oct 2026–15 Jan 2027; SI application 8–24 Feb 2027; admissions
announced 1 Apr 2027; scholarship announced 22 Apr 2027).

| Document | Deadline on fetched page? |
|---|---|
| sweden-si | Yes — 15 Jan 2027 (programmes), 24 Feb 2027 (scholarship) |
| chevening | No |
| fulbright | No — mentions "university deadline" but no date (country-specific) |
| daad | No — database, see §3 |
| erasmus-mundus | No — the only date is "Last updated: 26 June 2025", a page marker, not a deadline |
| eiffel | No — "contact the international relations department" for deadlines |
| vanier | No — and the programme is closed, see §1 |
| manaaki | No |
| hkpfs | No |
| mext | No |

The system must answer "not found in the corpus" for these, never
a guessed date. The golden set tests this: a Chevening deadline
question and a DAAD deadline question both abstain.

## 7. Stipend not stated on the fetched page (7 of 10)

Only Vanier ($50,000/year for three years, doctoral), Eiffel
(€1,200 / €2,000 per month, see §2) and HKPFS (HK$344,400/year,
see §2) state an amount. Chevening ("fully-funded", no figure),
Sweden SI, Fulbright, DAAD, Erasmus Mundus, Manaaki and MEXT state
no amount on the fetched page.

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

## How the system uses this

- **Abstention**: questions about a missing field return "Not found
  in the corpus." with a recorded reason (`missing_specific_term`
  / `weak_overlap`). Measured: abstention accuracy 0.8 (4/5).
- **Eligibility**: `unknown` verdicts where the corpus states no
  rule (Fulbright e10); `closed` where the corpus says the programme
  is closed (Vanier e02, e05, e11 — Phase 2, Step 4). Measured:
  eligibility accuracy 1.0 (11/11).
- **Golden set**: q21 (Vanier closure), q22/q23 (Eiffel amounts by
  level) were added for this step and all pass; e11 (Vanier doctoral
  applicant, verdict `closed`) was added in Phase 2, Step 4 and passes.
- **Re-ingest rule**: after any re-fetch, run
  `python -m scripts.check_data_quality` and update this document
  from its report before claiming the numbers still hold.
