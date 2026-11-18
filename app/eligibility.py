"""Eligibility-first answering.

Before the system answers a substantive question about a scholarship,
it checks whether the asker is eligible for that scholarship — degree
level and citizenship — using ONLY sentences that exist in the corpus.
Every verdict carries its reason and the verbatim evidence sentence
with its chunk id. When the corpus does not state a rule, the verdict
is "unknown" and the answer says so; the system never invents a rule
from general knowledge.

Why this exists: a scholarship answer is useless if the reader cannot
apply for it. Telling a Pakistani PhD applicant what Chevening funds
is worse than useless — Chevening funds one-year master's degrees,
and Pakistan is not among the SI Scholarship's 34 eligible
countries. Both facts are in the corpus, so the system must say so
first, with the citation, instead of answering the question as asked.

Design decisions:
- Rules are written in this file as a table, but each rule is
  validated at load time against the actual chunk text: if the
  evidence sentence is not found verbatim in the named chunk, the
  rule is dropped to "unknown" and a warning is printed. A rule
  that drifts from the corpus cannot silently survive a re-ingest.
- Degree levels are normalised to undergraduate / masters / phd.
  "Postgraduate" covers masters and phd. "Doctoral"/"doctorate"
  is phd. "Bachelor's" is undergraduate.
- Citizenship is one of four modes: "any" (the corpus says
  eligibility is irrespective of origin), "list" (an explicit
  country list in the corpus), "foreign" (the corpus says the
  scholarship is for foreign students of the host country), or
  "unknown" (the corpus states no citizenship rule).
- A verdict is "eligible" only when every check that the profile
  asks for is grounded and passes. One failed check makes the
  whole verdict "not_eligible". If no check can be grounded, the
  verdict is "unknown".
- A programme the corpus says is closed gets the distinct verdict
  "closed", whatever the profile is — even a profile that would
  otherwise pass every check. Closure is read from
  data/closures.json (driven by the data-quality checker's closure
  findings), validated verbatim against chunk text at load time
  exactly like the rules above: an entry whose evidence is not
  found in the named chunk is dropped with a warning and the
  programme is treated as open. Closure is checked before any
  degree or citizenship check, so a closed programme is never
  reported as eligible.
"""
import json
import re
import warnings
from pathlib import Path
from typing import Optional

from app.answer import _named_docs
from app.config import CLOSURES_FILE
from app.textnorm import norm_text as _norm_text

# ---------------------------------------------------------------- rules

# Canonical levels: "undergraduate", "masters", "phd".
# "postgraduate" in a rule expands to {"masters", "phd"}.
_LEVEL_SYNONYMS = {
    "undergraduate": "undergraduate", "bachelor": "undergraduate",
    "bachelors": "undergraduate", "bachelor's": "undergraduate",
    "masters": "masters", "master": "masters", "master's": "masters",
    "phd": "phd", "doctorate": "phd", "doctoral": "phd",
    "doctor": "phd", "postgraduate": "postgraduate",
}

# Each rule: doc -> {"levels": set[str] | None, "citizenship": {...},
# "evidence": {field: (sentence, chunk_id)}}.
# levels None means the corpus states no degree-level rule for this
# scholarship. citizenship None means the corpus states no
# citizenship rule. Evidence sentences were copied verbatim from the
# chunks named; load_rules() verifies them.
RULES = {
    "chevening": {
        "levels": {"masters"},
        "citizenship": None,
        "evidence": {
            "levels": ("Chevening Scholarships enable outstanding emerging leaders "
                       "from all over the world to pursue one-year master's degrees in the UK.",
                       "chevening-0"),
        },
    },
    "sweden-si": {
        "levels": {"masters"},
        "citizenship": {
            "mode": "list",
            "countries": ["Armenia", "Azerbaijan", "Bangladesh", "Belarus",
                          "Bolivia", "Brazil", "Colombia", "Ecuador", "Egypt",
                          "Ethiopia", "Georgia", "Guatemala", "Indonesia",
                          "Iraq", "Jordan", "Kenya", "Liberia", "Malaysia",
                          "Mexico", "Moldova", "Morocco", "Nigeria", "Peru",
                          "Philippines", "Rwanda", "Senegal", "South Africa",
                          "Tanzania", "Thailand", "Uganda", "Ukraine",
                          "Uzbekistan", "Vietnam", "Zambia"],
        },
        "evidence": {
            "levels": ("The scholarship supports full-time, one- or two-year "
                       "master’s studies at Swedish universities within the "
                       "following study fields: Governance.",
                       "sweden-si-0"),
            "citizenship": ("Citizens of the following countries are eligible to apply "
                            "for the SI Scholarship for Global Professionals: Armenia, "
                            "Azerbaijan, Bangladesh, Belarus*, Bolivia, Brazil, Colombia, "
                            "Ecuador, Egypt, Ethiopia, Georgia*, Guatemala, Indonesia, "
                            "Iraq, Jordan, Kenya, Liberia, Malaysia, Mexico, Moldova, "
                            "Morocco, Nigeria, Peru, Philippines, Rwanda, Senegal, South "
                            "Africa, Tanzania, Thailand, Uganda, Ukraine, Uzbekistan, "
                            "Vietnam, Zambia.",
                            "sweden-si-2"),
        },
    },
    "fulbright": {
        "levels": None,
        "citizenship": None,
        "evidence": {},
    },
    "daad": {
        "levels": None,
        "citizenship": {"mode": "international", "host": "Germany"},
        "evidence": {
            "citizenship": ("Here you will find information about our DAAD scholarship "
                            "programmes for international students and researchers as well "
                            "as offers from other selected funding organisations.",
                            "daad-0"),
        },
    },
    "erasmus-mundus": {
        "levels": {"masters"},
        "citizenship": {"mode": "any"},
        "evidence": {
            "levels": ("You must have a bachelor’s degree (first degree), or be in your "
                       "last year of bachelor studies graduate before the master’s "
                       "programme starts.",
                       "erasmus-mundus-1"),
            "citizenship": ("Masters may also offer full scholarships to the best-ranked "
                            "students worldwide.",
                            "erasmus-mundus-0"),
        },
    },
    "eiffel": {
        "levels": {"masters", "phd"},
        "citizenship": {"mode": "foreign", "host": "France"},
        "evidence": {
            "levels": ("The France Excellence Eiffel scholarship program is a tool "
                       "developed by the French Ministry of Foreign Affairs to assist "
                       "French institutions of higher education in attracting the best "
                       "foreign students to Master's and Doctoral/Phd degree programs.",
                       "eiffel-0"),
            "citizenship": ("The France Excellence Eiffel scholarship program is a tool "
                            "developed by the French Ministry of Foreign Affairs to assist "
                            "French institutions of higher education in attracting the best "
                            "foreign students to Master's and Doctoral/Phd degree programs.",
                            "eiffel-0"),
        },
    },
    "vanier": {
        "levels": {"phd"},
        "citizenship": {"mode": "any"},
        "evidence": {
            "levels": ("Valued at $50,000 per year for three years during doctoral studies.",
                       "vanier-0"),
            "citizenship": ("Citizenship: Canadian citizens. Permanent residents of "
                            "Canada. Foreign citizens.",
                            "vanier-0"),
        },
    },
    "manaaki": {
        "levels": {"undergraduate", "masters", "phd"},
        "citizenship": {"mode": "international", "host": "New Zealand"},
        "evidence": {
            "levels": ("Scholarships for undergraduate and postgraduate study.",
                       "manaaki-0"),
            "citizenship": ("Scholarships for international tertiary students.",
                            "manaaki-0"),
        },
    },
    "hkpfs": {
        "levels": {"phd"},
        "citizenship": {"mode": "any"},
        "evidence": {
            "levels": ("Established by the Research Grants Council (RGC) of Hong Kong in "
                       "2009, the Hong Kong PhD Fellowship Scheme (HKPFS) aims at "
                       "attracting the best and brightest students in the world to pursue "
                       "their PhD studies in Hong Kong's universities.",
                       "hkpfs-0"),
            "citizenship": ("Candidates who are seeking admission as new full time PhD "
                            "students in the following eight universities, irrespective of "
                            "their country of origin, prior work experience and ethnic "
                            "background, should be eligible to apply.",
                            "hkpfs-0"),
        },
    },
    "mext": {
        "levels": {"undergraduate", "masters", "phd"},
        "citizenship": None,
        "evidence": {
            "levels": ("You must be under 35 years of age and be eligible for admission "
                       "to a master's program or a doctoral program (first or second "
                       "stage) at a Japanese graduate school.",
                       "mext-1"),
        },
    },
    # Phase 2 improvement, Step 4: rules for the 9 Step 6 documents,
    # authored only where the corpus states a rule with a verbatim
    # evidence sentence (same standard as the rules above).
    "gates-cambridge": {
        "levels": {"masters", "phd"},
        "citizenship": {"mode": "foreign", "host": "UK"},
        "evidence": {
            "levels": ("Each year Gates Cambridge offers around 70 full-cost "
                       "scholarships to outstanding applicants from countries "
                       "outside the UK to pursue a postgraduate degree in any "
                       "subject available at the University of Cambridge.",
                       "gates-cambridge-0"),
            "citizenship": ("Each year Gates Cambridge offers around 70 full-cost "
                            "scholarships to outstanding applicants from countries "
                            "outside the UK to pursue a postgraduate degree in any "
                            "subject available at the University of Cambridge.",
                            "gates-cambridge-0"),
        },
    },
    # Levels only: the corpus states the Rhodes is postgraduate and
    # points undergraduate seekers elsewhere, but eligibility by
    # country runs through per-constituency rules with no country
    # list in the corpus, so no citizenship rule is authored.
    # Known limit: a second undergraduate degree via Senior Status
    # is possible in limited cases; this rule reports the standard
    # postgraduate-only position.
    "rhodes": {
        "levels": {"masters", "phd"},
        "citizenship": None,
        "evidence": {
            "levels": ("The Rhodes Scholarship is a fully funded, full-time "
                       "postgraduate scholarship which enables talented people "
                       "from around the world to study at the University of Oxford.",
                       "rhodes-2"),
        },
    },
    # Levels only: "any graduate degree(s) at Stanford". The funding
    # page in the corpus states no citizenship rule.
    "knight-hennessy": {
        "levels": {"masters", "phd"},
        "citizenship": None,
        "evidence": {
            "levels": ("Knight-Hennessy scholars receive funding for up to three "
                       "years for any graduate degree(s) at Stanford.",
                       "knight-hennessy-0"),
        },
    },
    # No rule: the FAQ in the corpus never states the funded degree
    # level (it only requires applicants to have completed an
    # undergraduate degree) and the 40/20/40 figures are class
    # composition targets, explicitly "not specific national or
    # regional targets", so no citizenship rule is authored either.
    # Verdict stays unknown, by design.
    "schwarzman": {
        "levels": None,
        "citizenship": None,
        "evidence": {},
    },
    "commonwealth-shared": {
        "levels": {"masters"},
        "citizenship": {
            "mode": "list",
            "countries": ["Bangladesh", "Belize", "Botswana", "Dominica",
                          "Eswatini", "Fiji", "Gabon", "Ghana", "Grenada",
                          "Guyana", "India", "Jamaica", "Kenya", "Kiribati",
                          "Lesotho", "Malawi", "Malaysia", "Maldives",
                          "Mauritius", "Montserrat", "Mozambique", "Namibia",
                          "Nauru", "Nigeria", "Pakistan", "Papua New Guinea",
                          "Rwanda", "Saint Helena", "Saint Lucia",
                          "Saint Vincent and The Grenadines", "Samoa",
                          "Sierra Leone", "Solomon Islands", "South Africa",
                          "Sri Lanka", "Tanzania", "The Gambia", "Togo",
                          "Tonga", "Tuvalu", "Uganda", "Vanuatu", "Zambia"],
        },
        "evidence": {
            "levels": ("Please note, these scholarships are for Master's courses "
                       "only and not for undergraduate or PhD study.",
                       "commonwealth-shared-1"),
            "citizenship": ("Bangladesh, Belize, Botswana, Dominica, Eswatini, Fiji, "
                            "Gabon, Ghana, Grenada, Guyana, India, Jamaica, Kenya, "
                            "Kiribati, Lesotho, Malawi, Malaysia, Maldives, Mauritius, "
                            "Montserrat, Mozambique, Namibia, Nauru, Nigeria, Pakistan, "
                            "Papua New Guinea, Rwanda, Saint Helena, Saint Lucia. Saint "
                            "Vincent and The Grenadines. Samoa, Sierra Leone, Solomon "
                            "Islands, South Africa, Sri Lanka, Tanzania, The Gambia, "
                            "Togo, Tonga, Tuvalu, Uganda, Vanuatu, Zambia.",
                            "commonwealth-shared-5"),
        },
    },
    "rtp": {
        "levels": {"masters", "phd"},
        "citizenship": {"mode": "any"},
        "evidence": {
            "levels": ("The Australian Government provides funding to eligible HEPs "
                       "through the RTP to support their training of domestic and "
                       "overseas students undertaking research doctorate and "
                       "research masters degrees.",
                       "rtp-0"),
            "citizenship": ("The Australian Government provides funding to eligible HEPs "
                            "through the RTP to support their training of domestic and "
                            "overseas students undertaking research doctorate and "
                            "research masters degrees.",
                            "rtp-0"),
        },
    },
    "marshall": {
        "levels": {"masters", "phd"},
        "citizenship": {"mode": "list", "countries": ["United States"]},
        "evidence": {
            "levels": ("Up to fifty Scholars are selected each year to study at "
                       "graduate level at an UK institution in any field of study.",
                       "marshall-0"),
            "citizenship": ("Open only to United States citizens who (at the time they "
                            "take up their Scholarship) hold a first degree from an "
                            "accredited four-year college or university in the United "
                            "States with a minimum GPA of 3.7.",
                            "marshall-0"),
        },
    },
    "erasmus-mundus-org": {
        "levels": {"masters"},
        "citizenship": {"mode": "any"},
        "evidence": {
            "levels": ("Erasmus Mundus Joint Masters are high-level and integrated "
                       "study programmes, at master level.",
                       "erasmus-mundus-org-0"),
            "citizenship": ("The EU provides funding for both running these programmes "
                            "and offering competitive scholarships to the best students "
                            "worldwide.",
                            "erasmus-mundus-org-0"),
        },
    },
    # Known limit: the checker models degree level and citizenship
    # only, not gender. The corpus also requires applicants to be
    # female citizens of the listed countries; a male applicant from
    # a listed country would be reported eligible here. The rule
    # covers what the checker can decide.
    "si-pioneering-women": {
        "levels": {"masters"},
        "citizenship": {
            "mode": "list",
            "countries": ["Bangladesh", "Indonesia", "Kenya", "Malaysia",
                          "Nigeria", "Philippines", "Rwanda", "South Africa",
                          "Thailand", "Vietnam"],
        },
        "evidence": {
            "levels": ("The scholarship is awarded for master's studies in Sweden "
                       "to women in Africa and Asia who are committed to advancing "
                       "their careers and driving change through STEM.",
                       "si-pioneering-women-0"),
            "citizenship": ("Female citizens of the following countries are eligible "
                            "to apply for Pioneering Women in STEM: Bangladesh, "
                            "Indonesia, Kenya, Malaysia, Nigeria, Philippines, Rwanda, "
                            "South Africa, Thailand, Vietnam.",
                            "si-pioneering-women-2"),
        },
    },
}

# A question is an eligibility question when it asks about applying
# or being eligible, rather than about a fact (amount, deadline).
ELIGIBILITY_QUESTION_RE = re.compile(
    r"\b(eligib|can i apply|am i eligible|can i get|do i qualify|"
    r"who can apply|can .* apply)\b", re.IGNORECASE)

# Display names for verdicts, used in answers.
_DOC_DISPLAY = {
    "chevening": "the Chevening Scholarship",
    "sweden-si": "the SI Scholarship for Global Professionals",
    "fulbright": "the Fulbright Foreign Student Program",
    "daad": "DAAD scholarships",
    "erasmus-mundus": "an Erasmus Mundus Joint Master",
    "eiffel": "the France Excellence Eiffel Scholarship",
    "vanier": "the Vanier Canada Graduate Scholarship",
    "manaaki": "a Manaaki New Zealand Scholarship",
    "hkpfs": "the Hong Kong PhD Fellowship Scheme",
    "mext": "a MEXT Scholarship",
    "gates-cambridge": "the Gates Cambridge Scholarship",
    "rhodes": "the Rhodes Scholarship",
    "knight-hennessy": "the Knight-Hennessy Scholarship",
    "schwarzman": "the Schwarzman Scholarship",
    "commonwealth-shared": "a Commonwealth Shared Scholarship",
    "rtp": "an RTP Scholarship",
    "marshall": "the Marshall Scholarship",
    "erasmus-mundus-org": "an Erasmus Mundus Joint Master",
    "si-pioneering-women": "the SI Scholarship Pioneering Women in STEM",
}


def normalize_level(raw: Optional[str]) -> Optional[str]:
    """Normalise a free-text degree level to undergraduate/masters/phd."""
    if not raw:
        return None
    key = raw.strip().lower().rstrip("s")
    if key in ("bachelor",):
        return "undergraduate"
    return _LEVEL_SYNONYMS.get(raw.strip().lower())


def normalize_country(raw: Optional[str]) -> Optional[str]:
    """Normalise a country name for comparison (case and whitespace)."""
    if not raw:
        return None
    return " ".join(raw.strip().lower().split())


def load_closures(chunks: list[dict], path: Path = CLOSURES_FILE) -> dict[str, dict]:
    """Read the closure flag file and validate it against the corpus.

    Returns {doc: {"replacement": str, "evidence": [{sentence, chunk_id}]}}.
    Every evidence sentence must be found verbatim (normalised) in
    the chunk it cites, and that chunk must belong to the doc; a
    sentence that fails is dropped with a warning, and an entry
    with no surviving evidence is dropped entirely — the same
    fail-safe as the eligibility rules. A missing or unreadable
    flag file means no programme is flagged closed (warn, don't
    crash): closure is a positive claim that needs its file.
    """
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        warnings.warn(f"eligibility: closure file {path} unreadable ({exc}); "
                      "no programme flagged closed")
        return {}
    by_id = {c["chunk_id"]: c for c in chunks}
    closures: dict[str, dict] = {}
    for entry in data.get("closures", []):
        doc = entry.get("doc")
        if entry.get("status") != "closed" or not doc:
            continue
        replacement = entry.get("replacement") or ""
        kept: list[dict] = []
        for ev in entry.get("evidence", []):
            sentence, chunk_id = ev.get("sentence", ""), ev.get("chunk_id", "")
            chunk = by_id.get(chunk_id)
            if chunk is None or chunk["doc"] != doc:
                warnings.warn(f"eligibility: closure evidence chunk {chunk_id} missing "
                              f"for {doc}; sentence dropped")
                continue
            if _norm_text(sentence) not in _norm_text(chunk["text"]):
                warnings.warn(f"eligibility: closure evidence sentence for {doc} not found "
                              f"verbatim in {chunk_id}; sentence dropped")
                continue
            kept.append({"sentence": sentence, "chunk_id": chunk_id})
        if not kept:
            warnings.warn(f"eligibility: closure entry for {doc} has no validated "
                          "evidence; programme treated as open")
            continue
        closures[doc] = {"replacement": replacement, "evidence": kept}
    return closures


class EligibilityChecker:
    """Checks a profile against the corpus-grounded rules.

    `chunks` is the processed chunk list; load validates every rule's
    evidence sentence against the chunk it claims to come from. A
    rule whose evidence is missing is dropped to unknown, so the
    checker can never cite a sentence that is not in the corpus.
    Closure entries from data/closures.json are validated the same
    way and checked first in check().
    """

    def __init__(self, chunks: list[dict], closures_file: Path = CLOSURES_FILE):
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.closures = load_closures(chunks, closures_file)
        self.rules = {}
        for doc, rule in RULES.items():
            kept = {"levels": rule["levels"], "citizenship": rule["citizenship"],
                    "evidence": {}}
            for field, (sentence, chunk_id) in rule["evidence"].items():
                chunk = self.chunks.get(chunk_id)
                if chunk is None or chunk["doc"] != doc:
                    warnings.warn(f"eligibility: evidence chunk {chunk_id} missing for {doc}; "
                                  f"{field} rule dropped")
                    kept["levels" if field == "levels" else "citizenship"] = None
                    continue
                if _norm_text(sentence) not in _norm_text(chunk["text"]):
                    warnings.warn(f"eligibility: evidence sentence for {doc}/{field} not found "
                                  f"verbatim in {chunk_id}; rule dropped")
                    if field == "levels":
                        kept["levels"] = None
                    else:
                        kept["citizenship"] = None
                    continue
                kept["evidence"][field] = {"sentence": sentence, "chunk_id": chunk_id}
            self.rules[doc] = kept

    def check(self, profile: dict, doc: str) -> dict:
        """Verdict for one scholarship. Profile keys: nationality,
        degree_level (free text, normalised here).

        A closed programme short-circuits everything: the verdict
        is "closed" whatever the profile, with the closure evidence
        and the replacement programme. Degree and citizenship are
        not checked — they are moot when nobody can apply, and
        reporting "doctoral only" for a closed programme would
        mislead a qualified applicant into thinking they can apply.
        """
        display = _DOC_DISPLAY.get(doc, doc)
        closed = self.closures.get(doc)
        if closed is not None:
            replacement = closed["replacement"]
            reasons = [
                f"{display} is closed: applications are no longer accepted.",
            ]
            if replacement:
                reasons.append(
                    f"It has been replaced by the {replacement}; see that "
                    "programme's page for current information.")
            return {"doc": doc, "verdict": "closed", "reasons": reasons,
                    "evidence": list(closed["evidence"])}
        rule = self.rules.get(doc)
        if rule is None:
            return {"doc": doc, "verdict": "unknown",
                    "reasons": [f"The corpus has no eligibility rules for {doc}."],
                    "evidence": []}
        reasons: list[str] = []
        evidence: list[dict] = []
        grounded = False
        failed = False

        level = normalize_level(profile.get("degree_level"))
        if level:
            article = "an" if level[0] in "aeiou" else "a"
            if rule["levels"] is None:
                reasons.append(
                    f"The corpus does not state which degree levels {display} covers, "
                    "so the degree check cannot be decided from the corpus.")
            else:
                grounded = True
                ev = rule["evidence"]["levels"]
                evidence.append(ev)
                if level in rule["levels"]:
                    levels = ", ".join(sorted(rule["levels"]))
                    reasons.append(
                        f"{display} covers {levels} level study, so {article} {level} "
                        "applicant meets the degree requirement.")
                else:
                    failed = True
                    levels = ", ".join(sorted(rule["levels"]))
                    reasons.append(
                        f"{display} covers {levels} level study only, so {article} {level} "
                        "applicant is not eligible on degree level.")

        nationality = normalize_country(profile.get("nationality"))
        if nationality:
            citizenship = rule["citizenship"]
            if citizenship is None:
                reasons.append(
                    f"The corpus does not state a citizenship rule for {display}, "
                    "so the citizenship check cannot be decided from the corpus.")
            else:
                grounded = True
                mode = citizenship["mode"]
                ev = rule["evidence"].get("citizenship")
                if ev:
                    evidence.append(ev)
                if mode == "any":
                    reasons.append(
                        f"{display} is open irrespective of country of origin, so a "
                        f"{profile['nationality']} citizen meets the citizenship requirement.")
                elif mode == "list":
                    countries = citizenship["countries"]
                    if nationality in {c.lower() for c in countries}:
                        reasons.append(
                            f"{profile['nationality']} is among the {len(countries)} eligible "
                            f"countries for {display}, so the citizenship requirement is met.")
                    else:
                        failed = True
                        reasons.append(
                            f"{profile['nationality']} is not among the {len(countries)} "
                            f"eligible countries for {display}, so the citizenship "
                            "requirement is not met.")
                elif mode in ("foreign", "international"):
                    host = citizenship["host"]
                    if nationality == host.lower():
                        failed = True
                        reasons.append(
                            f"{display} is for international (non-{host}) students, so a "
                            f"{host} citizen is not eligible on citizenship.")
                    else:
                        reasons.append(
                            f"{display} is for international students, so a "
                            f"{profile['nationality']} citizen meets the citizenship requirement.")

        if failed:
            verdict = "not_eligible"
        elif grounded:
            verdict = "eligible"
        else:
            verdict = "unknown"
        return {"doc": doc, "verdict": verdict, "reasons": reasons,
                "evidence": evidence}

    def check_many(self, profile: dict, docs: list[str]) -> list[dict]:
        seen: list[str] = []
        out = []
        for doc in docs:
            if doc not in seen:
                seen.append(doc)
                out.append(self.check(profile, doc))
        return out


def eligibility_statement(verdict: dict) -> str:
    """One human-readable answer from a verdict, for eligibility
    questions and for the eligibility-first override. The verbatim
    evidence sentence is appended so the answer is self-contained:
    the reader sees the rule as the official page states it, not
    just the system's paraphrase of it."""
    display = _DOC_DISPLAY.get(verdict["doc"], verdict["doc"])
    if verdict["verdict"] == "closed":
        lead = f"Closed: {display}."
    elif verdict["verdict"] == "not_eligible":
        lead = f"Not eligible for {display}."
    elif verdict["verdict"] == "eligible":
        lead = f"Eligible for {display}."
    else:
        lead = (f"Eligibility for {display} cannot be decided from the corpus.")
    text = lead + " " + " ".join(
        reason[0].upper() + reason[1:] if reason else reason
        for reason in verdict["reasons"])
    for ev in verdict["evidence"]:
        text += f" Evidence: {ev['sentence']}"
    return text


def resolve(question: str, profile: Optional[dict], retrieved: list[dict],
            chunks: list[dict], substantive: dict) -> tuple[Optional[dict], list[dict]]:
    """Eligibility-first routing.

    Returns (eligibility_answer_or_None, eligibility_list). When the
    question names a scholarship and the profile makes the asker
    not eligible for it, the eligibility answer replaces the
    substantive answer — answering the fact question would mislead.
    When the question itself asks about eligibility, the eligibility
    statement is the answer. Otherwise the substantive answer stands
    and the eligibility list is attached as supplementary information.
    """
    if not profile:
        return None, []
    checker = EligibilityChecker(chunks)
    docs = [c["doc"] for c in retrieved]
    eligibility = checker.check_many(profile, docs)
    named = _named_docs(question)
    if not named:
        return None, eligibility
    named_verdicts = [v for v in eligibility if v["doc"] in named]
    if not named_verdicts:
        return None, eligibility
    verdict = named_verdicts[0]
    if verdict["verdict"] in ("not_eligible", "closed") or ELIGIBILITY_QUESTION_RE.search(question):
        chunk_by_id = {c["chunk_id"]: c for c in chunks}
        citations = [
            {"doc": verdict["doc"], "chunk_id": ev["chunk_id"],
             "score": None, "text": chunk_by_id[ev["chunk_id"]]["text"]}
            for ev in verdict["evidence"]
            if ev["chunk_id"] in chunk_by_id
        ]
        answer = {
            "question": question,
            "found": True,
            "answer": eligibility_statement(verdict),
            "score": substantive.get("score", 0.0),
            "citations": citations,
            "eligibility": eligibility,
            # Faithfulness for an eligibility answer is judged on the
            # verbatim evidence sentences the verdict rests on, not on
            # the generated verdict wording.
            "answer_sentences": [ev["sentence"] for ev in verdict["evidence"]],
        }
        return answer, eligibility
    return None, eligibility
