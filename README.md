# Production RAG with Eval Harness — Phase 1

A small, honest, production-shaped RAG system over public official
scholarship pages. It answers questions **only from the fetched corpus**,
cites the exact chunks it used, and says **"Not found in the corpus"**
when the evidence is weak — instead of hallucinating.

Phase 1 is **retrieval + extractive answers + a measured eval harness**.
There is no generative LLM yet. That is deliberate: if retrieval is wrong,
a generator only hides it. Phase 2 adds a generator and an LLM-as-judge.

## Corpus (Phase 1: 10 documents)

Public official pages for 10 of the 21 family scholarships from
The Next Scholar. One page per scholarship, fetched 18 Nov 2026.
See `data/sources.json` for the exact URL, fetch method, HTTP status
and fetch date of every document. Raw HTML and extracted text live in
`data/raw/`.

| # | Scholarship | Source |
|---|-------------|--------|
| 1 | Chevening (UK) | https://www.chevening.org/scholarships/ |
| 2 | Sweden SI Global Professionals | https://si.se/en/apply/scholarships/swedish-institute-scholarships-for-global-professionals/ |
| 3 | Fulbright Foreign Student Program (USA) | https://foreign.fulbrightonline.org/about/foreign-student-program |
| 4 | DAAD (Germany) | https://www.daad.de/en/study-and-research-in-germany/scholarships/ |
| 5 | Erasmus Mundus Joint Masters | https://erasmus-plus.ec.europa.eu/opportunities/opportunities-for-individuals/students/erasmus-mundus-joint-masters |
| 6 | France Excellence Eiffel | https://www.campusfrance.org/en/the-france-excellence-eiffel-scholarship-program |
| 7 | Vanier Canada Graduate Scholarships | https://vanier.gc.ca/en/home-accueil.html |
| 8 | Manaaki New Zealand Scholarships | https://www.nzscholarships.govt.nz/ |
| 9 | Hong Kong PhD Fellowship Scheme (HKPFS) | https://cerg1.ugc.edu.hk/hkpfs/index.html |
| 10 | MEXT (Japan) | https://www.studyinjapan.go.jp/en/planning/scholarships/mext-scholarships/ |

Chevening returned HTTP 403 to a direct `curl`/`requests` fetch, so its
text was fetched with a text fetcher and saved with
`fetch_method: "browser-text-fetch"`. The other 9 returned HTTP 200.
No Reddit data. No tutorial corpus. No scraped private data.

## How it works

```
official pages -> ingest.py (fetch + extract text, save raw + metadata)
  -> chunking.py (recursive, ~512 tokens, 200-char overlap)
  -> embeddings.py (sentence-transformers all-MiniLM-L6-v2, 384-dim)
  -> retrieval.py (FAISS dense + BM25, 0.5/0.5 hybrid, lexical rerank)
  -> answer.py (extractive answer + citations + abstention)
  -> api.py (FastAPI, POST /ask)
  -> eval.py (golden set, recall@k, faithfulness, answer relevance)
```

### Chunking choice (why 512 tokens, why overlap)

Target **512 tokens, ~200 characters overlap (~50 tokens)**.
One token is roughly 4 characters of English text, so 512 tokens is
about 2,048 characters. That size is big enough to hold one complete
idea (a paragraph about eligibility, or a funding bullet list) and
small enough that the embedding still means something specific.
Overlap exists so a fact split across a chunk boundary is not lost.
Chunking is recursive and sentence-aware: split on paragraphs, then
sentences, then words only if a single sentence is still too long.
Implemented from scratch in `app/chunking.py` (no LangChain) so the
behaviour is fully visible and testable.

### Retrieval choice (why hybrid)

Dense embeddings alone miss exact names and numbers ("Vanier",
"$50,000", "SI Scholarship"). BM25 alone misses paraphrases
("fully funded master's in the UK" vs "one-year master's degrees").
So retrieval is hybrid: FAISS cosine similarity over
`all-MiniLM-L6-v2` embeddings, plus BM25 over chunk tokens, each
normalised to 0–1 and combined 0.5/0.5. The top 10 are then reranked
with a transparent lexical reranker (query-term coverage bonus).
That reranker is **not** a cross-encoder; Phase 2 adds one and
measures whether it actually helps.

### Answering choice (why extractive, why abstain)

Phase 1 has no generative LLM, so the "answer" is sentences taken
verbatim from the top cited chunks. That makes faithfulness
measurable instead of claimed. If the best hybrid score is below
the abstention threshold, or no retrieved chunk shares a content
word with the question, the API returns:

```json
{"found": false, "answer": "Not found in the corpus.", "citations": []}
```

Abstaining is a feature, not a failure. A scholarship assistant
that invents a deadline is worse than one that says it does not know.

## Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Fetch the corpus (9 pages by HTTP, Chevening text is already saved)
python -m scripts.ingest

# 2. Chunk + embed + build the FAISS index
python -m scripts.build_index

# 3. Run the eval harness (prints real numbers from this machine)
python -m scripts.run_eval

# 4. Run the API
uvicorn app.api:app --reload --port 8000

# 5. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the Chevening Scholarship fund?"}'
```

## API

`GET /health` — `{"status": "ok", "docs": 10, "chunks": N, "embedder": ...}`

`POST /ask` — body `{"question": "...", "top_k": 5}`

```json
{
  "question": "What does the Chevening Scholarship fund?",
  "found": true,
  "answer": "Chevening Scholarships enable outstanding emerging leaders ... one-year master's degrees in the UK. Chevening Scholarships are fully-funded ...",
  "score": 0.62,
  "embedder": "all-MiniLM-L6-v2",
  "citations": [
    {"doc": "chevening", "source_url": "https://www.chevening.org/scholarships/",
     "chunk_id": "chevening-0", "score": 0.62, "text": "..."}
  ]
}
```

If the sentence-transformers model cannot be downloaded (offline
machine), the embedder falls back to BM25-only retrieval and the
response reports `"embedder": "bm25-only"`. The API never silently
pretends dense retrieval ran when it did not.

## Eval harness

`eval/golden.json` holds hand-written questions. Every question was
checked against the actual fetched text before it was added — the
expected answer phrase must appear in the cited document.

`python -m scripts.run_eval` prints:

- **recall@3 / recall@5** — fraction of questions where the expected
  document is in the top 3 / top 5 retrieved documents.
- **faithfulness** — fraction of answer sentences that appear verbatim
  in a cited chunk (extractive answers, so this should be high; if it
  is not, something is broken).
- **answer relevance (proxy)** — fraction of questions where the
  expected answer phrase appears in the answer or cited chunks.
  This is a string-match proxy, **not** an LLM judge. Phase 2 adds
  an LLM-as-judge and compares.
- **abstention accuracy** — fraction of out-of-corpus questions
  correctly answered "Not found in the corpus."

All numbers in this README's "Results" section come from a real
local run — see `EVAL_RESULTS.md`. No number here is invented.

## Tests

```bash
python -m pytest tests/ -v
```

Tests cover chunking (size, overlap, sentence boundaries),
retrieval (known question finds the right document, hybrid beats
either signal alone on the golden set is checked in eval, not in
unit tests), and abstention (out-of-corpus question returns
`found: false` with no citations).

## Docker

```bash
docker build -t production-rag-eval-harness .
docker run -p 8000:8000 production-rag-eval-harness
```

The image builds the index at build time from the committed
`data/raw/` and `data/processed/chunks.jsonl`, so the container
starts without re-fetching or re-downloading the embedding model
at runtime beyond the cached model in the image.

## Results (Phase 1, real local run)

See `EVAL_RESULTS.md` for the exact numbers from the run on
18 Nov 2026, including the test count. Summary is written there
after the run, not guessed here.

## What Phase 1 does NOT do (Phase 2)

- No generative LLM answer (extractive only).
- No cross-encoder reranker (lexical rerank only).
- No LLM-as-judge (string-match proxies only).
- No query rewriting, no multi-hop retrieval.
- No deployment (local + Docker only; not deployed anywhere).
- No auto-refresh of the corpus (re-run `scripts.ingest` manually).

## What to learn from this (interview)

- **Chunking:** why 512 tokens, why overlap, what breaks when chunks
  are too small (lost context) or too big (diluted embedding).
- **Hybrid retrieval:** when dense wins, when BM25 wins, why you
  combine them, and how you prove it with recall@k instead of vibes.
- **Evals:** a golden set with a baseline beats a demo. Be able to
  say "recall@5 is X, and it went up/down when I changed Y".
- **Abstention:** the most important production behaviour in a
  scholarship assistant is refusing to answer when the evidence
  is not there.
- **Observability:** every answer carries its citations and its
  score, so any wrong answer can be traced back to the chunk.
