"""Tests for Phase 2 Step 3: the cross-encoder reranker.

The cross-encoder tests use the real local model
(cross-encoder/ms-marco-MiniLM-L-6-v2, ~91 MB, already downloaded
by the benchmark run), so they take ~30 s with the model load.
The fallback test monkeypatches the loader and needs no model.
The end-to-end config test runs the real benchmark path on a
3-question subset, not the full 28-question golden set.
"""
import json

import pytest

from app import cross_encoder
from app.config import CHUNKS_FILE, CROSS_ENCODER_MODEL
from app.eval import load_golden
from app.retrieval import Retriever
from benchmarks import run_benchmark
from benchmarks.run_benchmark import CONFIGS, build_summary, run_config

CHUNKS = [json.loads(line) for line in CHUNKS_FILE.read_text().splitlines() if line.strip()]

QUESTION = "What does the Chevening Scholarship fund?"
# Lexical trap: repeats every query term ("Chevening Scholarship
# fund") but says nothing about what the scholarship funds. The
# answering chunk shares fewer query terms but is the actual answer.
TRAP = ("The Chevening Scholarship fund opens for applications in August. Chevening Scholarship "
        "applicants for the fund must submit their Chevening Scholarship application before the "
        "deadline. The Chevening Scholarship fund team reviews applications.")
ANSWER = ("The Chevening Scholarship covers full tuition fees, a monthly living allowance, "
          "and return airfare to the United Kingdom.")
FILLERS = [
    "The MEXT scholarship is offered by the Japanese government to international students.",
    "DAAD scholarships support study and research in Germany for graduates.",
    "The Vanier Canada Graduate Scholarships attract doctoral students to Canada.",
    "Erasmus Mundus Joint Masters are international study programmes in Europe.",
]
TRAP_CHUNKS = (
    [{"chunk_id": "trap", "doc": "chevening", "text": TRAP},
     {"chunk_id": "answer", "doc": "chevening", "text": ANSWER}]
    + [{"chunk_id": f"f{i}", "doc": "other", "text": text}
       for i, text in enumerate(FILLERS)]
)


@pytest.fixture(scope="module")
def cross_encoder_model():
    assert cross_encoder.load_model() is not None, (
        f"cross-encoder model unavailable: {cross_encoder.load_error()}")


# --- loading and scoring -------------------------------------------------

def test_cross_encoder_loads_and_scores_pairs(cross_encoder_model):
    assert cross_encoder.reranker_name() == CROSS_ENCODER_MODEL
    scores = cross_encoder.score_pairs(QUESTION, [
        "Chevening Scholarships fund tuition fees and living costs for international students.",
        "The weather in London is rainy and cold in winter.",
    ])
    assert scores is not None
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    # Measured on ms-marco-MiniLM-L-6-v2: the relevant pair scores
    # about +5.4, the unrelated one about -11.4. The gap is large,
    # so this is a stable property of the model, not a knife-edge.
    assert scores[0] > 0.0
    assert scores[1] < 0.0
    assert scores[0] - scores[1] > 5.0


def test_cross_encoder_sigmoid_maps_logits_to_0_1():
    assert cross_encoder.sigmoid(0.0) == 0.5
    assert cross_encoder.sigmoid(10.0) > 0.99
    assert cross_encoder.sigmoid(-10.0) < 0.01
    assert 0.0 < cross_encoder.sigmoid(5.97) < 1.0


def test_invalid_reranker_raises():
    with pytest.raises(ValueError, match="unknown reranker"):
        Retriever(CHUNKS, reranker="tfidf")


# --- reranking behaviour -------------------------------------------------

def test_cross_encoder_reranking_flips_lexical_trap(cross_encoder_model):
    # Lexical rerank (combined score + coverage bonus) puts the
    # term-repeating trap first: coverage 1.0 vs 0.667, and the
    # repetition wins BM25. The cross-encoder, scoring the pair
    # jointly, must put the answering chunk first instead.
    lexical = Retriever(TRAP_CHUNKS, mode="hybrid", reranker="lexical")
    lexical_order = [c["chunk_id"] for c in lexical.search(QUESTION, top_k=2)]
    assert lexical_order[0] == "trap"

    ce = Retriever(TRAP_CHUNKS, mode="hybrid", reranker="cross-encoder")
    ranked = ce.search(QUESTION, top_k=2)
    assert [c["chunk_id"] for c in ranked][0] == "answer"
    assert ce.last_reranker_used == "cross-encoder"
    by_id = {c["chunk_id"]: c for c in ranked}
    # Raw logits, measured: answer ~7.13, trap ~5.97.
    assert by_id["answer"]["cross_encoder_score"] > by_id["trap"]["cross_encoder_score"]
    assert by_id["answer"]["cross_encoder_score"] > 5.0
    assert all(c["reranker"] == "cross-encoder" for c in ranked)
    # Rerank scores stay 0-1 (sigmoid of the logit) so the
    # abstention threshold keeps its meaning.
    assert all(0.0 <= c["rerank_score"] <= 1.0 for c in ranked)


def test_cross_encoder_falls_back_to_lexical_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(cross_encoder, "load_model", lambda: None)
    retriever = Retriever(TRAP_CHUNKS, mode="hybrid", reranker="cross-encoder")
    ranked = retriever.search(QUESTION, top_k=2)
    assert retriever.last_reranker_used == "lexical-fallback"
    assert all(c["reranker"] == "lexical-fallback" for c in ranked)
    # Fallback is the lexical order: the trap first.
    assert ranked[0]["chunk_id"] == "trap"
    assert "cross_encoder_score" not in ranked[0]


# --- ablation configuration ----------------------------------------------

def test_hybrid_cross_encoder_config_in_ablation():
    names = [c["name"] for c in CONFIGS]
    assert "hybrid-cross-encoder" in names
    config = next(c for c in CONFIGS if c["name"] == "hybrid-cross-encoder")
    assert config["mode"] == "hybrid"
    assert config["reranker"] == "cross-encoder"
    assert config["search_kwargs"] == {"rerank": True}
    assert config["answer_kwargs"] == {"gates": True, "scored_picking": True}
    # Nearest neighbour: identical to improved-hybrid except the reranker.
    improved = next(c for c in CONFIGS if c["name"] == "improved-hybrid")
    assert {k: v for k, v in config.items() if k != "reranker" and k != "name"
            and k != "description"} == \
        {k: v for k, v in improved.items() if k != "reranker" and k != "name"
         and k != "description"}


def test_hybrid_cross_encoder_config_runs_end_to_end(cross_encoder_model):
    golden = [q for q in load_golden() if not q.get("profile")
              and not q.get("out_of_corpus")][:3]
    config = next(c for c in CONFIGS if c["name"] == "hybrid-cross-encoder")
    result = run_config(CHUNKS, golden, config)
    assert result["config"] == "hybrid-cross-encoder"
    assert result["reranker"] == "cross-encoder"
    assert result["retrieval_mode"] == "hybrid"
    assert result["n_chunks"] == len(CHUNKS)
    assert result["cost_per_query_usd"] == 0.0
    for key in ("recall@3", "recall@5", "answer_relevance_proxy",
                "answer_only_relevance", "faithfulness", "abstention_accuracy"):
        assert key in result["quality"]
    assert len(result["per_question"]) == 3
    assert result["latency"]["n_queries"] == 3
    assert result["latency"]["retrieval"]["mean_ms"] > 0


def test_benchmark_summary_generated_from_results_files(tmp_path, monkeypatch,
                                                        cross_encoder_model):
    golden = [q for q in load_golden() if not q.get("profile")
              and not q.get("out_of_corpus")][:3]
    subset = [c for c in CONFIGS if c["name"] in
              ("improved-hybrid", "hybrid-cross-encoder")]
    monkeypatch.setattr(run_benchmark, "CONFIGS", subset)
    for config in subset:
        result = run_config(CHUNKS, golden, config)
        (tmp_path / f"{config['name']}.json").write_text(json.dumps(result, indent=2))
    first = build_summary(tmp_path)
    assert first == tmp_path / "summary.md"
    text = first.read_text()
    assert "improved-hybrid" in text
    assert "hybrid-cross-encoder" in text
    # Every table number comes from the files, not from memory:
    # the recall@5 of each config in the summary must match its JSON.
    for config in subset:
        saved = json.loads((tmp_path / f"{config['name']}.json").read_text())
        recall5 = str(saved["quality"]["recall@5"])
        assert f"| {config['name']} | hybrid |" in text
        assert recall5 in text
    # Re-running on the same files reproduces the same summary.
    assert build_summary(tmp_path).read_text() == text
