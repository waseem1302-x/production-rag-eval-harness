"""Tests for distillation finalize-only replay and proxy judge labels.

No teacher call, no model load, no network. The finalize-only path
replays saved teacher outputs and builds judge labels from the
deterministic proxy; these tests prove the labels are the proxy's
own values, that replay never appends duplicate gate rows, and
that the finalized files on disk satisfy the contamination rule
and the judge balance.
"""
import json
from pathlib import Path

import pytest

import scripts.build_distillation_data as dist
from app.eval import load_golden
from app.textnorm import norm_text
from scripts.build_distillation_data import (
    Builder,
    ReplayTeacher,
    faithfulness_proxy,
    parse_judge_lines,
    recorded_teacher_model,
    set_out_dir,
)

REPO = Path(__file__).resolve().parent.parent
DISTILL_DIR = REPO / "data" / "distillation"

CHUNKS = [
    {"chunk_id": "chevening-0", "doc": "chevening",
     "text": "Chevening Scholarships enable outstanding emerging "
             "leaders to pursue one-year master's degrees in the UK. "
             "Chevening Scholarships are fully-funded."},
    {"chunk_id": "sweden-si-0", "doc": "sweden-si",
     "text": "The SI Scholarship covers a monthly allowance of "
             "SEK 12,000 and a travel grant of SEK 15,000."},
]
GOLDEN = [{"id": "q01",
           "question": "What does the Chevening Scholarship fund?",
           "expected_answer": "one-year master"},
          {"id": "q10",
           "question": "How long does an Erasmus Mundus Joint Master last?",
           "expected_answer": "1 to 2 academic years"}]
GEN_EVAL = {"generative": {"per_question": []},
            "extractive": {"per_question": []}}


class StubTeacher:
    """Teacher that must never be called (finalize-only replay)."""

    def __init__(self):
        self.model = "gemini-3.5-flash-lite"

    def call(self, *args, **kwargs):
        raise AssertionError("teacher must not be called in finalize-only")


@pytest.fixture()
def tmp_out(tmp_path):
    """Point the distillation outputs at a temp dir, restore after."""
    set_out_dir(tmp_path / "distill")
    yield tmp_path / "distill"
    set_out_dir(DISTILL_DIR)


def _builder(tmp_out):
    return Builder(StubTeacher(), None, CHUNKS, GOLDEN, GEN_EVAL,
                   replay=True)


def test_replay_teacher_call_raises():
    with pytest.raises(AssertionError):
        ReplayTeacher("gemini-3.5-flash-lite").call("x", "answer", "y")


def test_recorded_teacher_model(tmp_out):
    (tmp_out / "teacher_raw.jsonl").write_text(
        "".join(json.dumps({"item_id": f"i{k}", "kind": "answer",
                            "teacher": "gemini/gemini-3.5-flash-lite",
                            "raw": "x"}) + "\n" for k in range(3)))
    assert recorded_teacher_model() == "gemini-3.5-flash-lite"


def test_recorded_teacher_model_rejects_mixed(tmp_out):
    (tmp_out / "teacher_raw.jsonl").write_text(
        json.dumps({"item_id": "i1", "kind": "answer",
                    "teacher": "gemini/gemini-3.5-flash-lite",
                    "raw": "x"}) + "\n"
        + json.dumps({"item_id": "i2", "kind": "answer",
                      "teacher": "gemini/gemini-3.8-flash",
                      "raw": "y"}) + "\n")
    with pytest.raises(SystemExit):
        recorded_teacher_model()


def test_proxy_label_wrong_answer(tmp_out):
    builder = _builder(tmp_out)
    label = builder._judge_proxy_item(
        "judge-w-q01", "What does the Chevening Scholarship fund?",
        "The Chevening Scholarship funds one academic year.",
        CHUNKS, "one-year master")
    # Paraphrase, not verbatim, and the expected phrase is absent.
    assert label == (0, 0)
    record = builder.records[-1]
    assert record["source"] == "judge"
    assert record["question_id"] == "judge-w-q01"
    assert record["expected_judge"] == {"faithfulness": 0, "relevance": 0}
    assert record["label_source"] == "deterministic_proxy"
    assert parse_judge_lines(record["messages"][1]["content"]) == (0, 0)
    assert record["teacher"] == "gemini/gemini-3.5-flash-lite"
    gate = [json.loads(line) for line in
            (tmp_out / "gate_log.jsonl").read_text().splitlines()]
    assert gate == [{"item_id": "judge-w-q01", "source": "judge",
                     "kept": True, "reasons": ["proxy_label"]}]


def test_proxy_label_right_answer(tmp_out):
    builder = _builder(tmp_out)
    label = builder._judge_proxy_item(
        "judge-r-q01", "What does the Chevening Scholarship fund?",
        "Chevening Scholarships enable outstanding emerging leaders "
        "to pursue one-year master's degrees in the UK.",
        CHUNKS, "one-year master")
    assert label == (1, 1)
    record = builder.records[-1]
    assert record["expected_judge"] == {"faithfulness": 1, "relevance": 1}
    assert parse_judge_lines(record["messages"][1]["content"]) == (1, 1)


def test_proxy_label_partial_credit(tmp_out):
    # The answer carries the expected phrase but rephrases it, so
    # relevance is 1 and verbatim faithfulness is 0. The proxy can
    # give mixed labels; the teacher judge could not be trusted
    # with this distinction.
    builder = _builder(tmp_out)
    label = builder._judge_proxy_item(
        "judge-w-q99", "What does the Chevening Scholarship fund?",
        "Chevening funds a one-year master programme in the UK.",
        CHUNKS, "one-year master")
    assert label == (0, 1)
    assert builder.records[-1]["expected_judge"] == {
        "faithfulness": 0, "relevance": 1}


def test_proxy_label_uses_stored_sentences(tmp_out):
    # Real q10 case: the extractive answer ends with a picked
    # sentence followed by a lowercase fragment line from another
    # chunk. Re-splitting the joined text merges them into one
    # unit that is not verbatim, so the re-split proxy says 0.
    # The stored sentences are the eval's own method, and they
    # are each verbatim.
    builder = _builder(tmp_out)
    chunks = [{
        "chunk_id": "erasmus-mundus-0", "doc": "erasmus-mundus",
        "text": "Masters duration 1 to 2 academic years (60, 90 or "
                "120 ECTS credits).\n"
                "A new batch of selected Erasmus Mundus Masters is "
                "added to the list each year.\n"
                "In most cases, you should submit your application."},
        {"chunk_id": "erasmus-mundus-1", "doc": "erasmus-mundus",
         "text": "Something else entirely.\n"
                 "of selected Erasmus Mundus Masters is added to the "
                 "list each year.\n"
                 "More text here."}]
    answer = ("Masters duration 1 to 2 academic years (60, 90 or 120 "
              "ECTS credits). A new batch of selected Erasmus Mundus "
              "Masters is added to the list each year. of selected "
              "Erasmus Mundus Masters is added to the list each year.")
    stored = ["Masters duration 1 to 2 academic years (60, 90 or 120 "
              "ECTS credits).",
              "A new batch of selected Erasmus Mundus Masters is added "
              "to the list each year.",
              "of selected Erasmus Mundus Masters is added to the list "
              "each year."]
    assert faithfulness_proxy(answer, chunks) == 0
    label = builder._judge_proxy_item(
        "judge-r-q10", "How long does an Erasmus Mundus Joint Master "
        "last?", answer, chunks, "1 to 2 academic years",
        answer_sentences=stored)
    assert label == (1, 1)


def test_proxy_label_matches_direct_proxy_computation(tmp_out):
    # The label is exactly (faithfulness_proxy, contains_phrase),
    # not a model judgement. Recompute both sides independently.
    from scripts.build_distillation_data import contains_phrase
    builder = _builder(tmp_out)
    answer = ("The SI Scholarship covers a monthly allowance of "
              "SEK 12,000 and a travel grant of SEK 15,000.")
    label = builder._judge_proxy_item(
        "judge-r-q99", "What is the SI allowance?", answer,
        CHUNKS, "SEK 12,000")
    assert label == (faithfulness_proxy(answer, CHUNKS),
                     1 if contains_phrase(answer, "SEK 12,000") else 0)
    assert label == (1, 1)


def test_replay_log_gate_skips_appends(tmp_out):
    gate_file = tmp_out / "gate_log.jsonl"
    gate_file.write_text(
        json.dumps({"item_id": "q01-v1", "source": "golden-variant",
                    "kept": True, "reasons": []}) + "\n")
    builder = _builder(tmp_out)
    builder.log_gate("q01-v1", "golden-variant", True, [])
    assert gate_file.read_text().count("\n") == 1
    builder.log_gate("judge-w-q01", "judge", True, ["proxy_label"],
                     force_append=True)
    assert gate_file.read_text().count("\n") == 2


def test_finalize_merges_records_by_question_id(tmp_out):
    (tmp_out / "gate_log.jsonl").write_text("")
    builder = _builder(tmp_out)
    builder.make_record("q01-v1", "golden-variant", "prompt one",
                        "Answer one.", CHUNKS, "phrase")
    builder.make_record("q01-v1", "golden-variant", "prompt two",
                        "Answer two.", CHUNKS, "phrase")
    builder.make_record("syn-x-000", "synthetic-qa", "prompt three",
                        "Answer three.", CHUNKS, "phrase", heldout=True)
    stats = builder.finalize()
    train = [json.loads(line) for line in
             (tmp_out / "train.jsonl").read_text().splitlines()]
    heldout = [json.loads(line) for line in
               (tmp_out / "heldout.jsonl").read_text().splitlines()]
    assert len(train) == 1
    assert train[0]["messages"][0]["content"] == "prompt two"
    assert len(heldout) == 1
    assert stats["train_records"] == 1
    assert stats["heldout_records"] == 1
    assert (tmp_out / "stats.json").exists()


# ---- tests on the finalized files on disk ----

def _load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text()
            .splitlines() if line.strip()]


def test_finalized_train_has_no_verbatim_golden():
    train = _load_jsonl(DISTILL_DIR / "train.jsonl")
    assert train, "train.jsonl is empty"
    golden_questions = {norm_text(g["question"])
                        for g in load_golden()}
    for record in train:
        prompt = record["messages"][0]["content"]
        question = prompt.split("Question: ", 1)[1].rsplit(
            "\nAnswer:", 1)[0].strip()
        assert norm_text(question) not in golden_questions, \
            f"train record {record['question_id']} is a verbatim golden"


def test_finalized_train_heldout_disjoint():
    train = _load_jsonl(DISTILL_DIR / "train.jsonl")
    heldout = _load_jsonl(DISTILL_DIR / "heldout.jsonl")
    assert heldout, "heldout.jsonl is empty"
    train_ids = {r["question_id"] for r in train}
    held_ids = {r["question_id"] for r in heldout}
    assert not train_ids & held_ids
    assert all(r["source"] in
               ("golden-variant", "golden-variant-heldout",
                "synthetic-qa", "abstention", "judge") for r in train)
    assert all(r["source"] in
               ("golden-variant-heldout", "synthetic-qa")
               for r in heldout)


def test_finalized_judge_balance_is_proxy_labels():
    train = _load_jsonl(DISTILL_DIR / "train.jsonl")
    judge = [r for r in train if r["source"] == "judge"]
    wrong = [r for r in judge
             if r["question_id"].startswith("judge-w-")]
    right = [r for r in judge
             if r["question_id"].startswith("judge-r-")]
    assert len(judge) == 40
    assert len(wrong) == 14
    assert len(right) == 26
    for record in judge:
        assert record.get("label_source") == "deterministic_proxy"
        assert record.get("teacher") == "gemini/gemini-3.5-flash-lite"
        assert parse_judge_lines(record["messages"][1]["content"]) == (
            record["expected_judge"]["faithfulness"],
            record["expected_judge"]["relevance"])
    # Wrong answers miss the expected phrase by construction:
    # every wrong label has relevance 0.
    assert all(r["expected_judge"]["relevance"] == 0 for r in wrong)
    # Right answers are verbatim extractive answers: both 1.
    assert all(r["expected_judge"] == {"faithfulness": 1, "relevance": 1}
               for r in right)


def test_finalized_counts_match_stats():
    stats = json.loads((DISTILL_DIR / "stats.json").read_text())
    train = _load_jsonl(DISTILL_DIR / "train.jsonl")
    heldout = _load_jsonl(DISTILL_DIR / "heldout.jsonl")
    assert stats["train_records"] == len(train)
    assert stats["heldout_records"] == len(heldout)
    assert stats["teacher"] == "gemini/gemini-3.5-flash-lite"
    assert stats["teacher_calls"] == 435
    assert stats["teacher_prompt_tokens"] == 109687
    assert stats["teacher_completion_tokens"] == 23628


def test_finalized_every_train_record_has_kept_gate_entry():
    train = _load_jsonl(DISTILL_DIR / "train.jsonl")
    last = {}
    for line in (DISTILL_DIR / "gate_log.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            last[row["item_id"]] = row["kept"]
    missing = [r["question_id"] for r in train
               if not last.get(r["question_id"])]
    assert not missing, f"train records with no kept gate entry: {missing}"
