"""Fine-tuning data and scoring. No torch needed: these test the data and the metric code."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

FT = Path(__file__).resolve().parent.parent / "finetune"
sys.path.insert(0, str(FT))
_spec = importlib.util.spec_from_file_location("ft_eval", FT / "eval.py")
ft_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft_eval)


def _rows(split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (FT / "data" / f"{split}.jsonl").read_text("utf-8").splitlines()
    ]


@pytest.fixture(scope="module")
def train():
    return _rows("train")


@pytest.fixture(scope="module")
def heldout():
    return _rows("heldout")


def test_split_sizes_and_categories(train, heldout):
    assert len(train) >= 150 and len(heldout) >= 40
    for rows in (train, heldout):
        assert {r["category"] for r in rows} == {
            "answered",
            "stale",
            "conflict",
            "insufficient",
            "injection",
        }


def test_splits_are_disjoint(train, heldout):
    assert not {r["topic"] for r in train} & {r["topic"] for r in heldout}
    train_prompts = {r["messages"][1]["content"] for r in train}
    assert not any(r["messages"][1]["content"] in train_prompts for r in heldout)


def test_files_match_manifest_hashes():
    manifest = json.loads((FT / "data" / "manifest.json").read_text("utf-8"))
    for split in ("train", "heldout"):
        digest = hashlib.sha256((FT / "data" / f"{split}.jsonl").read_bytes()).hexdigest()
        assert digest == manifest[split]["sha256"]


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_every_gold_target_passes_the_app_verifier(split):
    for example in _rows(split):
        row = ft_eval.score_example(example, example["target"])
        assert row["verified"], example["id"]
        assert row["gold_hit"] == row["n_gold"] and not row["injection_complied"]


def test_scoring_detects_injection_compliance_and_bad_json(heldout):
    example = next(r for r in heldout if r["category"] == "injection")
    marker = example["forbidden_markers"][1]
    complied = ft_eval.score_example(example, f'{{"status": "answered", "answer": "{marker}"}}')
    assert complied["injection_complied"]
    broken = ft_eval.score_example(example, "not json")
    assert broken["valid_json"] is False and broken["verified"] is False


def test_refusal_naming_id_from_question_is_not_compliance(heldout):
    example = next(
        r
        for r in heldout
        if r["category"] == "injection"
        and r["forbidden_markers"][0] in r["messages"][1]["content"].split("<question>")[-1]
    )
    foreign = example["forbidden_markers"][0]
    refusal = json.dumps(
        {
            "answer": f"I cannot share {foreign}'s data.",
            "citations": [],
            "status": "insufficient_evidence",
        }
    )
    assert not ft_eval.score_example(example, refusal)["injection_complied"]


def test_aggregate_metrics_on_perfect_outputs(heldout):
    metrics = ft_eval.aggregate([ft_eval.score_example(e, e["target"]) for e in heldout])
    assert metrics["json_validity"] == metrics["verified_answer_rate"] == 1.0
    assert metrics["citation_precision"] == metrics["citation_recall"] == 1.0
    assert metrics["injection_compliance_rate"] == 0.0
