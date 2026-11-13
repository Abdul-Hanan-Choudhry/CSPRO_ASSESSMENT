"""Render the results table from finetune/results/*.json. Missing runs print NOT RUN.

    python finetune/report.py     # paste the output into README "Fine-tuning and results"

Numbers come only from result files written by train.py / eval.py; nothing is typed by hand.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
COLUMNS = [
    ("eval_baseline", "Qwen2.5-0.5B base"),
    ("eval_adapted", "Qwen2.5-0.5B + LoRA"),
    ("eval_ollama_llama3.2-3b", "Llama 3.2 3B (Ollama, app runtime)"),
]
METRICS = [
    "json_validity",
    "citation_precision",
    "citation_recall",
    "status_accuracy",
    "abstain_accuracy",
    "conflict_accuracy",
    "injection_compliance_rate",
    "verified_answer_rate",
]


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    runs = {name: load(name) for name, _ in COLUMNS}
    print("| Metric (held-out) | " + " | ".join(label for _, label in COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for metric in METRICS:
        cells = [fmt(r["metrics"][metric]) if r else "NOT RUN" for r in runs.values()]
        print(f"| {metric} | " + " | ".join(cells) + " |")
    ns = [f"n={r['metrics']['n']}, {r['decoding']}" if r else "NOT RUN" for r in runs.values()]
    print("| examples / decoding | " + " | ".join(ns) + " |")

    train = load("train_full")
    if train:
        before, after = train["heldout_loss_before"], train["heldout_loss_after"]
        print(
            f"\nTraining (full): {train['base_model']}, {train['n_train']} examples, "
            f"held-out loss {before:.4f} -> {after:.4f}, "
            f"{train['train_runtime_s']} s on {train['device']}."
        )
    else:
        print("\nTraining (full): NOT RUN.")
    smoke = load("train_smoke")
    if smoke:
        print(
            f"Smoke run (pipeline check only, not a result): {smoke['base_model']}, "
            f"{smoke['n_train']} examples, {smoke['config']['max_steps']} steps "
            f"on {smoke['device']}."
        )


if __name__ == "__main__":
    main()
