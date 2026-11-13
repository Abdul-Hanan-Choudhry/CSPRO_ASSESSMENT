"""Shared helpers for train.py and eval.py: config, data loading, run metadata."""

import json
import platform
import random
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "finetune" / "results"
sys.path.insert(0, str(ROOT))


def load_config(path: Path, mode: str) -> dict:
    raw = yaml.safe_load(path.read_text("utf-8"))
    return {
        "seed": raw["seed"],
        "data": raw["data"],
        "lora": raw["lora"],
        "mode": mode,
        **raw[mode],
    }


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / path).read_text("utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_metadata() -> dict:
    packages = ["torch", "transformers", "peft", "accelerate", "httpx"]
    versions = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def write_result(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
