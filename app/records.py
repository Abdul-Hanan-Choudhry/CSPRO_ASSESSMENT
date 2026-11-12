"""Load the synthetic records shipped in data/."""

from pathlib import Path

from app.config import ROOT
from app.schemas import SourceRecord

DATA_FILES = ("kb.jsonl", "tickets.jsonl", "logs.jsonl")


def load_records(data_dir: Path = ROOT / "data") -> list[SourceRecord]:
    """Parse every data file; any invalid line fails loudly rather than being skipped."""
    records: list[SourceRecord] = []
    for name in DATA_FILES:
        for line_no, line in enumerate((data_dir / name).read_text("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(SourceRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{name}:{line_no}: {exc}") from exc
    return records
