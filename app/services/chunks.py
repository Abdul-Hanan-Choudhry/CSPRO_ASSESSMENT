"""The unit of retrieved evidence passed from retrieval to the prompt.

Kept dependency-free so finetune/ can build prompts identical to the app's without importing
Qdrant or fastembed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_index: int
    title: str
    text: str
    kind: str
    status: str
    updated_at: str
    superseded_by: str | None
    score: float
