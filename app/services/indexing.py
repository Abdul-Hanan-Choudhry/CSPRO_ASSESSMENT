"""Chunking, point construction and index integrity checks for the current (v2) schema."""

import uuid
from collections.abc import Iterable

from qdrant_client import models

from app.clients.embeddings import Embedder
from app.clients.qdrant_store import DENSE, SPARSE, QdrantStore
from app.schemas import SourceRecord

SCHEMA_VERSION = 2
MAX_CHUNK_CHARS = 800
# Fixed namespace: the same (source_id, chunk) always maps to the same point ID, so re-indexing
# overwrites instead of duplicating, and IDs never depend on file order.
POINT_NAMESPACE = uuid.UUID("6f1c2d8e-3b7a-4c55-9a0e-2f4b8c1d7e93")


def chunk_text(text: str) -> list[str]:
    """Split on blank lines; split long paragraphs at whitespace under MAX_CHUNK_CHARS."""
    chunks: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        while len(paragraph) > MAX_CHUNK_CHARS:
            cut = paragraph.rfind(" ", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 0 else MAX_CHUNK_CHARS
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if paragraph:
            chunks.append(paragraph)
    return chunks


def point_id(source_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source_id}#{chunk_index}"))


def build_points(records: Iterable[SourceRecord], embedder: Embedder) -> list[models.PointStruct]:
    """One point per chunk. source_id, ownership and text all come from the same record object."""
    pending: list[tuple[str, dict]] = []
    for record in records:
        meta = record.model_dump(mode="json", exclude={"text"})
        for index, chunk in enumerate(chunk_text(record.text)):
            payload = {
                **meta,
                "chunk_index": index,
                "text": chunk,
                "schema_version": SCHEMA_VERSION,
            }
            pending.append((point_id(record.source_id, index), payload))
    # Title is embedded for recall but not stored in "text": citations verify against chunk text.
    vectors = embedder.embed_documents([f"{p['title']}\n{p['text']}" for _, p in pending])
    return [
        models.PointStruct(id=pid, vector={DENSE: v.dense, SPARSE: v.sparse}, payload=payload)
        for (pid, payload), v in zip(pending, vectors, strict=True)
    ]


def index_records(
    store: QdrantStore, embedder: Embedder, collection: str, records: list[SourceRecord]
) -> int:
    points = build_points(records, embedder)
    store.upsert(collection, points)
    return len(points)


def integrity_problems(
    store: QdrantStore, collection: str, records: list[SourceRecord]
) -> list[str]:
    """Check every point against the canonical record carrying its source_id.

    A point whose text is not part of its labelled record, or whose owner/visibility differ from
    that record, means citations and permission filters are both wrong for that point.
    """
    canonical = {r.source_id: r for r in records}
    problems: list[str] = []
    for payload in store.scroll_payloads(collection):
        sid = payload.get("source_id")
        record = canonical.get(sid)
        if record is None:
            continue  # ingested through the API after seeding; no canonical copy on disk
        if payload.get("text", "") not in record.text:
            problems.append(f"{sid}: text does not belong to this source")
        if (payload.get("customer_id"), payload.get("visibility")) != (
            record.customer_id,
            record.visibility.value,
        ):
            problems.append(f"{sid}: owner/visibility differs from source record")
    return problems
