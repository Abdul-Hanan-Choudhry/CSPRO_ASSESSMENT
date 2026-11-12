"""Build versioned collections and move the live alias.

    python scripts/build_index.py seed            # kb_v1 (legacy) + kb_v2, alias -> kb_v2
    python scripts/build_index.py point kb_v1     # repoint alias (reproduce the incident)
    python scripts/build_index.py point kb_v2     # integrity-checked repoint (the fix)

kb_v1 is built by `build_v1_legacy`, a faithful copy of the pre-incident indexer, kept only to
reproduce the incident. kb_v2 uses app.services.indexing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models

from app.clients.embeddings import Embedder
from app.clients.qdrant_store import DENSE, SPARSE, QdrantStore
from app.config import Settings, get_settings
from app.records import load_records
from app.schemas import SourceRecord
from app.services.indexing import index_records, integrity_problems

V1, V2 = "kb_v1", "kb_v2"


def build_v1_legacy(store: QdrantStore, embedder: Embedder, records: list[SourceRecord]) -> int:
    """Pre-incident indexer. DO NOT USE for serving: it contains the ID-mismatch bug.

    Bug 1: vectors and text come from `records` in file order, but the metadata (source_id,
    owner, visibility) comes from a list sorted by updated_at, so point i is labelled with the
    wrong record. Bug 2: point IDs are row positions, so they change whenever the export order
    changes. Whole documents are embedded as a single chunk.
    """
    by_date = sorted(records, key=lambda r: r.updated_at)
    vectors = embedder.embed_documents([f"{r.title}\n{r.text}" for r in records])
    points = [
        models.PointStruct(
            id=row,
            vector={DENSE: v.dense, SPARSE: v.sparse},
            payload={
                **by_date[row].model_dump(mode="json", exclude={"text"}),
                "text": records[row].text,
                "chunk_index": 0,
                "schema_version": 1,
            },
        )
        for row, v in enumerate(vectors)
    ]
    store.upsert(V1, points)
    return len(points)


def point_alias_checked(
    store: QdrantStore, settings: Settings, target: str, records: list[SourceRecord]
) -> None:
    """Repoint the live alias, refusing targets that fail the integrity check."""
    problems = integrity_problems(store, target, records)
    if problems:
        print(f"REFUSED: {target} failed integrity check ({len(problems)} problems), e.g.:")
        for problem in problems[:5]:
            print(f"  - {problem}")
        raise SystemExit(1)
    store.point_alias(settings.collection_alias, target)
    print(f"alias {settings.collection_alias} -> {target}")


def seed(store: QdrantStore, embedder: Embedder, settings: Settings) -> None:
    """Idempotent: recreates both collections from data/ and leaves the alias on kb_v2."""
    records = load_records()
    for name in (V1, V2):
        store.recreate_collection(name, embedder.dense_dim)
    print(f"{V1}: {build_v1_legacy(store, embedder, records)} points (legacy schema)")
    print(f"{V2}: {index_records(store, embedder, V2, records)} points")
    point_alias_checked(store, settings, V2, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    point = sub.add_parser("point")
    point.add_argument("collection", choices=[V1, V2])
    point.add_argument(
        "--unsafe", action="store_true", help="skip the integrity check (incident repro only)"
    )
    args = parser.parse_args()

    settings = get_settings()
    store = QdrantStore(settings)
    try:
        if args.cmd == "seed":
            seed(store, Embedder(settings), settings)
        elif args.unsafe:
            store.point_alias(settings.collection_alias, args.collection)
            print(f"alias {settings.collection_alias} -> {args.collection} (UNCHECKED)")
        else:
            point_alias_checked(store, settings, args.collection, load_records())
    finally:
        store.close()


if __name__ == "__main__":
    main()
