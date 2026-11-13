"""Permission-aware hybrid retrieval: dense + BM25 candidates, fusion, access check, staleness."""

import logging
from dataclasses import dataclass

from app.clients.embeddings import Embedder
from app.clients.qdrant_store import DENSE, SPARSE, Hit, QdrantStore
from app.config import Principal, Settings
from app.errors import VectorStoreUnavailableError
from app.services.chunks import Chunk
from app.services.fusion import FusionMethod, fuse
from app.services.permissions import can_read, qdrant_filter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    collection: str
    fusion: str
    chunks: list[Chunk]
    excluded: list[str]


class Retriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def retrieve(
        self,
        question: str,
        principal: Principal,
        top_k: int,
        fusion: FusionMethod | None = None,
    ) -> RetrievalResult:
        s = self._settings
        method = fusion or s.fusion
        # DEBUG: if citations point to the wrong doc, first confirm alias 'kb_live' targets the
        # newest collection (GET /admin/collections), then that FUSION=rrf.
        collection = self._store.require_alias(s.collection_alias)
        query = self._embedder.embed_query(question)
        access = qdrant_filter(principal)
        n = s.candidates_per_retriever
        dense = self._store.search(collection, query.dense, DENSE, access, n)
        if not dense or dense[0].score < s.min_dense_score:
            # Nothing semantically close: small models otherwise "answer" from an irrelevant
            # chunk with a verbatim (so verifier-passing) but off-topic citation.
            return RetrievalResult(collection, method, [], ["below_relevance_threshold"])
        sparse = self._store.search(collection, query.sparse, SPARSE, access, n)

        hits = {h.point_id: h for h in dense + sparse}
        ranked = fuse(
            method,
            {h.point_id: h.score for h in dense},
            {h.point_id: h.score for h in sparse},
            rrf_k=s.rrf_k,
            bm25_weight=s.naive_bm25_weight,
        )
        chunks, excluded = [], []
        for pid, score in ranked:
            hit = hits[pid]
            if not can_read(principal, hit.payload):
                # Defence in depth: reaching here means the Qdrant filter is wrong. Never pass on.
                log.critical("permission filter leak blocked", extra={"point": pid})
                excluded.append("redacted:forbidden")  # even the foreign source_id is withheld
                continue
            if hit.payload.get("status") == "draft":
                excluded.append(f"{hit.payload['source_id']}:draft")
                continue
            chunks.append(_to_chunk(hit, score))
            if len(chunks) == top_k:
                break
        chunks, stale = drop_superseded(chunks)
        return RetrievalResult(collection, method, chunks, excluded + stale)


def drop_superseded(chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
    """Remove superseded chunks whose successor was also retrieved.

    A superseded chunk without its successor in context is kept (it is still the best evidence
    available) and is labelled superseded in the prompt so the model can say so.
    """
    present = {c.source_id for c in chunks}
    kept = [c for c in chunks if not (c.superseded_by and c.superseded_by in present)]
    dropped = sorted({f"{c.source_id}:superseded" for c in chunks if c not in kept})
    return kept, dropped


def _to_chunk(hit: Hit, score: float) -> Chunk:
    p = hit.payload
    if "source_id" not in p or "text" not in p:
        # DEBUG: points without source_id come from a pre-v2 schema; check which collection the
        # alias targets and rebuild with `python manage.py seed`.
        raise VectorStoreUnavailableError("index payload missing source_id/text; rebuild index")
    return Chunk(
        source_id=p["source_id"],
        chunk_index=p.get("chunk_index", 0),
        title=p.get("title", ""),
        text=p["text"],
        kind=p.get("kind", ""),
        status=p.get("status", "current"),
        updated_at=p.get("updated_at", ""),
        superseded_by=p.get("superseded_by"),
        score=score,
    )
