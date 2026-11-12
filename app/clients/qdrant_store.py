"""Thin wrapper over qdrant-client: versioned collections, the live alias, hybrid queries.

Every call goes through `_call`, so any storage failure surfaces as VectorStoreUnavailableError
(HTTP 503) instead of a raw client exception or stack trace.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from qdrant_client import QdrantClient, models

from app.config import Settings
from app.errors import VectorStoreUnavailableError
from app.retry import call_with_retries

log = logging.getLogger(__name__)
T = TypeVar("T")

DENSE = "dense"
SPARSE = "bm25"
FILTERABLE_FIELDS = ("source_id", "customer_id", "visibility", "status")


@dataclass(frozen=True)
class Hit:
    point_id: str
    score: float
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: QdrantClient | None = None
        # Embedded mode is not thread-safe; FastAPI runs sync endpoints in a thread pool.
        self._lock = threading.Lock()

    def _connect(self) -> QdrantClient:
        if self._client is None:
            s = self._settings
            # DEBUG: "already accessed by another instance" means embedded storage is locked by
            # another process (usually `manage.py run` while seeding). Stop one of them.
            self._client = (
                QdrantClient(url=s.qdrant_url, timeout=s.qdrant_timeout_s)
                if s.qdrant_url
                else QdrantClient(path=str(s.qdrant_path))
            )
        return self._client

    def _call(self, op: str, fn: Callable[[QdrantClient], T]) -> T:
        # Broad catch is deliberate: this is the storage boundary, and any failure here means
        # retrieval cannot proceed safely. The cause is logged; callers see a clean 503.
        try:
            with self._lock:
                return call_with_retries(
                    lambda: fn(self._connect()),
                    retries=self._settings.dependency_retries,
                    retry_on=(ConnectionError, TimeoutError),
                )
        except Exception as exc:
            log.exception("vector store call failed", extra={"op": op})
            raise VectorStoreUnavailableError(f"vector store unavailable during {op}") from exc

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def resolve_alias(self, alias: str) -> str | None:
        aliases = self._call("get_aliases", lambda c: c.get_aliases().aliases)
        return next((a.collection_name for a in aliases if a.alias_name == alias), None)

    def require_alias(self, alias: str) -> str:
        collection = self.resolve_alias(alias)
        if collection is None:
            # DEBUG: no alias means the index was never seeded: run `python manage.py seed`.
            raise VectorStoreUnavailableError(f"alias '{alias}' is not set; run the seed step")
        return collection

    def recreate_collection(self, name: str, dense_dim: int) -> None:
        def op(c: QdrantClient) -> None:
            if c.collection_exists(name):
                c.delete_collection(name)
            c.create_collection(
                name,
                vectors_config={
                    DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
                },
                # IDF modifier makes Qdrant compute BM25's IDF term server-side from corpus stats.
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            if self._settings.qdrant_url:  # payload indexes are a no-op in embedded mode
                for field in FILTERABLE_FIELDS:
                    c.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)

        self._call("recreate_collection", op)

    def upsert(self, collection: str, points: list[models.PointStruct]) -> None:
        self._call("upsert", lambda c: c.upsert(collection, points=points, wait=True))

    def search(
        self,
        collection: str,
        vector: list[float] | models.SparseVector,
        using: str,
        query_filter: models.Filter | None,
        limit: int,
    ) -> list[Hit]:
        result = self._call(
            f"search:{using}",
            lambda c: c.query_points(
                collection,
                query=vector,
                using=using,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            ),
        )
        return [Hit(str(p.id), p.score, p.payload or {}) for p in result.points]

    def point_alias(self, alias: str, collection: str) -> None:
        """Atomically repoint alias at collection (delete + create in one request)."""
        ops: list[Any] = []
        if self.resolve_alias(alias) is not None:
            ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
            )
        )
        self._call(
            "point_alias", lambda c: c.update_collection_aliases(change_aliases_operations=ops)
        )

    def list_collections(self) -> list[str]:
        return sorted(
            self._call(
                "get_collections", lambda c: [x.name for x in c.get_collections().collections]
            )
        )

    def count(self, collection: str) -> int:
        return self._call("count", lambda c: c.count(collection, exact=True).count)

    def scroll_payloads(
        self, collection: str, query_filter: models.Filter | None = None, limit: int = 10_000
    ) -> list[dict[str, Any]]:
        points, _ = self._call(
            "scroll",
            lambda c: c.scroll(
                collection, scroll_filter=query_filter, limit=limit, with_payload=True
            ),
        )
        return [p.payload or {} for p in points]

    def ping(self) -> None:
        self._call("ping", lambda c: c.get_collections())
