"""Health and collection inspection for operators."""

from app import __version__
from app.clients.llm import LLMClient
from app.clients.qdrant_store import QdrantStore
from app.config import Principal
from app.errors import AppError
from app.schemas import (
    CollectionInfo,
    CollectionsResponse,
    DependencyStatus,
    HealthResponse,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RetrievedChunk,
)
from app.services.retrieval import Retriever


def check_health(store: QdrantStore, llm: LLMClient, alias: str) -> HealthResponse:
    active: str | None = None
    try:
        active = store.resolve_alias(alias)
        vector = DependencyStatus(
            ok=active is not None,
            detail=f"alias {alias} -> {active}" if active else f"alias {alias} not set; run seed",
        )
    except AppError as exc:
        vector = DependencyStatus(ok=False, detail=exc.message)
    try:
        llm_status = DependencyStatus(ok=True, detail=llm.health())
    except AppError as exc:
        llm_status = DependencyStatus(ok=False, detail=exc.message)
    return HealthResponse(
        status="ok" if vector.ok and llm_status.ok else "degraded",
        version=__version__,
        active_collection=active,
        vector_store=vector,
        llm=llm_status,
    )


def debug_retrieve(retriever: Retriever, body: RetrievalDebugRequest) -> RetrievalDebugResponse:
    """Retrieval only, as a chosen customer (or admin), so fusion and filters can be inspected."""
    principal = (
        Principal(customer_id=body.as_customer, role="customer")
        if body.as_customer
        else Principal(customer_id=None, role="admin")
    )
    result = retriever.retrieve(body.question, principal, body.top_k, fusion=body.fusion)
    return RetrievalDebugResponse(
        collection=result.collection,
        fusion=result.fusion,
        chunks=[
            RetrievedChunk(
                source_id=c.source_id,
                chunk_index=c.chunk_index,
                kind=c.kind,
                status=c.status,
                score=round(c.score, 4),
                text=c.text,
            )
            for c in result.chunks
        ],
        excluded=result.excluded,
    )


def describe_collections(store: QdrantStore, alias: str) -> CollectionsResponse:
    infos = []
    for name in store.list_collections():
        sample = store.scroll_payloads(name, limit=1)
        infos.append(
            CollectionInfo(
                name=name,
                points=store.count(name),
                schema_version=sample[0].get("schema_version") if sample else None,
            )
        )
    return CollectionsResponse(
        alias=alias, active_collection=store.resolve_alias(alias), collections=infos
    )
