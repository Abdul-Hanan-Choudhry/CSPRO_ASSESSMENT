"""Health and collection inspection for operators."""

from app import __version__
from app.clients.llm import LLMClient
from app.clients.qdrant_store import QdrantStore
from app.errors import AppError
from app.schemas import CollectionInfo, CollectionsResponse, DependencyStatus, HealthResponse


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
