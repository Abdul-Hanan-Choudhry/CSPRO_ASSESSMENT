"""Ticket ingestion and lookup. Ticket text is untrusted: it is indexed as data, never executed."""

import uuid
from datetime import UTC, datetime

from qdrant_client import models

from app.clients.embeddings import Embedder
from app.clients.qdrant_store import QdrantStore
from app.config import Principal, Settings
from app.errors import InvalidRequestError, NotFoundError, PermissionDeniedError
from app.schemas import SourceRecord, TicketCreate, TicketCreated, TicketView
from app.services.guards import injection_flags
from app.services.indexing import index_records
from app.services.permissions import can_read


class TicketService:
    def __init__(self, store: QdrantStore, embedder: Embedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._alias = settings.collection_alias

    def create(self, body: TicketCreate, principal: Principal) -> TicketCreated:
        owner = _owner_for(body, principal)
        record = SourceRecord(
            source_id=f"T-{uuid.uuid4().hex[:8].upper()}",
            kind="ticket",
            title=body.subject,
            text=body.body,
            customer_id=owner,
            visibility="customer",
            updated_at=datetime.now(UTC),
        )
        # Writes go to the concrete collection the alias targets right now, so a ticket is never
        # split across versions mid-swap.
        collection = self._store.require_alias(self._alias)
        chunks = index_records(self._store, self._embedder, collection, [record])
        owner_view = Principal(customer_id=owner, role="customer")
        return TicketCreated(
            ticket_id=record.source_id,
            customer_id=owner,
            indexed_chunks=chunks,
            collection=collection,
            injection_flags=injection_flags(
                record.source_id, f"{body.subject}\n{body.body}", owner_view
            ),
        )

    def get(self, ticket_id: str, principal: Principal) -> TicketView:
        collection = self._store.require_alias(self._alias)
        only_this_ticket = models.Filter(
            must=[
                models.FieldCondition(key="source_id", match=models.MatchValue(value=ticket_id)),
                models.FieldCondition(key="kind", match=models.MatchValue(value="ticket")),
            ]
        )
        chunks = self._store.scroll_payloads(collection, only_this_ticket, limit=100)
        if not chunks:
            raise NotFoundError(f"ticket {ticket_id} not found")
        if not can_read(principal, chunks[0]):
            raise PermissionDeniedError("this ticket belongs to another customer")
        chunks.sort(key=lambda p: p.get("chunk_index", 0))
        first = chunks[0]
        return TicketView(
            ticket_id=ticket_id,
            customer_id=first["customer_id"],
            subject=first["title"],
            body="\n\n".join(p["text"] for p in chunks),
            updated_at=first["updated_at"],
        )


def _owner_for(body: TicketCreate, principal: Principal) -> str:
    """Customers always own their tickets; only admins may file on behalf of a customer."""
    if principal.role == "admin":
        if body.customer_id is None:
            raise InvalidRequestError("admins must set customer_id when creating a ticket")
        return body.customer_id
    if body.customer_id not in (None, principal.customer_id):
        raise PermissionDeniedError("customers can only create tickets for their own account")
    assert principal.customer_id is not None  # guaranteed by Principal's validator
    return principal.customer_id
