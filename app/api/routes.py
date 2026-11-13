"""HTTP routes. Deliberately thin: validate, authenticate, delegate to a service."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request

from app.api.deps import get_principal, get_services, require_admin
from app.config import Principal
from app.container import Services
from app.schemas import (
    AskRequest,
    AskResponse,
    CollectionsResponse,
    HealthResponse,
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    TicketCreate,
    TicketCreated,
    TicketView,
)
from app.services.status import check_health, debug_retrieve, describe_collections

router = APIRouter()
ServicesDep = Annotated[Services, Depends(get_services)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]
TicketId = Annotated[str, Path(pattern=r"^T-[0-9A-F]{4,8}$")]


@router.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    return check_health(services.store, services.llm, services.settings.collection_alias)


@router.post("/tickets", response_model=TicketCreated, status_code=201)
def create_ticket(body: TicketCreate, principal: PrincipalDep, services: ServicesDep):
    return services.tickets.create(body, principal)


@router.get("/tickets/{ticket_id}", response_model=TicketView)
def get_ticket(ticket_id: TicketId, principal: PrincipalDep, services: ServicesDep):
    return services.tickets.get(ticket_id, principal)


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, principal: PrincipalDep, services: ServicesDep):
    return services.answers.ask(body.question, principal, body.top_k, request.state.request_id)


@router.get(
    "/admin/collections",
    response_model=CollectionsResponse,
    dependencies=[Depends(require_admin)],
)
def admin_collections(services: ServicesDep) -> CollectionsResponse:
    return describe_collections(services.store, services.settings.collection_alias)


@router.post(
    "/admin/retrieve",
    response_model=RetrievalDebugResponse,
    dependencies=[Depends(require_admin)],
)
def admin_retrieve(body: RetrievalDebugRequest, services: ServicesDep) -> RetrievalDebugResponse:
    return debug_retrieve(services.retriever, body)
