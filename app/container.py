"""Wires clients and services together once per process. Tests pass in their own parts."""

from dataclasses import dataclass

from app.clients.embeddings import Embedder
from app.clients.llm import LLMClient, build_llm
from app.clients.qdrant_store import QdrantStore
from app.config import Settings
from app.services.answer import AnswerService
from app.services.retrieval import Retriever
from app.services.tickets import TicketService


@dataclass
class Services:
    settings: Settings
    store: QdrantStore
    llm: LLMClient
    answers: AnswerService
    tickets: TicketService


def build_services(
    settings: Settings,
    *,
    store: QdrantStore | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> Services:
    store = store or QdrantStore(settings)
    embedder = embedder or Embedder(settings)
    llm = llm or build_llm(settings)
    retriever = Retriever(store, embedder, settings)
    return Services(
        settings=settings,
        store=store,
        llm=llm,
        answers=AnswerService(retriever, llm),
        tickets=TicketService(store, embedder, settings),
    )
