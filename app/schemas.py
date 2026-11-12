"""Pydantic models for stored records, API payloads and the LLM's structured output."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_TICKET_SUBJECT_CHARS = 200
MAX_TICKET_BODY_CHARS = 4000
MAX_QUESTION_CHARS = 1000
MAX_EXCERPT_CHARS = 400

SOURCE_ID_PATTERN = r"^(T|KB|LOG)-[0-9A-F]{4,8}$"
CUSTOMER_ID_PATTERN = r"^C-\d{4}$"


class Visibility(StrEnum):
    CUSTOMER = "customer"
    INTERNAL_KB = "internal_kb"
    PUBLIC_KB = "public_kb"


class RecordStatus(StrEnum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


class RecordKind(StrEnum):
    TICKET = "ticket"
    LOG = "log"
    KB = "kb"


class SourceRecord(BaseModel):
    """One retrievable unit of evidence (ticket, log line or KB article)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    kind: RecordKind
    title: str = Field(min_length=1, max_length=MAX_TICKET_SUBJECT_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TICKET_BODY_CHARS)
    customer_id: str | None = Field(default=None, pattern=CUSTOMER_ID_PATTERN)
    visibility: Visibility
    status: RecordStatus = RecordStatus.CURRENT
    updated_at: datetime
    superseded_by: str | None = Field(default=None, pattern=SOURCE_ID_PATTERN)

    @model_validator(mode="after")
    def _check_ownership_and_lifecycle(self) -> "SourceRecord":
        # Security invariant: a customer-visible record without an owner would be readable by
        # nobody (or, with a buggy filter, by everybody). Reject it at the schema boundary.
        if (self.visibility is Visibility.CUSTOMER) != (self.customer_id is not None):
            raise ValueError("customer_id is required iff visibility is 'customer'")
        if self.kind is not RecordKind.KB and self.visibility is not Visibility.CUSTOMER:
            raise ValueError("tickets and logs must have visibility 'customer'")
        if (self.status is RecordStatus.SUPERSEDED) != (self.superseded_by is not None):
            raise ValueError("superseded_by is required iff status is 'superseded'")
        return self


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str = Field(min_length=3, max_length=MAX_TICKET_SUBJECT_CHARS)
    body: str = Field(min_length=3, max_length=MAX_TICKET_BODY_CHARS)
    # Only admins may set this; customers' tickets are always owned by the caller.
    customer_id: str | None = Field(default=None, pattern=CUSTOMER_ID_PATTERN)


class TicketCreated(BaseModel):
    ticket_id: str
    customer_id: str
    indexed_chunks: int
    collection: str
    injection_flags: list[str]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=5, ge=1, le=10)


AnswerStatus = Literal["answered", "insufficient_evidence", "conflicting_evidence"]


class Citation(BaseModel):
    source_id: str = Field(max_length=32)
    excerpt: str = Field(min_length=1, max_length=MAX_EXCERPT_CHARS)


class LLMAnswer(BaseModel):
    """The exact JSON contract the model must emit. Parsed strictly; extra keys rejected."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    answer: str = Field(max_length=2000)
    citations: list[Citation] = Field(default_factory=list, max_length=8)


class RejectedCitation(BaseModel):
    source_id: str
    reason: str


class AskMetadata(BaseModel):
    request_id: str
    collection: str
    fusion: str
    model: str
    retrieved: list[str]
    excluded: list[str]
    injection_flags: list[str]
    rejected_citations: list[RejectedCitation]
    downgraded: bool


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    metadata: AskMetadata


class DependencyStatus(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    active_collection: str | None
    vector_store: DependencyStatus
    llm: DependencyStatus


class CollectionInfo(BaseModel):
    name: str
    points: int
    schema_version: int | None


class CollectionsResponse(BaseModel):
    alias: str
    active_collection: str | None
    collections: list[CollectionInfo]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
