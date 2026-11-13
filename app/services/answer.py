"""Question answering: retrieve -> prompt -> LLM (one repair retry) -> verify -> output guard."""

import logging

from app.clients.llm import LLMClient, Message
from app.config import Principal
from app.errors import LLMOutputError
from app.schemas import AskMetadata, AskResponse, LLMAnswer, RejectedCitation
from app.services.citations import WITHHELD_ANSWER, verify_answer
from app.services.guards import foreign_identifiers, injection_flags
from app.services.parsing import parse_answer
from app.services.prompt import build_messages, repair_messages
from app.services.retrieval import Retriever

log = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = "No records you can access contain information about this question."


class AnswerService:
    def __init__(self, retriever: Retriever, llm: LLMClient) -> None:
        self._retriever = retriever
        self._llm = llm

    def ask(self, question: str, principal: Principal, top_k: int, request_id: str) -> AskResponse:
        retrieval = self._retriever.retrieve(question, principal, top_k)
        flags = injection_flags("question", question, principal)
        for chunk in retrieval.chunks:
            flags += injection_flags(chunk.source_id, chunk.text, principal)
        if flags:
            log.warning("suspicious instructions detected", extra={"flags": flags})

        def meta(rejected: list[RejectedCitation], downgraded: bool) -> AskMetadata:
            return AskMetadata(
                request_id=request_id,
                collection=retrieval.collection,
                fusion=retrieval.fusion,
                model=self._llm.model,
                retrieved=list(dict.fromkeys(c.source_id for c in retrieval.chunks)),
                excluded=retrieval.excluded,
                injection_flags=sorted(set(flags)),
                rejected_citations=rejected,
                downgraded=downgraded,
            )

        if not retrieval.chunks:
            # Nothing the caller may see matched: abstain without spending an LLM call.
            return AskResponse(
                status="insufficient_evidence",
                answer=NO_EVIDENCE_ANSWER,
                citations=[],
                metadata=meta([], False),
            )

        messages, shown = build_messages(question, retrieval.chunks)
        verification = verify_answer(self._generate(messages), shown)
        answer = verification.answer
        rejected = list(verification.rejected)

        leaked = foreign_identifiers(answer.answer, principal, shown)
        if leaked:
            log.error("answer withheld: foreign identifiers", extra={"ids": leaked})
            rejected.append(RejectedCitation(source_id="-", reason="foreign_identifier_in_answer"))
            answer = LLMAnswer(status="insufficient_evidence", answer=WITHHELD_ANSWER, citations=[])

        downgraded = verification.downgraded or bool(leaked)
        return AskResponse(
            status=answer.status,
            answer=answer.answer,
            citations=answer.citations,
            metadata=meta(rejected, downgraded),
        )

    def _generate(self, messages: list[Message]) -> LLMAnswer:
        """Call the LLM and parse strictly, with exactly one repair attempt."""
        raw = self._llm.complete(messages)
        try:
            return parse_answer(raw)
        except LLMOutputError as first:
            log.warning("malformed LLM output, repairing", extra={"err": first.message})
            # DEBUG: frequent repairs usually mean the model is too small for the JSON contract
            # or num_ctx truncated the prompt; check OLLAMA_MODEL and the prompt length.
            raw = self._llm.complete(repair_messages(messages, raw, first.message))
            return parse_answer(raw)
