"""Citation verification. Pure functions; the app and finetune/eval.py both use this module.

Rules: every cited source_id must be one the model was actually shown (retrieved AND permitted),
and every excerpt must appear verbatim (whitespace-normalised) in that source's shown text.
Any violation withholds the whole answer: an answer partly built on a fabricated source cannot
be trusted just because its other citations are real.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.schemas import Citation, LLMAnswer, RejectedCitation

MIN_EXCERPT_CHARS = 8
WITHHELD_ANSWER = (
    "The answer could not be verified against the available records, so it is withheld."
)


@dataclass(frozen=True)
class Verification:
    answer: LLMAnswer
    rejected: list[RejectedCitation]
    downgraded: bool


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def citation_problem(citation: Citation, shown: Mapping[str, Sequence[str]]) -> str | None:
    """Return why a citation is invalid, or None if it is valid."""
    texts = shown.get(citation.source_id)
    if texts is None:
        return "unknown_source"
    excerpt = normalize_ws(citation.excerpt)
    if len(excerpt) < MIN_EXCERPT_CHARS:
        return "excerpt_too_short"
    if not any(excerpt in normalize_ws(t) for t in texts):
        return "excerpt_not_verbatim"
    return None


def verify_answer(answer: LLMAnswer, shown: Mapping[str, Sequence[str]]) -> Verification:
    rejected = [
        # An unknown source_id is model output the caller was never shown; it could be a real
        # identifier from another tenant, so it is never echoed back.
        RejectedCitation(
            source_id=c.source_id if problem != "unknown_source" else "redacted", reason=problem
        )
        for c in answer.citations
        if (problem := citation_problem(c, shown))
    ]
    distinct_sources = {c.source_id for c in answer.citations}
    if rejected:
        return _withhold(rejected)
    if answer.status == "answered" and not answer.citations:
        return _withhold([RejectedCitation(source_id="-", reason="answer_without_citations")])
    if answer.status == "conflicting_evidence" and len(distinct_sources) < 2:
        return _withhold([RejectedCitation(source_id="-", reason="conflict_needs_two_sources")])
    return Verification(answer=answer, rejected=[], downgraded=False)


def _withhold(rejected: list[RejectedCitation]) -> Verification:
    safe = LLMAnswer(status="insufficient_evidence", answer=WITHHELD_ANSWER, citations=[])
    return Verification(answer=safe, rejected=rejected, downgraded=True)
