"""Citation verifier: the core guarantee that every citation is real and verbatim."""

from app.schemas import Citation, LLMAnswer
from app.services.citations import WITHHELD_ANSWER, verify_answer

SHOWN = {
    "KB-0004": ["Refunds can be requested within 14 days of the invoice date by opening a ticket."],
    "KB-0005": ["Refunds can be requested within 30 days of the invoice date."],
}


def _answer(status: str, *cites: tuple[str, str]) -> LLMAnswer:
    return LLMAnswer(
        status=status,
        answer="text",
        citations=[Citation(source_id=s, excerpt=e) for s, e in cites],
    )


def test_correct_citation_accepted():
    result = verify_answer(_answer("answered", ("KB-0005", "within 30 days")), SHOWN)
    assert not result.downgraded and result.answer.status == "answered"


def test_whitespace_differences_tolerated():
    excerpt = "within   30 days\nof the invoice date"
    assert not verify_answer(_answer("answered", ("KB-0005", excerpt)), SHOWN).downgraded


def test_fabricated_source_id_rejected():
    result = verify_answer(_answer("answered", ("KB-9999", "within 30 days")), SHOWN)
    assert result.downgraded
    assert result.rejected[0].reason == "unknown_source"
    assert result.answer.answer == WITHHELD_ANSWER and result.answer.citations == []


def test_non_verbatim_excerpt_rejected():
    result = verify_answer(_answer("answered", ("KB-0005", "within thirty days")), SHOWN)
    assert result.rejected[0].reason == "excerpt_not_verbatim"


def test_excerpt_from_another_source_rejected():
    # Real text, wrong label: exactly the incident's symptom.
    result = verify_answer(_answer("answered", ("KB-0004", "within 30 days")), SHOWN)
    assert result.rejected[0].reason == "excerpt_not_verbatim"


def test_one_bad_citation_withholds_whole_answer():
    answer = _answer("answered", ("KB-0005", "within 30 days"), ("KB-0004", "within 99 days"))
    assert verify_answer(answer, SHOWN).answer.status == "insufficient_evidence"


def test_trivially_short_excerpt_rejected():
    result = verify_answer(_answer("answered", ("KB-0005", "30")), SHOWN)
    assert result.rejected[0].reason == "excerpt_too_short"


def test_uncited_answer_downgraded():
    result = verify_answer(_answer("answered"), SHOWN)
    assert result.downgraded and result.rejected[0].reason == "answer_without_citations"


def test_conflict_with_two_sources_accepted():
    answer = _answer(
        "conflicting_evidence", ("KB-0004", "within 14 days"), ("KB-0005", "within 30 days")
    )
    assert verify_answer(answer, SHOWN).answer.status == "conflicting_evidence"


def test_conflict_with_one_source_downgraded():
    result = verify_answer(_answer("conflicting_evidence", ("KB-0005", "within 30 days")), SHOWN)
    assert result.rejected[0].reason == "conflict_needs_two_sources"


def test_insufficient_evidence_without_citations_accepted():
    result = verify_answer(_answer("insufficient_evidence"), SHOWN)
    assert not result.downgraded
