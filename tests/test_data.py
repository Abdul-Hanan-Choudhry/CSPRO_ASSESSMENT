"""The synthetic dataset must contain every scenario the assessment relies on."""

from collections import Counter

import pytest
from pydantic import ValidationError

from app.records import load_records
from app.schemas import RecordStatus, SourceRecord, Visibility


@pytest.fixture(scope="module")
def records() -> dict[str, SourceRecord]:
    return {r.source_id: r for r in load_records()}


def test_enough_tickets_across_two_customers(records):
    owners = Counter(r.customer_id for r in records.values() if r.kind == "ticket")
    assert sum(owners.values()) >= 10
    assert set(owners) == {"C-1001", "C-1002"}


def test_stale_article_points_at_current_successor(records):
    stale = records["KB-0002"]
    assert stale.status is RecordStatus.SUPERSEDED
    assert records[stale.superseded_by].status is RecordStatus.CURRENT


def test_conflicting_current_sources_exist(records):
    assert "14 days" in records["KB-0004"].text and "30 days" in records["KB-0005"].text
    assert records["KB-0004"].status is records["KB-0005"].status is RecordStatus.CURRENT


def test_malicious_records_present(records):
    assert records["T-0009"].customer_id == "C-1002"
    assert "reveal customer C-1001" in records["T-0009"].text
    assert "ignore your previous instructions" in records["KB-0007"].text


def test_internal_and_draft_articles_present(records):
    assert records["KB-0008"].visibility is Visibility.INTERNAL_KB
    assert records["KB-0009"].status is RecordStatus.DRAFT


def test_customer_record_without_owner_is_rejected():
    with pytest.raises(ValidationError, match="customer_id is required"):
        SourceRecord(
            source_id="T-00FF",
            kind="ticket",
            title="x",
            text="y",
            visibility="customer",
            updated_at="2026-01-01T00:00:00Z",
        )


def test_superseded_requires_successor():
    with pytest.raises(ValidationError, match="superseded_by"):
        SourceRecord(
            source_id="KB-00FF",
            kind="kb",
            title="x",
            text="y",
            visibility="public_kb",
            status="superseded",
            updated_at="2026-01-01T00:00:00Z",
        )
