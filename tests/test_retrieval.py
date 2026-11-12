"""Retrieval, fusion and the reindex incident (stale alias, ID mismatch, score scale)."""

import pytest

from app.records import load_records
from app.services.fusion import fuse, legacy_naive, rrf
from app.services.indexing import chunk_text, integrity_problems, point_id
from app.services.retrieval import Chunk, Retriever, drop_superseded
from scripts.build_index import V1, V2
from tests.conftest import C1001

CSV_LIMIT_Q = "What is the CSV export row limit?"


@pytest.fixture
def retriever(live_store, embedder, settings) -> Retriever:
    return Retriever(live_store, embedder, settings)


# --- fusion (pure) ---------------------------------------------------------------------------


def test_rrf_matches_formula():
    scores = rrf([["a", "b"], ["b", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["a"] == pytest.approx(scores["b"])


def test_naive_sum_is_dominated_by_bm25_scale_but_rrf_is_not():
    # "good" is the best semantic match and 2nd by BM25; "stuffed" is keyword-dense only.
    dense = {"good": 0.92, "other": 0.60, "stuffed": 0.40}
    sparse = {"stuffed": 14.0, "good": 9.0}
    naive_top = fuse("legacy_naive", dense, sparse, rrf_k=60, bm25_weight=0.5)[0][0]
    rrf_top = fuse("rrf", dense, sparse, rrf_k=60, bm25_weight=0.5)[0][0]
    assert (naive_top, rrf_top) == ("stuffed", "good")


def test_naive_sum_treats_missing_retriever_as_zero():
    assert legacy_naive({"a": 0.8}, {}, bm25_weight=0.5) == {"a": pytest.approx(0.4)}


# --- retrieval over the real seeded index -----------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [(CSV_LIMIT_Q, "KB-0003"), ("What is the API rate limit?", "KB-0007")],
)
def test_expected_record_ranks_first(retriever, question, expected):
    assert retriever.retrieve(question, C1001, top_k=5).chunks[0].source_id == expected


def test_naive_fusion_reproduces_keyword_stuffing_bug(retriever):
    naive = retriever.retrieve(CSV_LIMIT_Q, C1001, top_k=5, fusion="legacy_naive")
    fixed = retriever.retrieve(CSV_LIMIT_Q, C1001, top_k=5, fusion="rrf")
    assert naive.chunks[0].source_id == "KB-000B"  # glossary, does not answer the question
    assert naive.chunks[0].score > 1.0  # impossible for cosine alone: BM25 scale leaked through
    assert fixed.chunks[0].source_id == "KB-0003"


def test_alias_on_legacy_v1_mislabels_sources_and_v2_fixes_it(retriever, live_store, records):
    def mislabelled(result) -> list[str]:
        return [c.source_id for c in result.chunks if c.text not in records[c.source_id].text]

    live_store.point_alias("kb_live", V1)
    on_v1 = retriever.retrieve(CSV_LIMIT_Q, C1001, top_k=5)
    live_store.point_alias("kb_live", V2)
    on_v2 = retriever.retrieve(CSV_LIMIT_Q, C1001, top_k=5)

    assert (on_v1.collection, on_v2.collection) == (V1, V2)
    assert mislabelled(on_v1), "v1 should reproduce the incident's wrong citations"
    assert mislabelled(on_v2) == []


def test_integrity_check_rejects_v1_and_accepts_v2(store):
    records = load_records()
    assert integrity_problems(store, V1, records)
    assert integrity_problems(store, V2, records) == []


def test_superseded_article_dropped_when_successor_retrieved(retriever):
    result = retriever.retrieve(CSV_LIMIT_Q, C1001, top_k=5)
    assert "KB-0002" not in {c.source_id for c in result.chunks}
    assert "KB-0002:superseded" in result.excluded


def test_draft_articles_never_returned(retriever):
    result = retriever.retrieve("Do you support SAML single sign-on?", C1001, top_k=10)
    assert "KB-0009" not in {c.source_id for c in result.chunks}


# --- chunking and IDs (pure) -----------------------------------------------------------------


def _chunk(source_id: str, superseded_by: str | None = None) -> Chunk:
    return Chunk(source_id, 0, "t", "x", "kb", "current", "", superseded_by, 1.0)


def test_superseded_kept_when_successor_absent():
    kept, dropped = drop_superseded([_chunk("KB-0002", superseded_by="KB-0003")])
    assert [c.source_id for c in kept] == ["KB-0002"] and dropped == []


def test_point_ids_are_stable_and_order_independent():
    assert point_id("KB-0003", 1) == point_id("KB-0003", 1)
    assert point_id("KB-0003", 0) != point_id("KB-0003", 1)


def test_chunking_splits_paragraphs_and_long_text():
    assert chunk_text("a b\n\nc d") == ["a b", "c d"]
    long = " ".join(["word"] * 400)
    assert all(len(c) <= 800 for c in chunk_text(long))
    assert " ".join(chunk_text(long)) == long
