"""Reproduce the reindex incident and each fix, deterministically and offline.

Runs against a temporary Qdrant directory (so it never touches .qdrant or a running API) and
uses FakeLLM, which cites the first sentence of the top-ranked document: exactly the kind of
answer whose citation the incident corrupted.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.embeddings import Embedder
from app.clients.llm import FakeLLM
from app.clients.qdrant_store import QdrantStore
from app.config import Principal, Settings
from app.records import load_records
from app.services.answer import AnswerService
from app.services.indexing import integrity_problems
from app.services.retrieval import Retriever
from scripts.build_index import V1, V2, seed

QUESTION = "What is the CSV export row limit?"
CUSTOMER = Principal(customer_id="C-1001", role="customer")
OTHER = Principal(customer_id="C-1002", role="customer")


def main() -> None:
    records = {r.source_id: r for r in load_records()}
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(qdrant_path=Path(tmp), qdrant_url=None, _env_file=None)
        store = QdrantStore(settings)
        embedder = Embedder(settings)
        try:
            print("Building kb_v1 (legacy indexer) and kb_v2 (fixed indexer)...")
            seed(store, embedder, settings)
            run_steps(store, embedder, settings, records)
        finally:
            store.close()


def run_steps(store, embedder, settings, records) -> None:
    retriever = Retriever(store, embedder, settings)
    service = AnswerService(retriever, FakeLLM())

    step("1. SYMPTOM: alias on kb_v1, legacy_naive fusion (the pre-fix production state)")
    store.point_alias(settings.collection_alias, V1)
    settings_naive = settings.model_copy(update={"fusion": "legacy_naive"})
    show_answer(AnswerService(Retriever(store, embedder, settings_naive), FakeLLM()), records)

    step("2. ROOT CAUSE A: kb_v1 labels points with the wrong source record")
    problems = integrity_problems(store, V1, list(records.values()))
    print(f"   integrity check on kb_v1: {len(problems)} problems, e.g. {problems[:2]}")
    leaks = cross_tenant_text(retriever, records)
    print(f"   consequence for C-1002 on kb_v1: {leaks} retrieved chunks contain C-1001 text")

    step("3. ROOT CAUSE B: raw BM25 + cosine sum lets the keyword-stuffed glossary win (on kb_v2)")
    store.point_alias(settings.collection_alias, V2)
    for fusion in ("legacy_naive", "rrf"):
        top = retriever.retrieve(QUESTION, CUSTOMER, 3, fusion=fusion).chunks
        ranked = ", ".join(f"{c.source_id} ({c.score:.3f})" for c in top)
        print(f"   {fusion:>12}: {ranked}")

    step("4. FIX: alias -> kb_v2 (integrity-checked) and fusion=rrf")
    v2_problems = integrity_problems(store, V2, list(records.values()))
    print(f"   integrity problems on kb_v2: {len(v2_problems)}")
    show_answer(service, records)
    print(
        f"   consequence for C-1002 on kb_v2: {cross_tenant_text(retriever, records)} leaked chunks"
    )


def show_answer(service: AnswerService, records) -> None:
    result = service.ask(QUESTION, CUSTOMER, 5, "incident")
    print(f"   collection={result.metadata.collection} fusion={result.metadata.fusion}")
    for c in result.citations:
        owner = next(r.source_id for r in records.values() if c.excerpt in r.text)
        verdict = "correct" if owner == c.source_id else f"WRONG: text belongs to {owner}"
        print(f'   cited [{c.source_id}] "{c.excerpt}" -> {verdict} ({records[owner].status})')


def cross_tenant_text(retriever: Retriever, records) -> int:
    c1001_texts = [r.text for r in records.values() if r.customer_id == "C-1001"]
    questions = ["invoice charged twice", "webhook failing", "rate limited", "two-factor codes"]
    return sum(
        any(chunk.text in t for t in c1001_texts)
        for q in questions
        for chunk in retriever.retrieve(q, OTHER, 5).chunks
    )


def step(title: str) -> None:
    print(f"\n{title}")


if __name__ == "__main__":
    main()
