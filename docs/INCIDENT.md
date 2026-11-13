# Incident: wrong citations after the hybrid-search reindex

Everything below is reproducible offline with:

```
python manage.py incident
```

It builds both collection versions in a temporary directory, replays the broken and fixed
states, and prints the numbers quoted here. The regression tests named at the end lock in each fix.

## Symptom

After a reindex, the support bot answered "What is the CSV export row limit?" with:

```
cited [KB-000B] "CSV exports are limited to 10,000 rows per file"
```

Two things are wrong. KB-000B is the export *glossary*, not the limits article, and the
sentence actually comes from KB-0002, which is **superseded** (current limit: 50,000, KB-0003).
The citation verifier did not catch it, because the label and the text were wrong *together*
inside the index: the model faithfully quoted what it was shown under that label.

## Investigation

1. **Which collection is serving?** `GET /admin/collections` (admin key) showed alias `kb_live`
   still targeting `kb_v1` while `kb_v2` existed. The reindex built v2 but the alias swap was
   never applied, so production served the old index. **Root cause 1: stale alias.**

2. **Is v1 internally consistent?** The integrity check (`app/services/indexing.py:
   integrity_problems`) compares every point with the canonical record carrying its `source_id`.
   On kb_v1 it reports **29 problems** (out of 33 points): point text does not belong to the
   labelled source, and owner/visibility differ from the source record. The legacy indexer
   (`scripts/build_index.py: build_v1_legacy`, kept verbatim for reproduction) took vectors and
   text from the records in file order but metadata from a list sorted by `updated_at`, and used
   the row position as point ID. **Root cause 2: ID/metadata mismatch.**

   The same mismatch also breaks tenant isolation: the Qdrant payload filter trusts
   `customer_id` in the payload, which is wrong for most v1 points. With the alias on v1,
   customer C-1002 retrieves **6 chunks containing C-1001's text** across four probe questions.
   The post-retrieval permission check cannot catch this either, because it reads the same
   mislabelled payload. Index integrity is a precondition of every downstream control.

3. **Is ranking sane on the correct index?** On kb_v2 the old fusion formula still ranks the
   wrong document first:

   | fusion | top 3 for "What is the CSV export row limit?" |
   |---|---|
   | `legacy_naive` | KB-000B (7.296), KB-0002 (7.085), T-0001 (6.833) |
   | `rrf` | KB-0003 (0.032), T-0001 (0.032) |

   `legacy_naive` adds raw cosine similarity (0 to 1) to raw BM25 (unbounded, here 5 to 15).
   Fused scores above 1.0 are impossible for cosine alone, which shows BM25's scale dominating:
   the keyword-dense glossary wins on term frequency regardless of meaning.
   **Root cause 3: non-normalised score fusion.** RRF uses ranks only, so neither retriever's
   scale matters. (KB-0002 is absent under RRF because its successor KB-0003 was retrieved and
   the retriever drops superseded records in that case.)

## Fix

| Cause | Fix | Where |
|---|---|---|
| Stale alias | Alias moves only through `point_alias_checked`, which runs the integrity check and refuses a failing collection; seeding always ends with the alias on kb_v2. `/health` and `/admin/collections` show the active collection. | `scripts/build_index.py`, `app/services/status.py` |
| ID mismatch | One point per chunk built from a single record object; point ID = `uuid5(source_id#chunk)`, independent of file order; citations come from the `source_id` payload, never from point IDs or rank positions. | `app/services/indexing.py` |
| Score scale | `FUSION=rrf` (default); `legacy_naive` kept only to reproduce the bug. | `app/services/fusion.py` |

After the fix the same question cites `[KB-0003] "CSV exports are limited to 50,000 rows per
file"` (current), kb_v2 has 0 integrity problems, and C-1002 retrieves 0 chunks of C-1001 text.

## Regression tests

| Test | Locks in |
|---|---|
| `test_retrieval.py::test_alias_on_legacy_v1_mislabels_sources_and_v2_fixes_it` | v1 mislabels, v2 does not |
| `test_retrieval.py::test_integrity_check_rejects_v1_and_accepts_v2` | integrity check |
| `test_retrieval.py::test_checked_alias_swap_refuses_corrupt_collection` | alias gate |
| `test_retrieval.py::test_legacy_index_leaks_across_tenants_and_v2_does_not` | tenant consequence |
| `test_retrieval.py::test_naive_fusion_reproduces_keyword_stuffing_bug` | score-scale bug on real data |
| `test_retrieval.py::test_naive_sum_is_dominated_by_bm25_scale_but_rrf_is_not` | same, pure function |
| `test_retrieval.py::test_superseded_article_dropped_when_successor_retrieved` | stale evidence |
| `test_citations.py::test_excerpt_from_another_source_rejected` | real text under a wrong label |

## What this does not show

The incident is synthetic and staged by design: the legacy indexer was written to contain the
bug. The numbers above are from the bundled 33-record dataset and a fixed embedding model; they
show the mechanism, not production-scale rates.

## Found during the build

The security regression test caught one real leak while building the fix: `rejected_citations`
in the response echoed the model's cited `source_id` even when it was unknown to the caller.
With an LLM scripted to obey the forged ticket, that returned `T-0002` (a C-1001 ticket ID) to
C-1002. Unknown source IDs are now redacted (`app/services/citations.py`), and
`test_security.py::test_regression_malicious_ticket_never_leaks_other_customer_data` covers it.

A live run against llama3.2:3b found a second gap. Asked the C-1002 attack question, the model
returned `conflicting_evidence` citing KB-0007's hidden instruction verbatim, which contains the
string "C-1001". The citation was real and verbatim, so the verifier accepted it, and the output
guard only scanned the answer text. No C-1001 data was exposed (the string comes from a public
article), but a customer response must not carry another customer's identifier. The guard now
scans citation excerpts too; `test_citation_quoting_foreign_customer_id_is_withheld` covers it.
