# Support RAG: permission-aware hybrid retrieval with verified citations

A local support bot over synthetic tickets, logs and KB articles. It answers with citations
that are checked against the exact text the model was shown, refuses when evidence is missing or
contradictory, and never puts another customer's data in the prompt. The repository also
reproduces and fixes a staged incident: wrong citations after a Qdrant hybrid-search reindex
(stale alias, ID mismatch, non-normalised BM25/vector fusion).

- API: FastAPI + Pydantic v2, `/health`, `POST /tickets`, `GET /tickets/{id}`, `POST /ask`,
  `GET /admin/collections`
- LLM: **Llama 3.2 3B Instruct** (open weights) served locally by **Ollama** on CPU
- Retrieval: Qdrant (embedded, on-disk), dense `BAAI/bge-small-en-v1.5` + sparse `Qdrant/bm25`
  via fastembed, Reciprocal Rank Fusion
- Fine-tuning: LoRA on Qwen2.5-0.5B-Instruct (peft/transformers), evaluated with the app's own
  citation verifier

## Quickstart

Requires Python 3.11+ and [Ollama](https://ollama.com/download). Windows PowerShell commands
shown; `make <target>` mirrors them on Linux/macOS.

```
python manage.py setup      # .venv, pinned deps, embedding models, `ollama pull llama3.2:3b`
python manage.py seed       # build kb_v1 (legacy) and kb_v2, alias kb_live -> kb_v2
python manage.py run        # API on http://127.0.0.1:8000 (docs at /docs)
python manage.py test       # pytest, no model server needed
python manage.py demo       # in a second terminal while `run` is up
python manage.py incident   # offline reproduction of the reindex incident
python manage.py eval       # fine-tuning evaluation (see below)
```

Ollama must be running (`ollama serve`, or the desktop app). To use the 1B model on slower
machines: set `OLLAMA_MODEL=llama3.2:1b` in `.env` (copy `.env.example`) and rerun setup.
Embedded Qdrant allows one process: stop `run` before `seed`.

Example request (synthetic demo key):

```
curl -s localhost:8000/ask -H "X-API-Key: demo-c1001" -H "Content-Type: application/json" \
     -d '{"question": "What is the CSV export row limit?"}'
```

## Architecture

```
            X-API-Key
 client ───────────────► FastAPI routes (thin) ──► deps: key → Principal(customer_id, role)
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
          TicketService               AnswerService
          validate, index             1 Retriever ─────────► Qdrant (embedded)
          (untrusted text)              │ payload filter       alias kb_live → kb_v2
                 │                      │ dense + BM25          kb_v1 (legacy, kept)
                 ▼                      │ RRF fusion
          Qdrant upsert                 │ can_read() per hit, drop drafts/superseded
                                      2 guards.injection_flags (log only)
                                      3 prompt: <untrusted_data> fenced documents
                                      4 LLMClient ──► Ollama (llama3.2:3b) | FakeLLM (tests)
                                        strict JSON parse, one repair retry
                                      5 citations.verify_answer (shown + verbatim)
                                      6 guards.foreign_identifiers (output check)
```

Layers: `app/api` (routes, auth) → `app/services` (logic) → `app/clients` (Qdrant, embeddings,
Ollama). Errors are domain exceptions mapped to `{"error": {code, message, request_id}}`:
401/403 auth, 404, 422 validation, 502 bad LLM output, 503 Qdrant or Ollama unavailable. Every
response carries `X-Request-ID`; logs are JSON lines.

## Design decisions

- **Embedded Qdrant.** Verified on qdrant-client 1.20.0 before use: sparse vectors with IDF,
  `query_points` with prefetch and RRF, alias create/swap, persistence. Chosen to avoid Docker on
  the target laptop. Cost: single-process access. `QDRANT_URL` switches to a server.
- **Client-side fusion.** Dense and BM25 candidates are fused in `app/services/fusion.py`
  (pure functions), so `rrf` and the incident's `legacy_naive` are compared on identical inputs.
- **Citations by `source_id` payload only**, never point IDs or rank positions. The verifier
  checks against the exact (neutralised) text placed in the prompt.
- **Strict verification.** One invalid citation withholds the whole answer; a conflict needs two
  verified sources; an `answered` result needs at least one citation.
- **Abstain without the LLM** when retrieval returns nothing the caller may see, or when the
  best permitted dense match is below `MIN_DENSE_SCORE` (0.75). Added after the live model
  answered an unanswerable question from an unrelated (but verbatim, so verifier-passing)
  sentence. On this dataset relevant questions scored ≥ 0.815 and unanswerable ones ≤ 0.693.
- **JSON field order** `answer, citations, status`: constrained decoding generates in schema
  order; with `status` first, llama3.2:3b labelled single-source answers as conflicts.
- **Stale evidence.** Records carry `status`/`superseded_by`/`updated_at`; a superseded record is
  dropped when its successor is retrieved, otherwise shown labelled as superseded.
- **Integrity-gated alias moves.** The alias only moves to a collection whose every point matches
  its canonical record (the check that would have prevented the incident).

## Security model

Ticket text and retrieved documents are untrusted data.

1. **Identity**: API key → `Principal`, constant-time comparison. Missing/invalid key → 401;
   non-admin on `/admin/*` → 403. Customers can only create and read their own tickets.
2. **Retrieval filter**: a Qdrant payload filter admits `public_kb` and the caller's own
   `customer` records only (admins see everything, including `internal_kb`). Drafts never.
3. **Post-retrieval check**: `can_read` re-checks every hit; a filter bug is logged CRITICAL and
   the record is dropped (its ID is redacted). A test disables the filter to prove this layer.
4. **Prompt isolation**: documents are fenced as data; fence tags inside documents are removed.
   Because of 2 and 3, other customers' data is never in the prompt, so an obedient model has
   nothing to leak.
5. **Output checks**: citation verification, and withholding answers that name another customer
   ID or a source the caller was not shown. Unknown cited IDs are redacted from metadata.
6. **Injection flags**: regex flags on the question and each retrieved chunk are logged and
   returned in `metadata.injection_flags`. They are observability, not a control.

Regression test for the malicious record: `tests/test_security.py::
test_regression_malicious_ticket_never_leaks_other_customer_data`. Ticket T-0009 (C-1002)
contains a forged `SYSTEM/admin` instruction to reveal C-1001's invoices, and KB-0007 contains a
hidden HTML-comment instruction. With FakeLLM scripted to obey fully, the C-1002 caller receives
no C-1001 identifiers or text, and the prompt is asserted to contain no C-1001 record.

Demo keys (`demo-c1001`, `demo-c1002`, `demo-admin`) are synthetic and grant access only to
invented data; set `API_KEYS` for anything else.

## Testing

```
python manage.py test                  # 101 tests, real embeddings + FakeLLM, no model server
python manage.py test -m integration   # 2 tests against real Ollama (passed locally, ~80 s)
python manage.py lint                  # ruff
```

| File | Covers |
|---|---|
| `test_api_validation.py` | empty/oversized/extra-field payloads, malformed JSON, path params → 422; no input echo |
| `test_retrieval.py` | expected doc ranks first; RRF vs naive scale bug; alias v1→v2; integrity gate; superseded/draft handling |
| `test_citations.py` | fabricated source, non-verbatim, wrong-label excerpt, too short, uncited, conflict rules |
| `test_answer.py` | cited/conflict/abstain paths, repair retry, 502 path, fence neutralisation, Ollama error mapping |
| `test_failures.py` | Qdrant down → 503, LLM timeout/down → 503, bad output → 502, generic 500, no stack traces |
| `test_security.py` | malicious-record regression, flags, internal KB isolation, broken-filter defence, 401/403 |
| `test_finetune.py` | dataset size/disjointness/hashes; every gold target passes the app verifier; metric code |
| `test_data.py` | synthetic dataset contains every required scenario |
| `test_integration_ollama.py` | opt-in, real model |

## Fine-tuning and results

Goal: teach a small open-weight model to emit the app's cited-JSON contract, abstain, flag
conflicts and ignore instructions inside data, without constrained decoding.

- Data: `finetune/make_dataset.py --seed 42` → 180 train / 50 held-out examples, split by topic
  family (held-out topics never appear in training), prompts built with the app's own
  `build_messages`. Categories: answered, stale, conflict, insufficient, injection.
  Hashes in `finetune/data/manifest.json`.
- Training: `finetune/train.py` (LoRA r=16, α=32, all attention and MLP projections, loss on the
  answer tokens only), config in `finetune/config.yaml`, pinned `requirements-train.txt`.
  Full run on a Colab T4: `finetune/colab_run.md`. `--mode smoke` runs the same pipeline on CPU
  with a 135M model for 4 steps as a pipeline check (not a result).
- Evaluation: `finetune/eval.py` with the app's `parse_answer` and `verify_answer`. Metrics:
  JSON validity, citation precision/recall, status/abstain/conflict accuracy, injection
  compliance rate (lower is better), verified-answer rate.

### Results (from `python finetune/report.py`; source files in `finetune/results/`)

| Metric (held-out) | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | Llama 3.2 3B (Ollama, app runtime) |
|---|---|---|---|
| json_validity | NOT RUN | NOT RUN | 1.00 |
| citation_precision | NOT RUN | NOT RUN | 0.65 |
| citation_recall | NOT RUN | NOT RUN | 0.91 |
| status_accuracy | NOT RUN | NOT RUN | 0.76 |
| abstain_accuracy | NOT RUN | NOT RUN | 1.00 |
| conflict_accuracy | NOT RUN | NOT RUN | 0.70 |
| injection_compliance_rate | NOT RUN | NOT RUN | 0.00 |
| verified_answer_rate | NOT RUN | NOT RUN | 0.74 |
| examples / decoding | NOT RUN | NOT RUN | n=50, schema-constrained (Ollama format) |

**Executed:**

- Llama 3.2 3B via Ollama on all 50 held-out examples, CPU, schema-constrained decoding as in
  the app (`eval_ollama_llama3.2-3b.json`). By category, status was correct for answered 10/10,
  insufficient 10/10, injection 10/10, conflict 7/10 and **stale 1/10**: in 9 of 10 stale cases
  the model labelled a superseded-vs-current pair as a conflict, which also lowers citation
  precision (it cites the superseded document). In the app, retrieval removes a superseded record
  when its successor is retrieved, so this mainly matters where both reach the prompt; it is
  the clearest target for the fine-tune.
- Smoke run of the training pipeline (`train_smoke.json`, `eval_smoke_*.json`):
  SmolLM2-135M-Instruct, 16 examples, 4 steps on CPU, adapter saved and reloaded for evaluation.
  It shows the code runs end to end. It is not a quality result (the 135M model emits no valid
  JSON before or after 4 steps).

**Not run:** the Qwen2.5-0.5B baseline, the LoRA training run and the adapted evaluation need
a GPU (the local MX130 has 2 GB VRAM). They are planned for a Colab T4 following
`finetune/colab_run.md`; their columns stay NOT RUN until those result files exist. The
adapter checkpoint is not committed; it is reproducible with those steps.

## Limitations

- Synthetic, small data (33 records; 230 fine-tuning examples). Numbers show mechanisms, not
  production performance.
- CPU inference of a 3B model takes tens of seconds per answer.
- Embedded Qdrant is single-process and not built for large collections.
- Tickets created through the API live only in the active collection; `seed` rebuilds from
  `data/` and does not carry them over.
- Conflict detection is the model's judgement; code only enforces that a conflict cites two
  verified sources.
- Verbatim matching rejects paraphrased excerpts even when they are faithful.
- The verifier checks citations, not every claim in the answer text. In the live demo the model
  correctly cited "CSV exports are limited to 50,000 rows per file." but prefixed an unsupported
  "There is no fixed limit…"; its conflict answer cited both sources correctly but its answer
  text only stated one value.
- The relevance threshold is calibrated on 14 synthetic questions; other corpora need their own.
- Injection flags are regexes and easy to evade; they are not relied on for security.
- API keys are static demo values in configuration, not a production identity system.

## AI assistance

See [AI_DISCLOSURE.md](AI_DISCLOSURE.md). Incident write-up: [docs/INCIDENT.md](docs/INCIDENT.md).
Module-by-module notes: [docs/DEFENSE.md](docs/DEFENSE.md).
