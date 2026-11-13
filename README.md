# Support RAG

A local support bot that answers from synthetic tickets, logs and KB articles with **verified
citations**, abstains when evidence is missing, reports conflicting sources, and never puts
another customer's data in the prompt. It also reproduces and fixes a reindex incident (wrong
citations from a stale alias, mislabelled index and bad BM25/vector score fusion).

- **Model:** Llama 3.2 3B Instruct (open weights), run locally by Ollama on CPU
- **Stack:** Python 3.11+, FastAPI, Pydantic v2, Qdrant (embedded), fastembed (bge-small + BM25)
- **Fine-tuning:** LoRA on Qwen2.5-0.5B-Instruct, evaluated with the app's own citation verifier

Contents: FastAPI service (`app/`), synthetic data (`data/`), tests (`tests/`), index and demo
scripts (`scripts/`), LoRA fine-tuning and evaluation (`finetune/`), incident write-up
(`docs/INCIDENT.md`). The screen recording is sent with the submission email.

## 1. Install

Requirements: Python 3.11+ and [Ollama](https://ollama.com/download) running (`ollama serve` or
the desktop app). Commands are for Windows PowerShell; `make <target>` does the same on
Linux/macOS.

```
python manage.py setup     # .venv, pinned packages, embedding models, ollama pull llama3.2:3b
python manage.py seed      # builds kb_v1 (legacy) and kb_v2, points alias kb_live -> kb_v2
```

Slow machine: set `OLLAMA_MODEL=llama3.2:1b` in `.env` (copy `.env.example`) and rerun setup.

## 2. Run

```
python manage.py run       # API on http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000/ui/** for the test console, or http://127.0.0.1:8000/docs for the
OpenAPI page. Stop the API before running `seed` again (embedded Qdrant allows one process).

## 3. Check that it works

**In the console** (`/ui/`):

| Tab | What to do | What you should see |
|---|---|---|
| Feature checks | Click **Run all checks** (about 3 minutes; LLM checks are slow on CPU) | 19 of 19 pass |
| Ask | Click each preset, then **Ask** | answered with a KB-0003 citation; conflicting_evidence citing KB-0004 and KB-0005; insufficient_evidence; injection withheld |
| Tickets | Create a ticket as C-1002; look up T-0002 as C-1002 | 201 with an `ignore_instructions` flag; 403 |
| Retrieval lab | Click **Compare** | rrf ranks KB-0003 first; legacy_naive ranks the glossary KB-000B first |
| Index & health | Open the tab | alias `kb_live` → `kb_v2` |

**From the command line:**

```
python manage.py test                  # 106 tests, no model server needed
python manage.py test -m integration   # 2 tests against the real model
python manage.py demo                  # CLI demo; needs `run` in another terminal
python manage.py incident              # reproduces the incident and the fix, offline
python manage.py lint                  # ruff
```

Example API call (keys are synthetic: `demo-c1001`, `demo-c1002`, `demo-admin`):

```
curl -s localhost:8000/ask -H "X-API-Key: demo-c1001" -H "Content-Type: application/json" \
     -d '{"question": "What is the CSV export row limit?"}'
```

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Qdrant, active collection, Ollama model status |
| `POST /tickets` | customer/admin | Create and index a ticket (untrusted text) |
| `GET /tickets/{id}` | owner/admin | Read a ticket; other customers get 403 |
| `POST /ask` | customer/admin | Retrieve → model → verify; returns status, answer, citations, metadata |
| `GET /admin/collections` | admin | Alias target, collection versions, point counts |
| `POST /admin/retrieve` | admin | Retrieval only, as any customer, with either fusion method |

Errors are JSON `{"error": {code, message, request_id}}`: 401, 403, 404, 422, 502 (bad model
output), 503 (Qdrant or Ollama unavailable). No stack traces are returned.

## Architecture

```
client / console ─► FastAPI (auth, validation, errors) ─► services ─► clients
                                                           │            ├─ Qdrant: kb_live → kb_v2
  /ask: tenant-filtered dense + BM25 search ─► RRF ─► permission re-check ─► drop stale
        ─► relevance gate ─► fenced prompt ─► llama3.2:3b (Ollama) ─► strict JSON
        ─► citation verifier ─► output guard ─► answer | insufficient | conflicting
```

Security: identity from API key; the Qdrant filter and a second per-hit check keep other
customers' data out of the prompt; documents are fenced as data; citations must be real and
verbatim; answers naming foreign IDs (in the answer or in a quoted excerpt) are withheld;
injection patterns are flagged (logged only). Incident write-up: [docs/INCIDENT.md](docs/INCIDENT.md).

Regression test for the malicious records: `tests/test_security.py::
test_regression_malicious_ticket_never_leaks_other_customer_data`. Ticket T-0009 (C-1002)
contains a forged `SYSTEM/admin` instruction to reveal C-1001's invoices, and KB-0007 contains a
hidden instruction. With the fake model scripted to obey completely, the C-1002 caller receives no
C-1001 data and the prompt is asserted to contain no C-1001 record.

## Fine-tuning results

All numbers below come from the files in `finetune/results/` (`python finetune/report.py` prints
this table). Held-out set: 50 examples on topics never seen in training; one run per model.

| Metric | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | Llama 3.2 3B (Ollama, app runtime) |
|---|---|---|---|
| json_validity | 1.00 | 1.00 | 1.00 |
| citation_precision | 0.44 | 0.62 | 0.65 |
| citation_recall | 0.80 | 0.73 | 0.91 |
| status_accuracy | 0.48 | 0.70 | 0.76 |
| abstain_accuracy | 0.00 | 0.75 | 1.00 |
| conflict_accuracy | 0.00 | 0.50 | 0.70 |
| injection_compliance_rate (lower is better) | 0.20 | 0.00 | 0.00 |
| verified_answer_rate | 0.46 | 0.68 | 0.74 |
| decoding | unconstrained greedy | unconstrained greedy | schema-constrained (Ollama) |

**What was run**

- **LoRA training** (`train_full.json`): Qwen2.5-0.5B-Instruct, r=16, alpha=32, all attention
  and MLP projections (8.8 M of 503 M parameters trainable), 180 examples, 3 epochs, on a Colab
  T4 GPU in 108 s. Held-out loss 0.539 to 0.044.
- **Baseline and adapted evaluation** on the same 50 held-out examples, unconstrained greedy
  decoding (`eval_baseline.json`, `eval_adapted.json`).
- **Llama 3.2 3B** through Ollama on CPU, the model the app serves (`eval_ollama_llama3.2-3b.json`).
  Its decoding is schema-constrained, so it is not a like-for-like comparison with the Qwen columns.
- **CPU smoke run** of the training pipeline (SmolLM2-135M, 4 steps; `*_smoke.json`). It proves
  the code runs; it is not a quality result.

**How to read it**

- The base Qwen model answers `answered` to every example: it never abstains or flags a
  conflict. Its 10/10 on stale-evidence examples is only that.
- The adapter teaches abstaining (9/10 on "no evidence") and conflict flagging (5/10), and
  injection compliance falls from 2 of 10 examples to 0.
- It also gets worse in two places: stale-evidence status (10/10 to 5/10, it calls half of the
  superseded-vs-current pairs a conflict) and citation recall (0.80 to 0.73).
- Base Qwen already emitted valid JSON because the prompt specifies it, so the gain is in
  behaviour (abstain, conflict, injection), not in format.
- Small, template-generated data, 50 examples and one seed: differences of a few examples are
  not significant, and the held-out gains likely overstate what real tickets would show.
- The app does not use the adapted model. It serves Llama 3.2 3B; the adapter is the
  fine-tuning deliverable.

**Checkpoint:** the trained adapter is committed at
`finetune/checkpoint/qwen2.5-0.5b-support-lora/` (33.6 MB; the base model downloads from Hugging
Face). Reproduce the adapted numbers with a GPU (or slowly on CPU) in a venv built from
`requirements-train.txt`:

```
python finetune/eval.py --backend hf --tag adapted --adapter finetune/checkpoint/qwen2.5-0.5b-support-lora
```

Retrain from scratch: [finetune/colab_run.md](finetune/colab_run.md).

## Limitations

- Small synthetic data (33 records, 230 fine-tuning examples): results show mechanisms, not
  production performance.
- CPU inference takes 20 to 60 s per answer.
- The verifier checks that citations are real and verbatim, not that every sentence of the
  answer is supported.
- The relevance threshold (0.75) is calibrated on 14 synthetic questions.
- Embedded Qdrant is single-process; tickets added through the API are not kept by `seed`.
- Injection flags are regexes and easy to evade; security does not rely on them.
- API keys are static demo values.

## More

[docs/INCIDENT.md](docs/INCIDENT.md) (debugging write-up) ·
[AI_DISCLOSURE.md](AI_DISCLOSURE.md) (coding assistant and reused code) ·
MIT [LICENSE](LICENSE)
