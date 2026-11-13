# Support RAG

A local support bot that answers from synthetic tickets, logs and KB articles with **verified
citations**, abstains when evidence is missing, reports conflicting sources, and never puts
another customer's data in the prompt. It also reproduces and fixes a reindex incident (wrong
citations from a stale alias, mislabelled index and bad BM25/vector score fusion).

- **Model:** Llama 3.2 3B Instruct (open weights), run locally by Ollama on CPU
- **Stack:** Python 3.11+, FastAPI, Pydantic v2, Qdrant (embedded), fastembed (bge-small + BM25)
- **Fine-tuning:** LoRA on Qwen2.5-0.5B-Instruct, evaluated with the app's own citation verifier

How every part works, step by step: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).
Demo video: [docs/demo/demo.webm](docs/demo/demo.webm).

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

Re-record the demo video (optional, needs the API running and Playwright):
`pip install playwright==1.65.1; playwright install chromium; python scripts/record_demo.py`.
Captions in the video are generated from the results on screen.

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
verbatim; answers naming foreign IDs are withheld; injection patterns are flagged (logged only).
Details: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md), incident: [docs/INCIDENT.md](docs/INCIDENT.md).

## Fine-tuning results

From `python finetune/report.py` (files in `finetune/results/`). Held-out set: 50 examples on
topics never seen in training.

| Metric | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | Llama 3.2 3B (Ollama, app runtime) |
|---|---|---|---|
| json_validity | NOT RUN | NOT RUN | 1.00 |
| citation_precision | NOT RUN | NOT RUN | 0.65 |
| citation_recall | NOT RUN | NOT RUN | 0.91 |
| status_accuracy | NOT RUN | NOT RUN | 0.76 |
| abstain_accuracy | NOT RUN | NOT RUN | 1.00 |
| conflict_accuracy | NOT RUN | NOT RUN | 0.70 |
| injection_compliance_rate (lower is better) | NOT RUN | NOT RUN | 0.00 |
| verified_answer_rate | NOT RUN | NOT RUN | 0.74 |

- **Run:** Llama 3.2 3B baseline on all 50 examples (CPU, schema-constrained like the app).
  Weakest category: stale evidence, 1/10 (it calls superseded-vs-current a conflict). In the app,
  retrieval drops the superseded article first.
- **Run:** CPU smoke test of the training pipeline (135M model, 4 steps). Proves the code runs;
  not a quality result.
- **Not run:** Qwen baseline, LoRA training and adapted evaluation need a GPU. Steps:
  [finetune/colab_run.md](finetune/colab_run.md). The columns stay NOT RUN until those files exist.

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

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) (workflow and features) ·
[docs/INCIDENT.md](docs/INCIDENT.md) (debugging write-up) ·
[docs/DEFENSE.md](docs/DEFENSE.md) (module notes, interview Q&A) ·
[AI_DISCLOSURE.md](AI_DISCLOSURE.md) · MIT [LICENSE](LICENSE)
