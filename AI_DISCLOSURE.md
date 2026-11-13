# AI assistance and reused code

## Coding assistant

This repository was built with an AI coding assistant (Claude Code, Anthropic) used as a pair
programmer. It was used to:

- draft modules, tests, scripts and documentation from a written specification I provided
  (architecture, security model, incident scenario, evaluation metrics);
- verify library behaviour before relying on it (Qdrant embedded mode: sparse vectors, RRF via
  `query_points`/`prefetch`, collection aliases);
- run the test suite, linters, the incident reproduction and the local evaluation, and fix the
  failures they exposed.

All design decisions, the scenario and the acceptance criteria are mine. I reviewed every file
and can explain and modify it; `docs/DEFENSE.md` is my walkthrough of each module. Every number
reported in the README comes from a result file produced by a script in this repository.

## Reused code and libraries

No code was copied from other projects. The implementation uses these open-source libraries
through their public APIs (exact versions in `requirements.txt` and `requirements-train.txt`):

| Library | Use |
|---|---|
| FastAPI, Starlette, Uvicorn | HTTP API |
| Pydantic, pydantic-settings | schemas, configuration |
| qdrant-client | vector store (embedded mode) |
| fastembed | `BAAI/bge-small-en-v1.5` dense and `Qdrant/bm25` sparse embeddings |
| httpx | Ollama client |
| pytest, ruff | tests, lint/format |
| transformers, peft, accelerate, PyTorch | LoRA fine-tuning and evaluation |

Models (open weights): Llama 3.2 3B Instruct via Ollama (inference), Qwen2.5-0.5B-Instruct
(fine-tuning target), SmolLM2-135M-Instruct (CPU smoke test only).

Reciprocal Rank Fusion follows Cormack, Clarke and Büttcher (SIGIR 2009), with the common
constant k = 60.

## Data

All tickets, logs, KB articles, customers, invoices, keys and identifiers are invented. No
employer data, customer data, real credentials or tokens are included.
