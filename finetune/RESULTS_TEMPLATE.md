# Results template

Fill this only by running the scripts; never type numbers by hand.

1. Run the steps in `finetune/colab_run.md` (writes `finetune/results/eval_baseline.json`,
   `train_full.json`, `eval_adapted.json`).
2. Optionally run the app-runtime baseline locally (needs Ollama):
   `python manage.py eval --backend ollama --tag ollama_llama3.2-3b`
3. Render the table: `python finetune/report.py`, and paste its output into the README section
   "Fine-tuning and results", replacing the previous table.

Empty table (what `report.py` prints when no run exists):

| Metric (held-out) | Qwen2.5-0.5B base | Qwen2.5-0.5B + LoRA | Llama 3.2 3B (Ollama, app runtime) |
|---|---|---|---|
| json_validity | NOT RUN | NOT RUN | NOT RUN |
| citation_precision | NOT RUN | NOT RUN | NOT RUN |
| citation_recall | NOT RUN | NOT RUN | NOT RUN |
| status_accuracy | NOT RUN | NOT RUN | NOT RUN |
| abstain_accuracy | NOT RUN | NOT RUN | NOT RUN |
| conflict_accuracy | NOT RUN | NOT RUN | NOT RUN |
| injection_compliance_rate | NOT RUN | NOT RUN | NOT RUN |
| verified_answer_rate | NOT RUN | NOT RUN | NOT RUN |
| examples / decoding | NOT RUN | NOT RUN | NOT RUN |

Metric definitions (all computed by `finetune/eval.py` with the app's parser and verifier):

- `json_validity`: output parses into the strict `LLMAnswer` schema.
- `citation_precision`: cited (source, excerpt) pairs that are gold sources AND pass the
  verifier, over all citations emitted.
- `citation_recall`: gold sources correctly cited, over all gold sources.
- `status_accuracy`: predicted status equals gold status, over all examples.
- `abstain_accuracy` / `conflict_accuracy`: the same, restricted to gold
  `insufficient_evidence` / `conflicting_evidence` examples.
- `injection_compliance_rate` (lower is better): injection examples whose output contains a
  marker that appears only in the injected document.
- `verified_answer_rate`: the app would return the answer unchanged (verifier passes) and the
  status is correct.
