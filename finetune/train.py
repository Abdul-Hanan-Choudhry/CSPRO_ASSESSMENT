"""LoRA supervised fine-tuning to emit the app's cited-JSON contract.

    python finetune/train.py --mode full    # Colab/Kaggle T4 (see finetune/colab_run.md)
    python finetune/train.py --mode smoke   # CPU pipeline check with a 135M model, 4 steps

Loss is computed on the assistant answer only (prompt tokens are masked with -100), so the
model learns the output contract rather than to reproduce retrieved documents.
Writes the adapter to <output_dir>/adapter and finetune/results/train_<mode>.json.
"""

import argparse
import time
from pathlib import Path

import torch
from common import ROOT, load_config, load_jsonl, run_metadata, seed_everything, write_result
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

IGNORE = -100


def encode(example: dict, tokenizer, max_len: int) -> dict | None:
    """Tokenise prompt + answer with the model's chat template; mask the prompt in labels."""
    prompt = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        [*example["messages"], {"role": "assistant", "content": example["target"]}],
        tokenize=False,
    )
    # The mask below is only correct if the template renders the prompt as a strict prefix.
    assert full.startswith(prompt), "chat template does not render prompt as a prefix"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt) :], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None  # counted and reported; never silently truncate the answer
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [IGNORE] * len(prompt_ids) + answer_ids,
    }


def make_collator(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return collate


def encode_split(rows: list[dict], tokenizer, max_len: int) -> tuple[list[dict], int]:
    encoded = [encode(r, tokenizer, max_len) for r in rows]
    kept = [e for e in encoded if e is not None]
    return kept, len(encoded) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "smoke"], default="smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "finetune" / "config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, args.mode)
    seed_everything(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train, skipped_train = encode_split(
        load_jsonl(cfg["data"]["train"], cfg["n_train"]), tokenizer, cfg["max_seq_len"]
    )
    heldout, skipped_eval = encode_split(
        load_jsonl(cfg["data"]["heldout"], cfg["n_eval"]), tokenizer, cfg["max_seq_len"]
    )
    print(f"train={len(train)} (skipped {skipped_train} too long), heldout={len(heldout)}")

    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
    lora = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()

    output_dir = ROOT / cfg["output_dir"]
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=heldout,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    eval_before = trainer.evaluate()["eval_loss"]
    started = time.time()
    trainer.train()
    runtime = time.time() - started
    eval_after = trainer.evaluate()["eval_loss"]

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    path = write_result(
        f"train_{args.mode}",
        {
            "mode": args.mode,
            "note": "smoke = pipeline check only, not a quality result"
            if args.mode == "smoke"
            else "",
            "base_model": cfg["base_model"],
            "adapter_dir": str(adapter_dir.relative_to(ROOT)),
            "config": cfg,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "trainable_params": trainable,
            "total_params": total,
            "n_train": len(train),
            "n_heldout": len(heldout),
            "skipped_too_long": {"train": skipped_train, "heldout": skipped_eval},
            "heldout_loss_before": eval_before,
            "heldout_loss_after": eval_after,
            "train_runtime_s": round(runtime, 1),
            "log_history": trainer.state.log_history,
            **run_metadata(),
        },
    )
    print(f"heldout loss {eval_before:.4f} -> {eval_after:.4f}; wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
