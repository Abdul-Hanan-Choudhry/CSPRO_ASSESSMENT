# Running the real LoRA fine-tune on Colab (T4)

The local machine (MX130, 2 GB VRAM) cannot train. These steps run the full training and
evaluation on a free Colab T4 and produce the result files the README reports. Expect roughly
15-30 minutes in total; actual times are recorded in the output files.

## 1. Runtime

Colab → Runtime → Change runtime type → **T4 GPU**. Then, in a code cell:

```bash
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

## 2. Get the code

Either upload a zip of this repository and unzip it, or clone it:

```bash
!git clone <your-repo-url> support-rag
%cd support-rag
```

## 3. Install pinned training dependencies

```bash
!pip install -q -r requirements-train.txt
```

`Qwen/Qwen2.5-0.5B-Instruct` is ungated, so no Hugging Face token is needed.

## 4. (Optional) regenerate the dataset

The committed `finetune/data/*.jsonl` are already generated. Regenerating with the same seed must
reproduce the same hashes as `finetune/data/manifest.json`:

```bash
!python finetune/make_dataset.py --seed 42 && git diff --stat finetune/data
```

## 5. Baseline evaluation (before training)

```bash
!python finetune/eval.py --backend hf --tag baseline
```

## 6. Train

```bash
!python finetune/train.py --mode full
```

Writes `finetune/outputs/qwen2.5-0.5b-support-lora/adapter/` and `finetune/results/train_full.json`.

## 7. Adapted evaluation

```bash
!python finetune/eval.py --backend hf --tag adapted --adapter finetune/outputs/qwen2.5-0.5b-support-lora/adapter
```

## 8. Render the table and download the outputs

```bash
!python finetune/report.py
!zip -r ft_outputs.zip finetune/results finetune/outputs/qwen2.5-0.5b-support-lora/adapter
```

Download `ft_outputs.zip`, unzip it into the repository root locally, and commit
`finetune/results/*.json` (the adapter stays out of git; it is reproducible from these steps).
Then paste the output of `python finetune/report.py` into the README results section.

Only numbers produced by these files go into the README.
