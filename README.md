# Quantise or Downsize?

MSc thesis — Erasmus Mundus Geospatial Technologies (IFGI, University of Münster).

**Question:** At a similar weight-storage budget, is it better to quantise a large
geospatial foundation model (Prithvi-EO-2.0-600M → INT8, optionally + LoRA recovery)
or to use a smaller model at higher precision (Prithvi-EO-2.0-300M, FP16)?

**Task:** Wildfire burn-scar segmentation on the HLS Burn Scars dataset.

## Model variants

| # | Variant | Purpose |
|---|---------|---------|
| 1 | 600M FP16 | Baseline |
| 2 | 600M INT8 (post-training quantisation) | Effect of quantisation |
| 3 | 600M INT8 + LoRA | Accuracy recovery |
| 4 | 300M FP16 | Downsize alternative |

All variants: same data split, same test set, 3 seeds each.

## Pipeline

Scripts are numbered in the order they run. Each step only uses outputs of earlier steps.

1. `scripts/01_prepare_dataset.py` — build fixed train/val/test splits
2. `scripts/02_finetune.py --model 600m|300m` — fine-tune in FP16
3. `scripts/03_quantize_int8.py` — INT8 PTQ of the fine-tuned 600M model
4. `scripts/04_train_lora_int8.py` — LoRA adapters on the INT8 model
5. `scripts/05_evaluate.py --variant ...` — appends one row to `results/results.csv`

## Metrics

mIoU, F1 (burn class), model size on disk (MB), peak GPU memory during inference (MB).

## Large files

`data/` and `models/` contents are **not** tracked by Git (see `.gitignore`).
Record where they actually live here:

- HLS Burn Scars: `TODO path / source`
- Pretrained checkpoints: `TODO path / Hugging Face IDs`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
