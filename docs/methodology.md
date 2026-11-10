# Methodology notes

Record every decision here with a date and a reason.

- Baseline precision: FP16 (decide: keep FP32 reference? why / why not)
- INT8 method: TODO (e.g. bitsandbytes LLM.int8 / PyTorch PTQ) — check whether it
  actually lowers peak GPU memory or only file size
- LoRA: rank TODO, target modules TODO
- Fairness rule: 300M and 600M get identical data, epochs and tuning budget
