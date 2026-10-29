"""Fine-tune Prithvi-EO-2.0 (600M or 300M) in FP16 for burn-scar segmentation.

TODO: not implemented yet. This is a planned step in the thesis pipeline.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Fine-tune Prithvi-EO-2.0 (600M or 300M) in FP16 for burn-scar segmentation.")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--model", choices=["600m", "300m"], required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    raise NotImplementedError("02_finetune.py is a planned stub")


if __name__ == "__main__":
    main()
