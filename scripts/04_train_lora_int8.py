"""Train LoRA adapters on the INT8 600M model to recover accuracy.

TODO: not implemented yet. This is a planned step in the thesis pipeline.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Train LoRA adapters on the INT8 600M model to recover accuracy.")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    raise NotImplementedError("04_train_lora_int8.py is a planned stub")


if __name__ == "__main__":
    main()
