"""Evaluate one variant on the test set and append a row to results/results.csv.

TODO: not implemented yet. This is a planned step in the thesis pipeline.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Evaluate one variant on the test set and append a row to results/results.csv.")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--variant", choices=["600m_fp16", "600m_int8", "600m_int8_lora", "300m_fp16"], required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    raise NotImplementedError("05_evaluate.py is a planned stub")


if __name__ == "__main__":
    main()
