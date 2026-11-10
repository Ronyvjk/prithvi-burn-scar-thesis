"""Create fixed train/val/test splits from HLS Burn Scars.

TODO: not implemented yet. This is a planned step in the thesis pipeline.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Create fixed train/val/test splits from HLS Burn Scars.")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")

    args = p.parse_args()
    raise NotImplementedError("01_prepare_dataset.py is a planned stub")


if __name__ == "__main__":
    main()
