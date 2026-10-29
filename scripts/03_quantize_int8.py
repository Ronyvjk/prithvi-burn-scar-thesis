"""Apply INT8 post-training quantisation to the fine-tuned 600M model.

TODO: not implemented yet. This is a planned step in the thesis pipeline.
"""
import argparse


def main():
    p = argparse.ArgumentParser(description="Apply INT8 post-training quantisation to the fine-tuned 600M model.")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")

    args = p.parse_args()
    raise NotImplementedError("03_quantize_int8.py is a planned stub")


if __name__ == "__main__":
    main()
