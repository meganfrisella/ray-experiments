import argparse
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        type=int,
    )
    parser.add_argument(
        "--num-microbatches",
        type=int,
    )
    parser.add_argument(
        "--num-iters",
        type=int,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    args = vars(args)
    return args

