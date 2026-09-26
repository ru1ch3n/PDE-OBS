"""Portable entry point for all six CPU-first release examples.

Examples (run each in an independent CLI invocation)::

    python examples/release_workflows.py E1 --data-root ./demo-data --output-root ./demo-runs
    pdeobs demo --example E5 --data-root ./demo-data --output-root ./demo-runs --threads 1

E5 needs the train extra. No example requires a full dataset or a GPU.
Existing per-example output directories are never overwritten.
"""
from __future__ import annotations

import argparse

from pdeobs.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=[f"E{j}" for j in range(1, 7)])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--threads", choices=(1, 2), type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(main(["demo", "--example", args.example, "--data-root", args.data_root,
                          "--output-root", args.output_root, "--threads", str(args.threads)]))
