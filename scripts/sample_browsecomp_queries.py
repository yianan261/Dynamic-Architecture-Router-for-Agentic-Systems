#!/usr/bin/env python3
"""Randomly sample N lines from a BrowseComp-style JSONL (queries or corpus)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("-n", "--num", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.input_jsonl.is_file():
        print(f"ERROR: {args.input_jsonl}", file=sys.stderr)
        sys.exit(1)
    lines = [ln.strip() for ln in args.input_jsonl.open(encoding="utf-8") if ln.strip()]
    rng = random.Random(args.seed)
    k = min(args.num, len(lines))
    chosen = rng.sample(lines, k=k) if k < len(lines) else lines
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for ln in chosen:
            out.write(ln + "\n")
    print(f"Sampled {len(chosen)} lines -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
