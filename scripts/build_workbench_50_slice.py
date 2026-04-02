#!/usr/bin/env python3
"""
Build a reproducible 50-task WorkBench slice: 10 tasks per single-domain file
(email, calendar, analytics, project_management, customer_relationship_manager).

Output:
  benchmarks/workbench_50_queries.csv — same columns as upstream (query, answer, …)
  benchmarks/workbench_50_manifest.json — provenance (domain + 0-based row index)

Requires vendor/WorkBench (run scripts/setup_workbench.py first).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

# Upstream filenames under data/processed/queries_and_answers/
_DOMAIN_FILES: list[tuple[str, str]] = [
    ("email", "email_queries_and_answers.csv"),
    ("calendar", "calendar_queries_and_answers.csv"),
    ("analytics", "analytics_queries_and_answers.csv"),
    ("project_management", "project_management_queries_and_answers.csv"),
    ("customer_relationship_manager", "customer_relationship_manager_queries_and_answers.csv"),
]

_PER_DOMAIN = 10
_SEED = 42
_FIELDNAMES = ["query", "answer", "base_template", "chosen_template", "domains"]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    wb_qa = root / "vendor" / "WorkBench" / "data" / "processed" / "queries_and_answers"
    if not wb_qa.is_dir():
        print(f"ERROR: {wb_qa} not found. Run: python scripts/setup_workbench.py", file=sys.stderr)
        sys.exit(1)

    out_dir = root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "workbench_50_queries.csv"
    out_manifest = out_dir / "workbench_50_manifest.json"

    rng = random.Random(_SEED)
    combined: list[dict[str, str]] = []
    manifest: list[dict[str, str | int]] = []

    for domain, filename in _DOMAIN_FILES:
        path = wb_qa / filename
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f"ERROR: empty {path}", file=sys.stderr)
            sys.exit(1)
        n = min(_PER_DOMAIN, len(rows))
        indices = sorted(rng.sample(range(len(rows)), n))
        for i in indices:
            row = rows[i]
            combined.append({k: row[k] for k in _FIELDNAMES})
            manifest.append({"domain": domain, "source_file": filename, "row_index": i})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(combined)

    meta = {
        "seed": _SEED,
        "per_domain": _PER_DOMAIN,
        "total_tasks": len(combined),
        "tasks": manifest,
    }
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_csv} ({len(combined)} rows)")
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
