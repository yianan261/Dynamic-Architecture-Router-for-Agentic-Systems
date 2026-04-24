#!/usr/bin/env python3
"""
Train sklearn multinomial logistic regression on ``export_router_training_rows`` CSV.

Writes joblib bundle: ``{model, feature_names, classes}`` for ``ROUTER_LEARNED_MODEL_PATH``.

Requires: pip install '.[router-ml]'  (scikit-learn + joblib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    try:
        import joblib
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("ERROR: install router-ml extras: pip install -e '.[router-ml]'", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("training_csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=project_root / "models" / "router_policy.joblib")
    args = ap.parse_args()
    df = pd.read_csv(args.training_csv)
    feat_cols = [
        "estimated_sequential_depth",
        "parallelization_factor",
        "estimated_tool_count",
        "num_subgoals",
        "entity_count",
        "constraint_tightness",
        "open_endedness",
        "aggregation_required",
        "expected_retrieval_fanout",
        "domain_span",
        "expected_context_expansion",
        "final_synthesis_complexity",
        "cross_branch_dependency",
        "communication_load_estimate",
    ]
    for c in feat_cols:
        if c not in df.columns:
            print(f"ERROR: missing column {c}", file=sys.stderr)
            sys.exit(1)
    X = df[feat_cols].astype(float).values
    y = df["oracle_architecture"].astype(str).values
    classes = sorted(set(y.tolist()))
    clf = LogisticRegression(
        max_iter=200,
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
    )
    if len(set(y.tolist())) < 2:
        print("ERROR: need at least 2 distinct oracle_architecture values in CSV", file=sys.stderr)
        sys.exit(1)
    clf.fit(X, y)
    fitted_classes = list(getattr(clf, "classes_", classes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": feat_cols, "classes": fitted_classes}, args.output)
    print(f"Wrote {args.output.resolve()} classes={fitted_classes}")


if __name__ == "__main__":
    main()
