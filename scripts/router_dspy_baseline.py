#!/usr/bin/env python3
"""
Optional DSPy prompt-optimization baseline for the router (not required for core path).

Install: ``pip install dspy`` and set API keys per DSPy docs, then extend this
skeleton to optimize a small signature that maps ``user_query`` → architecture.

This script only checks import and prints next steps if ``dspy`` is missing.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import dspy  # noqa: F401

        print(
            "DSPy is available. Add a DSPy module + optimizer targeting your routing metric, "
            "then compare against threshold routing and ROUTER_LEARNED_MODEL_PATH."
        )
    except ImportError:
        print(
            "DSPy not installed. For a prompt-optimization baseline: pip install dspy\n"
            "See https://dspy.ai/ — keep the hybrid sklearn router as the primary learned policy."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
