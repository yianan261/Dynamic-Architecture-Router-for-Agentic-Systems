#!/usr/bin/env python3
"""
Example execution of the Dynamic Architecture Router.

Run from project root:
    python run_examples.py

Or with the package on PYTHONPATH:
    python -m run_examples
"""

import sys
from pathlib import Path

# Ensure src is on the path when running from project root
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from dynamic_routing.router import app


def main() -> None:
    print("=" * 60)
    print("Dynamic Architecture Router — Example Execution")
    print("=" * 60)

    # Test Case 1: High Parallelization (Context Aggregation)
    print("\n--- Test 1: High Parallelization (Context Aggregation) ---")
    test_1 = {"user_query": "Please aggregate my Calendar and Maps data simultaneously."}
    result_1 = app.invoke(test_1)
    print(f"Query: {test_1['user_query']}")
    print(f"Routed To: {result_1['selected_architecture']}")
    print(f"Response: {result_1['final_response']}")

    # Test Case 2: Deep Sequential / Tool-Heavy
    print("\n--- Test 2: Deep Sequential / Tool-Heavy ---")
    test_2 = {
        "user_query": "Execute this complex workflow step-by-step using all available tools."
    }
    result_2 = app.invoke(test_2)
    print(f"Query: {test_2['user_query']}")
    print(f"Routed To: {result_2['selected_architecture']}")
    print(f"Response: {result_2['final_response']}")

    # Test Case 3: Open-Ended Exploration (Decentralized)
    print("\n--- Test 3: Open-Ended Exploration ---")
    test_3 = {"user_query": "What are the pros and cons of different travel options?"}
    result_3 = app.invoke(test_3)
    print(f"Query: {test_3['user_query']}")
    print(f"Routed To: {result_3['selected_architecture']}")
    print(f"Response: {result_3['final_response']}")

    # Test Case 4: Centralized MAS with PCAB (Calendar + Drive + Commute)
    print("\n--- Test 4: Centralized MAS (Personalized Context Assembly Benchmark) ---")
    test_4 = {
        "user_query": "Cross-reference my calendar availability with maps for "
        "upstate NY hiking and check my drive notes."
    }
    result_4 = app.invoke(test_4)
    print(f"Query: {test_4['user_query']}")
    print(f"Routed To: {result_4['selected_architecture']}")
    print(f"Response: {result_4['final_response']}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
