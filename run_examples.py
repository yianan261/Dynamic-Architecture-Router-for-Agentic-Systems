#!/usr/bin/env python3
"""
Example execution of the Dynamic Architecture Router.

Run from project root:
    python run_examples.py

Without vLLM: Uses keyword fallback (set USE_LLM_ROUTER=false to suppress warnings).
With vLLM: Start server with:
    python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
"""

import os
import sys

# Use keyword fallback for demos when vLLM is not running
os.environ.setdefault("USE_LLM_ROUTER", "false")

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

    # Test Case 5: PCAB task by ID (data-driven, no hard-coded conditions)
    print("\n--- Test 5: PCAB Task by ID (Data-Driven Extraction) ---")
    from dynamic_routing.pcab_tasks import get_pcab_task

    pcab_task = get_pcab_task("PCAB-Par-01")
    if pcab_task:
        test_5 = {
            "user_query": pcab_task.description,
            "extraction_overrides": pcab_task.extraction_params.as_override_dict(),
            "required_tools": pcab_task.required_tools,
        }
        result_5 = app.invoke(test_5)
        print(f"Task ID: {pcab_task.id}")
        print(f"Query: {pcab_task.description}")
        print(f"Routed To: {result_5['selected_architecture']}")
        print(f"Response: {result_5['final_response']}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
