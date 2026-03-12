# Dynamic Architecture Router for Agentic Systems

A predictive meta-agent that routes incoming tasks to the optimal architectural topology (Single-Agent, Centralized MAS, or Decentralized MAS) to minimize coordination tax while maximizing accuracy.

## Overview

As agentic systems scale from isolated experiments to production, the assumption that "more agents are always better" becomes computationally flawed. Quantitative scaling principles show:

- **Centralized coordination** can improve performance by ~80.8% on parallelizable tasks
- **Multi-agent variants** degrade performance by 39–70% on tasks requiring sequential constraint satisfaction
- **Tool-heavy tasks** suffer from a severe coordination trade-off (β = -0.267)

This project implements a **Dynamic Architecture Router** that:

1. Analyzes incoming tasks (sequential depth, parallelization factor, tool count)
2. Routes to the optimal topology using literature-derived thresholds
3. Executes via Single-Agent System (SAS), Centralized MAS, or Decentralized MAS

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Dynamic Router (Meta-Agent)      │
                    │  • Extracts metadata from prompt         │
                    │  • Applies quantitative thresholds       │
                    └───────────────┬─────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌────────────────┐      ┌──────────────────┐
    │ Single-Agent  │      │ Centralized    │      │ Decentralized     │
    │ System (SAS)  │      │ MAS            │      │ MAS              │
    │ O(k)          │      │ Hub-and-spoke  │      │ Peer-to-peer     │
    └───────────────┘      └────────────────┘      └──────────────────┘
```

### Routing Logic

| Condition | Route to | Rationale |
|-----------|----------|-----------|
| `tools ≥ 12` or `depth > 5` | **SAS** | Tool-coordination trade-off; avoid multi-agent tax |
| `parallelism > 0.6` and `tools < 12` | **Centralized MAS** | Decompose orthogonal sub-goals safely |
| Default | **Decentralized MAS** | Open-ended tasks; multi-perspective consensus |

## Project Structure

```
DynamicRoutingAgents/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_examples.py              # Example execution script
├── evaluate_regret.py           # Oracle Evaluation Harness (Routing Regret)
├── scripts/
│   └── setup_pcab.py            # Initialize PCAB benchmark database
└── src/
    └── dynamic_routing/
        ├── __init__.py
        ├── state.py             # RouterState, CentralizedState schemas
        ├── router.py            # Main Dynamic Router graph
        ├── pcab.py               # PCAB database schema & agent tools
        └── centralized_mas.py   # Hub-and-spoke MAS (PCAB-backed)
```

## Installation

```bash
# From project root
pip install -r requirements.txt

# Or install in editable mode
pip install -e .

# Initialize the PCAB benchmark database (required for Centralized MAS)
python scripts/setup_pcab.py
```

## Usage

### Run Example Cases

```bash
python run_examples.py
```

This runs four test cases demonstrating routing to each topology.

### Run Oracle Evaluation Harness (Routing Regret)

```bash
python evaluate_regret.py
```

Demonstrates how Routing Regret is calculated vs. the Oracle Baseline across SAS, Centralized MAS, and Decentralized MAS.

### Run Oracle Evaluation Harness (Routing Regret)

```bash
python evaluate_regret.py
```

Demonstrates how Routing Regret is calculated vs. the Oracle Baseline across SAS, Centralized MAS, and Decentralized MAS.

### Programmatic Usage

```python
from dynamic_routing.router import app

result = app.invoke({"user_query": "Please aggregate my Calendar and Maps data simultaneously."})
print(result["selected_architecture"])  # Centralized MAS
print(result["final_response"])
```

### Routing by Query Patterns

The skeleton uses keyword-based metadata extraction. Example behaviors:

- **"simultaneously" / "aggregate"** → High parallelism (0.9) → Centralized MAS
- **"step-by-step" / "complex workflow"** → Deep sequential (depth 8, tools 14) → SAS
- **Default** → Moderate parallelism → Decentralized MAS

In production, replace the mock logic in `dynamic_router_node` with an LLM call (e.g., Mistral 7B) enforcing structured JSON output.

## Centralized MAS: Personalized Context Assembly Benchmark (PCAB)

The Centralized MAS implements a hub-and-spoke topology backed by the PCAB SQLite database:

- **Supervisor**: Delegates to workers and synthesizes outputs; acts as validation bottleneck
- **Workers**: Calendar, Drive, Commute (Maps), Contacts agents query `pcab_environment.db`; cannot communicate directly
- **State**: `Annotated[list, operator.add]` for `aggregated_context`—workers append results without passing full history

Example task: *"I need to meet with Dr. Hong Man on March 16th after my Deep Learning lecture. Where should I go, and how long will it take to walk there?"* → Supervisor delegates to Contacts → Calendar → Commute → synthesizes personalized context with gold trajectory API calls.

## Technology Stack (Proposal)

- **Orchestration**: LangGraph with `StateGraph`
- **AI Core (Local)**: Quantized Mistral 7B (routing), Llama 3 8B (execution), vLLM on V100 GPUs
- **API Fallbacks**: Google Gemini 1.5 Flash / OpenAI GPT-4o-mini via LangChain

## Evaluation (Proposal)

- **Environment A**: WorkBench dataset (30 stratified tasks) — validate SAS routing for tool-heavy sequential tasks
- **Environment B**: Personalized Context Assembly Benchmark (PCAB) — validate Centralized MAS for parallelizable aggregation with defined query families (Sequential, Parallel, Exploratory) and gold trajectories

## Metrics (Proposal)

- **Routing Regret (vs. Oracle Baseline)** — primary optimization objective; quantifies penalty for suboptimal routing
- Task Success Rate (accuracy)
- Total Token Efficiency
- Per-step Latency & End-to-end Wall-clock Time
- Coordination Overhead (% tokens on inter-agent communication)
- Error Amplification Rate (A_e)

## References

[1] Quantitative scaling principles for multi-agent systems; coordination tax on sequential vs. parallelizable tasks.
