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
└── src/
    └── dynamic_routing/
        ├── __init__.py
        ├── state.py             # RouterState, CentralizedState schemas
        ├── router.py            # Main Dynamic Router graph
        └── centralized_mas.py  # Hub-and-spoke MAS with mock APIs
```

## Installation

```bash
# From project root
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

## Usage

### Run Example Cases

```bash
python run_examples.py
```

This runs four test cases demonstrating routing to each topology.

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

## Centralized MAS: Personalized Context Sandbox

The Centralized MAS implements a hub-and-spoke topology with mock APIs:

- **Supervisor**: Delegates to workers and synthesizes outputs; acts as validation bottleneck
- **Workers**: Calendar, Maps, Drive agents (cannot communicate directly)
- **State**: `Annotated[list, operator.add]` for `aggregated_context`—workers append results without passing full history

Example task: *"Cross-reference my calendar with maps for upstate NY hiking and check my drive notes."* → Supervisor delegates to Calendar → Maps → Drive → synthesizes personalized context.

## Technology Stack (Proposal)

- **Orchestration**: LangGraph with `StateGraph`
- **AI Core (Local)**: Quantized Mistral 7B (routing), Llama 3 8B (execution), vLLM on V100 GPUs
- **API Fallbacks**: Google Gemini 1.5 Flash / OpenAI GPT-4o-mini via LangChain

## Evaluation (Proposal)

- **Environment A**: WorkBench dataset (30 stratified tasks) — validate SAS routing for tool-heavy sequential tasks
- **Environment B**: Personalized Context Sandbox — validate Centralized MAS for parallelizable aggregation

## Metrics (Proposal)

- Task Success Rate (accuracy)
- Total Token Efficiency
- Coordination Overhead (% tokens on inter-agent communication)
- Error Amplification Rate (A_e)

## References

[1] Quantitative scaling principles for multi-agent systems; coordination tax on sequential vs. parallelizable tasks.
