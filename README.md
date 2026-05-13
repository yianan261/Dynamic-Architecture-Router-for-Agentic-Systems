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
    │ System (SAS)  │      │ MAS            │      │ MAS               │
    │ ReAct loop    │      │ Hub-and-spoke  │      │ Parallel peers    │
    └───────────────┘      └────────────────┘      └──────────────────┘
```

### Routing Logic

| Condition | Route to | Rationale |
|-----------|----------|-----------|
| `tools ≥ 12` or `depth > 5` | **SAS** | Tool-coordination trade-off; avoid multi-agent tax |
| `parallelism > 0.85` and `tools < 12` | **Decentralized MAS** | Very parallel profile: peer batch + merge (no supervisor loop) |
| `parallelism > 0.6` and `tools < 12` | **Centralized MAS** | Moderately parallel: hub-and-spoke orchestration |
| Default | **SAS** | Matched-compute baseline when structure is ambiguous |

## Current Status

| Component | Status |
|-----------|--------|
| **Implemented** | Router graph (LangGraph), SAS, CMAS, **DMAS** (parallel PCAB peers + consensus; WorkBench fixed-order peer pass + merge), PCAB / WorkBench harnesses, BrowseComp-Plus and Fin-RATE fixed-corpus runners, regret oracle |
| **Partial** | vLLM routing — `predict_routing_metadata` + keyword fallback (`USE_LLM_ROUTER=false`). Optional **learned router**: train with `scripts/train_router_from_regret.py`, set `ROUTER_LEARNED_MODEL_PATH`. |
| **Benchmarks** | WorkBench 50-task pilot; BrowseComp-Plus and Fin-RATE 30-task fixed-corpus pilots with v1 deterministic/offline architectures and v2.0 LLM-backed SAS/CMAS/DMAS agents. |

## Project Structure

```
DynamicRoutingAgents/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_examples.py              # Example execution script
├── run_benchmark_sweep.py       # PCAB sweep: SAS + CMAS + DMAS
├── run_browsecomp_sweep.py      # BrowseComp-style local corpus + three architectures
├── run_finrate_sweep.py         # Fin-RATE-style QA + corpus (fixture or upstream paths)
├── evaluate_regret.py           # Oracle Evaluation Harness (v2, accuracy-based regret)
├── scripts/
│   ├── analyze_llm_fixed_corpus_results.py # v2.0 aggregate oracle/regret tables
│   └── setup_pcab.py            # Initialize PCAB benchmark database
└── src/
    └── dynamic_routing/
        ├── __init__.py
        ├── state.py             # RouterState, CentralizedState, SingleAgentState
        ├── router.py            # Main Dynamic Router graph
        ├── pcab.py              # PCAB database schema & agent tools
        ├── pcab_tasks.py        # PCAB task registry (gold schemas, extraction params)
        ├── agent_tools.py       # LangChain @tool bindings for Llama (JSON schema)
        ├── single_agent.py      # SAS: ReAct loop, unified memory
        ├── centralized_mas.py   # Hub-and-spoke MAS (PCAB-backed)
        ├── decentralized_mas.py # Parallel peers + merge (PCAB / router)
        ├── browsecomp_env.py    # Local corpus helper for BrowseComp-style runs
        ├── browsecomp_runner.py # SAS / CMAS / DMAS + hybrid judge hooks
        ├── finrate_runner.py    # Fin-RATE-style retrieval + three architectures
        ├── router_policy.py     # Optional sklearn policy (`ROUTER_LEARNED_MODEL_PATH`)
        └── vllm_integration.py  # Routing metadata (+ extended tabular features)
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

This runs five test cases demonstrating routing to each topology.

### Run Benchmark Sweep (PCAB End-to-End)

```bash
python scripts/setup_pcab.py   # Ensure database is seeded
python run_benchmark_sweep.py
```

Runs all PCAB tasks through **SAS, CMAS, and DMAS**, records latency and trajectory accuracy, and prints a summary. Rule-based mode (no LLM) is the default for a fast sanity check.

### Run Oracle Evaluation Harness (Routing Regret)

```bash
python run_benchmark_sweep.py   # Generates benchmark_results.json
python evaluate_regret.py      # Loads JSON, computes oracle and regret
```

Loads `benchmark_results.json` from the sweep and computes the Oracle Baseline (composite reward) per task. Regret metrics appear when `router_prediction` is populated (e.g. after running the full router).

### Programmatic Usage

```python
from dynamic_routing.router import app

result = app.invoke({"user_query": "Please aggregate my Calendar and Maps data simultaneously."})
print(result["selected_architecture"])  # Centralized MAS
print(result["final_response"])
```

### Single-Agent System (SAS): ReAct Loop

The SAS uses a ReAct (Reason + Act) loop with unified memory: every thought and tool result is appended to a single message stream. It invokes PCAB tools sequentially (contact → calendar → drive → commute as needed) and is optimal for deep sequential tasks with strict dependencies.

### Routing by Query Patterns

The router calls `predict_routing_metadata` from `vllm_integration`. When vLLM is unavailable (no server running, or `USE_LLM_ROUTER=false`), it falls back to keyword heuristics:

- **"simultaneously" / "aggregate"** or 2+ of (calendar, maps, drive) → High parallelism (0.9) → Centralized MAS
- **"step-by-step" / "complex workflow"** → Deep sequential (depth 8, tools 14) → SAS
- **Default** → Moderate parallelism → SAS (safe baseline)
- **Very high parallelism** (`> 0.85`) → Decentralized MAS (parallel peer batch + merge)

### LLM-powered Routing (vLLM)

The router uses Mistral-7B via vLLM when available. Start the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000
```

Then set `USE_LLM_ROUTER=true` (default when unset). If vLLM is unavailable, the router falls back to keyword heuristics.

## Centralized MAS: Personalized Context Assembly Benchmark (PCAB)

The Centralized MAS implements a hub-and-spoke topology backed by the PCAB SQLite database:

- **Supervisor**: Delegates to workers and synthesizes outputs; acts as validation bottleneck
- **Workers**: Calendar, Drive, Commute (Maps), Contacts agents query `pcab_environment.db`; cannot communicate directly
- **State**: `Annotated[list, operator.add]` for `aggregated_context`—workers append results without passing full history

Example task: *"I need to meet with Dr. Hong Man on March 16th after my Deep Learning lecture. Where should I go, and how long will it take to walk there?"* → Supervisor delegates to Contacts → Calendar → Commute → synthesizes personalized context with gold trajectory API calls.

## Technology Stack

- **Orchestration**: LangGraph with `StateGraph`
- **Worker LLMs**: configurable LangChain chat backends; v2.0 fixed-corpus pilots used the project worker backend defaulting to `gpt-5.4-mini`
- **Routing Core**: vLLM metadata extraction when available, keyword fallback, and optional sklearn learned router
- **API Fallbacks / Judges**: OpenAI or Google-backed semantic judges where configured; local exact/overlap judges remain available for fixtures

## Evaluation

- **Environment A**: WorkBench pilot — validates SAS, CMAS, and DMAS on tool-heavy and sequential tasks
- **Environment B**: BrowseComp-Plus-style fixed corpus — validates deep-research QA with fixed retrieval evidence
- **Environment C**: Fin-RATE-style fixed corpus — validates finance QA, comparison, and synthesis over SEC-filing evidence
- **Legacy/Pilot**: PCAB remains available for personalized context assembly experiments

## Metrics

- **Routing Regret (vs. Oracle Baseline)** — primary optimization objective; quantifies penalty for suboptimal routing
- Task Success Rate (accuracy)
- Total Token Efficiency
- Per-step Latency & End-to-end Wall-clock Time
- Coordination Overhead (% tokens on inter-agent communication)
- Error Amplification Rate (A_e)

## Experiment Versions

### v1: Deterministic fixed-corpus architecture pilots

The original BrowseComp-Plus and Fin-RATE pilots used fixed local corpora and deterministic architecture runners. This version was useful for validating schemas, oracle/regret export, failure diagnosis fields, and router-training plumbing without spending LLM tokens on every architecture.

Key v1 artifacts are retained under `results/browsecomp/`, `results/finrate/`, and `results/router/`. These runs support the older CMAS and DMAS implementation story, including centralized decomposition, decentralized peer comparison, local/semantic judges, and learned-router diagnostics.

### v2.0: LLM-backed fixed-corpus SAS/CMAS/DMAS pilots

The v2.0 update keeps retrieval evidence fixed but runs real LLM-backed agents for all three architectures:

- **SAS**: one ReAct agent with local corpus search.
- **CMAS**: supervisor-mediated workers using direct and alternate evidence queries, followed by LLM synthesis.
- **DMAS**: independent peer agents over the same fixed corpus, followed by LLM consensus synthesis.

This design isolates architecture effects from retrieval drift: BrowseComp-Plus and Fin-RATE agents reason with the same task-specific evidence pool, while accuracy, latency, token use, semantic failures, oracle wins, and regret are measured across SAS, CMAS, and DMAS.

Latest v2.0 30-task fixed-corpus results:

| Benchmark | Architecture | Avg. Accuracy | Oracle Wins | Main Failure Pattern |
|-----------|--------------|--------------:|------------:|----------------------|
| BrowseComp-Plus | SAS | 0.9333 | 25 / 30 | Imperfect or unlabeled misses |
| BrowseComp-Plus | CMAS | 0.9333 | 0 / 30 | Requirement Miss |
| BrowseComp-Plus | DMAS | 0.9333 | 5 / 30 | Requirement Miss; occasional incomplete aggregation |
| Fin-RATE | SAS | 0.6167 | 22 / 30 | Imperfect or unlabeled partial-credit answers |
| Fin-RATE | CMAS | 0.6667 | 0 / 30 | Requirement Miss |
| Fin-RATE | DMAS | 0.7500 | 8 / 30 | Requirement Miss; occasional incomplete aggregation |

Router/regret takeaway:

- BrowseComp-Plus accuracy tied across architectures, so SAS usually won the composite oracle by being cheaper.
- Fin-RATE showed the clearest MAS benefit: DMAS had the highest average accuracy, but SAS still won many oracle cases when partial-credit ties made latency and token cost decisive.
- Learned-router rows in the v2.0 aggregate tables used deterministic router-metadata fallback because the live router metadata backend was unavailable during table generation; interpret those rows as policy-shape diagnostics rather than a freshly optimized router result.

MAST-inspired semantic failure distribution for CMAS and DMAS in the v2.0 30-task pilots:

| Scope | Requirement Miss | Incorrect Synthesis / Incomplete Aggregation |
|-------|-----------------:|----------------------------------------------:|
| BrowseComp-Plus | 3 | 1 |
| Fin-RATE | 28 | 3 |
| Combined | 31 | 4 |

## BrowseComp-Plus–style sweep (local corpus)

Fixture queries and corpus live under `benchmarks/`. For the full [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) benchmark, decrypt queries per upstream docs and point `--corpus` at your indexed JSONL export.

```bash
python run_browsecomp_sweep.py
python run_browsecomp_sweep.py --queries path/to/queries.jsonl --corpus path/to/corpus.jsonl --use-llm-sas
python run_browsecomp_sweep.py --queries path/to/queries.jsonl --corpus path/to/corpus.jsonl --use-llm-all
```

**Hybrid judge:** set `BROWSECOMP_JUDGE_BACKEND=local|llm|auto`. `local` uses normalized exact / contains / overlap scoring. `llm` uses a configured semantic-equivalence judge, preferring `BROWSECOMP_QWEN_URL` when set and otherwise the worker LLM. `auto` runs local first and calls the LLM only for uncertain overlap scores.

Use `--use-llm-all` for the v2.0 fixed-corpus setting where SAS, CMAS, and DMAS all use LLM-backed agents over the same allowed document pool.

## Fin-RATE–style sweep

```bash
python run_finrate_sweep.py
python run_finrate_sweep.py --qa path/to/qa.jsonl --corpus path/to/corpus.jsonl --per-type 40
python run_finrate_sweep.py --qa path/to/qa.jsonl --corpus path/to/corpus.jsonl --limit 30 --use-llm-all
```

Optional: `FINRATE_USE_GPT_JUDGE=true` for an LLM score instead of local overlap. For full [Fin-RATE](https://github.com/jyd777/Fin-RATE), use their `qa/*.json` and `corpus/corpus.jsonl` (convert to JSONL if needed).

### v2.0 aggregate analysis

After running 30-task BrowseComp-Plus and Fin-RATE v2.0 sweeps, generate the aggregate architecture, oracle, router regret, and LaTeX tables with:

```bash
USE_LLM_ROUTER=false python scripts/analyze_llm_fixed_corpus_results.py \
  --browsecomp results/browsecomp/sweeps/<browsecomp-run>.json \
  --finrate results/finrate/converted/<finrate-run>.json \
  --balanced-model models/router_policy_pilots_110_20260509_100000.joblib \
  --unweighted-model models/router_policy_pilots_110_unweighted_20260509_101000.joblib
```

The latest aggregate outputs are under `results/llm_fixed_corpus_analysis/`, including architecture performance CSVs, router-regret CSVs, oracle-task exports, LaTeX tables, and the MAST-inspired semantic failure distribution used for charting.

## Learned router (hybrid: LLM features + sklearn)

1. Run any benchmark that writes three architectures per task (PCAB, WorkBench, BrowseComp, Fin-RATE fixtures).
2. `python scripts/export_router_training_rows.py results/<run>.json -o results/train.csv`
3. `pip install -e ".[router-ml]"` then `python scripts/train_router_from_regret.py results/train.csv -o models/router_policy.joblib`
4. `export ROUTER_LEARNED_MODEL_PATH=models/router_policy.joblib` — routing uses the multinomial logistic model before threshold fallbacks. Optional: `ROUTER_LEARNED_MIN_CONF=0.45` to abstain (falls through to thresholds).

Optional DSPy baseline check: `python scripts/router_dspy_baseline.py`

## Midstage report alignment

If you add `Midstage_report.pdf` to the repo (or point collaborators to it), keep these **code anchors** in sync with the write-up:

- **Composite oracle / regret:** defaults in [`evaluate_regret.py`](evaluate_regret.py) (`RegretEvaluator`: `accuracy_weight=0.50`, `latency_weight=0.40`, `token_weight=0.10`).
- **Routing thresholds:** [`route_task`](src/dynamic_routing/router.py) (tool-heavy / depth → SAS; parallelism bands → DMAS vs CMAS).
- **Extended routing features (for supervised policy):** [`RoutingMetadata`](src/dynamic_routing/vllm_integration.py).

## References

[1] Quantitative scaling principles for multi-agent systems; coordination tax on sequential vs. parallelizable tasks.

[2] [BrowseComp-Plus](https://arxiv.org/pdf/2508.06600) — fixed-corpus deep-research evaluation.

[3] [Fin-RATE](https://arxiv.org/pdf/2602.07294) — SEC filing QA benchmark.
