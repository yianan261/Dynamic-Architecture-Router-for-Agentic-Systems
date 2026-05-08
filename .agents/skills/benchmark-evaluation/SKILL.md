---
name: benchmark-evaluation
description: Use when running, modifying, debugging, or analyzing benchmark sweeps for SAS, CMAS, or DMAS.
---

# Benchmark Evaluation Workflow

## Goal

Evaluate architectures while measuring coordination tax and routing regret.

## Relevant Benchmarks

- WorkBench
- BrowseComp-Plus-style tasks
- Fin-RATE-style tasks
- PCAB (pilot/legacy only)

## Relevant Files

- `run_workbench_benchmark.py`
- `run_browsecomp_sweep.py`
- `run_finrate_sweep.py`
- `run_benchmark_sweep.py`
- `evaluate_regret.py`

## Workflow

1. Determine whether this is:
   - smoke test
   - debug run
   - pilot run
   - full sweep

2. Prefer fixture-first evaluation:
   - smoke: 3–5 tasks
   - debug: 10–20 tasks
   - pilot: ~50 tasks
   - full: only after validation

3. Run the smallest relevant benchmark first.

4. Confirm outputs include:
   - architecture
   - accuracy/success
   - latency
   - token usage
   - composite reward
   - oracle architecture
   - regret
   - semantic failure
   - trace/execution path

5. Compare SAS / CMAS / DMAS behavior.

6. Watch for:
   - tool explosion
   - synthesis drift
   - recursive coordination loops
   - early exit
   - verification failure

7. Summarize:
   - architecture tradeoffs
   - coordination-tax observations
   - failure patterns
   - runtime/cost concerns

## Important Rules

- Do not run expensive full sweeps without approval.
- Prefer reproducible fixture runs.
- Keep output schema consistent across benchmarks.
- PCAB is pilot/legacy unless explicitly requested.