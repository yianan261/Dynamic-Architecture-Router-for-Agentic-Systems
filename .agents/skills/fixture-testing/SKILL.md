---
name: fixture-testing
description: Use when validating new routing, benchmark, or failure-taxonomy changes before full benchmark sweeps.
---

# Fixture Testing Workflow

## Goal

Validate changes safely and cheaply before running expensive sweeps.

## Workflow

1. Select the smallest representative fixture:
   - WorkBench
   - BrowseComp
   - Fin-RATE

2. Prefer:
   - 3–5 task smoke runs first
   - then 10–20 task debug runs

3. Run:
   - SAS
   - CMAS
   - DMAS
   when applicable.

4. Validate:
   - no crashes
   - output schema consistency
   - regret calculation
   - semantic failure labels
   - execution traces
   - token/latency reporting

5. Compare outputs against:
   - prior runs
   - expected architecture behavior
   - known pilot results

6. Watch for:
   - runaway loops
   - token explosion
   - invalid JSON outputs
   - empty traces
   - missing metadata
   - router instability

7. Only recommend larger sweeps after fixture validation succeeds.

## Important Rules

- Do not jump directly to full datasets.
- Prefer reproducible small runs.
- Record limitations clearly.
- If results look suspicious, inspect traces before scaling evaluation.