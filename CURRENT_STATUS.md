# Current Status

## Last completed implementation: Project-scoped failure diagnostics

Implemented:
- Added `src/dynamic_routing/semantic_failure_judge.py`
- Wired conditional semantic labeling into:
  - `run_workbench_benchmark.py`
  - `run_browsecomp_sweep.py`
  - `run_finrate_sweep.py`
- Added `semantic_failure` field with label, confidence, reason, evidence, backend, model.
- Confirmed `semantic_failure_judge.py` is a post-hoc judge:
  - it runs only after benchmark grading,
  - it is applied only to failed CMAS/DMAS outputs,
  - SAS outputs are not assigned MAST-style semantic labels in the active sweeps.
- Added traceability:
  - `execution_path`
  - compact `trace` payloads
- Existing runtime `failure_taxonomy` tags remain inline in the architecture runners / WorkBench runner.
- Current taxonomy should be framed as a project-scoped, MAST-inspired diagnostic subset, not a full implementation of the MAST paper's 14 fine-grained modes.
- The subset focuses on:
  - runtime structural failures such as loop exhaustion, cyclic repetition, oscillating dispatch, context overflow, tool/dispatch explosion, and unhandled execution errors,
  - lightweight post-hoc semantic labels such as requirement miss, wrong assumption propagation, premature termination, incorrect synthesis / incomplete aggregation,
  - minimal DMAS-specific labels such as consensus failure, peer propagation drift, and unresolved conflict at termination.
- PCAB is pilot/legacy; active evaluation focus is WorkBench, BrowseComp-Plus-style, and Fin-RATE-style evaluation.

Validation status:
- The current tree has been inspected against this status.
- `python -m py_compile src/dynamic_routing/semantic_failure_judge.py run_workbench_benchmark.py run_browsecomp_sweep.py run_finrate_sweep.py src/dynamic_routing/browsecomp_runner.py src/dynamic_routing/finrate_runner.py` passed.
- Full fixture/benchmark runs are still pending.

## Important caution
Before building new functionality, compare this status against the current working tree:
- `git status`
- `git diff`
- inspect touched files
- confirm semantic failure fields still exist
