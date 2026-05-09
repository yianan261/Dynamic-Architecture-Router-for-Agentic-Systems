# Current Status

## Last completed resume task: Fin-RATE structured GPT/local judging

Implemented:
- Added structured Fin-RATE answer judging in `src/dynamic_routing/finrate_runner.py`.
- Kept local judge as the default fallback.
- Added judge backend controls:
  - `FINRATE_JUDGE_BACKEND=local|gpt|auto`
  - `FINRATE_GPT_JUDGE_MODEL`
  - backward-compatible `FINRATE_USE_GPT_JUDGE=true|false`
- GPT judging uses `OPENAI_API_KEY` from the environment and does not hardcode keys.
- GPT judge output is normalized to a strict machine-readable object:
  - `score`: one of `0.0`, `0.5`, `1.0`
  - `reason`
  - `judge_backend`
  - `judge_model`
- `run_finrate_sweep.py` now stores the structured judge object under each architecture result's `judgment` field while preserving `accuracy_score` for `evaluate_regret.py`.
- Preserved semantic failure behavior for failed CMAS/DMAS outputs.
- Added `scripts/rejudge_finrate_results.py` to rescore saved Fin-RATE results without rerunning SAS/CMAS/DMAS.
- Added Fin-RATE judge environment settings to `.env.example`.
- Added targeted tests in `tests/test_finrate_judge.py`.

Validation status:
- `python -m py_compile run_finrate_sweep.py src/dynamic_routing/finrate_runner.py src/dynamic_routing/semantic_failure_judge.py scripts/rejudge_finrate_results.py` passed.
- `pytest tests/test_finrate_judge.py` passed: 3 tests.
- Tiny local fixture run passed:
  - `FINRATE_JUDGE_BACKEND=local FINRATE_USE_GPT_JUDGE=false python run_finrate_sweep.py --limit 1 --output fixture_finrate_structured_local.json`
  - Output: `results/finrate/fixtures/fixture_finrate_structured_local_20260508_231048.json`
- Rejudge without rerunning architectures passed:
  - `OPENAI_API_KEY=` with `FINRATE_JUDGE_BACKEND=gpt` produced graceful `gpt_fallback_local` judgments.
  - Output: `results/finrate/rejudged/rejudge_finrate_gpt_fallback_20260508_231119.json`
- Regret evaluation on the rejudged output passed:
  - `results/finrate/regret/regret_finrate_rejudge_gpt_fallback_20260508_231140.json`
  - `results/finrate/regret/regret_finrate_rejudge_gpt_fallback_20260508_231140.csv`

Remaining risk:
- Live GPT judging was not exercised with a real API call in validation; only graceful no-key fallback was tested.
- Do not run full Fin-RATE sweeps until a tiny converted-data sample has passed with the intended judge backend.

## Last completed resume task: Fin-RATE dataset-mode metadata

Implemented:
- Updated `run_finrate_sweep.py` to distinguish fixture runs from official/custom converted Fin-RATE-style runs.
- Kept fixture defaults unchanged:
  - `benchmarks/finrate_fixture_qa.jsonl`
  - `benchmarks/finrate_corpus_fixture.jsonl`
- Added metadata fields to Fin-RATE outputs:
  - `dataset_mode`
  - `task_count`
  - `task_type_counts`
  - `qa_path`
  - `corpus_path`
  - `per_type`
  - `limit`
  - `seed`
- Kept legacy `qa` and `corpus` metadata aliases for compatibility.
- Set phase from dataset mode:
  - `finrate_fixture`
  - `finrate_official_or_custom_converted`
- Added comments documenting that official Fin-RATE `qa/*.json` and corpus files must be converted to this runner's JSONL schema first.
- Preserved SAS/CMAS/DMAS comparison rows and semantic failure behavior for failed CMAS/DMAS outputs.

Validation status:
- `python -m py_compile run_finrate_sweep.py src/dynamic_routing/finrate_runner.py src/dynamic_routing/results_paths.py` passed.
- One-task fixture smoke passed:
  - `FINRATE_USE_GPT_JUDGE=false python run_finrate_sweep.py --limit 1 --output fixture_finrate_metadata_smoke.json`
  - Output: `results/finrate/fixtures/fixture_finrate_metadata_smoke_20260508_230258.json`
- Metadata smoke check confirmed:
  - `dataset_mode=fixture`
  - `phase=finrate_fixture`
  - `task_count=1`
  - `task_type_counts={"DR": 1}`
  - default QA/corpus paths recorded as `qa_path` and `corpus_path`
- Regret evaluation passed:
  - `python evaluate_regret.py results/finrate/fixtures/fixture_finrate_metadata_smoke_20260508_230258.json --export-json regret_finrate_metadata_smoke.json --export-csv regret_finrate_metadata_smoke.csv`
  - Output: `results/finrate/regret/regret_finrate_metadata_smoke_20260508_230321.json`
  - CSV: `results/finrate/regret/regret_finrate_metadata_smoke_20260508_230321.csv`
- Dataset-mode detection check confirmed custom QA or corpus paths map to `official_or_custom_converted` without running a sweep.

Remaining risk:
- Official Fin-RATE download/conversion and leaderboard formatting are still intentionally unimplemented.
- Converted-data mode should be tested on a tiny converted sample before any pilot or full sweep.

## Last completed resume task: BrowseComp fixture judge modes

Implemented:
- Improved BrowseComp scoring in `src/dynamic_routing/browsecomp_runner.py` without adopting the official BrowseComp-Plus leaderboard format.
- Added judge modes through `BROWSECOMP_JUDGE_BACKEND`:
  - `local`: normalized exact / contains / overlap heuristic.
  - `llm`: semantic-equivalence judge, preferring `BROWSECOMP_QWEN_URL` when configured and otherwise using the worker LLM; falls back to local if unavailable.
  - `auto`: runs local first, then calls LLM only for uncertain overlap scores.
- Kept the existing benchmark result schema used by `evaluate_regret.py`.
- Added `final_answer` to raw per-architecture benchmark rows for easier output inspection.

Validation status:
- `python -m py_compile run_browsecomp_sweep.py src/dynamic_routing/browsecomp_runner.py` passed.
- `pytest tests/test_browsecomp_judge.py tests/test_browsecomp_env.py` passed: 7 tests.
- BrowseComp local-judge fixture passed:
  - `BROWSECOMP_JUDGE_BACKEND=local python run_browsecomp_sweep.py --limit 3 --output fixture_browsecomp_judge_local.json`
  - Output: `results/browsecomp/fixtures/fixture_browsecomp_judge_local_20260508_225553.json`
  - All 3 tasks x 3 architectures scored with `judge=local_contains_gold`.
- Regret evaluation passed:
  - `python evaluate_regret.py results/browsecomp/fixtures/fixture_browsecomp_judge_local_20260508_225553.json --export-json regret_browsecomp_judge_local.json --export-csv regret_browsecomp_judge_local.csv`
  - Output: `results/browsecomp/regret/regret_browsecomp_judge_local_20260508_225614.json`
  - CSV: `results/browsecomp/regret/regret_browsecomp_judge_local_20260508_225614.csv`
- Schema smoke check passed for the new fixture output:
  - 3 tasks,
  - 3 architecture results per task,
  - required fields present: `architecture`, `final_answer`, `accuracy_score`, `latency_sec`, `total_tokens`, `error_type`, `execution_path`, `trace`, `semantic_failure`.

Remaining risk:
- The local heuristic is stable for fixture/regret experiments, but semantic paraphrase cases still need `auto` or `llm` mode for better equivalence judgment.
- Do not treat these 3 fixture items as benchmark evidence; they only validate scoring and schema plumbing.

## Last completed resume task: Fixture smoke testing

Completed:
- Ran BrowseComp fixture smoke with SAS, CMAS, and DMAS:
  - `python run_browsecomp_sweep.py --limit 3 --output fixture_browsecomp_smoke.json`
  - Output: `results/browsecomp/fixtures/fixture_browsecomp_smoke_20260508_213728.json`
  - Regret exports: `results/browsecomp/regret/regret_browsecomp_fixture_smoke_20260508_213811.json`, `results/browsecomp/regret/regret_browsecomp_fixture_smoke_20260508_213811.csv`
- Ran Fin-RATE fixture smoke with SAS, CMAS, and DMAS:
  - `python run_finrate_sweep.py --per-type 1 --output fixture_finrate_smoke.json`
  - Output: `results/finrate/fixtures/fixture_finrate_smoke_20260508_213653.json`
  - Regret exports: `results/finrate/regret/regret_finrate_fixture_smoke_20260508_213829.json`, `results/finrate/regret/regret_finrate_fixture_smoke_20260508_213829.csv`
- Ran WorkBench 3-task smoke with SAS, CMAS, and DMAS using `LLM_BACKEND=openai`, router annotation disabled:
  - `python run_workbench_benchmark.py --limit 3 --no-annotate-router --write-csv --output fixture_workbench_smoke.json`
  - Output: `results/workbench/fixtures/fixture_workbench_smoke_20260508_213849.json`
  - CSV table: `results/workbench/fixtures/fixture_workbench_smoke_20260508_213849.csv`
  - Regret exports: `results/workbench/regret/regret_workbench_fixture_smoke_20260508_214009.json`, `results/workbench/regret/regret_workbench_fixture_smoke_20260508_214009.csv`

Result organization:
- Existing flat files under `results/` were moved into benchmark-specific subfolders.
- New output helpers route future BrowseComp and Fin-RATE fixtures to `results/<benchmark>/fixtures/`.
- New WorkBench outputs route to `results/workbench/fixtures/` for limited/smoke runs and `results/workbench/sweeps/` for full runs.
- New regret exports route to `results/<benchmark>/regret/` based on the source result metadata.
- Directory guide: `results/README.md`.

Validation status:
- `python -m py_compile run_browsecomp_sweep.py run_finrate_sweep.py run_workbench_benchmark.py src/dynamic_routing/semantic_failure_judge.py src/dynamic_routing/browsecomp_runner.py src/dynamic_routing/finrate_runner.py` passed.
- Schema smoke check passed for all three fixture JSONs:
  - each file has 3 tasks,
  - each task has 3 architecture results,
  - required fields are present: `architecture`, `accuracy_score`, `latency_sec`, `total_tokens`, `error_type`, `execution_path`, `trace`.
- Fin-RATE smoke produced 4 semantic failure labels on failed CMAS/DMAS outputs.
- BrowseComp and WorkBench smoke outputs had no failed CMAS/DMAS cases requiring semantic labels.
- Regret evaluation ran successfully for all three fixture outputs.

Fixture observations:
- BrowseComp fixture: all 3 smoke tasks scored 1.00 for SAS/CMAS/DMAS; oracle selected SAS for all three due lower latency.
- Fin-RATE fixture: one item per task type ran successfully; EC and LT fixture items produced partial local-overlap scores and semantic labels for failed CMAS/DMAS outputs.
- WorkBench fixture: first 3 email tasks all scored 1.00 for SAS/CMAS/DMAS; oracle selected SAS for 2 tasks and DMAS for 1 task under the current composite reward.

Pilot recommendation:
- Use the existing 50-instance WorkBench slice for the next pilot sweep, as planned.
- Keep BrowseComp and Fin-RATE pilot sweeps small first because the current fixtures are tiny and mostly synthetic; scale only after checking a larger 10-20 item debug run or real upstream data availability.

Remaining next item:
- Run pilot sweep across the three active benchmarks, then analyze project-scoped failure diagnostic results and compare failure modes by architecture.

## Last completed resume task: MAST-inspired diagnostic wording

Completed:
- Documented the active framing as a project-scoped MAST-inspired diagnostic subset:
  - runtime structural tags emitted during execution,
  - lightweight post-hoc semantic tags for failed CMAS/DMAS outputs,
  - not a full implementation of the MAST paper's taxonomy.
- Standardized that wording in:
  - `docs/research_notes.md`
  - `docs/papers/mast_multi_agent_failures.md`
  - benchmark result metadata notes for WorkBench, BrowseComp-Plus-style, and Fin-RATE-style sweeps.
- Marked the corresponding `TASKS.md` items complete.

Validation status:
- `python -m py_compile run_workbench_benchmark.py run_browsecomp_sweep.py run_finrate_sweep.py` passed.

Remaining next item:
- Fixture-test BrowseComp, Fin-RATE, and WorkBench before any pilot sweep.

Taxonomy constants decision:
- Do not add a central constants module yet.
- Existing runtime `failure_taxonomy` strings include dynamic details such as loop scores, tool signatures, and exception snippets.
- Keep the current inline strings until fixture outputs show a concrete grouping or serialization problem.
- Do not add the full MAST taxonomy unless explicitly needed for the paper claim.

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
