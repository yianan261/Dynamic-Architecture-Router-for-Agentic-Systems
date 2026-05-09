# TASKS.md

## Ready

### Status verification
- [x] Compare prior semantic judge summary against current working tree.
- [x] Confirm `src/dynamic_routing/semantic_failure_judge.py` still exists.
- [x] Confirm benchmark outputs include `semantic_failure`.
- [x] Confirm BrowseComp/Fin-RATE runners include `execution_path` and `trace`.
- [x] Rerun validation for touched evaluation files (`py_compile` or targeted tests).
- [x] Record validation outcome in `CURRENT_STATUS.md`.

### Project-scoped MAST-inspired diagnostics
- [x] Read `docs/papers/mast_multi_agent_failures.md`.
- [x] Confirm current semantic judge is post-hoc and limited to failed CMAS/DMAS outputs.
- [x] Document the project-scoped taxonomy framing: runtime structural tags + lightweight post-hoc semantic tags, not full MAST.
- [x] Standardize wording across docs/results notes: "project-scoped MAST-inspired diagnostic subset."
- [x] Decide whether existing inline runtime `failure_taxonomy` strings need central constants for consistency.
- [x] Do not add full MAST taxonomy unless explicitly needed for the paper claim.

### Fixture testing
- [x] Run BrowseComp fixture with SAS/CMAS/DMAS.
- [x] Run Fin-RATE fixture with SAS/CMAS/DMAS.
- [x] Run WorkBench on small subset with SAS/CMAS/DMAS.
- [x] Decide pilot instance count based on runtime and cost (use 50 instances for WorkBench).

### Fin-RATE runner safety
- [x] Keep fixture defaults unchanged for `run_finrate_sweep.py`.
- [x] Add explicit Fin-RATE `dataset_mode` metadata for fixture vs official/custom converted data.
- [x] Add Fin-RATE run metadata: `task_count`, `task_type_counts`, `qa_path`, `corpus_path`, `per_type`, `limit`, `seed`.
- [x] Preserve SAS/CMAS/DMAS result schema compatibility with `evaluate_regret.py`.
- [x] Document that official Fin-RATE `qa/*.json` must be converted to this runner's JSONL schema first.
- [x] Add structured Fin-RATE judge outputs with local/GPT/auto backend selection.
- [x] Add saved-result Fin-RATE rejudge script that does not rerun SAS/CMAS/DMAS.
- [x] Add conversion scripts for BrowseComp-Plus decrypted data/corpus and Fin-RATE HuggingFace snapshots.
- [ ] Validate conversion scripts against real downloaded upstream data.
- [ ] Implement official Fin-RATE download / leaderboard formatting only if explicitly needed later.

### Sweeps
- [ ] Run pilot sweep across:
      - WorkBench
      - BrowseComp-Plus
      - Fin-RATE
      using sampled real benchmark subsets.

- [ ] Execute SAS / CMAS / DMAS for all pilot tasks.

- [ ] Run semantic judging:
      - BrowseComp → Qwen/local
      - Fin-RATE → GPT judge

- [ ] Compute:
      - accuracy
      - latency
      - token usage
      - reward
      - oracle architecture
      - regret

- [ ] Analyze project-scoped failure diagnostics.

- [ ] Compare:
      - coordination tax
      - failure modes
      - regret trends
      across architectures.

### Learned router
- [ ] Review routing metadata features.

- [ ] Generate router training labels from oracle/regret outputs.

- [ ] Train logistic regression router.

- [ ] Compare learned router vs rule-based router:
      - routing accuracy
      - average regret
      - architecture selection distribution

## In Progress
- [ ] None

## Review
- [ ] None

## Done
- [ ] Research paper markdown notes created.
