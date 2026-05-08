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
- [ ] Document the project-scoped taxonomy framing: runtime structural tags + lightweight post-hoc semantic tags, not full MAST.
- [ ] Standardize wording across docs/results notes: "project-scoped MAST-inspired diagnostic subset."
- [ ] Decide whether existing inline runtime `failure_taxonomy` strings need central constants for consistency.
- [ ] Do not add full MAST taxonomy unless explicitly needed for the paper claim.

### Fixture testing
- [ ] Run BrowseComp fixture with SAS/CMAS/DMAS.
- [ ] Run Fin-RATE fixture with SAS/CMAS/DMAS.
- [ ] Run WorkBench on small subset with SAS/CMAS/DMAS.
- [ ] Decide pilot instance count based on runtime and cost (use 50 instances for WorkBench).

### Sweeps
- [ ] Run pilot sweep across 3 benchmarks × 3 architectures.
- [ ] Analyze project-scoped failure diagnostic results.
- [ ] Compare failure modes by architecture.

### Learned router
- [ ] Review current routing metadata features.
- [ ] Define LR training labels from oracle/regret outputs.
- [ ] Train logistic regression router.
- [ ] Compare LR router against rule-based router.

## In Progress
- [ ] None

## Review
- [ ] None

## Done
- [ ] Research paper markdown notes created.
