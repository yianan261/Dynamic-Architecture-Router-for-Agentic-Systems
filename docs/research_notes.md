# Research Notes for Dynamic Architecture Router

This file is the cross-paper synthesis. Detailed notes live in `docs/papers/`.

## Papers included

1. `docs/papers/browsecomp_plus.md`  
   Use for reproducible deep-research benchmark design, retrieval-vs-reasoning separation, search-call efficiency, and evidence/citation metrics.

2. `docs/papers/fin_rate.md`  
   Use for finance benchmark design, cross-entity comparison, longitudinal tracking, entity/year-aware retrieval, and fine-grained error labels.

3. `docs/papers/single_agent_with_skills.md`  
   Use for deciding when SAS with skills can replace MAS, when skill selection fails, and why hierarchical skill routing matters.

4. `docs/papers/scaling_agent_systems.md`  
   Use for architecture selection logic, measurable task features, coordination overhead, SAS/CMAS/DMAS tradeoffs, and learned router training.

5. `docs/papers/mast_multi_agent_failures.md`  
   Use for failure taxonomy, post-hoc judging, MAS failure labels, and review/evaluation loops.

## Cross-paper design principles

### 1. Do not assume more agents are better

The router should treat MAS as a cost-benefit decision, not a default upgrade. Multi-agent systems can help when tasks are decomposable, evidence is distributed, verification matters, or parallel exploration is useful. They can hurt when tasks are sequential, already handled well by SAS, or have high coordination overhead.

### 2. Separate retrieval failure from reasoning failure

For deep-research and finance tasks, wrong answers may come from missing evidence, bad ranking, hard negatives, entity/year mismatch, or bad synthesis. Router regret should not collapse all of these into one `incorrect` label.

### 3. Use task structure as routing metadata

Important routing features should include:

```text
decomposability_score
sequential_dependency_score
tool_count
num_required_evidence_sources
requires_parallel_exploration
requires_central_verification
requires_comparison
requires_temporal_reasoning
num_entities
num_time_periods
retrieval_confidence
single_agent_confidence
estimated_coordination_cost
```

### 4. Add failure taxonomy before complex repair loops

Before building autonomous repair agents, log why failures happen:

```text
specification_failure
inter_agent_misalignment
verification_failure
retrieval_failure
reasoning_failure
entity_mismatch
year_mismatch
coordination_overhead
```

### 5. Prefer hierarchical routing over flat selection

This applies at two levels:

```text
Task -> architecture: SAS / CMAS / DMAS
Task -> skill category -> specific skill
```

A large flat list of skills or tools will eventually confuse the model, especially when options are semantically similar.

## Suggested next implementation tasks

```md
- [ ] Add core routing metadata fields: decomposability_score, sequential_dependency_score, tool_count, num_required_evidence_sources.
- [ ] Add finance-specific fields: num_entities, num_time_periods, requires_comparison, requires_temporal_reasoning.
- [ ] Add retrieval diagnostics: retrieval_confidence, support_doc_hit_rate, hard_negative_hit_rate.
- [ ] Add failure taxonomy enums and output fields.
- [ ] Modify regret evaluation to store per-architecture score, oracle_architecture, and failure labels.
- [ ] Add report grouped by benchmark, task type, architecture, and failure category.
```

## Recommended Codex instruction

```text
Use AGENTS.md, TASKS.md, docs/PROJECT_PLAN.md, and docs/research_notes.md.
Before editing code, identify which research note applies to the task.
For benchmark/evaluation changes, preserve reproducibility and add the smallest useful test.
For routing-policy changes, update metadata or regret outputs before changing high-level architecture.
```
