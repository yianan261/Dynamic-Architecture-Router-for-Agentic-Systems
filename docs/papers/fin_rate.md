# Fin-RATE: Financial Analytics and Tracking Benchmark for LLMs on SEC Filings

**Source PDF:** `2602.07294v3.pdf`  
**Primary role in this project:** Benchmark design for multi-document, cross-entity, and temporal reasoning.  
**Use this note when:** working on Fin-RATE / FinanceBench-style adapters, routing metadata for finance tasks, temporal/entity-aware retrieval, or failure taxonomy.

## 1. One-paragraph summary

Fin-RATE evaluates LLMs on realistic financial analysis workflows using SEC filings. Instead of simple single-document question answering, it tests three paths: detail reasoning inside one disclosure, cross-company comparison, and longitudinal tracking of the same company across reporting periods. The paper is valuable for this project because it gives a concrete example of when multi-agent or structured routing may be useful: tasks that require aligning evidence across documents, entities, years, and disclosure sections. It also emphasizes fine-grained failure diagnosis rather than only final answer accuracy.

## 2. Key ideas to keep

- Real financial analysis requires cross-document, cross-temporal, and cross-entity reasoning.
- Single-document QA is too easy and does not capture professional analyst workflows.
- The benchmark defines three task families:
  - **DR-QA:** Detail & Reasoning within one disclosure chunk.
  - **EC-QA:** Enterprise Comparison across companies.
  - **LT-QA:** Longitudinal Tracking across reporting periods.
- Failures are not only wrong answers. They include retrieval errors, generation inconsistencies, financial reasoning failures, and comprehension issues.
- Entity/year mismatches are major failure sources in cross-company and longitudinal tasks.
- Hierarchical retrieval with entity-year constraints improves evidence coverage and ranking quality.
- Fine-grained scoring is more informative than binary correct/incorrect labels.

## 3. Relevance to Dynamic Architecture Router

Fin-RATE is a strong benchmark candidate because it naturally stresses architecture selection:

- DR-QA may often be enough for SAS.
- EC-QA may benefit from parallel agents comparing company-specific evidence.
- LT-QA may need temporal tracking and consistency checks.
- Centralized MAS may help when a verifier must align entities, years, and claims.

This gives your router real structure to learn from. The task type itself becomes a routing feature.

## 4. Implementation implications

### 4.1 Add finance task type metadata

Represent task type explicitly:

```text
task_family ∈ {detail_reasoning, enterprise_comparison, longitudinal_tracking}
```

Then map likely architecture preferences:

```text
DR-QA -> SAS or lightweight CMAS
EC-QA -> CMAS or DMAS depending on number of entities
LT-QA -> CMAS with temporal verifier
```

### 4.2 Add entity/year-aware metadata

Finance routing should track:

```text
num_companies
num_years
num_filings
num_chunks
requires_comparison
requires_temporal_reasoning
requires_numeric_reasoning
requires_evidence_alignment
```

### 4.3 Add failure labels beyond accuracy

Use labels such as:

```text
retrieval_error
entity_mismatch
year_mismatch
unsupported_comparison
trend_distortion
calculation_error
factual_inconsistency
intent_misunderstanding
```

These labels can improve regret analysis by explaining why an architecture failed.

### 4.4 Add hierarchical retrieval mode

For finance benchmarks, retrieval should not only rank all chunks globally. It should support entity/year scoped retrieval:

```text
company -> year -> filing type -> section/chunk
```

This is relevant to router design because the router can choose architecture partly based on retrieval structure.

## 5. Suggested routing features

```text
domain = finance
task_family
num_entities
num_time_periods
num_documents
requires_temporal_alignment
requires_entity_alignment
requires_numeric_computation
requires_comparison
retrieval_scope = global | entity_year_scoped
```

## 6. Suggested tasks for TASKS.md

```md
- [ ] Add finance-specific routing metadata: num_entities, num_years, requires_comparison, requires_temporal_alignment.
- [ ] Add failure labels for entity mismatch, year mismatch, trend distortion, and unsupported comparison.
- [ ] Add a fixture-style Fin-RATE adapter with DR-QA / EC-QA / LT-QA task families.
- [ ] Add architecture comparison report grouped by finance task family.
- [ ] Explore entity-year scoped retrieval for finance benchmark tasks.
```

## 7. Files likely affected

```text
benchmarks/finance*/
src/dynamic_routing/router.py
src/dynamic_routing/vllm_integration.py
src/dynamic_routing/router_policy.py
evaluate_regret.py
results/
```

## 8. What not to overbuild yet

- Do not implement a full SEC parser unless the benchmark requires it.
- Do not train on finance benchmark outputs without storing task type and retrieval diagnostics.
- Do not treat all finance tasks as equally multi-agent-worthy; DR-QA and LT-QA have different routing needs.

## 9. Codex prompt to use

```text
Use docs/papers/fin_rate.md. Inspect the current benchmark metadata and router metadata code. Propose minimal metadata fields needed to distinguish detail reasoning, enterprise comparison, and longitudinal tracking tasks. Do not modify source code until you show the plan.
```
