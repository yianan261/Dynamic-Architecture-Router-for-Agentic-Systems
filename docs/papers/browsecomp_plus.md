# BrowseComp-Plus: Fair and Transparent Evaluation for Deep-Research Agents

**Source PDF:** `2508.06600v1.pdf`  
**Primary role in this project:** Benchmark design for deep-research / web-search agent evaluation.  
**Use this note when:** working on BrowseComp-style benchmark adapters, retrieval-vs-reasoning attribution, search-call metrics, or reproducible evaluation.

## 1. One-paragraph summary

BrowseComp-Plus argues that deep-research agents should not be evaluated only through live black-box web search because that makes results hard to reproduce, expensive, and impossible to attribute cleanly between the retriever and the LLM agent. The paper introduces a fixed, human-verified corpus with supporting documents and hard negatives, allowing controlled evaluation of retrieval quality, agent reasoning, citation behavior, and search efficiency. For this Dynamic Architecture Router project, the most important lesson is that benchmark design should separate **retrieval failure** from **reasoning/orchestration failure**, otherwise the router may learn noisy or misleading regret labels.

## 2. Key ideas to keep

- Deep-research systems combine LLM reasoning with iterative search, search planning, and reflection.
- Live web-search APIs create fairness and reproducibility problems because the corpus and ranking behavior change over time.
- A fixed corpus with human-verified support documents allows cleaner component-level analysis.
- Supporting documents and hard negatives let the evaluator test whether the agent finds the right evidence, avoids distractors, and cites properly.
- Better retrievers can improve final accuracy while also reducing the number of search calls.
- Evaluation should track both answer correctness and process efficiency, not just final output.
- Deep-research benchmark results are sensitive to retriever-agent interaction.

## 3. Relevance to Dynamic Architecture Router

This paper is directly relevant because your router needs to decide when a task needs:

- a single agent with simple retrieval,
- centralized multi-agent search and synthesis,
- decentralized exploration,
- or a more expensive deep-research workflow.

The paper suggests that routing decisions should not only depend on task difficulty. They should also depend on:

- evidence availability,
- retrieval confidence,
- search-call budget,
- number of evidence pieces required,
- citation reliability,
- whether failures are due to bad retrieval or bad synthesis.

## 4. Implementation implications

### 4.1 Add retrieval-aware evaluation fields

When evaluating BrowseComp-style tasks, log fields like:

```text
retriever_name
num_search_calls
retrieved_doc_ids
support_doc_hit_rate
hard_negative_hit_rate
citation_coverage
answer_accuracy
retrieval_failure
reasoning_failure
```

This prevents the router from treating all wrong answers as architecture failures.

### 4.2 Separate retriever quality from architecture quality

If CMAS beats SAS only because it made more search calls, that is not necessarily a better architecture. The evaluator should normalize or at least report:

```text
accuracy_per_search_call
accuracy_per_token
accuracy_per_latency_second
```

### 4.3 Add controlled benchmark mode

Prefer a fixed local corpus or cached retrieval results for router training. Live web search is useful for demos, but bad for training regret labels.

### 4.4 Consider oracle-retrieval experiments

Run experiments under two conditions:

1. **retrieval-augmented setting:** model must retrieve evidence itself.
2. **oracle-context setting:** model receives gold evidence.

If architecture differences disappear under oracle context, then the real bottleneck is retrieval. If differences remain, then routing/orchestration matters.

## 5. Suggested routing features

Add or compute features such as:

```text
task_requires_search: bool
estimated_num_evidence_sources: int
retrieval_confidence: float
retrieval_entropy: float
hard_negative_risk: float
requires_citation: bool
requires_iterative_search: bool
search_budget: int
```

## 6. Suggested tasks for TASKS.md

```md
- [ ] Add retrieval-aware logging fields to benchmark outputs.
- [ ] Add support_doc_hit_rate and hard_negative_hit_rate metrics for BrowseComp-style evaluation.
- [ ] Add `retrieval_failure` vs `reasoning_failure` labels to regret outputs.
- [ ] Add accuracy-per-search-call reporting.
- [ ] Create a fixture/cached retrieval mode for reproducible benchmark runs.
```

## 7. Files likely affected

```text
benchmarks/
evaluate_regret.py
run_benchmark_sweep.py
src/dynamic_routing/router.py
src/dynamic_routing/vllm_integration.py
results/
```

## 8. What not to overbuild yet

- Do not build a full search engine before the router logic is stable.
- Do not train the router on live-web results without caching; the labels will drift.
- Do not merge retrieval failure and reasoning failure into one generic `incorrect` label.

## 9. Codex prompt to use

```text
Use docs/papers/browsecomp_plus.md. Inspect the benchmark and regret evaluation code. Propose a minimal change to separate retrieval failure from reasoning failure in benchmark outputs. Do not modify source code until you show the plan.
```
