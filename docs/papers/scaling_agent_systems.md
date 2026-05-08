# Towards a Science of Scaling Agent Systems

**Source PDF:** `2512.08296v3.pdf`  
**Primary role in this project:** Quantitative framework for architecture selection across SAS, centralized MAS, decentralized MAS, independent MAS, and hybrid MAS.  
**Use this note when:** designing router features, regret labels, architecture selection rules, or evaluation reports comparing SAS/CMAS/DMAS.

## 1. One-paragraph summary

This paper studies how agent system performance changes with model capability, coordination topology, task properties, and system metrics. It evaluates SAS and several MAS variants across multiple agentic benchmarks and finds that architecture-task alignment matters more than simply adding agents. Multi-agent coordination can help decomposable tasks, but it can harm sequential tasks or tasks where the single-agent baseline is already strong. The paper is probably the most directly aligned with your Dynamic Architecture Router project because it motivates learning architecture selection from measurable task/system features rather than using fixed heuristics.

## 2. Key ideas to keep

- More agents do not automatically improve performance.
- Coordination has overhead: message passing, context fragmentation, synchronization, and error propagation.
- Single-agent systems preserve unified context.
- MAS can help when tasks are decomposable or benefit from parallel exploration/verification.
- MAS can hurt when tasks require tight sequential reasoning or when the single-agent baseline is already strong.
- Centralized verification can reduce error propagation.
- Decentralized systems can help with parallel exploration but may increase redundancy and coordination cost.
- Architecture selection can be predicted from task features and empirical coordination metrics.
- The paper reports architecture selection accuracy around 87% on held-out configurations under its framework.

## 3. Relevance to Dynamic Architecture Router

This paper is essentially a research justification for your project.

Your router should learn or estimate:

```text
Task structure + model capability + coordination cost -> best architecture
```

Instead of using a fixed rule like:

```text
complex task -> MAS
```

use measurable features:

```text
decomposability
sequential_interdependence
tool_count
single_agent_baseline_confidence
coordination_overhead
error_amplification_risk
verification_need
```

## 4. Architecture-selection implications

### 4.1 When SAS is likely best

Use SAS when:

- the task has strong sequential dependence,
- the task needs unified context,
- the single-agent baseline is already strong,
- tool count is low or tool flow is simple,
- coordination overhead would dominate.

### 4.2 When CMAS is likely best

Use centralized MAS when:

- the task is decomposable,
- there is meaningful verification or synthesis work,
- error propagation risk is high,
- a coordinator/verifier can prevent unchecked mistakes,
- outputs from subagents must be reconciled.

### 4.3 When DMAS is likely best

Use decentralized MAS when:

- the task benefits from parallel exploration,
- multiple independent evidence paths exist,
- diversity is valuable,
- redundancy can reduce uncertainty,
- there is less strict sequential dependency.

### 4.4 When independent/hybrid MAS is risky

Be careful with:

- independent agents without verification,
- hybrid systems with very high message overhead,
- multi-agent setups on tasks with high single-agent baselines,
- sequential planning tasks.

## 5. Implementation implications

### 5.1 Add task decomposition features

```text
decomposability_score
sequential_dependency_score
tool_count
requires_parallel_exploration
requires_central_verification
single_agent_confidence
estimated_coordination_overhead
```

### 5.2 Add architecture-level metrics

Log per run:

```text
architecture
success
latency
tokens
num_messages
num_agents
num_tool_calls
error_labels
verification_passed
coordination_overhead_estimate
```

### 5.3 Add learned routing target

The regret evaluator should produce training rows like:

```text
task_id, features..., sas_score, cmas_score, dmas_score, oracle_architecture, regret
```

This directly supports a learned router.

## 6. Suggested routing features

```text
task_decomposability
sequential_interdependence
tool_count
num_required_evidence_sources
single_agent_baseline_score
estimated_coordination_cost
requires_verification
requires_parallel_search
error_propagation_risk
```

## 7. Suggested tasks for TASKS.md

```md
- [ ] Add decomposability_score and sequential_dependency_score to routing metadata.
- [ ] Add architecture-level run metrics: latency, tokens, messages, tool calls.
- [ ] Modify regret outputs to include oracle_architecture and per-architecture scores.
- [ ] Add a report grouped by architecture and task type.
- [ ] Add rules/tests showing SAS should win on highly sequential tasks.
```

## 8. Files likely affected

```text
src/dynamic_routing/router.py
src/dynamic_routing/router_policy.py
src/dynamic_routing/vllm_integration.py
src/dynamic_routing/centralized_mas.py
src/dynamic_routing/decentralized_mas.py
evaluate_regret.py
run_benchmark_sweep.py
results/
```

## 9. What not to overbuild yet

- Do not add more architectures before SAS/CMAS/DMAS are well evaluated.
- Do not assume DMAS is best for all search-heavy tasks.
- Do not ignore latency/token cost when measuring success.
- Do not use final answer accuracy alone as the router training target.

## 10. Codex prompt to use

```text
Use docs/papers/scaling_agent_systems.md. Inspect the current router metadata and regret evaluation. Propose a minimal set of measurable features needed to predict SAS vs CMAS vs DMAS. Do not implement until you show the plan.
```
