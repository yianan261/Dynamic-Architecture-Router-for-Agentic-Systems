# Why Do Multi-Agent LLM Systems Fail?

**Source PDF:** `2503.13657v3.pdf`  
**Primary role in this project:** Project-scoped MAST-inspired diagnostic subset and post-hoc diagnosis for multi-agent routing outcomes.  
**Use this note when:** designing failure labels, review loops, post-hoc judges, or evaluation metrics for CMAS/DMAS runs.

## 1. One-paragraph summary

This paper introduces MAST, a taxonomy of multi-agent system failures, and MAST-Data, a dataset of annotated traces from multiple MAS frameworks. The taxonomy groups failures into system design issues, inter-agent misalignment, and task verification failures. For this project, the core value is that router evaluation should not stop at “SAS/CMAS/DMAS was correct or wrong.” It should diagnose why an architecture failed, because routing regret is only useful if the system can tell whether the failure came from poor task specification, bad coordination, ignored agent input, repeated steps, premature termination, or missing verification.

The active implementation should be described as a project-scoped MAST-inspired
diagnostic subset, not a complete implementation of the paper's full taxonomy.
It currently combines runtime structural tags, such as loop exhaustion, cyclic
dispatch, context overflow, tool or dispatch explosion, and unhandled execution
errors, with lightweight post-hoc semantic tags for failed CMAS/DMAS outputs.

## 2. Key ideas to keep

- MAS failures are often structural, not just model weakness.
- Multi-agent systems can fail due to poor specification, coordination breakdown, or weak verification.
- The paper identifies 14 failure modes across three major categories.
- LLM-as-judge can be calibrated to annotate failure modes with reasonable agreement to humans.
- Failure diagnosis can guide architecture redesign, not only prompt tweaks.
- Some failures happen before execution, some during execution, and some after execution.

## 3. Useful failure categories

### 3.1 System design / specification failures

Examples:

```text
disobey_task_specification
disobey_role_specification
step_repetition
loss_of_conversation_history
unaware_of_termination_conditions
```

### 3.2 Inter-agent misalignment failures

Examples:

```text
conversation_reset
fail_to_ask_for_clarification
task_derailment
information_withholding
ignored_other_agent_input
reasoning_action_mismatch
```

### 3.3 Task verification failures

Examples:

```text
premature_termination
no_or_incomplete_verification
incorrect_verification
```

## 4. Relevance to Dynamic Architecture Router

This paper should drive your failure analysis layer.

Your router should not only ask:

```text
Which architecture got the highest score?
```

It should also ask:

```text
Why did this architecture fail?
```

That distinction matters because the fix is different:

```text
retrieval failure -> improve retriever / context selection
specification failure -> improve task metadata / prompts
inter-agent misalignment -> improve coordination protocol
verification failure -> add reviewer/verifier or stricter acceptance criteria
coordination overhead -> choose SAS next time
```

## 5. Implementation implications

### 5.1 Add failure labels to evaluation outputs

Extend benchmark/regret outputs with:

```text
failure_category
failure_mode
failure_stage
failure_confidence
judge_rationale
```

### 5.2 Add a post-hoc judge

A judge can inspect trace/log/output and assign:

```text
success: bool
failure_category: specification | coordination | verification | retrieval | reasoning | none
failure_mode: string
rationale: string
```

Keep this optional at first so tests do not depend on external LLM calls.

### 5.3 Use failure labels for router improvement

A simple next step:

```text
If DMAS fails due to ignored input / conversation reset / no verification,
then prefer CMAS or SAS for similar tasks.

If SAS fails due to missing evidence or incomplete reasoning,
then try CMAS or DMAS next time.

If CMAS fails due to bottleneck or premature aggregation,
then try DMAS for high-parallel-search tasks.
```

## 6. Suggested routing/evaluation fields

```text
failure_category
failure_mode
failure_stage
requires_role_separation
coordination_risk
verification_risk
termination_condition_clarity
history_length
num_agent_messages
num_repeated_steps
```

## 7. Suggested tasks for TASKS.md

```md
- [ ] Keep project-scoped MAST-inspired diagnostic subset labels in benchmark output schema.
- [ ] Decide whether current inline runtime `failure_taxonomy` strings need stable constants.
- [ ] Keep optional post-hoc judge interface for failed CMAS/DMAS semantic classification.
- [ ] Add tests for failure taxonomy serialization.
- [ ] Modify regret evaluation to group results by failure category.
```

## 8. Files likely affected

```text
src/dynamic_routing/
src/dynamic_routing/router.py
src/dynamic_routing/centralized_mas.py
src/dynamic_routing/decentralized_mas.py
evaluate_regret.py
results/
tests/
```

## 9. What not to overbuild yet

- Do not implement a complex autonomous repair loop before the taxonomy/logging works.
- Do not require live LLM judges in unit tests.
- Do not treat MAST as final truth; use it as a practical starting point for a project-scoped MAST-inspired diagnostic subset.
- Do not label every failure manually if basic structured labels are enough for this phase.

## 10. Codex prompt to use

```text
Use docs/papers/mast_multi_agent_failures.md. Inspect evaluation and result schemas. Preserve the current project-scoped MAST-inspired diagnostic subset: runtime structural tags plus lightweight post-hoc semantic tags for failed CMAS/DMAS outputs. Decide whether a minimal failure_taxonomy.py constants module is worth adding before editing code.
```
