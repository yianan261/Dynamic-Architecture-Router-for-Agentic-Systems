# When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail

**Source PDF:** `2601.04748v2.pdf`  
**Primary role in this project:** Design principle for when SAS can replace MAS, and when skill selection becomes the bottleneck.  
**Use this note when:** deciding whether a task should use SAS with skills vs CMAS/DMAS, designing skills, or adding hierarchical routing.

## 1. One-paragraph summary

This paper argues that many multi-agent systems can be compiled into a single-agent system with skills, replacing explicit inter-agent communication with internal skill selection. This can reduce token usage and latency while preserving performance for compilable workflows. However, skill selection itself has limits: as the skill library grows, selection accuracy can collapse sharply, especially when skills are semantically similar. The paper’s core lesson for this project is that SAS is not “weak”; it can be cheaper and competitive, but only when the skill set is small, well-organized, and not semantically confusing. For larger skill libraries, hierarchical routing is needed.

## 2. Key ideas to keep

- MAS can sometimes be converted into SAS by turning agent roles into skills.
- This saves inter-agent communication overhead.
- The tradeoff is skill-selection overhead.
- Skill-based SAS can reduce token usage and latency compared with explicit MAS.
- Not all MAS are compilable into SAS.
- Skill selection exhibits bounded capacity: performance stays stable up to a point, then drops sharply.
- Semantic confusability among skills matters more than raw skill count alone.
- Hierarchical skill routing can mitigate flat skill-selection failure.

## 3. Compilability rules

The paper’s useful boundary condition:

A MAS is more likely compilable into SAS when:

```text
- agent communication can be serialized
- all agents share the same history
- all agents use the same model/backbone
- no private state is required
- no true parallel independent sampling is required
- no adversarial role separation is essential
```

A MAS is less likely compilable when:

```text
- agents need hidden/private information
- agents use heterogeneous capabilities
- agents require genuine parallel exploration
- adversarial debate is central to success
- best-of-N independent sampling is the core method
```

## 4. Relevance to Dynamic Architecture Router

This is directly aligned with your project’s SAS / CMAS / DMAS choice.

A strong router should ask:

```text
Can this task be solved by one agent with the right skill?
Or does it truly need multi-agent decomposition?
```

This paper supports a bias toward SAS when:

- the workflow is serializable,
- context can remain unified,
- the needed skills are few and distinct,
- latency/token cost matters,
- no independent exploration is required.

It supports MAS when:

- the task needs parallel exploration,
- specialized agents need separate contexts,
- verification benefits from role separation,
- skill selection would be ambiguous or overloaded.

## 5. Implementation implications

### 5.1 Add compilability metadata

Add metadata fields such as:

```text
is_serializable
requires_parallel_exploration
requires_private_state
requires_heterogeneous_tools
requires_adversarial_review
skill_confusability_risk
estimated_skill_count
```

### 5.2 Add skill-library complexity to routing

If you later add many Codex/agent skills, the router should not use a flat skill list. Use categories:

```text
benchmark skills
research skills
coding skills
review skills
retrieval skills
finance skills
```

### 5.3 Prefer hierarchical skill routing

Instead of selecting from every skill directly:

```text
user task -> choose skill category -> choose specific skill -> execute
```

This is the same idea as your Dynamic Architecture Router, but applied inside the SAS skill layer.

## 6. Suggested routing features

```text
compilability_score
serializable_workflow: bool
needs_parallelism: bool
needs_role_separation: bool
skill_library_size: int
skill_confusability_score: float
estimated_coordination_cost: float
estimated_skill_selection_cost: float
```

## 7. Suggested tasks for TASKS.md

```md
- [ ] Add a `compilability_score` to routing metadata.
- [ ] Add SAS-vs-MAS decision rules based on serializability, parallelism, and role-separation needs.
- [ ] Create a small skill taxonomy instead of a flat skill list.
- [ ] Track token/latency cost of SAS-with-skills vs CMAS/DMAS.
- [ ] Add tests for tasks that should remain SAS despite looking complex.
```

## 8. Files likely affected

```text
src/dynamic_routing/router.py
src/dynamic_routing/router_policy.py
src/dynamic_routing/vllm_integration.py
.agents/skills/
docs/research_notes.md
```

## 9. What not to overbuild yet

- Do not create dozens of skills immediately.
- Do not assume MAS is better because a task is complex.
- Do not use a flat skill list once skills grow beyond a small number.
- Do not compile tasks into SAS when true parallel exploration or independent sampling is the point.

## 10. Codex prompt to use

```text
Use docs/papers/single_agent_with_skills.md. Inspect the router metadata and policy code. Propose a minimal compilability_score feature that helps decide when SAS with skills should replace CMAS/DMAS. Do not edit yet; show the plan first.
```
