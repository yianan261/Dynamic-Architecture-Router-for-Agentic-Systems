# Project Plan

## North Star

Build a regret-minimizing Dynamic Architecture Router that selects the most cost-effective agent topology — SAS, CMAS, or DMAS — based on task features, benchmark outcomes, and failure analysis.

The core research goal is to measure and reduce the Multi-Agent Coordination Tax: the latency, token, synchronization, and failure overhead introduced when multi-agent coordination is used on tasks where it is not structurally beneficial.

## Research Thesis

Multi-agent systems are not a universal upgrade over single-agent pipelines. They can improve trajectory accuracy on parallel or decomposable tasks, but they can also create severe coordination overhead, tool loops, synthesis drift, and catastrophic token bloat on sequential or tool-heavy tasks.

The router should therefore treat architecture choice as a dynamic decision variable, not a fixed design choice.

## Current Status

### Completed / Pilot Evidence

- PCAB was used as a pilot simulation benchmark.
- Phase I validated that graph/state mappings can execute gold trajectories without dropping state.
- Phase II live inference revealed severe CMAS coordination-tax failures.
- Observed CMAS failure modes include:
  - tool explosion
  - synthesis drift
  - recursive reprompting / coordination loops
- Although CMAS achieved higher average accuracy in the pilot, the composite reward oracle selected SAS for 100% of pilot tasks because CMAS latency and token costs dominated marginal accuracy gains.
- PCAB should now be treated as pilot/legacy evidence, not the primary final benchmark.

### Current Benchmark Direction

The project should transition toward broader evaluation on:

- WorkBench-style realistic workplace tasks
- BrowseComp-Plus-style deep research / retrieval-heavy tasks
- Fin-RATE-style financial analysis tasks

These benchmarks better test realistic agentic behavior, multi-document reasoning, retrieval, and tool-dependent workflows.

Fin-RATE evaluation should start with the local JSONL fixture, then use official
or custom converted data only after upstream `qa/*.json` and corpus files have
been converted to this repository's JSONL schema. The current runner is for
SAS/CMAS/DMAS comparison and regret-compatible outputs; it does not implement
official downloading or leaderboard formatting yet.

## Architecture Scope

The original midstage report focused on:

- SAS
- CMAS

The next stage should expand to:

- SAS
- CMAS
- DMAS

This is important because DMAS may expose different coordination-tax and failure patterns, especially around consensus, information withholding, verification failure, and inter-agent misalignment.

## Composite Reward and Routing Regret

Maintain the composite reward framing from the midstage report:

```text
Reward = α * Accuracy - β * NormalizedLatency - γ * NormalizedTokens
