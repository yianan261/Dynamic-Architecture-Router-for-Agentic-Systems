---
name: research-paper-ingestion
description: Use when integrating research papers or external methods into the project.
---

# Research Paper Ingestion Workflow

## Goal

Translate research papers into concrete implementation implications for the Dynamic Architecture Router project.

## Workflow

1. Read:
   - `docs/research_notes.md`
   - relevant files under `docs/papers/`

2. Extract:
   - main contribution
   - assumptions
   - relevant design principles
   - evaluation methodology
   - routing implications
   - failure-analysis implications

3. Identify which parts are relevant to:
   - SAS
   - CMAS
   - DMAS
   - routing
   - regret evaluation
   - benchmark design
   - failure taxonomy

4. Map ideas to current repo files.

5. Propose incremental implementation changes before editing code.

6. Avoid:
   - overengineering
   - copying full paper logic blindly
   - adding unnecessary agent complexity

7. Update:
   - `docs/research_notes.md` for cross-paper synthesis
   - `docs/papers/*.md` for paper-specific notes

## Important Rules

- Prefer distilled operational understanding over raw paper text.
- Keep implementation changes incremental.
- Do not assume papers are directly production-ready.