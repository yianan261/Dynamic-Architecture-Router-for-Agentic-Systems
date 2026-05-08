---
name: pr-review
description: Use when reviewing uncommitted changes, commits, or PR-style diffs before accepting code.
---

# PR Review Workflow

1. Inspect the diff:
   - `git status`
   - `git diff`
   - `git diff --stat`

2. Read `docs/review_checklist.md`.

3. Check:
   - architecture separation: SAS / CMAS / DMAS
   - benchmark output schema
   - regret calculation
   - semantic failure / MAST labels
   - tests or fixture validation
   - no secrets or local paths
   - no unrelated refactors

4. Run the closest safe validation:
   - targeted `pytest`
   - syntax check
   - fixture smoke test
   - benchmark schema check

5. Produce a review summary:
   - what changed
   - risks
   - missing validation
   - whether it is safe to commit

6. Do not commit or push.