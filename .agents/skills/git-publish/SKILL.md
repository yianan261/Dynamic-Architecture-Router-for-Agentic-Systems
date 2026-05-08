---
name: git-publish
description: Use only when the user explicitly asks to commit and push reviewed changes to the remote GitHub repository.
---

# Git Publish Workflow

## Safety Rules

- Never push without explicit user approval.
- Never force push unless the user explicitly requests it.
- Never commit secrets, `.env`, credentials, large datasets, cache files, or generated junk.
- Always run `git status` before committing.
- Always show the commit summary before pushing.

## Workflow

1. Confirm the branch:
   - `git branch --show-current`

2. Inspect changes:
   - `git status`
   - `git diff --stat`

3. Check for unsafe files:
   - `.env`
   - credentials
   - API keys
   - large benchmark outputs
   - cache directories
   - model weights
   - temporary files

4. Run the closest relevant validation.

5. Stage only intended files:
   - avoid `git add .` unless changes were already reviewed

6. Create a clear commit message:
   - example: `Add MAST failure judge and fixture validation docs`

7. Ask for final approval before:
   - `git push`

8. After push, summarize:
   - branch
   - commit hash
   - files changed
   - tests/checks run