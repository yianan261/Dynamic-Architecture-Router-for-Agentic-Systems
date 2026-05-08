# AGENTS.md

## Project goal
Build a Dynamic Architecture Router for Agentic Systems that routes tasks to SAS, CMAS, or DMAS based on task metadata, benchmark feedback, and regret evaluation.

## Core rules
- Inspect before editing.
- Plan before large changes.
- Prefer small, reviewable diffs.
- Preserve SAS / CMAS / DMAS separation.
- Do not rewrite architecture unless explicitly requested.
- Do not invent benchmark results.
- Do not commit secrets, API keys, or credentials.

## Required workflow
For every coding task:
1. Identify relevant files.
2. State a short implementation plan.
3. Implement the smallest useful change.
4. Run relevant tests/checks.
5. Review the diff.
6. Summarize what changed and what remains risky.

## Verification
Before saying done, run the closest relevant check:
- `pytest`
- targeted test file
- benchmark fixture
- regret evaluation script

## Output Documentation
For any runs (except experimental fixtures), be sure the output will be documented and recorded in some csv output file with timestamped names under organized subfolders in the `results` directory. Organize the files in the results directory so it's clear which files are the resulting output from what.

## Current project emphasis
This project currently focuses on WorkBench, BrowseComp-Plus-style, and Fin-RATE-style evaluation. PCAB is treated as pilot/legacy unless explicitly requested.

## Failure analysis rule
When changing MAS evaluation, preserve traceability fields and failure diagnosis outputs. Do not remove semantic failure or MAST-related fields without explicit approval.

If full benchmark data is unavailable, explain what is missing and run the closest local check. The project is in a conda environment on a shared remote server. 