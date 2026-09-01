# Baseline Report

## Date
2026-08-30

## Git Commit
`eeb1535` (HEAD of main)

## Environment Version
Official `arc-agi` pip package installed in `.venv`. OperationMode = NORMAL (uses live competition harness scoring).

## Hardware
Unknown/Local.

## Model
Local Ollama server via `ask_model` (qwen3:4b default or similar).

## Configuration
- MAX_ACTIONS defaults to 1400, or scaled based on `HUMAN` baseline (per-game).
- STALL_LIMIT = 400
- OPAQUE_LIMIT = 120
- STUCK_TRIGGER = 40

## Test Protocol
Run all 25 public ARC-AGI-3 environments (`run_baseline.py`). No cherry-picking. Record levels cleared and overall scoring.

## Public Development Set
Games: 25
Levels: 180 total across 25 games.

## Overall Score
**0.00026** (Mean across 25 games).

## Per-Game Results

| Game | Levels | Cleared | Actions | Score | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| ar25 | 8 | 0 | 544 | 0.0000 | 0.92 |
| bp35 | 9 | 0 | 1303 | 0.0000 | 5.51 |
| cd82 | 6 | 0 | 343 | 0.0000 | 0.60 |
| cn04 | 6 | 0 | 631 | 0.0000 | 0.89 |
| dc22 | 6 | 0 | 1770 | 0.0000 | 5.83 |
| ft09 | 6 | 0 | 123 | 0.0000 | 0.48 |
| g50t | 7 | 0 | 762 | 0.0000 | 1.34 |
| ka59 | 7 | 0 | 1461 | 0.0000 | 2.78 |
| lf52 | 10 | 0 | 191 | 0.0000 | 1.43 |
| lp85 | 8 | 2 | 614 | 0.0060 | 14.56 |
| ls20 | 7 | 0 | 642 | 0.0000 | 0.82 |
| m0r0 | 6 | 0 | 2215 | 0.0000 | 6.86 |
| r11l | 6 | 1 | 467 | 0.0005 | 5.64 |
| re86 | 8 | 0 | 427 | 0.0000 | 0.31 |
| s5i5 | 8 | 0 | 1277 | 0.0000 | 1.88 |
| sb26 | 8 | 0 | 427 | 0.0000 | 0.83 |
| sc25 | 6 | 0 | 701 | 0.0000 | 1.94 |
| sk48 | 8 | 0 | 2141 | 0.0000 | 5.78 |
| sp80 | 6 | 0 | 165 | 0.0000 | 0.24 |
| su15 | 9 | 0 | 249 | 0.0000 | 0.73 |
| tn36 | 7 | 0 | 188 | 0.0000 | 0.41 |
| tr87 | 6 | 0 | 638 | 0.0000 | 0.66 |
| tu93 | 9 | 0 | 330 | 0.0000 | 0.32 |
| vc33 | 7 | 1 | 534 | 0.0001 | 4.14 |
| wa30 | 9 | 0 | 2312 | 0.0000 | 1.61 |

## Runtime
Total baseline runtime: 66.52s. Mean per game: 2.66s. Mean per action: ~3.25ms.

## Agent Behavior
The agent extracts the grid, performs flood-fill segmentation, and identifies valid object signatures. It then chooses actions primarily on "novel" objects and falls back to simple actions when stuck. If stuck for >40 actions (`STUCK_TRIGGER = 40`), it invokes the LLM (`ask_model`) to get a single action string recommendation, then falls back to heuristics again.

## Failure Analysis

### Planning & World Model Failures (Primary)
The agent lacks an implementation for `verify(program, buffer)` or a World Model. Thus, there is NO hypothesis generation or transition code synthesis. It acts semi-randomly based on object signatures. This is the earliest point of unrecoverable failure for 22 out of 25 games.

### Exploration / Action-selection Failures
The agent relies on hitting the right combination of actions by heuristic chance. If the solution requires a specific sequence (e.g., `ACTION1` on a specific object, then `ACTION5`), the probability of discovering it by heuristic search without a plan is near zero. Consequently, it reaches `STALL_LIMIT` or exhaust action budgets and resets.

### Budget Penalties
Even when the agent succeeds (e.g. `lp85` level 1), it takes 73 actions compared to 17 human actions. This yields a ratio of 0.23, which is squared to give a negligible level score of 0.054. Because it is guessing instead of generating a python program that acts efficiently, it will NEVER achieve a competitive score even if it manages to clear the game.

## Strongest Games
`lp85` (Cleared 2 levels), `r11l` (Cleared 1 level), `vc33` (Cleared 1 level).

## Weakest Games
Games with huge action spaces like `m0r0` (2215 actions, 0 clears), `wa30` (2312 actions, 0 clears), `sk48` (2141 actions, 0 clears).

## Current Bottleneck
The complete absence of the Phase 1+ World Model and Planner. The LLM is only used as a fallback to guess a single action string when the agent gets stuck, rather than writing robust python code to transition states.

## Reproducibility
High reproducibility in action counts and score when run locally since it's predominantly deterministic heuristics (except when hitting the LLM API which has low temperature).

## Kaggle Readiness
The deployment pipeline (`make submit`) is functional, but `notebooks/kernel-metadata.json` requires the Kaggle username to be updated, and `.kaggle/access_token` must be present.

## Conclusion
The agent is solidly in Phase 0. It successfully wraps the official `arc-agi` harness, segments basic objects, and submits valid actions, but it is effectively a semi-random heuristic searcher. A structural upgrade to include the Planner, World Model (REPL), and proper prompt-driven python code generation is strictly required before any meaningful progress can be made.

