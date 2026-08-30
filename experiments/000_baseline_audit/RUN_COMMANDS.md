# Run Commands

These are the commands used to perform this baseline audit, without touching any agent behavior.

## Trace Capture
```bash
.venv\Scripts\python.exe experiments/000_baseline_audit/capture_trace.py
```
This runs the agent on `ls20` with a max action cap of 10, intercepting `agent.take_action` to record observations and actions directly into `one_game_trace.json`.

## Full 25-Game Baseline Run
```bash
.venv\Scripts\python.exe experiments/000_baseline_audit/run_baseline.py
```
This script enumerates all 25 public ARC-AGI-3 environments, initializes `MyAgent`, calls `agent.main()`, and collects raw telemetry (scores, actions, clears, runtimes) into `results.json` and `results.csv`.
