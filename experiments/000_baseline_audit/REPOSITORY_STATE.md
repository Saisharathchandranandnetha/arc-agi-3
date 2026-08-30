# Repository State

## File Audit
Based on the file tree and `ROADMAP.md` claims:

- `agent/my_agent.py`
  - ROADMAP CLAIM: exists (as my_agent.py)
  - ACTUAL REPOSITORY: exists (`agent/my_agent.py`)
- `baselines.py`
  - ROADMAP CLAIM: exists 
  - ACTUAL REPOSITORY: exists (`agent/baselines.py`)
- `rhae.py`
  - ROADMAP CLAIM: exists (Official scorer + strategy analysis)
  - ACTUAL REPOSITORY: missing
- `budget.py`
  - ROADMAP CLAIM: exists (Wall-clock modelling)
  - ACTUAL REPOSITORY: missing
- `eval_runs.py`
  - ACTUAL REPOSITORY: exists
- `PAPER.md`
  - ACTUAL REPOSITORY: exists
- `scripts/play_local.py`
  - ACTUAL REPOSITORY: exists
- `notebooks/kernel-metadata.json`
  - ACTUAL REPOSITORY: exists
- `vendor/ARC-AGI-3-Agents/`
  - ACTUAL REPOSITORY: exists

## Capabilities Audit

| Capability            | Status | Evidence/File | Notes |
| --------------------- | ------ | ------------- | ----- |
| Grid extraction       | IMPLEMENTED | `agent/my_agent.py` (extract_grid) | Successfully pulls 2D grid from FrameData |
| Segmentation          | IMPLEMENTED | `agent/my_agent.py` (segment_objects) | 4-connected flood fill to isolate objects |
| Object detection      | IMPLEMENTED | `agent/my_agent.py` (object_signature) | Identifies objects by color, size, and bbox dimensions |
| Action space          | IMPLEMENTED | `arcengine.GameAction` | Iterates over available actions provided by environment |
| Dead-action detection | IMPLEMENTED | `agent/my_agent.py` (_is_dead) | Prunes actions after DEAD_AFTER (3) fruitless tries |
| Loop avoidance        | IMPLEMENTED | `agent/my_agent.py` (seen_states) | Hashes grids to detect revisiting states |
| Per-game budgets      | IMPLEMENTED | `agent/my_agent.py` (MAX_ACTIONS) | Uses `baselines.py` HUMAN dict or defaults to 1400 |
| Replay buffer         | PARTIAL | `agent/my_agent.py` (_history) | Only tracks last 10 actions and level clearing paths, no full exact replay module |
| State logging         | IMPLEMENTED | `agent/my_agent.py` (seen_states) | State hashes are logged |
| Transition logging    | PARTIAL | `agent/my_agent.py` (_history) | Logs `(action, coords, changed_bool)` but not full state snapshots |
| World model           | MISSING | | No code for transition function synthesis exists |
| LLM                   | IMPLEMENTED | `agent/my_agent.py` (_consult_model) | Calls `ask_model` when stuck > 40 actions |
| REPL                  | MISSING | | No code found |
| Verification          | MISSING | | No exact replay verification code found |
| Planning              | MISSING | | No forward search over verified program exists |
| Goal inference        | MISSING | | No goal predicate fitting |
| Information gain      | MISSING | | No information gain metrics |
| Scheduler             | MISSING | | No play order optimization or opaque game dropping |
