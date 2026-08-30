# Current Architecture

## Execution Flow
Based on `agent/my_agent.py`, the agent follows this sequence:

1. **Environment Observation**
   - The environment provides `FrameData`.
2. **State Abstraction**
   - **Grid Extraction:** The 2D grid is extracted from the `FrameData`.
   - **Object Segmentation:** A 4-connected flood fill identifies objects (color, size, centroid, bbox).
   - **Object Identification:** Each object gets a `(color, size, height, width)` signature.
   - **Hashing:** The grid is hashed for state uniqueness.
3. **Outcome Recording (Bookkeeping)**
   - Checks if a new level was cleared. If so, records the action sequence cost and clears the local context caches (dead actions, seen states, etc.).
   - Adds the previous action and whether it changed the hash to `_history`.
   - Tracks revisiting states (`seen_states`) to detect loops (`_stall`).
4. **Early Termination Check (Efficiency Guard)**
   - Returns `GameAction.RESET` if game is over.
   - If stuck for `STALL_LIMIT` (400 actions) since last level-up, it abandons the game.
   - If stuck for `OPAQUE_LIMIT` (120 actions) with 0 levels cleared, it abandons the game.
5. **Action Filtering**
   - Obtains legal candidates from `available_actions`.
   - Checks `_is_dead` against simple actions (if an action was tried 3 times and never changed the frame, it is pruned).
6. **Stuck-Recovery Consult (LLM)**
   - If levels cleared > 0, and not already consulted this level, and stuck for `STUCK_TRIGGER` (40 actions):
     - Builds a prompt describing the grid, top 12 objects, and last 10 actions.
     - Calls `ask_model()` to suggest an action or click target.
     - If the model provides a legal suggestion, sets `_model_action`.
7. **Action Selection (Heuristics)**
   - If `_model_action` exists, returns it.
   - If there are untried simple actions, picks one randomly (Recon phase).
   - If there are complex (click) actions available and objects exist:
     - Selects an object type (signature) that isn't blacklisted (dead).
     - Prefers never-clicked types, otherwise picks a fresh one randomly (up to top 8).
     - Emits a click action at the object's centroid.
   - If all else fails, falls back to the simple action with the best historical change-rate, breaking ties randomly.
8. **Action Execution**
   - Submits the action to the environment.
