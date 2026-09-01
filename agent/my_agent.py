"""Your ARC-AGI-3 agent - recon build + stuck-recovery consult.

Replaces uniform-random selection with three things that cost no GPU:
  1. dead-action detection  - stop retrying actions that never change the frame
  2. object segmentation    - aim ACTION6 at real objects, not random pixels
  3. loop avoidance         - notice when we're cycling through seen states

On top of that, ONE rare model call per stuck episode. Heuristics clear level 1;
level 2 needs a hypothesis about the rule, which is what the model supplies. The
call is gated so it can never touch the ~430ms/action hot path: it fires only
after a level-up followed by STUCK_TRIGGER fruitless actions, at most once per
level. Every failure path falls straight back to the heuristics.

Contract unchanged: subclass Agent, class named MyAgent, implement
is_done() and choose_action().

License: CC0-1.0
"""
from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

try:
    from agent.model_client import ask_model, extract_json
except ImportError:                       # notebook/flat layout on Kaggle
    try:
        from model_client import ask_model, extract_json
    except ImportError:                   # no client at all -> pure heuristics
        def ask_model(prompt: str) -> str:      # type: ignore[misc]
            return ""

        def extract_json(text: str) -> dict:    # type: ignore[misc]
            return {}

# Set True for ONE run to dump the real FrameData layout, then set it back.
INTROSPECT = False


# ────────────────────────────────────────────────────────── grid helpers

def extract_grid(frame: FrameData):
    """
    Pull the 2-D grid off a FrameData. The stub docstring doesn't specify the
    layout, so try the known ARC-AGI-3 shapes: `.frame` is usually
    list[list[list[int]]] (a stack of layers) but may be a bare 2-D grid.
    Returns list[list[int]] or None.
    """
    g = getattr(frame, "frame", None)
    if g is None:
        return None
    try:
        if isinstance(g, list) and g and isinstance(g[0], list):
            if g[0] and isinstance(g[0][0], list):
                return g[-1]        # stack of layers -> current view
            return g                # already a 2-D grid
    except (IndexError, TypeError):
        pass
    return None


def grid_hash(grid) -> int:
    if grid is None:
        return 0
    return hash(tuple(tuple(row) for row in grid))


def segment_objects(grid, min_size: int = 1) -> list[dict]:
    """
    4-connected flood fill over same-coloured cells. Dependency-free.
    Collapses 4096 possible click targets into a handful of real objects.
    Returns [{color, size, centroid, bbox}], largest first.
    """
    if not grid:
        return []
    h, w = len(grid), len(grid[0])

    counts = defaultdict(int)
    for row in grid:
        for v in row:
            counts[v] += 1
    background = max(counts, key=counts.get)     # most common colour

    seen = [[False] * w for _ in range(h)]
    objects = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or grid[y][x] == background:
                continue
            color = grid[y][x]
            stack, cells = [(y, x)], []
            seen[y][x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny][nx]
                            and grid[ny][nx] == color):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_size:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            objects.append({
                "color": color,
                "size": len(cells),
                "centroid": (sum(ys) // len(ys), sum(xs) // len(xs)),
                "bbox": (min(ys), min(xs), max(ys), max(xs)),
            })
    objects.sort(key=lambda o: -o["size"])
    return objects


def object_signature(obj) -> tuple:
    """Position-invariant identity for an object TYPE, so 'clicking this kind
    of thing never works' generalises instead of being relearned per pixel."""
    y0, x0, y1, x1 = obj["bbox"]
    return (obj["color"], obj["size"], y1 - y0 + 1, x1 - x0 + 1)


def build_state_description(grid, objects, history) -> str:
    """Compact text view of the board for the model. Kept well under ~400
    tokens: the model is slow and a big prompt is the main cost driver.

    history: list of (action_name, coords_or_None, changed_bool) newest last.
    """
    lines = []
    if grid:
        lines.append(f"Grid: {len(grid)} rows x {len(grid[0])} cols "
                     f"(y=row 0..{len(grid)-1}, x=col 0..{len(grid[0])-1})")
    else:
        lines.append("Grid: unavailable")

    # Objects: cap the list so a noisy board can't blow up the prompt.
    lines.append(f"Objects ({len(objects)} total, largest {min(len(objects), 12)} shown):")
    for i, o in enumerate(objects[:12]):
        cy, cx = o["centroid"]
        y0, x0, y1, x1 = o["bbox"]
        # Spell the box out as ranges; "x3-4" reads as arithmetic to a small
        # model and derails the whole answer.
        lines.append(f"  [{i}] color={o['color']} size={o['size']} "
                     f"click_at x={cx} y={cy} "
                     f"(spans columns {x0} to {x1}, rows {y0} to {y1})")

    lines.append("Last actions (oldest first, 'changed' = frame changed):")
    for name, coords, changed in history[-10:]:
        where = ""
        if isinstance(coords, dict) and "x" in coords:
            where = f" at (x={coords['x']},y={coords['y']})"
        lines.append(f"  {name}{where} -> {'changed' if changed else 'NO CHANGE'}")
    if not history:
        lines.append("  (none)")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────── the agent

class MyAgent(Agent):
    """Segmentation-guided explorer with dead-action pruning."""

    MAX_ACTIONS = 80          # overwritten per-game in __init__
    DEAD_AFTER = 3            # no-op this many times -> stop trying it
    # Guard calibration. Deeper levels are weighted 2x, 3x... and the data shows
    # our reliable games CAN go deeper (lp85 hit L3 pre-guard). So be patient
    # after ANY progress, and only abandon games that never clear anything.
    STALL_LIMIT = 400        # actions since last level-up before giving up (was 150)
    OPAQUE_LIMIT = 120       # consecutive revisited states w/ 0 levels -> abandon (was 60)
    # Stuck-recovery consult. Fires at most ONCE per level, and only after we
    # have already cleared something - a game we never solved is a heuristic
    # problem, not a reasoning one. Tunable.
    STUCK_TRIGGER = 40       # fruitless actions since last level-up -> ask model

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(int(time.time() * 1e6) + hash(self.game_id) % 1_000_000)

        # Per-game budget: 2x the human total for known games, mean (~700) for
        # unseen games. The eval set is 110 games we've never seen, so the
        # fallback is what actually runs at competition time.
        try:
            from agent.baselines import HUMAN
            gid = self.game_id.split("-")[0]
            self.MAX_ACTIONS = 2 * sum(HUMAN.get(gid, [700]))
        except Exception:
            self.MAX_ACTIONS = 1400

        self.tried = defaultdict(int)          # action name -> times tried
        self.changed = defaultdict(int)        # action name -> times it changed frame
        self.dead_sigs: set = set()            # object signatures that do nothing
        self.clicked_sigs: set = set()         # signatures tried at least once
        self.seen_states: set = set()
        self.revisits = 0
        self.steps = 0

        self._prev_hash = None
        self._prev_action = None
        self._prev_sig = None
        self._prev_coords = None
        self._legal_names: list = []
        self._introspected = False

        # Level progress + efficiency-guard state.
        self.levels_seen = 0
        self.level_costs: dict[int, int] = {}   # level -> actions it cost
        self.level_paths: dict[int, list] = {}  # level -> action sequence that cleared it
        self._current_path: list = []           # actions since last level-up
        self._last_levelup_step = 0
        self._stall = 0                          # consecutive revisited states

        # Stuck-recovery consult state.
        self._consulted_levels: set = set()   # levels we already spent a call on
        self._model_action: tuple | None = None   # (action_name, coords) to play next
        self.model_calls = 0
        self.model_used = 0
        self._history: list = []              # (action_name, coords, changed)

    @property
    def name(self) -> str:
        return f"{super().name}.recon.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True

        # Efficiency guard. The score is (human_actions / our_actions)^2, so
        # every wasted action after our last level-up shrinks the score on the
        # levels we ALREADY cleared. If we've been stuck in seen-before states
        # for a long stretch with no new level, further play only bleeds score.
        actions_since_progress = self.steps - self._last_levelup_step
        if self.levels_seen > 0 and actions_since_progress > self.STALL_LIMIT:
            return True
        # Never cleared anything and deeply stuck in a loop -> this game is
        # opaque to the current policy; stop wasting budget on it.
        if self.levels_seen == 0 and self._stall > self.OPAQUE_LIMIT:
            return True
        return False

    # ---- bookkeeping -----------------------------------------------------

    def _introspect(self, frame: FrameData) -> None:
        """One-time dump of the real FrameData layout."""
        print(f"\n[INTROSPECT {self.game_id}]")
        print("  attrs:", [a for a in dir(frame) if not a.startswith("_")])
        raw = getattr(frame, "frame", None)
        print("  type(frame.frame):", type(raw).__name__)
        if isinstance(raw, list):
            print("  len:", len(raw))
            if raw and isinstance(raw[0], list):
                print("  len[0]:", len(raw[0]), "type[0][0]:", type(raw[0][0]).__name__)
        g = extract_grid(frame)
        print("  extracted grid:",
              f"{len(g)}x{len(g[0])}" if g else "FAILED -> fix extract_grid()")
        print("  score:", getattr(frame, "score", "n/a"))
        print("  available_actions:", getattr(frame, "available_actions", "n/a"))
        self._introspected = True

    def _record_outcome(self, grid, latest_frame, frames=None) -> None:
        """Attribute the frame change to whatever we did last turn, and track
        level progress so we can stop pouring actions into stuck games."""
        # Level progress: read the MAX levels_completed across the passed frame
        # and the harness's frames list, since the two can lag by one step.
        lvls = getattr(latest_frame, "levels_completed", 0) or 0
        if frames:
            for f in frames[-3:]:
                lvls = max(lvls, getattr(f, "levels_completed", 0) or 0)
        if lvls > self.levels_seen:
            # cleared a level - record how many actions THIS level cost
            cost = self.steps - self._last_levelup_step
            self.level_costs[lvls] = cost
            # capture the exact action sequence that cleared this level, for
            # path-caching: on a re-attempt we can replay instead of re-explore
            self.level_paths[lvls] = list(self._current_path)
            human = self._human_for_level(lvls)
            eff = (human / cost) if cost else 0.0
            print(f"  [CLEAR] {self.game_id} L{lvls} in {cost} actions "
                  f"(human {human}, ratio {eff:.2f}, "
                  f"level_score {min(1.0, eff)**2:.3f})")
            self.levels_seen = lvls
            self._last_levelup_step = self.steps
            self._stall = 0
            self._current_path = []
            # A new level may revive previously-dead actions/objects: what did
            # nothing on level N may be the key on level N+1. Clear the
            # blacklists so we re-explore fresh, but KEEP change-rate history
            # as a soft prior by halving it rather than wiping it.
            self.dead_sigs.clear()
            self.clicked_sigs.clear()
            for k in list(self.tried):
                self.tried[k] //= 2
                self.changed[k] //= 2
            self.seen_states.clear()      # states from the old level don't apply
            self._history.clear()         # nor does the old level's action log

        if self._prev_action is None:
            return
        h = grid_hash(grid)
        # Keep a short rolling log of what we did and whether it mattered; this
        # is the "what I've tried" half of the model prompt.
        self._history.append((self._prev_action, self._prev_coords,
                              h != self._prev_hash))
        del self._history[:-10]
        if h != self._prev_hash:
            self.changed[self._prev_action] += 1
        elif self._prev_sig is not None:
            # clicked this object type, nothing happened
            self.clicked_sigs.add(self._prev_sig)
            if self._prev_sig not in self.dead_sigs:
                self.dead_sigs.add(self._prev_sig)
        if h in self.seen_states:
            self.revisits += 1
            self._stall += 1
        else:
            self.seen_states.add(h)
            self._stall = 0

    # ---- stuck-recovery consult -----------------------------------------

    def _is_stuck(self) -> bool:
        """STUCK = we proved we can clear levels, then stopped making progress.

        Gating on levels_seen > 0 matters: a game we never solved at all is a
        heuristic/perception problem, and spending seconds of model time on it
        just bleeds the action budget.
        """
        if self.levels_seen <= 0:
            return False
        if self.levels_seen in self._consulted_levels:
            return False          # already spent this level's one call
        return (self.steps - self._last_levelup_step) > self.STUCK_TRIGGER

    def _build_prompt(self, grid, objects) -> str:
        n = self.levels_seen
        state = build_state_description(grid, objects, self._history)
        cleared = ", ".join(str(i) for i in range(1, n + 1)) or "none"
        legal = ", ".join(self._legal_names) if self._legal_names else "ACTION1-ACTION6"
        return (
            "This is a puzzle game. Here is the state and what I've tried. "
            f"I cleared level {n} but I'm stuck on level {n + 1}. "
            "Suggest which object to click or which action to try next, and "
            "briefly why. Respond as JSON: "
            '{"action": "ACTION1|ACTION2|...|ACTION6", "x": int, "y": int, '
            '"why": str}.\n\n'
            f"Levels cleared so far: {cleared}\n"
            f"Legal actions right now: {legal}\n"
            f'"action" must be EXACTLY ONE name copied from that list '
            f'(for example "{self._legal_names[0] if self._legal_names else "ACTION1"}") '
            "- not a list, not a pattern, no '|' characters.\n"
            "(x,y are only used for click-type actions; x=column, y=row.)\n\n"
            f"{state}\n\n"
            "Answer with the JSON object only."
        )

    def _consult_model(self, grid, objects, candidates) -> None:
        """One model call. Stores a validated action in self._model_action, or
        leaves it None so the heuristic path runs unchanged."""
        self._consulted_levels.add(self.levels_seen)   # spend the budget up-front
        self.model_calls += 1

        try:
            raw = ask_model(self._build_prompt(grid, objects))
        except Exception:          # a swapped-in client must never kill a run
            return
        if not raw:
            return                                  # unreachable / timeout / disabled
        try:
            data = extract_json(raw)
        except Exception:
            return
        if not isinstance(data, dict) or not data:
            return                                  # unparseable -> heuristics

        name = data.get("action")
        if not isinstance(name, str):
            return
        name = name.strip().upper().replace(" ", "_")
        # Legality is a hard veto: `candidates` is already filtered by the
        # frame's available_actions.
        match = next((a for a in candidates if a.name == name), None)
        if match is None:
            return

        coords = None
        if match.is_complex():
            x, y = data.get("x"), data.get("y")
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return                              # click with no target -> drop
            h = len(grid) if grid else 0
            w = len(grid[0]) if grid and grid[0] else 0
            if not (h and w and 0 <= int(y) < h and 0 <= int(x) < w):
                return                              # off-board -> drop
            coords = {"x": int(x), "y": int(y)}
            # For a click, "dead" is tracked per object TYPE, not per action:
            # ACTION6 itself goes dead on any stuck level, and vetoing on that
            # would kill every suggestion exactly when we need one. So veto only
            # if the model is pointing at an object type we've already ruled out.
            if self._sig_at(grid, coords) in self.dead_sigs:
                return
        elif self._is_dead(name):
            # A simple button that has provably never changed anything.
            return

        why = data.get("why")
        self._model_action = (match.name, coords)
        print(f"  [MODEL] {self.game_id} L{self.levels_seen}->L{self.levels_seen+1} "
              f"suggests {match.name}"
              f"{' at ' + str(coords) if coords else ''}: "
              f"{str(why)[:100] if why else 'no reason given'}")

    def _sig_at(self, grid, coords):
        """Signature of the object under a click target, or None."""
        if not grid or not coords:
            return None
        for o in segment_objects(grid):
            y0, x0, y1, x1 = o["bbox"]
            if y0 <= coords["y"] <= y1 and x0 <= coords["x"] <= x1:
                return object_signature(o)
        return None

    def _is_dead(self, name: str) -> bool:
        return self.tried[name] >= self.DEAD_AFTER and self.changed[name] == 0

    # ---- policy ----------------------------------------------------------

    def choose_action(self, frames: list[FrameData],
                      latest_frame: FrameData) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev_action = None
            self._prev_sig = None
            self._prev_hash = None
            self._prev_coords = None
            return GameAction.RESET

        grid = extract_grid(latest_frame)
        if INTROSPECT and not self._introspected:
            self._introspect(latest_frame)

        self._record_outcome(grid, latest_frame, frames)
        self.steps += 1

        # Respect the framework's legality list. available_actions is a list of
        # plain ints like [1,2,3,4,6], so map them back to GameAction members.
        avail = getattr(latest_frame, "available_actions", None)
        if avail:
            by_value = {a.value: a for a in GameAction}
            candidates = [by_value[v] for v in avail
                          if v in by_value and by_value[v] is not GameAction.RESET]
        else:
            candidates = [a for a in GameAction if a is not GameAction.RESET]
        if not candidates:
            candidates = [a for a in GameAction if a is not GameAction.RESET]

        self._legal_names = [a.name for a in candidates]

        simple = [a for a in candidates if not a.is_complex()]
        complex_ = [a for a in candidates if a.is_complex()]

        live_simple = [a for a in simple if not self._is_dead(a.name)]
        untried = [a for a in live_simple if self.tried[a.name] == 0]

        action, sig = None, None

        # Stuck-recovery consult. RARE by construction: at most one call per
        # level, and only once we've stalled STUCK_TRIGGER actions past a
        # clear. Everything below this block is the untouched heuristic path.
        if self._is_stuck():
            self._consult_model(grid, segment_objects(grid) if grid else [],
                                candidates)

        if self._model_action is not None:
            name, coords = self._model_action
            self._model_action = None          # one-shot: play it, then move on
            pick = next((a for a in candidates if a.name == name), None)
            if pick is not None:               # re-check legality at play time
                action = pick
                if coords is not None:
                    action.set_data(coords)
                action.reasoning = {"why": "model suggestion (stuck recovery)",
                                    "level": self.levels_seen}
                self.model_used += 1

        if action is not None:
            pass                               # model supplied it
        elif untried:
            # Recon phase: cost ~5 actions, tells us which buttons exist.
            action = random.choice(untried)
        elif complex_ and grid:
            # Aim a click at an object we haven't ruled out.
            objs = segment_objects(grid)
            fresh = [o for o in objs if object_signature(o) not in self.dead_sigs]
            novel = [o for o in fresh if object_signature(o) not in self.clicked_sigs]
            pick = None
            if novel:
                pick = random.choice(novel[:8])       # prefer never-clicked types
            elif fresh:
                pick = random.choice(fresh[:8])
            if pick is not None:
                action = complex_[0]
                sig = object_signature(pick)
                y, x = pick["centroid"]
                action.set_data({"x": int(x), "y": int(y)})
                action.reasoning = {
                    "why": "click segmented object",
                    "color": pick["color"], "size": pick["size"],
                    "objects_on_screen": len(objs),
                    "ruled_out_types": len(self.dead_sigs),
                }

        if action is None:
            # Fall back to the live action with the best observed change-rate.
            # Ties MUST be broken randomly: max() is deterministic, so on a
            # frozen board every rate is 0.0 and we'd spam one action forever.
            pool = live_simple or simple or candidates

            def rate(a):
                return (self.changed[a.name] / self.tried[a.name]
                        if self.tried[a.name] else 1.0)

            best = rate(max(pool, key=rate))
            action = random.choice([a for a in pool if rate(a) >= best - 1e-9])

        self.tried[action.name] += 1
        self._prev_action = action.name
        self._prev_sig = sig
        self._prev_hash = grid_hash(grid)
        self._prev_coords = getattr(action, "_data", None)

        # Record the action (with coords if complex) into the current level's
        # path, so a successful clear leaves a replayable sequence behind.
        coords = getattr(action, "_data", None)
        self._current_path.append((action.name, coords))

        if not action.is_complex():
            novelty = len(self.seen_states) / max(self.steps, 1)
            action.reasoning = (
                f"{action.name} | live={len(live_simple)}/{len(simple)} "
                f"novelty={novelty:.2f} revisits={self.revisits}")
        return action

    def _human_for_level(self, level: int) -> int:
        """Human action count for this level, or 0 if unknown (unseen game)."""
        try:
            from agent.baselines import HUMAN
            gid = self.game_id.split("-")[0]
            counts = HUMAN.get(gid)
            if counts and 1 <= level <= len(counts):
                return counts[level - 1]
        except Exception:
            pass
        return 0