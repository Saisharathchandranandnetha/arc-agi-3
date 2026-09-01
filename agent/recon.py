"""
recon.py - CPU-only reconnaissance + telemetry for ARC-AGI-3.

Zero GPU. Zero model. This is the layer that tells you WHICH of the 25 games
are tractable and WHICH actions do anything, so that when you eventually put
a model in the loop it isn't spending its (expensive) actions rediscovering
that ACTION3 is a no-op.

Three things nobody's published together:
  1. dead-action detection  - which of ACTION1..7 ever change the frame
  2. object segmentation    - turn 4096 click targets into ~10-30 candidates
  3. state-graph telemetry  - novelty, loops, and a tractability signal

Wire-up: you supply two adapter callables (see AdapterProtocol at bottom).

License: CC0-1.0
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------- frame hashing

def grid_hash(grid) -> int:
    """
    Canonical hash of a 64x64 grid. Accepts list-of-lists or numpy array.
    Used for no-op detection and state-revisit detection.
    """
    if hasattr(grid, "tobytes"):                 # numpy
        return hash(grid.tobytes())
    return hash(tuple(tuple(row) for row in grid))


def grid_diff_cells(a, b) -> int:
    """How many cells changed. 0 => the action was a no-op."""
    if hasattr(a, "shape"):                      # numpy
        return int((a != b).sum())
    return sum(1 for ra, rb in zip(a, b)
               for ca, cb in zip(ra, rb) if ca != cb)


# ---------------------------------------------------------- object segmentation

def segment_objects(grid, background: int | None = None,
                    min_size: int = 1) -> list[dict]:
    """
    4-connected flood fill over same-coloured cells. Dependency-free.

    Turns a 64x64 grid (4096 possible click targets) into a handful of
    candidate objects with centroids. This is the difference between random
    clicking being hopeless and being tractable.

    Returns list of {color, size, cells, centroid, bbox}, largest first.
    """
    rows = list(grid) if not hasattr(grid, "tolist") else grid.tolist()
    h = len(rows)
    w = len(rows[0]) if h else 0

    if background is None:                       # most common colour = background
        counts = defaultdict(int)
        for r in rows:
            for v in r:
                counts[v] += 1
        background = max(counts, key=counts.get)

    seen = [[False] * w for _ in range(h)]
    objects = []

    for y in range(h):
        for x in range(w):
            if seen[y][x] or rows[y][x] == background:
                continue
            color = rows[y][x]
            stack, cells = [(y, x)], []
            seen[y][x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < h and 0 <= nx < w
                            and not seen[ny][nx] and rows[ny][nx] == color):
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
                "cells": cells,
            })

    objects.sort(key=lambda o: -o["size"])
    return objects


def object_signature(obj) -> tuple:
    """
    Stable identity for an object TYPE (not instance), so 'clicking this kind
    of thing never does anything' generalises across positions. This is the
    dead-signature idea, made position-invariant.
    """
    y0, x0, y1, x1 = obj["bbox"]
    return (obj["color"], obj["size"], y1 - y0 + 1, x1 - x0 + 1)


# ------------------------------------------------------------------- telemetry

@dataclass
class ActionStat:
    tried: int = 0
    changed: int = 0
    total_cells_changed: int = 0
    score_gains: int = 0

    @property
    def is_dead(self) -> bool:
        """Tried enough times, never once changed the frame."""
        return self.tried >= 3 and self.changed == 0

    @property
    def hit_rate(self) -> float:
        return self.changed / self.tried if self.tried else 0.0


@dataclass
class GameTelemetry:
    game_id: str
    actions_taken: int = 0
    levels_cleared: int = 0
    wall_clock_s: float = 0.0
    action_stats: dict = field(default_factory=lambda: defaultdict(ActionStat))
    visited_states: set = field(default_factory=set)
    revisits: int = 0
    dead_signatures: set = field(default_factory=set)
    level_action_counts: dict = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)

    def start(self):
        self._t0 = time.time()

    def record(self, action_name: str, before, after,
               score_before: int, score_after: int):
        st = self.action_stats[action_name]
        st.tried += 1
        d = grid_diff_cells(before, after)
        if d > 0:
            st.changed += 1
            st.total_cells_changed += d
        if score_after > score_before:
            st.score_gains += 1
            self.level_action_counts[score_after] = self.actions_taken + 1
            self.levels_cleared = score_after

        self.actions_taken += 1
        h = grid_hash(after)
        if h in self.visited_states:
            self.revisits += 1
        self.visited_states.add(h)
        self.wall_clock_s = time.time() - self._t0

    # ---- derived signals

    @property
    def live_actions(self) -> list[str]:
        return [a for a, s in self.action_stats.items() if not s.is_dead]

    @property
    def dead_actions(self) -> list[str]:
        return [a for a, s in self.action_stats.items() if s.is_dead]

    @property
    def novelty_rate(self) -> float:
        """Fraction of actions that reached a NEW state. Low => stuck in a loop."""
        if not self.actions_taken:
            return 0.0
        return len(self.visited_states) / self.actions_taken

    @property
    def sec_per_action(self) -> float:
        return self.wall_clock_s / self.actions_taken if self.actions_taken else 0.0

    def tractability(self) -> tuple[str, str]:
        """
        Cheap early verdict, computed from the first N actions. This is the
        input to the scheduler: probe cheap, abandon opaque, reinvest.
        """
        if self.levels_cleared > 0:
            return "TRACTABLE", "already cleared a level"
        if self.novelty_rate < 0.15:
            return "OPAQUE", f"novelty {self.novelty_rate:.2f} - looping"
        if not self.live_actions:
            return "OPAQUE", "no action ever changed the frame"
        if self.novelty_rate > 0.60:
            return "PROMISING", f"novelty {self.novelty_rate:.2f} - state space opening up"
        return "UNKNOWN", f"novelty {self.novelty_rate:.2f}"

    def to_dict(self) -> dict:
        verdict, why = self.tractability()
        return {
            "game_id": self.game_id,
            "actions_taken": self.actions_taken,
            "levels_cleared": self.levels_cleared,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "sec_per_action": round(self.sec_per_action, 3),
            "distinct_states": len(self.visited_states),
            "revisits": self.revisits,
            "novelty_rate": round(self.novelty_rate, 3),
            "live_actions": sorted(self.live_actions),
            "dead_actions": sorted(self.dead_actions),
            "action_hit_rates": {a: round(s.hit_rate, 2)
                                 for a, s in sorted(self.action_stats.items())},
            "level_action_counts": self.level_action_counts,
            "verdict": verdict,
            "verdict_reason": why,
        }


# --------------------------------------------------------------- run reporting

def write_report(telemetries: list[GameTelemetry], path: str = "telemetry.json"):
    payload = [t.to_dict() for t in telemetries]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print("=" * 78)
    print(f"{'game':<8}{'acts':>6}{'lvls':>6}{'states':>8}{'novel':>8}"
          f"{'s/act':>8}  {'verdict':<11} dead")
    print("=" * 78)
    for d in payload:
        print(f"{d['game_id']:<8}{d['actions_taken']:>6}{d['levels_cleared']:>6}"
              f"{d['distinct_states']:>8}{d['novelty_rate']:>8.2f}"
              f"{d['sec_per_action']:>8.3f}  {d['verdict']:<11}"
              f" {','.join(a.replace('ACTION', 'A') for a in d['dead_actions'])}")

    n_tract = sum(1 for d in payload if d["verdict"] in ("TRACTABLE", "PROMISING"))
    print("-" * 78)
    print(f"tractable/promising: {n_tract}/{len(payload)}"
          f"   -> scheduler should concentrate budget on these")
    print(f"wrote {path}")


# ------------------------------------------------------------------ self-test

if __name__ == "__main__":
    # synthetic 12x12 grid: background 0, three objects
    g = [[0] * 12 for _ in range(12)]
    for y in range(2, 5):
        for x in range(2, 5):
            g[y][x] = 3                     # 3x3 square, colour 3
    for x in range(7, 11):
        g[1][x] = 5                         # 1x4 bar, colour 5
    g[9][9] = 7                             # single cell, colour 7

    objs = segment_objects(g)
    print("segmentation self-test")
    print(f"  found {len(objs)} objects (expected 3)")
    for o in objs:
        print(f"    colour={o['color']} size={o['size']:>2} "
              f"centroid={o['centroid']} sig={object_signature(o)}")
    assert len(objs) == 3
    assert objs[0]["size"] == 9 and objs[0]["color"] == 3

    print("\n  4096 raw click targets -> "
          f"{len(objs)} candidate centroids. That ratio is the whole point.")

    # telemetry self-test
    t = GameTelemetry("demo")
    t.start()
    before = [r[:] for r in g]
    after = [r[:] for r in g]
    after[0][0] = 9
    t.record("ACTION1", before, after, 0, 0)      # changed
    t.record("ACTION3", after, after, 0, 0)       # no-op
    t.record("ACTION3", after, after, 0, 0)
    t.record("ACTION3", after, after, 0, 0)
    print("\ntelemetry self-test")
    print(f"  dead_actions = {t.dead_actions} (expected ['ACTION3'])")
    assert t.dead_actions == ["ACTION3"]
    print("  OK")