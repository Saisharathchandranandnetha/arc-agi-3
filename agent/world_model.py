"""World-model substrate for ARC-AGI-3: snapshots, replay, verification, planning.

This module is DELIBERATELY NOT A SOLVER. It is the trust layer a solver gets
built on top of. It answers exactly one question honestly:

    "Does this candidate transition model reproduce what the game actually did?"

Nothing here infers rules, synthesises programs, or improves any score. Wiring
it into `MyAgent` costs a few microseconds per action and buys the ability to
*reject* a wrong world model before it is ever allowed to plan.

Layering (each layer only trusts the one below it):

    GridState / ActionRecord   immutable canonical snapshots
    Transition / ReplayBuffer  the ground-truth record, bounded
    verify_transition_model    exact replay -> PASS or first mismatch
    check_determinism          same inputs twice -> hidden state / RNG rejected
    certify                    the ONLY way to mint a VerifiedModel
    plan                       refuses to run without a VerifiedModel + goal

=== SAFETY MODEL ===

Candidate transition models arrive from an LLM. They are UNTRUSTED INPUT.

  * There is no `exec`, no `eval`, no `compile`, and no `import` of model text
    anywhere in this module. Model output never becomes Python.
  * Candidates are expressed in a declarative JSON-shaped DSL (see `DSL_OPS`).
    A candidate is DATA. `compile_transition_model` validates it against a
    closed allowlist of operations and rejects anything else with
    `CandidateRejected`.
  * Every DSL op is a total function grid -> grid. There are no loops, no
    recursion, and no unbounded constructs, so evaluation time is bounded by
    (rules x ops x grid cells) and cannot hang the 9-hour Kaggle rerun.
  * The DSL has no filesystem, network, subprocess, import, attribute-access,
    or ambient-mutable-state primitives. It cannot reach them because those
    concepts do not exist in its grammar - not because they are blacklisted.

LIMITATIONS - read before trusting this:

  1. The DSL is small on purpose. Most real ARC-AGI-3 dynamics are NOT
     expressible in it today. Its job right now is to prove the verify/reject
     pipeline works end to end, not to be sufficient. Growing it is the next
     unit of work, and every new op must stay a total pure grid -> grid function.
  2. `CallableTransitionModel` wraps an arbitrary Python callable. It is for
     TESTS and HAND-WRITTEN models only. It carries NO sandbox. Never hand it
     a callable derived from model output - route that through the DSL.
  3. Verification is only ever as strong as the buffer. PASS means "consistent
     with every transition recorded so far", never "correct". A model that
     passes 40 transitions can still be wrong on the 41st; that is why the
     planner re-checks each executed step against the live game.
  4. `check_determinism` detects hidden state only if it is observable within
     the replayed prefix. A model that changes behaviour after N calls, where
     N exceeds the buffer, is not caught.
  5. GridState carries only the grid, level, game state, and legal action ids.
     Off-grid game state (timers, inventories) is invisible to it, and will
     surface as an unexplainable mismatch rather than a silent wrong answer.

License: CC0-1.0
"""
from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "GridState",
    "ActionRecord",
    "Transition",
    "ReplayBuffer",
    "TransitionModel",
    "CallableTransitionModel",
    "DslTransitionModel",
    "CandidateRejected",
    "Mismatch",
    "VerificationResult",
    "verify_transition_model",
    "check_determinism",
    "certify",
    "VerifiedModel",
    "PlanResult",
    "PlannerRefused",
    "plan",
    "DSL_OPS",
    "compile_transition_model",
]


# =========================================================================
# canonical immutable state
# =========================================================================


def _freeze_grid(grid: Any) -> tuple[tuple[int, ...], ...]:
    """Deep-copy any 2-D int grid into nested tuples.

    Accepts list-of-lists, tuple-of-tuples, or a numpy array (via .tolist()).
    The copy is total: the returned value shares no mutable structure with the
    input, so a caller mutating its grid afterwards cannot corrupt a snapshot.
    """
    if grid is None:
        return ()
    if hasattr(grid, "tolist"):            # numpy array
        grid = grid.tolist()
    if not isinstance(grid, (list, tuple)):
        raise TypeError(f"grid must be a 2-D sequence, got {type(grid).__name__}")

    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for r in grid:
        if hasattr(r, "tolist"):
            r = r.tolist()
        if not isinstance(r, (list, tuple)):
            raise TypeError("grid rows must be sequences")
        row = tuple(int(v) for v in r)
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"ragged grid: row 0 has {width} cells, "
                f"row {len(rows)} has {len(row)}"
            )
        rows.append(row)
    return tuple(rows)


@dataclass(frozen=True)
class GridState:
    """An immutable canonical snapshot of everything we can observe.

    Frozen and hashable, so it can key a dict or live in a set. `rows` is
    nested tuples, so there is no way to mutate a recorded state after the
    fact - which is exactly what makes exact replay meaningful.
    """

    rows: tuple[tuple[int, ...], ...] = ()
    level: int = 0
    game_state: str = ""
    available_actions: tuple[int, ...] = ()

    # ---- constructors

    @classmethod
    def from_grid(cls, grid: Any, *, level: int = 0, game_state: str = "",
                  available_actions: Iterable[int] = ()) -> "GridState":
        return cls(
            rows=_freeze_grid(grid),
            level=int(level),
            game_state=str(game_state),
            available_actions=tuple(int(a) for a in available_actions),
        )

    @classmethod
    def from_frame(cls, frame: Any) -> "GridState":
        """Build a snapshot from a framework `FrameData` (or anything shaped
        like one).

        Tolerant of the layer-stack layout: `frame.frame` is
        list[list[list[int]]] where the LAST layer is the current view - the
        same convention `my_agent.extract_grid` uses. Never raises on a missing
        or malformed frame; returns an empty snapshot instead, so a bad frame
        shows up as an obvious empty state rather than a crash mid-run.
        """
        if frame is None:
            return cls()
        raw = getattr(frame, "frame", None)
        grid: Any = None
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)) and raw:
            first = raw[0]
            if hasattr(first, "tolist"):
                first = first.tolist()
            if isinstance(first, (list, tuple)) and first and isinstance(
                    first[0], (list, tuple)):
                grid = raw[-1]                 # stack of layers -> current view
            elif isinstance(first, (list, tuple)):
                grid = raw                     # already a bare 2-D grid
        try:
            rows = _freeze_grid(grid)
        except (TypeError, ValueError):
            rows = ()
        if rows and not rows[0]:
            # A zero-width grid (e.g. frame.frame == [[]]) is a degenerate
            # observation, not a 1x0 board. Normalise it to "no observation"
            # so downstream code has exactly one empty case to handle.
            rows = ()
        state = getattr(frame, "state", "")
        return cls(
            rows=rows,
            level=int(getattr(frame, "levels_completed", 0) or 0),
            game_state=getattr(state, "value", None) or str(state),
            available_actions=tuple(
                int(a) for a in (getattr(frame, "available_actions", None) or ())
            ),
        )

    # ---- accessors

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0]) if self.rows else 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def cell(self, y: int, x: int) -> int:
        return self.rows[y][x]

    def to_lists(self) -> list[list[int]]:
        """A FRESH mutable copy. Mutating it can never touch this snapshot."""
        return [list(r) for r in self.rows]

    def with_rows(self, grid: Any) -> "GridState":
        """A new snapshot with a different grid, same metadata."""
        return GridState(
            rows=_freeze_grid(grid),
            level=self.level,
            game_state=self.game_state,
            available_actions=self.available_actions,
        )

    def digest(self) -> str:
        """Short stable content hash, for logs and mismatch reports only.
        Equality checks use `==`, which is exact."""
        h = hashlib.sha256()
        h.update(repr((self.rows, self.level, self.game_state)).encode())
        return h.hexdigest()[:16]

    def __str__(self) -> str:      # pragma: no cover - cosmetic
        return (f"GridState({self.height}x{self.width} lvl={self.level} "
                f"{self.game_state} {self.digest()})")


@dataclass(frozen=True)
class ActionRecord:
    """An immutable, value-typed action.

    WHY THIS EXISTS: the framework's `GameAction` is an Enum whose members
    carry MUTABLE `action_data`, and every member is a process-wide singleton.
    `GameAction.ACTION6.set_data({"x": 5})` mutates that object for everyone,
    forever. Storing a `GameAction` in a replay buffer therefore stores a value
    that silently changes underneath you, and replay would report phantom
    mismatches. We snapshot name + data into frozen tuples at record time.
    """

    name: str
    data: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_game_action(cls, action: Any) -> "ActionRecord":
        """Snapshot a framework GameAction (or a plain name string).

        Reads `action.action_data.model_dump()` - the field the framework
        actually sends to the engine (`agents/agent.py`). Falls back to a
        `_data` mapping for hand-rolled fakes. `game_id` is dropped: it is
        constant per run and would only add noise to the action identity.
        """
        if isinstance(action, str):
            return cls(name=action.strip().upper())
        name = getattr(action, "name", None)
        if not isinstance(name, str):
            raise TypeError(
                f"cannot derive an action name from {type(action).__name__}")

        raw: Any = None
        payload = getattr(action, "action_data", None)
        if payload is not None and hasattr(payload, "model_dump"):
            raw = payload.model_dump()
        elif isinstance(payload, Mapping):
            raw = payload
        if raw is None:
            raw = getattr(action, "_data", None)

        items: list[tuple[str, int]] = []
        if isinstance(raw, Mapping):
            for k, v in raw.items():
                if k == "game_id":
                    continue
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue          # keep the identity numeric and hashable
                items.append((str(k), int(v)))
        return cls(name=name.strip().upper(), data=tuple(sorted(items)))

    def as_dict(self) -> dict[str, int]:
        return dict(self.data)

    def get(self, key: str, default: int | None = None) -> int | None:
        for k, v in self.data:
            if k == key:
                return v
        return default

    def __str__(self) -> str:      # pragma: no cover - cosmetic
        if not self.data:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in self.data)
        return f"{self.name}({inner})"


# =========================================================================
# replay record + bounded buffer
# =========================================================================


@dataclass(frozen=True)
class Transition:
    """One fully-specified ground-truth step: before -> action -> after."""

    index: int
    before: GridState
    action: ActionRecord
    after: GridState
    timestamp: float
    meta: tuple[tuple[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return self.before.rows != self.after.rows

    def meta_dict(self) -> dict[str, Any]:
        return dict(self.meta)

    def __str__(self) -> str:      # pragma: no cover - cosmetic
        return (f"#{self.index} {self.before.digest()} --{self.action}--> "
                f"{self.after.digest()}{'' if self.changed else ' (no-op)'}")


class ReplayBuffer:
    """Bounded FIFO of exact transitions.

    Bounded because a 9-hour run at ~430 ms/action is ~75k steps per game and
    a 64x64 snapshot is not free. Oldest transitions are dropped first, and
    `dropped` records how many - so a verifier PASS can never be mistaken for
    "verified against the whole run" when it wasn't.
    """

    __slots__ = ("_items", "_capacity", "_seq", "_dropped")

    def __init__(self, capacity: int = 2000) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive int")
        self._capacity = capacity
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._seq = 0
        self._dropped = 0

    # ---- writing

    def record(self, before: Any, action: Any, after: Any,
               meta: Mapping[str, Any] | None = None,
               timestamp: float | None = None) -> Transition:
        """Append one transition.

        Accepts raw frames/actions and canonicalises them, so a caller cannot
        accidentally store a mutable reference into the ground-truth record.
        """
        b = before if isinstance(before, GridState) else GridState.from_frame(before)
        a = after if isinstance(after, GridState) else GridState.from_frame(after)
        act = (action if isinstance(action, ActionRecord)
               else ActionRecord.from_game_action(action))

        meta_items: tuple[tuple[str, Any], ...] = ()
        if meta:
            meta_items = tuple(sorted((str(k), v) for k, v in meta.items()))

        tr = Transition(
            index=self._seq,
            before=b,
            action=act,
            after=a,
            timestamp=time.time() if timestamp is None else float(timestamp),
            meta=meta_items,
        )
        if len(self._items) == self._capacity:
            self._dropped += 1
        self._items.append(tr)
        self._seq += 1
        return tr

    def clear(self) -> None:
        """Drop retained transitions. Does NOT reset index numbering, so
        indices in a mismatch report stay globally unique across a run."""
        self._items.clear()

    # ---- reading

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def dropped(self) -> int:
        """Transitions evicted by the bound. Non-zero means a PASS covers only
        a suffix of the run."""
        return self._dropped

    @property
    def recorded(self) -> int:
        """Total ever recorded, including evicted."""
        return self._seq

    def transitions(self) -> tuple[Transition, ...]:
        return tuple(self._items)

    def changed_only(self) -> tuple[Transition, ...]:
        return tuple(t for t in self._items if t.changed)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __getitem__(self, i: int) -> Transition:
        return self._items[i]


# =========================================================================
# transition models
# =========================================================================


class TransitionModel:
    """Interface: predict the next state. Implementations must be PURE.

    `predict` may raise; the verifier catches it and reports it as a mismatch
    rather than letting it escape into the play loop.
    """

    name: str = "transition-model"

    def predict(self, state: GridState, action: ActionRecord) -> GridState:
        raise NotImplementedError

    def reset(self) -> None:
        """Optional hook, called before each determinism replay pass. A model
        with legitimate per-episode setup may use it. A model that NEEDS it to
        stay consistent has hidden state and should fail `check_determinism`."""


class CallableTransitionModel(TransitionModel):
    """Wrap a plain Python callable `(GridState, ActionRecord) -> GridState`.

    *** NOT A SANDBOX. *** The callable runs with this process's full
    privileges. Use for tests and hand-written reference models only. Never
    build one from LLM output - route that through `compile_transition_model`.
    """

    def __init__(self, fn: Callable[[GridState, ActionRecord], GridState],
                 name: str = "callable") -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn
        self.name = name

    def predict(self, state: GridState, action: ActionRecord) -> GridState:
        out = self._fn(state, action)
        if isinstance(out, GridState):
            return out
        return state.with_rows(out)      # tolerate a bare grid return


# =========================================================================
# the safe declarative candidate DSL
# =========================================================================


class CandidateRejected(ValueError):
    """A candidate transition program was malformed, or used a construct
    outside the allowlist. Raised at COMPILE time, never at play time."""


# Closed allowlist. Adding an entry here is the ONLY way to widen what a
# candidate may express. Every op must be a total, pure grid -> grid function.
DSL_OPS: tuple[str, ...] = (
    "identity",      # {}
    "set_cell",      # y, x (int or "$x"/"$y" from action data), value
    "fill_rect",     # y0, x0, y1, x1, value
    "recolor",       # from, to
    "translate",     # dy, dx, and optionally color (only that colour moves)
    "swap_colors",   # a, b
)

_MAX_RULES = 64
_MAX_OPS_PER_RULE = 16
_ALLOWED_REFS = ("$x", "$y")

_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "set_cell": ("y", "x", "value"),
    "fill_rect": ("y0", "x0", "y1", "x1", "value"),
    "recolor": ("from", "to"),
    "translate": ("dy", "dx"),
    "swap_colors": ("a", "b"),
}
_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {"translate": ("color",)}


def _resolve(val: Any, action: ActionRecord, *, field: str) -> int:
    """Resolve a DSL scalar: an int literal, or "$x"/"$y" bound from the
    action's own data. Nothing else is a valid scalar."""
    if isinstance(val, bool):
        raise CandidateRejected(f"{field}: bool is not a valid scalar")
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val in _ALLOWED_REFS:
        got = action.get(val[1:])
        if got is None:
            raise CandidateRejected(
                f"{field}: {val} referenced but action {action.name} has no "
                f"'{val[1:]}' in its data")
        return int(got)
    raise CandidateRejected(
        f"{field}: expected an int or one of {list(_ALLOWED_REFS)}, got {val!r}")


def _validate_op(op: Any, position: str) -> dict[str, Any]:
    if not isinstance(op, Mapping):
        raise CandidateRejected(
            f"{position}: op must be a mapping, got {type(op).__name__}")
    kind = op.get("op")
    if not isinstance(kind, str):
        raise CandidateRejected(f"{position}: op is missing a string 'op' key")
    if kind not in DSL_OPS:
        raise CandidateRejected(
            f"{position}: unknown op {kind!r}; allowed ops are {list(DSL_OPS)}")

    required = _REQUIRED_FIELDS[kind]
    optional = _OPTIONAL_FIELDS.get(kind, ())

    for key in required:
        if key not in op:
            raise CandidateRejected(f"{position}: op {kind!r} requires '{key}'")
    allowed = {"op", *required, *optional}
    extra = set(op) - allowed
    if extra:
        raise CandidateRejected(
            f"{position}: op {kind!r} has unknown key(s) {sorted(extra)}; "
            f"allowed: {sorted(allowed)}")

    # Scalars are checked for SHAPE here; "$x"/"$y" bind at evaluation time.
    for key in (*required, *optional):
        if key not in op:
            continue
        v = op[key]
        if isinstance(v, bool) or not (
                isinstance(v, int) or (isinstance(v, str) and v in _ALLOWED_REFS)):
            raise CandidateRejected(
                f"{position}: op {kind!r} field '{key}' must be an int or one "
                f"of {list(_ALLOWED_REFS)}, got {v!r}")
    return dict(op)


def _validate_when(when: Any, position: str) -> dict[str, Any]:
    if when is None:
        return {}
    if not isinstance(when, Mapping):
        raise CandidateRejected(f"{position}: 'when' must be a mapping")
    extra = set(when) - {"action", "level"}
    if extra:
        raise CandidateRejected(
            f"{position}: 'when' has unknown key(s) {sorted(extra)}; "
            f"allowed: ['action', 'level']")
    out: dict[str, Any] = {}
    if "action" in when:
        act = when["action"]
        if not isinstance(act, str) or not act.strip():
            raise CandidateRejected(
                f"{position}: 'when.action' must be a non-empty string")
        out["action"] = act.strip().upper()
    if "level" in when:
        lvl = when["level"]
        if isinstance(lvl, bool) or not isinstance(lvl, int):
            raise CandidateRejected(f"{position}: 'when.level' must be an int")
        out["level"] = lvl
    return out


def compile_transition_model(spec: Any, name: str = "dsl") -> "DslTransitionModel":
    """Validate an untrusted candidate spec and return a runnable model.

    This is the ONLY supported path from model output to a runnable transition
    model. It performs NO code execution: `spec` is plain data (the shape you
    get back from `json.loads`), and every field is checked against a closed
    allowlist before anything runs.

    Raises `CandidateRejected` - with a message naming the offending path - on
    any structural problem, unknown op, unknown key, or bad scalar type.
    """
    if not isinstance(spec, Mapping):
        raise CandidateRejected(
            f"spec must be a mapping, got {type(spec).__name__}")

    extra = set(spec) - {"version", "rules", "default", "name"}
    if extra:
        raise CandidateRejected(
            f"spec has unknown key(s) {sorted(extra)}; "
            f"allowed: ['version', 'rules', 'default', 'name']")

    version = spec.get("version", 1)
    if version != 1:
        raise CandidateRejected(
            f"unsupported spec version {version!r}; expected 1")

    default = spec.get("default", "identity")
    if default != "identity":
        raise CandidateRejected(
            f"'default' must be 'identity' (the only safe fallback), "
            f"got {default!r}")

    rules = spec.get("rules")
    if not isinstance(rules, (list, tuple)):
        raise CandidateRejected("'rules' must be a list")
    if len(rules) > _MAX_RULES:
        raise CandidateRejected(
            f"too many rules ({len(rules)}); the bound is {_MAX_RULES}")

    compiled: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = []
    for i, rule in enumerate(rules):
        pos = f"rules[{i}]"
        if not isinstance(rule, Mapping):
            raise CandidateRejected(f"{pos}: rule must be a mapping")
        extra = set(rule) - {"when", "do"}
        if extra:
            raise CandidateRejected(
                f"{pos}: unknown key(s) {sorted(extra)}; allowed: ['when', 'do']")
        when = _validate_when(rule.get("when"), pos)
        ops = rule.get("do")
        if not isinstance(ops, (list, tuple)):
            raise CandidateRejected(f"{pos}: 'do' must be a list of ops")
        if len(ops) > _MAX_OPS_PER_RULE:
            raise CandidateRejected(
                f"{pos}: too many ops ({len(ops)}); "
                f"the bound is {_MAX_OPS_PER_RULE}")
        compiled.append(
            (when,
             tuple(_validate_op(o, f"{pos}.do[{j}]") for j, o in enumerate(ops)))
        )

    return DslTransitionModel(compiled, name=str(spec.get("name", name)))


class DslTransitionModel(TransitionModel):
    """Evaluates a compiled DSL candidate. Pure, total, and bounded.

    Rules are scanned in order; the FIRST whose `when` matches fires, and its
    ops apply left to right. No rule matching means identity - a model that
    says nothing predicts nothing changed, which the verifier will then reject
    on the first real change. Out-of-bounds writes are clipped rather than
    raising, so a slightly-wrong candidate produces a diagnosable mismatch
    instead of an exception.
    """

    def __init__(self,
                 rules: Sequence[tuple[dict[str, Any], tuple[dict[str, Any], ...]]],
                 name: str = "dsl") -> None:
        self._rules = tuple(rules)
        self.name = name

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def _matches(self, when: Mapping[str, Any], state: GridState,
                 action: ActionRecord) -> bool:
        if "action" in when and when["action"] != action.name:
            return False
        if "level" in when and when["level"] != state.level:
            return False
        return True

    def predict(self, state: GridState, action: ActionRecord) -> GridState:
        for when, ops in self._rules:
            if not self._matches(when, state, action):
                continue
            grid = state.to_lists()
            for op in ops:
                grid = _apply_op(op, grid, action)
            return state.with_rows(grid)
        return state          # default: identity


def _apply_op(op: Mapping[str, Any], grid: list[list[int]],
              action: ActionRecord) -> list[list[int]]:
    """Apply one validated op to a mutable grid copy. Total: never raises for
    out-of-range coordinates, which are clipped or ignored."""
    kind = op["op"]
    h = len(grid)
    w = len(grid[0]) if h else 0

    if kind == "identity" or h == 0 or w == 0:
        return grid

    if kind == "set_cell":
        y = _resolve(op["y"], action, field="set_cell.y")
        x = _resolve(op["x"], action, field="set_cell.x")
        v = _resolve(op["value"], action, field="set_cell.value")
        if 0 <= y < h and 0 <= x < w:
            grid[y][x] = v
        return grid

    if kind == "fill_rect":
        y0 = _resolve(op["y0"], action, field="fill_rect.y0")
        x0 = _resolve(op["x0"], action, field="fill_rect.x0")
        y1 = _resolve(op["y1"], action, field="fill_rect.y1")
        x1 = _resolve(op["x1"], action, field="fill_rect.x1")
        v = _resolve(op["value"], action, field="fill_rect.value")
        for y in range(max(0, min(y0, y1)), min(h - 1, max(y0, y1)) + 1):
            for x in range(max(0, min(x0, x1)), min(w - 1, max(x0, x1)) + 1):
                grid[y][x] = v
        return grid

    if kind == "recolor":
        src = _resolve(op["from"], action, field="recolor.from")
        dst = _resolve(op["to"], action, field="recolor.to")
        for y in range(h):
            row = grid[y]
            for x in range(w):
                if row[x] == src:
                    row[x] = dst
        return grid

    if kind == "swap_colors":
        a = _resolve(op["a"], action, field="swap_colors.a")
        b = _resolve(op["b"], action, field="swap_colors.b")
        for y in range(h):
            row = grid[y]
            for x in range(w):
                if row[x] == a:
                    row[x] = b
                elif row[x] == b:
                    row[x] = a
        return grid

    if kind == "translate":
        dy = _resolve(op["dy"], action, field="translate.dy")
        dx = _resolve(op["dx"], action, field="translate.dx")
        color = (_resolve(op["color"], action, field="translate.color")
                 if "color" in op else None)
        if color is None:
            # Whole-grid shift; vacated cells take the most common colour.
            counts: dict[int, int] = {}
            for row in grid:
                for v in row:
                    counts[v] = counts.get(v, 0) + 1
            bg = max(counts, key=lambda k: counts[k])
            out = [[bg] * w for _ in range(h)]
            for y in range(h):
                ny = y + dy
                if not 0 <= ny < h:
                    continue
                for x in range(w):
                    nx = x + dx
                    if 0 <= nx < w:
                        out[ny][nx] = grid[y][x]
            return out
        # Move only cells of `color`; they leave behind the surrounding value.
        counts = {}
        for row in grid:
            for v in row:
                if v != color:
                    counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=lambda k: counts[k]) if counts else color
        moved = [(y, x) for y in range(h) for x in range(w) if grid[y][x] == color]
        for y, x in moved:
            grid[y][x] = bg
        for y, x in moved:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                grid[ny][nx] = color
        return grid

    return grid            # unreachable: the compile-time allowlist covers all ops


# =========================================================================
# exact replay verification
# =========================================================================


@dataclass(frozen=True)
class Mismatch:
    """The FIRST place a candidate disagreed with reality.

    First, not all - because after one wrong prediction every later state is
    off the rails and the rest of the report is noise. The single earliest
    divergence is the actionable counterexample.
    """

    index: int                     # the transition's buffer index
    position: int                  # its 0-based position in this verify pass
    action: ActionRecord
    reason: str                    # "cell" | "shape" | "exception" | "type"
    detail: str
    cell: tuple[int, int] | None = None      # (y, x) of first differing cell
    expected: int | None = None
    actual: int | None = None

    def __str__(self) -> str:
        head = (f"MISMATCH at transition #{self.index} "
                f"(position {self.position}) after {self.action}")
        if self.cell is not None:
            y, x = self.cell
            return (f"{head}: cell (y={y}, x={x}) expected {self.expected}, "
                    f"model predicted {self.actual}")
        return f"{head}: {self.reason} - {self.detail}"


@dataclass(frozen=True)
class VerificationResult:
    """PASS or the first mismatch. Truthy iff it passed."""

    ok: bool
    checked: int
    model_name: str
    mismatch: Mismatch | None = None
    buffer_dropped: int = 0

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if self.ok:
            tail = (f" (WARNING: {self.buffer_dropped} earlier transitions were "
                    f"evicted by the buffer bound and were NOT checked)"
                    if self.buffer_dropped else "")
            return (f"PASS: {self.model_name} reproduced all {self.checked} "
                    f"recorded transitions{tail}")
        return f"FAIL: {self.model_name} - {self.mismatch}"


def _first_diff(expected: GridState, actual: GridState
                ) -> tuple[str, str, tuple[int, int] | None, int | None, int | None]:
    """Locate the earliest disagreement in row-major order."""
    if expected.shape != actual.shape:
        return ("shape",
                f"expected grid {expected.height}x{expected.width}, "
                f"model produced {actual.height}x{actual.width}",
                None, None, None)
    for y in range(expected.height):
        er, ar = expected.rows[y], actual.rows[y]
        if er == ar:
            continue
        for x in range(expected.width):
            if er[x] != ar[x]:
                return ("cell", "grid cell differs", (y, x), er[x], ar[x])
    return ("", "", None, None, None)


def verify_transition_model(model: TransitionModel,
                            buffer: ReplayBuffer | Iterable[Transition],
                            ) -> VerificationResult:
    """Replay every recorded transition through `model`. Exact match required.

    Returns PASS only if `model.predict(before, action).rows == after.rows` for
    every transition, in order. Otherwise returns the FIRST mismatch, which is
    the counterexample a synthesiser should be re-prompted with.

    Only the grid is compared. Level and game-state fields are ground truth the
    model is not asked to predict, so a candidate is never penalised for them.

    A raising or wrong-typed `predict` is reported as a mismatch, not raised:
    a bad candidate must never be able to kill a run.
    """
    transitions = (buffer.transitions() if isinstance(buffer, ReplayBuffer)
                   else tuple(buffer))
    dropped = buffer.dropped if isinstance(buffer, ReplayBuffer) else 0
    model_name = getattr(model, "name", type(model).__name__)

    for position, tr in enumerate(transitions):
        try:
            predicted = model.predict(tr.before, tr.action)
        except Exception as exc:                    # noqa: BLE001 - by design
            return VerificationResult(
                ok=False, checked=position, model_name=model_name,
                buffer_dropped=dropped,
                mismatch=Mismatch(
                    index=tr.index, position=position, action=tr.action,
                    reason="exception",
                    detail=f"{type(exc).__name__}: {exc}"),
            )
        if not isinstance(predicted, GridState):
            return VerificationResult(
                ok=False, checked=position, model_name=model_name,
                buffer_dropped=dropped,
                mismatch=Mismatch(
                    index=tr.index, position=position, action=tr.action,
                    reason="type",
                    detail=f"predict returned {type(predicted).__name__}, "
                           f"expected GridState"),
            )
        if predicted.rows != tr.after.rows:
            reason, detail, cell, expected, actual = _first_diff(tr.after, predicted)
            return VerificationResult(
                ok=False, checked=position, model_name=model_name,
                buffer_dropped=dropped,
                mismatch=Mismatch(
                    index=tr.index, position=position, action=tr.action,
                    reason=reason, detail=detail, cell=cell,
                    expected=expected, actual=actual),
            )

    return VerificationResult(ok=True, checked=len(transitions),
                              model_name=model_name, buffer_dropped=dropped)


# =========================================================================
# hidden-state / determinism check
# =========================================================================


@dataclass(frozen=True)
class DeterminismResult:
    """Whether a candidate answered identically on identical inputs."""

    ok: bool
    passes: int
    checked: int
    model_name: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        if self.ok:
            return (f"DETERMINISTIC: {self.model_name} gave identical answers "
                    f"across {self.passes} passes over {self.checked} transitions")
        return f"NON-DETERMINISTIC: {self.model_name} - {self.detail}"


def check_determinism(model: TransitionModel,
                      buffer: ReplayBuffer | Iterable[Transition],
                      repeats: int = 2) -> DeterminismResult:
    """Replay the same inputs `repeats` times and require identical answers.

    This is the hidden-state detector. A model that counts its own calls,
    caches, reads a clock, or samples an RNG will drift between passes and be
    rejected here even when a single pass would have "verified".

    LIMITATION: it can only see hidden state that manifests within the replayed
    prefix. State that only changes behaviour after more calls than the buffer
    holds will not be caught.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("repeats must be an int >= 2")

    transitions = (buffer.transitions() if isinstance(buffer, ReplayBuffer)
                   else tuple(buffer))
    model_name = getattr(model, "name", type(model).__name__)

    baseline: list[Any] | None = None
    for p in range(repeats):
        try:
            model.reset()
        except Exception as exc:                    # noqa: BLE001 - by design
            return DeterminismResult(
                ok=False, passes=p, checked=len(transitions), model_name=model_name,
                detail=f"reset() raised {type(exc).__name__}: {exc}")

        answers: list[Any] = []
        for tr in transitions:
            try:
                out = model.predict(tr.before, tr.action)
                answers.append(out.rows if isinstance(out, GridState)
                               else ("<non-GridState>", type(out).__name__))
            except Exception as exc:                # noqa: BLE001 - by design
                answers.append(("<exception>", type(exc).__name__, str(exc)))

        if baseline is None:
            baseline = answers
            continue
        for i, (a, b) in enumerate(zip(baseline, answers)):
            if a != b:
                tr = transitions[i]
                return DeterminismResult(
                    ok=False, passes=p + 1, checked=len(transitions),
                    model_name=model_name,
                    detail=(f"pass 1 and pass {p + 1} disagree on transition "
                            f"#{tr.index} (position {i}) after {tr.action}; "
                            f"the model has hidden state or is stochastic"))

    return DeterminismResult(ok=True, passes=repeats, checked=len(transitions),
                             model_name=model_name)


# =========================================================================
# certification: the only gate to planning
# =========================================================================


class VerifiedModel:
    """A transition model that has PASSED exact replay AND the determinism check.

    Only `certify()` can produce one. The planner accepts nothing else. That is
    the structural guarantee behind "no planning on an unverified model" - it
    is enforced by the type, not by a comment asking nicely.

    A VerifiedModel is a claim about a specific buffer at a specific moment. It
    does NOT mean the model is correct; see LIMITATIONS at the top of this file.
    """

    __slots__ = ("_model", "_verification", "_determinism", "_evidence")

    def __init__(self, model: TransitionModel, verification: VerificationResult,
                 determinism: DeterminismResult, _token: object) -> None:
        if _token is not _CERTIFY_TOKEN:
            raise PermissionError(
                "VerifiedModel cannot be constructed directly; use certify()")
        self._model = model
        self._verification = verification
        self._determinism = determinism
        self._evidence = (verification.checked, verification.buffer_dropped)

    @property
    def name(self) -> str:
        return getattr(self._model, "name", type(self._model).__name__)

    @property
    def verification(self) -> VerificationResult:
        return self._verification

    @property
    def determinism(self) -> DeterminismResult:
        return self._determinism

    @property
    def transitions_verified(self) -> int:
        return self._evidence[0]

    def predict(self, state: GridState, action: ActionRecord) -> GridState:
        return self._model.predict(state, action)

    def __str__(self) -> str:      # pragma: no cover - cosmetic
        return (f"VerifiedModel({self.name}, "
                f"{self.transitions_verified} transitions)")


_CERTIFY_TOKEN = object()


def certify(model: TransitionModel, buffer: ReplayBuffer | Iterable[Transition],
            *, repeats: int = 2, min_transitions: int = 1,
            ) -> tuple[VerifiedModel | None, VerificationResult, DeterminismResult]:
    """Run both gates. Returns `(verified_or_None, verification, determinism)`.

    A caller that wants to plan MUST check the first element. Returning None
    rather than raising keeps the normal "candidate was wrong" path cheap -
    being wrong is the common case, not an exceptional one.

    `min_transitions` guards the degenerate case: a model trivially "passes"
    an empty buffer, and planning on that evidence would be self-deception.
    """
    verification = verify_transition_model(model, buffer)
    determinism = check_determinism(model, buffer, repeats=repeats)

    if not verification.ok or not determinism.ok:
        return None, verification, determinism
    if verification.checked < min_transitions:
        return None, verification, determinism
    return (VerifiedModel(model, verification, determinism, _CERTIFY_TOKEN),
            verification, determinism)


# =========================================================================
# bounded forward planner
# =========================================================================


class PlannerRefused(RuntimeError):
    """The planner was asked to run without the preconditions it requires:
    a certified model and an explicit goal predicate."""


@dataclass(frozen=True)
class PlanResult:
    """Outcome of a bounded search. Truthy iff a plan was found."""

    found: bool
    actions: tuple[ActionRecord, ...] = ()
    states_expanded: int = 0
    max_depth_reached: int = 0
    stop_reason: str = ""

    def __bool__(self) -> bool:
        return self.found

    def __str__(self) -> str:
        if self.found:
            path = " -> ".join(str(a) for a in self.actions)
            return (f"PLAN({len(self.actions)} actions, "
                    f"{self.states_expanded} expanded): {path}")
        return (f"NO PLAN ({self.stop_reason}); expanded "
                f"{self.states_expanded} states to depth "
                f"{self.max_depth_reached}")


def plan(model: Any,
         start: GridState,
         goal: Callable[[GridState], bool],
         actions: Sequence[ActionRecord],
         *,
         max_depth: int = 6,
         max_nodes: int = 20000,
         time_budget_s: float = 1.0) -> PlanResult:
    """Breadth-first search over a CERTIFIED transition model.

    Preconditions, both enforced:
      * `model` must be a `VerifiedModel` from `certify()`. Passing a raw
        TransitionModel raises `PlannerRefused` - planning over an unverified
        world model produces confident nonsense, which is worse than no plan.
      * `goal` must be an explicit callable predicate `GridState -> bool`.
        There is no default and no inferred goal. Goal inference is a separate
        problem this module does not pretend to solve.

    BFS, so the first plan found is the shortest - which matters directly,
    because the competition score is (human_actions / our_actions) squared.

    Bounded three ways (depth, nodes, wall-clock) because this runs inside a
    ~430 ms/action budget. Hitting a bound returns `found=False` with the
    reason named; it never runs long and never raises.
    """
    if not isinstance(model, VerifiedModel):
        raise PlannerRefused(
            "plan() requires a VerifiedModel from certify(); got "
            f"{type(model).__name__}. Verify the transition model against the "
            "replay buffer first - planning over an unverified model is not "
            "supported.")
    if not callable(goal):
        raise PlannerRefused(
            "plan() requires an explicit goal predicate (GridState -> bool); "
            f"got {type(goal).__name__}. There is no default goal.")
    if not actions:
        raise PlannerRefused("plan() requires a non-empty action set")
    if max_depth < 1 or max_nodes < 1 or time_budget_s <= 0:
        raise PlannerRefused(
            "max_depth, max_nodes and time_budget_s must all be positive")

    canonical = tuple(
        a if isinstance(a, ActionRecord) else ActionRecord.from_game_action(a)
        for a in actions
    )

    deadline = time.monotonic() + time_budget_s
    if goal(start):
        return PlanResult(found=True, actions=(), states_expanded=0,
                          max_depth_reached=0, stop_reason="start satisfies goal")

    frontier: deque[tuple[GridState, tuple[ActionRecord, ...]]] = deque(
        [(start, ())])
    seen: set[GridState] = {start}
    expanded = 0
    deepest = 0

    while frontier:
        if time.monotonic() > deadline:
            return PlanResult(found=False, states_expanded=expanded,
                              max_depth_reached=deepest,
                              stop_reason=f"time budget {time_budget_s}s exhausted")
        state, path = frontier.popleft()
        if len(path) >= max_depth:
            continue
        expanded += 1

        for action in canonical:
            try:
                nxt = model.predict(state, action)
            except Exception:                       # noqa: BLE001 - by design
                continue        # a model that throws on some input just prunes
            if not isinstance(nxt, GridState) or nxt in seen:
                continue
            new_path = path + (action,)
            deepest = max(deepest, len(new_path))
            if goal(nxt):
                return PlanResult(found=True, actions=new_path,
                                  states_expanded=expanded,
                                  max_depth_reached=len(new_path),
                                  stop_reason="goal reached")
            if len(seen) >= max_nodes:
                return PlanResult(found=False, states_expanded=expanded,
                                  max_depth_reached=deepest,
                                  stop_reason=f"node bound {max_nodes} reached")
            seen.add(nxt)
            frontier.append((nxt, new_path))

    return PlanResult(found=False, states_expanded=expanded,
                      max_depth_reached=deepest,
                      stop_reason="search space exhausted")
