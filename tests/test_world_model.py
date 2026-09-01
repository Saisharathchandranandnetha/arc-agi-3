"""Tests for agent/world_model.py.

Stdlib `unittest` on purpose: `.venv` has no pytest, and the Kaggle rerun runs
with internet disabled, so the substrate and its tests must stay dependency-free.

Run:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v      (Windows)
    .venv/bin/python          -m unittest discover -s tests -v      (POSIX)

License: CC0-1.0
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.world_model import (  # noqa: E402
    ActionRecord,
    CallableTransitionModel,
    CandidateRejected,
    GridState,
    PlannerRefused,
    ReplayBuffer,
    TransitionModel,
    certify,
    check_determinism,
    compile_transition_model,
    plan,
    verify_transition_model,
)


# --------------------------------------------------------------- fixtures


def blank(h: int = 4, w: int = 4, fill: int = 0) -> list[list[int]]:
    return [[fill] * w for _ in range(h)]


class FakeSimpleAction:
    """Stands in for a framework GameAction with a SimpleAction payload."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.action_data = _Payload({"game_id": "demo"})


class FakeComplexAction:
    """Stands in for GameAction.ACTION6 carrying click coordinates."""

    def __init__(self, name: str, x: int, y: int) -> None:
        self.name = name
        self.action_data = _Payload({"game_id": "demo", "x": x, "y": y})


class _Payload:
    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def model_dump(self) -> dict:
        return dict(self._data)


class FakeFrame:
    """Stands in for FrameData: a layer stack, last layer is the current view."""

    def __init__(self, grid, level=0, state="NOT_FINISHED", available=(1, 2, 6)):
        self.frame = [grid]
        self.levels_completed = level
        self.state = state
        self.available_actions = list(available)


def paint_click(state: GridState, action: ActionRecord) -> GridState:
    """Reference dynamics used across tests: a click paints its cell colour 5;
    ACTION1 recolours every 5 back to 0; anything else is a no-op."""
    if action.name == "ACTION6":
        x, y = action.get("x"), action.get("y")
        grid = state.to_lists()
        if x is not None and y is not None and 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            grid[y][x] = 5
        return state.with_rows(grid)
    if action.name == "ACTION1":
        return state.with_rows([[0 if v == 5 else v for v in row]
                                for row in state.rows])
    return state


def build_buffer(steps, capacity: int = 100) -> ReplayBuffer:
    """Drive `paint_click` forward and record ground truth."""
    buf = ReplayBuffer(capacity=capacity)
    state = GridState.from_grid(blank(), level=0, game_state="NOT_FINISHED")
    for action in steps:
        nxt = paint_click(state, action)
        buf.record(state, action, nxt)
        state = nxt
    return buf


CLICKS = [
    ActionRecord("ACTION6", (("x", 1), ("y", 1))),
    ActionRecord("ACTION6", (("x", 2), ("y", 0))),
    ActionRecord("ACTION1"),
    ActionRecord("ACTION6", (("x", 3), ("y", 3))),
]


# ================================================= 1. successful exact replay


class TestExactReplayPasses(unittest.TestCase):

    def test_correct_model_verifies_against_recorded_transitions(self):
        buf = build_buffer(CLICKS)
        self.assertEqual(len(buf), 4)

        model = CallableTransitionModel(paint_click, name="reference")
        result = verify_transition_model(model, buf)

        self.assertTrue(result.ok, msg=str(result))
        self.assertIsNone(result.mismatch)
        self.assertEqual(result.checked, 4)
        self.assertIn("PASS", str(result))

    def test_dsl_candidate_verifies_on_transitions_it_covers(self):
        """A DSL spec compiled from untrusted data can pass exact replay."""
        buf = build_buffer([ActionRecord("ACTION6", (("x", 1), ("y", 2)))])
        spec = {
            "version": 1,
            "rules": [
                {"when": {"action": "ACTION6"},
                 "do": [{"op": "set_cell", "y": "$y", "x": "$x", "value": 5}]},
            ],
        }
        model = compile_transition_model(spec, name="click-paints-5")
        self.assertTrue(verify_transition_model(model, buf).ok)

    def test_verification_result_is_truthy_on_pass(self):
        buf = build_buffer(CLICKS)
        self.assertTrue(bool(verify_transition_model(
            CallableTransitionModel(paint_click), buf)))

    def test_pass_reports_buffer_evictions_so_scope_is_not_overstated(self):
        """A bounded buffer that dropped history must say so on a PASS."""
        buf = build_buffer(CLICKS, capacity=2)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.dropped, 2)

        result = verify_transition_model(CallableTransitionModel(paint_click), buf)
        self.assertTrue(result.ok)
        self.assertEqual(result.buffer_dropped, 2)
        self.assertIn("evicted", str(result))


# ======================================================= 2. mismatch detection


class TestMismatchDetection(unittest.TestCase):

    def test_wrong_colour_is_caught_with_exact_cell_diagnostic(self):
        buf = build_buffer([ActionRecord("ACTION6", (("x", 2), ("y", 1)))])

        def wrong(state, action):
            grid = state.to_lists()
            grid[action.get("y")][action.get("x")] = 7      # should be 5
            return state.with_rows(grid)

        result = verify_transition_model(
            CallableTransitionModel(wrong, name="wrong-colour"), buf)

        self.assertFalse(result.ok)
        self.assertIsNotNone(result.mismatch)
        self.assertEqual(result.mismatch.reason, "cell")
        self.assertEqual(result.mismatch.cell, (1, 2))
        self.assertEqual(result.mismatch.expected, 5)
        self.assertEqual(result.mismatch.actual, 7)
        self.assertEqual(result.mismatch.action.name, "ACTION6")

    def test_reports_the_FIRST_mismatch_not_a_later_one(self):
        """Only the earliest divergence is an actionable counterexample."""
        buf = build_buffer(CLICKS)

        def wrong_from_second(state, action):
            out = paint_click(state, action)
            # correct on transition 0, wrong on every transition after it
            if state.rows != GridState.from_grid(blank()).rows:
                grid = out.to_lists()
                grid[0][0] = 9
                return out.with_rows(grid)
            return out

        result = verify_transition_model(
            CallableTransitionModel(wrong_from_second), buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.position, 1)
        self.assertEqual(result.mismatch.index, 1)
        self.assertEqual(result.checked, 1)      # one transition passed first

    def test_identity_model_is_rejected_by_the_first_real_change(self):
        buf = build_buffer(CLICKS)
        empty = compile_transition_model({"version": 1, "rules": []})
        result = verify_transition_model(empty, buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.position, 0)

    def test_shape_change_is_reported_as_a_shape_mismatch(self):
        buf = build_buffer([ActionRecord("ACTION6", (("x", 0), ("y", 0)))])
        result = verify_transition_model(
            CallableTransitionModel(
                lambda s, a: GridState.from_grid(blank(2, 2)), name="wrong-shape"),
            buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.reason, "shape")
        self.assertIn("4x4", str(result.mismatch))

    def test_a_raising_model_is_reported_never_propagated(self):
        """A bad candidate must not be able to kill a live run."""
        buf = build_buffer(CLICKS)

        def explode(state, action):
            raise RuntimeError("boom")

        result = verify_transition_model(
            CallableTransitionModel(explode, name="exploder"), buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.reason, "exception")
        self.assertIn("RuntimeError", result.mismatch.detail)
        self.assertIn("boom", result.mismatch.detail)

    def test_non_gridstate_return_is_reported_as_a_type_mismatch(self):
        buf = build_buffer(CLICKS)

        class BadType(TransitionModel):
            name = "bad-type"

            def predict(self, state, action):
                return "not a state"

        result = verify_transition_model(BadType(), buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.reason, "type")


# ============================================ 3. copied-grid / no-mutation
class TestSnapshotIntegrity(unittest.TestCase):

    def test_snapshot_does_not_alias_the_source_grid(self):
        src = blank()
        state = GridState.from_grid(src)
        src[0][0] = 99                       # mutate the caller's grid afterwards
        self.assertEqual(state.cell(0, 0), 0)

    def test_to_lists_returns_a_fresh_copy_each_time(self):
        state = GridState.from_grid(blank())
        a = state.to_lists()
        a[1][1] = 42
        self.assertEqual(state.cell(1, 1), 0)
        self.assertEqual(state.to_lists()[1][1], 0)
        self.assertIsNot(state.to_lists(), state.to_lists())

    def test_gridstate_is_frozen_and_hashable(self):
        state = GridState.from_grid(blank())
        with self.assertRaises(Exception):
            state.rows = ()                  # type: ignore[misc]
        self.assertEqual(len({state, GridState.from_grid(blank())}), 1)

    def test_recorded_transitions_survive_later_mutation_of_the_source(self):
        """The whole point of the buffer: ground truth cannot be edited later."""
        grid = blank()
        buf = ReplayBuffer()
        before = GridState.from_grid(grid)
        after_grid = blank()
        after_grid[0][0] = 5
        buf.record(before, ActionRecord("ACTION1"), GridState.from_grid(after_grid))

        grid[0][0] = 77
        after_grid[3][3] = 88
        self.assertEqual(buf[0].before.cell(0, 0), 0)
        self.assertEqual(buf[0].after.cell(3, 3), 0)
        self.assertEqual(buf[0].after.cell(0, 0), 5)

    def test_dsl_model_does_not_mutate_the_state_it_was_given(self):
        state = GridState.from_grid(blank())
        model = compile_transition_model({
            "version": 1,
            "rules": [{"when": {"action": "ACTION6"},
                       "do": [{"op": "fill_rect", "y0": 0, "x0": 0,
                               "y1": 3, "x1": 3, "value": 9}]}],
        })
        out = model.predict(state, ActionRecord("ACTION6"))
        self.assertEqual(out.cell(0, 0), 9)
        self.assertEqual(state.cell(0, 0), 0)     # input untouched

    def test_action_snapshot_survives_gameaction_singleton_mutation(self):
        """GameAction members are process-global with MUTABLE action_data.
        Recording must snapshot by value, or replay reports phantom mismatches."""
        act = FakeComplexAction("ACTION6", x=1, y=2)
        rec = ActionRecord.from_game_action(act)
        act.action_data = _Payload({"game_id": "demo", "x": 99, "y": 99})
        self.assertEqual(rec.as_dict(), {"x": 1, "y": 2})

    def test_action_record_reads_action_data_not_the_absent_underscore_data(self):
        """Regression guard: the framework exposes `action_data.model_dump()`.
        There is no `_data` attribute on a real GameAction."""
        rec = ActionRecord.from_game_action(FakeComplexAction("ACTION6", x=4, y=7))
        self.assertEqual(rec.name, "ACTION6")
        self.assertEqual(rec.as_dict(), {"x": 4, "y": 7})
        self.assertNotIn("game_id", rec.as_dict())

    def test_simple_action_records_no_coordinates(self):
        rec = ActionRecord.from_game_action(FakeSimpleAction("ACTION3"))
        self.assertEqual(rec, ActionRecord("ACTION3"))
        self.assertEqual(rec.as_dict(), {})

    def test_from_frame_takes_the_last_layer_of_a_layer_stack(self):
        top = [[1, 2], [3, 4]]
        frame = FakeFrame(top, level=2, state="NOT_FINISHED", available=(1, 6))
        frame.frame = [[[0, 0], [0, 0]], top]      # two layers; last is current
        state = GridState.from_frame(frame)
        self.assertEqual(state.rows, ((1, 2), (3, 4)))
        self.assertEqual(state.level, 2)
        self.assertEqual(state.available_actions, (1, 6))

    def test_from_frame_tolerates_a_missing_or_empty_frame(self):
        self.assertEqual(GridState.from_frame(None).rows, ())
        self.assertEqual(GridState.from_frame(FakeFrame([])).rows, ())

    def test_ragged_grid_is_rejected_at_snapshot_time(self):
        with self.assertRaises(ValueError):
            GridState.from_grid([[0, 0], [0]])


# ================================================ 4. malformed candidate rejection


class TestCandidateRejection(unittest.TestCase):
    """The DSL compiler is the trust boundary for LLM output."""

    def assertRejects(self, spec, *, contains: str = ""):
        with self.assertRaises(CandidateRejected) as ctx:
            compile_transition_model(spec)
        if contains:
            self.assertIn(contains, str(ctx.exception))

    def test_non_mapping_spec_is_rejected(self):
        self.assertRejects("print('pwned')", contains="mapping")
        self.assertRejects([1, 2, 3])
        self.assertRejects(None)

    def test_unknown_op_is_rejected_by_the_allowlist(self):
        self.assertRejects(
            {"version": 1,
             "rules": [{"when": {"action": "ACTION1"},
                        "do": [{"op": "exec", "code": "import os"}]}]},
            contains="unknown op")

    def test_unknown_key_inside_a_known_op_is_rejected(self):
        self.assertRejects(
            {"version": 1,
             "rules": [{"when": {"action": "ACTION1"},
                        "do": [{"op": "recolor", "from": 1, "to": 2,
                                "__import__": "os"}]}]},
            contains="unknown key")

    def test_missing_required_field_is_rejected(self):
        self.assertRejects(
            {"version": 1,
             "rules": [{"when": {"action": "ACTION1"},
                        "do": [{"op": "set_cell", "y": 0, "x": 0}]}]},
            contains="requires 'value'")

    def test_non_int_scalar_is_rejected(self):
        self.assertRejects(
            {"version": 1,
             "rules": [{"do": [{"op": "recolor", "from": "os.system", "to": 1}]}]},
            contains="must be an int")

    def test_bool_is_not_accepted_as_an_int_scalar(self):
        self.assertRejects(
            {"version": 1,
             "rules": [{"do": [{"op": "recolor", "from": True, "to": 1}]}]})

    def test_unknown_top_level_key_is_rejected(self):
        self.assertRejects({"version": 1, "rules": [], "eval": "1+1"},
                           contains="unknown key")

    def test_unknown_when_key_is_rejected(self):
        self.assertRejects(
            {"version": 1, "rules": [{"when": {"python": "os"}, "do": []}]},
            contains="unknown key")

    def test_unsupported_version_is_rejected(self):
        self.assertRejects({"version": 2, "rules": []}, contains="version")

    def test_non_identity_default_is_rejected(self):
        self.assertRejects({"version": 1, "rules": [], "default": "crash"},
                           contains="identity")

    def test_rules_must_be_a_list(self):
        self.assertRejects({"version": 1, "rules": {"a": 1}}, contains="list")

    def test_rule_and_op_counts_are_bounded(self):
        self.assertRejects(
            {"version": 1, "rules": [{"do": []} for _ in range(65)]},
            contains="too many rules")
        self.assertRejects(
            {"version": 1,
             "rules": [{"do": [{"op": "identity"} for _ in range(17)]}]},
            contains="too many ops")

    def test_unbound_coordinate_reference_fails_closed_at_eval_time(self):
        """A spec referencing $x compiles, but a simple action has no x. The
        failure surfaces as a verifier mismatch, never as a crash."""
        model = compile_transition_model({
            "version": 1,
            "rules": [{"do": [{"op": "set_cell", "y": "$y", "x": "$x",
                               "value": 5}]}],
        })
        buf = ReplayBuffer()
        s = GridState.from_grid(blank())
        buf.record(s, ActionRecord("ACTION1"), s)
        result = verify_transition_model(model, buf)
        self.assertFalse(result.ok)
        self.assertEqual(result.mismatch.reason, "exception")
        self.assertIn("CandidateRejected", result.mismatch.detail)

    def test_a_valid_spec_still_compiles(self):
        model = compile_transition_model({
            "version": 1,
            "name": "ok",
            "rules": [{"when": {"action": "ACTION1", "level": 0},
                       "do": [{"op": "recolor", "from": 0, "to": 1}]}],
            "default": "identity",
        })
        self.assertEqual(model.rule_count, 1)
        self.assertEqual(model.name, "ok")

    def test_callable_model_rejects_a_non_callable(self):
        with self.assertRaises(TypeError):
            CallableTransitionModel("not callable")   # type: ignore[arg-type]


# ============================== 5. hidden-state / non-determinism rejection


class TestDeterminism(unittest.TestCase):

    def test_pure_model_passes_the_determinism_check(self):
        buf = build_buffer(CLICKS)
        result = check_determinism(CallableTransitionModel(paint_click), buf)
        self.assertTrue(result.ok, msg=str(result))
        self.assertEqual(result.passes, 2)

    def test_call_counting_hidden_state_is_rejected(self):
        buf = build_buffer(CLICKS)

        class Counter(TransitionModel):
            name = "call-counter"

            def __init__(self):
                self.calls = 0

            def predict(self, state, action):
                self.calls += 1
                out = paint_click(state, action)
                if self.calls > len(CLICKS):        # drifts on the second pass
                    grid = out.to_lists()
                    grid[0][0] = 3
                    return out.with_rows(grid)
                return out

        # NOTE: the counter must be fresh. Calling predict() beforehand (e.g. a
        # verify pass) would push BOTH determinism passes past the threshold,
        # making them agree and hiding the very drift under test.
        result = check_determinism(Counter(), buf)
        self.assertFalse(result.ok)
        self.assertIn("hidden state", result.detail)
        self.assertEqual(result.passes, 2)

    def test_stochastic_model_is_rejected(self):
        buf = build_buffer(CLICKS)
        import random

        rng = random.Random(0)

        def noisy(state, action):
            grid = paint_click(state, action).to_lists()
            grid[0][0] = rng.randint(0, 9)
            return state.with_rows(grid)

        self.assertFalse(check_determinism(
            CallableTransitionModel(noisy, name="noisy"), buf).ok)

    def test_determinism_uses_reset_between_passes(self):
        """A model whose reset() clears its state passes; one that relies on
        never being replayed does not."""
        buf = build_buffer(CLICKS)

        class Resettable(TransitionModel):
            name = "resettable"

            def __init__(self):
                self.calls = 0

            def reset(self):
                self.calls = 0

            def predict(self, state, action):
                self.calls += 1
                out = paint_click(state, action)
                if self.calls > len(CLICKS):
                    grid = out.to_lists()
                    grid[0][0] = 3
                    return out.with_rows(grid)
                return out

        self.assertTrue(check_determinism(Resettable(), buf).ok)

    def test_a_raising_model_is_compared_not_propagated(self):
        buf = build_buffer(CLICKS)

        def explode(state, action):
            raise ValueError("nope")

        # Deterministically raising is deterministic; verification catches it.
        self.assertTrue(check_determinism(
            CallableTransitionModel(explode), buf).ok)
        self.assertFalse(verify_transition_model(
            CallableTransitionModel(explode), buf).ok)

    def test_repeats_below_two_is_a_usage_error(self):
        with self.assertRaises(ValueError):
            check_determinism(CallableTransitionModel(paint_click),
                              build_buffer(CLICKS), repeats=1)


# ============================================= 6. planner over a toy model


class TestPlanner(unittest.TestCase):
    """The planner runs only behind certify(), over a deterministic model."""

    def _certified(self):
        buf = build_buffer(CLICKS)
        verified, verification, determinism = certify(
            CallableTransitionModel(paint_click, name="reference"), buf)
        self.assertIsNotNone(verified, msg=f"{verification} / {determinism}")
        return verified

    def test_planner_finds_the_shortest_path_to_an_explicit_goal(self):
        verified = self._certified()
        start = GridState.from_grid(blank())

        def goal(s):                       # two specific cells painted
            return s.cell(0, 0) == 5 and s.cell(1, 1) == 5

        result = plan(
            verified, start, goal,
            actions=[ActionRecord("ACTION6", (("x", 0), ("y", 0))),
                     ActionRecord("ACTION6", (("x", 1), ("y", 1))),
                     ActionRecord("ACTION6", (("x", 3), ("y", 3))),
                     ActionRecord("ACTION1")],
            max_depth=4)

        self.assertTrue(result.found, msg=str(result))
        self.assertEqual(len(result.actions), 2)        # BFS => shortest
        self.assertEqual({a.as_dict()["x"] for a in result.actions}, {0, 1})

    def test_goal_already_satisfied_returns_an_empty_plan(self):
        verified = self._certified()
        result = plan(verified, GridState.from_grid(blank()), lambda s: True,
                      actions=[ActionRecord("ACTION1")])
        self.assertTrue(result.found)
        self.assertEqual(result.actions, ())

    def test_unreachable_goal_stops_at_a_bound_without_hanging(self):
        verified = self._certified()
        result = plan(verified, GridState.from_grid(blank()),
                      goal=lambda s: s.cell(0, 0) == 42,     # colour 42 is unreachable
                      actions=[ActionRecord("ACTION6", (("x", 0), ("y", 0))),
                               ActionRecord("ACTION1")],
                      max_depth=5, time_budget_s=2.0)
        self.assertFalse(result.found)
        self.assertTrue(result.stop_reason)

    def test_node_bound_is_enforced(self):
        verified = self._certified()
        acts = [ActionRecord("ACTION6", (("x", x), ("y", y)))
                for y in range(4) for x in range(4)]
        result = plan(verified, GridState.from_grid(blank()),
                      goal=lambda s: False, actions=acts,
                      max_depth=6, max_nodes=20, time_budget_s=5.0)
        self.assertFalse(result.found)
        self.assertIn("node bound", result.stop_reason)

    # ---- the gate itself

    def test_planner_refuses_an_uncertified_model(self):
        with self.assertRaises(PlannerRefused) as ctx:
            plan(CallableTransitionModel(paint_click),
                 GridState.from_grid(blank()), lambda s: True,
                 actions=[ActionRecord("ACTION1")])
        self.assertIn("VerifiedModel", str(ctx.exception))

    def test_planner_refuses_a_non_callable_goal(self):
        with self.assertRaises(PlannerRefused) as ctx:
            plan(self._certified(), GridState.from_grid(blank()),
                 goal="win", actions=[ActionRecord("ACTION1")])  # type: ignore[arg-type]
        self.assertIn("goal predicate", str(ctx.exception))

    def test_planner_refuses_an_empty_action_set(self):
        with self.assertRaises(PlannerRefused):
            plan(self._certified(), GridState.from_grid(blank()),
                 lambda s: False, actions=[])

    def test_certify_refuses_a_model_that_fails_replay(self):
        buf = build_buffer(CLICKS)
        verified, verification, _ = certify(
            CallableTransitionModel(lambda s, a: s, name="identity"), buf)
        self.assertIsNone(verified)
        self.assertFalse(verification.ok)

    def test_certify_refuses_an_empty_buffer_even_though_replay_is_vacuous(self):
        verified, verification, _ = certify(
            CallableTransitionModel(paint_click), ReplayBuffer())
        self.assertTrue(verification.ok)         # vacuously
        self.assertIsNone(verified)              # but not certifiable

    def test_verified_model_cannot_be_forged(self):
        from agent.world_model import VerifiedModel
        with self.assertRaises(PermissionError):
            VerifiedModel(CallableTransitionModel(paint_click),
                          None, None, object())  # type: ignore[arg-type]


# ============================================================ buffer mechanics


class TestReplayBuffer(unittest.TestCase):

    def test_buffer_is_bounded_and_evicts_oldest_first(self):
        buf = build_buffer(CLICKS, capacity=2)
        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.capacity, 2)
        self.assertEqual(buf.dropped, 2)
        self.assertEqual(buf.recorded, 4)
        self.assertEqual([t.index for t in buf], [2, 3])

    def test_indices_stay_globally_unique_across_a_clear(self):
        buf = build_buffer(CLICKS)
        buf.clear()
        self.assertEqual(len(buf), 0)
        s = GridState.from_grid(blank())
        self.assertEqual(buf.record(s, ActionRecord("ACTION1"), s).index, 4)

    def test_record_canonicalises_raw_frames_and_actions(self):
        buf = ReplayBuffer()
        tr = buf.record(FakeFrame(blank()),
                        FakeComplexAction("ACTION6", x=2, y=3),
                        FakeFrame(blank(), level=1))
        self.assertIsInstance(tr.before, GridState)
        self.assertIsInstance(tr.action, ActionRecord)
        self.assertEqual(tr.action.as_dict(), {"x": 2, "y": 3})
        self.assertEqual(tr.after.level, 1)

    def test_changed_flag_and_changed_only_filter(self):
        buf = build_buffer(CLICKS)
        self.assertTrue(buf[0].changed)
        s = GridState.from_grid(blank())
        buf.record(s, ActionRecord("ACTION3"), s)
        self.assertFalse(buf[-1].changed)
        self.assertEqual(len(buf.changed_only()), len(buf) - 1)

    def test_metadata_is_recorded_and_readable(self):
        buf = ReplayBuffer()
        s = GridState.from_grid(blank())
        tr = buf.record(s, ActionRecord("ACTION1"), s,
                        meta={"step": 7, "game_id": "ls20"})
        self.assertEqual(tr.meta_dict(), {"step": 7, "game_id": "ls20"})
        self.assertGreater(tr.timestamp, 0)

    def test_invalid_capacity_is_rejected(self):
        for bad in (0, -1, True, "10"):
            with self.assertRaises(ValueError):
                ReplayBuffer(capacity=bad)       # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
