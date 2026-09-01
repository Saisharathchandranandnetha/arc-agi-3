# ARC-AGI-3 Campaign Roadmap

**Owner:** Sai Sharath Chandra (NIAT Hyderabad)
**Target:** Milestone #2 top-3 by **Sep 30, 2026** · Final leaderboard by **Nov 2, 2026**
**Doc status:** living document — update the STATUS block after every run
**Last updated:** Aug 15, 2026 (evening — post Phase 1)

---

## 0. STATUS — where we actually are

| | |
|---|---|
| **Phase** | 1.5 of 5 — first submission in progress |
| **Local score** | mean ~0.035, std 74% of mean (stable, low-variance) |
| **Reliable clears** | lp85, vc33, r11l (level 1, every run) |
| **Kaggle submissions** | 0 (about to make first) |
| **Leaderboard rank** | unranked |
| **BLOCKER** | 2 config items: Kaggle username in kernel-metadata.json, and kaggle.json API token |

**Progress so far:**
- Phase 0 (instrument) DONE — agent runs, measurable, `eval_runs.py` for variance
- Phase 1 (efficient L1) DONE — per-game budgets, efficiency guard, smart click
  priority, premature-RESET fixed. 0.0 → stable 0.035.
- Key bug caught: baselines KeyError was zeroing the human counts, costing ~500x
  on every clear via the squared penalty. Fixed → single-game lp85 hit 1.52.

**Next single action:** set Kaggle username + install API token → `make submit`
(Phase A validates it runs) → click Submit to Competition (Phase B = real score).

**The wall ahead (Phase 2):** L1 is solved by heuristics. L2 needs reasoning about
a rule, not better clicking. That's where a model enters and where a prize-contending
score lives. First submission tells us whether the heuristics generalise to the
unseen 110 before we build on top.

---

## 1. The competition, in numbers

### Verified facts (sourced, not assumed)

| Fact | Value | Source |
|---|---|---|
| Eval set | **110 private games, never seen** | Kaggle data page |
| Public/private split | 50% public LB, 50% private LB | Kaggle data page |
| Local dev set | 25 public games (`environment_files/`) | Kaggle data page |
| Runtime | 9 hours, GPU or CPU | Kaggle code requirements |
| Hardware | RTX 6000 (`g4-standard-48`), ARC-AGI-3 only | Kaggle overview |
| Internet | **Disabled** — no cloud APIs | Kaggle code requirements |
| Submissions | **1 per day**, pick 2 finals | Rules §2.2 |
| Winner license | **CC-BY 4.0** + open weights (OSI checklist) | Rules §2.5 |
| External data/models | Allowed if free + public | Rules §2.6 |
| Pre-training compute | **Unlimited** | arcprize.org/policy |
| Team size | up to 8 | Rules §2.1 |
| Milestone #2 prizes | $25K / $7.5K / $5K | Kaggle overview |

### Scoring

```
level_score = min(human_actions / agent_actions, 1.0) ** 2
game_score  = Σ (level_index × level_score) / Σ (1..L)     # denominator FIXED
total       = mean(game_score across all games)
```

Three consequences that drive everything:

1. **Fixed denominator.** Uncleared levels stay in the denominator. So an extra
   cleared level *can never lower* your game score, however sloppily you clear it.
   → **Inside a game, always go deeper. Never bail on progress.**

2. **Level-1-only caps at 6.67%.** Clearing L1 on every game at perfect human pace
   still only scores 6.67. → **10% is structurally impossible without clearing L2+.**

3. **Squared penalty.** 2× human actions → 0.25. 5× → 0.04. Efficiency is a
   multiplier on clears, not a substitute for them.

### Leaderboard (Aug 12, 2026)

| Rank | Team | Score | Entries |
|---|---|---|---|
| 1 | cstl | 2.52 | 23 |
| 2 | YUTO KOJIMA | 1.86 | 65 |
| 3 | Andy liu | 1.69 | 7 |
| 6 | Tufa Labs | 1.62 | 103 |

Field is compressed between 1.4 and 2.5. Tufa (Milestone #1 winner) is now 6th.
Ranks 3–49 are separated by less than 0.3 — likely inside run-to-run noise.

### The number that defines the problem

Human play averages **685 actions/game** (source: OPINE-World Table 3).

```
110 games × 685 actions = 75,350 actions
9 hours                 = 32,400 seconds
                        → 430 ms per action
```

An LLM turn costs 5–20 s. **That is a 12–47× shortfall.** Calling a model once per
action is not slow — it is arithmetically impossible. Every visible top-10 approach
does it anyway. This is the central thesis of our campaign.

---

## 2. What the research says

**OPINE-World** — Courtis, Li & Sanner (U Toronto), arXiv:2607.01531, Jul 2026.
Scores **78.4%** on the same 25 public games. Clears 20/25 games, 160/183 levels.
Beats human action counts on 16 of its 20 wins.

The mechanism:
- LLM writes `transition_function(state, action)` as Python — a *world model*
- Program admitted **only if it replays every recorded transition exactly**
- Synthesis fires **on counterexamples**, not on a schedule
- A planner searches the verified program; each step re-checked against the live game
- Two agents (actor + synthesizer) share one replay buffer

Why the leaderboard is at 2.52 and the paper is at 78.4: the paper used Claude Opus 4.8
over the internet with no time limit. **The gap is compute, not ideas.** Our contribution
is making this survive 430 ms/action on one local GPU, offline.

**Tiny Recursive Models** — arXiv:2511.02886. A 7M-parameter net scored 7.8% on
ARC-AGI-2, and was fine-tuned *during* the competition run in 12,500 gradient steps.
Proof that test-time training inside the 9 hours is viable. Unused in this track.

**Related:** *Hill-climbing ARC-AGI-3* (Fox et al.) is `baseline1`, 63.8%.
*Explore Before You Solve* (arXiv:2605.25931) — speed/depth tradeoff, 3rd in 2025 Paper Award.

---

## 3. Gaps — where the field is weak

Ranked by confidence the gap is real and exploitable.

| # | Gap | Why it exists | Our move |
|---|---|---|---|
| 1 | **LLM called per action** | Nobody did the 430 ms arithmetic | LLM writes code; code plays |
| 2 | **No local ground truth** | Human baselines unpublished until Jul 2026 | `baselines.py` — we have per-level counts |
| 3 | **Uniform per-game budgets** | Convention | Human effort spans 171→1843 actions (10×); scale per game |
| 4 | **No pre-training** | Assumed disallowed; it isn't | Explicitly unlimited per arcprize.org/policy |
| 5 | **Bigger-model assumption** | 96 GB VRAM → "use 27B" | Throughput math favours smaller+faster |
| 6 | **Variance ignored** | σ≈0.45 on μ≈1.6 | 1 submission/day ⇒ local eval is the only real measurement |

**Constraint that kills half the obvious ideas:** the eval is 110 *unseen* games.
Per-game tuning (`if game_id == "ls20"`) scores zero at eval. Only mechanisms that
generalise survive. This is the strongest argument for synthesis over heuristics —
a program written from live observation works on a game nobody has seen.

---

## 4. Architecture

```
   ┌─────────────────────────────────────────────────────┐
   │  PERCEPTION          grid → objects (flood fill)    │  CPU, µs
   ├─────────────────────────────────────────────────────┤
   │  REPLAY BUFFER       every (state, action, next)    │  CPU
   ├─────────────────────────────────────────────────────┤
   │  VERIFIER            exact replay; 1 mismatch = no  │  CPU, ms
   ├─────────────────────────────────────────────────────┤
   │  SYNTHESIZER         LLM writes transition_function │  GPU, ~10-30×/game
   ├─────────────────────────────────────────────────────┤
   │  PLANNER             forward search over verified   │  CPU, ms
   ├─────────────────────────────────────────────────────┤
   │  SCHEDULER           probe → abandon → reallocate   │  CPU
   └─────────────────────────────────────────────────────┘
```

**The metric nobody reports: actions generated per LLM call.**
OPINE-World never needed it (unlimited time). For us it is *the* binding constraint.
Budget allows ~10–30 LLM calls per game → 10–30 s of thinking each, comfortable for
a local model. Optimising this number is both the leaderboard strategy and the paper.

---

## 5. Phases

### Phase 0 — Instrumentation ← **WE ARE HERE**
*Goal: measure honestly. No GPU needed.*

- [x] `rhae.py` — official scorer, validated against worked examples
- [x] `baselines.py` — per-level human counts, all 25 games, cross-validated
- [x] `my_agent.py` — segmentation + dead-action detection + loop avoidance
- [ ] **Verify `extract_grid()` on a real frame** ← BLOCKER
- [ ] Per-game action budgets (replace flat `MAX_ACTIONS = 80`)
- [ ] Telemetry: actions/level, wall-clock/action, novelty rate → JSON
- [ ] First non-zero local score

*Exit:* a real number from `game_score()`, not "levels cleared".

### Phase 1 — Replay verifier
*Goal: the trust anchor. Pure CPU, no model.*

- [ ] Log every `(state, action, next_state)` triple
- [ ] `verify(program, buffer)` → first mismatch or PASS
- [ ] Hand-write a transition program for one simple game; confirm it verifies
- [ ] Reject programs with hidden state (run twice, compare)

*Why first:* makes the LLM swappable. Develop against hand-written code, then a 4B
local model, then 27B on Kaggle — harness unchanged.

### Phase 2 — Synthesis loop
*Goal: LLM writes the program. First real GPU use.*

- [ ] Local 4B model (fits your 6 GB 4050) writes `transition_function`
- [ ] Counterexample-triggered re-synthesis + stall guard
- [ ] Context compaction (structured handoff on token threshold)
- [ ] **Measure actions-per-LLM-call**
- [ ] First Kaggle submission — validate the pipeline, not the score

*Exit:* ≥100 actions generated per LLM call.

### Phase 3 — Planning
*Goal: convert a verified model into level clears.*

- [ ] Bounded forward search over the verified program
- [ ] Gate: only plan after ≥1 level cleared
- [ ] Execute one step at a time; mismatch → counterexample → re-synthesise
- [ ] Goal predicate fitted after first reward observed

*Exit:* clearing L2+ on the local set. This is where 10% becomes reachable.

### Phase 4 — Scheduler
*Goal: spend the 9 hours where they score.*

- [ ] Cheap tractability probe (~8 actions/game)
- [ ] Abandon opaque games, reinvest surplus
- [ ] Per-game budgets from observed progress, not constants
- [ ] Play order optimisation (games are playable in any order)

*Modelled gain: ~2× more actions on games that can score, same policy.*

### Phase 5 — Paper + release
*Goal: Paper Prize track + CC-BY compliance.*

- [ ] Public repo, CC-BY 4.0, open weights
- [ ] Notebook public **before Sep 30** (required for Milestone #2)
- [ ] Paper: "Programmatic world models under a hard wall-clock budget"
- [ ] Optional: verifiers-spec ARC-AGI-3 environment → Prime Intellect Hub
- [ ] Mech-interp angle: predict verifier pass/fail from model activations

---

## 6. Calendar

| Window | Focus | Deliverable |
|---|---|---|
| Aug 15–22 | Phase 0 | Non-zero local score, telemetry live |
| Aug 23–Sep 5 | Phase 1–2 | Verifier + synthesis; first submission |
| Sep 6–19 | Phase 3 | Planner; L2+ clears locally |
| Sep 20–27 | Phase 4 | Scheduler; daily submissions |
| **Sep 28–30** | **Freeze** | **Notebook public, 2 finals selected** |
| Oct | Iterate | Private LB robustness |
| Nov 2 | Final | Final submission |

Only ~50 submissions exist between now and Sep 30. Spend them on validation,
never on iteration.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Overfit to 25 public games | Eval is 110 unseen. Ban per-game constants. |
| Local model too weak to write correct programs | Verifier rejects bad programs safely — worst case is wasted calls, not wrong behaviour |
| Model load eats the budget | Measure it on submission #1; count it against the 9 h |
| Score noise ≈ improvement size | Local eval on 25 games with `baselines.py`; never trust one submission |
| Kaggle GPU quota exhausted | Scored reruns reportedly don't consume quota, but exhausted quota blocks submitting — keep headroom |
| License non-compliance | CC-BY 4.0 + open weights; our files are CC0 (compatible) |

---

## 8. Files

| File | Purpose | License |
|---|---|---|
| `rhae.py` | Official scorer + strategy analysis | CC0 |
| `baselines.py` | Per-level human counts (25 games) | CC0 |
| `budget.py` | Wall-clock modelling | CC0 |
| `my_agent.py` | The agent (only file the harness reads) | CC0 |
| `ROADMAP.md` | This document | CC0 |

**Note:** `baselines.py` covers only the 25 *public* games. It is a development
instrument for honest local measurement — those games are not in the eval set.

---

## 9. The thesis, in one paragraph

The field is stuck at 2.52% because everyone calls a language model once per action,
which the arithmetic says is 12–47× too slow for 110 games in 9 hours. A July 2026
paper shows that programmatic world modelling — LLM writes a transition program,
exact-replay verification, planning over the verified model — reaches 78.4% on the
same games when compute is unlimited. Our contribution is not the algorithm; it is
making that architecture fit inside one offline GPU and a hard wall clock, by
maximising **actions generated per LLM call**. That single metric is unreported in
the literature and is the binding constraint of this competition.